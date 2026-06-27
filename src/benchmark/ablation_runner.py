"""Ablation runner: train + evaluate a single AblationConfig cell.

Stateless function — takes config, data splits, returns metrics dict.

Design:
1. none   -> runner trains a backbone model directly
2. early  -> EarlyFusionWrapper owns encoder training
3. late   -> LateFusionWrapper owns encoder training
4. cmtf   -> standalone Cross-Modal Temporal Fusion predictor
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .ablation_config import AblationConfig, BACKBONE_MODELS, CMTF_MODEL
from .baseline_models import (
    LSTMPredictor,
    RandomForestRegressor_Wrapper,
    CNNLSTMPredictor,
)
from .fusion_wrappers import EarlyFusionWrapper, LateFusionWrapper
from .hybrid_fusion import HybridFusionPredictor, build_market_encoder
from .metrics import compute_all
from .baseline_hpo import get_default_baseline_hpo_params
from src.pipeline.news_encoder import (
    SENTIMENT_TRACE_COLUMNS as _PIPELINE_SENTIMENT_TRACE_COLS,
)

SENTIMENT_FEATURE_COLUMNS = list(_PIPELINE_SENTIMENT_TRACE_COLS)

_ALLOWED_WRAPPER_NEWS_DIMS = {768, 128}


def _config_hash(cfg: AblationConfig) -> str:
    return hashlib.md5(cfg.cell_id.encode()).hexdigest()[:10]


def _supports_cmtf_market_encoder(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm"}


def _strip_sentiment_features(
    market_windows: np.ndarray,
    market_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    keep_idx = [i for i, c in enumerate(market_cols) if c not in SENTIMENT_FEATURE_COLUMNS]
    return market_windows[:, :, keep_idx], [market_cols[i] for i in keep_idx]


def _apply_sentiment_weighting(
    news_embs: np.ndarray,
    market_windows: np.ndarray,
    market_cols: list[str],
) -> np.ndarray:
    if "sentiment_mean" not in market_cols:
        raise ValueError(
            "sentiment_mode='weighted_emb' requires 'sentiment_mean' in market_cols, "
            "but it was not found."
        )
    idx = market_cols.index("sentiment_mean")
    sent_weight = np.abs(market_windows[:, :, idx])
    sent_weight = np.clip(sent_weight, 0.1, 5.0)[..., np.newaxis]
    return news_embs * sent_weight


def _validate_wrapper_news_dim(news_embs: np.ndarray) -> np.ndarray:
    """
    Wrapper/hybrid fusion paths support only:
      - raw text embeddings: 768
      - projected news embeddings: 128

    Fail fast on any other dimension instead of silently slicing.
    """
    news_embs = np.asarray(news_embs, dtype=np.float32)

    if news_embs.ndim != 3:
        raise ValueError(
            f"Expected news tensor with shape (N, S, D), got {news_embs.shape}"
        )

    last_dim = int(news_embs.shape[-1])

    if last_dim not in _ALLOWED_WRAPPER_NEWS_DIMS:
        raise ValueError(
            f"Fusion wrappers require news last dim in {_ALLOWED_WRAPPER_NEWS_DIMS}, got {last_dim}. "
            "Use text-only news embeddings for wrapper/hybrid benchmarks."
        )
    return news_embs


def _supports_early_fusion(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm"}


def _supports_late_fusion(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm", "rf"}


def _get_news_arrays(
    cfg: AblationConfig,
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve matched vs pooled-news tensors strictly."""
    if cfg.news_scope == "all":
        required = [
            ("train", "news_embs_all"),
            ("val", "news_embs_all"),
            ("test", "news_embs_all"),
        ]
        missing = [f"{split}.{key}" for split, key in required if key not in splits[split]]
        if missing:
            raise ValueError(
                f"news_scope='all' requested, but pooled news tensors are missing: {missing}"
            )

        ne_train = splits["train"]["news_embs_all"].copy()
        ne_val = splits["val"]["news_embs_all"].copy()
        ne_test = splits["test"]["news_embs_all"].copy()

        nm_train = splits["train"].get("news_masks_all")
        nm_val = splits["val"].get("news_masks_all")
        nm_test = splits["test"].get("news_masks_all")

        if nm_train is None or nm_val is None or nm_test is None:
            raise ValueError(
                "news_scope='all' requested, but one or more pooled news masks are missing."
            )

        return ne_train, ne_val, ne_test, nm_train, nm_val, nm_test

    ne_train = splits["train"]["news_embs"].copy()
    ne_val = splits["val"]["news_embs"].copy()
    ne_test = splits["test"]["news_embs"].copy()

    nm_train = splits["train"]["news_masks"]
    nm_val = splits["val"]["news_masks"]
    nm_test = splits["test"]["news_masks"]

    return ne_train, ne_val, ne_test, nm_train, nm_val, nm_test


def _apply_sentiment_mode(
    cfg: AblationConfig,
    mw_train: np.ndarray,
    mw_val: np.ndarray,
    mw_test: np.ndarray,
    ne_train: np.ndarray,
    ne_val: np.ndarray,
    ne_test: np.ndarray,
    market_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apply sentiment-mode transformations to market/news tensors."""
    cols = list(market_cols)

    if cfg.sentiment_mode == "weighted_emb":
        ne_train = _apply_sentiment_weighting(ne_train, mw_train, cols)
        ne_val = _apply_sentiment_weighting(ne_val, mw_val, cols)
        ne_test = _apply_sentiment_weighting(ne_test, mw_test, cols)

        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)

    elif cfg.sentiment_mode == "none":
        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)

    elif cfg.sentiment_mode == "scalars":
        # Keep scalar sentiment features in market windows.
        # News tensors must already be text-only (768) or projected (128)
        # for wrapper/hybrid fusion paths.
        pass

    else:
        raise ValueError(f"Unknown sentiment mode: {cfg.sentiment_mode}")

    return mw_train, mw_val, mw_test, ne_train, ne_val, ne_test, cols


def _build_encoder(
    cfg: AblationConfig,
    input_dim: int,
    device: str,
    chronos=None,
    hpo_params=None,
    seed: int = 42,
):
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()

    if cfg.model_name == "lstm":
        p = params["lstm"]
        return LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            device=device,
        )

    elif cfg.model_name == "rf":
        p = params["rf"]
        return RandomForestRegressor_Wrapper(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            max_features=p["max_features"],
            random_state=seed,
        )

    elif cfg.model_name == "cnn_lstm":
        p = params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3})
        return CNNLSTMPredictor(
            input_dim=input_dim,
            num_filters=p.get("num_filters", p.get("hidden_dim", 64)),
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            device=device,
        )

    raise ValueError(f"Unknown or unsupported backbone model for ablation_runner: {cfg.model_name}")


def _encoder_init_kwargs(
    cfg: AblationConfig,
    input_dim: int,
    device: str,
    chronos=None,
    hpo_params=None,
    seed: int = 42,
) -> dict:
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()

    if cfg.model_name == "lstm":
        p = params["lstm"]
        return dict(
            input_dim=input_dim,
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            device=device,
        )

    elif cfg.model_name == "cnn_lstm":
        p = params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3})
        return dict(
            input_dim=input_dim,
            num_filters=p.get("num_filters", p.get("hidden_dim", 64)),
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            device=device,
        )

    elif cfg.model_name == "rf":
        p = params["rf"]
        return dict(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            max_features=p["max_features"],
            random_state=seed,
        )

    raise ValueError(f"Early fusion not supported for {cfg.model_name}")


def _train_encoder(
    encoder,
    cfg: AblationConfig,
    mw_train: np.ndarray,
    y_train: np.ndarray,
    mw_val: np.ndarray,
    y_val: np.ndarray,
    horizon: int = 1,
):
    if cfg.model_name == "rf":
        encoder.fit(mw_train, y_train)
    else:
        warmup_epochs = min(horizon, 10)
        encoder.fit(mw_train, y_train, mw_val, y_val, warmup_epochs=warmup_epochs)


def run_ablation_cell(
    cfg: AblationConfig,
    splits: dict[str, dict[str, np.ndarray]],
    market_cols: list[str],
    horizon: int,
    device: str = "cpu",
    chronos=None,
    seed: int = 42,
    cache_dir: Path | None = None,
    hpo_params: dict | None = None,
) -> dict[str, float]:
    assert cfg.is_valid(), f"Invalid config: {cfg}"
    logger.info("▶ Running cell: {} (seed={})", cfg.cell_id, seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mw_train = splits["train"]["market_windows"].copy()
    mw_val = splits["val"]["market_windows"].copy()
    mw_test = splits["test"]["market_windows"].copy()

    ne_train, ne_val, ne_test, nm_train, nm_val, nm_test = _get_news_arrays(cfg, splits)

    mw_train, mw_val, mw_test, ne_train, ne_val, ne_test, _ = _apply_sentiment_mode(
        cfg,
        mw_train,
        mw_val,
        mw_test,
        ne_train,
        ne_val,
        ne_test,
        market_cols,
    )

    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    input_dim = mw_train.shape[-1] if mw_train.ndim == 3 else 1
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()

    # ------------------------------------------------------------------
    # NONE = raw backbone itself
    # ------------------------------------------------------------------
    if cfg.fusion_type == "none":
        if cfg.model_name not in BACKBONE_MODELS:
            raise ValueError(f"fusion_type='none' requires backbone model, got {cfg.model_name}")

        encoder = _build_encoder(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos=chronos,
            hpo_params=hpo_params,
            seed=seed,
        )
        _train_encoder(
            encoder,
            cfg,
            mw_train,
            y_train,
            mw_val,
            y_val,
            horizon=horizon,
        )
        preds = encoder.predict_market_only(mw_test)

    # ------------------------------------------------------------------
    # EARLY = wrapper-based fusion
    # ------------------------------------------------------------------
    elif cfg.fusion_type == "early":
        if cfg.model_name not in BACKBONE_MODELS:
            raise ValueError(f"fusion_type='early' requires backbone model, got {cfg.model_name}")

        if not _supports_early_fusion(cfg.model_name):
            raise ValueError(f"Early fusion not supported for model: {cfg.model_name}")

        ne_train_f = _validate_wrapper_news_dim(ne_train)
        ne_val_f = _validate_wrapper_news_dim(ne_val)
        ne_test_f = _validate_wrapper_news_dim(ne_test)
        raw_news_dim = int(ne_train_f.shape[-1])

        if ne_val_f.shape[-1] != raw_news_dim or ne_test_f.shape[-1] != raw_news_dim:
            raise ValueError(
                f"Inconsistent validated news dims: train={ne_train_f.shape[-1]}, "
                f"val={ne_val_f.shape[-1]}, test={ne_test_f.shape[-1]}"
            )

        wrapper = EarlyFusionWrapper(
            encoder_cls={"lstm": LSTMPredictor, "cnn_lstm": CNNLSTMPredictor}[cfg.model_name],
            encoder_kwargs=_encoder_init_kwargs(cfg, input_dim, device, chronos, hpo_params, seed),
            raw_news_dim=raw_news_dim,
            projected_news_dim=params.get("news", {}).get("projected_news_dim", 128),
        )

        p = params.get(cfg.model_name, {})
        warmup_epochs = min(horizon, 10)

        wrapper.fit(
            mw_train,
            ne_train_f,
            y_train,
            mw_val,
            ne_val_f,
            y_val,
            epochs=100,
            batch_size=p.get("batch_size", 32),
            learning_rate=p.get("lr", 1e-3),
            warmup_epochs=warmup_epochs,
        )
        preds = wrapper.predict(mw_test, ne_test_f)

    # ------------------------------------------------------------------
    # LATE = wrapper-based residual fusion
    # ------------------------------------------------------------------
    elif cfg.fusion_type == "late":
        if cfg.model_name not in BACKBONE_MODELS:
            raise ValueError(f"fusion_type='late' requires backbone model, got {cfg.model_name}")

        if not _supports_late_fusion(cfg.model_name):
            raise ValueError(f"Late fusion not supported for model: {cfg.model_name}")

        ne_train_f = _validate_wrapper_news_dim(ne_train)
        ne_val_f = _validate_wrapper_news_dim(ne_val)
        ne_test_f = _validate_wrapper_news_dim(ne_test)
        raw_news_dim = int(ne_train_f.shape[-1])

        if ne_val_f.shape[-1] != raw_news_dim or ne_test_f.shape[-1] != raw_news_dim:
            raise ValueError(
                f"Inconsistent validated news dims: train={ne_train_f.shape[-1]}, "
                f"val={ne_val_f.shape[-1]}, test={ne_test_f.shape[-1]}"
            )

        encoder = _build_encoder(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos=chronos,
            hpo_params=hpo_params,
            seed=seed,
        )

        wrapper = LateFusionWrapper(
            encoder=encoder,
            raw_news_dim=raw_news_dim,
            projected_news_dim=params.get("news", {}).get("projected_news_dim", 128),
            seq_len=mw_train.shape[1],
            device=device,
            horizon=horizon,
        )

        p_news = params.get("news", {})

        wrapper.fit(
            mw_train,
            ne_train_f,
            y_train,
            mw_val,
            ne_val_f,
            y_val,
            news_mask_train=nm_train,
            news_mask_val=nm_val,
            epochs_news=p_news.get("epochs", 30),
            batch_size_news=p_news.get("batch_size", 32),
            lr_news=p_news.get("lr", 1e-3),
            patience_news=p_news.get("patience", 8),
        )
        preds = wrapper.predict(mw_test, ne_test_f, nm_test)

    # ------------------------------------------------------------------
    # CMTF = standalone Cross-Modal Temporal Fusion predictor
    # ------------------------------------------------------------------
    elif cfg.fusion_type == "cmtf":
        if cfg.model_name != CMTF_MODEL:
            raise ValueError(
                f"fusion_type='cmtf' requires model_name='{CMTF_MODEL}', got {cfg.model_name}"
            )

        ne_train_f = _validate_wrapper_news_dim(ne_train)
        ne_val_f = _validate_wrapper_news_dim(ne_val)
        ne_test_f = _validate_wrapper_news_dim(ne_test)
        raw_news_dim = int(ne_train_f.shape[-1])

        if ne_val_f.shape[-1] != raw_news_dim or ne_test_f.shape[-1] != raw_news_dim:
            raise ValueError(
                f"Inconsistent validated news dims: train={ne_train_f.shape[-1]}, "
                f"val={ne_val_f.shape[-1]}, test={ne_test_f.shape[-1]}"
            )

        hybrid_params = params.get("hybrid_fusion", {})

        hybrid_market_model_name = cfg.market_encoder_name
        if hybrid_market_model_name is None:
            raise ValueError("CMTF requires cfg.market_encoder_name to be set")

        if hybrid_market_model_name not in {"lstm", "cnn_lstm"}:
            raise ValueError(
                f"Unsupported CMTF market_encoder_name: {hybrid_market_model_name}"
            )

        encoder_hpo = params.get(hybrid_market_model_name, {})
        market_encoder = build_market_encoder(
            model_name=hybrid_market_model_name,
            input_dim=input_dim,
            seq_len=mw_train.shape[1],
            horizon=horizon,
            device=device,
            hidden_dim=encoder_hpo.get("hidden_dim", 64),
            num_layers=encoder_hpo.get("num_layers", 2),
            dropout=encoder_hpo.get("dropout", 0.3),
            sign_penalty_weight=encoder_hpo.get("sign_penalty_weight", 0.02),
        )

        hybrid_model = HybridFusionPredictor(
            market_encoder=market_encoder,
            raw_news_dim=raw_news_dim,
            projected_news_dim=cfg.projected_news_dim,
            fusion_market_dim=cfg.fusion_market_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            seq_len=mw_train.shape[1],
            huber_delta=1.0,
            sign_penalty_weight=cfg.sign_penalty_weight,
            use_cross_attention=cfg.use_cross_attention,
            use_positional_encoding=cfg.use_positional_encoding,
            use_news_gate=cfg.use_news_gate,
            use_variance_reg=cfg.use_variance_reg,
            use_two_stage=cfg.use_two_stage,
            use_aux_loss=cfg.use_aux_loss,
            freeze_market_encoder=False,
            recency_gate_k=cfg.recency_gate_k,
            target_scale=100.0,
            aux_loss_weight=cfg.aux_loss_weight,
            encoder_lr_scale=cfg.encoder_lr_scale,
            stage1_ratio=cfg.stage1_ratio,
            market_epochs=cfg.market_epochs,
            fusion_epochs=cfg.fusion_epochs,
            market_patience=cfg.market_patience,
            fusion_patience=cfg.fusion_patience,
            news_gate_alpha=cfg.news_gate_alpha,
            variance_reg_coeff=cfg.variance_reg_coeff,
            output_mode=cfg.output_mode,
            device=device,
            use_interaction_prod=cfg.use_interaction_prod,
            use_interaction_diff=cfg.use_interaction_diff,
            use_news_context_prod=cfg.use_news_context_prod,
            use_cosine_sim=cfg.use_cosine_sim,
            use_pooled_news=cfg.use_pooled_news,
            fusion_style=cfg.fusion_style,
            market_query_mode=cfg.market_query_mode,
        )

        market_encoder_params = params.get(hybrid_market_model_name, {})
        market_fit_kwargs = {
            "epochs": cfg.market_epochs,
            "batch_size": market_encoder_params.get("batch_size", 32),
            "learning_rate": market_encoder_params.get("lr", 1e-3),
            "patience": cfg.market_patience,
            "warmup_epochs": min(horizon, 10),
        }

        hybrid_model.fit(
            mw_train,
            ne_train_f,
            y_train,
            mw_val,
            ne_val_f,
            y_val,
            news_mask_train=nm_train,
            news_mask_val=nm_val,
            market_fit_kwargs=market_fit_kwargs,
            epochs=cfg.fusion_epochs,
            batch_size=hybrid_params.get("batch_size", 32),
            lr=hybrid_params.get("lr", 5e-4),
            patience=cfg.fusion_patience,
        )

        preds = hybrid_model.predict(mw_test, ne_test_f, nm_test)

    else:
        raise ValueError(f"Unknown fusion type: {cfg.fusion_type}")

    metrics = compute_all(y_test, preds, horizon=horizon)
    logger.info(
        "  ✓ {} → DA%={:.1f}  Sharpe={:.3f}  IC={:.3f}",
        cfg.cell_id,
        metrics["DA%"],
        metrics["Sharpe"],
        metrics["IC"],
    )

    is_degen = metrics.get("F1", 1.0) < 0.01 and abs(metrics.get("DA%", 50.0) - 50.0) < 1.5
    metrics["degenerate"] = is_degen
    if is_degen:
        logger.warning("  ⚠ Degenerate prediction (constant output) for {} at {}D", cfg.cell_id, horizon)

    if cache_dir is not None:
        pred_dir = Path(cache_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        short_id = _config_hash(cfg)
        pred_file = pred_dir / f"{short_id}__seed{seed}__{horizon}d.npy"
        np.save(str(pred_file), preds)

        truth_file = pred_dir / f"truth__{horizon}d.npy"
        if truth_file.exists():
            existing_truth = np.load(str(truth_file))
            if existing_truth.shape != y_test.shape or not np.allclose(existing_truth, y_test):
                raise ValueError(
                    f"Existing truth file mismatch for horizon={horizon}. "
                    "Refusing to reuse inconsistent cached truth."
                )
        else:
            np.save(str(truth_file), y_test)

    return metrics