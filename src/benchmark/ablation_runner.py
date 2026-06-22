"""Ablation runner: train + evaluate a single AblationConfig cell.

Stateless function — takes config, data splits, returns metrics dict.
"""

from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .ablation_config import AblationConfig
from .baseline_models import (
    LSTMPredictor,
    LSTMHybridPredictor,
    RandomForestRegressor_Wrapper,
    CNNLSTMPredictor,
    CNNLSTMHybridPredictor,
    extract_market_summary_features,
)
from .fusion_wrappers import EarlyFusionWrapper, LateFusionWrapper, HybridFusionWrapper
from .metrics import compute_all
from .baseline_hpo import get_default_baseline_hpo_params
from src.pipeline.news_encoder import (
    SENTIMENT_TRACE_COLUMNS as _PIPELINE_SENTIMENT_TRACE_COLS,
)

SENTIMENT_FEATURE_COLUMNS = list(_PIPELINE_SENTIMENT_TRACE_COLS)


def _config_hash(cfg: AblationConfig) -> str:
    return hashlib.md5(cfg.cell_id.encode()).hexdigest()[:10]


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
        return news_embs
    idx = market_cols.index("sentiment_mean")
    sent_weight = np.abs(market_windows[:, :, idx])
    sent_weight = np.clip(sent_weight, 0.1, 5.0)[..., np.newaxis]
    return news_embs * sent_weight


def _supports_early_fusion(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm"}


def _build_encoder(
    cfg: AblationConfig,
    input_dim: int,
    device: str,
    chronos=None,
    hpo_params=None,
    seed: int = 42,
    tabular_dim: int | None = None,
):
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()

    if cfg.model_name == "lstm":
        p = params["lstm"]
        return LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            device=device,
        )

    elif cfg.model_name == "lstm_hybrid":
        p = params.get("lstm_hybrid", params["lstm"])
        return LSTMHybridPredictor(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
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
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            device=device,
        )

    elif cfg.model_name == "cnn_lstm_hybrid":
        p = params.get("cnn_lstm_hybrid", params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3}))
        return CNNLSTMHybridPredictor(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            num_filters=p.get("num_filters", 64),
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            device=device,
        )

    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")


def _train_encoder(
    encoder,
    cfg: AblationConfig,
    mw_train: np.ndarray,
    y_train: np.ndarray,
    mw_val: np.ndarray,
    y_val: np.ndarray,
    horizon: int = 1,
    market_tab_train: np.ndarray | None = None,
    market_tab_val: np.ndarray | None = None,
):
    if cfg.model_name == "rf":
        encoder.fit(mw_train, y_train)

    elif cfg.model_name in {"lstm_hybrid", "cnn_lstm_hybrid"}:
        warmup_epochs = min(horizon, 10)
        encoder.fit(
            mw_train,
            y_train,
            mw_val,
            y_val,
            market_tabular_train=market_tab_train,
            market_tabular_val=market_tab_val,
            warmup_epochs=warmup_epochs,
        )

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
    cols = list(market_cols)

    ne_train = splits["train"]["news_embs"][:, :, :768].copy()
    ne_val = splits["val"]["news_embs"][:, :, :768].copy()
    ne_test = splits["test"]["news_embs"][:, :, :768].copy()

    nm_train = splits["train"]["news_masks"]
    nm_val = splits["val"]["news_masks"]
    nm_test = splits["test"]["news_masks"]

    if cfg.news_scope == "all" and "news_embs_all" in splits["train"]:
        ne_train = splits["train"]["news_embs_all"][:, :, :768].copy()
        ne_val = splits["val"]["news_embs_all"][:, :, :768].copy()
        ne_test = splits["test"]["news_embs_all"][:, :, :768].copy()
        nm_train = splits["train"].get("news_masks_all", nm_train)
        nm_val = splits["val"].get("news_masks_all", nm_val)
        nm_test = splits["test"].get("news_masks_all", nm_test)

    if cfg.sentiment_mode == "weighted_emb":
        ne_train = _apply_sentiment_weighting(ne_train, splits["train"]["market_windows"], market_cols)
        ne_val = _apply_sentiment_weighting(ne_val, splits["val"]["market_windows"], market_cols)
        ne_test = _apply_sentiment_weighting(ne_test, splits["test"]["market_windows"], market_cols)

        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)

    elif cfg.sentiment_mode == "none":
        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)

    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    market_tab_train = extract_market_summary_features(mw_train)
    market_tab_val = extract_market_summary_features(mw_val)
    market_tab_test = extract_market_summary_features(mw_test)
    tabular_dim = market_tab_train.shape[1]

    input_dim = mw_train.shape[-1] if mw_train.ndim == 3 else 1

    if cfg.fusion_type == "early" and not _supports_early_fusion(cfg.model_name):
        raise ValueError(f"Early fusion not supported for best-state backbone: {cfg.model_name}")

    encoder = None
    if cfg.fusion_type != "early":
        encoder = _build_encoder(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos=chronos,
            hpo_params=hpo_params,
            seed=seed,
            tabular_dim=tabular_dim,
        )
        _train_encoder(
            encoder,
            cfg,
            mw_train,
            y_train,
            mw_val,
            y_val,
            horizon=horizon,
            market_tab_train=market_tab_train,
            market_tab_val=market_tab_val,
        )

    # ----------------------------------------------------------
    # NONE = raw backbone itself
    # ----------------------------------------------------------
    if cfg.fusion_type == "none":
        if cfg.model_name in {"lstm_hybrid", "cnn_lstm_hybrid"}:
            preds = encoder.predict(mw_test, market_tabular=market_tab_test)
        else:
            preds = encoder.predict_market_only(mw_test)

    # ----------------------------------------------------------
    # EARLY = only for single-input encoders
    # ----------------------------------------------------------
    elif cfg.fusion_type == "early":
        wrapper = EarlyFusionWrapper(
            encoder_cls=type(_build_encoder(cfg, input_dim, device, chronos, hpo_params, seed)),
            encoder_kwargs={
                **_encoder_init_kwargs(cfg, input_dim, device, chronos, hpo_params, seed),
            },
            news_dim=768,
        )
        params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()
        p = params.get(cfg.model_name, {})
        warmup_epochs = min(horizon, 10)

        wrapper.fit(
            mw_train, ne_train, y_train,
            mw_val, ne_val, y_val,
            epochs=100,
            batch_size=p.get("batch_size", 32),
            learning_rate=p.get("lr", 1e-3),
            warmup_epochs=warmup_epochs,
        )
        preds = wrapper.predict(mw_test, ne_test)

    # ----------------------------------------------------------
    # LATE = kept as benchmark/reference only
    # ----------------------------------------------------------
    elif cfg.fusion_type == "late":
        # For transition safety, late fusion stays on plain backbone host.
        if cfg.model_name == "lstm_hybrid":
            host_cfg = AblationConfig(
                model_name="lstm",
                fusion_type="none",
                news_scope=cfg.news_scope,
                sentiment_mode=cfg.sentiment_mode,
            )
            encoder = _build_encoder(host_cfg, input_dim, device, chronos, hpo_params, seed, tabular_dim=None)
            _train_encoder(encoder, host_cfg, mw_train, y_train, mw_val, y_val, horizon=horizon)

        elif cfg.model_name == "cnn_lstm_hybrid":
            host_cfg = AblationConfig(
                model_name="cnn_lstm",
                fusion_type="none",
                news_scope=cfg.news_scope,
                sentiment_mode=cfg.sentiment_mode,
            )
            encoder = _build_encoder(host_cfg, input_dim, device, chronos, hpo_params, seed, tabular_dim=None)
            _train_encoder(encoder, host_cfg, mw_train, y_train, mw_val, y_val, horizon=horizon)

        wrapper = LateFusionWrapper(
            encoder=encoder,
            news_dim=768,
            device=device,
            horizon=horizon,
        )

        params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()
        p_news = params.get("news", {}) if "news" in params else {}

        wrapper.fit(
            mw_train, ne_train, y_train,
            mw_val, ne_val, y_val,
            news_mask_train=nm_train,
            news_mask_val=nm_val,
            epochs_news=p_news.get("epochs", 30),
            batch_size_news=p_news.get("batch_size", 32),
            lr_news=p_news.get("lr", 1e-3),
            patience_news=p_news.get("patience", 8),
        )
        preds = wrapper.predict(mw_test, ne_test, nm_test)

    # ----------------------------------------------------------
    # HYBRID = main joint fusion path
    # ----------------------------------------------------------
    elif cfg.fusion_type == "hybrid":
        if cfg.model_name == "lstm_hybrid":
            host_cfg = AblationConfig(
                model_name="lstm",
                fusion_type="none",
                news_scope=cfg.news_scope,
                sentiment_mode=cfg.sentiment_mode,
            )
            encoder = _build_encoder(host_cfg, input_dim, device, chronos, hpo_params, seed, tabular_dim=None)
            _train_encoder(encoder, host_cfg, mw_train, y_train, mw_val, y_val, horizon=horizon)

        elif cfg.model_name == "cnn_lstm_hybrid":
            host_cfg = AblationConfig(
                model_name="cnn_lstm",
                fusion_type="none",
                news_scope=cfg.news_scope,
                sentiment_mode=cfg.sentiment_mode,
            )
            encoder = _build_encoder(host_cfg, input_dim, device, chronos, hpo_params, seed, tabular_dim=None)
            _train_encoder(encoder, host_cfg, mw_train, y_train, mw_val, y_val, horizon=horizon)

        wrapper = HybridFusionWrapper(
            encoder=encoder,
            news_dim=768,
            use_positional_encoding=cfg.use_positional_encoding,
            recency_gate_k=cfg.recency_gate_k,
            use_news_gate=cfg.use_news_gate,
            horizon=horizon,
            device=device,
            use_two_stage=cfg.use_two_stage,
            use_aux_loss=cfg.use_aux_loss,
            use_variance_reg=cfg.use_variance_reg,
        )
        wrapper.fit(
            mw_train, ne_train, y_train,
            mw_val, ne_val, y_val,
            news_mask_train=nm_train,
            news_mask_val=nm_val,
        )
        preds = wrapper.predict(mw_test, ne_test, nm_test)

    else:
        raise ValueError(f"Unknown fusion type: {cfg.fusion_type}")

    metrics = compute_all(y_test, preds, horizon=horizon)
    logger.info(
        "  ✓ {} → DA%={:.1f}  Sharpe={:.3f}  IC={:.3f}",
        cfg.cell_id, metrics["DA%"], metrics["Sharpe"], metrics["IC"]
    )

    is_degen = metrics.get("F1", 1.0) < 0.01 and abs(metrics.get("DA%", 50.0) - 50.0) < 1.5
    metrics["degenerate"] = is_degen
    if is_degen:
        logger.warning("  ⚠ Degenerate prediction (constant output) for {} at {}D", cfg.cell_id, horizon)

    if cache_dir is not None:
        pred_dir = Path(cache_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"{cfg.cell_id}__seed{seed}__{horizon}d.npy"
        np.save(str(pred_file), preds)
        truth_file = pred_dir / f"truth__{horizon}d.npy"
        if not truth_file.exists():
            np.save(str(truth_file), y_test)

    return metrics


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
            device=device,
        )

    elif cfg.model_name == "cnn_lstm":
        p = params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3})
        return dict(
            input_dim=input_dim,
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
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

    else:
        raise ValueError(f"Early fusion not supported for {cfg.model_name}")