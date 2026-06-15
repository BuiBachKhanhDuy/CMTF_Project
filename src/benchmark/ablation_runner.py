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
    RandomForestRegressor_Wrapper,
    CNNLSTMPredictor,
)
from .fusion_wrappers import EarlyFusionWrapper, LateFusionWrapper, HybridFusionWrapper
from .metrics import compute_all
from .baseline_hpo import get_default_baseline_hpo_params
from src.pipeline.news_encoder import (
    SENTIMENT_FEATURE_COLUMNS as _PIPELINE_SENTIMENT_COLS,
    SENTIMENT_TRACE_COLUMNS as _PIPELINE_SENTIMENT_TRACE_COLS,
)

# Use the pipeline's actual column names for stripping sentiment features.
# SENTIMENT_TRACE_COLUMNS includes the missing flag — strip all of them.
SENTIMENT_FEATURE_COLUMNS = list(_PIPELINE_SENTIMENT_TRACE_COLS)


def _config_hash(cfg: AblationConfig) -> str:
    """Short hash for cache key."""
    return hashlib.md5(cfg.cell_id.encode()).hexdigest()[:10]


def _strip_sentiment_features(
    market_windows: np.ndarray,
    market_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Remove sentiment columns from market windows."""
    keep_idx = [i for i, c in enumerate(market_cols) if c not in SENTIMENT_FEATURE_COLUMNS]
    return market_windows[:, :, keep_idx], [market_cols[i] for i in keep_idx]


def _apply_sentiment_weighting(news_embs: np.ndarray, market_windows: np.ndarray, market_cols: list[str]) -> np.ndarray:
    """Multiply each bar's news_emb by that bar's own abs(sentiment_mean)."""
    if "sentiment_mean" not in market_cols:
        return news_embs
    idx = market_cols.index("sentiment_mean")
    # Per-bar weighting: each bar's news embedding scaled by its own sentiment
    sent_weight = np.abs(market_windows[:, :, idx])  # (N, S)
    sent_weight = np.clip(sent_weight, 0.1, 5.0)[..., np.newaxis]  # (N, S, 1)
    return news_embs * sent_weight


def _build_encoder(cfg: AblationConfig, input_dim: int, device: str, chronos=None, hpo_params=None, seed: int = 42):
    """Instantiate the base encoder for this config."""
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
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")


def _train_encoder(
    encoder,
    cfg: AblationConfig,
    mw_train: np.ndarray,
    y_train: np.ndarray,
    mw_val: np.ndarray,
    y_val: np.ndarray,
    splits: dict,
    horizon: int = 1,
):
    """Train an encoder on market data.

    warmup_epochs scales with horizon to prevent constant-prediction degeneration
    on long-horizon targets where the gradient signal is weak early in training.
    """
    if cfg.model_name == "rf":
        encoder.fit(mw_train, y_train)
    else:
        # min(horizon, 10): 1/5/10 for 1D/5D/20D.
        # horizon//2 gave only 2 epochs at 5D — insufficient to prevent
        # constant-prediction degeneration on the weak weekly signal.
        warmup_epochs = min(horizon, 10)
        encoder.fit(mw_train, y_train, mw_val, y_val, warmup_epochs=warmup_epochs)


def _clone_encoder(encoder):
    """Deep-copy a trained encoder to reuse weights without mutation."""
    return copy.deepcopy(encoder)


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
    """Train and evaluate one ablation cell.

    The encoder is trained once and reused for late/hybrid fusion.
    Only early fusion requires a separate encoder (expanded input_dim).
    """
    assert cfg.is_valid(), f"Invalid config: {cfg}"
    logger.info("▶ Running cell: {} (seed={})", cfg.cell_id, seed)

    # Seed model randomness (data splits are deterministic from dataset cache)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Prepare data based on config ---
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

    # --- Bug 1 fix: Select news array based on news_scope ---
    if cfg.news_scope == "all" and "news_embs_all" in splits["train"]:
        ne_train = splits["train"]["news_embs_all"][:, :, :768].copy()
        ne_val = splits["val"]["news_embs_all"][:, :, :768].copy()
        ne_test = splits["test"]["news_embs_all"][:, :, :768].copy()
        nm_train = splits["train"].get("news_masks_all", nm_train)
        nm_val = splits["val"].get("news_masks_all", nm_val)
        nm_test = splits["test"].get("news_masks_all", nm_test)

    # FIX: Handle sentiment data correctly to avoid double-dipping
    if cfg.sentiment_mode == "weighted_emb":
        # Step 1: Apply sentiment weighting to news embeddings FIRST
        ne_train = _apply_sentiment_weighting(ne_train, splits["train"]["market_windows"], market_cols)
        ne_val = _apply_sentiment_weighting(ne_val, splits["val"]["market_windows"], market_cols)
        ne_test = _apply_sentiment_weighting(ne_test, splits["test"]["market_windows"], market_cols)
        
        # Step 2: Strip sentiment features from market_windows immediately after
        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)
    elif cfg.sentiment_mode == "none":
        # Strip sentiment if not using it
        mw_train, cols = _strip_sentiment_features(mw_train, cols)
        mw_val, _ = _strip_sentiment_features(mw_val, market_cols)
        mw_test, _ = _strip_sentiment_features(mw_test, market_cols)
    # else: sentiment_mode == "scalars" → keep sentiment in market_windows

    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    input_dim = mw_train.shape[-1] if mw_train.ndim == 3 else 1

    # --- Train encoder once (reused by none/late/hybrid) ---
    if cfg.fusion_type != "early":
        encoder = _build_encoder(cfg, input_dim, device, chronos, hpo_params, seed)
        _train_encoder(encoder, cfg, mw_train, y_train, mw_val, y_val, splits, horizon=horizon)

    # --- Dispatch by fusion type ---
    if cfg.fusion_type == "none":
        preds = encoder.predict_market_only(mw_test)

    elif cfg.fusion_type == "early":
        wrapper = EarlyFusionWrapper(
            encoder_cls=type(_build_encoder(cfg, input_dim, device, chronos, hpo_params, seed)),
            encoder_kwargs=_encoder_init_kwargs(cfg, input_dim, device, chronos, hpo_params, seed),
            news_dim=768,
        )
        # Extract HPO params for early fusion training
        params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()
        p = params.get(cfg.model_name, {})
        warmup_epochs = min(horizon, 10)
        
        wrapper.fit(
            mw_train, ne_train, y_train, mw_val, ne_val, y_val,
            epochs=100,
            batch_size=p.get("batch_size", 32),
            learning_rate=p.get("lr", 1e-3),
            warmup_epochs=warmup_epochs,
        )
        preds = wrapper.predict(mw_test, ne_test)

    elif cfg.fusion_type == "late":
        freeze_encoder = getattr(cfg, "freeze_encoder", True)
        wrapper = LateFusionWrapper(encoder=encoder, news_dim=768, device=device, horizon=horizon, freeze_encoder=freeze_encoder)
        # Extract HPO params for news training
        params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()
        p_news = params.get("news", {}) if "news" in params else {}
        wrapper.fit(
            mw_train, ne_train, y_train, mw_val, ne_val, y_val,
            news_mask_train=nm_train, news_mask_val=nm_val,
            epochs_news=p_news.get("epochs", 30),
            batch_size_news=p_news.get("batch_size", 32),
            lr_news=p_news.get("lr", 1e-3),
            patience_news=p_news.get("patience", 8),
        )
        preds = wrapper.predict(mw_test, ne_test, nm_test)

    elif cfg.fusion_type == "hybrid":
        freeze_encoder = getattr(cfg, "freeze_encoder", True)
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
            freeze_encoder=freeze_encoder,
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

    # --- Compute metrics ---
    metrics = compute_all(y_test, preds, horizon=horizon)
    logger.info("  ✓ {} → DA%={:.1f}  Sharpe={:.3f}  IC={:.3f}", cfg.cell_id, metrics["DA%"], metrics["Sharpe"], metrics["IC"])

    # --- Degeneration detection: flag cells where the model collapsed to constant prediction ---
    is_degen = metrics.get("F1", 1.0) < 0.01 and abs(metrics.get("DA%", 50.0) - 50.0) < 1.5
    metrics["degenerate"] = is_degen
    if is_degen:
        logger.warning("  ⚠ Degenerate prediction (constant output) for {} at {}D", cfg.cell_id, horizon)

    # --- Save predictions for post-hoc statistical testing ---
    if cache_dir is not None:
        pred_dir = Path(cache_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"{cfg.cell_id}__seed{seed}__{horizon}d.npy"
        np.save(str(pred_file), preds)
        truth_file = pred_dir / f"truth__{horizon}d.npy"
        if not truth_file.exists():
            np.save(str(truth_file), y_test)

    return metrics


def _encoder_init_kwargs(cfg: AblationConfig, input_dim: int, device: str, chronos=None, hpo_params=None, seed: int = 42) -> dict:
    """Return the kwargs needed to re-instantiate the encoder class (for EarlyFusionWrapper)."""
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()

    if cfg.model_name == "lstm":
        p = params["lstm"]
        return dict(input_dim=input_dim, hidden_dim=p["hidden_dim"], num_layers=p["num_layers"], dropout=p["dropout"], device=device)
    elif cfg.model_name == "cnn_lstm":
        p = params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3})
        return dict(input_dim=input_dim, hidden_dim=p.get("hidden_dim", 64), num_layers=p.get("num_layers", 2), dropout=p.get("dropout", 0.3), device=device)
    elif cfg.model_name == "rf":
        p = params["rf"]
        return dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_split=p["min_samples_split"], max_features=p["max_features"], random_state=seed)
    else:
        raise ValueError(f"Early fusion not supported for {cfg.model_name}")
