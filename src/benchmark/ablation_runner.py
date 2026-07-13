"""Ablation runner: train + evaluate a single AblationConfig cell.

Stateless function — takes config, data splits, returns metrics dict.

Design:
1. none   -> runner trains a backbone model directly
2. early  -> EarlyFusionWrapper owns encoder training
3. late   -> LateFusionWrapper owns encoder training
4. cmtf   -> standalone Cross-Modal Temporal Fusion predictor

Refactor notes:
- FIX #5: encoder cache keys now include feature/schema/config/split identity
  to prevent invalid cache reuse across incompatible ablation settings.
- FIX #6: anchor predictions now come from a shared market-only baseline
  per (backbone, horizon, seed, processed market input definition, split),
  making ModalDisagreement / CompositeScore comparable across ablations.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from .ablation_config import (
    AblationConfig,
    BACKBONE_MODELS,
    CMTF_MODEL,
)
from .baseline_models import (
    LSTMPredictor,
    CNNLSTMPredictor,
)
from .gpt4ts_encoder import GPT4TSPredictor
from .chronos_encoder import ChronosAdapter
from .fusion_wrappers import EarlyFusionWrapper, LateFusionWrapper
from .hybrid_fusion import HybridFusionPredictor, build_market_encoder
from .metrics import compute_all, compute_composite_metrics, flag_degenerate
from .baseline_hpo import get_default_baseline_hpo_params
from src.pipeline.news_encoder import (
    SENTIMENT_TRACE_COLUMNS as _PIPELINE_SENTIMENT_TRACE_COLS,
)

SENTIMENT_FEATURE_COLUMNS = list(_PIPELINE_SENTIMENT_TRACE_COLS)
_ALLOWED_WRAPPER_NEWS_DIMS = {768, 128}

# Bump this whenever the target-scaling scheme OR the encoder training recipe
# changes so that on-disk encoder/anchor/prediction caches trained under a
# different scheme are invalidated.
# unitstd_v2 = unit-std target scaling (1/train_std) for all trainable models
# (matching the Phase-1 harness, run_model_benchmark.py) PLUS a unified canonical
# encoder training recipe (cfg.market_epochs / cfg.market_patience) applied
# identically across none/late/cmtf so a shared encoder is bit-identical
# regardless of which cell trains it first. The recipe is also folded into the
# encoder/anchor cache keys (recipe_sig) so mismatched recipes never share.
_SCALING_VERSION = "unitstd_v2"

# ---------------------------------------------------------------------------
# Late-fusion OOF (Phase 1) budget.
#
# LateFusionWrapper.fit builds leakage-free residual targets by refitting the
# market backbone once per CV fold (TimeSeriesSplit). For the heavy foundation
# backbones (GPT4TS fine-tunes a transformer block; Chronos runs a large T5
# encoder) this Phase-1 loop is by far the dominant cost of a late-fusion cell —
# the backbone is trained `n_splits` extra times purely to estimate OOF
# residuals. Those residual targets only need to be leakage-free and roughly
# calibrated, NOT a fully-converged model, so we cut the OOF budget (fewer CV
# folds + capped per-fold epochs). The FINAL full-data encoder fit (Phase 2) is
# untouched, so deployed accuracy is unaffected.
#
# APPLES-TO-APPLES: this budget is applied UNIFORMLY to every backbone (lstm,
# cnn_lstm, gpt4ts, chronos). The fusion comparison contrasts late vs cmtf vs
# none across backbones, so the residual-generation protocol that feeds each
# late-fusion news branch must be identical — otherwise an LSTM late cell would
# be trained on residuals from a different CV protocol (5 folds / 50 epochs)
# than a GPT4TS late cell (3 folds / 15 epochs), confounding "backbone" with
# "OOF protocol". Only the wall-clock cost differs (foundation backbones are far
# heavier per fit); the procedure is the same.
_OOF_N_SPLITS = 3
_OOF_MAX_EPOCHS = 15

# Per-process cache of the canonical test targets used for each horizon within THIS
# run. All cells for one horizon share the exact same `splits["test"]["targets"]`
# object (computed once in `_extract_and_split`), so a mismatch here means a real
# intra-run data/splits bug, unlike the on-disk truth cache below which can go
# legitimately stale across separate runs as the data pipeline evolves.
_RUN_TRUTH_CACHE: dict[int, np.ndarray] = {}

_ENCODER_CLS = {
    "lstm": LSTMPredictor,
    "cnn_lstm": CNNLSTMPredictor,
    "gpt4ts": GPT4TSPredictor,
    "chronos": ChronosAdapter,
}

# Stores state_dict snapshots for trained torch encoders.
_ENCODER_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}

# Stores shared market-only anchor predictions.
_ANCHOR_PRED_CACHE: dict[tuple, np.ndarray] = {}

# Stores shared out-of-fold (OOF) market predictions used by late fusion.
_OOF_PRED_CACHE: dict[tuple, np.ndarray] = {}


# ============================================================================
# Cache control
# ============================================================================

def clear_encoder_cache() -> None:
    _ENCODER_CACHE.clear()
    _ANCHOR_PRED_CACHE.clear()
    _OOF_PRED_CACHE.clear()


def clear_anchor_cache() -> None:
    _ANCHOR_PRED_CACHE.clear()


# ============================================================================
# Stable hashing helpers
# ============================================================================

def _freeze_for_hash(obj: Any) -> Any:
    """Convert nested structures into deterministic hashable forms."""
    if isinstance(obj, dict):
        return tuple((k, _freeze_for_hash(v)) for k, v in sorted(obj.items(), key=lambda x: str(x[0])))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_for_hash(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return ("__ndarray__", obj.shape, str(obj.dtype), hashlib.md5(obj.tobytes()).hexdigest())
    return obj


def _safe_hash_obj(obj: Any) -> str:
    payload = repr(_freeze_for_hash(obj)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _time_bounds_from_split(split: dict[str, np.ndarray]) -> tuple[str | None, str | None]:
    """Extract stable time bounds from split if available."""
    if "times" not in split:
        return None, None

    arr = np.asarray(split["times"])
    if arr.size == 0:
        return None, None

    # Convert to pandas-compatible string representation if possible
    try:
        tmin = str(arr.min())
        tmax = str(arr.max())
        return tmin, tmax
    except Exception:
        return None, None


# ============================================================================
# Encoder cache key helpers
# ============================================================================

def _resolve_market_encoder_name(cfg: AblationConfig) -> str | None:
    if cfg.fusion_type == "cmtf":
        return cfg.market_encoder_name
    if cfg.fusion_type in ("none", "late") and cfg.model_name in BACKBONE_MODELS:
        return cfg.model_name
    return None


def _resolve_anchor_backbone_name(cfg: AblationConfig) -> str | None:
    if cfg.fusion_type == "none":
        return cfg.model_name if cfg.model_name in BACKBONE_MODELS else None
    if cfg.fusion_type in {"early", "late"}:
        return cfg.model_name if cfg.model_name in BACKBONE_MODELS else None
    if cfg.fusion_type == "cmtf":
        return cfg.market_encoder_name
    return None


def _config_hash(cfg: AblationConfig) -> str:
    return hashlib.md5(cfg.cell_id.encode()).hexdigest()[:10]


def _supports_early_fusion(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm", "gpt4ts", "chronos"}


def _supports_late_fusion(model_name: str) -> bool:
    return model_name in {"lstm", "cnn_lstm", "gpt4ts", "chronos"}


def _extract_encoder_cache_params(
    cfg: AblationConfig,
    params: dict,
    encoder_name: str,
) -> dict[str, Any]:
    """Normalize encoder-defining hyperparameters for safe cache identity."""
    p = params.get(encoder_name, {})

    if encoder_name == "lstm":
        return {
            "encoder_name": "lstm",
            "hidden_dim": p.get("hidden_dim", 64),
            "num_layers": p.get("num_layers", 2),
            "dropout": p.get("dropout", 0.3),
            "sign_penalty_weight": p.get("sign_penalty_weight", 0.02),
        }

    if encoder_name == "cnn_lstm":
        return {
            "encoder_name": "cnn_lstm",
            "num_filters": p.get("num_filters", p.get("hidden_dim", 64)),
            "hidden_dim": p.get("hidden_dim", 64),
            "num_layers": p.get("num_layers", 2),
            "dropout": p.get("dropout", 0.3),
            "sign_penalty_weight": p.get("sign_penalty_weight", 0.02),
        }

    if encoder_name == "gpt4ts":
        return {
            "encoder_name": "gpt4ts",
            "hidden_dim": p.get("hidden_dim", 64),
            "num_layers": p.get("num_layers", 3),
            "patch_length": p.get("patch_length", 6),
            "dropout": p.get("dropout", 0.3),
            "sign_penalty_weight": p.get("sign_penalty_weight", 0.02),
        }

    if encoder_name == "chronos":
        return {
            "encoder_name": "chronos",
            "model_name": "amazon/chronos-t5-small",
            "dropout": p.get("dropout", 0.3),
        }

    return {"encoder_name": encoder_name}


def _encoder_recipe(
    cfg: AblationConfig,
    params: dict,
    enc_name: str,
    horizon: int,
) -> dict[str, Any]:
    """Canonical market-encoder training recipe.

    Returns the exact fit-kwargs used to train a torch market encoder, applied
    IDENTICALLY across none / late / cmtf-stage1 so the shared encoder cache is
    bit-identical regardless of which fusion cell trains it first. The recipe is
    also folded into the encoder/anchor cache keys (recipe_sig) so that two cells
    only share a cached encoder when their training recipe truly matches.
    """
    p = params.get(enc_name, {}) if params else {}
    return {
        "epochs": cfg.market_epochs,
        "batch_size": p.get("batch_size", 32),
        "learning_rate": p.get("lr", 1e-3),
        "patience": cfg.market_patience,
        "warmup_epochs": min(horizon, 10),
    }


def _oof_budget(base_recipe: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Resolve the late-fusion OOF (Phase 1) CV budget.

    Returns ``(n_splits, oof_recipe)``, applied UNIFORMLY to every backbone so
    the fusion comparison stays apple-to-apple: the leakage-free residual targets
    that feed each late-fusion news branch are generated with an identical CV
    configuration (folds + per-fold epochs) regardless of backbone. The OOF folds
    train with a capped-epoch recipe (leakage-free residuals don't need a
    fully-converged per-fold model); the final full-data encoder fit (Phase 2)
    still uses ``base_recipe``, so deployed accuracy is unchanged.
    """
    oof_recipe = dict(base_recipe)
    full_epochs = int(oof_recipe.get("epochs", _OOF_MAX_EPOCHS))
    capped_epochs = max(1, min(full_epochs, _OOF_MAX_EPOCHS))
    oof_recipe["epochs"] = capped_epochs
    # Keep the direction-loss warmup strictly shorter than the capped budget so
    # the sign-aware objective still activates within the reduced schedule.
    warmup = oof_recipe.get("warmup_epochs")
    if warmup is not None:
        oof_recipe["warmup_epochs"] = max(0, min(int(warmup), capped_epochs - 1))
    return _OOF_N_SPLITS, oof_recipe


def _encoder_cache_key(
    enc_name: str,
    horizon: int,
    seed: int,
    market_cols: list[str],
    sentiment_mode: str,
    encoder_params: dict[str, Any],
    train_shape: tuple,
    val_shape: tuple,
    train_time_min: str | None = None,
    train_time_max: str | None = None,
    val_time_min: str | None = None,
    val_time_max: str | None = None,
    recipe: dict | None = None,
) -> tuple:
    feature_sig = _safe_hash_obj(tuple(market_cols))
    param_sig = _safe_hash_obj(encoder_params)
    recipe_sig = _safe_hash_obj(recipe or {})
    split_sig = _safe_hash_obj(
        {
            "train_shape": train_shape,
            "val_shape": val_shape,
            "train_time_min": train_time_min,
            "train_time_max": train_time_max,
            "val_time_min": val_time_min,
            "val_time_max": val_time_max,
        }
        )

    return (
        "encoder",
        _SCALING_VERSION,
        enc_name,
        horizon,
        seed,
        sentiment_mode,
        feature_sig,
        param_sig,
        recipe_sig,
        split_sig,
    )


def _anchor_cache_key(
    enc_name: str,
    horizon: int,
    seed: int,
    market_cols: list[str],
    sentiment_mode: str,
    encoder_params: dict[str, Any],
    train_shape: tuple,
    val_shape: tuple,
    test_shape: tuple,
    train_time_min: str | None = None,
    train_time_max: str | None = None,
    val_time_min: str | None = None,
    val_time_max: str | None = None,
    test_time_min: str | None = None,
    test_time_max: str | None = None,
    recipe: dict | None = None,
) -> tuple:
    feature_sig = _safe_hash_obj(tuple(market_cols))
    param_sig = _safe_hash_obj(encoder_params)
    recipe_sig = _safe_hash_obj(recipe or {})
    split_sig = _safe_hash_obj(
        {
            "train_shape": train_shape,
            "val_shape": val_shape,
            "test_shape": test_shape,
            "train_time_min": train_time_min,
            "train_time_max": train_time_max,
            "val_time_min": val_time_min,
            "val_time_max": val_time_max,
            "test_time_min": test_time_min,
            "test_time_max": test_time_max,
        }
        )

    return (
        "anchor",
        _SCALING_VERSION,
        enc_name,
        horizon,
        seed,
        sentiment_mode,
        feature_sig,
        param_sig,
        recipe_sig,
        split_sig,
    )

def _oof_cache_key(
    enc_name: str,
    horizon: int,
    seed: int,
    market_cols: list[str],
    sentiment_mode: str,
    encoder_params: dict[str, Any],
    train_shape: tuple,
    val_shape: tuple,
    n_splits: int,
    gap: int,
    train_time_min: str | None = None,
    train_time_max: str | None = None,
    val_time_min: str | None = None,
    val_time_max: str | None = None,
    recipe: dict | None = None,
) -> tuple:
    """Cache key for out-of-fold market predictions consumed by late fusion.

    OOF predictions are a pure function of the market encoder training recipe and
    the (train) market features/targets, plus the CV configuration (n_splits/gap).
    They are independent of the news branch, so late-fusion cells that share a
    backbone + market inputs (e.g. real-matched / real-full / placebo variants)
    can reuse the same OOF tensor instead of refitting the backbone once per fold.
    """
    feature_sig = _safe_hash_obj(tuple(market_cols))
    param_sig = _safe_hash_obj(encoder_params)
    recipe_sig = _safe_hash_obj(recipe or {})
    split_sig = _safe_hash_obj(
        {
            "train_shape": train_shape,
            "val_shape": val_shape,
            "train_time_min": train_time_min,
            "train_time_max": train_time_max,
            "val_time_min": val_time_min,
            "val_time_max": val_time_max,
        }
        )

    return (
        "oof",
        _SCALING_VERSION,
        enc_name,
        horizon,
        seed,
        sentiment_mode,
        feature_sig,
        param_sig,
        recipe_sig,
        n_splits,
        gap,
        split_sig,
    )


_ENCODER_CACHE_SUBDIR = "encoders"
_ANCHOR_CACHE_SUBDIR = "anchors"
_OOF_CACHE_SUBDIR = "oof"


def _key_to_filename(key: tuple, suffix: str) -> str:
    return hashlib.md5(repr(key).encode("utf-8")).hexdigest() + suffix


def _has_cached_encoder(key: tuple, cache_dir: Path | None = None) -> bool:
    """Check memory first, then disk, without loading anything."""
    if key in _ENCODER_CACHE:
        return True
    if cache_dir is not None:
        path = Path(cache_dir) / _ENCODER_CACHE_SUBDIR / _key_to_filename(key, ".pt")
        return path.exists()
    return False


def _save_encoder_to_cache(key: tuple, encoder, cache_dir: Path | None = None) -> None:
    if not hasattr(encoder, "state_dict"):
        return
    state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    logger.info("Saving encoder to memory cache | key={}", key)
    _ENCODER_CACHE[key] = state

    if cache_dir is not None:
        enc_dir = Path(cache_dir) / _ENCODER_CACHE_SUBDIR
        enc_dir.mkdir(parents=True, exist_ok=True)
        path = enc_dir / _key_to_filename(key, ".pt")
        torch.save(state, path)
        logger.info("Persisted encoder to disk | path={}", path)


def _load_encoder_from_cache(encoder, key: tuple, cache_dir: Path | None = None) -> None:
    if key in _ENCODER_CACHE:
        logger.info("Loading encoder from memory cache | key={}", key)
        encoder.load_state_dict(_ENCODER_CACHE[key])
        return

    if cache_dir is not None:
        path = Path(cache_dir) / _ENCODER_CACHE_SUBDIR / _key_to_filename(key, ".pt")
        if path.exists():
            logger.info("Loading encoder from disk cache | path={}", path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            encoder.load_state_dict(state)
            _ENCODER_CACHE[key] = state
            return

    raise KeyError(f"Encoder cache miss for key={key}")


def _save_anchor_to_cache(key: tuple, anchor_pred: np.ndarray, cache_dir: Path | None = None) -> None:
    _ANCHOR_PRED_CACHE[key] = anchor_pred.copy()
    if cache_dir is not None:
        anc_dir = Path(cache_dir) / _ANCHOR_CACHE_SUBDIR
        anc_dir.mkdir(parents=True, exist_ok=True)
        path = anc_dir / _key_to_filename(key, ".npy")
        np.save(path, anchor_pred)
        logger.info("Persisted anchor prediction to disk | path={}", path)


def _load_anchor_from_cache(key: tuple, cache_dir: Path | None = None) -> np.ndarray | None:
    if key in _ANCHOR_PRED_CACHE:
        return _ANCHOR_PRED_CACHE[key].copy()
    if cache_dir is not None:
        path = Path(cache_dir) / _ANCHOR_CACHE_SUBDIR / _key_to_filename(key, ".npy")
        if path.exists():
            arr = np.load(path)
            _ANCHOR_PRED_CACHE[key] = arr.copy()
            return arr.copy()
    return None


def _has_cached_oof(key: tuple, cache_dir: Path | None = None) -> bool:
    """Check memory first, then disk, without loading the array."""
    if key in _OOF_PRED_CACHE:
        return True
    if cache_dir is not None:
        path = Path(cache_dir) / _OOF_CACHE_SUBDIR / _key_to_filename(key, ".npy")
        return path.exists()
    return False


def _save_oof_to_cache(key: tuple, oof_pred: np.ndarray, cache_dir: Path | None = None) -> None:
    _OOF_PRED_CACHE[key] = oof_pred.copy()
    if cache_dir is not None:
        oof_dir = Path(cache_dir) / _OOF_CACHE_SUBDIR
        oof_dir.mkdir(parents=True, exist_ok=True)
        path = oof_dir / _key_to_filename(key, ".npy")
        np.save(path, oof_pred)
        logger.info("Persisted OOF market prediction to disk | path={}", path)


def _load_oof_from_cache(key: tuple, cache_dir: Path | None = None) -> np.ndarray | None:
    if key in _OOF_PRED_CACHE:
        return _OOF_PRED_CACHE[key].copy()
    if cache_dir is not None:
        path = Path(cache_dir) / _OOF_CACHE_SUBDIR / _key_to_filename(key, ".npy")
        if path.exists():
            arr = np.load(path)
            _OOF_PRED_CACHE[key] = arr.copy()
            return arr.copy()
    return None


# ============================================================================
# Cell-level prediction cache (Layer 3)
# ============================================================================

def _cell_prediction_paths(
    cfg: AblationConfig, seed: int, horizon: int, cache_dir: Path
) -> tuple[Path, Path]:
    short_id = _config_hash(cfg)
    pred_dir = Path(cache_dir) / "predictions"
    npy = pred_dir / f"{short_id}__seed{seed}__{horizon}d.npy"
    js = pred_dir / f"{short_id}__seed{seed}__{horizon}d.json"
    return npy, js


def _try_load_cell_prediction(
    cfg: AblationConfig,
    seed: int,
    horizon: int,
    cache_dir: Path | None,
    data_sig: str,
    news_sig: str,
) -> np.ndarray | None:
    """Return cached test predictions for this exact cell iff the on-disk
    provenance sidecar matches the *current* data, scaling scheme, and news
    configuration. Any mismatch (or missing sidecar) is treated as a miss so a
    stale artifact is never silently reused — the cell is recomputed instead.
    """
    if cache_dir is None:
        return None
    npy, js = _cell_prediction_paths(cfg, seed, horizon, cache_dir)
    if not (npy.exists() and js.exists()):
        return None
    try:
        prov = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        prov.get("scaling_version") != _SCALING_VERSION
        or prov.get("cell_id") != cfg.cell_id
        or prov.get("data_sig") != data_sig
        or prov.get("news_sig") != news_sig
    ):
        return None
    try:
        return np.load(str(npy))
    except Exception:
        return None


# ============================================================================
# Feature / news processing helpers
# ============================================================================

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


# ============================================================================
# Encoder construction / training
# ============================================================================

def _get_encoder_kwargs(
    cfg: AblationConfig,
    input_dim: int,
    device: str,
    chronos_pipeline=None,
    hpo_params=None,
    seed: int = 42,
    target_scale: float = 1.0,
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
            target_scale=target_scale,
            device=device,
        )

    if cfg.model_name == "cnn_lstm":
        p = params.get("cnn_lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3})
        return dict(
            input_dim=input_dim,
            num_filters=p.get("num_filters", p.get("hidden_dim", 64)),
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            target_scale=target_scale,
            device=device,
        )

    if cfg.model_name == "gpt4ts":
        p = params.get("gpt4ts", {"hidden_dim": 64, "num_layers": 3, "dropout": 0.3})
        return dict(
            input_dim=input_dim,
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 3),
            patch_length=p.get("patch_length", 6),
            dropout=p.get("dropout", 0.3),
            sign_penalty_weight=p.get("sign_penalty_weight", 0.02),
            target_scale=target_scale,
            device=device,
        )

    if cfg.model_name == "chronos":
        return dict(
            input_dim=input_dim,
            model_name="amazon/chronos-t5-small",
            dropout=params.get("chronos", {}).get("dropout", 0.3),
            device=device,
            pipeline=chronos_pipeline,
            target_scale=target_scale,
        )

    raise ValueError(f"Unknown or unsupported backbone model for ablation_runner: {cfg.model_name}")


def _build_encoder(
    cfg: AblationConfig,
    input_dim: int,
    device: str,
    chronos_pipeline=None,
    hpo_params=None,
    seed: int = 42,
    target_scale: float = 1.0,
):
    encoder_cls = _ENCODER_CLS.get(cfg.model_name)
    if encoder_cls is None:
        raise ValueError(f"Unknown or unsupported backbone model for ablation_runner: {cfg.model_name}")
    return encoder_cls(
        **_get_encoder_kwargs(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos_pipeline=chronos_pipeline,
            hpo_params=hpo_params,
            seed=seed,
            target_scale=target_scale,
        )
    )


def _train_encoder(
    encoder,
    mw_train: np.ndarray,
    y_train: np.ndarray,
    mw_val: np.ndarray,
    y_val: np.ndarray,
    horizon: int = 1,
    recipe: dict | None = None,
):
    if recipe is None:
        recipe = {"warmup_epochs": min(horizon, 10)}
    encoder.fit(mw_train, y_train, mw_val, y_val, **recipe)


def _build_shared_encoder_cache_key(
    encoder_name: str,
    cfg: AblationConfig,
    params: dict,
    effective_market_cols: list[str],
    mw_train: np.ndarray,
    mw_val: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    horizon: int,
    seed: int,
) -> tuple:
    encoder_params = _extract_encoder_cache_params(cfg, params, encoder_name)
    recipe = _encoder_recipe(cfg, params, encoder_name, horizon)
    train_time_min, train_time_max = _time_bounds_from_split(splits["train"])
    val_time_min, val_time_max = _time_bounds_from_split(splits["val"])

    return _encoder_cache_key(
        enc_name=encoder_name,
        horizon=horizon,
        seed=seed,
        market_cols=effective_market_cols,
        sentiment_mode=cfg.sentiment_mode,
        encoder_params=encoder_params,
        train_shape=tuple(mw_train.shape),
        val_shape=tuple(mw_val.shape),
        train_time_min=train_time_min,
        train_time_max=train_time_max,
        val_time_min=val_time_min,
        val_time_max=val_time_max,
        recipe=recipe,
    )


def _build_oof_cache_key(
    encoder_name: str,
    cfg: AblationConfig,
    params: dict,
    effective_market_cols: list[str],
    mw_train: np.ndarray,
    mw_val: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    horizon: int,
    seed: int,
    n_splits: int,
    gap: int,
    oof_recipe: dict | None = None,
) -> tuple:
    """Cache key for the leakage-free OOF market predictions used by late fusion.

    Mirrors _build_shared_encoder_cache_key but adds the CV configuration
    (n_splits/gap). OOF predictions depend only on the market encoder recipe and
    the training market features/targets, so they can be reused across late-fusion
    cells that share the same backbone + market inputs (e.g. real-matched /
    real-full / placebo news variants), avoiding a repeated k-fold backbone refit.

    ``oof_recipe`` lets the caller fold the ACTUAL Phase-1 training recipe into
    the key. Foundation backbones train OOF folds with a cheaper (capped-epoch)
    recipe than the final encoder fit, so the OOF cache must key off that reduced
    recipe — otherwise a cell run under a different OOF budget could silently
    reuse a stale tensor. When omitted it defaults to the full encoder recipe,
    keeping custom-baseline keys byte-identical to before.
    """
    encoder_params = _extract_encoder_cache_params(cfg, params, encoder_name)
    recipe = oof_recipe if oof_recipe is not None else _encoder_recipe(cfg, params, encoder_name, horizon)
    train_time_min, train_time_max = _time_bounds_from_split(splits["train"])
    val_time_min, val_time_max = _time_bounds_from_split(splits["val"])

    return _oof_cache_key(
        enc_name=encoder_name,
        horizon=horizon,
        seed=seed,
        market_cols=effective_market_cols,
        sentiment_mode=cfg.sentiment_mode,
        encoder_params=encoder_params,
        train_shape=tuple(mw_train.shape),
        val_shape=tuple(mw_val.shape),
        n_splits=n_splits,
        gap=gap,
        train_time_min=train_time_min,
        train_time_max=train_time_max,
        val_time_min=val_time_min,
        val_time_max=val_time_max,
        recipe=recipe,
    )


def _get_shared_anchor_pred(
    cfg: AblationConfig,
    splits: dict[str, dict[str, np.ndarray]],
    mw_train: np.ndarray,
    mw_val: np.ndarray,
    mw_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    effective_market_cols: list[str],
    horizon: int,
    device: str,
    chronos_pipeline,
    hpo_params: dict,
    seed: int,
    cache_dir: Path | None = None,
) -> np.ndarray | None:
    """Get a shared market-only anchor prediction for comparable disagreement diagnostics."""
    anchor_name = _resolve_anchor_backbone_name(cfg)
    if anchor_name is None:
        return None

    encoder_params = _extract_encoder_cache_params(cfg, hpo_params, anchor_name)
    recipe = _encoder_recipe(cfg, hpo_params, anchor_name, horizon)
    train_time_min, train_time_max = _time_bounds_from_split(splits["train"])
    val_time_min, val_time_max = _time_bounds_from_split(splits["val"])
    test_time_min, test_time_max = _time_bounds_from_split(splits["test"])

    anchor_key = _anchor_cache_key(
        enc_name=anchor_name,
        horizon=horizon,
        seed=seed,
        market_cols=effective_market_cols,
        sentiment_mode=cfg.sentiment_mode,
        encoder_params=encoder_params,
        train_shape=tuple(mw_train.shape),
        val_shape=tuple(mw_val.shape),
        test_shape=tuple(mw_test.shape),
        train_time_min=train_time_min,
        train_time_max=train_time_max,
        val_time_min=val_time_min,
        val_time_max=val_time_max,
        test_time_min=test_time_min,
        test_time_max=test_time_max,
        recipe=recipe,
    )

    cached_anchor = _load_anchor_from_cache(anchor_key, cache_dir)
    if cached_anchor is not None:
        logger.info("Loading anchor prediction from cache | key={}", anchor_key)
        return cached_anchor

    input_dim = mw_train.shape[-1] if mw_train.ndim == 3 else 1

    # Unit-std target scaling (matches Phase-1 harness); anchor must use the
    # same scheme as the main cell for comparable diagnostics.
    _train_std = float(np.std(np.asarray(y_train, dtype=np.float64), ddof=1))
    target_scale = 1.0 / max(_train_std, 1e-6)

    anchor_cfg = dataclasses.replace(cfg, model_name=anchor_name, fusion_type="none")

    encoder_cache_key = _build_shared_encoder_cache_key(
        encoder_name=anchor_name,
        cfg=anchor_cfg,
        params=hpo_params,
        effective_market_cols=effective_market_cols,
        mw_train=mw_train,
        mw_val=mw_val,
        splits=splits,
        horizon=horizon,
        seed=seed,
    )

    encoder = _build_encoder(
        anchor_cfg,
        input_dim=input_dim,
        device=device,
        chronos_pipeline=chronos_pipeline,
        hpo_params=hpo_params,
        seed=seed,
        target_scale=target_scale,
    )

    if _has_cached_encoder(encoder_cache_key, cache_dir):
        _load_encoder_from_cache(encoder, encoder_cache_key, cache_dir)
    else:
        _train_encoder(
            encoder,
            mw_train,
            y_train,
            mw_val,
            y_val,
            horizon=horizon,
            recipe=recipe,
        )
        _save_encoder_to_cache(encoder_cache_key, encoder, cache_dir)

    anchor_pred = np.asarray(encoder.predict_market_only(mw_test), dtype=np.float32)
    logger.info("Saving anchor prediction to cache | key={}", anchor_key)
    _save_anchor_to_cache(anchor_key, anchor_pred, cache_dir)
    return anchor_pred


# ============================================================================
# Main ablation cell runner
# ============================================================================

def run_ablation_cell(
    cfg: AblationConfig,
    splits: dict[str, dict[str, np.ndarray]],
    market_cols: list[str],
    horizon: int,
    device: str = "cpu",
    chronos_pipeline=None,
    seed: int = 42,
        cache_dir: Path | None = None,
    use_cache: bool = True,
    hpo_params: dict | None = None,
    artifacts: dict | None = None,
    compute_gate: bool = False,
    gate_conviction: bool = True,
    gate_coverage: float | None = 0.25,
) -> dict[str, float]:
    assert cfg.is_valid(), f"Invalid config: {cfg}"
    logger.info("▶ Running cell: {} (seed={})", cfg.cell_id, seed)
    _cell_start_time = time.perf_counter()

    # The validation-calibrated confidence-gate / conviction-sizing decision
    # policy (src/benchmark/decision_policy.py) needs fresh val predictions,
    # which only exist on a live training/inference pass (the on-disk cache
    # only stores TEST predictions, see _try_load_cell_prediction). Force a
    # fresh pass whenever gating is requested, and make sure `artifacts` is a
    # real dict so every fusion-type branch below (which only populates it
    # `if artifacts is not None`) captures val_pred/test_pred for us.
    if compute_gate:
        if use_cache:
            logger.info(
                "compute_gate=True: forcing use_cache=False for {} (need fresh val predictions)",
                cfg.cell_id,
            )
            use_cache = False
        if artifacts is None:
            artifacts = {}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mw_train = splits["train"]["market_windows"].copy()
    mw_val = splits["val"]["market_windows"].copy()
    mw_test = splits["test"]["market_windows"].copy()

    ne_train, ne_val, ne_test, nm_train, nm_val, nm_test = _get_news_arrays(cfg, splits)

    mw_train, mw_val, mw_test, ne_train, ne_val, ne_test, effective_market_cols = _apply_sentiment_mode(
        cfg,
        mw_train,
        mw_val,
        mw_test,
        ne_train,
        ne_val,
        ne_test,
        market_cols,
    )

    # ------------------------------------------------------------------
    # B3 shuffled-news placebo control: destroy the market<->news row
    # alignment on TRAIN/VAL only (test is left intact). If CMTF's news lift
    # is real, this cell should collapse toward the market-only baseline.
    # ------------------------------------------------------------------
    if getattr(cfg, "shuffle_news", False):
        _news_rng = np.random.default_rng(seed + 777)
        _perm_tr = _news_rng.permutation(ne_train.shape[0])
        _perm_v = _news_rng.permutation(ne_val.shape[0])
        ne_train = ne_train[_perm_tr]
        ne_val = ne_val[_perm_v]
        if nm_train is not None:
            nm_train = np.asarray(nm_train)[_perm_tr]
        if nm_val is not None:
            nm_val = np.asarray(nm_val)[_perm_v]
        logger.info(
            "B3 shuffle_news: permuted {} train / {} val news rows (placebo)",
            len(_perm_tr), len(_perm_v),
        )

    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    # ------------------------------------------------------------------
    # Unit-std target scaling (A1): scale targets to unit std so that the
    # loss margins, huber_delta and direction_epsilon operate at a
    # consistent scale across every fusion path. predict() divides back
    # by target_scale, so metrics stay in raw return units. This mirrors
    # the Phase-1 harness (run_model_benchmark.py) so that `x::none` cells
    # reproduce the Phase-1 baselines.
    # ------------------------------------------------------------------
    _train_std = float(np.std(np.asarray(y_train, dtype=np.float64), ddof=1))
    target_scale = 1.0 / max(_train_std, 1e-6)
    logger.info(
        "{} {}D target std={:.6f} → target_scale={:.4f}",
        cfg.cell_id, horizon, _train_std, target_scale,
    )

    input_dim = mw_train.shape[-1] if mw_train.ndim == 3 else 1
    params = hpo_params if hpo_params is not None else get_default_baseline_hpo_params()
    wrapper = None
    hybrid_model = None

    # ------------------------------------------------------------------
    # Cell-level prediction cache (Layer 3): if this exact cell was already
    # computed for the current data + scaling + news configuration, reuse its
    # fully-trained predictions and skip ALL training/inference. Signatures are
    # computed unconditionally so they can be written into the provenance
    # sidecar below regardless of hit/miss.
    # ------------------------------------------------------------------
    data_sig = _safe_hash_obj(
        {
            "cols": tuple(effective_market_cols),
            "mw_train": tuple(mw_train.shape),
            "mw_val": tuple(mw_val.shape),
            "mw_test": tuple(mw_test.shape),
            "y_train": np.asarray(y_train, dtype=np.float64),
            "y_val": np.asarray(y_val, dtype=np.float64),
            "y_test": np.asarray(y_test, dtype=np.float64),
            "scaling_version": _SCALING_VERSION,
            "target_scale": round(float(target_scale), 10),
        }
    )
    news_sig = _safe_hash_obj(
        {
            "scope": cfg.news_scope,
            "shuffle": bool(getattr(cfg, "shuffle_news", False)),
            "ne_train_shape": tuple(np.asarray(ne_train).shape),
            "ne_val_shape": tuple(np.asarray(ne_val).shape),
            "ne_test": np.asarray(ne_test, dtype=np.float32),
        }
    )
    preds = None
    if use_cache:
        cached_preds = _try_load_cell_prediction(
            cfg, seed, horizon, cache_dir, data_sig, news_sig
        )
        if cached_preds is not None:
            preds = cached_preds
            logger.info(
                "\u2713 prediction cache hit \u2014 skipping training for {} (seed={}, {}D)",
                cfg.cell_id, seed, horizon,
            )

    # ------------------------------------------------------------------
    # NONE = raw backbone itself
    # ------------------------------------------------------------------
    if preds is not None:
        # Whole-cell prediction cache hit: skip all training/inference.
        pass
    elif cfg.fusion_type == "none":
        if cfg.model_name not in BACKBONE_MODELS:
            raise ValueError(f"fusion_type='none' requires backbone model, got {cfg.model_name}")

        recipe = _encoder_recipe(cfg, params, cfg.model_name, horizon)
        encoder = _build_encoder(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos_pipeline=chronos_pipeline,
            hpo_params=hpo_params,
            seed=seed,
            target_scale=target_scale,
        )

        enc_name = _resolve_market_encoder_name(cfg)
        assert enc_name is not None

        cache_key = _build_shared_encoder_cache_key(
            encoder_name=enc_name,
            cfg=cfg,
            params=params,
            effective_market_cols=effective_market_cols,
            mw_train=mw_train,
            mw_val=mw_val,
            splits=splits,
            horizon=horizon,
            seed=seed,
        )

        if _has_cached_encoder(cache_key, cache_dir):
            _load_encoder_from_cache(encoder, cache_key, cache_dir)
        else:
            _train_encoder(
                encoder,
                mw_train,
                y_train,
                mw_val,
                y_val,
                horizon=horizon,
                recipe=recipe,
            )
            _save_encoder_to_cache(cache_key, encoder, cache_dir)
        preds = encoder.predict_market_only(mw_test)
        if artifacts is not None:
            # Expose val+test predictions from the exact production path so a
            # validation-calibrated decision policy can be layered on top.
            artifacts.update(
                model_kind="none",
                test_pred=np.asarray(preds, dtype=np.float32),
                val_pred=np.asarray(encoder.predict_market_only(mw_val), dtype=np.float32),
            )

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
            encoder_cls=_ENCODER_CLS[cfg.model_name],
            encoder_kwargs=_get_encoder_kwargs(cfg, input_dim, device, chronos_pipeline, hpo_params, seed, target_scale=target_scale),
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
            epochs=cfg.market_epochs,
            batch_size=p.get("batch_size", 32),
            learning_rate=p.get("lr", 1e-3),
            patience=cfg.market_patience,
            warmup_epochs=warmup_epochs,
        )
        preds = wrapper.predict(mw_test, ne_test_f)
        if artifacts is not None:
            artifacts.update(
                model_kind="early",
                test_pred=np.asarray(preds, dtype=np.float32),
                val_pred=np.asarray(wrapper.predict(mw_val, ne_val_f), dtype=np.float32),
            )

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

        enc_name = _resolve_market_encoder_name(cfg)
        assert enc_name is not None

        cache_key = _build_shared_encoder_cache_key(
            encoder_name=enc_name,
            cfg=cfg,
            params=params,
            effective_market_cols=effective_market_cols,
            mw_train=mw_train,
            mw_val=mw_val,
            splits=splits,
            horizon=horizon,
            seed=seed,
        )

        encoder = _build_encoder(
            cfg,
            input_dim=input_dim,
            device=device,
            chronos_pipeline=chronos_pipeline,
            hpo_params=hpo_params,
            seed=seed,
            target_scale=target_scale,
        )

        encoder_from_cache = False
        if _has_cached_encoder(cache_key, cache_dir):
            _load_encoder_from_cache(encoder, cache_key, cache_dir)
            encoder = copy.deepcopy(encoder)
            encoder_from_cache = True

        wrapper = LateFusionWrapper(
            encoder=encoder,
            raw_news_dim=raw_news_dim,
            projected_news_dim=params.get("news", {}).get("projected_news_dim", 128),
            seq_len=mw_train.shape[1],
            device=device,
            horizon=horizon,
        )

        p_news = params.get("news", {})
        encoder_recipe = _encoder_recipe(cfg, params, enc_name, horizon)

                # Phase 1 (OOF) is the dominant cost of a late-fusion cell: it refits the
        # encoder once per CV fold. OOF predictions depend only on the market
        # recipe + inputs (not the news variant), so we cache them and reuse the
        # tensor across late-fusion cells that share this backbone + market inputs.
        # The Phase-1 budget (folds + per-fold epochs) is applied UNIFORMLY to
        # every backbone (see _oof_budget) so the fusion comparison is
        # apple-to-apple; the final encoder fit (Phase 2) still uses the full
        # `encoder_recipe`, so deployed accuracy is unchanged.
        oof_n_splits, oof_recipe = _oof_budget(encoder_recipe)
        oof_gap = max(horizon, 1)
        oof_key = _build_oof_cache_key(
            encoder_name=enc_name,
            cfg=cfg,
            params=params,
            effective_market_cols=effective_market_cols,
            mw_train=mw_train,
            mw_val=mw_val,
            splits=splits,
            horizon=horizon,
            seed=seed,
            n_splits=oof_n_splits,
            gap=oof_gap,
                        oof_recipe=oof_recipe,
        )
        cached_oof = _load_oof_from_cache(oof_key, cache_dir)
        if cached_oof is not None:
            logger.info("Late fusion: reusing cached OOF market predictions | key={}", oof_key)
        else:
            logger.info(
                "Late fusion: OOF budget for backbone '{}' "
                "(n_splits={}, epochs={} vs main fit {})",
                enc_name, oof_n_splits,
                oof_recipe.get("epochs"), encoder_recipe.get("epochs"),
            )

        wrapper.fit(
            mw_train,
            ne_train_f,
            y_train,
            mw_val,
            ne_val_f,
            y_val,
            news_mask_train=nm_train,
            news_mask_val=nm_val,
            n_splits=oof_n_splits,
            epochs_news=p_news.get("epochs", 30),
            batch_size_news=p_news.get("batch_size", 32),
            lr_news=p_news.get("lr", 1e-3),
            patience_news=p_news.get("patience", 8),
            skip_encoder_fit=encoder_from_cache,
            precomputed_oof=cached_oof,
            oof_fit_kwargs=oof_recipe,
            **encoder_recipe,
        )
        if not encoder_from_cache:
            _save_encoder_to_cache(cache_key, wrapper.encoder, cache_dir)
        if cached_oof is None:
            oof_to_cache = getattr(wrapper, "_last_oof_preds_train", None)
            if oof_to_cache is not None:
                _save_oof_to_cache(oof_key, np.asarray(oof_to_cache, dtype=np.float32), cache_dir)

        preds = wrapper.predict(mw_test, ne_test_f, nm_test)
        if artifacts is not None:
            artifacts.update(
                model_kind="late",
                test_pred=np.asarray(preds, dtype=np.float32),
                val_pred=np.asarray(wrapper.predict(mw_val, ne_val_f, nm_val), dtype=np.float32),
            )

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

        if hybrid_market_model_name not in {"lstm", "cnn_lstm", "gpt4ts", "chronos"}:
            raise ValueError(
                f"Unsupported CMTF market_encoder_name: {hybrid_market_model_name}"
            )

        market_encoder_cache_key = _build_shared_encoder_cache_key(
            encoder_name=hybrid_market_model_name,
            cfg=cfg,
            params=params,
            effective_market_cols=effective_market_cols,
            mw_train=mw_train,
            mw_val=mw_val,
            splits=splits,
            horizon=horizon,
            seed=seed,
        )

        encoder_hpo = _extract_encoder_cache_params(cfg, params, hybrid_market_model_name)

        market_encoder = build_market_encoder(
            model_name=hybrid_market_model_name,
            input_dim=input_dim,
            seq_len=mw_train.shape[1],
            horizon=horizon,
            device=device,
            chronos_pipeline=chronos_pipeline,
            target_scale=target_scale,
            hidden_dim=encoder_hpo.get("hidden_dim", 64),
            num_layers=encoder_hpo.get("num_layers", 2),
            dropout=encoder_hpo.get("dropout", 0.3),
            sign_penalty_weight=encoder_hpo.get("sign_penalty_weight", 0.02),
        )

        skip_encoder_fit = False
        if _has_cached_encoder(market_encoder_cache_key, cache_dir):
            _load_encoder_from_cache(market_encoder, market_encoder_cache_key, cache_dir)
            market_encoder = copy.deepcopy(market_encoder)
            skip_encoder_fit = True

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
            sharpe_surrogate_weight=cfg.sharpe_surrogate_weight,
            sharpe_surrogate_k=cfg.sharpe_surrogate_k,
            use_cross_attention=cfg.use_cross_attention,
            use_positional_encoding=cfg.use_positional_encoding,
            use_news_gate=cfg.use_news_gate,
            use_variance_reg=cfg.use_variance_reg,
            use_two_stage=cfg.use_two_stage,
            use_aux_loss=cfg.use_aux_loss,
            freeze_market_encoder=False,
            recency_gate_k=cfg.recency_gate_k,
            target_scale=target_scale,
            aux_loss_weight=cfg.aux_loss_weight,
            encoder_lr_scale=cfg.encoder_lr_scale,
            stage1_ratio=cfg.stage1_ratio,
            market_epochs=cfg.market_epochs,
            fusion_epochs=cfg.fusion_epochs,
            market_patience=cfg.market_patience,
            fusion_patience=cfg.fusion_patience,
            news_gate_alpha=cfg.news_gate_alpha,
            gate_mode=cfg.gate_mode,
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

        # Canonical recipe: identical to the one folded into the encoder cache
        # key, so a cmtf stage-1 encoder is bit-identical to (and shared with)
        # the matching none/late encoder and market-only anchor.
        market_fit_kwargs = _encoder_recipe(cfg, params, hybrid_market_model_name, horizon)

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
            skip_encoder_fit=skip_encoder_fit,
        )

        if not skip_encoder_fit:
            _save_encoder_to_cache(market_encoder_cache_key, hybrid_model.market_encoder, cache_dir)

        preds = hybrid_model.predict(mw_test, ne_test_f, nm_test)
        if artifacts is not None:
            artifacts.update(
                model_kind="cmtf",
                test_pred=np.asarray(preds, dtype=np.float32),
                val_pred=np.asarray(
                    hybrid_model.predict(mw_val, ne_val_f, nm_val), dtype=np.float32
                ),
            )

    else:
        raise ValueError(f"Unknown fusion type: {cfg.fusion_type}")

    # ------------------------------------------------------------------
    # Shared anchor: same market-only baseline for comparable diagnostics
    # ------------------------------------------------------------------
    if cfg.fusion_type == "none":
        # Keep anchor neutral for none cells to avoid self-disagreement = 0 artifact.
        anchor_pred = None
    else:
        anchor_pred = _get_shared_anchor_pred(
            cfg=cfg,
            splits=splits,
            mw_train=mw_train,
            mw_val=mw_val,
            mw_test=mw_test,
            y_train=y_train,
            y_val=y_val,
            effective_market_cols=effective_market_cols,
            horizon=horizon,
            device=device,
            chronos_pipeline=chronos_pipeline,
            hpo_params=params,
            seed=seed,
            cache_dir=cache_dir,
        )

    metrics = compute_all(y_test, preds, horizon=horizon)
    metrics.update(
        compute_composite_metrics(
            y_test,
            preds,
            horizon=horizon,
            anchor_pred=anchor_pred,
        )
    )

    logger.info(
        "  ✓ {} → DA%={:.1f}  Sharpe={:.3f}  IC={:.3f}",
        cfg.cell_id,
        metrics["DA%"],
        metrics["Sharpe"],
        metrics["IC"],
    )

    is_degen = flag_degenerate(metrics, preds)
    metrics["degenerate"] = is_degen
    if is_degen:
        logger.warning("  ⚠ Degenerate prediction (collapsed / no-skill) for {} at {}D", cfg.cell_id, horizon)

    # ------------------------------------------------------------------
    # Optional: validation-calibrated confidence-gate + conviction-sizing
    # decision policy (src/benchmark/decision_policy.py), layered on top of
    # the already-trained prediction — no retraining. Calibrated on VAL only
    # (leak-free) then applied to the frozen TEST predictions. See
    # RESULTS_IMPROVEMENT_LEVERS.md for the evidence this is worthwhile for
    # CMTF specifically; computed for every fusion_type here so the
    # fusion_comparison table lets you honestly compare "best CMTF, gated"
    # against every other backbone/fusion cell under the SAME policy.
    # ------------------------------------------------------------------
    if compute_gate:
        val_pred = artifacts.get("val_pred") if artifacts else None
        if val_pred is not None:
            from .decision_policy import (
                calibrate_gate,
                calibrate_gate_fixed_coverage,
                evaluate_policy,
            )

            if gate_coverage is not None:
                # Fixed coverage across every cell (apples-to-apples): each
                # model trades the SAME top-fraction of its own confidence
                # ranking, so gated DA/Sharpe/IC compare confidence-ranking
                # QUALITY rather than rewarding a model for how aggressively
                # a per-model coverage search happened to gate it.
                policy = calibrate_gate_fixed_coverage(
                    val_pred, y_val, coverage=gate_coverage, conviction=gate_conviction
                )
            else:
                # Legacy per-model best-coverage search (NOT apples-to-apples
                # across cells — kept only for single-model deployment tuning).
                policy = calibrate_gate(val_pred, y_val, conviction=gate_conviction)
            gated = evaluate_policy(y_test, preds, policy, horizon=horizon)
            metrics["DA%_gated"] = gated["DA%"]
            metrics["Sharpe_gated"] = gated["Sharpe"]
            metrics["IC_gated"] = gated["IC"]
            metrics["gate_coverage"] = gated["coverage"]
            metrics["gate_tau"] = policy.tau
            metrics["gate_conviction"] = policy.conviction
            logger.info(
                "  ✓ {} GATED → DA%={:.1f}  Sharpe={:.3f}  IC={:.3f}  coverage={:.2f}",
                cfg.cell_id, gated["DA%"], gated["Sharpe"], gated["IC"], gated["coverage"],
            )
        else:
            logger.warning(
                "compute_gate=True but no val_pred available for {} — leaving gated metrics NaN",
                cfg.cell_id,
            )
            metrics["DA%_gated"] = float("nan")
            metrics["Sharpe_gated"] = float("nan")
            metrics["IC_gated"] = float("nan")
            metrics["gate_coverage"] = float("nan")
            metrics["gate_tau"] = float("nan")
            metrics["gate_conviction"] = False

    metrics["train_time_sec"] = round(time.perf_counter() - _cell_start_time, 2)

    if cache_dir is not None:
        pred_dir = Path(cache_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        short_id = _config_hash(cfg)
        pred_file = pred_dir / f"{short_id}__seed{seed}__{horizon}d.npy"
        np.save(str(pred_file), preds)

        # D2 provenance sidecar: a cached prediction can always be traced back
        # to the exact config + scaling scheme that produced it, so stale or
        # cross-scheme artifacts can never be silently reused.
        provenance = {
            "cell_id": cfg.cell_id,
            "config_hash": short_id,
            "seed": int(seed),
            "horizon": int(horizon),
            "scaling_version": _SCALING_VERSION,
            "target_scale": float(target_scale),
            "shuffle_news": bool(getattr(cfg, "shuffle_news", False)),
            "data_sig": data_sig,
            "news_sig": news_sig,
        }
        (pred_dir / f"{short_id}__seed{seed}__{horizon}d.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        # In-process guard: catches real intra-run leakage/aliasing bugs, since every
        # cell for one horizon in this run must share identical test targets.
        run_truth = _RUN_TRUTH_CACHE.get(horizon)
        if run_truth is not None:
            if run_truth.shape != y_test.shape or not np.allclose(run_truth, y_test):
                raise ValueError(
                    f"In-run truth mismatch for horizon={horizon}. All cells in one run "
                    "must share the same test targets — this indicates a real data/splits bug."
                )
        else:
            _RUN_TRUTH_CACHE[horizon] = y_test

        # On-disk cache: only meant to speed up cross-run consistency checks. The
        # data pipeline evolves across runs/sessions, so a mismatch here just means
        # the cache predates the current pipeline — refresh it instead of failing
        # the whole cell (the in-process guard above already covers real bugs).
        truth_file = pred_dir / f"truth__{horizon}d.npy"
        if truth_file.exists():
            existing_truth = np.load(str(truth_file))
            if existing_truth.shape != y_test.shape or not np.allclose(existing_truth, y_test):
                logger.warning(
                    "Stale cached truth for horizon={}d (cached shape={}, current shape={}) "
                    "— refreshing from current pipeline output instead of failing the run.",
                    horizon, existing_truth.shape, y_test.shape,
                )
                np.save(str(truth_file), y_test)
        else:
            np.save(str(truth_file), y_test)

    return metrics