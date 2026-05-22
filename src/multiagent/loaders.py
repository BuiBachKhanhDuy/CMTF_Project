"""Lazy artifact loader with singleton/keyed cache and test override hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig


class ArtifactMissingError(FileNotFoundError):
    """Raised when a required model artifact cannot be found on disk."""

    pass


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {}
_overrides: dict[str, Any] = {}


def _cmtf_version_rank(path: Path) -> tuple[int, str]:
    """Sort helper for cnn_lstm_cmtf_vN_* artifacts."""
    parts = path.stem.split("_")
    version = parts[3] if len(parts) > 3 else "v0"
    if version.startswith("v") and version[1:].isdigit():
        return int(version[1:]), version
    return -1, version


def _resolve_cmtf_checkpoint(
    cfg: MultiAgentConfig,
    symbol: str,
    horizon: int,
    seed: int,
) -> Path:
    """Resolve the preferred CNN-LSTM CMTF checkpoint for one ensemble seed."""
    exact_pattern = f"cnn_lstm_cmtf_{cfg.cmtf_version}_{symbol}_{horizon}d_seed{seed}_*.pt"
    exact_matches = sorted(cfg.cmtf_models_dir.glob(exact_pattern))
    if exact_matches:
        return exact_matches[0]

    fallback_pattern = f"cnn_lstm_cmtf_v*_{symbol}_{horizon}d_seed{seed}_*.pt"
    fallback_matches = sorted(
        cfg.cmtf_models_dir.glob(fallback_pattern),
        key=_cmtf_version_rank,
        reverse=True,
    )
    if fallback_matches:
        return fallback_matches[0]

    raise ArtifactMissingError(
        "No CNN-LSTM CMTF checkpoint matching "
        f"'{exact_pattern}' or '{fallback_pattern}' in {cfg.cmtf_models_dir}"
    )


def _build_cnn_lstm_cmtf_from_checkpoint(
    ckpt: dict,
    hpo_params: dict[str, Any],
    seq_len: int,
):
    """Reconstruct a CNN-LSTM CMTF predictor from its checkpoint shapes."""
    state_dict = ckpt.get("state_dict", ckpt)

    input_dim = int(state_dict["input_proj.weight"].shape[1])
    num_filters = int(state_dict["input_proj.weight"].shape[0])
    hidden_dim = int(state_dict["lstm.weight_hh_l0"].shape[1])
    num_layers = len([key for key in state_dict if key.startswith("lstm.weight_ih_l")])
    news_dim = int(state_dict["fusion.news_proj.0.weight"].shape[1])
    fusion_market_dim = int(state_dict["fusion.market_query_proj.0.weight"].shape[0])
    fusion_dim = int(state_dict["fusion.residual_head.0.weight"].shape[0])

    from src.benchmark.cnn_lstm_cmtf import CNNLSTMCMTFPredictor

    return CNNLSTMCMTFPredictor(
        input_dim=input_dim,
        news_dim=news_dim,
        hidden_dim=hidden_dim,
        num_filters=num_filters,
        num_layers=num_layers,
        dropout=float(hpo_params.get("dropout", 0.2)),
        fusion_dim=fusion_dim,
        fusion_market_dim=fusion_market_dim,
        n_heads=int(hpo_params.get("n_heads", 4)),
        sign_penalty_weight=float(hpo_params.get("dir_penalty_weight", 0.05)),
        seq_len=seq_len,
        device="cpu",
    )


def set_loader_override(name: str, obj: Any) -> None:
    """Inject a fake artifact for testing. Bypasses disk loading."""
    _overrides[name] = obj


def clear_overrides() -> None:
    """Remove all test overrides and clear the cache."""
    _overrides.clear()
    _cache.clear()


# ---------------------------------------------------------------------------
# LoRA backbone (keyed by symbol × horizon)
# ---------------------------------------------------------------------------
def get_lora_backbone(symbol: str, horizon: int, config: MultiAgentConfig | None = None):
    """Load the fine-tuned ChronosLoRA backbone for (symbol, horizon)."""
    key = f"lora_backbone_{symbol}_{horizon}d"
    if key in _overrides:
        return _overrides[key]
    if key in _cache:
        return _cache[key]

    cfg = config or DEFAULT_CONFIG
    pattern = f"ft_chronos_lora_backbone_{cfg.backbone_version}_{symbol}_{horizon}d_*.pt"
    matches = list(cfg.cmtf_models_dir.glob(pattern))
    if not matches:
        raise ArtifactMissingError(
            f"No backbone checkpoint matching '{pattern}' in {cfg.cmtf_models_dir}"
        )
    ckpt_path = matches[0]  # Take first match (should be unique per symbol/horizon)

    # Load HPO params to reconstruct the predictor
    baseline_hpo_path = cfg.optuna_dir / f"best_baseline_params_{horizon}d.json"
    if not baseline_hpo_path.exists():
        raise ArtifactMissingError(f"Baseline HPO params not found: {baseline_hpo_path}")
    baseline_params = json.loads(baseline_hpo_path.read_text(encoding="utf-8"))
    # The LoRA backbone params are under the "finetuned_chronos" key
    ft_params = baseline_params.get("finetuned_chronos", baseline_params)

    from src.benchmark.chronos_encoder import ChronosMarketPredictor
    from src.benchmark.baseline_models import ChronosLoRAPredictor

    chronos = ChronosMarketPredictor(device="cpu")
    lora_predictor = ChronosLoRAPredictor(
        chronos,
        hidden_dim=int(ft_params.get("hidden_dim", 192)),
        dropout=float(ft_params.get("dropout", 0.2)),
        tabular_dim=0,  # backbone was trained without tabular features
        market_input_dim=23,  # OHLCV + technicals (fixed across all checkpoints)
        market_hidden_dim=int(ft_params.get("market_hidden_dim", 32)),
        device="cpu",
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_predictor.load_checkpoint_state(ckpt)
    lora_predictor.is_fitted = True
    logger.info("Loaded LoRA backbone: {}", ckpt_path.name)

    _cache[key] = lora_predictor
    return lora_predictor


# ---------------------------------------------------------------------------
# CMTF ensemble (keyed by symbol × horizon, returns list of 3 predictors)
# ---------------------------------------------------------------------------
def get_cmtf_ensemble(
    symbol: str, horizon: int, config: MultiAgentConfig | None = None
) -> list:
    """Load the CNN-LSTM CMTF ensemble predictors for (symbol, horizon)."""
    key = f"cmtf_ensemble_{symbol}_{horizon}d"
    if key in _overrides:
        return _overrides[key]
    if key in _cache:
        return _cache[key]

    cfg = config or DEFAULT_CONFIG

    hpo_path = cfg.optuna_dir / f"best_params_{cfg.hpo_version}_{horizon}d.json"
    hpo_params = (
        json.loads(hpo_path.read_text(encoding="utf-8"))
        if hpo_path.exists() else {}
    )

    ensemble = []
    for seed in cfg.ensemble_seeds:
        ckpt_path = _resolve_cmtf_checkpoint(cfg, symbol, horizon, seed)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        predictor = _build_cnn_lstm_cmtf_from_checkpoint(
            ckpt,
            hpo_params=hpo_params,
            seq_len=cfg.sequence_len,
        )
        predictor.load_checkpoint(ckpt)
        predictor.is_fitted = True
        predictor.loaded_checkpoint_name = ckpt_path.name
        predictor.loaded_checkpoint_version = ckpt.get("version", _cmtf_version_rank(ckpt_path)[1])
        ensemble.append(predictor)
        logger.debug("Loaded CMTF seed {}: {}", seed, ckpt_path.name)

    logger.info("Loaded CMTF ensemble ({} seeds) for {} {}d", len(ensemble), symbol, horizon)
    _cache[key] = ensemble
    return ensemble
