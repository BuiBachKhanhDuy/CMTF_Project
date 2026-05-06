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


def set_loader_override(name: str, obj: Any) -> None:
    """Inject a fake artifact for testing. Bypasses disk loading."""
    _overrides[name] = obj


def clear_overrides() -> None:
    """Remove all test overrides and clear the cache."""
    _overrides.clear()
    _cache.clear()


# ---------------------------------------------------------------------------
# PhoBERT bundle (singleton)
# ---------------------------------------------------------------------------
def get_phobert_bundle(config: MultiAgentConfig | None = None):
    """Load Phase2 PhoBERT inference bundle (singleton)."""
    key = "phobert_bundle"
    if key in _overrides:
        return _overrides[key]
    if key in _cache:
        return _cache[key]

    cfg = config or DEFAULT_CONFIG
    from src.phase2.inference import load_phase2_phobert_inference_bundle

    bundle = load_phase2_phobert_inference_bundle(
        output_dir=cfg.phase2_output_dir,
        device="cpu",
    )
    _cache[key] = bundle
    logger.info("Loaded PhoBERT bundle from {}", cfg.phase2_output_dir)
    return bundle


# ---------------------------------------------------------------------------
# News encoder (singleton)
# ---------------------------------------------------------------------------
def get_news_encoder(config: MultiAgentConfig | None = None):
    """Load NewsEncoder (singleton)."""
    key = "news_encoder"
    if key in _overrides:
        return _overrides[key]
    if key in _cache:
        return _cache[key]

    from src.pipeline.news_encoder import NewsEncoder

    encoder = NewsEncoder()
    _cache[key] = encoder
    logger.info("Loaded NewsEncoder (vietnamese-embedding 768-dim)")
    return encoder


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

    from src.benchmark.chronos_market import ChronosMarketPredictor
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
    """Load 3 CMTF v8 ensemble predictors for (symbol, horizon)."""
    key = f"cmtf_ensemble_{symbol}_{horizon}d"
    if key in _overrides:
        return _overrides[key]
    if key in _cache:
        return _cache[key]

    cfg = config or DEFAULT_CONFIG

    # Load HPO params for CMTF fusion
    hpo_path = cfg.optuna_dir / f"best_params_{cfg.hpo_version}_{horizon}d.json"
    if not hpo_path.exists():
        raise ArtifactMissingError(f"CMTF HPO params not found: {hpo_path}")
    hpo_params = json.loads(hpo_path.read_text(encoding="utf-8"))

    # Get shared backbone
    backbone = get_lora_backbone(symbol, horizon, cfg)

    from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

    ensemble = []
    for seed in cfg.ensemble_seeds:
        pattern = f"cmtf_lora_{cfg.cmtf_version}_{symbol}_{horizon}d_seed{seed}_*.pt"
        matches = list(cfg.cmtf_models_dir.glob(pattern))
        if not matches:
            raise ArtifactMissingError(
                f"No CMTF checkpoint matching '{pattern}' in {cfg.cmtf_models_dir}"
            )
        ckpt_path = matches[0]

        predictor = ChronosCMTFPredictor(
            chronos_lora_predictor=backbone,
            news_dim=773,  # 768 vietnamese-embedding + 5 PhoBERT sentiment features
            tabular_dim=0,  # tabular features already folded into backbone combined_feature_dim
            fusion_dim=int(hpo_params.get("fusion_dim", 64)),
            n_heads=int(hpo_params.get("n_heads", 2)),
            dropout=float(hpo_params.get("dropout", 0.2)),
            seq_len=cfg.sequence_len,
            device="cpu",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        predictor.load_checkpoint(ckpt)
        predictor.is_fitted = True
        ensemble.append(predictor)
        logger.debug("Loaded CMTF seed {}: {}", seed, ckpt_path.name)

    logger.info("Loaded CMTF ensemble ({} seeds) for {} {}d", len(ensemble), symbol, horizon)
    _cache[key] = ensemble
    return ensemble
