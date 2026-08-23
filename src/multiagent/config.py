"""Configuration for the multi-agent inference system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MultiAgentConfig:
    """All tunable parameters for the multi-agent graph."""

    # Sequence settings
    sequence_len: int = 30

    # Artifact paths
    news_cache_dir: Path = field(default_factory=lambda: Path("cache/news"))
    cmtf_models_dir: Path = field(default_factory=lambda: Path("cache/cmtf_models"))
    optuna_dir: Path = field(default_factory=lambda: Path("cache/optuna"))
    sentiment_output_dir: Path = field(default_factory=lambda: Path("outputs/sentiment/latest"))
    chronos_emb_dir: Path = field(default_factory=lambda: Path("cache/chronos_emb"))

    # Ollama settings
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30  # seconds

    # Disable LLM calls during deterministic evaluation.
    evaluation_mode: bool = False

    # Model versions
    cmtf_version: str = "v4"
    backbone_version: str = "v3"
    hpo_version: str = "v7"
    # Seeds used to calibrate and serve the ensemble.
    ensemble_seeds: list[int] = field(default_factory=lambda: [1, 42, 123])

    # VN-Index symbol
    vnindex_symbol: str = "VNINDEX"

    # Decision gate
    # Directory containing validation-calibrated ``GatePolicy`` artifacts.
    gate_policy_dir: Path = field(default_factory=lambda: Path("results/gate_policies"))
    gate_coverage: float = 0.25  # Calibration coverage target.
    use_conviction_sizing: bool = True
    # Use the ensemble mean by default; set this only for single-seed experiments.
    gate_on_raw_seed: bool = False

    # Live inference runs the deployed ensemble and returns model evidence.
    enable_live_inference: bool = True

    # Routing and news scope
    enable_intent_routing: bool = True
    news_scope_default: str = "matched"

    # Risk veto thresholds
    hard_block_vol: float = 40.0  # 20d annualised vol %; above → veto a trade to abstain
    hard_block_drawdown: float = 20.0  # max drawdown %; above → veto a trade to abstain

    # Cross-horizon conviction adjustment.
    enable_horizon_interaction: bool = True
    horizon_interaction_dir: Path = field(default_factory=lambda: Path("results/horizon_interaction"))

    # Reasoning settings
    reasoning_min_news_coverage: int = 3
    reasoning_widen_lookback_days_to: int = 20
    reasoning_widen_sequence_len_to: int = 60

    # Critic settings
    critic_max_retries: int = 2

    # Trace settings
    trace_enabled: bool = False


# Shared default configuration.
DEFAULT_CONFIG = MultiAgentConfig()
