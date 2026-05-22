"""Configuration for the multi-agent inference system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MultiAgentConfig:
    """All tunable parameters for the multi-agent graph."""

    # --- Sequence ---
    sequence_len: int = 30

    # --- Paths ---
    news_cache_dir: Path = field(default_factory=lambda: Path("cache/news"))
    cmtf_models_dir: Path = field(default_factory=lambda: Path("cache/cmtf_models"))
    optuna_dir: Path = field(default_factory=lambda: Path("cache/optuna"))
    phase2_output_dir: Path = field(default_factory=lambda: Path("outputs/phase2/latest"))
    chronos_emb_dir: Path = field(default_factory=lambda: Path("cache/chronos_emb"))

    # --- Ollama ---
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30  # seconds

    # --- Evaluation mode: disables ALL LLM calls globally ---
    evaluation_mode: bool = False

    # --- Model versions ---
    cmtf_version: str = "v4"
    backbone_version: str = "v3"
    hpo_version: str = "v7"
    ensemble_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # --- VN-Index ---
    vnindex_symbol: str = "VNINDEX"

    # --- Fusion: confidence modulation ---
    market_agree_bonus: float = 0.15
    market_disagree_penalty: float = 0.10
    news_agree_bonus: float = 0.10
    news_disagree_penalty: float = 0.05

    # --- Fusion: agent consensus correction ---
    override_alpha: float = 0.3  # how much agent consensus scales final_pred

    # --- Policy / reflection ---
    policy_store_path: Path = field(default_factory=lambda: Path("results/multiagent_policy.json"))
    reflection_min_samples: int = 30


# Singleton default config — importable everywhere
DEFAULT_CONFIG = MultiAgentConfig()
