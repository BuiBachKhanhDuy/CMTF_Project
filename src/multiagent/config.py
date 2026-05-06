"""Configuration for the multi-agent inference system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MultiAgentConfig:
    """All tunable parameters for the multi-agent graph."""

    # --- Decision thresholds ---
    buy_threshold: float = 0.002
    sell_threshold: float = 0.002

    # --- Risk/Regime Critic ---
    vol_high_pct: float = 0.30  # 20d annualized vol threshold (VN banking typical: 19-32%)
    dd_max_pct: float = 0.08  # max drawdown threshold → force flat

    # --- News-Quality Critic ---
    min_news_bars: int = 3  # minimum bars with articles to trust news
    max_stale_frac: float = 0.7  # fraction of stale articles → discount news
    staleness_days: int = 14  # articles older than this are "stale"

    # --- Ensemble Disagreement ---
    # (no tunable params — pure sign check)

    # --- Sequence ---
    sequence_len: int = 30

    # --- Paths ---
    news_cache_dir: Path = field(default_factory=lambda: Path("cache/news"))
    cmtf_models_dir: Path = field(default_factory=lambda: Path("cache/cmtf_models"))
    optuna_dir: Path = field(default_factory=lambda: Path("cache/optuna"))
    phase2_output_dir: Path = field(default_factory=lambda: Path("outputs/phase2/latest"))
    chronos_emb_dir: Path = field(default_factory=lambda: Path("cache/chronos_emb"))

    # --- Ollama ---
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30  # seconds

    # --- Model versions ---
    cmtf_version: str = "v8"
    backbone_version: str = "v3"
    hpo_version: str = "v7"
    ensemble_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # --- VN-Index ---
    vnindex_symbol: str = "VNINDEX"


# Singleton default config — importable everywhere
DEFAULT_CONFIG = MultiAgentConfig()
