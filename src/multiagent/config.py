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
    sentiment_output_dir: Path = field(default_factory=lambda: Path("outputs/sentiment/latest"))
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
    # Registry default seeds (run_ablation_registry --seeds 1 42 123). The gate is
    # calibrated on the mean of these seeds' validation predictions and predict_agent
    # averages the same seeds, so runtime == research.
    ensemble_seeds: list[int] = field(default_factory=lambda: [1, 42, 123])

    # --- VN-Index ---
    vnindex_symbol: str = "VNINDEX"

    # --- Gate (the decision core) ---
    # Directory of frozen, validation-calibrated GatePolicy artifacts (VN_{H}d.json).
    gate_policy_dir: Path = field(default_factory=lambda: Path("results/gate_policies"))
    gate_coverage: float = 0.25  # fixed-coverage calibration target (apples-to-apples)
    use_conviction_sizing: bool = True
    # Gate on the 3-seed ENSEMBLE mean (default), not a single raw seed. Empirically
    # (results/gate_policies diagnostics) single-seed gated DA is high-variance
    # (49.6–58.0% across seeds) and validation selection_score does NOT reliably rank
    # seeds, so a single seed is not a defensible, leak-free deployment choice. The
    # ensemble is the pre-registered standard and its gated IC (0.129) exceeds every
    # single seed, so the RESULTS_IMPROVEMENT_LEVERS tail-shrink concern does not bind
    # here. gate_on_raw_seed=True is retained as a labelled research toggle only.
    gate_on_raw_seed: bool = False

    # --- Live inference (product) ---
    # When a (symbol, date) is not in the frozen research cache, run a real forward
    # pass of the deployed champion (cache/deploy_models) instead of raising. This is
    # what makes the system realtime. Frozen cache is still used for known dates.
    enable_live_inference: bool = True

    # --- Routing / news scope ---
    enable_intent_routing: bool = True
    news_scope_default: str = "matched"

    # --- Risk: one-way safety veto (never a decision-maker) ---
    hard_block_vol: float = 40.0  # 20d annualised vol %; above → veto a trade to abstain
    hard_block_drawdown: float = 20.0  # max drawdown %; above → veto a trade to abstain

    # --- Critic ---
    critic_max_retries: int = 2

    # --- Trace / observability ---
    trace_enabled: bool = False


# Singleton default config — importable everywhere
DEFAULT_CONFIG = MultiAgentConfig()
