"""Shared state definition for the multi-agent inference graph.

Topology:
    orchestrator → [market_agent | news_agent] → predict_agent
    → gate_agent → risk_agent (veto) → narrator → critic → END
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import numpy as np


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merge two dicts (later values overwrite)."""
    merged = dict(a) if a else {}
    if b:
        merged.update(b)
    return merged


def _merge_lists(a: list, b: list) -> list:
    """Reducer: concatenate two lists."""
    return (a or []) + (b or [])


class MultiAgentState(TypedDict, total=False):
    # --- Request (raw input) ---
    query_text: str
    symbol: str
    prediction_time: str  # ISO date e.g. "2025-03-31"
    data_end: str  # optional: extend the live-inference data fetch to >= prediction_time
    target_horizon_days: int  # 1, 5, or 20
    sequence_len: int  # default 30

    # --- Orchestrator routing output ---
    query_intent: str  # "prediction" | "comparison" | "explanation" | "research"
    target_symbols: list[str]  # symbols to act on (single for prediction, N for comparison)
    target_horizon: str  # e.g. "5d"
    aspect_filter: str  # optional topic filter for research/news
    route_reason: str  # why this branch was chosen (never a silent default — R1)

    # --- Orchestrator output (data fetch) ---
    close_window: np.ndarray  # (seq_len,)
    market_window: np.ndarray  # (seq_len, n_feat)
    market_tabular: np.ndarray  # (n_feat,)
    market_feature_cols: list[str]  # column names for market_tabular
    news_emb: np.ndarray  # (seq_len, 773)
    news_mask: np.ndarray  # (seq_len,) True=missing
    articles: list[dict[str, Any]]
    data_cutoff: str

    # --- Market agent output ---
    volatility_metrics: dict[str, float]  # vol_20d, max_drawdown_pct, trend_pct
    market_proposal: dict[str, Any]  # direction, score, confidence, rationale, quality

    # --- News agent output ---
    sentiment_metrics: dict[str, Any]  # coverage, staleness_frac, sentiment_mean, sentiment_std
    news_proposal: dict[str, Any]  # direction, score (raw sentiment), confidence (trust_weight)

    # --- Predict agent output ---
    baseline_pred: float
    final_pred: float  # seed-mean (metadata only; NOT what the gate consumes)
    gate_pred: float  # RAW magnitude the gate consumes (single-seed or mean per config)
    seed_preds: list[float]
    news_residual: float
    attn_weights: np.ndarray  # (seq_len,) mean attention
    news_weight: float
    predict_confidence: float  # demoted to metadata — the gate does NOT use this
    model_evidence: dict[str, Any]  # full evidence payload (incl. agreement annotations)
    model_proposal: dict[str, Any]  # direction, score, confidence, rationale

    # --- Gate agent output (the decision core) ---
    gated_action: str  # "long" | "short" | "abstain" (gate's decision, pre-veto)
    gate_tau: float
    gate_coverage: float
    gate_val_score: float
    gate_reason: str

    # --- Risk agent output (one-way safety veto only) ---
    action: str  # final action: "long" | "short" | "abstain"
    position_scale: float  # signed conviction size (gate) or 0 (abstain/veto)
    risk_vetoed: bool
    veto_reasons: list[str]
    decision_reasoning: str

    # --- Metalabel agent output (one-way qualitative event-flag veto) ---
    metalabel_flags: list[str]
    metalabel_vetoed: bool

    # --- Rank agent output (COMPARISON branch) ---
    ranking: list[dict[str, Any]]  # one row per symbol, sorted by conviction
    rank_longs: list[str]
    rank_shorts: list[str]
    rank_abstained: list[str]

    # --- Research agent output (RESEARCH branch) ---
    retrieved_docs: list[dict[str, Any]]
    research_summary_vi: str

    # --- Narrator + critic output ---
    answer_text: str  # final Vietnamese answer (verified)
    grounded_answer: str  # deterministic state-only answer (critic fallback + reference)
    critic_status: str  # "ok" | "regenerated" | "failed"
    critic_findings: list[str]

    # --- Audit ---
    data_cutoff: str
    artifact_versions: Annotated[dict[str, str], _merge_dicts]
    errors: Annotated[list[str], _merge_lists]
    warnings: Annotated[list[str], _merge_lists]
    node_timings: Annotated[dict[str, float], _merge_dicts]
    trace: Annotated[list[dict[str, Any]], _merge_lists]
    decision_id: str
