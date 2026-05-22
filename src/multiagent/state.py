"""Shared state definition for the multi-agent inference graph.

Topology:
    orchestrator → [market_agent | news_agent] → predict_agent
    → fusion_agent → risk_agent → answer_agent → END
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
    target_horizon_days: int  # 1, 5, or 20
    sequence_len: int  # default 30

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
    final_pred: float
    adjusted_pred: float  # consensus-corrected prediction
    mkt_adjusted_pred: float  # market-agent-only correction
    news_adjusted_pred: float  # news-agent-only correction
    seed_preds: list[float]
    news_residual: float
    attn_weights: np.ndarray  # (seq_len,) mean attention
    news_weight: float
    predict_confidence: float  # derived from seed agreement + pred strength
    model_evidence: dict[str, Any]  # full evidence payload for answer agent
    model_proposal: dict[str, Any]  # direction, score, confidence, rationale

    # --- Fusion agent output ---
    fusion_decision: dict[str, Any]  # fused decision trace over model+market+news

    # --- Risk agent output (final decision authority) ---
    action: str  # "long" | "short" | "flat"
    position_scale: float  # [0.0, 1.0]
    final_confidence: float
    risk_checks: dict[str, Any]  # individual check results
    decision_reasoning: str

    # --- Answer agent output ---
    explanation_text_vi: str

    # --- Audit ---
    data_cutoff: str
    artifact_versions: Annotated[dict[str, str], _merge_dicts]
    errors: Annotated[list[str], _merge_lists]
    warnings: Annotated[list[str], _merge_lists]
    node_timings: Annotated[dict[str, float], _merge_dicts]
    policy_version: int
    decision_id: str
