"""Shared state definition for the multi-agent inference graph."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class MultiAgentState(TypedDict, total=False):
    # --- Request ---
    symbol: str
    prediction_time: str  # ISO date string e.g. "2025-03-31"
    target_horizon_days: int  # 1, 5, or 20
    sequence_len: int  # default 30

    # --- Market ---
    close_window: np.ndarray  # (seq_len,)
    market_window: np.ndarray  # (seq_len, 23)
    market_tabular: np.ndarray  # (23,)
    token_ids: np.ndarray  # Chronos token IDs
    attention_mask: np.ndarray  # Chronos attention mask

    # --- News ---
    articles: list[dict[str, Any]]  # [{title, published_at, sentiment_score, bar_index}, ...]
    news_emb: np.ndarray  # (seq_len, 773) hybrid embedding
    news_mask: np.ndarray  # (seq_len,) boolean — True means bar has no news
    sentiment_features: dict[str, float]  # 6 scalar sentiment features

    # --- Fusion ---
    baseline_pred: float  # market-only forecast (news_mask all True)
    final_pred: float  # mean over 3 seeds
    seed_preds: list[float]  # per-seed predictions
    news_residual: float  # final_pred - baseline_pred
    attn_weights: np.ndarray  # (seq_len,) mean attention over seeds
    news_weight: float  # mean of fusion.news_weight over seeds

    # --- Critics ---
    regime_flags: dict[str, Any]  # {high_vol, drawdown_breach, vnindex_zscore, ...}
    position_scale_regime: float  # [0, 1]
    news_quality_flags: dict[str, Any]  # {coverage, staleness_frac, sentiment_std, ...}
    news_residual_scale: float  # [0, 1]
    final_pred_adjusted: float  # baseline + scale * residual
    disagreement_force_flat: bool

    # --- Decision ---
    action: str  # "long", "short", or "flat"
    position_scale: float  # [0, 1]

    # --- Explanation ---
    evidence_dict: dict[str, Any]
    explanation_text_vi: str

    # --- Audit ---
    data_cutoff: str  # ISO date
    artifact_versions: dict[str, str]
    errors: list[str]
    warnings: list[str]
    node_timings: dict[str, float]  # node_name -> seconds
