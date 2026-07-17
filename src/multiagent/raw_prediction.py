"""Shared (symbol, date, horizon) → raw CMTF prediction fetch.

Always runs a real forward pass via ``live_inference.predict_live`` — the frozen
``.npy`` prediction cache is deliberately NOT used here. Two reasons this changed
from the earlier cache-first design:

1. Only a real forward pass can expose the model's internal attention/recency-gate
   tensors (``HybridFusionPredictor.predict_with_attention``) — the frozen cache
   only ever stored bare scalar predictions, so there was no way to get
   explainability out of it for any (symbol, date), no matter how the cache was
   extended.
2. Verified cheap for the common case: ``predict_live``'s Tier 1 (a date already
   inside the cached research range) hits an in-process ``@lru_cache``d dataset
   split, not a network fetch — so switching away from the frozen cache does not
   make already-served dates slow, only the first call per horizon per process
   (subsequent calls in the same run are cache-hits). Tier 2 (genuinely new dates)
   was already the slow path either way.

This is the product/runtime serving path only. `frozen_predictions.get_store` is
UNCHANGED and continues to back every research/evaluation script
(`h3_faithfulness.py`, `h4_interaction_eval.py`, `metalabel_eval.py`,
`eval_ladder.py`, `rank_agent`'s matched-scope lookups, `readiness.py`) — none of
those call this function, so none of their reproducibility is affected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MultiAgentConfig
from .loaders import ArtifactMissingError


@dataclass(frozen=True)
class PredictionRecord:
    seed_preds: list[float]
    ensemble_pred: float
    gate_pred: float
    truth: float | None
    source: str  # always "live_inference" now — kept for model_evidence compatibility
    attn_weights: np.ndarray | None = None
    recency_gate: np.ndarray | None = None
    attention_top_days: list[dict] | None = None


def summarize_attention(attn_weights: np.ndarray | None, top_k: int = 3) -> list[dict] | None:
    """Turn a raw per-trailing-day attention vector into a grounded, numeric summary
    — the top-k days by attention weight, each as
    ``{"days_before_cutoff": int, "weight": float}`` (day 0 = the cutoff date itself,
    i.e. the last/most-recent trailing day). State-derived numbers only — this is
    never an LLM's interpretation of the raw tensor, so the critic can verify it the
    same way it verifies every other disclosed number.
    """
    if attn_weights is None or len(attn_weights) == 0:
        return None
    seq_len = len(attn_weights)
    order = np.argsort(-attn_weights)[:top_k]
    return [
        {
            "days_before_cutoff": int(seq_len - 1 - i),
            "weight": round(float(attn_weights[i]), 4),
        }
        for i in order
    ]


def fetch_prediction_record(
    symbol: str,
    date: str,
    horizon: int,
    config: MultiAgentConfig,
    data_end: str | None = None,
) -> PredictionRecord:
    """Real forward-pass prediction for (symbol, date, horizon), always via
    ``predict_live``. Raises ``ArtifactMissingError`` if live inference is disabled
    or the row cannot be served — never invents a prediction (R1).
    """
    if not config.enable_live_inference:
        raise ArtifactMissingError(
            "Live inference is disabled (config.enable_live_inference=False) — "
            "no other prediction backend is available."
        )

    from .live_inference import predict_live
    lp = predict_live(symbol, date, horizon, config, data_end=data_end or date)
    ensemble_pred = float(np.mean(lp.seed_preds))
    gate_pred = lp.gate_pred if not config.gate_on_raw_seed else lp.seed_preds[0]
    return PredictionRecord(
        seed_preds=lp.seed_preds, ensemble_pred=ensemble_pred,
        gate_pred=gate_pred, truth=lp.truth, source="live_inference",
        attn_weights=lp.attn_weights, recency_gate=lp.recency_gate,
        attention_top_days=summarize_attention(lp.attn_weights),
    )
