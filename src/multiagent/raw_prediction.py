"""Fetch a live CMTF prediction for a symbol, date, and horizon.

The product path runs the deployed model so it can return attention and
recency-gate evidence. Evaluation code uses frozen prediction stores instead.
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
    source: str  # ``live_inference`` for compatibility with model evidence.
    attn_weights: np.ndarray | None = None
    recency_gate: np.ndarray | None = None
    attention_top_days: list[dict] | None = None


def summarize_attention(attn_weights: np.ndarray | None, top_k: int = 3) -> list[dict] | None:
    """Return the highest-weight trailing days from an attention vector."""
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
