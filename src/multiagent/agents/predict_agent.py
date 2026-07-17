"""Predict Agent — the CMTF prediction for a single name.

Always runs a real forward pass of the deployed champion via
``raw_prediction.fetch_prediction_record`` (never the frozen `.npy` cache — see that
module's docstring for why). This is what makes the model's internal attention/
recency-gate tensors genuinely available for every request, not just new/live dates.

Emits the RAW magnitude the gate consumes. R1: if live inference can't serve the
(symbol, date), it raises loudly; it never invents a prediction. ``news_residual`` is
not exposed by the current champion architecture, so it is reported as ``None`` (not
fabricated).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..raw_prediction import fetch_prediction_record
from ..state import MultiAgentState


def predict_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: serve the CMTF prediction (always a real forward pass).

    Reads: symbol, target_horizon_days, prediction_time
    Writes: final_pred (seed-mean), gate_pred (raw, gated), seed_preds, baseline_pred
            (None), news_residual (None), attn_weights, news_weight,
            attention_top_days, model_evidence, artifact_versions, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbol = state["symbol"]
    horizon = state["target_horizon_days"]
    date = state["prediction_time"]

    rec = fetch_prediction_record(symbol, date, horizon, cfg, data_end=state.get("data_end"))
    seed_preds, final_pred, gate_pred, truth, source = (
        rec.seed_preds, rec.ensemble_pred, rec.gate_pred, rec.truth, rec.source,
    )

    model_evidence = {
        "final_pred": final_pred,
        "gate_pred": gate_pred,
        "gate_on_raw_seed": bool(cfg.gate_on_raw_seed),
        "baseline_pred": None,
        "seed_preds": [round(p, 6) for p in seed_preds],
        "seed_variance": round(float(np.var(seed_preds)), 8),
        "spread": round(float(np.max(seed_preds) - np.min(seed_preds)), 6),
        "all_same_sign": bool(np.all(np.sign(seed_preds) == np.sign(seed_preds[0])) and seed_preds[0] != 0),
        "news_residual": None,
        "truth": truth,  # realised return (backtest use only; never feeds the decision)
        "source": source,
        "attention_top_days": rec.attention_top_days,
    }

    elapsed = time.time() - t0
    logger.info(
        "PredictAgent | {} {} {}d | seed_mean={:.5f} gate_pred={:.5f} ({} seeds) | {:.3f}s",
        symbol, date, horizon, final_pred, gate_pred, len(seed_preds), elapsed,
    )

    artifact_versions = dict(state.get("artifact_versions", {}))
    artifact_versions["cmtf_version"] = cfg.cmtf_version
    artifact_versions["backbone_version"] = cfg.backbone_version
    artifact_versions["ensemble_seeds"] = str(cfg.ensemble_seeds)

    return {
        "baseline_pred": None,
        "final_pred": final_pred,
        "gate_pred": gate_pred,
        "seed_preds": seed_preds,
        "news_residual": None,
        "attn_weights": rec.attn_weights,
        "news_weight": rec.recency_gate,
        "attention_top_days": rec.attention_top_days,
        "model_evidence": model_evidence,
        "artifact_versions": artifact_versions,
        "node_timings": {"predict_agent": elapsed},
    }
