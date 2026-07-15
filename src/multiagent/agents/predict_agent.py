"""Predict Agent — the CMTF prediction for a single name.

Two backends, tried in order:
1. **Frozen cache** (`frozen_predictions`): for dates already in the research book,
   returns the cached per-seed predictions — fast, and byte-identical to research.
2. **Live inference** (`live_inference.predict_live`): for a NEW date not in the cache
   (e.g. today), runs a real forward pass of the deployed champion over features built
   by the exact training pipeline — verified to reproduce the cache bit-for-bit, so
   there is no train/serve skew. This is what makes the product realtime.

Emits the RAW magnitude the gate consumes. R1: if neither backend can serve the
(symbol, date) — no cache row and no deployed model — it raises loudly; it never
invents a prediction. ``news_residual`` is not in the frozen cache for the all-scope
champion, so it is reported as ``None`` (not fabricated).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..frozen_predictions import PredictionNotCachedError, get_store
from ..state import MultiAgentState


def predict_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: serve the CMTF prediction (frozen cache, else live inference).

    Reads: symbol, target_horizon_days, prediction_time
    Writes: final_pred (seed-mean), gate_pred (raw, gated), seed_preds, baseline_pred
            (None), news_residual (None), model_evidence, artifact_versions, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbol = state["symbol"]
    horizon = state["target_horizon_days"]
    date = state["prediction_time"]

    source = "frozen_prediction_cache"
    truth = None
    try:
        fp = get_store(horizon, cfg).get(symbol, date)
        seed_preds = fp.seed_preds
        final_pred = fp.ensemble_pred
        gate_pred = fp.gate_pred
        truth = fp.truth
    except PredictionNotCachedError:
        # Not in the research book → real forward pass of the deployed champion.
        if not getattr(cfg, "enable_live_inference", True):
            raise
        from ..live_inference import predict_live
        logger.info("PredictAgent | {} {} not cached → LIVE inference (deployed champion)…", symbol, date)
        lp = predict_live(symbol, date, horizon, cfg, data_end=state.get("data_end") or date)
        seed_preds = lp.seed_preds
        final_pred = float(np.mean(seed_preds))
        # Live path is the 3-seed ensemble mean (matches the calibrated gate).
        gate_pred = lp.gate_pred if not cfg.gate_on_raw_seed else seed_preds[0]
        truth = lp.truth
        source = "live_inference"

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
    }

    elapsed = time.time() - t0
    logger.info(
        "PredictAgent | {} {} {}d | seed_mean={:.5f} gate_pred={:.5f} ({}, {} seeds) | {:.3f}s",
        symbol, date, horizon, final_pred, gate_pred, source, len(seed_preds), elapsed,
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
        "model_evidence": model_evidence,
        "artifact_versions": artifact_versions,
        "node_timings": {"predict_agent": elapsed},
    }
