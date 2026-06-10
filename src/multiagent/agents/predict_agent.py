"""Predict Agent — runs CMTF v8 ensemble inference and produces prediction output.

This is the ML core: takes market data + news embeddings, runs cross-modal
fusion model (3 seeds), and outputs predictions with model evidence.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..loaders import get_cmtf_ensemble
from ..state import MultiAgentState


def _compute_predict_confidence(final_pred: float, seed_preds: list[float]) -> float:
    """Compute prediction confidence from pred strength and seed agreement."""
    abs_pred = abs(final_pred)
    seed_var = float(np.var(seed_preds))

    pred_strength = min(1.0, abs_pred / 0.05)
    # max_expected_var calibrated to real 3-seed CMTF ensemble variance
    max_expected_var = 1e-4
    agreement_factor = max(0.0, 1.0 - seed_var / max_expected_var)

    return round(pred_strength * agreement_factor, 3)


def predict_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Predict Agent — CMTF ensemble inference.

    Reads: symbol, target_horizon_days, close_window, market_window, market_tabular,
           news_emb, news_mask
    Writes: baseline_pred, final_pred, seed_preds, news_residual, attn_weights,
            news_weight, predict_confidence, model_evidence, artifact_versions, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbol = state["symbol"]
    horizon = state["target_horizon_days"]
    close_window = state["close_window"]
    market_window = state["market_window"]
    market_tabular = state["market_tabular"]
    news_emb = state["news_emb"]
    news_mask = state["news_mask"]

    # Load ensemble (cached after first call)
    ensemble = get_cmtf_ensemble(symbol, horizon, cfg)

    # Tokenize close window
    close_2d = close_window.reshape(1, -1)
    token_ids, attention_mask = ensemble[0].tokenize_windows(close_2d)

    # Prepare batched inputs
    news_batch = news_emb[np.newaxis, ...]
    news_mask_batch = news_mask[np.newaxis, ...]
    market_batch = market_window[np.newaxis, ...] if market_window is not None else None
    tabular_batch = market_tabular[np.newaxis, ...] if market_tabular is not None else None

    # Run each seed
    seed_preds: list[float] = []
    baseline_preds: list[float] = []
    all_attn_weights: list[np.ndarray] = []
    all_news_weights: list[float] = []

    for predictor in ensemble:
        result = predictor.predict_with_explanation(
            token_ids=token_ids,
            attention_mask=attention_mask,
            news_test=news_batch,
            tabular_test=tabular_batch,
            market_windows_test=market_batch,
            news_mask_test=news_mask_batch,
        )
        seed_preds.append(result["final_pred"])
        baseline_preds.append(result["baseline_pred"])
        all_attn_weights.append(result["attn_weights"])
        all_news_weights.append(result["news_weight"])

    # Aggregate
    final_pred = float(np.mean(seed_preds))
    baseline_pred = float(np.mean(baseline_preds))
    news_residual = final_pred - baseline_pred
    attn_weights = np.mean(np.stack(all_attn_weights), axis=0)
    news_weight = float(np.mean(all_news_weights))
    predict_confidence = _compute_predict_confidence(final_pred, seed_preds)

    # Model evidence for risk agent and answer agent
    model_evidence = {
        "final_pred": final_pred,
        "baseline_pred": baseline_pred,
        "seed_preds": [round(p, 6) for p in seed_preds],
        "seed_variance": round(float(np.var(seed_preds)), 8),
        "spread": round(float(np.max(seed_preds) - np.min(seed_preds)), 6),
        "all_same_sign": bool(np.all(np.sign(seed_preds) == np.sign(seed_preds[0])) and seed_preds[0] != 0),
        "news_residual": round(news_residual, 6),
        "news_weight": round(news_weight, 3),
        "predict_confidence": predict_confidence,
    }

    elapsed = time.time() - t0
    logger.info(
        "PredictAgent | {} {}d | baseline={:.5f} final={:.5f} conf={:.3f} | {:.2f}s",
        symbol, horizon, baseline_pred, final_pred, predict_confidence, elapsed,
    )

    artifact_versions = dict(state.get("artifact_versions", {}))
    artifact_versions["cmtf_version"] = cfg.cmtf_version
    artifact_versions["ensemble_seeds"] = str(cfg.ensemble_seeds)

    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "baseline_pred": baseline_pred,
        "final_pred": final_pred,
        "seed_preds": seed_preds,
        "news_residual": news_residual,
        "attn_weights": attn_weights,
        "news_weight": news_weight,
        "predict_confidence": predict_confidence,
        "model_evidence": model_evidence,
        "artifact_versions": artifact_versions,
        "node_timings": {"predict_agent": elapsed},
    }
