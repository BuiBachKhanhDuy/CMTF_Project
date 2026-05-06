"""Fusion Agent — runs CMTF v8 ensemble inference with full explanation payload."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .loaders import get_cmtf_ensemble
from .state import MultiAgentState


def fusion_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: run CMTF v8 ensemble (3 seeds) and return interpretability payload.

    Reads: symbol, target_horizon_days, close_window, market_window, market_tabular,
           news_emb, news_mask
    Writes: baseline_pred, final_pred, seed_preds, news_residual, attn_weights,
            news_weight, token_ids, attention_mask, artifact_versions, node_timings
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

    # Tokenize close window using the backbone from the first predictor
    # close_window shape: (seq_len,) → needs to be (1, seq_len) for tokenize_windows
    close_2d = close_window.reshape(1, -1)
    token_ids, attention_mask = ensemble[0].tokenize_windows(close_2d)

    # Prepare inputs for predict_with_explanation
    # news_emb: (seq_len, news_dim) → (1, seq_len, news_dim)
    news_batch = news_emb[np.newaxis, ...]  # (1, seq_len, news_dim)
    news_mask_batch = news_mask[np.newaxis, ...]  # (1, seq_len)

    # market_window: (seq_len, n_feat) → (1, seq_len, n_feat)
    market_batch = market_window[np.newaxis, ...] if market_window is not None else None

    # market_tabular: (n_feat,) → (1, n_feat)
    tabular_batch = market_tabular[np.newaxis, ...] if market_tabular is not None else None

    # Run each seed and collect results
    seed_preds: list[float] = []
    baseline_preds: list[float] = []
    all_attn_weights: list[np.ndarray] = []
    all_news_weights: list[float] = []

    for i, predictor in enumerate(ensemble):
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

    # Average across seeds
    final_pred = float(np.mean(seed_preds))
    baseline_pred = float(np.mean(baseline_preds))
    news_residual = final_pred - baseline_pred
    attn_weights = np.mean(np.stack(all_attn_weights), axis=0)  # (seq_len,)
    news_weight = float(np.mean(all_news_weights))

    elapsed = time.time() - t0
    logger.info(
        "FusionAgent | {} {}d | baseline={:.5f} final={:.5f} residual={:.5f} | {:.2f}s",
        symbol, horizon, baseline_pred, final_pred, news_residual, elapsed,
    )

    timings = dict(state.get("node_timings", {}))
    timings["fusion_agent"] = elapsed

    artifact_versions = dict(state.get("artifact_versions", {}))
    artifact_versions["cmtf_version"] = cfg.cmtf_version
    artifact_versions["ensemble_seeds"] = str(cfg.ensemble_seeds)

    return {
        "baseline_pred": baseline_pred,
        "final_pred": final_pred,
        "seed_preds": seed_preds,
        "news_residual": news_residual,
        "attn_weights": attn_weights,
        "news_weight": news_weight,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "artifact_versions": artifact_versions,
        "node_timings": timings,
    }
