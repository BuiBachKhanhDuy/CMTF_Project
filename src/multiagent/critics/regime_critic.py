"""Risk/Regime Critic — deterministic volatility and drawdown checker."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def regime_critic_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: assess market regime and set position_scale_regime.

    Deterministic rules based on realized volatility and max drawdown of the
    close price window. No ML, no LLM.

    Reads: close_window, market_window
    Writes: regime_flags, position_scale_regime, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    close_window = state["close_window"]

    # --- Compute regime indicators ---

    # 1. Realized 20-day volatility (annualized)
    if len(close_window) >= 2:
        log_returns = np.diff(np.log(close_window))
        vol_20d = float(np.std(log_returns[-20:]) * np.sqrt(252)) if len(log_returns) >= 20 else float(np.std(log_returns) * np.sqrt(252))
    else:
        vol_20d = 0.0

    # 2. Maximum drawdown over the window
    cummax = np.maximum.accumulate(close_window)
    drawdowns = (close_window - cummax) / np.where(cummax > 0, cummax, 1.0)
    max_drawdown = float(np.min(drawdowns))  # Most negative value
    max_drawdown_abs = abs(max_drawdown)

    # 3. VN-Index z-score (from market features if available)
    market_window = state.get("market_window")
    vnindex_zscore = 0.0
    if market_window is not None and market_window.shape[1] > 0:
        # VN-Index ret is typically one of the last columns in market features
        # We'll compute z-score of the last window's VN-Index returns if available
        # For now, use the sentiment features or market_tabular
        pass  # VN-Index z-score is optional — skipped in v1

    # --- Apply rules ---
    high_vol = vol_20d > cfg.vol_high_pct
    drawdown_breach = max_drawdown_abs > cfg.dd_max_pct

    if drawdown_breach:
        position_scale_regime = 0.0
    elif high_vol:
        position_scale_regime = 0.5
    else:
        position_scale_regime = 1.0

    regime_flags = {
        "vol_20d": vol_20d,
        "max_drawdown": max_drawdown,
        "max_drawdown_abs": max_drawdown_abs,
        "high_vol": high_vol,
        "drawdown_breach": drawdown_breach,
        "vnindex_zscore": vnindex_zscore,
        "vol_threshold": cfg.vol_high_pct,
        "dd_threshold": cfg.dd_max_pct,
    }

    elapsed = time.time() - t0
    logger.info(
        "RegimeCritic | vol={:.4f} dd={:.4f} | scale={:.1f} | {:.3f}s",
        vol_20d, max_drawdown_abs, position_scale_regime, elapsed,
    )

    timings = dict(state.get("node_timings", {}))
    timings["regime_critic"] = elapsed

    return {
        "regime_flags": regime_flags,
        "position_scale_regime": position_scale_regime,
        "node_timings": timings,
    }
