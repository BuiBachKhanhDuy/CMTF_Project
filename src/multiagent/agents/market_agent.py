"""Market Agent — computes volatility metrics and technical-indicator proposal.

Pure analytical node: reads market data from state (fetched by orchestrator).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def _compute_volatility_metrics(close_window: np.ndarray) -> dict[str, float]:
    """Compute volatility metrics from close price window."""
    if len(close_window) < 2:
        return {"vol_20d": 0.0, "max_drawdown_pct": 0.0, "trend_pct": 0.0}

    log_returns = np.diff(np.log(close_window))
    if len(log_returns) >= 20:
        vol_20d = float(np.std(log_returns[-20:]) * np.sqrt(252))
    else:
        vol_20d = float(np.std(log_returns) * np.sqrt(252))

    cummax = np.maximum.accumulate(close_window)
    drawdowns = (close_window - cummax) / np.where(cummax > 0, cummax, 1.0)
    max_drawdown_pct = float(abs(np.min(drawdowns))) * 100

    trend_pct = float((close_window[-1] / close_window[0] - 1) * 100)

    return {
        "vol_20d": round(vol_20d * 100, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trend_pct": round(trend_pct, 2),
    }


def _get_feature(name: str, tabular: np.ndarray, col_names: list[str]) -> float:
    """Look up a named feature value from market_tabular."""
    if name in col_names:
        return float(tabular[col_names.index(name)])
    return 0.0


def _build_market_proposal(
    tabular: np.ndarray,
    col_names: list[str],
    volatility_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build forward-looking market proposal from technical indicators."""
    rsi = _get_feature("rsi_14", tabular, col_names)
    macd_hist = _get_feature("macd_hist", tabular, col_names)
    atr = _get_feature("atr_14", tabular, col_names)
    close = _get_feature("close", tabular, col_names)
    bb_lower = _get_feature("bb_lower", tabular, col_names)
    bb_mid = _get_feature("bb_mid", tabular, col_names)
    bb_upper = _get_feature("bb_upper", tabular, col_names)

    # RSI mean-reversion: oversold (< 50) → positive, overbought (> 50) → negative
    rsi_score = max(-1.0, min(1.0, (50.0 - rsi) / 50.0))

    # MACD histogram normalized by ATR
    macd_score = max(-1.0, min(1.0, macd_hist / max(abs(atr), 1e-6)))

    # Bollinger Band position: below mid → long bias, above mid → short bias
    bb_width = bb_upper - bb_lower
    bb_score = max(-1.0, min(1.0, (bb_mid - close) / max(abs(bb_width), 1e-6)))

    # Aggregate and scale to ±0.05 (same magnitude as CMTF preds)
    raw = (rsi_score + macd_score + bb_score) / 3.0
    score = max(-0.05, min(0.05, raw * 0.05))

    # Confidence = indicator agreement
    signs = [np.sign(rsi_score), np.sign(macd_score), np.sign(bb_score)]
    nonzero = [s for s in signs if s != 0]
    if nonzero:
        majority = np.sign(sum(nonzero))
        agreement = sum(1 for s in nonzero if s == majority) / len(nonzero)
    else:
        agreement = 0.0

    vol_20d = volatility_metrics.get("vol_20d", 0.0)
    vol_discount = min(1.0, vol_20d / 60.0)
    confidence = round(max(0.0, min(1.0, agreement * (1.0 - vol_discount))), 3)

    direction = "flat" if abs(score) < 0.001 else ("long" if score > 0 else "short")

    return {
        "direction": direction,
        "score": round(score, 6),
        "confidence": confidence,
        "rationale": (
            f"rsi={rsi:.1f}({rsi_score:+.2f}) macd_h={macd_hist:+.4f}({macd_score:+.2f}) "
            f"bb_pos=({bb_score:+.2f}) vol={vol_20d:.1f}%"
        ),
        "quality": {
            "vol_20d": vol_20d,
            "max_drawdown_pct": volatility_metrics.get("max_drawdown_pct", 0.0),
        },
    }


def market_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Market Agent — compute volatility and technical proposal.

    Reads from state (populated by orchestrator):
        close_window, market_tabular, market_feature_cols
    Writes: volatility_metrics, market_proposal, node_timings
    """
    t0 = time.time()

    close_window = state["close_window"]
    market_tabular = state["market_tabular"]
    col_names = state.get("market_feature_cols", [])

    volatility_metrics = _compute_volatility_metrics(close_window)
    market_proposal = _build_market_proposal(market_tabular, col_names, volatility_metrics)

    elapsed = time.time() - t0
    logger.info(
        "MarketAgent | vol={:.1f}% dd={:.1f}% proposal={} | {:.2f}s",
        volatility_metrics["vol_20d"],
        volatility_metrics["max_drawdown_pct"],
        market_proposal["direction"],
        elapsed,
    )

    return {
        "volatility_metrics": volatility_metrics,
        "market_proposal": market_proposal,
        "node_timings": {"market_agent": elapsed},
    }
