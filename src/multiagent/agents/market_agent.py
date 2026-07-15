"""Market Agent — collects market data (OHLCV, volatility) for predict and risk agents."""

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


def market_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Market Agent — fetch OHLCV and compute volatility metrics.

    Reads: symbol, prediction_time, sequence_len
    Writes: close_window, market_window, market_tabular, token_ids, attention_mask,
            volatility_metrics, data_cutoff, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbol = state["symbol"]
    cutoff = state["prediction_time"]
    seq_len = state.get("sequence_len", cfg.sequence_len)

    from src.pipeline.orchestrator import prepare_single_cutoff

    result = prepare_single_cutoff(
        symbol=symbol,
        cutoff=cutoff,
        sequence_len=seq_len,
        news_cache_dir=str(cfg.news_cache_dir),
        sentiment_output_dir=str(cfg.sentiment_output_dir),
    )

    close_window = result["close_window"]
    volatility_metrics = _compute_volatility_metrics(close_window)

    elapsed = time.time() - t0
    logger.info(
        "MarketAgent | {} cutoff={} | vol={:.1f}% dd={:.1f}% | {:.2f}s",
        symbol, cutoff, volatility_metrics["vol_20d"],
        volatility_metrics["max_drawdown_pct"], elapsed,
    )

    return {
        "close_window": close_window,
        "market_window": result["market_window"],
        "market_tabular": result["market_tabular"],
        "volatility_metrics": volatility_metrics,
        "data_cutoff": cutoff,
        "node_timings": {"market_agent": elapsed},
    }
