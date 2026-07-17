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


def compute_range_stats(frame, symbol: str, date_start: str, date_end: str) -> dict[str, Any]:
    """Real market statistics over an EXPLICIT calendar date range — no hardcoded
    lookback window. Splits the requested range at the boundary of what's actually
    known (the last date with a real forward return in ``frame``): the portion
    within that boundary is analyzed from real historical data; anything past it
    cannot be known from data alone and must come from the model's own forecast,
    which the caller is told about via ``coverage``/``needs_prediction_from``
    rather than this function inventing a number for it.

    Args:
        frame: the dataset parquet loaded via resolve_price_parquet (has
            `symbol` column, DatetimeIndex, and `fwd_ret_1d`).
        symbol, date_start, date_end: the exact range the query named.
    """
    import numpy as np
    import pandas as pd

    s = frame[frame["symbol"] == symbol].sort_index()
    if s.empty:
        return {"coverage": "none", "n_days": 0}

    start_ts, end_ts = pd.Timestamp(date_start), pd.Timestamp(date_end)
    known = s["fwd_ret_1d"].dropna()
    last_known = known.index.max() if not known.empty else None

    if last_known is None or start_ts > last_known:
        return {
            "coverage": "none", "n_days": 0,
            "needs_prediction_from": str(start_ts.date()),
            "requested_start": str(start_ts.date()), "requested_end": str(end_ts.date()),
        }

    covered_end = min(end_ts, last_known)
    window = known[(known.index >= start_ts) & (known.index <= covered_end)]
    daily = window.to_numpy(dtype=float)
    if len(daily) == 0:
        return {
            "coverage": "none", "n_days": 0,
            "needs_prediction_from": str(start_ts.date()),
            "requested_start": str(start_ts.date()), "requested_end": str(end_ts.date()),
        }

    # `fwd_ret_1d` is a LOG return (log(close[t+1]/close[t]), see
    # feature_engineer.py). Log returns compound by SUMMING then exponentiating —
    # `prod(1+r)` is the formula for SIMPLE returns and is wrong here (it overstated
    # a -7.63% month as -8.20%). Cumulative simple return = exp(sum(log_r)) - 1.
    cum_return = float((np.exp(np.sum(daily)) - 1.0) * 100)
    # Volatility: std of daily log returns, annualized — correct as-is for logs.
    vol = float(np.std(daily) * np.sqrt(252) * 100) if len(daily) > 1 else 0.0
    # Drawdown: build the real price-relative curve from the log returns
    # (exp of the cumulative sum), then measure the largest peak-to-trough drop as
    # a simple percentage — not a drop in log space.
    curve = np.exp(np.cumsum(daily))
    peak = np.maximum.accumulate(curve)
    max_dd = float(((peak - curve) / peak).max() * 100) if len(curve) else 0.0
    coverage = "full" if covered_end >= end_ts else "partial"

    return {
        "coverage": coverage,
        "n_days": int(len(daily)),
        "return_pct": round(cum_return, 2),
        "volatility_pct": round(vol, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "covered_start": str(window.index.min().date()),
        "covered_end": str(covered_end.date()),
        "requested_start": str(start_ts.date()),
        "requested_end": str(end_ts.date()),
        "needs_prediction_from": (
            str((covered_end + pd.Timedelta(days=1)).date()) if coverage == "partial" else None
        ),
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
