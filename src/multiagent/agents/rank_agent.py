"""Rank Agent — cross-sectional ranking (plan §3.8; honest, universe-limited).

For a single date, rank the requested symbols by their matched-scope CMTF prediction
and bucket into top longs / bottom shorts / abstained middle. This is the ONE place
the genuine cross-sectional signal (§0: matched-scope per-date IC +0.04, all-scope ~0)
is deployed, so it reads the MATCHED cell's frozen predictions, not the all-scope core.

The bucketing is purely rank-based (long the top fraction, short the bottom, abstain
the middle) — a cross-sectional policy needs no magnitude tau and no calibration, so
there is no leak. This is a SECONDARY, universe-limited claim (underpowered at 7
names, p≈0.06); the node always reports how many names were ranked vs dropped (R1).
"""

from __future__ import annotations

import math
import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..frozen_predictions import MATCHED_CELL_ID, PredictionNotCachedError, get_store
from ..state import MultiAgentState


def rank_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: rank symbols cross-sectionally for a single date.

    Reads: target_symbols, target_horizon_days, prediction_time
    Writes: ranking, rank_longs, rank_shorts, rank_abstained, node_timings, warnings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbols = state.get("target_symbols") or ([state["symbol"]] if state.get("symbol") else [])
    horizon = state["target_horizon_days"]
    date = state["prediction_time"]

    store = get_store(horizon, cfg, cell_id=MATCHED_CELL_ID)

    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    for sym in symbols:
        try:
            fp = store.get(sym, date)
        except PredictionNotCachedError:
            dropped.append(sym)  # surfaced, never silently ignored (R1)
            continue
        rows.append({"symbol": sym, "gate_pred": fp.gate_pred, "conviction": abs(fp.gate_pred)})

    # Sort by signed prediction (this ordering IS the cross-sectional IC signal).
    rows.sort(key=lambda r: r["gate_pred"], reverse=True)
    n = len(rows)
    # Long the top tranche, short the bottom tranche, abstain the middle.
    k = max(1, math.floor(cfg.gate_coverage * n)) if n else 0
    longs, shorts, abstained = [], [], []
    for i, r in enumerate(rows):
        if i < k and r["gate_pred"] > 0:
            r["bucket"] = "long"; longs.append(r["symbol"])
        elif i >= n - k and r["gate_pred"] < 0:
            r["bucket"] = "short"; shorts.append(r["symbol"])
        else:
            r["bucket"] = "abstain"; abstained.append(r["symbol"])
        r["rank"] = i + 1

    elapsed = time.time() - t0
    logger.info("RankAgent | {} ranked, {} dropped | longs={} shorts={} abstain={} | {:.3f}s",
                n, len(dropped), longs, shorts, abstained, elapsed)

    warnings = []
    if dropped:
        warnings.append(f"rank_agent: no frozen prediction for {dropped} on {date} (insufficient data)")

    return {
        "ranking": rows,
        "rank_longs": longs,
        "rank_shorts": shorts,
        "rank_abstained": abstained,
        "warnings": warnings,
        "node_timings": {"rank_agent": elapsed},
    }
