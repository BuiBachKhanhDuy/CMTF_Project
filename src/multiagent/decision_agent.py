"""Decision Agent — applies threshold policy after critic overrides."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .state import MultiAgentState


def decision_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: determine final trading action and position scale.

    Combines critic outputs with threshold-based policy to produce
    the final action ("long", "short", "flat") and position_scale [0, 1].

    Reads: final_pred_adjusted, position_scale_regime, disagreement_force_flat
    Writes: action, position_scale, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    final_pred_adjusted = state["final_pred_adjusted"]
    position_scale_regime = state["position_scale_regime"]
    disagreement_force_flat = state["disagreement_force_flat"]

    # --- Decision logic ---
    if disagreement_force_flat or position_scale_regime == 0.0:
        action = "flat"
        position_scale = 0.0
    elif final_pred_adjusted >= cfg.buy_threshold:
        action = "long"
        position_scale = position_scale_regime
    elif final_pred_adjusted <= -cfg.sell_threshold:
        action = "short"
        position_scale = position_scale_regime
    else:
        action = "flat"
        position_scale = 0.0

    elapsed = time.time() - t0
    logger.info(
        "DecisionAgent | pred_adj={:.5f} | action={} scale={:.2f} | {:.3f}s",
        final_pred_adjusted, action, position_scale, elapsed,
    )

    timings = dict(state.get("node_timings", {}))
    timings["decision_agent"] = elapsed

    return {
        "action": action,
        "position_scale": position_scale,
        "node_timings": timings,
    }
