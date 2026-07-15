"""Risk Agent — one-way safety veto (plan §3.7).

Risk is NEVER the decision-maker. It starts from the gate's action and can only
DOWNGRADE a trade to abstain when the market is too dangerous (extreme volatility
or drawdown). It can never turn an abstain into a trade, never up-size, and never
change a long into a short. No LLM, no policy file, no tiers.

Invariant (enforced by test): abstain-in ⇒ abstain-out.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def risk_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: one-way safety veto over the gate's decision.

    Reads: gated_action, position_scale, volatility_metrics
    Writes: action, position_scale, risk_vetoed, veto_reasons, decision_reasoning,
            node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    gated_action = state["gated_action"]
    position_scale = float(state.get("position_scale", 0.0))

    vol_metrics = state.get("volatility_metrics", {})
    vol_20d = float(vol_metrics.get("vol_20d", 0.0))
    max_drawdown_pct = float(vol_metrics.get("max_drawdown_pct", 0.0))

    veto_reasons: list[str] = []
    is_trade = gated_action in ("long", "short")
    if is_trade:
        if vol_20d > cfg.hard_block_vol:
            veto_reasons.append(f"vol={vol_20d:.1f}%>{cfg.hard_block_vol:.0f}%")
        if max_drawdown_pct > cfg.hard_block_drawdown:
            veto_reasons.append(f"dd={max_drawdown_pct:.1f}%>{cfg.hard_block_drawdown:.0f}%")

    if veto_reasons:
        action = "abstain"
        position_scale = 0.0
        risk_vetoed = True
        decision_reasoning = (
            f"Risk VETO ({', '.join(veto_reasons)}): {gated_action}→abstain. "
            f"vol={vol_20d:.1f}% dd={max_drawdown_pct:.1f}%."
        )
    else:
        # Pass the gate's decision through unchanged (one-way: only downgrades).
        action = gated_action
        risk_vetoed = False
        decision_reasoning = (
            f"Risk OK: action={action} size={position_scale:+.2f}. "
            f"vol={vol_20d:.1f}% dd={max_drawdown_pct:.1f}%."
        )

    elapsed = time.time() - t0
    logger.info(
        "RiskAgent | {}→{} size={:+.2f} vetoed={} | {:.4f}s",
        gated_action, action, position_scale, risk_vetoed, elapsed,
    )

    return {
        "action": action,
        "position_scale": position_scale,
        "risk_vetoed": risk_vetoed,
        "veto_reasons": veto_reasons,
        "decision_reasoning": decision_reasoning,
        "node_timings": {"risk_agent": elapsed},
    }
