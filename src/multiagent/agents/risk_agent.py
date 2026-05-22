"""Risk Agent — final decision authority via tiered position sizing.

Owns the final trading action and position_scale. No LLM, no loops.
Three tiers: hard block → reduced position → full position.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..reflection import load_policy
from ..state import MultiAgentState


def _determine_action(score: float, buy_threshold: float, sell_threshold: float, weak_signal: float) -> str:
    """Map prediction to action via thresholds."""
    if abs(score) < weak_signal:
        return "flat"
    if score >= buy_threshold:
        return "long"
    if score <= sell_threshold:
        return "short"
    return "flat"


def _tiered_risk(
    score: float,
    confidence: float,
    vol_20d: float,
    max_drawdown_pct: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Tiered position sizing: hard block / reduced / full.

    Returns dict with tier, position_scale, action, and individual checks.
    """
    hard_vol = float(policy.get("hard_block_vol", 40.0))
    hard_dd = float(policy.get("hard_block_drawdown", 20.0))
    hard_conf = float(policy.get("hard_block_min_confidence", 0.10))
    reduced_vol = float(policy.get("reduced_vol", 30.0))
    reduced_conf = float(policy.get("reduced_min_confidence", 0.25))

    buy_thresh = float(policy["buy_threshold"])
    sell_thresh = float(policy["sell_threshold"])
    weak = float(policy["weak_signal"])

    action = _determine_action(score, buy_thresh, sell_thresh, weak)

    # --- Hard block ---
    hard_block_reasons = []
    if vol_20d > hard_vol:
        hard_block_reasons.append(f"vol={vol_20d:.1f}%>{hard_vol:.0f}%")
    if max_drawdown_pct > hard_dd:
        hard_block_reasons.append(f"dd={max_drawdown_pct:.1f}%>{hard_dd:.0f}%")
    if confidence < hard_conf:
        hard_block_reasons.append(f"conf={confidence:.3f}<{hard_conf}")

    if hard_block_reasons or action == "flat":
        tier = "blocked" if hard_block_reasons else "flat_signal"
        return {
            "tier": tier,
            "action": "flat" if hard_block_reasons else action,
            "position_scale": 0.0,
            "hard_block_reasons": hard_block_reasons,
            "vol_20d": vol_20d,
            "max_drawdown_pct": max_drawdown_pct,
            "confidence": confidence,
        }

    # --- Reduced position ---
    reduced_reasons = []
    if confidence < reduced_conf:
        reduced_reasons.append(f"conf={confidence:.3f}<{reduced_conf}")
    if vol_20d > reduced_vol:
        reduced_reasons.append(f"vol={vol_20d:.1f}%>{reduced_vol:.0f}%")

    if reduced_reasons:
        # Scale linearly in [0.3, 0.5] based on confidence distance from hard_conf
        conf_range = reduced_conf - hard_conf
        if conf_range > 0:
            frac = min(1.0, (confidence - hard_conf) / conf_range)
        else:
            frac = 0.5
        scale = round(0.3 + 0.2 * frac, 2)
        return {
            "tier": "reduced",
            "action": action,
            "position_scale": scale,
            "reduced_reasons": reduced_reasons,
            "vol_20d": vol_20d,
            "max_drawdown_pct": max_drawdown_pct,
            "confidence": confidence,
        }

    # --- Full position ---
    scale = round(max(0.3, min(1.0, confidence)), 2)
    return {
        "tier": "full",
        "action": action,
        "position_scale": scale,
        "vol_20d": vol_20d,
        "max_drawdown_pct": max_drawdown_pct,
        "confidence": confidence,
    }


def risk_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Risk Agent — tiered position sizing.

    Three tiers based on vol, drawdown, and fused confidence:
      - Hard block: extreme risk → flat, scale=0
      - Reduced: marginal conditions → trade at 0.3–0.5 scale
      - Full: confident + calm market → scale ∝ confidence

    Reads: fusion_decision, volatility_metrics
    Writes: action, position_scale, final_confidence, risk_checks, decision_reasoning, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    policy = load_policy(cfg.policy_store_path)

    fusion = state.get("fusion_decision", {})
    if not fusion:
        raise ValueError("fusion_decision missing in state; fusion_agent must run before risk_agent")

    fusion_score = float(fusion.get("score", 0.0))
    fusion_confidence = float(fusion.get("confidence", 0.0))
    vol_metrics = state.get("volatility_metrics", {})

    vol_20d = float(vol_metrics.get("vol_20d", 0.0))
    max_drawdown_pct = float(vol_metrics.get("max_drawdown_pct", 0.0))

    risk_result = _tiered_risk(
        fusion_score, fusion_confidence, vol_20d, max_drawdown_pct, policy,
    )

    action = risk_result["action"]
    position_scale = risk_result["position_scale"]
    tier = risk_result["tier"]
    final_confidence = round(fusion_confidence * position_scale, 3)

    # Build reasoning
    if tier == "blocked":
        decision_reasoning = (
            f"Risk BLOCKED ({', '.join(risk_result['hard_block_reasons'])}). "
            f"score={fusion_score:+.5f} vol={vol_20d:.1f}% dd={max_drawdown_pct:.1f}% conf={fusion_confidence:.3f}."
        )
    elif tier == "reduced":
        decision_reasoning = (
            f"Risk REDUCED ({', '.join(risk_result['reduced_reasons'])}): "
            f"action={action} scale={position_scale:.2f}. "
            f"score={fusion_score:+.5f} vol={vol_20d:.1f}% dd={max_drawdown_pct:.1f}% conf={fusion_confidence:.3f}."
        )
    else:
        decision_reasoning = (
            f"Risk APPROVED ({tier}): action={action} scale={position_scale:.2f}. "
            f"score={fusion_score:+.5f} vol={vol_20d:.1f}% dd={max_drawdown_pct:.1f}% conf={fusion_confidence:.3f}."
        )

    elapsed = time.time() - t0
    logger.info(
        "RiskAgent | {} tier={} scale={:.2f} conf={:.3f} | {:.4f}s",
        action, tier, position_scale, final_confidence, elapsed,
    )

    return {
        "action": action,
        "position_scale": position_scale,
        "final_confidence": final_confidence,
        "risk_checks": risk_result,
        "decision_reasoning": decision_reasoning,
        "policy_version": int(policy.get("version", 1)),
        "node_timings": {"risk_agent": elapsed},
    }
