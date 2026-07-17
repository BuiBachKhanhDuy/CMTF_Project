"""Horizon Interaction Agent — cross-horizon conviction adjustment (NOT a veto).

Every other node between the gate and the narrator (`risk_agent`, `metalabel_agent`)
follows a strict one-way-veto invariant: they can only zero `position_scale`, never
scale it up. This node is deliberately different, and that difference is the entire
point of it: the OTHER two horizons' independently-calibrated gates see evidence about
the same (symbol, date) that the primary horizon's gate never looks at. When they
corroborate the primary horizon's direction, that is genuine extra evidence, not risk —
so this node scales conviction symmetrically (up on agreement, down on disagreement),
using a multiplier table frozen by `horizon_interaction_io.calibrate_interaction_from_cache`
(validation-only, leak-free, monotonicity-constrained, placebo-checked).

Missing/stale artifact handling is intentionally NOT the gate's fail-loud rule: this is
an enhancement layer, not a decision boundary, so a missing calibration or an
unreachable other-horizon prediction degrades to a neutral multiplier (1.0) plus a
logged warning — it never crashes the decision chain over a secondary signal.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..horizon_interaction_io import interaction_policy_path, load_interaction_policy
from ..raw_prediction import fetch_prediction_record
from ..state import MultiAgentState

_OTHER_HORIZONS = {1: (5, 20), 5: (1, 20), 20: (1, 5)}


def horizon_interaction_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: scale `position_scale` by cross-horizon sign agreement.

    Runs between `gate_agent` and `risk_agent` — at this point in the chain the gate's
    decision is in `gated_action`, NOT `action` (`risk_agent` is what first derives
    `action` from `gated_action`). Reading `action` here would silently see nothing
    and always skip the adjustment, so this node deliberately mirrors gate_agent's own
    field name rather than risk_agent's.

    Reads: gated_action, gate_pred, position_scale, symbol, prediction_time, target_horizon_days
    Writes: position_scale, horizon_agreement_score, horizon_interaction_multiplier,
            node_timings, warnings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    gated_action = state.get("gated_action", "abstain")
    position_scale = float(state.get("position_scale", 0.0))
    is_trade = gated_action in ("long", "short")

    warnings: list[str] = []
    agreement_score: int | None = None
    multiplier: float | None = None

    if is_trade and cfg.enable_horizon_interaction:
        primary_horizon = state["target_horizon_days"]
        symbol = state["symbol"]
        date = state["prediction_time"]
        other_horizons = _OTHER_HORIZONS[primary_horizon]

        try:
            policy, _ = load_interaction_policy(
                interaction_policy_path(cfg.horizon_interaction_dir, primary_horizon, symbol="VN"),
                expect_cmtf_version=cfg.cmtf_version,
                expect_backbone_version=cfg.backbone_version,
            )
            primary_sign = np.sign(float(state.get("gate_pred", 0.0)))
            agreement_score = 0
            for h in other_horizons:
                try:
                    other_pred = fetch_prediction_record(symbol, date, h, cfg, data_end=state.get("data_end")).gate_pred
                    if np.sign(other_pred) == primary_sign:
                        agreement_score += 1
                except Exception as e:  # noqa: BLE001 — secondary signal, degrade not crash
                    warnings.append(f"horizon_interaction: could not fetch {h}d prediction for "
                                    f"{symbol} {date} ({type(e).__name__}) — treated as non-agreeing")
            multiplier = policy.multiplier_by_agreement[agreement_score]
            position_scale = position_scale * multiplier
        except Exception as e:  # noqa: BLE001 — enhancement layer: degrade, never crash
            warnings.append(f"horizon_interaction: no calibrated artifact for {primary_horizon}d "
                            f"({type(e).__name__}) — no adjustment applied "
                            f"(run `python -m src.multiagent calibrate-interaction --horizon {primary_horizon}`)")
            agreement_score = None
            multiplier = 1.0

    elapsed = time.time() - t0
    logger.info(
        "HorizonInteraction | gated_action={} agreement={} multiplier={} size={:+.2f} | {:.4f}s",
        gated_action, agreement_score, multiplier, position_scale, elapsed,
    )

    return {
        "position_scale": position_scale,
        "horizon_agreement_score": agreement_score,
        "horizon_interaction_multiplier": multiplier,
        "warnings": warnings,
        "node_timings": {"horizon_interaction_agent": elapsed},
    }
