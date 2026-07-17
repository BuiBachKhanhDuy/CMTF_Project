"""Gate Agent — the decision core (plan §3.6).

Loads the frozen, validation-calibrated :class:`GatePolicy` for the horizon and
applies it verbatim, reusing ``src/benchmark/decision_policy.py`` unchanged so the
runtime decision is byte-identical to the research pipeline. This is the ONLY node
that turns a prediction into a trade/abstain.

R1: a missing or version-mismatched policy artifact raises loudly (no ad-hoc tau,
no "confident" default). Below tau ⇒ honest abstain, size 0.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from src.benchmark.decision_policy import apply_positions

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..gate_io import load_gate_policy, policy_path
from ..state import MultiAgentState


def gate_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: apply the frozen GatePolicy to the raw prediction.

    Reads: gate_pred, target_horizon_days, artifact_versions
    Writes: gated_action, position_scale, gate_tau, gate_coverage, gate_val_score,
            gate_disclosure_da_pct, gate_disclosure_base_rate_pct, gate_reason,
            node_timings, warnings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    horizon = state["target_horizon_days"]
    gate_pred = float(state["gate_pred"])

    # Version stamps of the model actually loaded upstream, so a policy calibrated
    # on a different backbone/CMTF version is refused (StalePolicyError, R1).
    versions = state.get("artifact_versions", {})
    path = policy_path(cfg.gate_policy_dir, horizon, symbol="VN")
    policy, meta = load_gate_policy(
        path,
        expect_cmtf_version=versions.get("cmtf_version"),
        expect_backbone_version=versions.get("backbone_version"),
    )

    tau = policy.tau
    if abs(gate_pred) < tau:
        gated_action = "abstain"
        position_scale = 0.0
        reason = f"|pred|={abs(gate_pred):.4f} < tau={tau:.4f} -> abstain"
    else:
        # Reuse the exact research sizing (conviction-clipped) so runtime == research.
        pos = float(apply_positions(np.array([gate_pred]), policy)[0])
        gated_action = "long" if pos > 0 else "short"
        position_scale = pos
        reason = (
            f"|pred|={abs(gate_pred):.4f} >= tau={tau:.4f} -> {gated_action} "
            f"@ size={position_scale:+.2f}"
        )

    disclosure_da = meta.get("val_gated_DA%")
    disclosure_base_rate = meta.get("val_base_rate_DA%")
    warnings = []
    if disclosure_da is None or disclosure_base_rate is None:
        warnings.append(
            f"gate: {path} has no honest disclosure numbers (schema predates v2) — "
            f"recalibrate (`python -m src.multiagent calibrate --horizon {horizon}`)"
        )

    elapsed = time.time() - t0
    logger.info(
        "GateAgent | pred={:+.4f} tau={:.4f} cov={:.2f} → {} size={:+.2f} | {:.4f}s",
        gate_pred, tau, policy.coverage, gated_action, position_scale, elapsed,
    )

    return {
        "gated_action": gated_action,
        "position_scale": position_scale,
        "gate_tau": tau,
        "gate_coverage": policy.coverage,
        "gate_val_score": policy.val_score,
        "gate_disclosure_da_pct": disclosure_da,
        "gate_disclosure_base_rate_pct": disclosure_base_rate,
        "gate_reason": reason,
        "artifact_versions": {"gate_policy": f"{meta.get('config_hash')}@cov{meta.get('calibration_coverage')}"},
        "warnings": warnings,
        "node_timings": {"gate_agent": elapsed},
    }
