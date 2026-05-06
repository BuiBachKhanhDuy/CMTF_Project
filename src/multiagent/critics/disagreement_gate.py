"""Ensemble Disagreement Gate — deterministic seed sign check."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def disagreement_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: check if ensemble seeds agree on direction.

    If any seed prediction has a different sign from the mean, force flat.
    This is the simplest ensemble-calibration gate: disagreement → abstain.

    Reads: seed_preds
    Writes: disagreement_force_flat, node_timings
    """
    _ = config  # No config params needed for this gate
    t0 = time.time()

    seed_preds = state["seed_preds"]

    # Determine mean sign
    mean_pred = float(np.mean(seed_preds))
    mean_sign = np.sign(mean_pred)

    # Check if any seed disagrees with the mean sign
    disagreement_force_flat = False
    if mean_sign != 0:
        for sp in seed_preds:
            if np.sign(sp) != mean_sign:
                disagreement_force_flat = True
                break
    else:
        # Mean is exactly zero → ambiguous, treat as disagreement
        disagreement_force_flat = True

    elapsed = time.time() - t0
    logger.info(
        "DisagreementGate | seeds={} mean_sign={} | force_flat={} | {:.3f}s",
        [f"{s:.5f}" for s in seed_preds],
        int(mean_sign),
        disagreement_force_flat,
        elapsed,
    )

    timings = dict(state.get("node_timings", {}))
    timings["disagreement_gate"] = elapsed

    return {
        "disagreement_force_flat": disagreement_force_flat,
        "node_timings": timings,
    }
