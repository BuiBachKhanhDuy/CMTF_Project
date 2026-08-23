"""Persist and calibrate validation-based ``GatePolicy`` artifacts.

Policies include their calibration provenance and schema version. Calibration
uses validation predictions only, with one pooled policy per horizon.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.decision_policy import (
    GatePolicy, calibrate_gate, calibrate_gate_fixed_coverage, evaluate_policy,
)
from src.benchmark.metrics import compute_all

from .loaders import ArtifactMissingError

# Increment when the policy schema or calibration procedure changes.
GATE_ARTIFACT_SCHEMA_VERSION = 2

# Default registry cell used to calibrate the gate.
CORE_CELL_ID = "0"

# Horizon-specific cell overrides used by the deployed policy.
CORE_CELL_BY_HORIZON: dict[int, str] = {5: "13", 20: "13"}


def core_cell_for(horizon: int) -> str:
    """Return the configured registry cell for ``horizon``."""
    return CORE_CELL_BY_HORIZON.get(int(horizon), CORE_CELL_ID)


class StalePolicyError(RuntimeError):
    """Raised when a loaded GatePolicy's version stamps do not match the request."""


def policy_path(gate_dir: str | Path, horizon: int, symbol: str = "VN") -> Path:
    """Path of the frozen policy artifact for (symbol, horizon)."""
    return Path(gate_dir) / f"{symbol}_{horizon}d.json"


def save_gate_policy(
    policy: GatePolicy,
    meta: dict[str, Any],
    path: str | Path,
) -> None:
    """Serialise ``policy`` + provenance ``meta`` to ``path`` (pretty JSON)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GATE_ARTIFACT_SCHEMA_VERSION,
        "policy": asdict(policy),
        **meta,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_gate_policy(
    path: str | Path,
    *,
    expect_cmtf_version: str | None = None,
    expect_backbone_version: str | None = None,
) -> tuple[GatePolicy, dict[str, Any]]:
    """Load a frozen GatePolicy, failing loud on missing/stale artifacts (R1).

    Raises
    ------
    ArtifactMissingError
        The artifact file does not exist. The caller must not invent a tau.
    StalePolicyError
        The artifact schema version is unknown, or its backbone/CMTF version
        stamps do not match the model actually loaded at runtime.
    """
    p = Path(path)
    if not p.exists():
        raise ArtifactMissingError(
            f"GatePolicy artifact not found: {p} — run `python -m src.multiagent "
            f"calibrate --horizon <H>` first (no ad-hoc threshold fallback)."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))

    schema = payload.get("schema_version")
    if schema != GATE_ARTIFACT_SCHEMA_VERSION:
        raise StalePolicyError(
            f"GatePolicy {p} has schema_version={schema}, runtime expects "
            f"{GATE_ARTIFACT_SCHEMA_VERSION} — recalibrate."
        )
    if expect_cmtf_version is not None and payload.get("cmtf_version") != expect_cmtf_version:
        raise StalePolicyError(
            f"GatePolicy {p} calibrated on cmtf_version={payload.get('cmtf_version')} "
            f"but runtime loaded {expect_cmtf_version} — recalibrate."
        )
    if expect_backbone_version is not None and payload.get("backbone_version") != expect_backbone_version:
        raise StalePolicyError(
            f"GatePolicy {p} calibrated on backbone_version={payload.get('backbone_version')} "
            f"but runtime loaded {expect_backbone_version} — recalibrate."
        )

    pol = payload["policy"]
    policy = GatePolicy(
        tau=float(pol["tau"]),
        conviction=bool(pol["conviction"]),
        conviction_scale=float(pol["conviction_scale"]),
        coverage=float(pol["coverage"]),
        val_score=float(pol["val_score"]),
    )
    return policy, payload


def _load_val_predictions(
    pred_dir: Path,
    config_hash: str,
    horizon: int,
    seed: int | None,
) -> np.ndarray:
    """Load frozen validation predictions for a cell.

    ``seed=None`` averages every cached seed (the ensemble val prediction);
    otherwise the single named seed's raw prediction is used (``gate_on_raw_seed``).
    """
    if seed is not None:
        f = pred_dir / f"{config_hash}__seed{seed}__val__{horizon}d.npy"
        if not f.exists():
            raise ArtifactMissingError(
                f"Missing validation predictions: {f} — re-run the registry "
                f"(cell {CORE_CELL_ID}) so val predictions are cached."
            )
        return np.load(str(f)).astype(np.float64)

    seed_files = sorted(glob.glob(str(pred_dir / f"{config_hash}__seed*__val__{horizon}d.npy")))
    if not seed_files:
        raise ArtifactMissingError(
            f"No cached validation predictions for {config_hash} at {horizon}d in "
            f"{pred_dir} — re-run the registry (cell {CORE_CELL_ID})."
        )
    return np.mean([np.load(s).astype(np.float64) for s in seed_files], axis=0)


def calibrate_from_cache(
    *,
    pred_dir: str | Path,
    gate_dir: str | Path,
    horizon: int,
    coverage: float,
    gate_on_raw_seed: bool,
    seed: int,
    cmtf_version: str,
    backbone_version: str,
    conviction: bool = True,
    adaptive: bool = False,
) -> tuple[GatePolicy, dict[str, Any], Path]:
    """Freeze a universe GatePolicy from the cached validation predictions.

    Reads VALIDATION predictions + truth for the pre-registered CORE cell, and
    writes ``VN_{H}d.json``. TEST predictions are never touched (leak-free, R1).

    ``adaptive=False`` (default): calibrate at a FIXED ``coverage`` (apples-to-
    apples across horizons/models — the right choice for a controlled research
    comparison table, e.g. Phase 2/3's cross-backbone/cross-cell numbers).

    ``adaptive=True``: use :func:`~src.benchmark.decision_policy.calibrate_gate`'s
    coverage-grid search instead — picks whichever coverage on ``DEFAULT_COVERAGE_GRID``
    scores best (by the same DA-aware ``selection_score`` used everywhere else in
    this project) for THIS horizon's own validation predictions, rather than
    forcing every horizon onto the same fixed coverage. A validation-only sweep
    (`logs/coverage_sweep_result.json`, generated while diagnosing why 1D/20D's
    gated validation DA fell *below* their own base rate at the fixed 25%
    coverage) found the fixed-25% convention is not any of the three horizons'
    own validation-optimal operating point — 1D scores ~4x better at 70%
    coverage, 5D and 20D score meaningfully better at ~40%. This is still
    calibrated on VALIDATION alone; it changes *which* coverage is chosen, not
    which split calibrates it.
    """
    from src.benchmark.ablation_registry import get_cell
    from src.benchmark.ablation_runner import _config_hash

    pred_dir = Path(pred_dir)
    cell_id = core_cell_for(horizon)
    config_hash = _config_hash(get_cell(cell_id))

    val_truth_file = pred_dir / f"val_truth__{horizon}d.npy"
    if not val_truth_file.exists():
        raise ArtifactMissingError(
            f"Missing {val_truth_file} — re-run the registry so validation truth is cached."
        )
    val_truth = np.load(str(val_truth_file)).astype(np.float64)

    use_seed = seed if gate_on_raw_seed else None
    val_pred = _load_val_predictions(pred_dir, config_hash, horizon, use_seed)

    if val_pred.shape != val_truth.shape:
        raise ValueError(
            f"val_pred shape {val_pred.shape} != val_truth shape {val_truth.shape} "
            f"for horizon={horizon} — cache is inconsistent, re-run the registry."
        )

    if adaptive:
        policy = calibrate_gate(val_pred, val_truth, conviction=conviction)
        coverage = policy.coverage  # the meta block below records what was actually chosen
    else:
        policy = calibrate_gate_fixed_coverage(
            val_pred, val_truth, coverage=coverage, conviction=conviction
        )

    # Honest, per-horizon operating-point disclosure numbers (plan §0/§3.10): computed
    # HERE, from this horizon's own validation predictions at the calibrated tau/
    # coverage — never a hardcoded literal copied from a different horizon's one-off
    # measurement. Same epistemic status as any other calibration-set metric: it is
    # the accuracy/base-rate AT the tau chosen on this same validation set, not a
    # held-out claim. Threaded through gate_agent_node -> state -> narrator/critic so
    # every horizon discloses its own real numbers instead of a fixed 5D string.
    gated = evaluate_policy(val_truth, val_pred, policy, horizon=horizon)
    mask = np.abs(val_pred) >= policy.tau
    base_rate = (
        compute_all(val_truth[mask], val_pred[mask], horizon=horizon)["base_rate_DA%"]
        if mask.sum() >= 3 else float("nan")
    )

    def _finite_or_none(x: float) -> float | None:
        return round(float(x), 4) if np.isfinite(x) else None

    meta = {
        "symbol": "VN",
        "horizon": int(horizon),
        "cell_id": cell_id,
        "config_hash": config_hash,
        "cmtf_version": cmtf_version,
        "backbone_version": backbone_version,
        "calibrated_on": "validation",
        "calibration_coverage": float(coverage),
        "gate_on_raw_seed": bool(gate_on_raw_seed),
        "calibration_seed": int(seed) if gate_on_raw_seed else None,
        "n_val": int(val_pred.size),
        "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_gated_DA%": _finite_or_none(gated["DA%"]),
        "val_gated_coverage": _finite_or_none(gated["coverage"]),
        "val_base_rate_DA%": _finite_or_none(base_rate),
        "val_gated_Sharpe": _finite_or_none(gated["Sharpe"]),
        "val_gated_IC": _finite_or_none(gated["IC"]),
    }

    out_path = policy_path(gate_dir, horizon, symbol="VN")
    save_gate_policy(policy, meta, out_path)
    return policy, meta, out_path
