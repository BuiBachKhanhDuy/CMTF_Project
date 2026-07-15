"""Frozen GatePolicy artifact I/O + offline calibration (plan §4).

The deployed decision layer is a single validation-calibrated
:class:`~src.benchmark.decision_policy.GatePolicy`. This module serialises it to
JSON with full provenance (version stamps, calibration coverage, source config
hash + seed), reloads it with a hard staleness check, and calibrates it from the
frozen validation predictions the ablation registry writes to
``cache/predictions`` — so the deployed gate is calibrated on the *exact same*
frozen predictions the registry gated in-memory (runtime == research).

R1 rules enforced here:
- A missing artifact raises :class:`ArtifactMissingError` (no ad-hoc tau).
- A version-stamp mismatch raises :class:`StalePolicyError` (no stale fallback).
- Calibration reads VALIDATION predictions only; TEST never enters (leak-free).

The CMTF champion is a single POOLED model over the whole symbol universe, so the
honest artifact is one universe policy per horizon (``VN_{H}d.json``), consumed by
both the single-name ``gate_agent`` and the ``rank_agent``. We do not fabricate
per-symbol policies from ~40 thin validation samples each.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.decision_policy import GatePolicy, calibrate_gate_fixed_coverage

from .loaders import ArtifactMissingError

# Bumped whenever the artifact schema or the calibration procedure changes, so an
# old on-disk policy can never be silently loaded against new calibration logic.
GATE_ARTIFACT_SCHEMA_VERSION = 1

# The pre-registered champion (plan §0): cell 0 = CMTF_CORE. The gate is
# calibrated on this cell's validation predictions, never a point-estimate winner.
CORE_CELL_ID = "0"


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
) -> tuple[GatePolicy, dict[str, Any], Path]:
    """Freeze a universe GatePolicy from the cached validation predictions.

    Reads VALIDATION predictions + truth for the pre-registered CORE cell,
    calibrates at a FIXED coverage (apples-to-apples, matching the registry), and
    writes ``VN_{H}d.json``. TEST predictions are never touched (leak-free, R1).
    """
    from src.benchmark.ablation_registry import get_cell
    from src.benchmark.ablation_runner import _config_hash

    pred_dir = Path(pred_dir)
    config_hash = _config_hash(get_cell(CORE_CELL_ID))

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

    policy = calibrate_gate_fixed_coverage(
        val_pred, val_truth, coverage=coverage, conviction=conviction
    )

    meta = {
        "symbol": "VN",
        "horizon": int(horizon),
        "cell_id": CORE_CELL_ID,
        "config_hash": config_hash,
        "cmtf_version": cmtf_version,
        "backbone_version": backbone_version,
        "calibrated_on": "validation",
        "calibration_coverage": float(coverage),
        "gate_on_raw_seed": bool(gate_on_raw_seed),
        "calibration_seed": int(seed) if gate_on_raw_seed else None,
        "n_val": int(val_pred.size),
        "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = policy_path(gate_dir, horizon, symbol="VN")
    save_gate_policy(policy, meta, out_path)
    return policy, meta, out_path
