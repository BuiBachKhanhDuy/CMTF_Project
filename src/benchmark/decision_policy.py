"""decision_policy.py

Validation-calibrated CONFIDENCE-GATE + CONVICTION-SIZING decision layer.

Why this exists
---------------
CMTF is trained/deployed as a point regressor, but the priority metrics (DA,
Sharpe) are sign/decision-layer quantities. The placebo-controlled improvement
study (RESULTS_IMPROVEMENT_LEVERS.md) showed that CMTF's *news-driven* directional
signal is concentrated in its high-|pred| (high-confidence) tail: the bottom
confidence deciles flip away from the market anchor and are pure noise, while the
top decile reaches DA ~= 66% / IC ~= +0.25. Trading the full book drowns the
signal; trading only the confident subset recovers it, and the gain transfers
out of sample for REAL news but not for a shuffled-news placebo.

This module operationalises that finding as a deployable, leak-free policy:

    1. CONFIDENCE GATE  (Lever 1): only act on predictions whose |pred| clears a
       threshold ``tau``. ``tau`` is calibrated on the VALIDATION split (never the
       test set) by maximising the DA-aware ``selection_score`` over a grid of
       coverage levels, subject to a coverage floor.
    2. CONVICTION SIZING (Lever 2): size each position by |pred| (normalised on
       the traded subset) instead of a flat unit position — |pred| is a genuine
       confidence signal only when news is real, so this lifts Sharpe.

Naive seed-ensembling (Lever 4) is deliberately NOT included: averaging shrinks
predictions toward zero and erodes the very tail conviction this policy exploits.

Everything here is numpy-only and a pure function of predictions + targets, so it
adds no training cost and is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion_selection import selection_score


# Coverage grid = fraction of the book we are willing to trade. 1.0 (full book)
# is always evaluated so the gate can decline to gate when that is best on
# validation (the policy is strictly opt-in: coverage=1.0 recovers the
# trade-everything baseline exactly when no gate helps).
DEFAULT_COVERAGE_GRID: tuple[float, ...] = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15)

# Never calibrate to a book so thin the validation estimate is meaningless.
DEFAULT_MIN_COVERAGE: float = 0.15


@dataclass(frozen=True)
class GatePolicy:
    """A calibrated confidence-gate + conviction-sizing decision rule.

    Attributes
    ----------
    tau
        Absolute-magnitude threshold on |pred|. Predictions with |pred| < tau are
        NOT traded (flat / no position).
    conviction
        If True, size traded positions by |pred| normalised on the traded subset;
        else use a flat unit position (sign only).
    conviction_scale
        Normaliser for conviction sizing (median |pred| of the traded VAL subset).
        Frozen at calibration time so test-time sizing uses no test statistics.
    coverage
        Fraction of the VALIDATION book that cleared ``tau`` at calibration.
    val_score
        The DA-aware ``selection_score`` achieved on the traded VAL subset.
    """

    tau: float
    conviction: bool
    conviction_scale: float
    coverage: float
    val_score: float


def _traded_mask(pred: np.ndarray, tau: float) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64).ravel()) >= tau


def calibrate_gate(
    val_pred: np.ndarray,
    val_truth: np.ndarray,
    conviction: bool = True,
    coverage_grid: tuple[float, ...] = DEFAULT_COVERAGE_GRID,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> GatePolicy:
    """Calibrate a :class:`GatePolicy` on the VALIDATION split.

    For each target coverage ``c`` we set ``tau`` to the ``(1-c)`` quantile of
    |val_pred| (so ~c of the book is traded), then score the traded subset with
    the DA-aware ``selection_score``. The coverage with the best score (subject to
    the coverage floor) wins. ``tau`` and the conviction normaliser are frozen for
    deployment — the test set never enters calibration.
    """
    val_pred = np.asarray(val_pred, dtype=np.float64).ravel()
    val_truth = np.asarray(val_truth, dtype=np.float64).ravel()
    abs_pred = np.abs(val_pred)

    best: GatePolicy | None = None
    for cov in coverage_grid:
        if cov <= 0.0:
            continue
        if cov >= 1.0:
            tau = 0.0
        else:
            tau = float(np.quantile(abs_pred, 1.0 - cov))
        mask = _traded_mask(val_pred, tau)
        realised_cov = float(np.mean(mask))
        if realised_cov < min_coverage or mask.sum() < 20:
            continue
        score = selection_score(val_pred[mask], val_truth[mask])
        conv_scale = float(np.median(abs_pred[mask])) or 1.0
        cand = GatePolicy(
            tau=tau,
            conviction=conviction,
            conviction_scale=conv_scale,
            coverage=realised_cov,
            val_score=float(score),
        )
        if best is None or cand.val_score > best.val_score:
            best = cand

    if best is None:
        # Degenerate fallback: trade the full book, flat sizing.
        return GatePolicy(
            tau=0.0,
            conviction=False,
            conviction_scale=1.0,
            coverage=1.0,
            val_score=float(selection_score(val_pred, val_truth)),
        )
    return best


def calibrate_gate_fixed_coverage(
    val_pred: np.ndarray,
    val_truth: np.ndarray,
    coverage: float = 0.25,
    conviction: bool = True,
) -> GatePolicy:
    """Calibrate a :class:`GatePolicy` at a FIXED target coverage (apples-to-apples).

    ``calibrate_gate`` searches a coverage grid per-model and keeps whichever
    coverage scores best on validation. That is the right *deployment* policy
    for a single model, but it is NOT a fair basis for cross-model comparison:
    two models end up trading different fractions of the book (e.g. one at 25%,
    another at 60%), so their gated DA/Sharpe/IC are different operating points,
    not the same policy. This function fixes every model to the SAME coverage
    (e.g. "trade the top 25% most confident predictions"), so the comparison
    isolates the *quality* of each model's confidence ranking rather than
    rewarding/punishing it for its raw output scale or how aggressively the
    auto-search happened to gate it.

    ``tau`` is still frozen on VALIDATION only (the ``(1-coverage)`` quantile of
    ``|val_pred|``) — the test set never enters calibration.
    """
    val_pred = np.asarray(val_pred, dtype=np.float64).ravel()
    val_truth = np.asarray(val_truth, dtype=np.float64).ravel()
    abs_pred = np.abs(val_pred)

    coverage = float(np.clip(coverage, 1e-6, 1.0))
    tau = 0.0 if coverage >= 1.0 else float(np.quantile(abs_pred, 1.0 - coverage))
    mask = _traded_mask(val_pred, tau)
    realised_cov = float(np.mean(mask)) if mask.size else 0.0
    conv_scale = float(np.median(abs_pred[mask])) if mask.any() else 1.0
    conv_scale = conv_scale or 1.0
    score = float(selection_score(val_pred[mask], val_truth[mask])) if mask.sum() >= 3 else float("nan")

    return GatePolicy(
        tau=tau,
        conviction=conviction,
        conviction_scale=conv_scale,
        coverage=realised_cov,
        val_score=score,
    )


def apply_positions(pred: np.ndarray, policy: GatePolicy) -> np.ndarray:
    """Map raw predictions to signed positions under ``policy``.

    Returns a per-sample position in roughly [-1, 1] (conviction) or {-1, 0, +1}
    (flat). Non-traded samples (|pred| < tau) get position 0.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mask = _traded_mask(pred, policy.tau)
    pos = np.zeros_like(pred)
    if not np.any(mask):
        return pos
    if policy.conviction:
        scale = policy.conviction_scale if policy.conviction_scale > 1e-12 else 1.0
        # Size by |pred| normalised on the calibration median, clipped so a few
        # huge-magnitude names cannot dominate the book.
        sized = np.clip(pred[mask] / scale, -3.0, 3.0)
        pos[mask] = sized
    else:
        pos[mask] = np.sign(pred[mask])
    return pos


def _gated_sharpe(positions: np.ndarray, y_true: np.ndarray, horizon: int) -> float:
    """Annualised Sharpe of the position-weighted book (0 = no trade)."""
    positions = np.asarray(positions, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pnl = positions * y_true
    if horizon <= 1:
        if pnl.size < 5 or np.std(pnl) == 0:
            return float("nan")
        return float(np.mean(pnl) / np.std(pnl) * np.sqrt(252.0))
    ann = np.sqrt(252.0 / horizon)
    phase = []
    for off in range(horizon):
        seg = pnl[off::horizon]
        if len(seg) < 5 or np.std(seg) == 0:
            continue
        phase.append(float(np.mean(seg) / np.std(seg) * ann))
    return float(np.mean(phase)) if phase else float("nan")


def evaluate_policy(
    y_true: np.ndarray,
    pred: np.ndarray,
    policy: GatePolicy,
    horizon: int = 1,
) -> dict[str, float]:
    """Evaluate DA / Sharpe / IC on the TRADED subset under ``policy``.

    DA and IC are computed on the confident (traded) names only — the names the
    policy actually acts on — while Sharpe uses the (optionally conviction-sized)
    position-weighted PnL so both levers are reflected. ``coverage`` reports the
    fraction of the test book that was traded.
    """
    from .metrics import directional_accuracy, information_coefficient

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mask = _traded_mask(pred, policy.tau)
    coverage = float(np.mean(mask)) if mask.size else 0.0

    positions = apply_positions(pred, policy)

    if mask.sum() >= 3:
        da = directional_accuracy(y_true[mask], pred[mask])
        ic = information_coefficient(y_true[mask], pred[mask])
    else:
        da = float("nan")
        ic = float("nan")

    sharpe = _gated_sharpe(positions, y_true, horizon)

    return {
        "DA%": da,
        "Sharpe": sharpe,
        "IC": ic,
        "coverage": coverage,
        "n_traded": int(mask.sum()),
        "tau": policy.tau,
        "conviction": policy.conviction,
    }
