"""Selective-prediction calibration metrics (plan §10.3, H2).

The flagship calibration claim: the confidence gate yields *honest selective
prediction* — trading only confident names lowers directional risk. We measure this
with the risk-coverage curve and its area (AURC): sort samples by confidence, and at
each coverage level report the directional error on the covered (most-confident)
subset. A model whose confidence ranks skill well has risk that falls as coverage
tightens, hence a low AURC. Significance is a paired bootstrap over the test book on
ΔAURC (MAS gate confidence vs a comparator's confidence on the SAME samples).

Everything here is numpy-only, deterministic, and a pure function of
(prediction, truth, confidence) — no LLM, no retraining.
"""

from __future__ import annotations

import numpy as np


def _directional_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-sample directional error (1 = wrong sign, 0 = right sign)."""
    return (np.sign(pred) != np.sign(truth)).astype(np.float64)


def risk_coverage_curve(
    pred: np.ndarray,
    truth: np.ndarray,
    confidence: np.ndarray,
    min_coverage: float = 0.05,
):
    """Return (coverages, risks) sweeping from most-confident to full book.

    At coverage c we keep the ⌈c·n⌉ most-confident samples and report their mean
    directional error. Ties in confidence are broken deterministically by index.
    """
    pred = np.asarray(pred, np.float64).ravel()
    truth = np.asarray(truth, np.float64).ravel()
    confidence = np.asarray(confidence, np.float64).ravel()
    n = len(pred)
    order = np.argsort(-confidence, kind="stable")  # high confidence first
    err = _directional_error(pred, truth)[order]
    covs, risks = [], []
    start = max(1, int(np.ceil(min_coverage * n)))
    for k in range(start, n + 1):
        covs.append(k / n)
        risks.append(float(err[:k].mean()))
    return np.asarray(covs), np.asarray(risks)


def aurc(pred: np.ndarray, truth: np.ndarray, confidence: np.ndarray,
         min_coverage: float = 0.05) -> float:
    """Area under the risk-coverage curve (lower = better selective prediction)."""
    covs, risks = risk_coverage_curve(pred, truth, confidence, min_coverage)
    if len(covs) < 2:
        return float("nan")
    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy>=2 renamed trapz→trapezoid
    return float(_trapz(risks, covs) / (covs[-1] - covs[0]))


def paired_bootstrap_aurc(
    pred_a: np.ndarray, conf_a: np.ndarray,
    pred_b: np.ndarray, conf_b: np.ndarray,
    truth: np.ndarray,
    n_boot: int = 5000, seed: int = 0, min_coverage: float = 0.05,
) -> dict:
    """95% CI on Δ AURC = A − B, resampling the test book (paired).

    Negative Δ means A has lower risk-coverage area (better calibration) than B.
    """
    truth = np.asarray(truth, np.float64).ravel()
    n = len(truth)
    point = aurc(pred_a, truth, conf_a, min_coverage) - aurc(pred_b, truth, conf_b, min_coverage)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (aurc(pred_a[idx], truth[idx], conf_a[idx], min_coverage)
                     - aurc(pred_b[idx], truth[idx], conf_b[idx], min_coverage))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_aurc": point, "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo * hi > 0), "n": n}


def selective_da_at_coverage(pred, truth, confidence, coverage: float) -> dict:
    """Directional accuracy on the top-`coverage` most-confident subset."""
    pred = np.asarray(pred, np.float64).ravel()
    truth = np.asarray(truth, np.float64).ravel()
    confidence = np.asarray(confidence, np.float64).ravel()
    n = len(pred)
    k = max(1, int(np.ceil(coverage * n)))
    order = np.argsort(-confidence, kind="stable")[:k]
    da = float((np.sign(pred[order]) == np.sign(truth[order])).mean()) * 100
    return {"coverage": k / n, "n": k, "DA%": da}
