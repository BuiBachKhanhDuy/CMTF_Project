"""Numpy-only validation-selection utilities for fusion strategies.

The selection score combines rank information with directional performance for
use by fusion training and decision-gate calibration.
"""

from __future__ import annotations

import numpy as np

# Weights for the directional validation objective.
DEFAULT_W_IC: float = 0.25
DEFAULT_W_DA: float = 2.0
DEFAULT_W_SHARPE: float = 1.0


def _direction_threshold(values: np.ndarray) -> float:
    """Half of the 20th-percentile absolute magnitude (matches metrics.py)."""
    abs_v = np.abs(np.asarray(values, dtype=np.float64).ravel())
    if abs_v.size == 0:
        return 0.0
    return max(0.5 * float(np.percentile(abs_v, 20)), 1e-6)


def rank_ic(pred, target) -> float:
    """Spearman rank correlation (numpy-only). 0.0 for degenerate inputs."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.size < 3 or np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0
    pr = np.argsort(np.argsort(pred)).astype(np.float64)
    tr = np.argsort(np.argsort(target)).astype(np.float64)
    pr -= pr.mean()
    tr -= tr.mean()
    denom = np.sqrt((pr ** 2).sum() * (tr ** 2).sum())
    return float((pr * tr).sum() / denom) if denom > 0 else 0.0


def _signed(values: np.ndarray, thr: float) -> np.ndarray:
    lab = np.zeros_like(values, dtype=np.int8)
    lab[values > thr] = 1
    lab[values < -thr] = -1
    return lab


def da_fraction(pred, target) -> float:
    """Directional accuracy (fraction in [0,1]) on the active-target subset.

    Uses per-series adaptive dead-zone thresholds, mirroring metrics.py, so a
    low-magnitude prediction is not unfairly zeroed by the target's scale.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.size == 0:
        return 0.0
    t_sign = _signed(target, _direction_threshold(target))
    p_sign = _signed(pred, _direction_threshold(pred))
    active = t_sign != 0
    if not np.any(active):
        return 0.0
    return float(np.mean(t_sign[active] == p_sign[active]))


def sharpe_proxy(pred, target) -> float:
    """Sign-based Sharpe proxy (per-step, un-annualised) on all samples.

    pnl_t = sign(pred_t) * target_t;  returns mean(pnl) / std(pnl). Un-annualised
    (no sqrt(252/h) factor) because it is only used for *relative* comparison
    between lambda candidates on the same validation set. Depends only on the
    prediction sign, so it directly tracks the reported Sharpe metric.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.size == 0:
        return 0.0
    pnl = np.sign(pred) * target
    sd = float(np.std(pnl))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(pnl) / sd)


def base_rate_fraction(target) -> float:
    """Majority-class fraction on the active-target subset (in [0,1])."""
    target = np.asarray(target, dtype=np.float64).ravel()
    t_sign = _signed(target, _direction_threshold(target))
    active = t_sign != 0
    if not np.any(active):
        return 0.5
    pos = float(np.mean(t_sign[active] == 1))
    return max(pos, 1.0 - pos)


def selection_score(
    pred,
    target,
    w_ic: float = DEFAULT_W_IC,
    w_da: float = DEFAULT_W_DA,
    w_sharpe: float = DEFAULT_W_SHARPE,
) -> float:
    """DA/Sharpe-first blended validation objective (higher is better).

    score = w_da * (DA_fraction - base_rate_fraction)
          + w_sharpe * sharpe_proxy
          + w_ic * rank_IC

    The DA term is DA-*skill* (over the majority-class base rate) so trivially
    predicting the majority direction earns nothing. Sharpe and DA dominate;
    IC is a low-weight tiebreaker so it cannot outweigh a DA/Sharpe regression.
    """
    ic = rank_ic(pred, target)
    da_skill = da_fraction(pred, target) - base_rate_fraction(target)
    shp = sharpe_proxy(pred, target)
    return w_da * da_skill + w_sharpe * shp + w_ic * ic
