"""Evaluation metrics for forecasting benchmarks."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct sign (up/down) predictions.

    Samples where y_true == 0 (no directional move) are excluded.
    """
    if len(y_true) == 0:
        return 0.0
    nonzero = y_true != 0
    if not nonzero.any():
        return 0.0
    correct = np.sum(np.sign(y_true[nonzero]) == np.sign(y_pred[nonzero]))
    return float(correct / nonzero.sum() * 100)


def sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> float:
    """Annualised Sharpe ratio of a strategy that goes long/short based on pred sign.

    Strategy return on bar t = sign(y_pred[t]) * y_true[t].

    For H-day forward returns the daily-sampled strategy returns are
    overlapping (consecutive samples share H-1 bars).  To remove the
    resulting autocorrelation bias we sub-sample at stride H.  To avoid
    dependence on the arbitrary starting offset we average across all H
    possible phase offsets.  The annualisation factor is sqrt(252 / H)
    (one independent observation every H trading days).
    """
    strategy_returns = np.sign(y_pred) * y_true

    if horizon <= 1:
        if len(strategy_returns) < 5 or strategy_returns.std() == 0:
            return float('nan')
        ann_factor = np.sqrt(252.0)
        return float((strategy_returns.mean() / strategy_returns.std()) * ann_factor)

    # Phase-averaged Sharpe for H > 1
    ann_factor = np.sqrt(252.0 / horizon)
    phase_sharpes: list[float] = []
    for offset in range(horizon):
        phase = strategy_returns[offset::horizon]
        if len(phase) < 5 or phase.std() == 0:
            continue
        phase_sharpes.append(
            float((phase.mean() / phase.std()) * ann_factor)
        )
    if not phase_sharpes:
        return float('nan')
    return float(np.mean(phase_sharpes))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual returns."""
    if len(y_true) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else 0.0


def direction_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision for 'up' direction: TP / (TP + FP)."""
    pred_up = y_pred > 0
    true_up = y_true > 0
    tp = np.sum(pred_up & true_up)
    fp = np.sum(pred_up & ~true_up)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def direction_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for 'up' direction: TP / (TP + FN)."""
    pred_up = y_pred > 0
    true_up = y_true > 0
    tp = np.sum(pred_up & true_up)
    fn = np.sum(~pred_up & true_up)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def direction_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for 'up' direction prediction."""
    p = direction_precision(y_true, y_pred)
    r = direction_recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA%": directional_accuracy(y_true, y_pred),
        "Sharpe": sharpe_ratio(y_true, y_pred, horizon=horizon),
        "IC": information_coefficient(y_true, y_pred),
        "Prec": direction_precision(y_true, y_pred),
        "Rec": direction_recall(y_true, y_pred),
        "F1": direction_f1(y_true, y_pred),
    }
