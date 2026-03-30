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
    """Percentage of correct sign (up/down) predictions."""
    if len(y_true) == 0:
        return 0.0
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    return float(correct / len(y_true) * 100)


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Annualised Sharpe ratio of a strategy that goes long/short based on pred sign.

    Strategy return on day t = sign(y_pred[t]) * y_true[t].
    """
    strategy_returns = np.sign(y_pred) * y_true
    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return 0.0
    return float((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual returns."""
    if len(y_true) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else 0.0


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA%": directional_accuracy(y_true, y_pred),
        "Sharpe": sharpe_ratio(y_true, y_pred),
        "IC": information_coefficient(y_true, y_pred),
    }
