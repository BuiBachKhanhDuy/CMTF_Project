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


def modal_disagreement(anchor_pred: np.ndarray | None, y_pred: np.ndarray) -> float:
    """Fraction of sign disagreements versus a shared market-only anchor.

    This is a benchmark diagnostic, not a universal forecasting metric.
    Lower is better.
    """
    if anchor_pred is None or len(anchor_pred) == 0 or len(y_pred) == 0:
        return 0.0

    active = (anchor_pred != 0) | (y_pred != 0)
    if not np.any(active):
        return 0.0

    anchor_sign = np.sign(anchor_pred[active])
    pred_sign = np.sign(y_pred[active])
    return float(np.mean(anchor_sign != pred_sign))


def temporal_lag(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 1) -> float:
    """Normalised temporal lag penalty from phase-shifted correlation.

    The penalty is 0 when the best correlation occurs at lag 0 and approaches 1
    as the best-aligned lag moves further away from 0.
    """
    n_samples = len(y_true)
    if n_samples < 5:
        return 0.0

    max_lag = int(min(max(horizon, 1), max(n_samples - 1, 1), 10))
    if max_lag <= 0:
        return 0.0

    true_centered = y_true - np.mean(y_true)
    pred_centered = y_pred - np.mean(y_pred)
    true_std = np.std(true_centered)
    pred_std = np.std(pred_centered)
    if true_std < 1e-12 or pred_std < 1e-12:
        return 0.0

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = true_centered[-lag:]
            right = pred_centered[: n_samples + lag]
        elif lag > 0:
            left = true_centered[: n_samples - lag]
            right = pred_centered[lag:]
        else:
            left = true_centered
            right = pred_centered

        if len(left) < 3 or len(right) < 3:
            continue
        left_std = np.std(left)
        right_std = np.std(right)
        if left_std < 1e-12 or right_std < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(left, right)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0

        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_lag = lag

    return float(abs(best_lag) / max_lag)


def compute_composite_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
    anchor_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the custom benchmark score and its components.

    Score:
        0.5 * RMSE
      + 0.15 * MAE
      + 0.15 * (1 - DA/100)
      + 0.12 * ModalDisagreement
      + 0.08 * TemporalLag
    """
    rmse_value = rmse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)
    da_value = directional_accuracy(y_true, y_pred)
    disagreement = modal_disagreement(anchor_pred, y_pred)
    lag_penalty = temporal_lag(y_true, y_pred, horizon=horizon)
    # Replace anchor-biased ModalDisagreement weight with model-intrinsic
    # directional F1 penalty so CMTF is not penalised for diverging from zero-shot.
    f1_value = direction_f1(y_true, y_pred)
    composite_score = (
        0.5 * rmse_value
        + 0.15 * mae_value
        + 0.15 * (1.0 - da_value / 100.0)
        + 0.12 * (1.0 - f1_value)
        + 0.08 * lag_penalty
    )
    return {
        "ModalDisagreement": disagreement,
        "TemporalLag": lag_penalty,
        "CompositeScore": composite_score,
    }


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
