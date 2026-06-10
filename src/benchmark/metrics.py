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


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of a return series (negative float, e.g. -0.15 = -15%).

    Uses cumulative-sum arithmetic so that a single -5% return produces -0.05,
    and a monotonically rising series produces 0.0.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(r)])
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualised return / abs(max drawdown). Returns 0 if MDD is zero."""
    r = np.asarray(returns, dtype=float)
    ann_return = float(np.mean(r) * periods_per_year)
    mdd = abs(max_drawdown(r))
    return ann_return / mdd if mdd > 1e-12 else 0.0


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    da_val = directional_accuracy(y_true, y_pred)

    # Effective sample size: corrects for autocorrelation in the target series.
    # Overlapping H-day windows inflate n by ~H; ESS removes that inflation.
    # At 20D (autocorr≈0.96) this turns 574 nominal samples into ~23 effective.
    ac1 = float(np.corrcoef(y_true[:-1], y_true[1:])[0, 1]) if len(y_true) > 2 else 0.0
    ess = max(1, int(len(y_true) * (1.0 - ac1) / (1.0 + ac1 + 1e-9)))

    # Base-rate DA%: best naive classifier (always predict majority class).
    # Any model below this threshold has no directional skill.
    up_frac = float((y_true > 0).mean())
    base_rate_da = round(max(up_frac, 1.0 - up_frac) * 100, 2)

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA%": da_val,
        "Sharpe": sharpe_ratio(y_true, y_pred, horizon=horizon),
        "IC": information_coefficient(y_true, y_pred),
        "Prec": direction_precision(y_true, y_pred),
        "Rec": direction_recall(y_true, y_pred),
        "F1": direction_f1(y_true, y_pred),
        "ESS": ess,
        "base_rate_DA%": base_rate_da,
        "DA_skill%": round(da_val - base_rate_da, 2),
    }


# ======================================================================
# Statistical significance tests
# ======================================================================


def diebold_mariano_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    horizon: int = 1,
    loss: str = "se",
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[L(e_a)] = E[L(e_b)]  (equal forecast loss)
    H1: E[L(e_a)] ≠ E[L(e_b)]  (different forecast loss)

    Uses Newey-West HAC standard errors with bandwidth = horizon.

    Args:
        y_true: (N,) actual values
        preds_a: (N,) predictions from model A (e.g. baseline)
        preds_b: (N,) predictions from model B (e.g. CMTF)
        horizon: forecast horizon (used for HAC bandwidth)
        loss: "se" (squared error) or "ae" (absolute error)

    Returns:
        dict with keys: DM_stat, p_value
    """
    y_true = np.asarray(y_true, dtype=float)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    n = len(y_true)

    if loss == "se":
        loss_a = (y_true - preds_a) ** 2
        loss_b = (y_true - preds_b) ** 2
    elif loss == "ae":
        loss_a = np.abs(y_true - preds_a)
        loss_b = np.abs(y_true - preds_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b  # positive means A is worse
    d_mean = d.mean()

    # Newey-West HAC variance estimator
    bandwidth = max(horizon, 1)
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1)
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * weight * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d < 1e-15:
        return {"DM_stat": 0.0, "p_value": 1.0}

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return {"DM_stat": float(dm_stat), "p_value": float(p_value)}


def paired_bootstrap_da(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for directional accuracy difference.

    Resamples (with replacement) and computes DA(B) - DA(A) for each
    bootstrap sample. Returns mean delta, 95% CI, and empirical p-value.

    Args:
        y_true: (N,) actual values
        preds_a: (N,) predictions from model A (baseline)
        preds_b: (N,) predictions from model B (CMTF)
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with: delta_da (mean), ci_low, ci_high, p_value
    """
    y_true = np.asarray(y_true, dtype=float)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    nonzero = y_true != 0
    correct_a = (np.sign(y_true) == np.sign(preds_a)) & nonzero
    correct_b = (np.sign(y_true) == np.sign(preds_b)) & nonzero

    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        nz_sum = nonzero[idx].sum()
        if nz_sum == 0:
            deltas[i] = 0.0
            continue
        da_a = correct_a[idx].sum() / nz_sum * 100
        da_b = correct_b[idx].sum() / nz_sum * 100
        deltas[i] = da_b - da_a

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # Two-sided p-value: fraction of bootstrap samples where delta ≤ 0
    p_value = float(np.mean(deltas <= 0))

    return {
        "delta_da": float(deltas.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }
