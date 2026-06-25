"""Evaluation metrics for forecasting benchmarks.

Refactored for financial forecasting:
1. Direction metrics use a configurable dead-zone threshold instead of exact zero.
2. Precision/Recall/F1 are symmetric over {-1, +1}, not biased to only 'up'.
3. Base-rate DA is computed on the same active directional subset.
4. Composite metrics use the corrected directional F1.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ======================================================================
# Basic regression metrics
# ======================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ======================================================================
# Direction helpers
# ======================================================================

def _direction_threshold(y_true: np.ndarray, eps: float | None = None) -> float:
    """Choose a robust dead-zone threshold for directional metrics.

    If eps is provided, use it directly.
    Otherwise use a small data-adaptive threshold based on target scale.
    """
    y_true = np.asarray(y_true, dtype=float)
    if eps is not None:
        return float(max(eps, 0.0))

    abs_y = np.abs(y_true)
    if abs_y.size == 0:
        return 0.0

    # Robust small-move threshold:
    # half of the 20th percentile magnitude, bounded below.
    q20 = float(np.percentile(abs_y, 20))
    return max(0.5 * q20, 1e-6)


def _signed_labels(values: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Map values to {-1, 0, +1} using a dead-zone threshold."""
    values = np.asarray(values, dtype=float)
    thr = _direction_threshold(values, eps=eps)

    labels = np.zeros_like(values, dtype=np.int8)
    labels[values > thr] = 1
    labels[values < -thr] = -1
    return labels


def _active_direction_mask(y_true: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Samples where true direction is meaningfully non-neutral."""
    true_sign = _signed_labels(y_true, eps=eps)
    return true_sign != 0


# ======================================================================
# Directional metrics
# ======================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Percentage of correct sign predictions on meaningful moves only.

    Uses a dead-zone threshold so tiny/noisy moves are ignored.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    true_sign = _signed_labels(y_true, eps=eps)
    pred_sign = _signed_labels(y_pred, eps=eps)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    correct = np.sum(true_sign[active] == pred_sign[active])
    return float(correct / np.sum(active) * 100.0)


def direction_precision(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Macro precision over {-1, +1} on meaningful true-direction samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_sign = _signed_labels(y_true, eps=eps)
    pred_sign = _signed_labels(y_pred, eps=eps)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    true_active = true_sign[active]
    pred_active = pred_sign[active]

    precisions = []
    for cls in (-1, 1):
        pred_cls = pred_active == cls
        tp = np.sum(pred_cls & (true_active == cls))
        fp = np.sum(pred_cls & (true_active != cls))
        denom = tp + fp
        if denom > 0:
            precisions.append(tp / denom)

    return float(np.mean(precisions)) if precisions else 0.0


def direction_recall(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Macro recall over {-1, +1} on meaningful true-direction samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_sign = _signed_labels(y_true, eps=eps)
    pred_sign = _signed_labels(y_pred, eps=eps)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    true_active = true_sign[active]
    pred_active = pred_sign[active]

    recalls = []
    for cls in (-1, 1):
        true_cls = true_active == cls
        tp = np.sum(true_cls & (pred_active == cls))
        fn = np.sum(true_cls & (pred_active != cls))
        denom = tp + fn
        if denom > 0:
            recalls.append(tp / denom)

    return float(np.mean(recalls)) if recalls else 0.0


def direction_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Macro F1 over {-1, +1} on meaningful true-direction samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_sign = _signed_labels(y_true, eps=eps)
    pred_sign = _signed_labels(y_pred, eps=eps)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    true_active = true_sign[active]
    pred_active = pred_sign[active]

    f1s = []
    for cls in (-1, 1):
        tp = np.sum((true_active == cls) & (pred_active == cls))
        fp = np.sum((true_active != cls) & (pred_active == cls))
        fn = np.sum((true_active == cls) & (pred_active != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1s.append(2.0 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)

    return float(np.mean(f1s))


# ======================================================================
# Trading / rank metrics
# ======================================================================

def sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> float:
    """Annualised Sharpe ratio of a sign-based long/short strategy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    strategy_returns = np.sign(y_pred) * y_true

    if horizon <= 1:
        if len(strategy_returns) < 5 or np.std(strategy_returns) == 0:
            return float("nan")
        ann_factor = np.sqrt(252.0)
        return float((np.mean(strategy_returns) / np.std(strategy_returns)) * ann_factor)

    ann_factor = np.sqrt(252.0 / horizon)
    phase_sharpes: list[float] = []
    for offset in range(horizon):
        phase = strategy_returns[offset::horizon]
        if len(phase) < 5 or np.std(phase) == 0:
            continue
        phase_sharpes.append(float((np.mean(phase) / np.std(phase)) * ann_factor))

    if not phase_sharpes:
        return float("nan")
    return float(np.mean(phase_sharpes))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual returns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else 0.0


# ======================================================================
# Diagnostics
# ======================================================================

def modal_disagreement(anchor_pred: np.ndarray | None, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Fraction of sign disagreements versus a shared market-only anchor."""
    if anchor_pred is None:
        return 0.0

    anchor_pred = np.asarray(anchor_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(anchor_pred) == 0 or len(y_pred) == 0:
        return 0.0

    anchor_sign = _signed_labels(anchor_pred, eps=eps)
    pred_sign = _signed_labels(y_pred, eps=eps)

    active = (anchor_sign != 0) | (pred_sign != 0)
    if not np.any(active):
        return 0.0

    return float(np.mean(anchor_sign[active] != pred_sign[active]))


def temporal_lag(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 1) -> float:
    """Normalised temporal lag penalty from phase-shifted correlation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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
    """Compute diagnostic composite metrics.

    IMPORTANT:
    CompositeScore is diagnostic only and should not be used as the sole
    model-selection criterion. It is designed to summarize multiple aspects
    of forecast behavior on roughly normalized scales.
    """
    rmse_value = rmse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)
    da_value = directional_accuracy(y_true, y_pred)
    disagreement = modal_disagreement(anchor_pred, y_pred)
    lag_penalty = temporal_lag(y_true, y_pred, horizon=horizon)
    f1_value = direction_f1(y_true, y_pred)
    ic_value = information_coefficient(y_true, y_pred)

    # Scale-normalized error terms
    target_scale = max(float(np.std(np.asarray(y_true, dtype=float))), 1e-8)
    rmse_norm = rmse_value / target_scale
    mae_norm = mae_value / target_scale

    # Convert "higher is better" metrics into penalties
    da_penalty = 1.0 - da_value / 100.0
    f1_penalty = 1.0 - f1_value
    ic_clipped = max(min(ic_value, 1.0), -1.0)
    ic_penalty = (1.0 - ic_clipped) / 2.0  # maps IC from [-1, 1] to penalty [1, 0]

    # Diagnostic composite: lower is better
    composite_score = (
        0.28 * rmse_norm
        + 0.18 * mae_norm
        + 0.16 * da_penalty
        + 0.14 * f1_penalty
        + 0.12 * ic_penalty
        + 0.07 * lag_penalty
        + 0.05 * disagreement
    )

    return {
        "ModalDisagreement": disagreement,
        "TemporalLag": lag_penalty,
        "CompositeScore": float(composite_score),
    }


# ======================================================================
# Drawdown / risk
# ======================================================================

def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of a return series."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(r)])
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    r = np.asarray(returns, dtype=float)
    ann_return = float(np.mean(r) * periods_per_year)
    mdd = abs(max_drawdown(r))
    return ann_return / mdd if mdd > 1e-12 else 0.0


# ======================================================================
# Main metric bundle
# ======================================================================
def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """Compute all benchmark metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    da_val = directional_accuracy(y_true, y_pred)

    # ESS from rough AR(1)-style approximation on target autocorrelation.
    if len(y_true) > 2:
        ac1 = float(np.corrcoef(y_true[:-1], y_true[1:])[0, 1])
        if not np.isfinite(ac1):
            ac1 = 0.0
        ac1 = float(np.clip(ac1, -0.99, 0.99))
    else:
        ac1 = 0.0
    ess = max(1, int(len(y_true) * (1.0 - ac1) / (1.0 + ac1 + 1e-9)))

    # Base-rate directional accuracy on the same active subset.
    eps = _direction_threshold(y_true)
    true_sign = _signed_labels(y_true, eps=eps)
    active = true_sign != 0
    if np.any(active):
        pos_frac = float(np.mean(true_sign[active] == 1))
        neg_frac = float(np.mean(true_sign[active] == -1))
        base_rate_da = round(max(pos_frac, neg_frac) * 100.0, 2)
    else:
        base_rate_da = 0.0

    sharpe_val = sharpe_ratio(y_true, y_pred, horizon=horizon)
    ic_val = information_coefficient(y_true, y_pred)
    prec_val = direction_precision(y_true, y_pred)
    rec_val = direction_recall(y_true, y_pred)
    f1_val = direction_f1(y_true, y_pred)

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA%": da_val,
        "Sharpe": sharpe_val,
        "IC": ic_val,
        "Prec": prec_val,
        "Rec": rec_val,
        "F1": f1_val,
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
    """Diebold-Mariano test for equal predictive accuracy."""
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

    d = loss_a - loss_b
    d_mean = d.mean()

    bandwidth = max(horizon, 1)
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        if n - k <= 0:
            break
        weight = 1.0 - k / (bandwidth + 1)
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * weight * gamma_k
    var_d = (gamma_0 + gamma_sum) / max(n, 1)

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
    """Paired bootstrap test for directional accuracy difference."""
    y_true = np.asarray(y_true, dtype=float)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    eps = _direction_threshold(y_true)
    true_sign = _signed_labels(y_true, eps=eps)
    pred_a_sign = _signed_labels(preds_a, eps=eps)
    pred_b_sign = _signed_labels(preds_b, eps=eps)
    active = true_sign != 0

    correct_a = (true_sign == pred_a_sign) & active
    correct_b = (true_sign == pred_b_sign) & active

    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        active_count = np.sum(active[idx])
        if active_count == 0:
            deltas[i] = 0.0
            continue
        da_a = correct_a[idx].sum() / active_count * 100.0
        da_b = correct_b[idx].sum() / active_count * 100.0
        deltas[i] = da_b - da_a

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_left = float(np.mean(deltas <= 0))
    p_right = float(np.mean(deltas >= 0))
    p_value = min(1.0, 2.0 * min(p_left, p_right))

    return {
        "delta_da": float(np.mean(deltas)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }