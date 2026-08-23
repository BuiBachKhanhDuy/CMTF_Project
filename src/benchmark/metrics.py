"""Evaluation metrics for forecasting benchmarks.

Directional metrics use adaptive dead-zone thresholds, symmetric class metrics,
and comparable active subsets for base-rate and model performance.
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

def _direction_threshold(values: np.ndarray, eps: float | None = None) -> float:
    """Choose a robust dead-zone threshold for directional metrics.

    If eps is provided, use it directly.
    Otherwise use a small data-adaptive threshold based on the scale of
    the given series (half of its own 20th-percentile absolute magnitude).

    NOTE: this is deliberately generic over "values" (not "y_true") because
    it is now called separately on y_true, y_pred, preds_a, preds_b, and
    anchor_pred -- each series gets a threshold sized to ITS OWN scale.
    See FIX-1..FIX-4 in the module docstring.
    """
    values = np.asarray(values, dtype=float)
    if eps is not None:
        return float(max(eps, 0.0))

    abs_v = np.abs(values)
    if abs_v.size == 0:
        return 0.0

    # Robust small-move threshold:
    # half of the 20th percentile magnitude, bounded below.
    q20 = float(np.percentile(abs_v, 20))
    return max(0.5 * q20, 1e-6)


def _resolve_pair_thresholds(
    series_a: np.ndarray,
    series_b: np.ndarray,
    eps: float | None = None,
) -> tuple[float, float]:
    """Return (thr_a, thr_b) for two series being compared directionally.

    If eps is given, both series share that fixed threshold (a deliberate
    manual dead-zone, valid because both series are in the same return
    units). If eps is None, each series gets its own adaptive threshold
    from its own distribution -- this is FIX-1/3/4: don't let one series'
    scale determine whether the other series' values count as "active".
    """
    if eps is not None:
        thr = float(max(eps, 0.0))
        return thr, thr
    return _direction_threshold(series_a), _direction_threshold(series_b)


def _signed_labels(values: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Map values to {-1, 0, +1} using a dead-zone threshold."""
    values = np.asarray(values, dtype=float)

    if eps is None:
        raise ValueError("eps must be provided as a fixed threshold")

    thr = float(eps)

    labels = np.zeros_like(values, dtype=np.int8)
    labels[values > thr] = 1
    labels[values < -thr] = -1
    return labels

# Directional metrics

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    # Resolve thresholds independently for returns and predictions.
    thr_true, thr_pred = _resolve_pair_thresholds(y_true, y_pred, eps)

    true_sign = _signed_labels(y_true, eps=thr_true)
    pred_sign = _signed_labels(y_pred, eps=thr_pred)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    return float(np.mean(true_sign[active] == pred_sign[active]) * 100.0)


def direction_precision(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    thr_true, thr_pred = _resolve_pair_thresholds(y_true, y_pred, eps)

    true_sign = _signed_labels(y_true, eps=thr_true)
    pred_sign = _signed_labels(y_pred, eps=thr_pred)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    true_active = true_sign[active]
    pred_active = pred_sign[active]

    precisions = []
    for cls in (-1, 1):
        tp = np.sum((true_active == cls) & (pred_active == cls))
        fp = np.sum((true_active != cls) & (pred_active == cls))
        denom = tp + fp
        if denom > 0:
            precisions.append(tp / denom)

    return float(np.mean(precisions)) if precisions else 0.0


def direction_recall(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    thr_true, thr_pred = _resolve_pair_thresholds(y_true, y_pred, eps)

    true_sign = _signed_labels(y_true, eps=thr_true)
    pred_sign = _signed_labels(y_pred, eps=thr_pred)
    active = true_sign != 0

    if not np.any(active):
        return 0.0

    true_active = true_sign[active]
    pred_active = pred_sign[active]

    recalls = []
    for cls in (-1, 1):
        tp = np.sum((true_active == cls) & (pred_active == cls))
        fn = np.sum((true_active == cls) & (pred_active != cls))
        denom = tp + fn
        if denom > 0:
            recalls.append(tp / denom)

    return float(np.mean(recalls)) if recalls else 0.0


def direction_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    thr_true, thr_pred = _resolve_pair_thresholds(y_true, y_pred, eps)

    true_sign = _signed_labels(y_true, eps=thr_true)
    pred_sign = _signed_labels(y_pred, eps=thr_pred)
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

        denom_p = tp + fp
        denom_r = tp + fn

        precision = tp / denom_p if denom_p > 0 else 0.0
        recall = tp / denom_r if denom_r > 0 else 0.0

        if precision + recall > 0:
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
    """Annualised Sharpe ratio of a sign-based long/short strategy.

    FIX-2: the trading threshold is now sized from y_pred's OWN
    distribution, not y_true's. Previously a model with small-magnitude
    predictions had most of its signals zeroed out by a threshold sized
    from the (larger-scale) actual returns, producing a sparse, noisy
    trade set whose Sharpe could look good or bad almost by chance --
    and disagreeing with DA_skill%, which penalizes those same zeroed
    predictions as wrong. Sizing the threshold from y_pred's own scale
    means "roughly the top ~80% of this model's own signals" get traded,
    regardless of how that model's output scale compares to actual
    return magnitude.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    thr_pred = _direction_threshold(y_pred)

    pred_sign = _signed_labels(y_pred, thr_pred)
    strategy_returns = pred_sign * y_true

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
    """Fraction of sign disagreements versus a shared market-only anchor.

    FIX-3: anchor_pred and y_pred now each get their own adaptive
    threshold (previously both were thresholded using y_pred's scale
    alone, inconsistent with the rest of the file and unfair whenever
    the anchor and the candidate prediction have different scales).
    """
    if anchor_pred is None:
        return 0.0

    anchor_pred = np.asarray(anchor_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(anchor_pred) == 0 or len(y_pred) == 0:
        return 0.0

    thr_anchor, thr_pred = _resolve_pair_thresholds(anchor_pred, y_pred, eps)

    anchor_sign = _signed_labels(anchor_pred, thr_anchor)
    pred_sign = _signed_labels(y_pred, thr_pred)

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

    NOTE: no longer computes a single shared `thr` up front -- each call
    below resolves its own per-series threshold(s) internally (FIX-1/3).
    """
    rmse_value = rmse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)
    da_value = directional_accuracy(y_true, y_pred)
    f1_value = direction_f1(y_true, y_pred)
    disagreement = modal_disagreement(anchor_pred, y_pred)
    lag_penalty = temporal_lag(y_true, y_pred, horizon=horizon)
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


# Main metric bundle
def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """Compute all benchmark metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # The true-return threshold defines the active subset for the base rate.
    thr_true = _direction_threshold(y_true)

    da_val = directional_accuracy(y_true, y_pred)

    # ESS adjusted for overlapping horizons
    n = len(y_true)

    if n <= 2:
        ess = n
    else:
        if horizon <= 1:
            # no overlap → use original n
            ess = n
        else:
            # remove overlap by subsampling
            ess = max(1, n // horizon)

    # Base-rate directional accuracy on the same active subset.
    true_sign = _signed_labels(y_true, thr_true)
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
        **compute_directional_independent(y_true, y_pred, horizon),
    }


def flag_degenerate(metrics: dict, y_pred: np.ndarray | None = None) -> bool:
    """Robust collapse / no-skill detector for a computed metrics bundle.

    The old rule (F1 < 0.01 and |DA% - 50| < 1.5) only caught the narrow
    exact-50%% case and silently passed real collapses such as a constant
    single-sign output that lands exactly on the base rate (IC == 0,
    Rec == 1.0, DA%% == base_rate_DA%%, F1 ~ 0.33). This ORs in the
    unambiguous collapse signatures:
      - zero prediction variance (constant output),
      - IC == 0 exactly (spearman of a constant series -> nan -> 0),
      - Recall pinned to a single class (~0 or ~1),
      - sitting on the base rate with no directional skill.
    """
    da = float(metrics.get("DA%", 0.0))
    base = float(metrics.get("base_rate_DA%", 50.0))
    f1 = float(metrics.get("F1", 0.0))
    ic = float(metrics.get("IC", 0.0))
    rec = float(metrics.get("Rec", 0.5))

    pred_std_zero = (
        y_pred is not None
        and float(np.std(np.asarray(y_pred, dtype=float))) < 1e-8
    )

    return bool(
        pred_std_zero
        or abs(ic) < 1e-6                              # constant output -> spearman nan->0
        or rec > 0.98 or rec < 0.02                    # single-sign predictions
        or (abs(da - base) < 0.25 and f1 < 0.40)       # sits on base rate, no skill
        or (f1 < 0.01 and abs(da - 50.0) < 1.5)        # original narrow rule, retained
    )


def compute_directional_independent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """Directional metrics on non-overlapping subsamples (phase-averaged stride=horizon).

    For a horizon-h forward return, consecutive targets overlap by h-1 steps.
    Evaluating on stride-h indices (h phase offsets, then averaged) gives
    ~n/h statistically independent evaluations per phase — correcting the
    inflated DA% / Prec / Rec / F1 that results from overlapping targets.

    For horizon <= 1 returns the same values as the standard directional metrics.

    NOTE: each phase now delegates to directional_accuracy/precision/recall/f1
    directly, which resolve per-series thresholds internally (FIX-1), instead
    of computing a single thr from the phase's y_true slice and reusing it for
    y_pred.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if horizon <= 1 or len(y_true) == 0:
        return {
            "DA_ind%": directional_accuracy(y_true, y_pred),
            "Prec_ind": direction_precision(y_true, y_pred),
            "Rec_ind": direction_recall(y_true, y_pred),
            "F1_ind": direction_f1(y_true, y_pred),
        }

    n = len(y_true)
    phase_da: list[float] = []
    phase_prec: list[float] = []
    phase_rec: list[float] = []
    phase_f1: list[float] = []

    for offset in range(horizon):
        idx = np.arange(offset, n, horizon)
        if len(idx) < 2:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        phase_da.append(directional_accuracy(yt, yp))
        phase_prec.append(direction_precision(yt, yp))
        phase_rec.append(direction_recall(yt, yp))
        phase_f1.append(direction_f1(yt, yp))

    if not phase_da:
        return {"DA_ind%": 0.0, "Prec_ind": 0.0, "Rec_ind": 0.0, "F1_ind": 0.0}

    return {
        "DA_ind%": float(np.mean(phase_da)),
        "Prec_ind": float(np.mean(phase_prec)),
        "Rec_ind": float(np.mean(phase_rec)),
        "F1_ind": float(np.mean(phase_f1)),
    }

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
    """Paired bootstrap test for directional accuracy difference.

    FIX-4: preds_a and preds_b now each get their own adaptive threshold
    (previously both were thresholded using y_true's scale alone). This is
    symmetric-fair between the two models being compared but no longer
    penalizes whichever model happens to have smaller output magnitude,
    independent of whether its sign calls are actually correct.
    """
    y_true = np.asarray(y_true, dtype=float)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    thr_true = _direction_threshold(y_true)
    thr_a = _direction_threshold(preds_a)
    thr_b = _direction_threshold(preds_b)

    true_sign = _signed_labels(y_true, thr_true)
    pred_a_sign = _signed_labels(preds_a, thr_a)
    pred_b_sign = _signed_labels(preds_b, thr_b)

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
