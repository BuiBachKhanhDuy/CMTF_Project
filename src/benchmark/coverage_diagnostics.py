"""coverage_diagnostics.py

Coverage-accuracy diagnostics for gate-related ablation cells.

For a gate to be trustworthy as a *deployable confidence ranking* (not just a
lucky post-hoc threshold), trading a smaller, more-confident top-fraction of
the book should never do WORSE than trading more of it — i.e. DA / Sharpe / IC
should be monotonically non-decreasing as coverage shrinks toward the most
confident predictions. This module operationalises that check:

    coverage_deciles(...)
        Sweeps coverage from 100% (trade everything) down to 10% (trade only
        the most confident decile), calibrating ``tau`` on VALIDATION at each
        coverage level (leak-free) and evaluating on the frozen TEST
        predictions. Returns one row per decile with coverage / DA / Sharpe /
        IC / gate_tau.

    monotonicity_report(...)
        Spearman rank correlation between "confidence rank" (deciles ordered
        from least- to most-confident) and each performance metric. A
        well-ranked gate should show a strong POSITIVE Spearman correlation
        (more confidence -> better performance).

Pure function of (val_pred, val_truth, test_pred, test_truth); no retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .decision_policy import calibrate_gate_fixed_coverage, evaluate_policy

# 100% (full book) down to 10% (top decile only), in 10 steps.
DEFAULT_COVERAGE_DECILES: tuple[float, ...] = tuple(round(c, 2) for c in np.arange(1.0, 0.0, -0.1))


def coverage_deciles(
    val_pred: np.ndarray,
    val_truth: np.ndarray,
    test_pred: np.ndarray,
    test_truth: np.ndarray,
    horizon: int = 1,
    coverage_grid: tuple[float, ...] = DEFAULT_COVERAGE_DECILES,
    conviction: bool = True,
) -> pd.DataFrame:
    """Per-decile coverage -> DA% / Sharpe / IC table on frozen TEST predictions.

    ``tau`` at each coverage level is calibrated on VALIDATION only
    (``calibrate_gate_fixed_coverage``), then applied unchanged to TEST — the
    test set never enters calibration.
    """
    rows = []
    for decile_rank, cov in enumerate(coverage_grid, start=1):
        policy = calibrate_gate_fixed_coverage(val_pred, val_truth, coverage=cov, conviction=conviction)
        gated = evaluate_policy(test_truth, test_pred, policy, horizon=horizon)
        rows.append({
            "decile_rank": decile_rank,  # 1 = least confident (full book) ... N = most confident
            "target_coverage": cov,
            "coverage": round(gated["coverage"], 4),
            "gate_tau": policy.tau,
            "n_traded": gated["n_traded"],
            "DA%": gated["DA%"],
            "Sharpe": gated["Sharpe"],
            "IC": gated["IC"],
        })
    return pd.DataFrame(rows)


def monotonicity_report(deciles_df: pd.DataFrame) -> dict[str, float]:
    """Spearman(rank_confidence, performance) for DA% / Sharpe / IC.

    ``rank_confidence`` increases with ``decile_rank`` (lower coverage = more
    confident subset). A well-calibrated gate should show a strong positive
    correlation: shrinking to the most-confident predictions should not hurt
    (and ideally improves) performance.
    """
    if deciles_df.empty or len(deciles_df) < 3:
        return {
            "spearman_DA": float("nan"),
            "spearman_Sharpe": float("nan"),
            "spearman_IC": float("nan"),
        }

    df = deciles_df.dropna(subset=["DA%", "Sharpe", "IC"], how="all")
    rank_confidence = df["decile_rank"].to_numpy(dtype=np.float64)

    out: dict[str, float] = {}
    for metric, key in (("DA%", "spearman_DA"), ("Sharpe", "spearman_Sharpe"), ("IC", "spearman_IC")):
        vals = df[metric].to_numpy(dtype=np.float64)
        valid = np.isfinite(vals) & np.isfinite(rank_confidence)
        if valid.sum() < 3:
            out[key] = float("nan")
            continue
        rho, _p = spearmanr(rank_confidence[valid], vals[valid])
        out[key] = float(rho) if np.isfinite(rho) else float("nan")
    return out


def coverage_curves(deciles_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return coverage_vs_accuracy and coverage_vs_sharpe curve tables.

    These are simple (coverage, metric) projections of ``deciles_df`` sorted
    by ascending coverage, convenient for plotting or for a quick monotonicity
    eyeball without recomputing anything.
    """
    if deciles_df.empty:
        empty = pd.DataFrame(columns=["coverage", "DA%", "Sharpe", "IC"])
        return {"coverage_vs_accuracy": empty, "coverage_vs_sharpe": empty}

    sorted_df = deciles_df.sort_values("coverage").reset_index(drop=True)
    return {
        "coverage_vs_accuracy": sorted_df[["coverage", "DA%", "IC"]].copy(),
        "coverage_vs_sharpe": sorted_df[["coverage", "Sharpe"]].copy(),
    }


def plot_coverage_curves(deciles_df: pd.DataFrame, out_path, title: str = "") -> None:
    """Save a 2-panel PNG: coverage-vs-DA/IC and coverage-vs-Sharpe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if deciles_df.empty:
        return

    sorted_df = deciles_df.sort_values("coverage").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sorted_df["coverage"], sorted_df["DA%"], marker="o", label="DA%")
    ax0b = axes[0].twinx()
    ax0b.plot(sorted_df["coverage"], sorted_df["IC"], marker="s", color="orange", label="IC")
    axes[0].set_xlabel("coverage (fraction of book traded)")
    axes[0].set_ylabel("DA%")
    ax0b.set_ylabel("IC")
    axes[0].set_title("coverage vs DA / IC")
    axes[0].invert_xaxis()

    axes[1].plot(sorted_df["coverage"], sorted_df["Sharpe"], marker="o", color="green")
    axes[1].set_xlabel("coverage (fraction of book traded)")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_title("coverage vs Sharpe")
    axes[1].invert_xaxis()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
