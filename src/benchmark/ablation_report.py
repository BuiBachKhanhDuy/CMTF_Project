"""ablation_report.py

Aggregated reporting framework for the registry-driven ablation study
(src/benchmark/ablation_registry.py):

    rank_cells(...)
        Ranks every cell by (gated DA%, gated Sharpe, gated IC) descending —
        the PRIMARY reported metrics per the roadmap.

    real_minus_placebo_table(...)
        For every registry real/placebo twin pair, computes
        real_minus_placebo_{DA,Sharpe,IC} point estimates plus bootstrap 95%
        confidence intervals from the frozen per-seed TEST predictions cached
        on disk (no retraining).

    generate_markdown_report(...)
        Renders everything (rankings, placebo deltas, coverage diagnostics,
        per-cell documentation) into one markdown summary.

All functions are pure transforms of an aggregated results DataFrame (as
produced by ``run_ablation_registry.py``) plus the on-disk prediction cache;
nothing here retrains a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ablation_config import AblationConfig
from .ablation_registry import ABLATION_CELLS, CELL_GROUP, CELL_NOTES, PLACEBO_PAIRS
from .ablation_runner import _config_hash
from .metrics import directional_accuracy, information_coefficient, sharpe_ratio

PRIMARY_METRICS: tuple[str, str, str] = ("DA%_gated", "Sharpe_gated", "IC_gated")
SECONDARY_METRICS: tuple[str, ...] = ("DA%", "Sharpe", "IC", "RMSE", "gate_coverage")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_cells(df: pd.DataFrame, metrics: tuple[str, ...] = PRIMARY_METRICS) -> pd.DataFrame:
    """Rank rows by ``metrics`` lexicographically, descending (best first).

    Expects a ``cell`` column identifying each registry cell id. Rows missing
    any ranking metric sort last.
    """
    if df.empty:
        return df.copy()
    work = df.copy()
    for m in metrics:
        if m not in work.columns:
            work[m] = float("nan")
    # NaNs must sort last even though we want descending order overall.
    sort_cols = list(metrics)
    na_rank = work[sort_cols].isna().any(axis=1).astype(int)
    work["_na_rank"] = na_rank
    work = work.sort_values(
        by=["_na_rank", *sort_cols],
        ascending=[True] + [False] * len(sort_cols),
        kind="mergesort",
    ).drop(columns="_na_rank")
    work.insert(0, "rank", range(1, len(work) + 1))
    return work.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real-minus-placebo + bootstrap CI
# ---------------------------------------------------------------------------

def _load_prediction(cfg: AblationConfig, seed: int, horizon: int, cache_dir: Path) -> np.ndarray | None:
    short_id = _config_hash(cfg)
    path = cache_dir / "predictions" / f"{short_id}__seed{seed}__{horizon}d.npy"
    if not path.exists():
        return None
    return np.load(str(path))


def _load_truth(horizon: int, cache_dir: Path) -> np.ndarray | None:
    path = cache_dir / "predictions" / f"truth__{horizon}d.npy"
    if not path.exists():
        return None
    return np.load(str(path))


def bootstrap_real_minus_placebo(
    y_true: np.ndarray,
    real_pred: np.ndarray,
    placebo_pred: np.ndarray,
    horizon: int = 1,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap 95%% CI for (real - placebo) on DA% / Sharpe / IC.

    Resamples TEST row indices with replacement (paired: the same resampled
    indices are applied to both the real and placebo predictions and to
    truth), so each bootstrap draw is a consistent alternate "test set".
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    real_pred = np.asarray(real_pred, dtype=np.float64).ravel()
    placebo_pred = np.asarray(placebo_pred, dtype=np.float64).ravel()
    n = y_true.size
    rng = np.random.RandomState(seed)

    deltas = {"DA%": np.empty(n_bootstrap), "Sharpe": np.empty(n_bootstrap), "IC": np.empty(n_bootstrap)}
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, rp, pp = y_true[idx], real_pred[idx], placebo_pred[idx]
        deltas["DA%"][i] = directional_accuracy(yt, rp) - directional_accuracy(yt, pp)
        deltas["Sharpe"][i] = sharpe_ratio(yt, rp, horizon=horizon) - sharpe_ratio(yt, pp, horizon=horizon)
        deltas["IC"][i] = information_coefficient(yt, rp) - information_coefficient(yt, pp)

    out: dict[str, float] = {}
    for metric, arr in deltas.items():
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"real_minus_placebo_{metric.rstrip('%')}"] = float("nan")
            out[f"{metric.rstrip('%')}_ci_low"] = float("nan")
            out[f"{metric.rstrip('%')}_ci_high"] = float("nan")
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[f"real_minus_placebo_{metric.rstrip('%')}"] = float(np.mean(arr))
        out[f"{metric.rstrip('%')}_ci_low"] = float(lo)
        out[f"{metric.rstrip('%')}_ci_high"] = float(hi)
    return out


def real_minus_placebo_table(
    horizon: int,
    seeds: tuple[int, ...] = (1, 42, 123),
    cache_dir: Path = Path("cache"),
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Real-vs-placebo comparison for every registry twin pair (0/0p, 8/8p, 18/18p).

    Bootstrap CI is computed on the FIRST seed with both real and placebo
    predictions cached on disk (documented limitation: with only 3 seeds a
    per-seed bootstrap is more informative than pooling non-iid seeds
    together). The point-estimate real_minus_placebo_* columns average the
    delta across every seed that has both predictions cached.
    """
    y_true = _load_truth(horizon, cache_dir)
    rows = []
    for real_id, placebo_id in PLACEBO_PAIRS.items():
        real_cfg = ABLATION_CELLS[real_id]
        placebo_cfg = ABLATION_CELLS[placebo_id]

        per_seed_deltas: list[dict[str, float]] = []
        boot_seed_used = None
        boot_result: dict[str, float] = {}

        for seed in seeds:
            real_pred = _load_prediction(real_cfg, seed, horizon, cache_dir)
            placebo_pred = _load_prediction(placebo_cfg, seed, horizon, cache_dir)
            if real_pred is None or placebo_pred is None or y_true is None:
                continue
            per_seed_deltas.append({
                "DA%": directional_accuracy(y_true, real_pred) - directional_accuracy(y_true, placebo_pred),
                "Sharpe": sharpe_ratio(y_true, real_pred, horizon=horizon) - sharpe_ratio(y_true, placebo_pred, horizon=horizon),
                "IC": information_coefficient(y_true, real_pred) - information_coefficient(y_true, placebo_pred),
            })
            if boot_seed_used is None:
                boot_seed_used = seed
                boot_result = bootstrap_real_minus_placebo(
                    y_true, real_pred, placebo_pred, horizon=horizon, n_bootstrap=n_bootstrap, seed=seed,
                )

        if not per_seed_deltas:
            rows.append({
                "real_cell": real_id, "placebo_cell": placebo_id,
                "real_minus_placebo_DA": float("nan"),
                "real_minus_placebo_Sharpe": float("nan"),
                "real_minus_placebo_IC": float("nan"),
                "n_seeds_available": 0,
                "bootstrap_seed": None,
            })
            continue

        delta_df = pd.DataFrame(per_seed_deltas)
        row = {
            "real_cell": real_id,
            "placebo_cell": placebo_id,
            "real_minus_placebo_DA": float(delta_df["DA%"].mean()),
            "real_minus_placebo_Sharpe": float(delta_df["Sharpe"].mean()),
            "real_minus_placebo_IC": float(delta_df["IC"].mean()),
            "n_seeds_available": len(per_seed_deltas),
            "bootstrap_seed": boot_seed_used,
            **boot_result,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame, cols: list[str] | None = None, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Hand-rolled (no ``tabulate`` dependency): joins string cells with ``|``.
    """
    if df.empty:
        return "_(no rows)_\n"
    work = df[cols].copy() if cols else df.copy()
    for c in work.columns:
        if pd.api.types.is_float_dtype(work[c]):
            work[c] = work[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            work[c] = work[c].map(lambda v: "" if pd.isna(v) else str(v))

    header = "| " + " | ".join(str(c) for c in work.columns) + " |"
    sep = "|" + "|".join(["---"] * len(work.columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(row) + " |"
        for row in work.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def generate_markdown_report(
    horizon: int,
    ranked_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render the full registry report for one horizon to markdown."""
    lines: list[str] = []
    lines.append(f"# CMTF Component Ablation Registry — {horizon}D Report\n")
    lines.append(
        "Reproducible, config-driven ablation of the CMTF(LSTM) champion. "
        "Every cell derives from `CMTF_CORE` via a single-field override "
        "(src/benchmark/ablation_registry.py); GATED metrics "
        "(`DA%_gated`, `Sharpe_gated`, `IC_gated`) are the PRIMARY reported "
        "metrics, computed by a validation-calibrated fixed-coverage "
        "confidence gate layered on top of each cell's frozen predictions.\n"
    )

    lines.append("## Ranking (by gated DA%, then gated Sharpe, then gated IC)\n")
    rank_cols = [c for c in (
        "rank", "cell", "group", *PRIMARY_METRICS, *SECONDARY_METRICS, "seed_count",
    ) if c in ranked_df.columns]
    lines.append(_df_to_md(ranked_df, cols=rank_cols))
    lines.append("")

    lines.append("## Real-minus-placebo comparisons (bootstrap 95% CI)\n")
    lines.append(
        "Positive `real_minus_placebo_*` means the REAL-news cell beats its "
        "shuffled-news placebo twin — evidence of genuine news signal rather "
        "than a generic decision-layer artifact. CI computed via paired "
        "bootstrap (2000 resamples) on one seed's frozen test predictions "
        "(see `bootstrap_seed` column); point estimates average the delta "
        "across every seed with both predictions cached.\n"
    )
    lines.append(_df_to_md(placebo_df))
    lines.append("")

    lines.append("## Gate monotonicity (coverage-accuracy diagnostics)\n")
    lines.append(
        "Spearman correlation between confidence rank (decile 1 = full book, "
        "decile 10 = most-confident 10%) and test-set performance. Strong "
        "positive correlation means the gate ranks conviction correctly — "
        "trading only the most confident predictions should not hurt.\n"
    )
    lines.append(_df_to_md(monotonicity_df, float_fmt="{:.3f}"))
    lines.append("")

    lines.append("## Cell documentation\n")
    lines.append("| cell | group | research question |")
    lines.append("|---|---|---|")
    for cid in ABLATION_CELLS:
        note = CELL_NOTES.get(cid, "").replace("\n", " ")
        group = CELL_GROUP.get(cid, "")
        lines.append(f"| {cid} | {group} | {note} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
