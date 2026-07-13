"""Ablation benchmark visualization.

Two chart types per table × horizon:
  1. heatmap  — absolute metric values, green=best, red=worst, globally normalised
  2. delta    — lollipop of Δmetric vs the within-table baseline

Public API
----------
plot_table_charts(df, table, horizon, figures_dir)
    Generates both charts for all primary metrics. Called from run_ablation_benchmark
    for the 'fusion_comparison' study table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_MODEL_PALETTE = {
    "lstm": "#4C72B0",
    "chronos": "#DD8452",
    "cnn_lstm": "#55A868",
    "gpt4ts": "#C44E52",
    "cmtf": "#8172B3",
}

_PRIMARY_METRICS: list[tuple[str, bool, str]] = [
    ("DA%", True, ".1f"),
    ("Sharpe", True, ".3f"),
    ("IC", True, ".3f"),
    ("RMSE", False, ".5f"),
    ("MAE", False, ".5f"),
    ("F1", True, ".3f"),
]

_TABLE_TITLE: dict[str, str] = {
    "fusion_comparison": "Fusion Comparison",
}

def _color_for(model: str) -> str:
    return _MODEL_PALETTE.get(model, "#8C8C8C")


def _truthy(value, default: bool = False) -> bool:
    """Interpret CSV/object values as booleans without treating NaN as True."""
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)








def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_plot_category(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, str]:
    """
        Return (plot_df, category_col) where category_col is guaranteed to exist.
    """
    plot_df = df.copy()

    # -------------------------
    # Final study tables
    # -------------------------
    if table == "fusion_comparison":
        if "fusion_type" not in plot_df.columns:
            raise KeyError("fusion_comparison plotting requires column 'fusion_type'")

        def _fusion_label(row):
            if str(row.get("fusion_type")) == "cmtf" and _truthy(row.get("shuffle_news", False)):
                return "cmtf(placebo)"
            return str(row.get("fusion_type"))

        plot_df["_plot_category"] = plot_df.apply(_fusion_label, axis=1)

        return plot_df, "_plot_category"

    raise KeyError(f"Unknown table for plotting: {table}")

def _baseline_mask(df: pd.DataFrame, table: str) -> pd.Series:
    """
        Return a boolean mask selecting baseline rows for delta plots.
    """
    mask = pd.Series([False] * len(df), index=df.index)

    # -------------------------
    # Final study tables
    # -------------------------
    if table == "fusion_comparison":
        if "fusion_type" in df.columns:
            mask = df["fusion_type"].astype(str) == "none"
        return mask

    return mask


def _plot_heatmap(
    df: pd.DataFrame,
    category_col: str,
    metric: str,
    higher_is_better: bool,
    fmt: str,
    title: str,
    save_path: Path,
) -> None:
    agg = (
        df.groupby(["model_name", category_col])[metric]
        .mean()
        .unstack()
    )

    if agg.empty:
        return

    norm = agg.copy().astype(float)

    if "degenerate" in df.columns:
        degen_agg = (
            df.groupby(["model_name", category_col])["degenerate"]
            .any()
            .unstack()
            .reindex(index=agg.index, columns=agg.columns, fill_value=False)
        )
    else:
        degen_agg = None

    finite_vals = norm.values.flatten()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if len(finite_vals) == 0:
        return

    g_min, g_max = finite_vals.min(), finite_vals.max()
    g_rng = g_max - g_min
    if g_rng > 1e-12:
        norm = (norm - g_min) / g_rng
    else:
        norm = norm.where(norm.isna(), 0.5)

    if not higher_is_better:
        norm = 1.0 - norm

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#D0D0D0")

    n_rows, n_cols = agg.shape
    fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.6), max(2.5, n_rows * 0.8 + 0.8)))
    im = ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(agg.columns.tolist(), rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(agg.index.tolist(), fontsize=9)

    for i, row_lbl in enumerate(agg.index):
        for j, col_lbl in enumerate(agg.columns):
            v = agg.loc[row_lbl, col_lbl]
            if not np.isfinite(v):
                continue
            nv = float(norm.loc[row_lbl, col_lbl])
            txt_color = "white" if nv < 0.25 or nv > 0.80 else "black"
            is_cell_degen = degen_agg is not None and bool(degen_agg.loc[row_lbl, col_lbl])
            suffix = "†" if is_cell_degen else ""
            label = f"{v:{fmt}}%{suffix}" if metric == "DA%" else f"{v:{fmt}}{suffix}"
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=8, color=txt_color, fontweight="bold"
            )

    arrow = "↑ higher=better" if higher_is_better else "↓ lower=better"
    ax.set_title(f"{title}  [{metric}  {arrow}]", fontsize=11, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    if higher_is_better:
        cbar.set_ticklabels([f"{g_min:{fmt}}", f"{(g_min + g_max) / 2:{fmt}}", f"{g_max:{fmt}}"])
    else:
        cbar.set_ticklabels([f"{g_max:{fmt}}", f"{(g_min + g_max) / 2:{fmt}}", f"{g_min:{fmt}}"])

    fig.tight_layout()
    _save(fig, save_path)


def _plot_delta(
    df: pd.DataFrame,
    category_col: str,
    table: str,
    metric: str,
    higher_is_better: bool,
    fmt: str,
    title: str,
    save_path: Path,
) -> None:
    baseline_sel = _baseline_mask(df, table)
    baseline_df = df[baseline_sel].copy()
    if baseline_df.empty:
        return

    baseline = baseline_df.groupby("model_name")[metric].mean()
    if baseline.empty:
        return

    agg = df.groupby(["model_name", category_col])[metric].mean().unstack()
    if agg.empty:
        return

    delta = agg.subtract(baseline, axis=0)
    if not higher_is_better:
        delta = -delta

    cats = delta.columns.tolist()
    models = delta.index.tolist()
    n_cats = len(cats)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(8, max(3, n_cats * 0.55 + 1.2)))
    spacing = 0.22

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * spacing
        y = np.arange(n_cats) + offset
        vals = delta.loc[model, cats].values.astype(float)
        color = _color_for(model)
        ax.scatter(vals, y, color=color, s=55, zorder=3, label=model)
        for xi, yi in zip(vals, y):
            if np.isfinite(xi):
                ax.hlines(yi, 0, xi, color=color, linewidth=1.8, alpha=0.75)

    ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
    ax.set_yticks(np.arange(n_cats))
    ax.set_yticklabels(cats, fontsize=9)

    xlabel = (
        f"Δ{metric} vs baseline  (positive = better)"
        if higher_is_better
        else f"−Δ{metric} vs baseline  (positive = better)"
    )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"{title}  [Δ{metric}]", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, save_path)


def plot_table_charts(
    df: pd.DataFrame,
    table: str,
    horizon: int,
    figures_dir: Path,
) -> None:
    """Generate heatmap + delta lollipop for every primary metric."""
    if df.empty:
        return

    plot_df, cat_col = _resolve_plot_category(df, table)
    tbl_title = f"{_TABLE_TITLE.get(table, table)} — {horizon}D"

    for metric, higher, fmt in _PRIMARY_METRICS:
        if metric not in plot_df.columns:
            continue

        _plot_heatmap(
            plot_df,
            cat_col,
            metric,
            higher,
            fmt,
            title=tbl_title,
            save_path=figures_dir / f"{table}_heatmap_{metric.replace('%', 'pct')}.png",
        )
        _plot_delta(
            plot_df,
            cat_col,
            table,
            metric,
            higher,
            fmt,
            title=tbl_title,
            save_path=figures_dir / f"{table}_delta_{metric.replace('%', 'pct')}.png",
        )