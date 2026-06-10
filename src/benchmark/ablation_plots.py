"""Ablation benchmark visualization.

Two chart types per table × horizon:
  1. heatmap  — absolute metric values, green=best, red=worst, per-column normalised
  2. delta    — lollipop of Δmetric vs the within-table baseline (market-only or full hybrid)

Public API
----------
plot_table_charts(df, table, horizon, figures_dir)
    Generates both charts for all primary metrics.  Called from run_ablation_benchmark.

Backward-compat wrappers kept so existing imports do not break:
    plot_fusion_comparison, plot_news_scope,
    plot_sentiment_ablation, plot_component_ablation
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
    "chronos_ft": "#DD8452",
    "cnn_lstm": "#55A868",
    "rf": "#C44E52",
}

# Metrics shown in every chart pair
_PRIMARY_METRICS: list[tuple[str, bool, str]] = [
    # (column_name, higher_is_better, display_format)
    ("DA%",    True,  ".1f"),
    ("Sharpe", True,  ".3f"),
    ("IC",     True,  ".3f"),
    ("RMSE",   False, ".5f"),
    ("MAE",    False, ".5f"),
    ("F1",     True,  ".3f"),
]

# Per-table baseline row selector: which column value identifies the "no-news" baseline
_BASELINE_FILTER: dict[str, dict[str, object]] = {
    "fusion":     {"fusion_type": "none"},
    "news_scope": {"news_scope": "none"},
    "sentiment":  {"sentiment_mode": "none"},
    "component":  {
        "use_positional_encoding": True, "use_news_gate": True, "recency_gate_k": 5,
        "use_two_stage": True,
        "use_aux_loss": True, "use_variance_reg": True,
    },
}

# Human-readable category column per table
_CATEGORY_COL: dict[str, str] = {
    "fusion":     "fusion_type",
    "news_scope": "news_scope",
    "sentiment":  "sentiment_mode",
    "component":  "toggle",
}

_TABLE_TITLE: dict[str, str] = {
    "fusion":     "Fusion Strategy",
    "news_scope": "News Scope",
    "sentiment":  "Sentiment Mode",
    "component":  "CMTF Component Toggles (LSTM & CNN-LSTM only)",
}


def _color_for(model: str) -> str:
    return _MODEL_PALETTE.get(model, "#8C8C8C")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _add_toggle_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable 'toggle' column for component ablation rows."""
    df = df.copy()
    df["toggle"] = df.apply(
        lambda r: (
            f"PE={'Y' if r.get('use_positional_encoding', True) else 'N'} "
            f"Gate={'Y' if r.get('use_news_gate', True) else 'N'} "
            f"K={int(r.get('recency_gate_k', 5))} "
            f"TS={'Y' if r.get('use_two_stage', True) else 'N'} "
            f"Aux={'Y' if r.get('use_aux_loss', True) else 'N'} "
            f"VR={'Y' if r.get('use_variance_reg', True) else 'N'}"
        ),
        axis=1,
    )
    return df


# ------------------------------------------------------------------
# Chart 1: Heatmap — absolute values, globally normalised so colors
#           are comparable across all cells in the matrix.
#           green = best value, red = worst value (direction-aware).
# ------------------------------------------------------------------

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
    # rows = models, cols = categories
    norm = agg.copy().astype(float)

    # Build degenerate mask if the column exists (from ablation_runner detection)
    if "degenerate" in df.columns:
        degen_agg = (
            df.groupby(["model_name", category_col])["degenerate"]
            .any()
            .unstack()
            .reindex(index=agg.index, columns=agg.columns, fill_value=False)
        )
    else:
        degen_agg = None

    # Global min/max normalisation across the entire matrix so that
    # colors are comparable cell-to-cell (not just within each column).
    finite_vals = norm.values.flatten()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if len(finite_vals) == 0:
        return
    g_min, g_max = finite_vals.min(), finite_vals.max()
    g_rng = g_max - g_min
    if g_rng > 1e-12:
        norm = (norm - g_min) / g_rng
    else:
        # All values identical → neutral (NaN cells preserved via pandas ops)
        norm = norm.where(norm.isna(), 0.5)
    # For lower-is-better metrics (RMSE, MAE): invert so green = lowest = best
    if not higher_is_better:
        norm = 1.0 - norm

    # Use gray for cells with no data (model not valid for that category)
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
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color=txt_color, fontweight="bold")

    has_degen = degen_agg is not None and degen_agg.values.any()
    arrow = "↑ higher=better" if higher_is_better else "↓ lower=better"
    ax.set_title(f"{title}  [{metric}  {arrow}]", fontsize=11, fontweight="bold", pad=8)
    # Add colorbar so the absolute scale is visible
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    if higher_is_better:
        cbar.set_ticklabels([f"{g_min:{fmt}}", f"{(g_min+g_max)/2:{fmt}}", f"{g_max:{fmt}}"])
    else:
        # Inverted: 0=worst(g_max raw), 1=best(g_min raw)
        cbar.set_ticklabels([f"{g_max:{fmt}}", f"{(g_min+g_max)/2:{fmt}}", f"{g_min:{fmt}}"])
    fig.tight_layout()
    _save(fig, save_path)


# ------------------------------------------------------------------
# Chart 2: Delta lollipop — Δmetric vs baseline
# ------------------------------------------------------------------

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
    baseline_filter = _BASELINE_FILTER.get(table, {})

    # Build per-model baseline
    mask = pd.Series([True] * len(df), index=df.index)
    for k, v in baseline_filter.items():
        if k in df.columns:
            mask &= df[k] == v
    baseline = df[mask].groupby("model_name")[metric].mean()
    if baseline.empty:
        return

    agg = df.groupby(["model_name", category_col])[metric].mean().unstack()
    delta = agg.subtract(baseline, axis=0)
    if not higher_is_better:
        delta = -delta  # flip so positive = good

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
    xlabel = f"Δ{metric} vs baseline  (positive = better)" if higher_is_better \
             else f"−Δ{metric} vs baseline  (positive = better)"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"{title}  [Δ{metric}]", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, save_path)


# ------------------------------------------------------------------
# Main public entry-point
# ------------------------------------------------------------------

def plot_table_charts(
    df: pd.DataFrame,
    table: str,
    horizon: int,
    figures_dir: Path,
) -> None:
    """Generate heatmap + delta lollipop for every primary metric.

    Produces files:
        figures_dir / {table}_heatmap_{metric}.png
        figures_dir / {table}_delta_{metric}.png
    """
    if df.empty:
        return

    if table == "component":
        df = _add_toggle_col(df)

    cat_col = _CATEGORY_COL.get(table, table)
    tbl_title = f"{_TABLE_TITLE.get(table, table)} — {horizon}D"

    for metric, higher, fmt in _PRIMARY_METRICS:
        if metric not in df.columns:
            continue

        _plot_heatmap(
            df, cat_col, metric, higher, fmt,
            title=tbl_title,
            save_path=figures_dir / f"{table}_heatmap_{metric.replace('%','pct')}.png",
        )
        _plot_delta(
            df, cat_col, table, metric, higher, fmt,
            title=tbl_title,
            save_path=figures_dir / f"{table}_delta_{metric.replace('%','pct')}.png",
        )


# ------------------------------------------------------------------
# Backward-compat wrappers (existing imports keep working)
# ------------------------------------------------------------------

def plot_fusion_comparison(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "fusion", horizon, save_path.parent)


def plot_news_scope(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "news_scope", horizon, save_path.parent)


def plot_sentiment_ablation(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "sentiment", horizon, save_path.parent)


def plot_component_ablation(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "component", horizon, save_path.parent)
