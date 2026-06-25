"""Ablation benchmark visualization.

Two chart types per table × horizon:
  1. heatmap  — absolute metric values, green=best, red=worst, globally normalised
  2. delta    — lollipop of Δmetric vs the within-table baseline

Public API
----------
plot_table_charts(df, table, horizon, figures_dir)
    Generates both charts for all primary metrics. Called from run_ablation_benchmark.

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
    # New tables
    "data_ablation": "Data Ablation",
    "architecture_ablation": "Architecture Ablation",
    "feature_extractor_ablation": "Feature Extractor Ablation",
    "cmtf_search": "CMTF Search",

    # Legacy tables
    "fusion": "Fusion Strategy",
    "news_scope": "News Scope",
    "sentiment": "Sentiment Mode",
    "component": "CMTF Component Toggles",
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
            f"PE={'Y' if bool(r.get('use_positional_encoding', True)) else 'N'} "
            f"Gate={'Y' if bool(r.get('use_news_gate', True)) else 'N'} "
            f"K={int(r.get('recency_gate_k', 5))} "
            f"TS={'Y' if bool(r.get('use_two_stage', True)) else 'N'} "
            f"Aux={'Y' if bool(r.get('use_aux_loss', True)) else 'N'} "
            f"VR={'Y' if bool(r.get('use_variance_reg', True)) else 'N'}"
        ),
        axis=1,
    )
    return df


def _resolve_plot_category(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, str]:
    """
    Return (plot_df, category_col) where category_col is guaranteed to exist.
    """
    plot_df = df.copy()

    # -------------------------
    # New final-study tables
    # -------------------------
    if table == "data_ablation":
        if "fusion_type" not in plot_df.columns:
            raise KeyError("data_ablation plotting requires column 'fusion_type'")
        plot_df["_plot_category"] = plot_df["fusion_type"].astype(str)
        return plot_df, "_plot_category"

    if table == "architecture_ablation":
        if "use_cross_attention" not in plot_df.columns:
            raise KeyError("architecture_ablation plotting requires column 'use_cross_attention'")
        plot_df["_plot_category"] = plot_df["use_cross_attention"].map(
            lambda x: "xattn_on" if bool(x) else "xattn_off"
        )
        return plot_df, "_plot_category"

    if table == "feature_extractor_ablation":
        required_cols = {"model_name", "fusion_type"}
        missing = required_cols - set(plot_df.columns)
        if missing:
            raise KeyError(f"feature_extractor_ablation plotting missing columns: {sorted(missing)}")

        def _feat_label(row):
            fusion = str(row["fusion_type"])
            model = str(row["model_name"])
            enc = row["market_encoder_name"] if "market_encoder_name" in plot_df.columns else "na"
            enc = "na" if pd.isna(enc) else str(enc)

            if fusion == "cmtf":
                return f"cmtf::{enc}"
            return f"{model}::{fusion}"

        plot_df["_plot_category"] = plot_df.apply(_feat_label, axis=1)
        return plot_df, "_plot_category"

    if table == "cmtf_search":
        required_cols = {
            "recency_gate_k",
            "aux_loss_weight",
            "encoder_lr_scale",
            "fusion_market_dim",
            "fusion_hidden_dim",
        }
        missing = required_cols - set(plot_df.columns)
        if missing:
            raise KeyError(f"cmtf_search plotting missing columns: {sorted(missing)}")

        def _search_label(row):
            return (
                f"k={int(row['recency_gate_k'])}"
                f"|auxw={float(row['aux_loss_weight']):.2f}"
                f"|elr={float(row['encoder_lr_scale']):.2f}"
                f"|fmd={int(row['fusion_market_dim'])}"
                f"|fhd={int(row['fusion_hidden_dim'])}"
            )

        plot_df["_plot_category"] = plot_df.apply(_search_label, axis=1)
        return plot_df, "_plot_category"

    # -------------------------
    # Legacy tables
    # -------------------------
    if table == "fusion":
        plot_df["_plot_category"] = plot_df["fusion_type"].astype(str)
        return plot_df, "_plot_category"

    if table == "news_scope":
        plot_df["_plot_category"] = plot_df["news_scope"].astype(str)
        return plot_df, "_plot_category"

    if table == "sentiment":
        plot_df["_plot_category"] = plot_df["sentiment_mode"].astype(str)
        return plot_df, "_plot_category"

    if table == "component":
        plot_df = _add_toggle_col(plot_df)
        return plot_df, "toggle"

    raise KeyError(f"Unknown table for plotting: {table}")


def _baseline_mask(df: pd.DataFrame, table: str) -> pd.Series:
    """
    Return a boolean mask selecting baseline rows for delta plots.
    """
    mask = pd.Series([False] * len(df), index=df.index)

    # -------------------------
    # New final-study tables
    # -------------------------
    if table == "data_ablation":
        if "fusion_type" in df.columns:
            mask = df["fusion_type"].astype(str) == "none"
        return mask

    if table == "architecture_ablation":
        if "use_cross_attention" in df.columns:
            mask = df["use_cross_attention"].astype(bool) == True
        return mask

    if table == "feature_extractor_ablation":
        if "fusion_type" in df.columns:
            mask = df["fusion_type"].astype(str) == "none"
        return mask

    if table == "cmtf_search":
        # Preferred internal baseline:
        # 1) gate=True, vreg=True, k=5
        # 2) otherwise k=5
        # 3) otherwise first row per model
        if all(col in df.columns for col in ["use_news_gate", "use_variance_reg", "recency_gate_k"]):
            preferred = (
                (df["use_news_gate"].astype(bool) == True)
                & (df["use_variance_reg"].astype(bool) == True)
                & (df["recency_gate_k"] == 5)
            )
            if preferred.any():
                return preferred

            preferred_k5 = df["recency_gate_k"] == 5
            if preferred_k5.any():
                return preferred_k5

        # fallback: first row per model
        mask = pd.Series([False] * len(df), index=df.index)
        if "model_name" in df.columns and not df.empty:
            first_idx = df.groupby("model_name", sort=False).head(1).index
            mask.loc[first_idx] = True
        return mask

    # -------------------------
    # Legacy tables
    # -------------------------
    if table == "fusion":
        if "fusion_type" in df.columns:
            mask = df["fusion_type"].astype(str) == "none"
        return mask

    if table == "news_scope":
        if "news_scope" in df.columns:
            mask = df["news_scope"].astype(str) == "none"
        return mask

    if table == "sentiment":
        if "sentiment_mode" in df.columns:
            mask = df["sentiment_mode"].astype(str) == "none"
        return mask

    if table == "component":
        needed = [
            ("use_positional_encoding", True),
            ("use_news_gate", True),
            ("recency_gate_k", 5),
            ("use_two_stage", True),
            ("use_aux_loss", True),
            ("use_variance_reg", True),
        ]
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in needed:
            if col in df.columns:
                mask &= df[col] == val
            else:
                mask &= False
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


# ------------------------------------------------------------------
# Backward-compat wrappers
# ------------------------------------------------------------------

def plot_fusion_comparison(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "fusion", horizon, save_path.parent)


def plot_news_scope(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "news_scope", horizon, save_path.parent)


def plot_sentiment_ablation(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "sentiment", horizon, save_path.parent)


def plot_component_ablation(df: pd.DataFrame, horizon: int, save_path: Path) -> None:
    plot_table_charts(df, "component", horizon, save_path.parent)