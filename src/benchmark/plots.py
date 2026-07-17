"""Visualization functions for A/B benchmark results.

Public functions:
    - plot_agent_waterfall(df, out_path)   — agent contribution bridge chart
    - plot_agent_temporal(df, out_path)    — temporal contribution area chart
    - plot_risk_benchmark(df, out_path)
    - plot_equity_curves(df, out_path, buy_threshold)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Color palettes ---
SLATE, TEAL, AMBER = "#475569", "#0d9488", "#d97706"
EMERALD, ROSE = "#059669", "#e11d48"
INDIGO = "#4f46e5"
_RISK_COLORS = {"AT": SLATE, "Thresh": EMERALD, "RG": ROSE}
_RISK_LABELS = {"AT": "Always-Trade", "Thresh": "Threshold", "RG": "Risk-Gated"}
_AGENT_COLORS = {"CMTF": SLATE, "Mkt": TEAL, "News": AMBER, "Both": INDIGO}
_AGENT_LABELS = {"CMTF": "CMTF Only", "Mkt": "+Market Agent",
                 "News": "+News Agent", "Both": "+Both Agents"}


# ======================================================================
# Agent contribution — Waterfall / Bridge chart
# ======================================================================

def plot_agent_waterfall(df: pd.DataFrame, out: Path) -> None:
    """Plot waterfall chart showing each agent's contribution to DA% and IC."""
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.Symbol} {r.Horizon}" for r in df.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Agent Contribution: Prediction Correction Waterfall",
                 fontsize=13, fontweight="bold")

    panels = [
        ("DA%", "Directional Accuracy %"),
        ("IC", "Information Coefficient"),
        ("MAE", "Mean Absolute Error"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        w = 0.18
        offsets = {"CMTF": -1.5 * w, "Mkt": -0.5 * w, "News": 0.5 * w, "Both": 1.5 * w}
        for prefix, offset in offsets.items():
            col = f"{prefix}_{metric}"
            vals = df[col].tolist()
            bars = ax.bar(x + offset, vals, w, label=_AGENT_LABELS[prefix],
                          color=_AGENT_COLORS[prefix], alpha=0.85)
            for b in bars:
                h = b.get_height()
                va = "bottom" if h >= 0 else "top"
                ax.text(b.get_x() + w / 2, h + np.sign(h) * (abs(h) * 0.03 + 0.003),
                        f"{h:.2f}", ha="center", va=va, fontsize=7)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    if len(df) > 0:
        da_delta = (df["Both_DA%"] - df["CMTF_DA%"]).mean()
        ic_delta = (df["Both_IC"] - df["CMTF_IC"]).mean()
        mae_delta = (df["Both_MAE"] - df["CMTF_MAE"]).mean()
        summary = (f"Avg DA% Δ(Both−CMTF): {da_delta:+.1f}  |  "
                   f"Avg IC Δ: {ic_delta:+.4f}  |  "
                   f"Avg MAE Δ: {mae_delta:+.6f}")
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eef2ff", ec=INDIGO, alpha=0.8))

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ======================================================================
# Agent contribution — Temporal area chart
# ======================================================================

def plot_agent_temporal(df: pd.DataFrame, out: Path) -> None:
    """Plot temporal contribution of each agent per symbol × horizon."""
    out.parent.mkdir(parents=True, exist_ok=True)

    # Check required columns exist
    required = {"Mkt_Contribution_Pct", "News_Contribution_Pct", "Cutoff"}
    if not required.issubset(df.columns):
        print(f"Skipping temporal plot — missing columns: {required - set(df.columns)}")
        return

    symbols = sorted(df["Symbol"].unique())
    horizons = sorted(df["Horizon"].unique(), key=lambda h: int(h.replace("d", "")))

    nrows, ncols = len(symbols), len(horizons)
    if nrows == 0 or ncols == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle("Agent Contribution Over Time: Scale Factor by Agent",
                 fontsize=13, fontweight="bold")

    for ri, symbol in enumerate(symbols):
        for ci, horizon in enumerate(horizons):
            ax = axes[ri, ci]
            sub = df[(df["Symbol"] == symbol) & (df["Horizon"] == horizon)].copy()
            sub = sub.sort_values("Cutoff").reset_index(drop=True)

            if len(sub) == 0:
                ax.set_title(f"{symbol} {horizon} (no data)", fontsize=9)
                ax.set_visible(False)
                continue

            dates = pd.to_datetime(sub["Cutoff"])
            mkt_pct = sub["Mkt_Contribution_Pct"].values
            news_pct = sub["News_Contribution_Pct"].values

            ax.fill_between(dates, 0, mkt_pct, alpha=0.5, color=TEAL, label="Market Agent")
            ax.fill_between(dates, 0, news_pct, alpha=0.5, color=AMBER, label="News Agent")
            ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)

            ax.set_title(f"{symbol} {horizon}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Contribution %", fontsize=8)
            ax.tick_params(axis="x", rotation=20, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ======================================================================
# Risk management
# ======================================================================

def _risk_grouped_bars(ax, x, df, metric, ylabel, title):
    w = 0.25
    offsets = {"AT": -w, "Thresh": 0, "RG": w}
    for prefix, offset in offsets.items():
        col = f"{prefix}_{metric}"
        vals = df[col].tolist()
        bars = ax.bar(x + offset, vals, w, label=_RISK_LABELS[prefix],
                      color=_RISK_COLORS[prefix], alpha=0.85)
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(b.get_x() + w / 2, h + np.sign(h) * (abs(h) * 0.04 + 0.003),
                    f"{h:.2f}", ha="center", va=va, fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def plot_risk_benchmark(df: pd.DataFrame, out: Path) -> None:
    """Plot Sharpe, MaxDD, TradeFreq×WinRate scatter, Calmar (2×2 grid)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.Symbol} {r.Horizon}" for r in df.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Risk Management: Always-Trade vs Threshold vs Risk-Gated",
                 fontsize=13, fontweight="bold")

    _risk_grouped_bars(axes[0, 0], x, df, "Sharpe", "Sharpe", "Annualized Sharpe Ratio")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    _risk_grouped_bars(axes[0, 1], x, df, "MaxDD%", "MaxDD %", "Max Drawdown %")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    # TradeFreq vs WinRate scatter
    ax3 = axes[1, 0]
    for prefix in ["AT", "Thresh", "RG"]:
        tf = df[f"{prefix}_TradeFreq%"].values
        wr = df[f"{prefix}_WinRate%"].values
        ax3.scatter(tf, wr, color=_RISK_COLORS[prefix], label=_RISK_LABELS[prefix],
                    s=60, alpha=0.85, edgecolors="white", linewidth=0.5)
        for i, lbl in enumerate(labels):
            ax3.annotate(lbl, (tf[i], wr[i]), fontsize=6, alpha=0.7,
                         xytext=(4, 4), textcoords="offset points")
    ax3.axhline(50, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax3.set_xlabel("Trade Frequency %", fontsize=9)
    ax3.set_ylabel("Win Rate %", fontsize=9)
    ax3.set_title("Trade Frequency vs Win Rate", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    _risk_grouped_bars(axes[1, 1], x, df, "Calmar", "Calmar", "Calmar Ratio")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    if len(df) > 0:
        summary = (
            f"Avg AT Sharpe: {df['AT_Sharpe'].mean():.3f}  |  "
            f"Avg RG Sharpe: {df['RG_Sharpe'].mean():.3f}  |  "
            f"Avg RG TradeFreq: {df['RG_TradeFreq%'].mean():.0f}%  |  "
            f"Rows: {len(df)}"
        )
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f8", ec=ROSE, alpha=0.8))

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ======================================================================
# Equity curves
# ======================================================================

def _equity(returns: np.ndarray) -> np.ndarray:
    """Cumulative wealth curve starting at 1.0."""
    return np.concatenate(([1.0], np.cumprod(1.0 + returns)))


def plot_equity_curves(df: pd.DataFrame, out: Path, buy_threshold: float = 0.012) -> None:
    """Plot equity curves grid: symbols (rows) × horizons (cols)."""
    out.parent.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["Symbol"].unique())
    horizons = sorted(df["Horizon"].unique(), key=lambda h: int(h.replace("d", "")))

    nrows, ncols = len(symbols), len(horizons)
    if nrows == 0 or ncols == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Equity Curves: Always-Trade vs Threshold vs Risk-Gated",
                 fontsize=13, fontweight="bold")

    for ri, symbol in enumerate(symbols):
        for ci, horizon in enumerate(horizons):
            ax = axes[ri, ci]
            sub = df[(df["Symbol"] == symbol) & (df["Horizon"] == horizon)].copy()
            sub = sub.sort_values("Cutoff").reset_index(drop=True)

            if len(sub) == 0:
                ax.set_title(f"{symbol} {horizon} (no data)", fontsize=9)
                ax.set_visible(False)
                continue

            actual = sub["Actual"].values
            pred = sub["Final_Pred"].values
            actions = sub["Action"].values
            scales = sub["Position_Scale"].values.astype(float)

            at_ret = np.sign(pred) * actual
            th_ret = np.where(np.abs(pred) >= buy_threshold,
                              np.sign(pred) * actual, 0.0)
            rg_ret = np.zeros_like(actual)
            for i in range(len(actual)):
                if actions[i] == "long":
                    rg_ret[i] = actual[i] * scales[i]
                elif actions[i] == "short":
                    rg_ret[i] = -actual[i] * scales[i]

            dates = pd.to_datetime(sub["Cutoff"])
            x_dates = np.concatenate(([dates.iloc[0]], dates.values))

            ax.plot(x_dates, _equity(at_ret), color=SLATE, lw=1.2,
                    label="Always-Trade", alpha=0.85)
            ax.plot(x_dates, _equity(th_ret), color=EMERALD, lw=1.2,
                    label="Threshold", alpha=0.85)
            ax.plot(x_dates, _equity(rg_ret), color=ROSE, lw=1.2,
                    label="Risk-Gated", alpha=0.85)
            ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)

            ax.set_title(f"{symbol} {horizon}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Wealth", fontsize=8)
            ax.tick_params(axis="x", rotation=20, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
