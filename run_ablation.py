"""Ablation study: 4-layer progression from raw CMTF to full multiagent.

Loads existing result CSVs (no new inference). Outputs:
  - results/ablation_results.csv
  - results/figures/ablation_layers.png

Layers:
  1. Always-Trade    — sign(final_pred) every day, no filtering
  2. Threshold-Only  — trade only if |final_pred| >= buy_threshold
  3. Risk-Gated      — multiagent risk_agent controls action + position_scale
  4. Risk-Gated + Agent Correction — same as L3, but using adjusted_pred for direction

Usage:
    python run_ablation.py
    python run_ablation.py --risk-csv results/benchmark_risk.csv --trace-csv results/ab_test_multiagent.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLATE, TEAL, EMERALD, ROSE = "#475569", "#0d9488", "#059669", "#e11d48"
INDIGO = "#4f46e5"

LAYER_COLORS = [SLATE, EMERALD, ROSE, INDIGO]
LAYER_LABELS = [
    "L1: Always-Trade",
    "L2: Threshold",
    "L3: Risk-Gated",
    "L4: RG + Agent Correction",
]


def _sharpe_from_returns(returns: np.ndarray, horizon: int) -> float:
    if len(returns) < 3 or returns.std() == 0:
        return float("nan")
    ann = np.sqrt(252.0 / max(horizon, 1))
    return float((returns.mean() / returns.std()) * ann)


def _max_dd(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(dd.min()) * 100 if len(dd) > 0 else 0.0


def _calmar(returns: np.ndarray, horizon: int) -> float:
    ann_ret = float(returns.mean()) * (252.0 / max(horizon, 1))
    mdd = abs(_max_dd(returns))
    if mdd < 1e-8:
        return float("nan")
    return round(ann_ret / (mdd / 100), 3)


def compute_layer4(trace_df: pd.DataFrame, buy_threshold: float = 0.012) -> pd.DataFrame:
    """Compute Layer 4 metrics from the raw trace using Adjusted_Pred."""
    rows = []
    if "Adjusted_Pred" not in trace_df.columns:
        return pd.DataFrame(rows)

    for (symbol, horizon), grp in trace_df.groupby(["Symbol", "Horizon"]):
        grp = grp.sort_values("Cutoff").reset_index(drop=True)
        actual = grp["Actual"].values
        adj_pred = grp["Adjusted_Pred"].values
        actions = grp["Action"].values
        scales = grp["Position_Scale"].values.astype(float)
        h = int(str(horizon).replace("d", ""))

        # L4: Risk-Gated but using adjusted_pred for direction signal
        rg_adj_ret = np.zeros(len(actual))
        for i in range(len(actual)):
            adj_dir = np.sign(adj_pred[i]) if not np.isnan(adj_pred[i]) else 0
            if actions[i] == "long":
                rg_adj_ret[i] = actual[i] * scales[i]
            elif actions[i] == "short":
                rg_adj_ret[i] = -actual[i] * scales[i]
            elif abs(adj_pred[i]) >= buy_threshold:
                # Agent correction can activate trades that were flat
                rg_adj_ret[i] = adj_dir * actual[i] * 0.5

        rows.append({
            "Symbol": symbol,
            "Horizon": horizon,
            "L4_Sharpe": round(_sharpe_from_returns(rg_adj_ret, h), 3),
            "L4_MaxDD%": round(_max_dd(rg_adj_ret), 2),
            "L4_Calmar": _calmar(rg_adj_ret, h),
        })

    return pd.DataFrame(rows)


def build_ablation_table(risk_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    """Combine Layer 1-3 from risk_df with Layer 4 from trace_df."""
    result = risk_df[["Symbol", "Horizon", "N_Samples"]].copy()

    # Layer 1: Always-Trade
    result["L1_Sharpe"] = risk_df["AT_Sharpe"]
    result["L1_MaxDD%"] = risk_df["AT_MaxDD%"]
    result["L1_Calmar"] = risk_df["AT_Calmar"]

    # Layer 2: Threshold
    result["L2_Sharpe"] = risk_df["Thresh_Sharpe"]
    result["L2_MaxDD%"] = risk_df["Thresh_MaxDD%"]
    result["L2_Calmar"] = risk_df["Thresh_Calmar"]

    # Layer 3: Risk-Gated
    result["L3_Sharpe"] = risk_df["RG_Sharpe"]
    result["L3_MaxDD%"] = risk_df["RG_MaxDD%"]
    result["L3_Calmar"] = risk_df["RG_Calmar"]

    # Layer 4: Risk-Gated + Agent Correction
    l4 = compute_layer4(trace_df)
    if len(l4) > 0:
        result = result.merge(l4, on=["Symbol", "Horizon"], how="left")
    else:
        result["L4_Sharpe"] = np.nan
        result["L4_MaxDD%"] = np.nan
        result["L4_Calmar"] = np.nan

    return result


def plot_ablation(df: pd.DataFrame, out: Path) -> None:
    """Plot 4-layer ablation progression: Sharpe, MaxDD, Calmar."""
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.Symbol} {r.Horizon}" for r in df.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study: 4-Layer System Progression",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("Sharpe", "Sharpe Ratio"),
        ("MaxDD%", "Max Drawdown %"),
        ("Calmar", "Calmar Ratio"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        w = 0.18
        for li in range(4):
            col = f"L{li + 1}_{metric}"
            if col not in df.columns:
                continue
            vals = df[col].fillna(0).tolist()
            offset = (li - 1.5) * w
            bars = ax.bar(x + offset, vals, w, label=LAYER_LABELS[li],
                          color=LAYER_COLORS[li], alpha=0.85)
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
        for col in ["L1_Sharpe", "L3_Sharpe", "L4_Sharpe"]:
            if col not in df.columns:
                df[col] = np.nan
        summary = (
            f"Avg L1 Sharpe: {df['L1_Sharpe'].mean():.3f}  →  "
            f"L3: {df['L3_Sharpe'].mean():.3f}  →  "
            f"L4: {df['L4_Sharpe'].mean():.3f}"
        )
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eef2ff", ec=INDIGO, alpha=0.8))

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study from existing CSVs")
    parser.add_argument("--risk-csv", default="results/benchmark_risk.csv")
    parser.add_argument("--trace-csv", default="results/ab_test_multiagent.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    risk_path = Path(args.risk_csv)
    trace_path = Path(args.trace_csv)

    if not risk_path.exists():
        print(f"Risk CSV not found: {risk_path}")
        print("Run `python run_ab_benchmark.py` first.")
        return
    if not trace_path.exists():
        print(f"Trace CSV not found: {trace_path}")
        print("Run `python run_ab_benchmark.py` first.")
        return

    risk_df = pd.read_csv(risk_path)
    trace_df = pd.read_csv(trace_path)

    out_dir = Path(args.output_dir)
    ablation = build_ablation_table(risk_df, trace_df)

    csv_path = out_dir / "ablation_results.csv"
    ablation.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY: 4-Layer System Progression")
    print("=" * 80)
    print(ablation.to_string(index=False))

    fig_path = out_dir / "figures" / "ablation_layers.png"
    plot_ablation(ablation, fig_path)


if __name__ == "__main__":
    main()
