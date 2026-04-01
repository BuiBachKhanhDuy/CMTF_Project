"""Run Chronos benchmark: Market-Only vs Cross-Modal Temporal Fusion.

Usage:
    python run_chronos_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

# Project imports
from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.benchmark.metrics import compute_all
from src.benchmark.chronos_market import ChronosMarketPredictor
from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


# ======================================================================
# Data extraction helpers
# ======================================================================

def extract_per_symbol_data(
    dataset,
    raw_ohlcv: dict[str, pd.DataFrame],
    seq_len: int = 30,
    target_horizon_days: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-symbol arrays of close windows, news embeddings, targets.

    Returns:
        {symbol: {'close_windows': (N, seq_len),
                  'last_close': (N,),
                  'market_tabular': (N, n_market_features),
                  'news_embs': (N, seq_len, 768),
                  'targets': (N,),
                  'times': array of Timestamps}}
    """
    df = dataset.df.copy()
    symbols = df["symbol"].unique()
    result: dict[str, dict[str, np.ndarray]] = {}

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("time").reset_index(drop=True)
        n = len(sym_df)

        if n < seq_len + 1:
            logger.warning("{} has only {} rows, need {} — skipping", sym, n, seq_len + 1)
            continue

        # Raw close prices from cached fetcher
        raw_close = raw_ohlcv[sym]["close"].values

        # Build aligned raw_close column using time matching
        raw_df = raw_ohlcv[sym][["close"]].rename(columns={"close": "raw_close"})
        sym_df = sym_df.merge(
            raw_df, left_on="time", right_index=True, how="left",
        )

        # Drop rows where raw_close is missing (shouldn't happen)
        sym_df = sym_df.dropna(subset=["raw_close"]).reset_index(drop=True)
        n = len(sym_df)

        raw_c = sym_df["raw_close"].values.astype(np.float64)

        # News embeddings
        news_list = sym_df["news_emb"].tolist()
        news_arr = np.stack(news_list).astype(np.float32)  # (n, 768)

        # Engineered market features from pipeline (OHLCV + technical indicators)
        market_cols = list(getattr(dataset, "market_cols", []))
        market_tabular = sym_df[market_cols].values.astype(np.float32)

        # Forward returns (target)
        target_col_name = f"fwd_ret_{int(target_horizon_days)}d"
        if target_col_name not in sym_df.columns:
            logger.warning("{} missing target column {} — skipping", sym, target_col_name)
            continue
        target_col = sym_df[target_col_name].values.astype(np.float32)

        # Build windows
        close_windows = []
        last_closes = []
        market_tabs = []
        news_embs = []
        targets = []
        times = []

        valid_start = seq_len - 1
        valid_end = n

        for i in range(valid_start, valid_end):
            window = raw_c[i - seq_len + 1 : i + 1]   # (seq_len,)
            # Keep per-bar news embeddings as a sequence for cross-attention
            news_window = news_arr[i - seq_len + 1 : i + 1]  # (seq_len, 768)

            target_val = target_col[i]

            if np.isnan(target_val):
                continue

            close_windows.append(window)
            last_closes.append(raw_c[i])
            market_tabs.append(market_tabular[i])
            news_embs.append(news_window)
            targets.append(target_val)
            times.append(sym_df.iloc[i]["time"])

        result[sym] = {
            "close_windows": np.array(close_windows),
            "last_close": np.array(last_closes),
            "market_tabular": np.array(market_tabs),
            "news_embs": np.array(news_embs),
            "targets": np.array(targets),
            "times": np.array(times),
        }
        logger.info("{} → {} samples extracted", sym, len(targets))

    return result


def split_by_date(
    data: dict[str, np.ndarray],
    times: np.ndarray,
    train_end: str,
    val_end: str,
    target_horizon_days: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Walk-forward split arrays by date with horizon-aware purge buffer.

    For horizons H > 1, a sample at time T has a label using price at time T+H.
    To prevent label leakage, we exclude the last H trading days before each
    split boundary from that split (purge buffer). This ensures:
    - Train samples: T+H <= val_start (no val prices in train labels)
    - Val samples: T+H <= test_start (no test prices in val labels)
    - Test samples: Go to the end (no future prices exist; labels may be NaN)

    Args:
        data: Dict of arrays (targets, close_windows, etc.)
        times: Array of timestamps for each sample.
        train_end: End date of training period (exclusive after purge).
        val_end: End date of validation period (exclusive after purge).
        target_horizon_days: Prediction horizon H; purge H bars at boundaries.

    Returns:
        {'train': {...}, 'val': {...}, 'test': {...}}
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    # Apply purge buffer: exclude last H bars before each boundary
    purge_days = pd.Timedelta(days=target_horizon_days)
    train_end_purged = train_end_ts - purge_days
    val_end_purged = val_end_ts - purge_days

    train_mask = times <= train_end_purged
    val_mask = (times > train_end_ts) & (times <= val_end_purged)
    test_mask = times > val_end_ts

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = {k: v[mask] for k, v in data.items()}

    logger.info(
        "Walk-forward split (horizon={}D, purge={}D) | train={} | val={} | test={}",
        target_horizon_days,
        target_horizon_days,
        np.sum(train_mask),
        np.sum(val_mask),
        np.sum(test_mask),
    )
    return splits


def impute_tabular_splits(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Impute NaNs in market_tabular using train-only column means.

    This avoids leakage while keeping validation/test aligned with train stats.
    """
    if "market_tabular" not in splits.get("train", {}):
        return splits

    train_tab = splits["train"]["market_tabular"]
    if train_tab.size == 0:
        return splits

    col_means = np.nanmean(train_tab, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means).astype(np.float32)

    for split_name in ("train", "val", "test"):
        tab = splits[split_name].get("market_tabular")
        if tab is None or tab.size == 0:
            continue
        splits[split_name]["market_tabular"] = np.where(
            np.isnan(tab),
            col_means,
            tab,
        ).astype(np.float32)

    return splits


# ======================================================================
# Visualization
# ======================================================================

def plot_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    """Overlay actual vs predicted returns for each experiment."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual", color="black", linewidth=1.2, alpha=0.7)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (name, preds) in enumerate(predictions.items()):
        ax.plot(x, preds, label=name, linewidth=0.9, alpha=0.7, color=colors[i % len(colors)])
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Log Return")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot → {}", save_path)


def plot_ablation(results_df: pd.DataFrame, save_path: Path) -> None:
    """Multi-subplot ablation: one panel per metric with zoomed y-axis."""
    metrics = ["MAE", "RMSE", "DA%", "Sharpe", "IC"]
    experiments = results_df["Experiment"].unique()
    n_exp = len(experiments)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    short_labels = ["Zero-Shot", "Linear", "CMTF"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for ax, metric in zip(axes, metrics):
        vals = []
        for i, exp in enumerate(experiments):
            row = results_df[results_df["Experiment"] == exp].iloc[0]
            v = row[metric]
            vals.append(v)
            bar = ax.bar(i, v, color=colors[i % len(colors)], width=0.6)
            # Value label on top of bar
            va = "bottom" if v >= 0 else "top"
            ax.text(i, v, f"{v:.4f}", ha="center", va=va, fontsize=8, fontweight="bold")

        # Zoom y-axis to magnify differences
        v_min, v_max = min(vals), max(vals)
        margin = max(abs(v_max - v_min) * 0.4, abs(v_max) * 0.05)
        if v_min >= 0:
            ax.set_ylim(max(0, v_min - margin), v_max + margin)
        else:
            ax.set_ylim(v_min - margin, v_max + margin)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(range(n_exp))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # Shared legend at top
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, short_labels)]
    fig.legend(handles=legend_handles, loc="upper center", ncol=n_exp, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Chronos Ablation: Market-Only vs CMTF Fusion (AVG)", fontsize=14,
                 fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation chart → {}", save_path)


def plot_per_symbol(results_df: pd.DataFrame, save_path: Path) -> None:
    """Per-symbol heatmap: color-coded table of all metrics × experiments × symbols."""
    metrics = ["MAE", "RMSE", "DA%", "Sharpe", "IC"]
    # Higher-is-better for DA%, Sharpe, IC; lower-is-better for MAE, RMSE
    higher_better = {"MAE": False, "RMSE": False, "DA%": True, "Sharpe": True, "IC": True}

    # Filter out AVG row
    df = results_df[results_df["Symbol"] != "AVG"].copy()
    symbols = df["Symbol"].unique()
    experiments = df["Experiment"].unique()

    # Build row labels: "Symbol / Experiment"
    row_labels = []
    data_matrix = []
    for sym in symbols:
        for exp in experiments:
            row = df[(df["Symbol"] == sym) & (df["Experiment"] == exp)]
            if row.empty:
                continue
            row = row.iloc[0]
            short_exp = exp.replace("Chronos ", "").replace("+ ", "")
            row_labels.append(f"{sym} / {short_exp}")
            data_matrix.append([row[m] for m in metrics])

    data_arr = np.array(data_matrix)  # (n_rows, 5)

    fig, ax = plt.subplots(figsize=(12, max(4, len(row_labels) * 0.5 + 1)))

    # Normalize per column for coloring (0=worst, 1=best)
    norm_data = np.zeros_like(data_arr)
    for j, metric in enumerate(metrics):
        col = data_arr[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-12:
            normalized = (col - col_min) / (col_max - col_min)
        else:
            normalized = np.full_like(col, 0.5)
        if not higher_better[metric]:
            normalized = 1.0 - normalized  # Invert so lower is "better" (green)
        norm_data[:, j] = normalized

    # Plot heatmap
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            val = data_arr[i, j]
            # Format based on metric scale
            if metrics[j] == "DA%":
                text = f"{val:.1f}%"
            elif metrics[j] in ("MAE", "RMSE"):
                text = f"{val:.4f}"
            else:
                text = f"{val:.3f}"
            # Choose text color based on background brightness
            brightness = norm_data[i, j]
            color = "white" if brightness < 0.35 or brightness > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Draw horizontal separators between symbols
    n_exp_per_sym = len(experiments)
    for k in range(1, len(symbols)):
        ax.axhline(y=k * n_exp_per_sym - 0.5, color="black", linewidth=1.5)

    ax.set_title("Per-Symbol Benchmark Heatmap (green = best)", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-symbol heatmap → {}", save_path)


# ======================================================================
# Main benchmark
# ======================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Pipeline config -----
    config = {
        "symbols": ["VCB", "MBB"],
        "start": "2025-04-01",
        "end": "2026-03-31",
        "interval": "1D",
        "ohlcv_source": "KBS",
        "news_source": "web",
        "news_sources": ("cafef_banking", "vietstock"),
        "news_use_cache": False,
        "news_export_trace": True,
        "news_similarity_threshold": 85.0,
        "log_news_coverage": True,
        "sequence_len": 30,
        "horizon": 1,
        "target_horizons_days": [1, 5, 20],
        "train_end": "2025-10-31",
        "val_end": "2025-12-31",
        "normalize_method": "zscore",
    }

    # ----- 2. Fetch raw OHLCV (cached) for Chronos -----
    logger.info("═══ Fetching raw OHLCV for Chronos ═══")
    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(
        config["symbols"], config["start"], config["end"],
    )

    # ----- 3. Load Chronos (once, shared) -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("═══ Loading Chronos (device={}) ═══", device)
    chronos = ChronosMarketPredictor(device=device)

    # ----- 4. Run experiments for each target horizon -----
    horizons = [int(h) for h in config.get("target_horizons_days", [1])]
    for target_h in horizons:
        logger.info("═══ Horizon {}D benchmark ═══", target_h)
        run_cfg = {**config, "target_horizon_days": target_h}

        logger.info("Building dataset for target horizon {}D", target_h)
        dataset = run_pipeline(run_cfg)

        logger.info("Extracting per-symbol arrays for target horizon {}D", target_h)
        per_symbol = extract_per_symbol_data(
            dataset,
            raw_ohlcv,
            seq_len=run_cfg["sequence_len"],
            target_horizon_days=target_h,
        )

        all_results: list[dict] = []
        all_preds: dict[str, list[np.ndarray]] = {
            "Chronos Zero-Shot": [],
            "Chronos Linear-Probe": [],
            "Chronos + CMTF": [],
        }
        all_y_true: list[np.ndarray] = []

        for sym, data in per_symbol.items():
            logger.info("━━━ Benchmark: {} ({}D) ━━━", sym, target_h)

            splits = split_by_date(
                {k: v for k, v in data.items() if k != "times"},
                data["times"],
                run_cfg["train_end"],
                run_cfg["val_end"],
                target_horizon_days=target_h,
            )
            splits = impute_tabular_splits(splits)

            if len(splits["test"]["targets"]) == 0:
                logger.warning("{}: no test samples — skipping", sym)
                continue

            y_test = splits["test"]["targets"]
            all_y_true.append(y_test)

            # --- Experiment 1: Zero-shot ---
            logger.info("[{}] Running Chronos zero-shot …", sym)
            preds_zs = chronos.zero_shot_predict(
                splits["test"]["close_windows"],
                splits["test"]["last_close"],
            )
            metrics_zs = compute_all(y_test, preds_zs)
            metrics_zs.update({
                "Experiment": "Chronos Zero-Shot",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_zs)
            all_preds["Chronos Zero-Shot"].append(preds_zs)

            # --- Experiment 2: Linear-probe (market + tabular) ---
            logger.info("[{}] Running Chronos linear-probe (+tabular market) …", sym)
            preds_lp = chronos.linear_probe_predict(
                splits["train"]["close_windows"], splits["train"]["targets"],
                splits["val"]["close_windows"], splits["val"]["targets"],
                splits["test"]["close_windows"],
                tabular_train=splits["train"].get("market_tabular"),
                tabular_val=splits["val"].get("market_tabular"),
                tabular_test=splits["test"].get("market_tabular"),
            )
            metrics_lp = compute_all(y_test, preds_lp)
            metrics_lp.update({
                "Experiment": "Chronos Linear-Probe",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lp)
            all_preds["Chronos Linear-Probe"].append(preds_lp)

            # --- Experiment 3: CMTF fusion (+tabular market) ---
            logger.info("[{}] Running Chronos + CMTF fusion (+tabular market) …", sym)
            tab_dim = int(splits["train"].get("market_tabular", np.zeros((1, 0))).shape[1])
            cmtf = ChronosCMTFPredictor(chronos, tabular_dim=tab_dim, device=device)
            cmtf.fit(
                splits["train"]["close_windows"], splits["train"]["news_embs"],
                splits["train"]["targets"],
                splits["val"]["close_windows"], splits["val"]["news_embs"],
                splits["val"]["targets"],
                tabular_train=splits["train"].get("market_tabular"),
                tabular_val=splits["val"].get("market_tabular"),
            )
            preds_cmtf = cmtf.predict(
                splits["test"]["close_windows"],
                splits["test"]["news_embs"],
                tabular_test=splits["test"].get("market_tabular"),
            )
            metrics_cmtf = compute_all(y_test, preds_cmtf)
            metrics_cmtf.update({
                "Experiment": "Chronos + CMTF",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_cmtf)
            all_preds["Chronos + CMTF"].append(preds_cmtf)

            # Per-symbol prediction plot
            plot_predictions(
                y_test,
                {"Zero-Shot": preds_zs, "Linear-Probe": preds_lp, "CMTF": preds_cmtf},
                f"Chronos Predictions — {sym} ({target_h}D target)",
                FIGURES_DIR / f"predictions_{sym}_{target_h}d.png",
            )

        # Aggregate results
        results_df = pd.DataFrame(all_results)
        for exp_name in results_df["Experiment"].unique():
            exp_rows = results_df[results_df["Experiment"] == exp_name]
            avg = {
                "Experiment": exp_name,
                "Symbol": "AVG",
                "TargetHorizonD": target_h,
                "MAE": exp_rows["MAE"].mean(),
                "RMSE": exp_rows["RMSE"].mean(),
                "DA%": exp_rows["DA%"].mean(),
                "Sharpe": exp_rows["Sharpe"].mean(),
                "IC": exp_rows["IC"].mean(),
            }
            all_results.append(avg)

        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 76)
        print(f"  CHRONOS BENCHMARK RESULTS — TARGET HORIZON {target_h}D")
        print("=" * 76)
        col_order = ["Experiment", "Symbol", "TargetHorizonD", "MAE", "RMSE", "DA%", "Sharpe", "IC"]
        results_df = results_df[col_order]
        print(results_df.to_string(index=False, float_format="{:.4f}".format))
        print("=" * 76 + "\n")

        suffix = f"{target_h}d"
        csv_path = RESULTS_DIR / f"chronos_benchmark_{suffix}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info("Results saved → {}", csv_path)

        avg_df = results_df[results_df["Symbol"] == "AVG"]
        plot_ablation(avg_df, FIGURES_DIR / f"ablation_chronos_{suffix}.png")
        plot_per_symbol(results_df, FIGURES_DIR / f"per_symbol_heatmap_{suffix}.png")

        if all_y_true:
            combined_y = np.concatenate(all_y_true)
            combined_preds = {
                k: np.concatenate(v) for k, v in all_preds.items() if v
            }
            plot_predictions(
                combined_y,
                combined_preds,
                f"Chronos Predictions — All Symbols Combined ({target_h}D target)",
                FIGURES_DIR / f"predictions_combined_{suffix}.png",
            )

    logger.info("═══ Benchmark complete ═══")


if __name__ == "__main__":
    main()
