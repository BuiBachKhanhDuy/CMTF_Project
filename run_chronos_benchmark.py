"""Run Chronos benchmark: Market-Only vs Cross-Modal Temporal Fusion.

Usage:
    python run_chronos_benchmark.py                  # full run (use caches)
    python run_chronos_benchmark.py --stage data     # rebuild dataset only
    python run_chronos_benchmark.py --stage predict  # rerun models, reuse data
    python run_chronos_benchmark.py --stage cmtf     # retrain CMTF only
    python run_chronos_benchmark.py --stage hpo      # run Optuna HPO + retrain CMTF
    python run_chronos_benchmark.py --stage plot     # regenerate plots from CSVs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


def set_global_seed(seed: int) -> None:
    """Pin every RNG source for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to {}", seed)

# Project imports
from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.benchmark.metrics import compute_all
from src.benchmark.chronos_market import ChronosMarketPredictor
from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_EMB_DIR = Path("cache/chronos_emb")
CACHE_PRED_DIR = Path("cache/predictions")
CACHE_CMTF_DIR = Path("cache/cmtf_models")
CACHE_HPO_DIR = Path("cache/optuna")


def _split_hash(splits: dict, sym: str, horizon: int) -> str:
    """Short hash of split sizes for cache key."""
    h = hashlib.sha256()
    h.update(f"{sym}_{horizon}".encode())
    for name in ("train", "val", "test"):
        n = len(splits[name]["targets"])
        h.update(f"{name}={n}".encode())
    return h.hexdigest()[:12]


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_npy(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path)
    return None


def _run_optuna_hpo(
    chronos: ChronosMarketPredictor,
    all_symbol_splits: dict[str, dict[str, dict[str, np.ndarray]]],
    all_symbol_embs: dict[str, dict[str, np.ndarray]],
    target_h: int,
    use_tabular: bool,
    device: str,
    n_trials: int = 30,
    seq_len: int = 30,
) -> dict:
    """Run Optuna HPO across all symbols jointly.

    Objective: average validation loss over all symbols (single seed=42).
    Searched hyperparameters:
        fusion_dim ∈ {32, 64, 128}
        lr ∈ [1e-4, 1e-2] (log-uniform)
        bce_weight ∈ [0.1, 0.3]
        dropout ∈ [0.1, 0.5]
    Fixed: n_heads = 1 (FiLM/GRN does not use attention heads)

    Returns:
        Best hyperparameter dict.
    """
    import optuna
    import json

    CACHE_HPO_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_HPO_DIR / f"best_params_{target_h}d.json"

    # Check for cached result
    if cache_file.exists():
        with open(cache_file) as f:
            best = json.load(f)
        logger.info("Optuna HPO loaded from cache: {}", best)
        return best

    def objective(trial: optuna.Trial) -> float:
        fusion_dim = trial.suggest_categorical("fusion_dim", [32, 64, 128])
        # n_heads is not used by FiLM/GRN architecture — fixed at 1
        n_heads = 1
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        bce_weight = trial.suggest_float("bce_weight", 0.1, 0.3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        total_val_loss = 0.0
        n_symbols = 0

        for sym, splits in all_symbol_splits.items():
            embs = all_symbol_embs[sym]
            tab_dim = (
                int(splits["train"].get("market_tabular", np.zeros((1, 0))).shape[1])
                if use_tabular else 0
            )
            cmtf = ChronosCMTFPredictor(
                chronos, tabular_dim=tab_dim, device=device,
                fusion_dim=fusion_dim, n_heads=n_heads, dropout=dropout,
                bce_weight=bce_weight, seq_len=seq_len,
            )
            history = cmtf.fit(
                splits["train"]["close_windows"], splits["train"]["news_embs"],
                splits["train"]["targets"],
                splits["val"]["close_windows"], splits["val"]["news_embs"],
                splits["val"]["targets"],
                tabular_train=splits["train"].get("market_tabular") if use_tabular else None,
                tabular_val=splits["val"].get("market_tabular") if use_tabular else None,
                seed=42, lr=lr, epochs=50, patience=15, batch_size=32,
                precomputed_emb_train=embs["train"],
                precomputed_emb_val=embs["val"],
            )
            total_val_loss += min(history["val_loss"])
            n_symbols += 1

        return total_val_loss / max(n_symbols, 1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info("Optuna best params ({}D): {}", target_h, best)
    logger.info("Optuna best val loss: {:.6f}", study.best_value)

    # Cache result
    with open(cache_file, "w") as f:
        json.dump(best, f, indent=2)

    return best


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

    # Apply purge buffer using actual trading days (not calendar days)
    # so that H trading days are excluded before each boundary.
    sorted_times = np.sort(np.unique(times))

    def _trading_day_offset(boundary: pd.Timestamp, n: int) -> pd.Timestamp:
        """Return the timestamp n *trading* days before boundary."""
        idx = np.searchsorted(sorted_times, boundary, side="right") - 1
        idx = max(idx - n, 0)
        return pd.Timestamp(sorted_times[idx])

    train_end_purged = _trading_day_offset(train_end_ts, target_horizon_days)
    val_end_purged = _trading_day_offset(val_end_ts, target_horizon_days)

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
            if np.isnan(v):
                vals.append(0.0)
                ax.text(i, 0, "N/A", ha="center", va="bottom", fontsize=9, fontstyle="italic")
                continue
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
    parser = argparse.ArgumentParser(description="Chronos benchmark")
    parser.add_argument(
        "--stage", choices=["data", "predict", "cmtf", "hpo", "plot"],
        default=None,
        help="Run only a specific stage (default: full run using caches)",
    )
    args = parser.parse_args()
    stage = args.stage  # None = full run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Pipeline config -----
    config = {
        "seed": 42,
        "rebuild_data": False,        # Use cached dataset if available
        "symbols": ["VCB", "BID"],
        "start": "2022-01-01",
        "end": "2026-03-31",
        "interval": "1D",
        "ohlcv_source": "KBS",
        "news_source": "web",
        "news_sources": ("vnexpress", "cafef_banking", "vietstock"),
        "news_use_cache": True,       # Reuse cached news articles
        "news_export_trace": True,
        "news_similarity_threshold": 85.0,
        "log_news_coverage": True,
        "sequence_len": 30,
        "horizon": 1,
        "target_horizons_days": [1, 5, 20],
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        "use_tabular_market_features": True,
        "allow_unfair_baselines": True,
    }

    # Override for --stage data: force rebuild
    if stage == "data":
        config["rebuild_data"] = True

    # ----- 1. Set global seed -----
    set_global_seed(config["seed"])

    # ----- 2. Fetch raw OHLCV (cached) for Chronos -----
    if stage == "plot":
        # Plot-only mode: skip everything, just regenerate from CSVs
        horizons = [int(h) for h in config.get("target_horizons_days", [1])]
        for target_h in horizons:
            suffix = f"{target_h}d"
            csv_path = RESULTS_DIR / f"chronos_benchmark_{suffix}.csv"
            if not csv_path.exists():
                logger.warning("No results CSV for {}D — skipping plot", target_h)
                continue
            results_df = pd.read_csv(csv_path)
            avg_df = results_df[results_df["Symbol"] == "AVG"]
            plot_ablation(avg_df, FIGURES_DIR / f"ablation_chronos_{suffix}.png")
            plot_per_symbol(results_df, FIGURES_DIR / f"per_symbol_heatmap_{suffix}.png")
            logger.info("Plots regenerated for {}D", target_h)
        logger.info("═══ Plot-only mode complete ═══")
        return

    logger.info("═══ Fetching raw OHLCV for Chronos ═══")
    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(
        config["symbols"], config["start"], config["end"],
    )

    # ----- 3. Load Chronos (once, shared) -----
    if stage == "data":
        # Data-only mode: just build dataset, no model needed
        horizons = [int(h) for h in config.get("target_horizons_days", [1])]
        for target_h in horizons:
            run_cfg = {**config, "target_horizon_days": target_h}
            logger.info("Building dataset for target horizon {}D", target_h)
            run_pipeline(run_cfg)
        logger.info("═══ Data-only mode complete ═══")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("═══ Loading Chronos (device={}) ═══", device)
    chronos = ChronosMarketPredictor(device=device)
    use_tabular_market_features = bool(config.get("use_tabular_market_features", False))
    allow_unfair_baselines = bool(config.get("allow_unfair_baselines", False))
    if use_tabular_market_features and not allow_unfair_baselines:
        raise ValueError(
            "Tabular market features for trainable baselines are disabled by default for fairness. "
            "Set allow_unfair_baselines=True only for explicit ablation runs."
        )
    if use_tabular_market_features:
        logger.warning(
            "Benchmark fairness mode OFF: trainable models receive extra tabular market features"
        )
    else:
        logger.info(
            "Benchmark fairness mode ON: all models share the same market input via Chronos close windows"
        )

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

        # ---- Phase 1: splits, ZS, LP, embeddings ----
        _all_splits: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        _all_embs: dict[str, dict[str, np.ndarray]] = {}
        _all_sh: dict[str, str] = {}
        _sym_order: list[str] = []  # track order for indexing

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

            # News coverage summary — check the *last* bar in each window
            # (the current bar), not the entire 30-bar lookback window.
            if config.get("log_news_coverage", True) and "news_embs" in data:
                total_bars = len(data["times"])
                embs = data["news_embs"]  # (N, seq_len, 768)
                if embs.ndim == 3:
                    # Per-bar: check only the last position in each window
                    last_bar_norm = np.linalg.norm(embs[:, -1, :], axis=-1)  # (N,)
                else:
                    last_bar_norm = np.linalg.norm(embs, axis=-1)
                has_news = last_bar_norm > 0
                bars_with_news = int(has_news.sum())
                coverage_pct = bars_with_news / total_bars * 100 if total_bars > 0 else 0
                logger.info(
                    "[{}] News coverage: {}/{} bars ({:.1f}%)",
                    sym, bars_with_news, total_bars, coverage_pct,
                )

            if len(splits["test"]["targets"]) == 0:
                logger.warning("{}: no test samples — skipping", sym)
                continue

            y_test = splits["test"]["targets"]
            all_y_true.append(y_test)
            _sym_order.append(sym)

            sh = _split_hash(splits, sym, target_h)
            _all_splits[sym] = splits
            _all_sh[sym] = sh

            # --- Experiment 1: Zero-shot ---
            zs_cache = CACHE_PRED_DIR / f"zs_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_zs := _load_npy(zs_cache)) is not None:
                logger.info("[{}] Zero-shot loaded from cache", sym)
                preds_zs = cached_zs
            else:
                logger.info("[{}] Running Chronos zero-shot (horizon={}) …", sym, target_h)
                preds_zs = chronos.zero_shot_predict(
                    splits["test"]["close_windows"],
                    splits["test"]["last_close"],
                    seed=config["seed"],
                    horizon=target_h,
                )
                _save_npy(zs_cache, preds_zs)
            metrics_zs = compute_all(y_test, preds_zs, horizon=target_h)
            metrics_zs.update({
                "Experiment": "Chronos Zero-Shot",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_zs)
            all_preds["Chronos Zero-Shot"].append(preds_zs)

            # --- Experiment 2: Linear-probe ---
            lp_cache = CACHE_PRED_DIR / f"lp_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_lp := _load_npy(lp_cache)) is not None:
                logger.info("[{}] Linear-probe loaded from cache", sym)
                preds_lp = cached_lp
            else:
                logger.info("[{}] Running Chronos linear-probe …", sym)
                preds_lp = chronos.linear_probe_predict(
                    splits["train"]["close_windows"], splits["train"]["targets"],
                    splits["val"]["close_windows"], splits["val"]["targets"],
                    splits["test"]["close_windows"],
                    tabular_train=splits["train"].get("market_tabular") if use_tabular_market_features else None,
                    tabular_val=splits["val"].get("market_tabular") if use_tabular_market_features else None,
                    tabular_test=splits["test"].get("market_tabular") if use_tabular_market_features else None,
                    allow_tabular_features=use_tabular_market_features,
                )
                _save_npy(lp_cache, preds_lp)
            metrics_lp = compute_all(y_test, preds_lp, horizon=target_h)
            metrics_lp.update({
                "Experiment": "Chronos Linear-Probe",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lp)
            all_preds["Chronos Linear-Probe"].append(preds_lp)

            # --- Pre-compute Chronos embeddings (cached) ---
            _emb_cache_base = CACHE_EMB_DIR / f"{sym}_{target_h}d_{sh}"
            _emb_cache_base.parent.mkdir(parents=True, exist_ok=True)
            _emb_train = _load_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_train.npy"))
            _emb_val = _load_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_val.npy"))
            _emb_test = _load_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_test.npy"))
            if _emb_train is None or _emb_val is None or _emb_test is None:
                logger.info("[{}] Computing Chronos embeddings …", sym)
                _emb_train = chronos.get_embeddings(splits["train"]["close_windows"])
                _emb_val = chronos.get_embeddings(splits["val"]["close_windows"])
                _emb_test = chronos.get_embeddings(splits["test"]["close_windows"])
                _save_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_train.npy"), _emb_train)
                _save_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_val.npy"), _emb_val)
                _save_npy(_emb_cache_base.with_name(f"{_emb_cache_base.name}_test.npy"), _emb_test)
            else:
                logger.info("[{}] Chronos embeddings loaded from cache", sym)
            _all_embs[sym] = {"train": _emb_train, "val": _emb_val, "test": _emb_test}

        # ---- Optuna HPO (once per horizon, across all symbols) ----
        if stage == "hpo" or (stage in (None, "cmtf") and not (CACHE_HPO_DIR / f"best_params_{target_h}d.json").exists()):
            logger.info("═══ Running Optuna HPO for {}D ═══", target_h)
            hpo_params = _run_optuna_hpo(
                chronos, _all_splits, _all_embs, target_h,
                use_tabular=use_tabular_market_features,
                device=device, n_trials=15, seq_len=run_cfg["sequence_len"],
            )
        elif (CACHE_HPO_DIR / f"best_params_{target_h}d.json").exists():
            import json
            with open(CACHE_HPO_DIR / f"best_params_{target_h}d.json") as f:
                hpo_params = json.load(f)
            logger.info("Using cached HPO params ({}D): {}", target_h, hpo_params)
        else:
            hpo_params = {}

        if stage == "hpo":
            logger.info("═══ HPO-only mode complete for {}D ═══", target_h)
            continue

        # ---- Phase 2: CMTF ensemble with HPO params ----
        _ensemble_seeds = [42, 123, 456]
        for sym in _sym_order:
            splits = _all_splits[sym]
            sh = _all_sh[sym]
            sym_embs = _all_embs[sym]
            y_test = splits["test"]["targets"]

            logger.info("[{}] Running Chronos + CMTF fusion (ensemble × {} seeds) …", sym, len(_ensemble_seeds))
            tab_dim = (
                int(splits["train"].get("market_tabular", np.zeros((1, 0))).shape[1])
                if use_tabular_market_features else 0
            )

            # CMTF hyperparameters (HPO or defaults)
            cmtf_fusion_dim = hpo_params.get("fusion_dim", 128)
            cmtf_n_heads = 1  # FiLM/GRN architecture does not use attention heads
            cmtf_dropout = hpo_params.get("dropout", 0.2)
            cmtf_bce_weight = hpo_params.get("bce_weight", 0.2)
            cmtf_lr = hpo_params.get("lr", 5e-4)

            # Hash HPO params into checkpoint path so architecture changes
            # (e.g. different fusion_dim) invalidate stale caches.
            import json as _json
            _hpo_hash = hashlib.md5(
                _json.dumps(hpo_params, sort_keys=True).encode()
            ).hexdigest()[:8]

            _ensemble_preds: list[np.ndarray] = []
            for _seed in _ensemble_seeds:
                _ckpt_path = CACHE_CMTF_DIR / f"{sym}_{target_h}d_seed{_seed}_{_hpo_hash}_{sh}.pt"
                _ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                _cmtf = ChronosCMTFPredictor(
                    chronos, tabular_dim=tab_dim, device=device,
                    fusion_dim=cmtf_fusion_dim, n_heads=cmtf_n_heads,
                    dropout=cmtf_dropout, bce_weight=cmtf_bce_weight,
                    seq_len=run_cfg["sequence_len"],
                )

                if _ckpt_path.exists() and stage not in ("cmtf", "hpo"):
                    # Load cached checkpoint
                    _cmtf.load_checkpoint(
                        torch.load(_ckpt_path, map_location=device, weights_only=False)
                    )
                    logger.info("[{}] CMTF seed {} loaded from checkpoint", sym, _seed)
                else:
                    _cmtf.fit(
                        splits["train"]["close_windows"], splits["train"]["news_embs"],
                        splits["train"]["targets"],
                        splits["val"]["close_windows"], splits["val"]["news_embs"],
                        splits["val"]["targets"],
                        tabular_train=splits["train"].get("market_tabular") if use_tabular_market_features else None,
                        tabular_val=splits["val"].get("market_tabular") if use_tabular_market_features else None,
                        seed=_seed, lr=cmtf_lr, patience=40,
                        precomputed_emb_train=sym_embs["train"],
                        precomputed_emb_val=sym_embs["val"],
                    )
                    torch.save(_cmtf.get_checkpoint(), _ckpt_path)
                    logger.info("[{}] CMTF seed {} checkpoint saved", sym, _seed)

                _ensemble_preds.append(_cmtf.predict(
                    splits["test"]["close_windows"],
                    splits["test"]["news_embs"],
                    tabular_test=splits["test"].get("market_tabular") if use_tabular_market_features else None,
                    precomputed_emb_test=sym_embs["test"],
                ))
            preds_cmtf = np.mean(_ensemble_preds, axis=0)
            metrics_cmtf = compute_all(y_test, preds_cmtf, horizon=target_h)
            metrics_cmtf.update({
                "Experiment": "Chronos + CMTF",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_cmtf)
            all_preds["Chronos + CMTF"].append(preds_cmtf)

            # Per-symbol prediction plot (use index from _sym_order)
            _sym_idx = _sym_order.index(sym)
            plot_predictions(
                y_test,
                {
                    "Zero-Shot": all_preds["Chronos Zero-Shot"][_sym_idx],
                    "Linear-Probe": all_preds["Chronos Linear-Probe"][_sym_idx],
                    "CMTF": preds_cmtf,
                },
                f"Chronos Predictions — {sym} ({target_h}D target)",
                FIGURES_DIR / f"predictions_{sym}_{target_h}d.png",
            )

        # Aggregate results — pooled metrics across all symbols
        results_df = pd.DataFrame(all_results)
        for exp_name in results_df["Experiment"].unique():
            # Concatenate per-symbol predictions for pooled computation
            exp_y_parts: list[np.ndarray] = []
            exp_p_parts: list[np.ndarray] = []
            for sym_name in per_symbol:
                sym_rows = results_df[
                    (results_df["Experiment"] == exp_name) & (results_df["Symbol"] == sym_name)
                ]
                if sym_rows.empty:
                    continue
                idx = list(per_symbol.keys()).index(sym_name)
                exp_y_parts.append(all_y_true[idx])
                exp_p_parts.append(all_preds[exp_name][idx])
            if exp_y_parts:
                pooled_y = np.concatenate(exp_y_parts)
                pooled_p = np.concatenate(exp_p_parts)
                avg = compute_all(pooled_y, pooled_p, horizon=target_h)
            else:
                avg = {"MAE": 0.0, "RMSE": 0.0, "DA%": 0.0, "Sharpe": 0.0, "IC": 0.0, "Prec": 0.0, "Rec": 0.0, "F1": 0.0}
            avg.update({
                "Experiment": exp_name,
                "Symbol": "AVG",
                "TargetHorizonD": target_h,
            })
            all_results.append(avg)

        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 76)
        print(f"  CHRONOS BENCHMARK RESULTS — TARGET HORIZON {target_h}D")
        print("=" * 76)
        col_order = ["Experiment", "Symbol", "TargetHorizonD", "MAE", "RMSE", "DA%", "Sharpe", "IC", "Prec", "Rec", "F1"]
        results_df = results_df[col_order]

        def _fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        print(results_df.to_string(index=False, formatters={c: _fmt for c in col_order}))
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
