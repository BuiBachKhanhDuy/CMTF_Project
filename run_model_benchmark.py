"""Run model benchmark: all baselines.

Trains and evaluates:
    - Chronos Zero-Shot
    - LSTM Baseline (raw sequence)
    - CNN-LSTM (raw sequence)
    - Random Forest Baseline (engineered summary features)
    - Linear Summary Baseline (engineered summary features)
    - MLP Summary Baseline (engineered summary features)
    - LSTM Hybrid (raw sequence + engineered summary features)
    - CNN-LSTM Hybrid (raw sequence + engineered summary features)

Usage:
    python run_model_benchmark.py                  # full run (use caches)
    python run_model_benchmark.py --stage data     # rebuild dataset only
    python run_model_benchmark.py --stage predict  # rerun models, reuse data
    python run_model_benchmark.py --stage hpo      # run Optuna HPO for baselines
    python run_model_benchmark.py --stage plot     # regenerate plots from CSVs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.benchmark.metrics import compute_all, compute_composite_metrics
from src.benchmark.chronos_encoder import ChronosMarketPredictor
from src.benchmark.baseline_models import (
    CNNLSTMPredictor,
    CNNLSTMHybridPredictor,
    LSTMPredictor,
    LSTMHybridPredictor,
    LinearSummaryRegressor_Wrapper,
    MLPSummaryPredictor,
    RandomForestRegressor_Wrapper,
    extract_market_summary_features,
)
from src.benchmark.gpt4ts_encoder import GPT4TSPredictor, GPT4TSHybridPredictor
from src.benchmark.baseline_hpo import (
    get_default_baseline_hpo_params,
    load_or_run_baseline_hpo,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_PRED_DIR = Path("cache/predictions")
CACHE_MODEL_DIR = Path("cache/cmtf_models")
CACHE_HPO_DIR = Path("cache/optuna")


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

def _use_prediction_cache(stage: str | None) -> bool:
    return stage is None or stage == "plot"

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


# ======================================================================
# Data extraction helpers
# ======================================================================

def extract_per_symbol_data(
    dataset,
    raw_ohlcv: dict[str, pd.DataFrame],
    seq_len: int = 30,
    target_horizon_days: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-symbol arrays of close windows, market windows, news embeddings, targets."""
    df = dataset.df.copy()

    if "time" not in df.columns:
        if df.index.name == "time":
            df = df.reset_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "time"})
        else:
            raise ValueError("dataset.df must contain a 'time' column or DatetimeIndex")

    if "symbol" not in df.columns:
        raise ValueError("dataset.df must contain a 'symbol' column")

    stray_cols = [c for c in ("index", "level_0", "Unnamed: 0") if c in df.columns]
    if stray_cols:
        df = df.drop(columns=stray_cols)

    symbols = df["symbol"].unique()
    result: dict[str, dict[str, np.ndarray]] = {}

    dataset_market_cols = list(getattr(dataset, "market_cols", []))
    filtered_market_cols = [
        c for c in dataset_market_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not filtered_market_cols:
        raise ValueError("dataset.market_cols is empty or invalid after schema filtering")

    logger.info("Using {} market-only features for benchmark", len(filtered_market_cols))

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("time").reset_index(drop=True)
        n = len(sym_df)

        if sym not in raw_ohlcv:
            logger.warning("{} missing from raw_ohlcv — skipping", sym)
            continue

        raw_df = raw_ohlcv[sym][["close"]].rename(columns={"close": "raw_close"})
        sym_df = sym_df.merge(raw_df, left_on="time", right_index=True, how="left")

        missing = int(sym_df["raw_close"].isna().sum())
        assert missing == 0, f"{sym}: {missing} timestamps missing after merge with raw OHLCV"

        sym_df = sym_df.reset_index(drop=True)
        n = len(sym_df)
        raw_c = sym_df["raw_close"].to_numpy(dtype=np.float64)

        if n < seq_len + 1:
            logger.warning("{} has only {} rows, need {} — skipping", sym, n, seq_len + 1)
            continue

        news_col = "news_hybrid_emb" if "news_hybrid_emb" in sym_df.columns else "news_emb"
        if news_col not in sym_df.columns:
            raise ValueError(f"{sym}: missing news embedding column {news_col}")

        news_list = sym_df[news_col].tolist()
        news_arr = np.stack(news_list).astype(np.float32)
        has_news_arr = sym_df["has_news"].astype(bool).to_numpy(copy=True)
        market_values = sym_df[filtered_market_cols].astype(np.float32).to_numpy(copy=True)

        target_col_name = f"fwd_ret_{int(target_horizon_days)}d"
        if target_col_name not in sym_df.columns:
            logger.warning("{} missing target column {} — skipping", sym, target_col_name)
            continue
        target_col = sym_df[target_col_name].to_numpy(dtype=np.float32)

        close_windows = []
        market_windows = []
        last_closes = []
        market_tabs = []
        news_embs = []
        news_masks = []
        targets = []
        times = []

        valid_start = seq_len - 1
        valid_end = n

        for i in range(valid_start, valid_end):
            window = raw_c[i - seq_len + 1 : i + 1]
            market_window = market_values[i - seq_len + 1 : i + 1]
            news_window = news_arr[i - seq_len + 1 : i + 1]
            news_mask_window = ~has_news_arr[i - seq_len + 1 : i + 1]

            target_val = target_col[i]
            if np.isnan(target_val):
                continue

            close_windows.append(window)
            market_windows.append(market_window)
            last_closes.append(raw_c[i])
            market_tabs.append(market_values[i])
            news_embs.append(news_window)
            news_masks.append(news_mask_window)
            targets.append(target_val)
            times.append(sym_df.iloc[i]["time"])

        n_samples = len(targets)
        if n_samples == 0:
            logger.warning("{} has zero valid samples after windowing — skipping", sym)
            continue

        close_windows_arr = np.asarray(close_windows, dtype=np.float64)
        market_windows_arr = np.asarray(market_windows, dtype=np.float32)
        news_embs_arr = np.asarray(news_embs, dtype=np.float32)

        assert close_windows_arr.ndim == 2
        assert market_windows_arr.ndim == 3
        assert news_embs_arr.ndim == 3
        assert close_windows_arr.shape[1] == seq_len
        assert market_windows_arr.shape[1] == seq_len
        assert news_embs_arr.shape[1] == seq_len

        result[sym] = {
            "close_windows": close_windows_arr,
            "market_windows": market_windows_arr,
            "news_embs": news_embs_arr,
            "last_close": np.asarray(last_closes, dtype=np.float64),
            "market_tabular": np.asarray(market_tabs, dtype=np.float32),
            "news_masks": np.asarray(news_masks, dtype=bool),
            "targets": np.asarray(targets, dtype=np.float32),
            "times": np.asarray(times),
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
    """Walk-forward split arrays by date with horizon-aware purge buffer."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    sorted_times = np.sort(np.unique(times))

    def _trading_day_offset(boundary: pd.Timestamp, n: int) -> pd.Timestamp:
        idx = np.searchsorted(sorted_times, boundary, side="right") - 1
        idx = max(idx - n, 0)
        return pd.Timestamp(sorted_times[idx])

    train_end_purged = _trading_day_offset(train_end_ts, target_horizon_days)
    val_end_purged = _trading_day_offset(val_end_ts, target_horizon_days)

    train_mask = times <= train_end_purged
    val_mask = (times > train_end_ts) & (times <= val_end_purged)
    test_mask = times > val_end_ts

    splits: dict[str, dict[str, np.ndarray]] = {}
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


def _impute_splits_key(
    splits: dict[str, dict[str, np.ndarray]],
    key: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Impute NaNs in splits[*][key] using train-only feature means."""
    if key not in splits.get("train", {}):
        return splits

    train_arr = splits["train"][key]
    if train_arr.size == 0:
        return splits

    reduce_axes = tuple(range(train_arr.ndim - 1))
    feature_means = np.nanmean(train_arr, axis=reduce_axes)
    feature_means = np.where(np.isnan(feature_means), 0.0, feature_means).astype(np.float32)
    fill_shape = (1,) * (train_arr.ndim - 1) + (feature_means.shape[0],)
    fill_values = feature_means.reshape(fill_shape)

    for split_name in ("train", "val", "test"):
        arr = splits[split_name].get(key)
        if arr is None or arr.size == 0:
            continue
        splits[split_name][key] = np.where(np.isnan(arr), fill_values, arr).astype(np.float32)

    return splits


def impute_tabular_splits(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    return _impute_splits_key(splits, "market_tabular")


def impute_market_window_splits(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    return _impute_splits_key(splits, "market_windows")


# ======================================================================
# Visualization
# ======================================================================

def plot_ablation(results_df: pd.DataFrame, save_path: Path) -> None:
    metrics = ["MAE", "RMSE", "DA%", "Sharpe", "IC"]
    experiments = results_df["Experiment"].unique()
    n_exp = len(experiments)

    colors = [
        "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
        "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
    ]

    label_map = {
        "Chronos Zero-Shot": "Zero-Shot",
        "LSTM Baseline": "LSTM",
        "LSTM Hybrid": "LSTM+Tab",
        "Random Forest Baseline": "RF",
        "Linear Summary Baseline": "Linear+Feat",
        "MLP Summary Baseline": "MLP+Feat",
        "CNN-LSTM": "CNN-LSTM",
        "CNN-LSTM Hybrid": "CNN-LSTM+Tab",
        "GPT4TS Baseline": "GPT4TS",
        "GPT4TS Hybrid": "GPT4TS+Tab",
    }
    short_labels = [label_map.get(exp, exp[:15]) for exp in experiments]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

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
            ax.bar(i, v, color=colors[i % len(colors)], width=0.6)
            va = "bottom" if v >= 0 else "top"
            ax.text(i, v, f"{v:.4f}", ha="center", va=va, fontsize=8, fontweight="bold")

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

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[i % len(colors)], label=l) for i, l in enumerate(short_labels)]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(n_exp, 6),
        fontsize=10,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle("Baseline Benchmark (AVG)", fontsize=14, fontweight="bold", y=1.10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation chart → {}", save_path)


def plot_per_symbol(results_df: pd.DataFrame, save_path: Path) -> None:
    metrics = ["MAE", "RMSE", "DA%", "Sharpe", "IC"]
    higher_better = {"MAE": False, "RMSE": False, "DA%": True, "Sharpe": True, "IC": True}

    df = results_df[results_df["Symbol"] != "AVG"].copy()
    symbols = df["Symbol"].unique()
    experiments = df["Experiment"].unique()

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

    data_arr = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(12, max(4, len(row_labels) * 0.5 + 1)))

    norm_data = np.zeros_like(data_arr)
    for j, metric in enumerate(metrics):
        col = data_arr[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-12:
            normalized = (col - col_min) / (col_max - col_min)
        else:
            normalized = np.full_like(col, 0.5)
        if not higher_better[metric]:
            normalized = 1.0 - normalized
        norm_data[:, j] = normalized

    ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            val = data_arr[i, j]
            if metrics[j] == "DA%":
                text = f"{val:.1f}%"
            elif metrics[j] in ("MAE", "RMSE"):
                text = f"{val:.4f}"
            else:
                text = f"{val:.3f}"
            brightness = norm_data[i, j]
            color = "white" if brightness < 0.35 or brightness > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    n_exp_per_sym = len(experiments)
    for k in range(1, len(symbols)):
        ax.axhline(y=k * n_exp_per_sym - 0.5, color="black", linewidth=1.5)

    ax.set_title("Per-Symbol Benchmark Heatmap (green = best)", fontsize=13, fontweight="bold", pad=12)
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
        "--stage",
        choices=["data", "predict", "hpo", "plot"],
        default=None,
        help="Run only a specific stage (default: full run using caches)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional list of target horizons to run, e.g. --horizons 20",
    )
    args = parser.parse_args()
    stage = args.stage

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PRED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_HPO_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": 42,
        "rebuild_data": False,
        "symbols": ["VCB", "BID"],
        "start": "2022-01-01",
        "end": "2026-03-31",
        "interval": "1D",
        "ohlcv_source": "KBS",
        "news_source": "web",
        "news_sources": ("vnexpress", "cafef_banking", "vietstock", "google_news"),
        "news_use_cache": True,
        "news_export_trace": True,
        "news_sentiment_enabled": True,
        "news_sentiment_device": "cpu",
        "news_sentiment_export_trace": True,
        "phase2_output_dir": "outputs/phase2/latest",
        "news_similarity_threshold": 85.0,
        "log_news_coverage": True,
        "sequence_len": 30,
        "horizon": 1,
        "target_horizons_days": [1, 5, 20],
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        "stability_selection_enabled": False,
        "stability_corr_threshold": 0.95,
        "stability_lasso_alpha": 0.001,
        "stability_n_folds": 5,
        "stability_threshold": 0.6,
        "stability_min_train_rows": 120,
        "use_tabular_market_features": True,
    }

    if stage == "data":
        config["rebuild_data"] = True
    if args.horizons:
        config["target_horizons_days"] = [int(h) for h in args.horizons]

    set_global_seed(config["seed"])

    if stage == "plot":
        horizons = [int(h) for h in config.get("target_horizons_days", [1])]
        for target_h in horizons:
            suffix = f"{target_h}d"
            csv_path = RESULTS_DIR / f"chronos_benchmark_{suffix}.csv"
            if not csv_path.exists():
                logger.warning("No results CSV for {}D — skipping plot", target_h)
                continue
            results_df = pd.read_csv(csv_path)
            avg_df = results_df[results_df["Symbol"] == "AVG"]
            plot_ablation(avg_df, FIGURES_DIR / f"model_{suffix}.png")
            plot_per_symbol(results_df, FIGURES_DIR / f"per_symbol_heatmap_{suffix}.png")
            logger.info("Plots regenerated for {}D", target_h)
        logger.info("═══ Plot-only mode complete ═══")
        return

    if stage == "data":
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
    if use_tabular_market_features:
        logger.info("Active trainable models share market-only engineered features; zero-shot remains close-only anchor")
    else:
        logger.warning("Tabular market features disabled; trainable baselines fall back to close-only inputs")

    logger.info("═══ Fetching raw OHLCV for Chronos ═══")
    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(config["symbols"], config["start"], config["end"])

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

        logger.info("Extracting 365-day close windows for Chronos zero-shot ({}D)", target_h)
        per_symbol_365 = extract_per_symbol_data(
            dataset,
            raw_ohlcv,
            seq_len=365,
            target_horizon_days=target_h,
        )

        all_results: list[dict] = []
        all_preds: dict[str, list[np.ndarray]] = {
            "Chronos Zero-Shot": [],
            "LSTM Baseline": [],
            "LSTM Hybrid": [],
            "Random Forest Baseline": [],
            "Linear Summary Baseline": [],
            "MLP Summary Baseline": [],
            "CNN-LSTM": [],
            "CNN-LSTM Hybrid": [],
            "GPT4TS Baseline": [],
            "GPT4TS Hybrid": [],
        }
        all_y_true: list[np.ndarray] = []

        _all_splits: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        _all_sh: dict[str, str] = {}
        _sym_order: list[str] = []

        for sym, data in per_symbol.items():
            logger.info("━━━ Benchmark: {} ({}D) ━━━", sym, target_h)

            splits = split_by_date(
                {k: v for k, v in data.items() if k != "times"},
                data["times"],
                run_cfg["train_end"],
                run_cfg["val_end"],
                target_horizon_days=target_h,
            )
            splits = impute_market_window_splits(splits)
            splits = impute_tabular_splits(splits)

            if config.get("log_news_coverage", True) and "news_embs" in data:
                total_bars = len(data["times"])
                embs = data["news_embs"]
                if embs.ndim == 3:
                    last_bar_norm = np.linalg.norm(embs[:, -1, :], axis=-1)
                else:
                    last_bar_norm = np.linalg.norm(embs, axis=-1)
                has_news = last_bar_norm > 0
                bars_with_news = int(has_news.sum())
                coverage_pct = bars_with_news / total_bars * 100 if total_bars > 0 else 0
                logger.info("[{}] News coverage: {}/{} bars ({:.1f}%)", sym, bars_with_news, total_bars, coverage_pct)

            if len(splits["test"]["targets"]) == 0:
                logger.warning("{}: no test samples — skipping", sym)
                continue

            y_test = splits["test"]["targets"]
            all_y_true.append(y_test)
            _sym_order.append(sym)

            sh = _split_hash(splits, sym, target_h)
            _all_splits[sym] = splits
            _all_sh[sym] = sh

            zs_splits_source = per_symbol_365.get(sym, data)
            zs_splits = split_by_date(
                {k: v for k, v in zs_splits_source.items() if k != "times"},
                zs_splits_source["times"],
                run_cfg["train_end"],
                run_cfg["val_end"],
                target_horizon_days=target_h,
            )

            zs_sh = _split_hash(zs_splits, sym, target_h)
            zs_cache = CACHE_PRED_DIR / f"zs_v2_{sym}_{target_h}d_{zs_sh}.npy"
            if _use_prediction_cache(stage) and (cached_zs := _load_npy(zs_cache)) is not None:
                logger.info("[{}] Zero-shot loaded from cache", sym)
                preds_zs = cached_zs
            else:
                logger.info("[{}] Running Chronos zero-shot (horizon={}) …", sym, target_h)
                preds_zs = chronos.zero_shot_predict(
                    zs_splits["test"]["close_windows"],
                    zs_splits["test"]["last_close"],
                    seed=config["seed"],
                    horizon=target_h,
                )
                _save_npy(zs_cache, preds_zs)

            metrics_zs = compute_all(y_test, preds_zs, horizon=target_h)
            metrics_zs.update(compute_composite_metrics(y_test, preds_zs, horizon=target_h, anchor_pred=preds_zs))
            metrics_zs.update({
                "Experiment": "Chronos Zero-Shot",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_zs)
            all_preds["Chronos Zero-Shot"].append(preds_zs)

        baseline_hpo_params = None
        if _sym_order:
            baseline_sym = _sym_order[0]
            baseline_splits = _all_splits[baseline_sym]
            logger.info("Loading or running baseline HPO for {}D", target_h)
            baseline_hpo_params = load_or_run_baseline_hpo(
                CACHE_HPO_DIR,
                baseline_splits["train"]["market_windows"],
                baseline_splits["train"]["targets"],
                baseline_splits["val"]["market_windows"],
                baseline_splits["val"]["targets"],
                chronos,
                close_windows_train=baseline_splits["train"]["close_windows"],
                close_windows_val=baseline_splits["val"]["close_windows"],
                market_tabular_train=baseline_splits["train"].get("market_tabular") if use_tabular_market_features else None,
                market_tabular_val=baseline_splits["val"].get("market_tabular") if use_tabular_market_features else None,
                target_h=target_h,
                device=device,
                fallback_to_defaults=True,
            )

        if baseline_hpo_params is None:
            baseline_hpo_params = get_default_baseline_hpo_params()

        baseline_hpo_params.setdefault("mlp_summary", {
            "hidden_dim": 64,
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 32,
        })
        baseline_hpo_params.setdefault("lstm_hybrid", baseline_hpo_params.get("lstm", {
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 32,
        }))
        baseline_hpo_params.setdefault("cnn_lstm_hybrid", baseline_hpo_params.get("cnn_lstm", {
            "num_filters": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 32,
        }))

        baseline_hpo_params.setdefault("gpt4ts", {
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 32,
        })
        baseline_hpo_params.setdefault("gpt4ts_hybrid", {
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 32,
        })

        for sym in _sym_order:
            splits = _all_splits[sym]
            sh = _all_sh[sym]
            y_test = splits["test"]["targets"]
            anchor_pred = all_preds["Chronos Zero-Shot"][_sym_order.index(sym)]

            train_tab = extract_market_summary_features(splits["train"]["market_windows"])
            val_tab = extract_market_summary_features(splits["val"]["market_windows"])
            test_tab = extract_market_summary_features(splits["test"]["market_windows"])
            tabular_dim = train_tab.shape[1]

            # ----------------------------------------------------------
            # LSTM
            # ----------------------------------------------------------
            lstm_cache = CACHE_PRED_DIR / f"lstm_{sym}_{target_h}d_{sh}.npy"
            lstm_params = baseline_hpo_params["lstm"]
            lstm_param_hash = hashlib.md5(str(sorted(lstm_params.items())).encode()).hexdigest()[:8]
            lstm_ckpt = CACHE_MODEL_DIR / f"lstm_backbone_v3_{sym}_{target_h}d_{lstm_param_hash}_{sh}.pt"
            lstm_ckpt.parent.mkdir(parents=True, exist_ok=True)

            lstm_model = LSTMPredictor(
                input_dim=splits["train"]["market_windows"].shape[-1],
                hidden_dim=lstm_params.get("hidden_dim", 64),
                num_layers=lstm_params.get("num_layers", 2),
                dropout=lstm_params.get("dropout", 0.3),
                device=device,
            )

            cached_lstm = _load_npy(lstm_cache)
            cached_lstm_is_finite = cached_lstm is not None and np.isfinite(cached_lstm).all()
            if stage not in ("data", "predict") and cached_lstm_is_finite and lstm_ckpt.exists():
                logger.info("[{}] LSTM loaded from cache", sym)
                preds_lstm = cached_lstm
                lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location=device, weights_only=False))
            else:
                logger.info("[{}] Training LSTM…", sym)
                lstm_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    epochs=100,
                    batch_size=lstm_params.get("batch_size", 32),
                    learning_rate=lstm_params.get("lr", 1e-3),
                    patience=15,
                )
                preds_lstm = lstm_model.predict(splits["test"]["market_windows"])
                _save_npy(lstm_cache, preds_lstm)
                torch.save(lstm_model.state_dict(), lstm_ckpt)

            metrics_lstm = compute_all(y_test, preds_lstm, horizon=target_h)
            metrics_lstm.update(
                compute_composite_metrics(y_test, preds_lstm, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_lstm.update({
                "Experiment": "LSTM Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lstm)
            all_preds["LSTM Baseline"].append(preds_lstm)

            # ----------------------------------------------------------
            # LSTM Hybrid
            # ----------------------------------------------------------
            lstm_hybrid_cache = CACHE_PRED_DIR / f"lstm_hybrid_{sym}_{target_h}d_{sh}.npy"
            lstm_hybrid_params = baseline_hpo_params["lstm_hybrid"]

            if stage not in ("data", "predict") and (cached_lstm_hybrid := _load_npy(lstm_hybrid_cache)) is not None:
                logger.info("[{}] LSTM Hybrid loaded from cache", sym)
                preds_lstm_hybrid = cached_lstm_hybrid
            else:
                logger.info("[{}] Training LSTM Hybrid…", sym)
                lstm_hybrid_model = LSTMHybridPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    tabular_dim=tabular_dim,
                    hidden_dim=lstm_hybrid_params.get("hidden_dim", 64),
                    num_layers=lstm_hybrid_params.get("num_layers", 2),
                    dropout=lstm_hybrid_params.get("dropout", 0.3),
                    device=device,
                )
                lstm_hybrid_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    market_tabular_train=train_tab,
                    market_tabular_val=val_tab,
                    epochs=100,
                    batch_size=lstm_hybrid_params.get("batch_size", 32),
                    learning_rate=lstm_hybrid_params.get("lr", 1e-3),
                    patience=15,
                )
                preds_lstm_hybrid = lstm_hybrid_model.predict(
                    splits["test"]["market_windows"],
                    market_tabular=test_tab,
                )
                _save_npy(lstm_hybrid_cache, preds_lstm_hybrid)

            metrics_lstm_hybrid = compute_all(y_test, preds_lstm_hybrid, horizon=target_h)
            metrics_lstm_hybrid.update(
                compute_composite_metrics(y_test, preds_lstm_hybrid, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_lstm_hybrid.update({
                "Experiment": "LSTM Hybrid",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lstm_hybrid)
            all_preds["LSTM Hybrid"].append(preds_lstm_hybrid)

            # ----------------------------------------------------------
            # Random Forest
            # ----------------------------------------------------------
            rf_cache = CACHE_PRED_DIR / f"rf_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_rf := _load_npy(rf_cache)) is not None:
                logger.info("[{}] Random Forest loaded from cache", sym)
                preds_rf = cached_rf
            else:
                logger.info("[{}] Training Random Forest…", sym)
                rf_params = baseline_hpo_params["rf"]
                rf_model = RandomForestRegressor_Wrapper(
                    n_estimators=rf_params.get("n_estimators", 100),
                    max_depth=rf_params.get("max_depth", 10),
                    min_samples_split=rf_params.get("min_samples_split", 5),
                    max_features=rf_params.get("max_features", "sqrt"),
                    random_state=42,
                )
                rf_model.fit(splits["train"]["market_windows"], splits["train"]["targets"])
                preds_rf = rf_model.predict(splits["test"]["market_windows"])
                _save_npy(rf_cache, preds_rf)

            metrics_rf = compute_all(y_test, preds_rf, horizon=target_h)
            metrics_rf.update(
                compute_composite_metrics(y_test, preds_rf, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_rf.update({
                "Experiment": "Random Forest Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_rf)
            all_preds["Random Forest Baseline"].append(preds_rf)

            # ----------------------------------------------------------
            # Linear Summary
            # ----------------------------------------------------------
            lin_cache = CACHE_PRED_DIR / f"linear_summary_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_lin := _load_npy(lin_cache)) is not None:
                logger.info("[{}] Linear Summary loaded from cache", sym)
                preds_lin = cached_lin
            else:
                logger.info("[{}] Training Linear Summary…", sym)
                lin_model = LinearSummaryRegressor_Wrapper(alpha=1.0)
                lin_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                )
                preds_lin = lin_model.predict(splits["test"]["market_windows"])
                _save_npy(lin_cache, preds_lin)

            metrics_lin = compute_all(y_test, preds_lin, horizon=target_h)
            metrics_lin.update(
                compute_composite_metrics(y_test, preds_lin, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_lin.update({
                "Experiment": "Linear Summary Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lin)
            all_preds["Linear Summary Baseline"].append(preds_lin)

            # ----------------------------------------------------------
            # MLP Summary
            # ----------------------------------------------------------
            mlp_summary_cache = CACHE_PRED_DIR / f"mlp_summary_{sym}_{target_h}d_{sh}.npy"
            mlp_summary_params = baseline_hpo_params["mlp_summary"]

            if stage not in ("data", "predict") and (cached_mlp_sum := _load_npy(mlp_summary_cache)) is not None:
                logger.info("[{}] MLP Summary loaded from cache", sym)
                preds_mlp_summary = cached_mlp_sum
            else:
                logger.info("[{}] Training MLP Summary…", sym)
                mlp_summary_model = MLPSummaryPredictor(
                    hidden_dim=mlp_summary_params.get("hidden_dim", 64),
                    dropout=mlp_summary_params.get("dropout", 0.2),
                    device=device,
                )
                mlp_summary_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    epochs=100,
                    batch_size=mlp_summary_params.get("batch_size", 32),
                    learning_rate=mlp_summary_params.get("lr", 1e-3),
                    patience=15,
                )
                preds_mlp_summary = mlp_summary_model.predict(splits["test"]["market_windows"])
                _save_npy(mlp_summary_cache, preds_mlp_summary)

            metrics_mlp_summary = compute_all(y_test, preds_mlp_summary, horizon=target_h)
            metrics_mlp_summary.update(
                compute_composite_metrics(y_test, preds_mlp_summary, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_mlp_summary.update({
                "Experiment": "MLP Summary Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_mlp_summary)
            all_preds["MLP Summary Baseline"].append(preds_mlp_summary)

            # ----------------------------------------------------------
            # CNN-LSTM
            # ----------------------------------------------------------
            cnn_lstm_params = baseline_hpo_params.get("cnn_lstm", baseline_hpo_params["lstm"])
            cnn_lstm_param_hash = hashlib.md5(str(sorted(cnn_lstm_params.items())).encode()).hexdigest()[:8]
            cnn_lstm_cache = CACHE_PRED_DIR / f"cnn_lstm_v3_{sym}_{target_h}d_{cnn_lstm_param_hash}_{sh}.npy"
            cnn_lstm_ckpt = CACHE_MODEL_DIR / f"cnn_lstm_model_v3_{sym}_{target_h}d_{cnn_lstm_param_hash}_{sh}.pt"
            cnn_lstm_ckpt.parent.mkdir(parents=True, exist_ok=True)

            cached_cnn_lstm = _load_npy(cnn_lstm_cache)
            if stage not in ("data", "predict") and cached_cnn_lstm is not None and cnn_lstm_ckpt.exists():
                logger.info("[{}] CNN-LSTM loaded from cache", sym)
                preds_cnn_lstm = cached_cnn_lstm
            else:
                logger.info("[{}] Training CNN-LSTM…", sym)
                cnn_lstm_model = CNNLSTMPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    num_filters=cnn_lstm_params.get("num_filters", cnn_lstm_params.get("hidden_dim", 64)),
                    hidden_dim=cnn_lstm_params.get("hidden_dim", 64),
                    num_layers=cnn_lstm_params.get("num_layers", 2),
                    dropout=cnn_lstm_params.get("dropout", 0.3),
                    device=device,
                )
                cnn_lstm_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    epochs=100,
                    batch_size=cnn_lstm_params.get("batch_size", 32),
                    learning_rate=cnn_lstm_params.get("lr", 1e-3),
                    patience=15,
                )
                preds_cnn_lstm = cnn_lstm_model.predict(splits["test"]["market_windows"])
                _save_npy(cnn_lstm_cache, preds_cnn_lstm)
                torch.save(cnn_lstm_model.state_dict(), cnn_lstm_ckpt)

            metrics_cnn_lstm = compute_all(y_test, preds_cnn_lstm, horizon=target_h)
            metrics_cnn_lstm.update(
                compute_composite_metrics(y_test, preds_cnn_lstm, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_cnn_lstm.update({
                "Experiment": "CNN-LSTM",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_cnn_lstm)
            all_preds["CNN-LSTM"].append(preds_cnn_lstm)

            # ----------------------------------------------------------
            # CNN-LSTM Hybrid
            # ----------------------------------------------------------
            cnn_lstm_hybrid_cache = CACHE_PRED_DIR / f"cnn_lstm_hybrid_{sym}_{target_h}d_{sh}.npy"
            cnn_lstm_hybrid_params = baseline_hpo_params["cnn_lstm_hybrid"]

            if stage not in ("data", "predict") and (cached_cnn_lstm_hybrid := _load_npy(cnn_lstm_hybrid_cache)) is not None:
                logger.info("[{}] CNN-LSTM Hybrid loaded from cache", sym)
                preds_cnn_lstm_hybrid = cached_cnn_lstm_hybrid
            else:
                logger.info("[{}] Training CNN-LSTM Hybrid…", sym)
                cnn_lstm_hybrid_model = CNNLSTMHybridPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    tabular_dim=tabular_dim,
                    num_filters=cnn_lstm_hybrid_params.get("num_filters", 64),
                    hidden_dim=cnn_lstm_hybrid_params.get("hidden_dim", 64),
                    num_layers=cnn_lstm_hybrid_params.get("num_layers", 2),
                    dropout=cnn_lstm_hybrid_params.get("dropout", 0.3),
                    device=device,
                )
                cnn_lstm_hybrid_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    market_tabular_train=train_tab,
                    market_tabular_val=val_tab,
                    epochs=100,
                    batch_size=cnn_lstm_hybrid_params.get("batch_size", 32),
                    learning_rate=cnn_lstm_hybrid_params.get("lr", 1e-3),
                    patience=15,
                )
                preds_cnn_lstm_hybrid = cnn_lstm_hybrid_model.predict(
                    splits["test"]["market_windows"],
                    market_tabular=test_tab,
                )
                _save_npy(cnn_lstm_hybrid_cache, preds_cnn_lstm_hybrid)

            metrics_cnn_lstm_hybrid = compute_all(y_test, preds_cnn_lstm_hybrid, horizon=target_h)
            metrics_cnn_lstm_hybrid.update(
                compute_composite_metrics(y_test, preds_cnn_lstm_hybrid, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_cnn_lstm_hybrid.update({
                "Experiment": "CNN-LSTM Hybrid",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_cnn_lstm_hybrid)
            all_preds["CNN-LSTM Hybrid"].append(preds_cnn_lstm_hybrid)

            # ----------------------------------------------------------
            # GPT4TS Baseline
            # ----------------------------------------------------------
            gpt4ts_params = baseline_hpo_params.get("gpt4ts", {})
            gpt4ts_param_hash = hashlib.md5(str(sorted(gpt4ts_params.items())).encode()).hexdigest()[:8]
            gpt4ts_cache = CACHE_PRED_DIR / f"gpt4ts_{sym}_{target_h}d_{gpt4ts_param_hash}_{sh}.npy"
            gpt4ts_ckpt = CACHE_MODEL_DIR / f"gpt4ts_model_{sym}_{target_h}d_{gpt4ts_param_hash}_{sh}.pt"
            gpt4ts_ckpt.parent.mkdir(parents=True, exist_ok=True)

            cached_gpt4ts = _load_npy(gpt4ts_cache)
            if stage not in ("data", "predict") and cached_gpt4ts is not None and gpt4ts_ckpt.exists():
                logger.info("[{}] GPT4TS Baseline loaded from cache", sym)
                preds_gpt4ts = cached_gpt4ts
            else:
                logger.info("[{}] Training GPT4TS Baseline…", sym)
                gpt4ts_model = GPT4TSPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    hidden_dim=gpt4ts_params.get("hidden_dim", 64),
                    num_layers=gpt4ts_params.get("num_layers", 3),
                    dropout=gpt4ts_params.get("dropout", 0.3),
                    device=device,
                )
                gpt4ts_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    epochs=50,
                    batch_size=gpt4ts_params.get("batch_size", 32),
                    learning_rate=gpt4ts_params.get("lr", 1e-3),
                    patience=10,
                )
                preds_gpt4ts = gpt4ts_model.predict(splits["test"]["market_windows"])
                _save_npy(gpt4ts_cache, preds_gpt4ts)
                torch.save(gpt4ts_model.state_dict(), gpt4ts_ckpt)

            metrics_gpt4ts = compute_all(y_test, preds_gpt4ts, horizon=target_h)
            metrics_gpt4ts.update(
                compute_composite_metrics(y_test, preds_gpt4ts, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_gpt4ts.update({
                "Experiment": "GPT4TS Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_gpt4ts)
            all_preds["GPT4TS Baseline"].append(preds_gpt4ts)

            # ----------------------------------------------------------
            # GPT4TS Hybrid
            # ----------------------------------------------------------
            gpt4ts_hybrid_cache = CACHE_PRED_DIR / f"gpt4ts_hybrid_{sym}_{target_h}d_{sh}.npy"
            gpt4ts_hybrid_params = baseline_hpo_params.get("gpt4ts_hybrid", {})

            if stage not in ("data", "predict") and (cached_gpt4ts_hybrid := _load_npy(gpt4ts_hybrid_cache)) is not None:
                logger.info("[{}] GPT4TS Hybrid loaded from cache", sym)
                preds_gpt4ts_hybrid = cached_gpt4ts_hybrid
            else:
                logger.info("[{}] Training GPT4TS Hybrid…", sym)
                gpt4ts_hybrid_model = GPT4TSHybridPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    tabular_dim=tabular_dim,
                    hidden_dim=gpt4ts_hybrid_params.get("hidden_dim", 64),
                    num_layers=gpt4ts_hybrid_params.get("num_layers", 3),
                    dropout=gpt4ts_hybrid_params.get("dropout", 0.3),
                    device=device,
                )
                gpt4ts_hybrid_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                    splits["val"]["market_windows"],
                    splits["val"]["targets"],
                    market_tabular_train=train_tab,
                    market_tabular_val=val_tab,
                    epochs=50,
                    batch_size=gpt4ts_hybrid_params.get("batch_size", 32),
                    learning_rate=gpt4ts_hybrid_params.get("lr", 1e-3),
                    patience=10,
                )
                preds_gpt4ts_hybrid = gpt4ts_hybrid_model.predict(
                    splits["test"]["market_windows"],
                    market_tabular=test_tab,
                )
                _save_npy(gpt4ts_hybrid_cache, preds_gpt4ts_hybrid)

            metrics_gpt4ts_hybrid = compute_all(y_test, preds_gpt4ts_hybrid, horizon=target_h)
            metrics_gpt4ts_hybrid.update(
                compute_composite_metrics(y_test, preds_gpt4ts_hybrid, horizon=target_h, anchor_pred=anchor_pred)
            )
            metrics_gpt4ts_hybrid.update({
                "Experiment": "GPT4TS Hybrid",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_gpt4ts_hybrid)
            all_preds["GPT4TS Hybrid"].append(preds_gpt4ts_hybrid)

        results_df = pd.DataFrame(all_results)

        for exp_name in results_df["Experiment"].unique():
            exp_y_parts: list[np.ndarray] = []
            exp_p_parts: list[np.ndarray] = []
            anchor_parts: list[np.ndarray] = []

            for idx, sym_name in enumerate(_sym_order):
                sym_rows = results_df[
                    (results_df["Experiment"] == exp_name) & (results_df["Symbol"] == sym_name)
                ]
                if sym_rows.empty:
                    continue
                exp_y_parts.append(all_y_true[idx])
                exp_p_parts.append(all_preds[exp_name][idx])
                anchor_parts.append(all_preds["Chronos Zero-Shot"][idx])

            if exp_y_parts:
                pooled_y = np.concatenate(exp_y_parts)
                pooled_p = np.concatenate(exp_p_parts)
                avg = compute_all(pooled_y, pooled_p, horizon=target_h)
                avg.update(
                    compute_composite_metrics(
                        pooled_y,
                        pooled_p,
                        horizon=target_h,
                        anchor_pred=np.concatenate(anchor_parts) if anchor_parts else None,
                    )
                )
            else:
                avg = {
                    "MAE": 0.0,
                    "RMSE": 0.0,
                    "DA%": 0.0,
                    "Sharpe": 0.0,
                    "IC": 0.0,
                    "Prec": 0.0,
                    "Rec": 0.0,
                    "F1": 0.0,
                    "ModalDisagreement": 0.0,
                    "TemporalLag": 0.0,
                    "CompositeScore": 0.0,
                }

            avg.update({
                "Experiment": exp_name,
                "ComparisonSet": "fairness",
                "Symbol": "AVG",
                "TargetHorizonD": target_h,
            })
            all_results.append(avg)

        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 100)
        print(f"  BASELINE BENCHMARK RESULTS — TARGET HORIZON {target_h}D")
        print("=" * 100)
        col_order = [
            "Experiment",
            "ComparisonSet",
            "Symbol",
            "TargetHorizonD",
            "MAE",
            "RMSE",
            "DA%",
            "ModalDisagreement",
            "TemporalLag",
            "CompositeScore",
            "Sharpe",
            "IC",
            "Prec",
            "Rec",
            "F1",
        ]
        results_df = results_df[col_order]

        def _fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        print(results_df.to_string(index=False, formatters={c: _fmt for c in col_order}))
        print("=" * 100 + "\n")

        suffix = f"{target_h}d"
        csv_path = RESULTS_DIR / f"chronos_benchmark_{suffix}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info("Results saved → {}", csv_path)

        avg_df = results_df[results_df["Symbol"] == "AVG"]
        plot_ablation(avg_df, FIGURES_DIR / f"model_{suffix}.png")
        plot_per_symbol(results_df, FIGURES_DIR / f"per_symbol_heatmap_{suffix}.png")

    logger.info("═══ Benchmark complete ═══")


if __name__ == "__main__":
    main()