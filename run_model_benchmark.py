"""Run model benchmark: full-scope research comparison.

Evaluates:
    - Chronos Zero-Shot
    - LSTM Baseline
    - LSTM Hybrid
    - Random Forest Baseline
    - Linear Summary Baseline
    - MLP Summary Baseline
    - CNN-LSTM
    - CNN-LSTM Hybrid
    - GPT4TS Baseline
    - GPT4TS Hybrid

Research-fair additions:
    - Explicit protocol metadata columns:
        ComparisonSet, InputRegime, AdaptationRegime, ContextLength
    - Shared trained anchor (LSTM Baseline) for disagreement/composite diagnostics
    - Standardized target_scale=1.0 for trainable torch models where applicable
    - Preserves all models; does not drop or hide any comparison

Audit additions:
    - Optional symbol filter via --symbols
    - Optional experiment filter via --experiments
    - Optional cache bypass via --no-cache
    - Optional direction-loss warmup via --warmup-epochs
    - GPT4TS patch_stride passthrough if present in HPO params

Global Option-1 tuning additions:
    - One global config value for sign_penalty_weight
    - Shared across all trainable torch models
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.common import generate_walkforward_folds
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
    GLOBAL_LOSS_CONFIG,
)
from src.benchmark.gpt4ts_encoder import GPT4TSPredictor, GPT4TSHybridPredictor, GPT4TS_DEFAULTS
from src.benchmark.baseline_hpo import (
    get_default_baseline_hpo_params,
    load_or_run_baseline_hpo,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_PRED_DIR = Path("cache/predictions")
CACHE_MODEL_DIR = Path("cache/cmtf_models")
CACHE_HPO_DIR = Path("cache/optuna")


# ============================================================================
# Utilities
# ============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to {}", seed)


def _use_prediction_cache(stage: str | None, no_cache: bool = False) -> bool:
    if no_cache:
        return False
    return stage is None or stage == "plot"


def _split_hash(splits: dict, sym: str, horizon: int) -> str:
    h = hashlib.sha256()
    h.update(f"{sym}_{horizon}".encode())
    for name in ("train", "val", "test"):
        split = splits[name]
        targets = np.asarray(split["targets"], dtype=np.float64)
        h.update(f"{name}=n{len(targets)}".encode())
        times = split.get("times")
        if times is not None and len(times) > 0:
            times_arr = np.asarray(times)
            h.update(f"{name}_tmin={times_arr.min()}".encode())
            h.update(f"{name}_tmax={times_arr.max()}".encode())
        if targets.size > 0:
            h.update(targets.tobytes())
    return h.hexdigest()[:12]


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_npy(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path)
    return None


# ============================================================================
# Protocol metadata
# ============================================================================

@dataclass(frozen=True)
class ProtocolMeta:
    comparison_set: str
    input_regime: str
    adaptation_regime: str
    context_length: int


PROTOCOLS: dict[str, ProtocolMeta] = {
    "Chronos Zero-Shot": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="close_only",
        adaptation_regime="zero_shot",
        context_length=365,
    ),
    "LSTM Baseline": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window",
        adaptation_regime="trained",
        context_length=30,
    ),
    "LSTM Hybrid": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window_plus_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
    "Random Forest Baseline": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="engineered_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
    "Linear Summary Baseline": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="engineered_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
    "MLP Summary Baseline": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="engineered_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
    "CNN-LSTM": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window",
        adaptation_regime="trained",
        context_length=30,
    ),
    "CNN-LSTM Hybrid": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window_plus_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
    "GPT4TS Baseline": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window",
        adaptation_regime="trained",
        context_length=30,
    ),
    "GPT4TS Hybrid": ProtocolMeta(
        comparison_set="cross_model",
        input_regime="multivariate_window_plus_summary",
        adaptation_regime="trained",
        context_length=30,
    ),
}


# ============================================================================
# Data extraction helpers
# ============================================================================

def instance_normalize_windows(windows: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RevIN-style per-window instance normalization (anti-collapse fix).

    Removes non-stationary absolute level from each window by standardizing every
    (window, feature) channel over its own time axis. This is the standard fix for
    covariate shift in time-series forecasting (RevIN / PatchTST / GPT4TS).

    Rationale: the canonical market features include absolute price-level columns
    (open/high/low/close/bb_*/macd/atr). Under a train-fit global z-score these
    become pinned at large ~constant z-values in a later bull/bear regime, so the
    model receives a near-constant out-of-distribution input and emits a constant
    sign (collapse). Per-window standardization makes each window's *shape*
    comparable across regimes, eliminating the OOD level.

    Args:
        windows: (N, seq_len, n_features) float array.
        eps: variance floor to avoid division by near-zero std on flat channels.

    Returns:
        Per-window standardized copy, same shape and dtype float32.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1] < 2:
        return windows
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return ((windows - mean) / (std + eps)).astype(np.float32)


def extract_per_symbol_data(
    dataset,
    raw_ohlcv: dict[str, pd.DataFrame],
    seq_len: int = 30,
    target_horizon_days: int = 1,
    close_only: bool = False,
    allow_missing_target: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
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
    skipped: dict[str, str] = {}  # sym → skip reason

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
            reason = "not in raw_ohlcv"
            logger.warning("{} skipped — {}", sym, reason)
            skipped[sym] = reason
            continue

        raw_df = raw_ohlcv[sym][["close"]].rename(columns={"close": "raw_close"})
        sym_df = sym_df.merge(raw_df, left_on="time", right_index=True, how="left")

        missing = int(sym_df["raw_close"].isna().sum())
        assert missing == 0, f"{sym}: {missing} timestamps missing after merge with raw OHLCV"

        raw_c = sym_df["raw_close"].to_numpy(dtype=np.float64)

        if n < seq_len + 1:
            reason = f"only {n} rows, need {seq_len + 1}"
            logger.warning("{} skipped — {}", sym, reason)
            skipped[sym] = reason
            continue

        news_col = "news_hybrid_emb" if "news_hybrid_emb" in sym_df.columns else "news_emb"
        if not close_only:
            if news_col not in sym_df.columns:
                raise ValueError(f"{sym}: missing news embedding column {news_col}")
            news_list = sym_df[news_col].tolist()
            news_arr = np.stack(news_list).astype(np.float32)
            has_news_arr = sym_df["has_news"].astype(bool).to_numpy(copy=True)
        market_values = sym_df[filtered_market_cols].astype(np.float32).to_numpy(copy=True)

        target_col_name = f"fwd_ret_{int(target_horizon_days)}d"
        if target_col_name not in sym_df.columns:
            reason = f"missing target column {target_col_name}"
            logger.warning("{} skipped — {}", sym, reason)
            skipped[sym] = reason
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

        _emb_dim = news_arr.shape[-1] if not close_only else 1  # placeholder not used

        for i in range(valid_start, valid_end):
            target_val = target_col[i]
            # A NaN target means the row is too close to the end of the fetched
            # range for a full forward-return window to exist yet (e.g. today, or
            # any date within `target_horizon_days` of the most recent bar) — this
            # is the ONLY case for genuinely live/current-date rows, since the
            # future hasn't happened. Dropping these is correct for
            # training/backtesting (a label is required), but wrong for live
            # single-row inference, which needs only the feature window, never a
            # label. `allow_missing_target` keeps the row (features intact, target
            # NaN) instead of silently discarding the exact rows live inference
            # exists to serve.
            if np.isnan(target_val) and not allow_missing_target:
                continue

            close_windows.append(raw_c[i - seq_len + 1 : i + 1])
            market_windows.append(market_values[i - seq_len + 1 : i + 1])
            last_closes.append(raw_c[i])
            market_tabs.append(market_values[i])
            if not close_only:
                news_embs.append(news_arr[i - seq_len + 1 : i + 1])
                news_masks.append(~has_news_arr[i - seq_len + 1 : i + 1])
            targets.append(target_val)
            times.append(sym_df.iloc[i]["time"])

        if not targets:
            reason = "zero valid samples after windowing (all targets NaN?)"
            logger.warning("{} skipped — {}", sym, reason)
            skipped[sym] = reason
            continue

        n_samples = len(targets)
        result[sym] = {
            "close_windows": np.asarray(close_windows, dtype=np.float64),
            "market_windows": instance_normalize_windows(
                np.asarray(market_windows, dtype=np.float32)
            ),
            # Skip news arrays when close_only=True (e.g. Chronos 365-context
            # extraction) to avoid allocating N×365×768 float32 arrays that are
            # never read by the zero-shot predictor.
            "news_embs": (
                np.asarray(news_embs, dtype=np.float32) if not close_only
                else np.zeros((n_samples, 1, 1), dtype=np.float32)
            ),
            "last_close": np.asarray(last_closes, dtype=np.float64),
            "market_tabular": np.asarray(market_tabs, dtype=np.float32),
            "news_masks": (
                np.asarray(news_masks, dtype=bool) if not close_only
                else np.zeros((n_samples, 1), dtype=bool)
            ),
            "targets": np.asarray(targets, dtype=np.float32),
            "times": np.asarray(times),
        }
        logger.info("{} → {} samples extracted", sym, len(targets))

    if skipped:
        logger.warning(
            "extract_per_symbol_data: {} symbol(s) skipped — {}",
            len(skipped),
            {s: r for s, r in skipped.items()},
        )
    logger.info(
        "extract_per_symbol_data: {}/{} symbols extracted successfully",
        len(result), len(symbols),
    )
    return result


def split_by_date(
    data: dict[str, np.ndarray],
    times: np.ndarray,
    train_end: str,
    val_end: str,
    target_horizon_days: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    times = np.asarray(times, dtype="datetime64[ns]")
    train_end_ts = pd.Timestamp(train_end).to_datetime64()
    val_end_ts = pd.Timestamp(val_end).to_datetime64()

    sorted_times = np.sort(np.unique(times))

    def _trading_day_offset(boundary: np.datetime64, n: int) -> np.datetime64:
        idx = int(np.searchsorted(sorted_times, boundary, side="right")) - 1
        idx = max(idx - n, 0)
        return sorted_times[idx]

    train_end_purged = _trading_day_offset(train_end_ts, target_horizon_days)
    val_end_purged = _trading_day_offset(val_end_ts, target_horizon_days)

    train_mask = times <= train_end_purged
    val_mask = (times > train_end_ts) & (times <= val_end_purged)
    test_mask = times > val_end_ts

    splits: dict[str, dict[str, np.ndarray]] = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split_data = {k: v[mask] for k, v in data.items() if k != "times"}
        split_data["times"] = times[mask]
        splits[name] = split_data

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


# ============================================================================
# Plotting
# ============================================================================

def plot_ablation(results_df: pd.DataFrame, save_path: Path) -> None:
    metrics = ["MAE", "RMSE", "DA%", "Sharpe", "IC"]
    experiments = results_df["Experiment"].unique()
    n_exp = len(experiments)

    colors = [
        "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
        "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
        "#7f8c8d", "#c0392b",
    ]

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
        ax.set_xticklabels(experiments, rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Model Benchmark (AVG)", fontsize=14, fontweight="bold", y=1.05)
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
            row_labels.append(f"{sym} / {exp}")
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
            text = (
                f"{val:.4f}"
                if metrics[j] in ("MAE", "RMSE")
                else (f"{val:.1f}%" if metrics[j] == "DA%" else f"{val:.3f}")
            )
            brightness = norm_data[i, j]
            color = "white" if brightness < 0.35 or brightness > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Per-Symbol Benchmark Heatmap", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-symbol heatmap → {}", save_path)


# ============================================================================
# Result packaging
# ============================================================================

def package_result(
    experiment_name: str,
    symbol: str,
    target_h: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchor_pred: np.ndarray | None,
) -> dict[str, Any]:
    protocol = PROTOCOLS[experiment_name]
    metrics = compute_all(y_true, y_pred, horizon=target_h)
    metrics.update(
        compute_composite_metrics(
            y_true,
            y_pred,
            horizon=target_h,
            anchor_pred=anchor_pred,
        )
    )

    # Prediction-distribution diagnostics (Phase 1 transparency).
    pred_std = float(np.std(y_pred))
    pred_pos_pct = float(100.0 * np.mean(y_pred > 0))
    sign_bias = float(abs(pred_pos_pct - 50.0))
    # Degenerate = effectively single-sign: >95% or <5% of predictions are positive.
    # This catches both exact-0.5 recall cases AND dead-zone-shifted ones.
    is_degenerate = int(pred_pos_pct > 95.0 or pred_pos_pct < 5.0)
    if is_degenerate:
        logger.warning(
            "DEGENERATE CELL: {} | {} | {}D — single-sign predictions detected "
            "(PredPosPct={:.1f}%, DA%={:.3f}%, DA_skill%={:.2f})",
            experiment_name, symbol, target_h,
            pred_pos_pct, metrics.get("DA%", 0.0), metrics.get("DA_skill%", 0.0),
        )
    metrics.update({
        "PredStd": pred_std,
        "PredPosPct": pred_pos_pct,
        "SignBias": sign_bias,
        "Degenerate": is_degenerate,
    })

    metrics.update({
        "Experiment": experiment_name,
        "ComparisonSet": protocol.comparison_set,
        "InputRegime": protocol.input_regime,
        "AdaptationRegime": protocol.adaptation_regime,
        "ContextLength": protocol.context_length,
        "Symbol": symbol,
        "TargetHorizonD": target_h,
    })
    return metrics


# ============================================================================
# Training logs saving
# ============================================================================

def save_training_logs(
    logs_dict: dict[str, dict],
    target_h: int,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Save training logs for all models to CSV.
    
    Args:
        logs_dict: Dict with keys like "LSTM Baseline", values are dicts with keys:
                   {symbol: {"train_loss": [...], "val_loss": [...], ...}}
        target_h: Target horizon in days (1, 5, 20, etc.)
        results_dir: Directory to save logs
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_rows = []
    
    for model_name, model_logs in logs_dict.items():
        for symbol, metrics in model_logs.items():
            row = {
                "Model": model_name,
                "Symbol": symbol,
                "Horizon": target_h,
            }
            
            # Add training history
            train_losses = metrics.get("train_losses", [])
            val_losses = metrics.get("val_losses", [])
            val_losses_clean = metrics.get("val_losses_clean", [])
            pred_means = metrics.get("pred_means", [])
            pred_pct_pos = metrics.get("pred_pct_pos", [])
            pred_pct_neg = metrics.get("pred_pct_neg", [])
            
            # Epoch-wise metrics
            max_epochs = max(len(train_losses), len(val_losses))
            for epoch in range(max_epochs):
                train_loss = train_losses[epoch] if epoch < len(train_losses) else np.nan
                val_loss = val_losses[epoch] if epoch < len(val_losses) else np.nan
                val_clean = val_losses_clean[epoch] if epoch < len(val_losses_clean) else np.nan
                pred_mean = pred_means[epoch] if epoch < len(pred_means) else np.nan
                pred_pos = pred_pct_pos[epoch] if epoch < len(pred_pct_pos) else np.nan
                pred_neg = pred_pct_neg[epoch] if epoch < len(pred_pct_neg) else np.nan
                
                row[f"Epoch_{epoch:03d}_TrainLoss"] = train_loss
                row[f"Epoch_{epoch:03d}_ValLoss"] = val_loss
                row[f"Epoch_{epoch:03d}_ValClean"] = val_clean
                row[f"Epoch_{epoch:03d}_PredMean"] = pred_mean
                row[f"Epoch_{epoch:03d}_PredPos%"] = pred_pos
                row[f"Epoch_{epoch:03d}_PredNeg%"] = pred_neg
            
            # Summary metrics
            row["BestValLoss"] = metrics.get("best_val_loss", np.nan)
            row["FinalTrainLoss"] = train_losses[-1] if train_losses else np.nan
            row["FinalValLoss"] = val_losses[-1] if val_losses else np.nan
            row["FinalValClean"] = val_losses_clean[-1] if val_losses_clean else np.nan
            row["FinalPredMean"] = pred_means[-1] if pred_means else np.nan
            row["FinalPredPos%"] = pred_pct_pos[-1] if pred_pct_pos else np.nan
            row["FinalPredNeg%"] = pred_pct_neg[-1] if pred_pct_neg else np.nan
            
            # Test metrics (computed from predictions)
            row["TestMAE"] = metrics.get("test_mae", np.nan)
            row["TestRMSE"] = metrics.get("test_rmse", np.nan)
            row["TestDA%"] = metrics.get("test_da", np.nan)
            row["TestSharpe"] = metrics.get("test_sharpe", np.nan)
            
            log_rows.append(row)
    
    if log_rows:
        logs_df = pd.DataFrame(log_rows)
        logs_csv_path = results_dir / f"training_logs_{target_h}d.csv"
        logs_df.to_csv(logs_csv_path, index=False)
        logger.info("Training logs saved → {}", logs_csv_path)
    else:
        logger.warning("No training logs to save for {}D", target_h)


# ============================================================================
# Main benchmark
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scope model benchmark")
    parser.add_argument(
        "--stage",
        choices=["data", "predict", "hpo", "plot"],
        default=None,
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        type=str,
        default=None,
        help="Optional symbol filter, e.g. --symbols VCB BID",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=None,
        help='Optional experiment filter, e.g. --experiments "GPT4TS Baseline" "CNN-LSTM"',
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prediction cache and force fresh retraining/inference",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs before enabling direction loss for trainable torch models (default: 5)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Walk-forward folds via generate_walkforward_folds (default 1 = fixed split). "
             "When >1, results include a 'Fold' column and CSV is appended across folds.",
    )
    parser.add_argument(
        "--skip-chronos",
        action="store_true",
        help="Skip the Chronos zero-shot baseline (slowest; no adaptation; worst average). "
             "Saves the expensive 365-context extraction + autoregressive inference.",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=0,
        help="Subsample the TRAIN split every N samples (0=auto: max(1,horizon//4)). "
             "Val/test stay at full resolution so metrics are unaffected. "
             "Cuts redundant overlapping-window training at long horizons.",
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
        "symbols": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB"],
        "start": "2020-01-01",
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
        "sentiment_output_dir": "outputs/sentiment/latest",
        "news_similarity_threshold": 85.0,
        "log_news_coverage": True,
        "sequence_len": 30,
        "horizon": 1,
        "target_horizons_days": [1, 5, 20],
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        "use_tabular_market_features": True,

        # Global Option-1 tuning knob for all trainable torch models
        # (single source of truth — also the default in GLOBAL_LOSS_CONFIG)
        "sign_penalty_weight": GLOBAL_LOSS_CONFIG.sign_penalty_weight,
    }

    if stage == "data":
        config["rebuild_data"] = True
    if args.horizons:
        config["target_horizons_days"] = [int(h) for h in args.horizons]
    if args.symbols:
        config["symbols"] = [str(s) for s in args.symbols]

    set_global_seed(config["seed"])

    logger.info(
        "Audit controls | symbols={} | horizons={} | experiments={} | no_cache={} | warmup_epochs={}",
        config["symbols"],
        config.get("target_horizons_days"),
        args.experiments if args.experiments else "ALL",
        args.no_cache,
        args.warmup_epochs,
    )
    logger.info(
        "Global loss setting | sign_penalty_weight={}",
        config["sign_penalty_weight"],
    )

    if stage == "plot":
        horizons = [int(h) for h in config.get("target_horizons_days", [1])]
        for target_h in horizons:
            csv_path = RESULTS_DIR / f"model_benchmark_{target_h}d.csv"
            if not csv_path.exists():
                logger.warning("No results CSV for {}D — skipping plot", target_h)
                continue
            results_df = pd.read_csv(csv_path)
            avg_df = results_df[results_df["Symbol"] == "AVG"]
            plot_ablation(avg_df, FIGURES_DIR / f"model_{target_h}d.png")
            plot_per_symbol(results_df, FIGURES_DIR / f"per_symbol_heatmap_{target_h}d.png")
        return

    if stage == "data":
        horizons = [int(h) for h in config.get("target_horizons_days", [1])]
        for target_h in horizons:
            run_cfg = {**config, "target_horizon_days": target_h}
            run_pipeline(run_cfg)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("═══ Loading Chronos (device={}) ═══", device)
    chronos = ChronosMarketPredictor(device=device)

    logger.info("═══ Fetching raw OHLCV ═══")
    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(config["symbols"], config["start"], config["end"])

    horizons = [int(h) for h in config.get("target_horizons_days", [1])]

    # Walk-forward fold support: build fold list once from the full date range.
    # With --folds 1 (default), this yields a single (train_end, val_end) pair
    # identical to the fixed config split — fully backward-compatible.
    if args.folds > 1:
        all_dates = pd.date_range(config["start"], config["end"], freq="B")
        fold_pairs = generate_walkforward_folds(
            all_dates,
            n_folds=args.folds,
            test_months=6,
            min_train_months=36,
        )
        logger.info("Walk-forward mode: {} folds → {}", len(fold_pairs), fold_pairs)
    else:
        fold_pairs = [(config["train_end"], config["val_end"])]

    experiments = [
        "Chronos Zero-Shot",
        "LSTM Baseline",
        "LSTM Hybrid",
        "Random Forest Baseline",
        "Linear Summary Baseline",
        "MLP Summary Baseline",
        "CNN-LSTM",
        "CNN-LSTM Hybrid",
        "GPT4TS Baseline",
        "GPT4TS Hybrid",
    ]
    if args.skip_chronos:
        experiments = [e for e in experiments if e != "Chronos Zero-Shot"]
        logger.info("--skip-chronos: Chronos excluded from experiments.")
    if args.experiments:
        requested = set(args.experiments)
        experiments = [exp for exp in experiments if exp in requested]
        logger.info("Filtered experiments: {}", experiments)

    all_fold_results: list[pd.DataFrame] = []

    for fold_idx, (fold_train_end, fold_val_end) in enumerate(fold_pairs):
        fold_label = f"fold{fold_idx}" if len(fold_pairs) > 1 else None
        if fold_label:
            logger.info("═══ Walk-forward {} ({} / {}) train_end={} val_end={} ═══",
                        fold_label, fold_idx + 1, len(fold_pairs), fold_train_end, fold_val_end)
        fold_config = {**config, "train_end": fold_train_end, "val_end": fold_val_end}

    # NOTE: for --folds 1 (default), fold_config equals config exactly.
    # Multi-fold aggregation (nest the horizon loop inside the fold loop)
    # is deferred; the helper in src/common.py is the single source of truth.

    # E1: Build dataset once per fold — all horizons share the same pipeline output
    # (compute_technical produces fwd_ret_1d/5d/20d together; only train_end matters
    # for normalization). Eliminates 2/3 of the expensive pipeline+PhoBERT cost.
    _pipeline_cfg = {**fold_config, "target_horizon_days": horizons[0]}
    logger.info("═══ Building dataset once (shared across all horizons) ═══")
    _shared_dataset = run_pipeline(_pipeline_cfg)

    # Pre-extract per-symbol arrays for every horizon at once.
    _per_symbol_by_h: dict[int, dict] = {}
    _per_symbol_365_by_h: dict[int, dict] = {}
    for _h in horizons:
        _per_symbol_by_h[_h] = extract_per_symbol_data(
            _shared_dataset, raw_ohlcv,
            seq_len=fold_config["sequence_len"], target_horizon_days=_h,
        )
        if "Chronos Zero-Shot" in experiments:
            _per_symbol_365_by_h[_h] = extract_per_symbol_data(
                _shared_dataset, raw_ohlcv,
                seq_len=365, target_horizon_days=_h, close_only=True,
            )
    logger.info("Pre-extraction done for horizons {}", horizons)

    for target_h in horizons:
        logger.info("═══ Horizon {}D benchmark ═══", target_h)
        run_cfg = {**fold_config, "target_horizon_days": target_h}
        per_symbol = _per_symbol_by_h[target_h]
        per_symbol_365 = _per_symbol_365_by_h.get(target_h, {})

        all_results: list[dict[str, Any]] = []
        all_preds: dict[str, list[np.ndarray]] = {exp: [] for exp in experiments}
        all_y_true: list[np.ndarray] = []
        sym_order: list[str] = []
        all_splits: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        all_sh: dict[str, str] = {}
        all_training_logs: dict[str, dict[str, dict]] = {exp: {} for exp in experiments}

        # --------------------------------------------------------------
        # First pass: symbol splits + zero-shot
        # --------------------------------------------------------------
        for sym, data in per_symbol.items():
            splits = split_by_date(
                {k: v for k, v in data.items() if k != "times"},
                data["times"],
                run_cfg["train_end"],
                run_cfg["val_end"],
                target_horizon_days=target_h,
            )
            splits = impute_market_window_splits(splits)
            splits = impute_tabular_splits(splits)

            if len(splits["test"]["targets"]) == 0:
                continue

            sym_order.append(sym)
            all_splits[sym] = splits
            all_sh[sym] = _split_hash(splits, sym, target_h)
            all_y_true.append(splits["test"]["targets"])
            # Phase 0: persist per-symbol test truth for collapse diagnostics.
            _save_npy(CACHE_PRED_DIR / f"truth_{sym}_{target_h}d.npy", splits["test"]["targets"])

            if "Chronos Zero-Shot" in experiments:
                zs_source = per_symbol_365.get(sym, data)
                zs_splits = split_by_date(
                    {k: v for k, v in zs_source.items() if k != "times"},
                    zs_source["times"],
                    run_cfg["train_end"],
                    run_cfg["val_end"],
                    target_horizon_days=target_h,
                )

                zs_sh = _split_hash(zs_splits, sym, target_h)
                zs_cache = CACHE_PRED_DIR / f"zs_v3_{sym}_{target_h}d_{zs_sh}.npy"
                cached_zs = _load_npy(zs_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached_zs is not None:
                    preds_zs = cached_zs
                else:
                    preds_zs = chronos.zero_shot_predict(
                        zs_splits["test"]["close_windows"],
                        zs_splits["test"]["last_close"],
                        seed=config["seed"],
                        horizon=target_h,
                    )
                    _save_npy(zs_cache, preds_zs)

                all_preds["Chronos Zero-Shot"].append(preds_zs)

        if not sym_order:
            logger.warning("No symbols available for {}D", target_h)
            continue

        # E3: subsample the TRAINING split to remove redundant overlapping windows.
        # Only applied when --train-stride > 1 is explicitly set (default=0=disabled).
        # Aggressive strides cause model collapse at long horizons with few samples,
        # so no auto mode — the user must opt in explicitly.
        _eff_stride = args.train_stride
        if _eff_stride > 1:
            logger.info("{}D train-stride={} (was: {} train samples/sym → ~{})",
                        target_h, _eff_stride,
                        next(iter(all_splits.values()))["train"]["targets"].shape[0],
                        -(-next(iter(all_splits.values()))["train"]["targets"].shape[0] // _eff_stride))
            for _sym in all_splits:
                for _key in all_splits[_sym]["train"]:
                    all_splits[_sym]["train"][_key] = all_splits[_sym]["train"][_key][::_eff_stride]

        # --------------------------------------------------------------
        # HPO
        # --------------------------------------------------------------
        baseline_sym = sym_order[0]
        baseline_splits = all_splits[baseline_sym]

        baseline_hpo_params = load_or_run_baseline_hpo(
            CACHE_HPO_DIR,
            baseline_splits["train"]["market_windows"],
            baseline_splits["train"]["targets"],
            baseline_splits["val"]["market_windows"],
            baseline_splits["val"]["targets"],
            chronos,
            close_windows_train=baseline_splits["train"]["close_windows"],
            close_windows_val=baseline_splits["val"]["close_windows"],
            market_tabular_train=baseline_splits["train"].get("market_tabular"),
            market_tabular_val=baseline_splits["val"].get("market_tabular"),
            target_h=target_h,
            device=device,
            fallback_to_defaults=True,
        )
        if baseline_hpo_params is None:
            baseline_hpo_params = get_default_baseline_hpo_params()

        baseline_hpo_params.setdefault("mlp_summary", {
            "hidden_dim": 64, "dropout": 0.2, "lr": 1e-3, "batch_size": 32
        })
        baseline_hpo_params.setdefault("lstm_hybrid", baseline_hpo_params.get("lstm", {
            "hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 32
        }))
        baseline_hpo_params.setdefault("cnn_lstm_hybrid", baseline_hpo_params.get("cnn_lstm", {
            "num_filters": 64, "hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 32
        }))
        # Horizon-adaptive GPT4TS patch size:
        # - 1D/5D (short horizon): small patches (L3 stride1 → 28 tokens) for fine-grained temporal detail.
        # - 20D+ (long horizon): larger patches (L6 stride3 → ~9 tokens) for efficiency + macro patterns.
        _gpt4ts_patch_length = 6 if target_h >= 10 else 3
        _gpt4ts_patch_stride = 3 if target_h >= 10 else 1
        baseline_hpo_params.setdefault("gpt4ts", {
            "hidden_dim": GPT4TS_DEFAULTS.head_hidden_dim,
            "num_layers": GPT4TS_DEFAULTS.num_layers,
            "patch_length": _gpt4ts_patch_length,
            "patch_stride": _gpt4ts_patch_stride,
            "dropout": GPT4TS_DEFAULTS.dropout,
            "pooling": GPT4TS_DEFAULTS.pooling,
            "unfreeze_top_k_blocks": GPT4TS_DEFAULTS.unfreeze_top_k_blocks,
            "unfreeze_final_layer_norm": GPT4TS_DEFAULTS.unfreeze_final_layer_norm,
            "use_recent_residual": True,
            "backbone_lr": GPT4TS_DEFAULTS.backbone_lr,
            "head_lr": GPT4TS_DEFAULTS.head_lr,
            "batch_size": 32,
        })
        baseline_hpo_params.setdefault("gpt4ts_hybrid", {
            "hidden_dim": GPT4TS_DEFAULTS.head_hidden_dim,
            "num_layers": GPT4TS_DEFAULTS.num_layers,
            "patch_length": _gpt4ts_patch_length,
            "patch_stride": _gpt4ts_patch_stride,
            "dropout": GPT4TS_DEFAULTS.dropout,
            "pooling": GPT4TS_DEFAULTS.pooling,
            "unfreeze_top_k_blocks": GPT4TS_DEFAULTS.unfreeze_top_k_blocks,
            "unfreeze_final_layer_norm": GPT4TS_DEFAULTS.unfreeze_final_layer_norm,
            "use_recent_residual": True,
            "backbone_lr": GPT4TS_DEFAULTS.backbone_lr,
            "head_lr": GPT4TS_DEFAULTS.head_lr,
            "batch_size": 32,
        })

        # --------------------------------------------------------------
        # Trainable models
        # --------------------------------------------------------------
        # Horizon-adaptive loss weights: longer horizons need a stronger
        # directional signal and variance regulariser to avoid constant-sign
        # collapse caused by regression-to-mean on autocorrelated targets.
        # NOTE: changing these values invalidates cached predictions; use
        # --no-cache to force a fresh retrain after adjusting weights.
        _SPW_BY_HORIZON = {1: 0.3, 5: 0.5, 20: 0.7}
        _VRW_BY_HORIZON = {1: 0.0, 5: 0.02, 20: 0.05}
        effective_spw = _SPW_BY_HORIZON.get(target_h, config["sign_penalty_weight"])
        effective_vrw = _VRW_BY_HORIZON.get(target_h, 0.0)
        logger.info(
            "{}D adaptive loss | sign_penalty_weight={:.2f} | variance_reg_weight={:.3f}",
            target_h, effective_spw, effective_vrw,
        )
        lstm_anchor_by_sym: dict[str, np.ndarray] = {}

        for sym in sym_order:
            splits = all_splits[sym]
            sh = all_sh[sym]

            # ----------------------------------------------------------
            # Target standardisation: scale targets to unit std so that
            # loss margins and huber_delta operate at a consistent scale.
            # predict() divides back by target_scale, so metrics remain
            # in raw return units.
            # ----------------------------------------------------------
            _train_targets = splits["train"]["targets"]
            _train_std = float(np.std(_train_targets, ddof=1))
            target_scale = 1.0 / max(_train_std, 1e-6)
            ts_tag = hashlib.md5(f"{target_scale:.4f}".encode()).hexdigest()[:4]
            logger.info(
                "{} {}D target std={:.6f} → target_scale={:.2f} (ts={})",
                sym, target_h, _train_std, target_scale, ts_tag,
            )
            train_tab = extract_market_summary_features(splits["train"]["market_windows"])
            val_tab = extract_market_summary_features(splits["val"]["market_windows"])
            test_tab = extract_market_summary_features(splits["test"]["market_windows"])
            tabular_dim = train_tab.shape[1]

            # LSTM Baseline
            if "LSTM Baseline" in experiments:
                lstm_params = baseline_hpo_params["lstm"]
                lstm_hash = hashlib.md5(str(sorted(lstm_params.items())).encode()).hexdigest()[:8]
                lstm_cache = CACHE_PRED_DIR / f"lstm_{sym}_{target_h}d_{lstm_hash}_{sh}_{ts_tag}.npy"
                lstm_ckpt = CACHE_MODEL_DIR / f"lstm_{sym}_{target_h}d_{lstm_hash}_{sh}_{ts_tag}.pt"
                lstm_model = LSTMPredictor(
                    input_dim=splits["train"]["market_windows"].shape[-1],
                    hidden_dim=lstm_params.get("hidden_dim", 64),
                    num_layers=lstm_params.get("num_layers", 2),
                    dropout=lstm_params.get("dropout", 0.3),
                    sign_penalty_weight=effective_spw,
                    target_scale=target_scale,
                    device=device,
                )
                lstm_model.variance_reg_weight = effective_vrw
                cached = _load_npy(lstm_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None and lstm_ckpt.exists():
                    preds_lstm = cached
                    lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location=device, weights_only=False))
                    logs_lstm = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    logs_lstm = lstm_model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        epochs=100,
                        batch_size=lstm_params.get("batch_size", 32),
                        learning_rate=lstm_params.get("lr", 1e-3),
                        patience=15,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_lstm = lstm_model.predict(splits["test"]["market_windows"])
                    _save_npy(lstm_cache, preds_lstm)
                    lstm_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(lstm_model.state_dict(), lstm_ckpt)
                
                # Compute test metrics for logging
                test_metrics_lstm = compute_all(splits["test"]["targets"], preds_lstm, horizon=target_h)
                logs_lstm.update({
                    "test_mae": test_metrics_lstm.get("MAE", np.nan),
                    "test_rmse": test_metrics_lstm.get("RMSE", np.nan),
                    "test_da": test_metrics_lstm.get("DA%", np.nan),
                    "test_sharpe": test_metrics_lstm.get("Sharpe", np.nan),
                })
                all_training_logs["LSTM Baseline"][sym] = logs_lstm
                all_preds["LSTM Baseline"].append(preds_lstm)
                lstm_anchor_by_sym[sym] = preds_lstm

            # LSTM Hybrid
            if "LSTM Hybrid" in experiments:
                lstm_hybrid_params = baseline_hpo_params["lstm_hybrid"]
                lstm_hybrid_cache = CACHE_PRED_DIR / f"lstm_hybrid_{sym}_{target_h}d_{sh}_{ts_tag}.npy"
                cached = _load_npy(lstm_hybrid_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_lstm_hybrid = cached
                    logs_lstm_hybrid = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = LSTMHybridPredictor(
                        input_dim=splits["train"]["market_windows"].shape[-1],
                        tabular_dim=tabular_dim,
                        hidden_dim=lstm_hybrid_params.get("hidden_dim", 64),
                        num_layers=lstm_hybrid_params.get("num_layers", 2),
                        dropout=lstm_hybrid_params.get("dropout", 0.3),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_lstm_hybrid = model.fit(
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
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_lstm_hybrid = model.predict(splits["test"]["market_windows"], market_tabular=test_tab)
                    _save_npy(lstm_hybrid_cache, preds_lstm_hybrid)
                
                # Compute test metrics for logging
                test_metrics_lstm_h = compute_all(splits["test"]["targets"], preds_lstm_hybrid, horizon=target_h)
                logs_lstm_hybrid.update({
                    "test_mae": test_metrics_lstm_h.get("MAE", np.nan),
                    "test_rmse": test_metrics_lstm_h.get("RMSE", np.nan),
                    "test_da": test_metrics_lstm_h.get("DA%", np.nan),
                    "test_sharpe": test_metrics_lstm_h.get("Sharpe", np.nan),
                })
                all_training_logs["LSTM Hybrid"][sym] = logs_lstm_hybrid
                all_preds["LSTM Hybrid"].append(preds_lstm_hybrid)

            # Random Forest
            if "Random Forest Baseline" in experiments:
                rf_params = baseline_hpo_params["rf"]
                rf_cache = CACHE_PRED_DIR / f"rf_{sym}_{target_h}d_{sh}.npy"
                cached = _load_npy(rf_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_rf = cached
                    logs_rf = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = RandomForestRegressor_Wrapper(
                        n_estimators=rf_params.get("n_estimators", 100),
                        max_depth=rf_params.get("max_depth", 10),
                        min_samples_split=rf_params.get("min_samples_split", 5),
                        max_features=rf_params.get("max_features", "sqrt"),
                        random_state=42,
                    )
                    model.fit(splits["train"]["market_windows"], splits["train"]["targets"])
                    preds_rf = model.predict(splits["test"]["market_windows"])
                    _save_npy(rf_cache, preds_rf)
                    logs_rf = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                
                # Compute test metrics for logging
                test_metrics_rf = compute_all(splits["test"]["targets"], preds_rf, horizon=target_h)
                logs_rf.update({
                    "test_mae": test_metrics_rf.get("MAE", np.nan),
                    "test_rmse": test_metrics_rf.get("RMSE", np.nan),
                    "test_da": test_metrics_rf.get("DA%", np.nan),
                    "test_sharpe": test_metrics_rf.get("Sharpe", np.nan),
                })
                all_training_logs["Random Forest Baseline"][sym] = logs_rf
                all_preds["Random Forest Baseline"].append(preds_rf)

            # Linear Summary
            if "Linear Summary Baseline" in experiments:
                lin_cache = CACHE_PRED_DIR / f"linear_summary_{sym}_{target_h}d_{sh}.npy"
                cached = _load_npy(lin_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_lin = cached
                    logs_lin = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = LinearSummaryRegressor_Wrapper(alpha=1.0)
                    logs_lin = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                    )
                    preds_lin = model.predict(splits["test"]["market_windows"])
                    _save_npy(lin_cache, preds_lin)
                
                # Compute test metrics for logging
                test_metrics_lin = compute_all(splits["test"]["targets"], preds_lin, horizon=target_h)
                logs_lin.update({
                    "test_mae": test_metrics_lin.get("MAE", np.nan),
                    "test_rmse": test_metrics_lin.get("RMSE", np.nan),
                    "test_da": test_metrics_lin.get("DA%", np.nan),
                    "test_sharpe": test_metrics_lin.get("Sharpe", np.nan),
                })
                all_training_logs["Linear Summary Baseline"][sym] = logs_lin
                all_preds["Linear Summary Baseline"].append(preds_lin)

            # MLP Summary
            if "MLP Summary Baseline" in experiments:
                mlp_summary_params = baseline_hpo_params["mlp_summary"]
                mlp_cache = CACHE_PRED_DIR / f"mlp_summary_{sym}_{target_h}d_{sh}_{ts_tag}.npy"
                cached = _load_npy(mlp_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_mlp = cached
                    logs_mlp = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = MLPSummaryPredictor(
                        hidden_dim=mlp_summary_params.get("hidden_dim", 64),
                        dropout=mlp_summary_params.get("dropout", 0.2),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_mlp = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        epochs=100,
                        batch_size=mlp_summary_params.get("batch_size", 32),
                        learning_rate=mlp_summary_params.get("lr", 1e-3),
                        patience=15,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_mlp = model.predict(splits["test"]["market_windows"])
                    _save_npy(mlp_cache, preds_mlp)
                
                # Compute test metrics for logging
                test_metrics_mlp = compute_all(splits["test"]["targets"], preds_mlp, horizon=target_h)
                logs_mlp.update({
                    "test_mae": test_metrics_mlp.get("MAE", np.nan),
                    "test_rmse": test_metrics_mlp.get("RMSE", np.nan),
                    "test_da": test_metrics_mlp.get("DA%", np.nan),
                    "test_sharpe": test_metrics_mlp.get("Sharpe", np.nan),
                })
                all_training_logs["MLP Summary Baseline"][sym] = logs_mlp
                all_preds["MLP Summary Baseline"].append(preds_mlp)

            # CNN-LSTM
            if "CNN-LSTM" in experiments:
                cnn_params = baseline_hpo_params.get("cnn_lstm", baseline_hpo_params["lstm"])
                cnn_hash = hashlib.md5(str(sorted(cnn_params.items())).encode()).hexdigest()[:8]
                cnn_cache = CACHE_PRED_DIR / f"cnn_lstm_{sym}_{target_h}d_{cnn_hash}_{sh}_{ts_tag}.npy"
                cnn_ckpt = CACHE_MODEL_DIR / f"cnn_lstm_{sym}_{target_h}d_{cnn_hash}_{sh}_{ts_tag}.pt"
                cached = _load_npy(cnn_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None and cnn_ckpt.exists():
                    preds_cnn = cached
                    logs_cnn = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = CNNLSTMPredictor(
                        input_dim=splits["train"]["market_windows"].shape[-1],
                        num_filters=cnn_params.get("num_filters", cnn_params.get("hidden_dim", 64)),
                        hidden_dim=cnn_params.get("hidden_dim", 64),
                        num_layers=cnn_params.get("num_layers", 2),
                        dropout=cnn_params.get("dropout", 0.3),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_cnn = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        epochs=100,
                        batch_size=cnn_params.get("batch_size", 32),
                        learning_rate=cnn_params.get("lr", 1e-3),
                        patience=15,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_cnn = model.predict(splits["test"]["market_windows"])
                    _save_npy(cnn_cache, preds_cnn)
                    cnn_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), cnn_ckpt)
                
                # Compute test metrics for logging
                test_metrics_cnn = compute_all(splits["test"]["targets"], preds_cnn, horizon=target_h)
                logs_cnn.update({
                    "test_mae": test_metrics_cnn.get("MAE", np.nan),
                    "test_rmse": test_metrics_cnn.get("RMSE", np.nan),
                    "test_da": test_metrics_cnn.get("DA%", np.nan),
                    "test_sharpe": test_metrics_cnn.get("Sharpe", np.nan),
                })
                all_training_logs["CNN-LSTM"][sym] = logs_cnn
                all_preds["CNN-LSTM"].append(preds_cnn)

            # CNN-LSTM Hybrid
            if "CNN-LSTM Hybrid" in experiments:
                cnn_h_params = baseline_hpo_params["cnn_lstm_hybrid"]
                cnn_h_cache = CACHE_PRED_DIR / f"cnn_lstm_hybrid_{sym}_{target_h}d_{sh}_{ts_tag}.npy"
                cached = _load_npy(cnn_h_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_cnn_h = cached
                    logs_cnn_h = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = CNNLSTMHybridPredictor(
                        input_dim=splits["train"]["market_windows"].shape[-1],
                        tabular_dim=tabular_dim,
                        num_filters=cnn_h_params.get("num_filters", 64),
                        hidden_dim=cnn_h_params.get("hidden_dim", 64),
                        num_layers=cnn_h_params.get("num_layers", 2),
                        dropout=cnn_h_params.get("dropout", 0.3),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_cnn_h = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        market_tabular_train=train_tab,
                        market_tabular_val=val_tab,
                        epochs=100,
                        batch_size=cnn_h_params.get("batch_size", 32),
                        learning_rate=cnn_h_params.get("lr", 1e-3),
                        patience=15,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_cnn_h = model.predict(splits["test"]["market_windows"], market_tabular=test_tab)
                    _save_npy(cnn_h_cache, preds_cnn_h)
                
                # Compute test metrics for logging
                test_metrics_cnn_h = compute_all(splits["test"]["targets"], preds_cnn_h, horizon=target_h)
                logs_cnn_h.update({
                    "test_mae": test_metrics_cnn_h.get("MAE", np.nan),
                    "test_rmse": test_metrics_cnn_h.get("RMSE", np.nan),
                    "test_da": test_metrics_cnn_h.get("DA%", np.nan),
                    "test_sharpe": test_metrics_cnn_h.get("Sharpe", np.nan),
                })
                all_training_logs["CNN-LSTM Hybrid"][sym] = logs_cnn_h
                all_preds["CNN-LSTM Hybrid"].append(preds_cnn_h)

            # GPT4TS Baseline
            if "GPT4TS Baseline" in experiments:
                gpt_params = baseline_hpo_params.get("gpt4ts", {})
                gpt_hash = hashlib.md5(str(sorted(gpt_params.items())).encode()).hexdigest()[:8]
                gpt_cache = CACHE_PRED_DIR / f"gpt4ts_{sym}_{target_h}d_{gpt_hash}_{sh}_{ts_tag}.npy"
                gpt_ckpt = CACHE_MODEL_DIR / f"gpt4ts_{sym}_{target_h}d_{gpt_hash}_{sh}_{ts_tag}.pt"
                cached = _load_npy(gpt_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None and gpt_ckpt.exists():
                    preds_gpt = cached
                    logs_gpt = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = GPT4TSPredictor(
                        input_dim=splits["train"]["market_windows"].shape[-1],
                        hidden_dim=gpt_params.get("hidden_dim", GPT4TS_DEFAULTS.head_hidden_dim),
                        num_layers=gpt_params.get("num_layers", GPT4TS_DEFAULTS.num_layers),
                        patch_length=gpt_params.get("patch_length", GPT4TS_DEFAULTS.patch_length),
                        patch_stride=gpt_params.get("patch_stride", GPT4TS_DEFAULTS.patch_stride),
                        dropout=gpt_params.get("dropout", GPT4TS_DEFAULTS.dropout),
                        pooling=gpt_params.get("pooling", GPT4TS_DEFAULTS.pooling),
                        unfreeze_top_k_blocks=gpt_params.get("unfreeze_top_k_blocks", GPT4TS_DEFAULTS.unfreeze_top_k_blocks),
                        unfreeze_final_layer_norm=gpt_params.get("unfreeze_final_layer_norm", GPT4TS_DEFAULTS.unfreeze_final_layer_norm),
                        use_recent_residual=gpt_params.get("use_recent_residual", True),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_gpt = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        epochs=50,
                        batch_size=gpt_params.get("batch_size", 32),
                        backbone_lr=gpt_params.get("backbone_lr", GPT4TS_DEFAULTS.backbone_lr),
                        head_lr=gpt_params.get("head_lr", GPT4TS_DEFAULTS.head_lr),
                        patience=10,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_gpt = model.predict(splits["test"]["market_windows"])
                    _save_npy(gpt_cache, preds_gpt)
                    gpt_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), gpt_ckpt)
                
                # Compute test metrics for logging
                test_metrics_gpt = compute_all(splits["test"]["targets"], preds_gpt, horizon=target_h)
                logs_gpt.update({
                    "test_mae": test_metrics_gpt.get("MAE", np.nan),
                    "test_rmse": test_metrics_gpt.get("RMSE", np.nan),
                    "test_da": test_metrics_gpt.get("DA%", np.nan),
                    "test_sharpe": test_metrics_gpt.get("Sharpe", np.nan),
                })
                all_training_logs["GPT4TS Baseline"][sym] = logs_gpt
                all_preds["GPT4TS Baseline"].append(preds_gpt)

            # GPT4TS Hybrid
            if "GPT4TS Hybrid" in experiments:
                gpt_h_params = baseline_hpo_params.get("gpt4ts_hybrid", {})
                gpt_h_cache = CACHE_PRED_DIR / f"gpt4ts_hybrid_{sym}_{target_h}d_{sh}_{ts_tag}.npy"
                cached = _load_npy(gpt_h_cache) if _use_prediction_cache(stage, args.no_cache) else None
                if cached is not None:
                    preds_gpt_h = cached
                    logs_gpt_h = {
                        "train_losses": [],
                        "val_losses": [],
                        "val_losses_clean": [],
                        "best_val_loss": np.nan,
                        "pred_means": [],
                        "pred_pct_pos": [],
                        "pred_pct_neg": [],
                    }
                else:
                    model = GPT4TSHybridPredictor(
                        input_dim=splits["train"]["market_windows"].shape[-1],
                        tabular_dim=tabular_dim,
                        hidden_dim=gpt_h_params.get("hidden_dim", GPT4TS_DEFAULTS.head_hidden_dim),
                        num_layers=gpt_h_params.get("num_layers", GPT4TS_DEFAULTS.num_layers),
                        patch_length=gpt_h_params.get("patch_length", GPT4TS_DEFAULTS.patch_length),
                        patch_stride=gpt_h_params.get("patch_stride", GPT4TS_DEFAULTS.patch_stride),
                        dropout=gpt_h_params.get("dropout", GPT4TS_DEFAULTS.dropout),
                        pooling=gpt_h_params.get("pooling", GPT4TS_DEFAULTS.pooling),
                        unfreeze_top_k_blocks=gpt_h_params.get("unfreeze_top_k_blocks", GPT4TS_DEFAULTS.unfreeze_top_k_blocks),
                        unfreeze_final_layer_norm=gpt_h_params.get("unfreeze_final_layer_norm", GPT4TS_DEFAULTS.unfreeze_final_layer_norm),
                        use_recent_residual=gpt_h_params.get("use_recent_residual", True),
                        sign_penalty_weight=effective_spw,
                        target_scale=target_scale,
                        device=device,
                    )
                    model.variance_reg_weight = effective_vrw
                    logs_gpt_h = model.fit(
                        splits["train"]["market_windows"],
                        splits["train"]["targets"],
                        splits["val"]["market_windows"],
                        splits["val"]["targets"],
                        market_tabular_train=train_tab,
                        market_tabular_val=val_tab,
                        epochs=50,
                        batch_size=gpt_h_params.get("batch_size", 32),
                        backbone_lr=gpt_h_params.get("backbone_lr", GPT4TS_DEFAULTS.backbone_lr),
                        head_lr=gpt_h_params.get("head_lr", GPT4TS_DEFAULTS.head_lr),
                        patience=10,
                        warmup_epochs=args.warmup_epochs,
                    )
                    preds_gpt_h = model.predict(splits["test"]["market_windows"], market_tabular=test_tab)
                    _save_npy(gpt_h_cache, preds_gpt_h)
                
                # Compute test metrics for logging
                test_metrics_gpt_h = compute_all(splits["test"]["targets"], preds_gpt_h, horizon=target_h)
                logs_gpt_h.update({
                    "test_mae": test_metrics_gpt_h.get("MAE", np.nan),
                    "test_rmse": test_metrics_gpt_h.get("RMSE", np.nan),
                    "test_da": test_metrics_gpt_h.get("DA%", np.nan),
                    "test_sharpe": test_metrics_gpt_h.get("Sharpe", np.nan),
                })
                all_training_logs["GPT4TS Hybrid"][sym] = logs_gpt_h
                all_preds["GPT4TS Hybrid"].append(preds_gpt_h)

        # --------------------------------------------------------------
        # Package per-symbol results
        # --------------------------------------------------------------
        for idx, sym in enumerate(sym_order):
            y_test = all_y_true[idx]
            anchor_pred = lstm_anchor_by_sym.get(sym)

            for exp_name in experiments:
                if exp_name == "LSTM Baseline":
                    anchor = None
                else:
                    anchor = anchor_pred

                all_results.append(
                    package_result(
                        exp_name,
                        sym,
                        target_h,
                        y_test,
                        all_preds[exp_name][idx],
                        anchor,
                    )
                )

        # --------------------------------------------------------------
        # AVG rows
        # --------------------------------------------------------------
        if len(sym_order) == 1:
            logger.warning(
                "AVG row for {}D contains only 1 symbol ({}). "
                "AVG metrics will equal that symbol's metrics — run with multiple symbols for a meaningful aggregate.",
                target_h, sym_order[0],
            )
        for exp_name in experiments:
            y_concat = np.concatenate(all_y_true)
            p_concat = np.concatenate(all_preds[exp_name])

            if exp_name == "LSTM Baseline":
                a_concat = None
            else:
                available_anchor = [lstm_anchor_by_sym[sym] for sym in sym_order if sym in lstm_anchor_by_sym]
                a_concat = np.concatenate(available_anchor) if available_anchor else None

            all_results.append(
                package_result(
                    exp_name,
                    "AVG",
                    target_h,
                    y_concat,
                    p_concat,
                    a_concat,
                )
            )

        final_df = pd.DataFrame(all_results)

        col_order = [
            "Experiment",
            "ComparisonSet",
            "InputRegime",
            "AdaptationRegime",
            "ContextLength",
            "Symbol",
            "TargetHorizonD",
            "MAE",
            "RMSE",
            "DA%",
            "DA_ind%",
            "base_rate_DA%",
            "DA_skill%",
            "ModalDisagreement",
            "TemporalLag",
            "CompositeScore",
            "Sharpe",
            "IC",
            "Prec",
            "Rec",
            "F1",
            "Prec_ind",
            "Rec_ind",
            "F1_ind",
            "ESS",
            "PredStd",
            "PredPosPct",
            "SignBias",
            "Degenerate",
        ]
        final_df = final_df[col_order]

        csv_path = RESULTS_DIR / f"model_benchmark_{target_h}d.csv"
        final_df.to_csv(csv_path, index=False)
        logger.info("Results saved → {}", csv_path)

        # Warn when every model is below the naive majority baseline
        for sym_chk in final_df["Symbol"].unique():
            if sym_chk == "AVG":
                continue
            sub_chk = final_df[(final_df["Symbol"] == sym_chk) & (final_df["AdaptationRegime"] == "trained")]
            if len(sub_chk) > 0 and (sub_chk["DA_skill%"] < 0).all():
                logger.warning(
                    "NO SKILL: {}/{} — all trained models below majority baseline "
                    "(max DA_skill%={:.2f}%, ESS={}). "
                    "Consistent with regime shift or insufficient signal.",
                    sym_chk, target_h,
                    sub_chk["DA_skill%"].max(),
                    int(sub_chk["ESS"].iloc[0]),
                )
        # Warn on very low ESS
        ess_val = int(final_df[final_df["Symbol"] != "AVG"]["ESS"].iloc[0])
        if ess_val < 30:
            logger.warning(
                "LOW ESS at {}D: ESS={} — directional conclusions rest on very few "
                "statistically independent samples; interpret with caution.",
                target_h, ess_val,
            )

        # Save training logs for all models
        save_training_logs(all_training_logs, target_h, results_dir=RESULTS_DIR)

        avg_df = final_df[final_df["Symbol"] == "AVG"]
        plot_ablation(avg_df, FIGURES_DIR / f"model_{target_h}d.png")
        plot_per_symbol(final_df, FIGURES_DIR / f"per_symbol_heatmap_{target_h}d.png")

    logger.info("═══ Benchmark complete ═══")


