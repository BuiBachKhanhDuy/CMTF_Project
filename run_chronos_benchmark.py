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
from src.benchmark.metrics import compute_all, compute_composite_metrics
from src.benchmark.chronos_market import ChronosMarketPredictor
from src.benchmark.chronos_cmtf import ChronosCMTFPredictor, CMTF_VERSION
from src.benchmark.baseline_models import (
    ChronosLoRAPredictor,
    LSTMPredictor,
    RandomForestRegressor_Wrapper,
    FineTunedChronosPredictor,
)
from src.benchmark.baseline_hpo import get_default_baseline_hpo_params, load_or_run_baseline_hpo

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_EMB_DIR = Path("cache/chronos_emb")
CACHE_PRED_DIR = Path("cache/predictions")
CACHE_CMTF_DIR = Path("cache/cmtf_models")
CACHE_HPO_DIR = Path("cache/optuna")


def _cmtf_hpo_cache_file(target_h: int) -> Path:
    """Versioned CMTF HPO cache path for stable reproducibility."""
    return CACHE_HPO_DIR / f"best_params_v7_{target_h}d.json"


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


def _tokenize_chronos_windows(
    chronos: ChronosMarketPredictor,
    close_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    close_tensor = torch.as_tensor(close_windows, dtype=torch.float32)
    if close_tensor.ndim != 2:
        raise ValueError("close_windows must have shape (N, seq_len) for Chronos tokenization")
    token_ids, attention_mask, _ = chronos.pipeline.tokenizer.context_input_transform(close_tensor)
    return token_ids.cpu().numpy(), attention_mask.cpu().numpy()


def _load_or_cache_chronos_tokens(
    chronos: ChronosMarketPredictor,
    cache_base: Path,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_windows: np.ndarray,
    sym: str,
) -> dict[str, np.ndarray]:
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_ids": cache_base.with_name(f"{cache_base.name}_train_ids.npy"),
        "train_mask": cache_base.with_name(f"{cache_base.name}_train_mask.npy"),
        "val_ids": cache_base.with_name(f"{cache_base.name}_val_ids.npy"),
        "val_mask": cache_base.with_name(f"{cache_base.name}_val_mask.npy"),
        "test_ids": cache_base.with_name(f"{cache_base.name}_test_ids.npy"),
        "test_mask": cache_base.with_name(f"{cache_base.name}_test_mask.npy"),
    }
    cached = {name: _load_npy(path) for name, path in paths.items()}
    if any(value is None for value in cached.values()):
        logger.info("[{}] Tokenizing Chronos windows for LoRA-backed models…", sym)
        train_ids, train_mask = _tokenize_chronos_windows(chronos, train_windows)
        val_ids, val_mask = _tokenize_chronos_windows(chronos, val_windows)
        test_ids, test_mask = _tokenize_chronos_windows(chronos, test_windows)
        cached = {
            "train_ids": train_ids,
            "train_mask": train_mask,
            "val_ids": val_ids,
            "val_mask": val_mask,
            "test_ids": test_ids,
            "test_mask": test_mask,
        }
        for name, path in paths.items():
            _save_npy(path, cached[name])
    else:
        logger.info("[{}] Chronos token cache loaded", sym)
    return cached


def _build_ft_chronos_model(
    chronos: ChronosMarketPredictor,
    ft_params: dict,
    market_input_dim: int,
    device: str,
) -> ChronosLoRAPredictor:
    return ChronosLoRAPredictor(
        chronos,
        hidden_dim=ft_params.get("hidden_dim", 128),
        dropout=ft_params.get("dropout", 0.2),
        market_input_dim=market_input_dim,
        market_hidden_dim=ft_params.get("market_hidden_dim", 64),
        tabular_dim=0,
        sign_penalty_weight=ft_params.get("sign_penalty_weight", 0.05),
        lora_rank=ft_params.get("lora_rank", 4),
        lora_alpha=ft_params.get("lora_alpha", 8),
        lora_dropout=ft_params.get("lora_dropout", 0.0),
        device=device,
    )


def _load_or_train_ft_chronos_model(
    chronos: ChronosMarketPredictor,
    sym: str,
    target_h: int,
    ft_params: dict,
    split_hash: str,
    sym_tokens: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    device: str,
    stage: str | None,
    return_test_predictions: bool,
) -> tuple[ChronosLoRAPredictor, np.ndarray | None, str]:
    ft_param_hash = hashlib.md5(str(sorted(ft_params.items())).encode()).hexdigest()[:8]
    pred_cache = CACHE_PRED_DIR / f"ft_chronos_lora_v5_{sym}_{target_h}d_{ft_param_hash}_{split_hash}.npy"
    ckpt_cache = CACHE_CMTF_DIR / f"ft_chronos_lora_backbone_v3_{sym}_{target_h}d_{ft_param_hash}_{split_hash}.pt"
    ckpt_cache.parent.mkdir(parents=True, exist_ok=True)

    model = _build_ft_chronos_model(
        chronos,
        ft_params,
        market_input_dim=splits["train"]["market_windows"].shape[-1],
        device=device,
    )

    can_load_cache = stage not in ("data", "predict")
    loaded_from_checkpoint = False
    if can_load_cache and ckpt_cache.exists():
        model.load_checkpoint_state(
            torch.load(ckpt_cache, map_location=device, weights_only=False)
        )
        loaded_from_checkpoint = True
        logger.info("[{}] Chronos LoRA backbone loaded from checkpoint", sym)
    else:
        logger.info("[{}] Training Chronos LoRA market backbone…", sym)
        model.fit_tokenized(
            sym_tokens["train_ids"],
            sym_tokens["train_mask"],
            splits["train"]["targets"],
            sym_tokens["val_ids"],
            sym_tokens["val_mask"],
            splits["val"]["targets"],
            market_windows_train=splits["train"]["market_windows"],
            market_windows_val=splits["val"]["market_windows"],
            epochs=25,
            batch_size=32,
            learning_rate=ft_params.get("lr", 1e-4),
            patience=5,
        )
        torch.save(model.checkpoint_state(), ckpt_cache)
        logger.info("[{}] Chronos LoRA backbone checkpoint saved", sym)

    test_preds = None
    if return_test_predictions:
        if can_load_cache and loaded_from_checkpoint and (cached_preds := _load_npy(pred_cache)) is not None:
            test_preds = cached_preds
            logger.info("[{}] Chronos LoRA fine-tuned baseline loaded from cache", sym)
        else:
            test_preds = model.predict_tokenized(
                sym_tokens["test_ids"],
                sym_tokens["test_mask"],
                market_windows=splits["test"]["market_windows"],
            )
            _save_npy(pred_cache, test_preds)

    return model, test_preds, ft_param_hash


def _run_optuna_hpo(
    chronos: ChronosMarketPredictor,
    all_symbol_splits: dict[str, dict[str, dict[str, np.ndarray]]],
    all_symbol_tokens: dict[str, dict[str, np.ndarray]],
    all_symbol_anchor_val_preds: dict[str, np.ndarray],
    target_h: int,
    use_tabular: bool,
    device: str,
    n_trials: int = 30,
    seq_len: int = 30,
    ft_backbone_params: dict | None = None,
) -> dict:
    """Run Optuna HPO across all symbols jointly.

    Objective: composite validation score with regression dominance,
    directional penalty, and temporal lag.
    Searched hyperparameters:
        fusion_dim ∈ {32, 64, 128}
        n_heads ∈ {1, 2, 4}
        lr ∈ [1e-4, 1e-2] (log-uniform)
        dir_penalty_weight ∈ [0.01, 0.20]
        dropout ∈ [0.1, 0.5]

    Returns:
        Best hyperparameter dict.
    """
    import optuna
    import json

    CACHE_HPO_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cmtf_hpo_cache_file(target_h)

    # Check for cached result
    if cache_file.exists():
        with open(cache_file) as f:
            best = json.load(f)
        logger.info("Optuna HPO loaded from cache: {}", best)
        return best

    ft_backbone_params = ft_backbone_params or get_default_baseline_hpo_params()["finetuned_chronos"]

    def objective(trial: optuna.Trial) -> float:
        fusion_dim = trial.suggest_categorical("fusion_dim", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        if target_h >= 20:
            lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        else:
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        if target_h >= 20:
            dir_penalty_weight = trial.suggest_float("dir_penalty_weight", 0.03, 0.20)
        else:
            dir_penalty_weight = trial.suggest_float("dir_penalty_weight", 0.01, 0.10)
        if target_h >= 20:
            dropout = trial.suggest_float("dropout", 0.1, 0.35)
        else:
            dropout = trial.suggest_float("dropout", 0.1, 0.5)

        total_objective = 0.0
        n_symbols = 0

        for sym, splits in all_symbol_splits.items():
            tokens = all_symbol_tokens[sym]
            sh = _split_hash(splits, sym, target_h)
            tab_dim = (
                int(splits["train"].get("market_tabular", np.zeros((1, 0))).shape[1])
                if use_tabular else 0
            )
            news_dim = int(splits["train"]["news_embs"].shape[-1])
            ft_chronos_model, _, _ = _load_or_train_ft_chronos_model(
                chronos,
                sym=sym,
                target_h=target_h,
                ft_params=ft_backbone_params,
                split_hash=sh,
                sym_tokens=tokens,
                splits=splits,
                device=device,
                stage=None,
                return_test_predictions=False,
            )
            cmtf = ChronosCMTFPredictor(
                ft_chronos_model, news_dim=news_dim, tabular_dim=tab_dim, device=device,
                fusion_dim=fusion_dim, n_heads=n_heads, dropout=dropout,
                dir_penalty_weight=dir_penalty_weight,
                seq_len=seq_len,
                freeze_backbone=False,
            )
            cmtf.fit_tokenized(
                tokens["train_ids"], tokens["train_mask"],
                splits["train"]["news_embs"],
                splits["train"]["targets"],
                tokens["val_ids"], tokens["val_mask"],
                splits["val"]["news_embs"],
                splits["val"]["targets"],
                tabular_train=splits["train"].get("market_tabular") if use_tabular else None,
                tabular_val=splits["val"].get("market_tabular") if use_tabular else None,
                market_windows_train=splits["train"].get("market_windows"),
                market_windows_val=splits["val"].get("market_windows"),
                news_mask_train=splits["train"].get("news_masks"),
                news_mask_val=splits["val"].get("news_masks"),
                seed=42, lr=lr, epochs=12, patience=4, batch_size=32,
            )
            val_preds = cmtf.predict_tokenized(
                tokens["val_ids"], tokens["val_mask"],
                splits["val"]["news_embs"],
                tabular_test=splits["val"].get("market_tabular") if use_tabular else None,
                market_windows_test=splits["val"].get("market_windows"),
                news_mask_test=splits["val"].get("news_masks"),
            )
            total_objective += compute_composite_metrics(
                splits["val"]["targets"],
                val_preds,
                horizon=target_h,
                anchor_pred=all_symbol_anchor_val_preds.get(sym),
            )["CompositeScore"]
            n_symbols += 1

        return total_objective / max(n_symbols, 1)

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
    """Build per-symbol arrays of close windows, market windows, news embeddings, targets.

    Returns:
        {symbol: {'close_windows': (N, seq_len),
                  'market_windows': (N, seq_len, n_market_features),
                  'last_close': (N,),
                  'market_tabular': (N, n_market_features),
                  'news_embs': (N, seq_len, news_dim),
                  'news_masks': (N, seq_len),
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
        news_col = "news_hybrid_emb" if "news_hybrid_emb" in sym_df.columns else "news_emb"
        news_list = sym_df[news_col].tolist()
        news_arr = np.stack(news_list).astype(np.float32)  # (n, 768)
        has_news_arr = sym_df["has_news"].astype(bool).to_numpy(copy=True)

        # Engineered market features from pipeline (OHLCV + technical indicators)
        market_cols = list(getattr(dataset, "market_cols", []))
        market_values = sym_df[market_cols].values.astype(np.float32)

        # Forward returns (target)
        target_col_name = f"fwd_ret_{int(target_horizon_days)}d"
        if target_col_name not in sym_df.columns:
            logger.warning("{} missing target column {} — skipping", sym, target_col_name)
            continue
        target_col = sym_df[target_col_name].values.astype(np.float32)

        # Build windows
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
            window = raw_c[i - seq_len + 1 : i + 1]   # (seq_len,)
            market_window = market_values[i - seq_len + 1 : i + 1]  # (seq_len, n_market_features)
            # Keep per-bar news embeddings as a sequence for cross-attention
            news_window = news_arr[i - seq_len + 1 : i + 1]  # (seq_len, 768)
            news_mask_window = ~has_news_arr[i - seq_len + 1 : i + 1]  # True = no news

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

        result[sym] = {
            "close_windows": np.array(close_windows),
            "market_windows": np.array(market_windows),
            "last_close": np.array(last_closes),
            "market_tabular": np.array(market_tabs),
            "news_embs": np.array(news_embs),
            "news_masks": np.array(news_masks, dtype=bool),
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


def impute_market_window_splits(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Impute NaNs in market_windows using train-only feature means.

    Market windows feed the LSTM and Random Forest baselines directly, so they
    need the same train-only NaN handling as the tabular branch.
    """
    if "market_windows" not in splits.get("train", {}):
        return splits

    train_windows = splits["train"]["market_windows"]
    if train_windows.size == 0:
        return splits

    reduce_axes = tuple(range(train_windows.ndim - 1))
    feature_means = np.nanmean(train_windows, axis=reduce_axes)
    feature_means = np.where(np.isnan(feature_means), 0.0, feature_means).astype(np.float32)
    fill_shape = (1,) * (train_windows.ndim - 1) + (feature_means.shape[0],)
    fill_values = feature_means.reshape(fill_shape)

    for split_name in ("train", "val", "test"):
        windows = splits[split_name].get("market_windows")
        if windows is None or windows.size == 0:
            continue
        splits[split_name]["market_windows"] = np.where(
            np.isnan(windows),
            fill_values,
            windows,
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
    
    # Extended color palette for up to 5 experiments
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    
    # Shorten labels for display
    label_map = {
        "Chronos Zero-Shot": "Zero-Shot",
        "CMTF": "CMTF",
        "LSTM Baseline": "LSTM",
        "Random Forest Baseline": "RF",
        "Chronos Fine-Tuned (LoRA)": "Chronos FT LoRA",
    }
    short_labels = [label_map.get(exp, exp[:15]) for exp in experiments]

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
    legend_handles = [Patch(facecolor=colors[i % len(colors)], label=l) for i, l in enumerate(short_labels)]
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(n_exp, 5), fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Chronos Ablation Benchmark (AVG)", fontsize=14,
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
    parser.add_argument(
        "--comparison-set",
        choices=["base", "full"],
        default="base",
        help="Benchmark scope: 'base' skips all CMTF models, 'full' includes CMTF.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional list of target horizons to run, e.g. --horizons 20",
    )
    args = parser.parse_args()
    stage = args.stage  # None = full run
    comparison_set = args.comparison_set
    include_cmtf = comparison_set == "full"

    if stage in ("cmtf", "hpo") and not include_cmtf:
        raise ValueError("CMTF-only stages require --comparison-set full")

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
        "use_tabular_market_features": True,
    }

    # Override for --stage data: force rebuild
    if stage == "data":
        config["rebuild_data"] = True
    if args.horizons:
        config["target_horizons_days"] = [int(h) for h in args.horizons]

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
    if use_tabular_market_features:
        logger.info(
            "Active trainable models share OHLCV + technical-indicator inputs; zero-shot remains a close-only anchor"
        )
    else:
        logger.warning(
            "Tabular market features disabled; trainable baselines fall back to close-only inputs"
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
            "LSTM Baseline": [],
            "Random Forest Baseline": [],
            "Chronos Fine-Tuned (LoRA)": [],
        }
        if include_cmtf:
            all_preds["CMTF"] = []
        all_y_true: list[np.ndarray] = []

        # ---- Phase 1: splits, zero-shot anchor, token cache ----
        _all_splits: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        _all_tokens: dict[str, dict[str, np.ndarray]] = {}
        _all_sh: dict[str, str] = {}
        _all_zs_val_preds: dict[str, np.ndarray] = {}
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
            splits = impute_market_window_splits(splits)
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

            zs_val_cache = CACHE_PRED_DIR / f"zs_val_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_zs_val := _load_npy(zs_val_cache)) is not None:
                preds_zs_val = cached_zs_val
            else:
                preds_zs_val = chronos.zero_shot_predict(
                    splits["val"]["close_windows"],
                    splits["val"]["last_close"],
                    seed=config["seed"],
                    horizon=target_h,
                )
                _save_npy(zs_val_cache, preds_zs_val)
            _all_zs_val_preds[sym] = preds_zs_val

            metrics_zs = compute_all(y_test, preds_zs, horizon=target_h)
            metrics_zs.update(
                compute_composite_metrics(
                    y_test,
                    preds_zs,
                    horizon=target_h,
                    anchor_pred=preds_zs,
                )
            )
            metrics_zs.update({
                "Experiment": "Chronos Zero-Shot",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_zs)
            all_preds["Chronos Zero-Shot"].append(preds_zs)

            token_cache_base = CACHE_EMB_DIR / f"chronos_lora_tokens_{sym}_{target_h}d_{sh}"
            _all_tokens[sym] = _load_or_cache_chronos_tokens(
                chronos,
                token_cache_base,
                splits["train"]["close_windows"],
                splits["val"]["close_windows"],
                splits["test"]["close_windows"],
                sym,
            )

        baseline_hpo_params = None
        if _sym_order:
            _baseline_sym = _sym_order[0]
            _baseline_splits = _all_splits[_baseline_sym]
            logger.info("Loading or running baseline HPO for {}D", target_h)
            baseline_hpo_params = load_or_run_baseline_hpo(
                CACHE_HPO_DIR,
                _baseline_splits["train"]["market_windows"],
                _baseline_splits["train"]["targets"],
                _baseline_splits["val"]["market_windows"],
                _baseline_splits["val"]["targets"],
                chronos,
                close_windows_train=_baseline_splits["train"]["close_windows"],
                close_windows_val=_baseline_splits["val"]["close_windows"],
                market_tabular_train=_baseline_splits["train"].get("market_tabular") if use_tabular_market_features else None,
                market_tabular_val=_baseline_splits["val"].get("market_tabular") if use_tabular_market_features else None,
                target_h=target_h,
                device=device,
                fallback_to_defaults=not include_cmtf,
            )

        # ---- Optuna HPO (once per horizon, across all symbols) ----
        cmtf_hpo_params = {}
        if include_cmtf:
            _cmtf_hpo_cache = _cmtf_hpo_cache_file(target_h)
            if stage == "hpo" or (stage in (None, "cmtf") and not _cmtf_hpo_cache.exists()):
                logger.info("═══ Running Optuna HPO for {}D ═══", target_h)
                cmtf_hpo_params = _run_optuna_hpo(
                    chronos, _all_splits, _all_tokens, _all_zs_val_preds, target_h,
                    use_tabular=use_tabular_market_features,
                    device=device, n_trials=6, seq_len=run_cfg["sequence_len"],
                    ft_backbone_params=(baseline_hpo_params or {}).get("finetuned_chronos"),
                )
            elif _cmtf_hpo_cache.exists():
                import json
                with open(_cmtf_hpo_cache) as f:
                    cmtf_hpo_params = json.load(f)
                logger.info("Using cached HPO params ({}D): {}", target_h, cmtf_hpo_params)

        if stage == "hpo":
            logger.info("═══ HPO-only mode complete for {}D ═══", target_h)
            continue

        # ---- Phase 2: trainable baselines and optional CMTF ----
        _ensemble_seeds = [42, 123, 456]
        for sym in _sym_order:
            splits = _all_splits[sym]
            sh = _all_sh[sym]
            sym_tokens = _all_tokens[sym]
            y_test = splits["test"]["targets"]

            ft_chronos_model = None
            preds_ft_chronos = None
            ft_param_hash = None
            if include_cmtf:
                ft_chronos_model, preds_ft_chronos, ft_param_hash = _load_or_train_ft_chronos_model(
                    chronos,
                    sym=sym,
                    target_h=target_h,
                    ft_params=baseline_hpo_params["finetuned_chronos"],
                    split_hash=sh,
                    sym_tokens=sym_tokens,
                    splits=splits,
                    device=device,
                    stage=stage,
                    return_test_predictions=True,
                )

            if include_cmtf:
                logger.info("[{}] Running CMTF (unfrozen LoRA Chronos + fusion, ensemble × {} seeds) …", sym, len(_ensemble_seeds))
                tab_dim = (
                    int(splits["train"].get("market_tabular", np.zeros((1, 0))).shape[1])
                    if use_tabular_market_features else 0
                )
                news_dim = int(splits["train"]["news_embs"].shape[-1])

                cmtf_fusion_dim = cmtf_hpo_params.get("fusion_dim", 128)
                cmtf_n_heads = cmtf_hpo_params.get("n_heads", 1)
                cmtf_dropout = cmtf_hpo_params.get("dropout", 0.2)
                cmtf_dir_penalty_weight = cmtf_hpo_params.get("dir_penalty_weight", 0.08 if target_h >= 20 else 0.05)
                cmtf_lr = cmtf_hpo_params.get("lr", 5e-4)

                import json as _json
                _hpo_hash = hashlib.md5(
                    _json.dumps(cmtf_hpo_params, sort_keys=True).encode()
                ).hexdigest()[:8]

                _ensemble_preds: list[np.ndarray] = []
                for _seed in _ensemble_seeds:
                    _ckpt_path = CACHE_CMTF_DIR / f"cmtf_lora_{CMTF_VERSION}_{sym}_{target_h}d_seed{_seed}_{_hpo_hash}_{ft_param_hash}_{sh}.pt"
                    _ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    _cmtf = ChronosCMTFPredictor(
                        ft_chronos_model, news_dim=news_dim, tabular_dim=tab_dim, device=device,
                        fusion_dim=cmtf_fusion_dim, n_heads=cmtf_n_heads,
                        dropout=cmtf_dropout,
                        dir_penalty_weight=cmtf_dir_penalty_weight,
                        seq_len=run_cfg["sequence_len"],
                        freeze_backbone=False,
                    )

                    if _ckpt_path.exists() and stage not in ("cmtf", "hpo"):
                        _cmtf.load_checkpoint(
                            torch.load(_ckpt_path, map_location=device, weights_only=False)
                        )
                        logger.info("[{}] CMTF seed {} loaded from checkpoint", sym, _seed)
                    else:
                        _cmtf.fit_tokenized(
                            sym_tokens["train_ids"], sym_tokens["train_mask"],
                            splits["train"]["news_embs"],
                            splits["train"]["targets"],
                            sym_tokens["val_ids"], sym_tokens["val_mask"],
                            splits["val"]["news_embs"],
                            splits["val"]["targets"],
                            tabular_train=splits["train"].get("market_tabular") if use_tabular_market_features else None,
                            tabular_val=splits["val"].get("market_tabular") if use_tabular_market_features else None,
                            market_windows_train=splits["train"].get("market_windows"),
                            market_windows_val=splits["val"].get("market_windows"),
                            news_mask_train=splits["train"].get("news_masks"),
                            news_mask_val=splits["val"].get("news_masks"),
                            seed=_seed, lr=cmtf_lr, epochs=25, patience=8,
                        )
                        torch.save(_cmtf.get_checkpoint(), _ckpt_path)
                        logger.info("[{}] CMTF seed {} checkpoint saved", sym, _seed)

                    _ensemble_preds.append(_cmtf.predict_tokenized(
                        sym_tokens["test_ids"], sym_tokens["test_mask"],
                        splits["test"]["news_embs"],
                        tabular_test=splits["test"].get("market_tabular") if use_tabular_market_features else None,
                        market_windows_test=splits["test"].get("market_windows"),
                        news_mask_test=splits["test"].get("news_masks"),
                    ))
                preds_cmtf = np.mean(_ensemble_preds, axis=0)
                metrics_cmtf = compute_all(y_test, preds_cmtf, horizon=target_h)
                metrics_cmtf.update(
                    compute_composite_metrics(
                        y_test,
                        preds_cmtf,
                        horizon=target_h,
                        anchor_pred=all_preds["Chronos Zero-Shot"][_sym_order.index(sym)],
                    )
                )
                metrics_cmtf.update({
                    "Experiment": "CMTF",
                    "ComparisonSet": "fairness",
                    "Symbol": sym,
                    "TargetHorizonD": target_h,
                })
                all_results.append(metrics_cmtf)
                all_preds["CMTF"].append(preds_cmtf)

            # ---- Baseline Models: LSTM and Random Forest ----
            logger.info("[{}] Training baseline models (LSTM, Random Forest) …", sym)
            
            # --- LSTM Baseline ---
            lstm_cache = CACHE_PRED_DIR / f"lstm_{sym}_{target_h}d_{sh}.npy"
            lstm_params = baseline_hpo_params["lstm"]
            lstm_param_hash = hashlib.md5(str(sorted(lstm_params.items())).encode()).hexdigest()[:8]
            lstm_backbone_ckpt = CACHE_CMTF_DIR / f"lstm_backbone_v3_{sym}_{target_h}d_{lstm_param_hash}_{sh}.pt"
            lstm_backbone_ckpt.parent.mkdir(parents=True, exist_ok=True)
            lstm_model = LSTMPredictor(
                input_dim=splits["train"]["market_windows"].shape[-1],
                hidden_dim=lstm_params.get("hidden_dim", 64),
                num_layers=lstm_params.get("num_layers", 2),
                dropout=lstm_params.get("dropout", 0.3),
                device=device,
            )
            cached_lstm = _load_npy(lstm_cache)
            cached_lstm_is_finite = cached_lstm is not None and np.isfinite(cached_lstm).all()
            if cached_lstm is not None and not cached_lstm_is_finite:
                logger.warning("[{}] LSTM cache contains non-finite values; retraining", sym)
            if stage not in ("data", "predict") and cached_lstm_is_finite and lstm_backbone_ckpt.exists():
                logger.info("[{}] LSTM loaded from cache", sym)
                preds_lstm = cached_lstm
                lstm_model.load_state_dict(
                    torch.load(lstm_backbone_ckpt, map_location=device, weights_only=False)
                )
            else:
                logger.info("[{}] Training LSTM with HPO params…", sym)
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
                torch.save(lstm_model.state_dict(), lstm_backbone_ckpt)
            
            metrics_lstm = compute_all(y_test, preds_lstm, horizon=target_h)
            metrics_lstm.update(
                compute_composite_metrics(
                    y_test,
                    preds_lstm,
                    horizon=target_h,
                    anchor_pred=all_preds["Chronos Zero-Shot"][_sym_order.index(sym)],
                )
            )
            metrics_lstm.update({
                "Experiment": "LSTM Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_lstm)
            all_preds["LSTM Baseline"].append(preds_lstm)
            
            # --- Random Forest Baseline ---
            rf_cache = CACHE_PRED_DIR / f"rf_{sym}_{target_h}d_{sh}.npy"
            if stage not in ("data", "predict") and (cached_rf := _load_npy(rf_cache)) is not None:
                logger.info("[{}] Random Forest loaded from cache", sym)
                preds_rf = cached_rf
            else:
                logger.info("[{}] Training Random Forest with HPO params…", sym)
                rf_params = baseline_hpo_params["rf"]
                rf_model = RandomForestRegressor_Wrapper(
                    n_estimators=rf_params.get("n_estimators", 100),
                    max_depth=rf_params.get("max_depth", 10),
                    min_samples_split=rf_params.get("min_samples_split", 5),
                    max_features=rf_params.get("max_features", "sqrt"),
                    random_state=42,
                )
                rf_model.fit(
                    splits["train"]["market_windows"],
                    splits["train"]["targets"],
                )
                preds_rf = rf_model.predict(splits["test"]["market_windows"])
                _save_npy(rf_cache, preds_rf)
            
            metrics_rf = compute_all(y_test, preds_rf, horizon=target_h)
            metrics_rf.update(
                compute_composite_metrics(
                    y_test,
                    preds_rf,
                    horizon=target_h,
                    anchor_pred=all_preds["Chronos Zero-Shot"][_sym_order.index(sym)],
                )
            )
            metrics_rf.update({
                "Experiment": "Random Forest Baseline",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_rf)
            all_preds["Random Forest Baseline"].append(preds_rf)
            
            # --- Fine-tuned Chronos Baseline ---
            if ft_chronos_model is None or preds_ft_chronos is None:
                ft_chronos_model, preds_ft_chronos, _ = _load_or_train_ft_chronos_model(
                    chronos,
                    sym=sym,
                    target_h=target_h,
                    ft_params=baseline_hpo_params["finetuned_chronos"],
                    split_hash=sh,
                    sym_tokens=sym_tokens,
                    splits=splits,
                    device=device,
                    stage=stage,
                    return_test_predictions=True,
                )
            
            metrics_ft_chronos = compute_all(y_test, preds_ft_chronos, horizon=target_h)
            metrics_ft_chronos.update(
                compute_composite_metrics(
                    y_test,
                    preds_ft_chronos,
                    horizon=target_h,
                    anchor_pred=all_preds["Chronos Zero-Shot"][_sym_order.index(sym)],
                )
            )
            metrics_ft_chronos.update({
                "Experiment": "Chronos Fine-Tuned (LoRA)",
                "ComparisonSet": "fairness",
                "Symbol": sym,
                "TargetHorizonD": target_h,
            })
            all_results.append(metrics_ft_chronos)
            all_preds["Chronos Fine-Tuned (LoRA)"].append(preds_ft_chronos)

            # Per-symbol prediction plot (use index from _sym_order)
            _sym_idx = _sym_order.index(sym)
            plot_predictions(
                y_test,
                {
                    "Zero-Shot": all_preds["Chronos Zero-Shot"][_sym_idx],
                    "LSTM": preds_lstm,
                    "RF": preds_rf,
                    "Chronos FT LoRA": preds_ft_chronos,
                    **({"CMTF": all_preds["CMTF"][_sym_idx]} if include_cmtf else {}),
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
            anchor_parts: list[np.ndarray] = []
            for sym_name in per_symbol:
                sym_rows = results_df[
                    (results_df["Experiment"] == exp_name) & (results_df["Symbol"] == sym_name)
                ]
                if sym_rows.empty:
                    continue
                idx = list(per_symbol.keys()).index(sym_name)
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

        print("\n" + "=" * 76)
        print(f"  CHRONOS BENCHMARK RESULTS — TARGET HORIZON {target_h}D")
        print("=" * 76)
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
