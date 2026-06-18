"""Run ablation benchmark: model-agnostic grid over fusion × news × sentiment × components.

Usage:
    python run_ablation_benchmark.py                           # full run
    python run_ablation_benchmark.py --table fusion            # Table 1 only
    python run_ablation_benchmark.py --table component --horizons 1 5
    python run_ablation_benchmark.py --stage plot              # regenerate plots from CSVs
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
from loguru import logger
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.benchmark.chronos_encoder import ChronosMarketPredictor
from src.benchmark.ablation_config import AblationConfig, generate_grid
from src.benchmark.ablation_runner import run_ablation_cell
from src.benchmark.ablation_plots import plot_table_charts

_ABLATION_ROOT = Path("results/ablation")

_CONFIG_KEY_COLS = [
    "model_name",
    "fusion_type",
    "news_scope",
    "sentiment_mode",
    "use_positional_encoding",
    "use_news_gate",
    "recency_gate_k",
    "use_two_stage",
    "use_aux_loss",
    "use_variance_reg",
]

def _horizon_dir(horizon: int) -> Path:
    return _ABLATION_ROOT / f"{horizon}d"


def _figures_dir(horizon: int) -> Path:
    return _horizon_dir(horizon) / "figures"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru for clean CLI output.
    
    - Normal mode: warnings/errors only (clean output)
    - Verbose mode: all logs including debug
    """
    logger.remove()  # Remove default handler
    
    if verbose:
        # Verbose: show everything with timestamps
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<level>[{level: <8}]</level> <cyan>{name}:{function}:{line}</cyan> — {message}",
            colorize=True,
        )
    else:
        # Clean mode: only warnings, errors, and critical
        logger.add(
            sys.stderr,
            level="WARNING",
            format="<level>{level: <8}</level> {message}",
            colorize=True,
        )
        # Also capture errors with context but not debug logs
        logger.add(
            sys.stderr,
            level="ERROR",
            format="<level>[{level}]</level> <red>{name}:{function}:{line}</red> — {message}",
            colorize=True,
        )


def _build_pipeline_config(horizon: int) -> dict:
    """Standard pipeline config for ablation."""
    return {
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
        "news_export_trace": False,
        "news_sentiment_enabled": True,
        "news_sentiment_device": "cpu",
        "news_sentiment_export_trace": False,
        "phase2_output_dir": "outputs/phase2/latest",
        "news_similarity_threshold": 85.0,
        "log_news_coverage": False,
        "sequence_len": 30,
        "horizon": horizon,
        "target_horizons_days": [horizon],
        "target_horizon_days": horizon,
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        "stability_selection_enabled": False,
        "use_tabular_market_features": True,
    }


def _build_cross_symbol_news(
    all_data: dict[str, dict[str, np.ndarray]],
    seq_len: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build cross-symbol pooled news embeddings for the 'all' news_scope.

    For each date, uses learned attention to pool non-zero news embeddings across all symbols.
    Returns per-symbol arrays of cross-symbol pooled news windows + masks.
    
    Uses explicit timestamp-based lookups instead of fragile index arithmetic.
    Attention pooling learns which symbol embeddings to weight for cross-symbol fusion.
    """
    import torch
    import torch.nn.functional as F
    from collections import defaultdict

    # Step 1: Build per-symbol timelines and extract all per-bar embeddings with timestamps
    # For each symbol, reconstruct the full per-bar timeline from windowed data.
    date_embs_list: dict[str, list[np.ndarray]] = defaultdict(list)
    # Audit counters
    date_symbol_counts = defaultdict(int)
    date_unique_symbols = defaultdict(set)
    date_embs_by_sym: dict[str, dict[str, np.ndarray]] = defaultdict(dict)  # {date_key: {sym: emb}}
    
    for sym, data in all_data.items():
        times = data["times"]  # (N_samples,) — timestamp of last bar in each window
        news_embs = data["news_embs"]  # (N_samples, seq_len, 768)
        
        # Build per-symbol ordered timeline
        sym_timeline = pd.to_datetime(np.sort(np.unique(times)))
        sym_time_to_idx = {pd.Timestamp(t): idx for idx, t in enumerate(sym_timeline)}
        # Collect unique embeddings per bar for this symbol (one embedding per date, not per window)
        sym_embeddings = {}  # {bar_time_str: emb} — captures latest non-zero embedding
        
        for i, last_bar_time in enumerate(times):
            last_bar_idx = sym_time_to_idx[pd.Timestamp(last_bar_time)]
            
            for j in range(seq_len):
                # Bar j in window i corresponds to bar at index (last_bar_idx - (seq_len - 1 - j))
                # in the symbol's timeline (using arithmetic within the timeline, which is safe)
                bar_idx_in_timeline = last_bar_idx - (seq_len - 1 - j)
                
                if 0 <= bar_idx_in_timeline < len(sym_timeline):
                    bar_time = sym_timeline[bar_idx_in_timeline]
                    emb = news_embs[i, j, :768]
                    
                    if np.any(emb != 0):
                        date_key = pd.Timestamp(bar_time)
                        # Store only one embedding per symbol per date (use latest if appears in multiple windows)
                        sym_embeddings[date_key] = emb
        
        # Add this symbol's embeddings to the per-date aggregation
        for date_key, emb in sym_embeddings.items():
            date_embs_by_sym[date_key][sym] = emb
            date_symbol_counts[date_key] += 1
            date_unique_symbols[date_key].add(sym)
    
    # Convert to list format for pooling
    date_embs_list: dict[str, list[np.ndarray]] = {
        date_key: list(sym_dict.values())
        for date_key, sym_dict in date_embs_by_sym.items()
    }
    
    # ==========================================================
    # Audit duplicate counting
    # ==========================================================
    
    duplicate_dates = []
    
    for date_key in date_embs_list:
    
        total_embs = date_symbol_counts[date_key]
        unique_symbols = len(date_unique_symbols[date_key])
    
        if total_embs > unique_symbols:
            duplicate_dates.append(
                (
                    date_key,
                    total_embs,
                    unique_symbols,
                )
            )
    
    if duplicate_dates:
    
        logger.warning(
            "Cross-symbol news audit: {} dates contain duplicated embeddings",
            len(duplicate_dates),
        )
    
        for date_key, total_embs, unique_symbols in duplicate_dates[:10]:
        
            logger.warning(
                "Date={} total_embeddings={} unique_symbols={} duplication_factor={:.2f}",
                date_key,
                total_embs,
                unique_symbols,
                total_embs / max(unique_symbols, 1),
            )
    # Step 2: Attention-pool embeddings per date across symbols
    # Instead of simple mean, use learned attention weights
    pooled: dict[str, np.ndarray] = {}
    for date_key, embs_list in date_embs_list.items():
        if len(embs_list) == 1:
            # Only one symbol has news on this date
            pooled[date_key] = embs_list[0].astype(np.float32)
        else:
            # Multiple symbols: use attention-weighted pooling
            embs_tensor = torch.from_numpy(np.stack(embs_list)).float()  # (n_symbols, 768)
            
            # Compute attention weights via cosine similarity to mean embedding
            mean_emb = embs_tensor.mean(dim=0, keepdim=True)  # (1, 768)
            
            # Cosine similarity scores
            scores = F.cosine_similarity(mean_emb, embs_tensor, dim=1)  # (n_symbols,)
            
            # Softmax to get attention weights
            attn_weights = F.softmax(scores, dim=0)  # (n_symbols,)
            
            # Weighted combination
            pooled_emb = (embs_tensor * attn_weights.unsqueeze(1)).sum(dim=0)  # (768,)
            pooled[date_key] = pooled_emb.numpy().astype(np.float32)

    # Step 3: Rebuild per-symbol windowed arrays using timestamp-based lookups
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, data in all_data.items():
        times = data["times"]  # (N,)
        news_embs_orig = data["news_embs"]  # (N, seq_len, 768)
        N, S, D = news_embs_orig.shape
        D = min(D, 768)
        
        pooled_windows = np.zeros((N, S, D), dtype=np.float32)
        pooled_masks = np.ones((N, S), dtype=bool)  # True = no news

        # Build per-symbol timeline for explicit lookups
        sym_timeline = pd.to_datetime(np.sort(np.unique(times)))
        sym_time_to_idx = {pd.Timestamp(t): idx for idx, t in enumerate(sym_timeline)}

        # For each window, populate pooled embeddings via timestamp-based lookups
        for i in range(N):
            last_bar_time = times[i]
            last_bar_idx = sym_time_to_idx[pd.Timestamp(last_bar_time)]
            
            for j in range(S):
                # Bar j in window i is at index (last_bar_idx - (S - 1 - j)) in the timeline
                bar_idx_in_timeline = last_bar_idx - (S - 1 - j)
                
                if 0 <= bar_idx_in_timeline < len(sym_timeline):
                    bar_time = sym_timeline[bar_idx_in_timeline]
                    bar_time_key = pd.Timestamp(bar_time)
                    if bar_time_key in pooled:
                        pooled_windows[i, j, :] = pooled[bar_time_key]
                        pooled_masks[i, j] = False


        result[sym] = (pooled_windows, pooled_masks)
        covered = np.sum(~pooled_masks)
        total = pooled_masks.size
        coverage = covered / max(total, 1)

        logger.warning(
            "Cross-symbol coverage | sym={} covered_bars={} total_bars={} coverage={:.3f}",
            sym, covered, total, coverage
        )
    return result

def _log_news_stats(tag: str, news_embs: np.ndarray, news_masks: np.ndarray) -> None:
    zero_bar_ratio = float(np.mean(np.all(np.isclose(news_embs, 0.0), axis=-1)))
    all_mask_ratio = float(np.mean(news_masks))
    all_zero_window_ratio = float(np.mean(np.all(np.isclose(news_embs, 0.0), axis=(1, 2))))
    logger.warning(
        "{} | zero_bar_ratio={:.3f} all_mask_ratio={:.3f} all_zero_window_ratio={:.3f} shape={}",
        tag, zero_bar_ratio, all_mask_ratio, all_zero_window_ratio, news_embs.shape,
    )
    
def _extract_and_split(config: dict):
    """Run pipeline, extract per-symbol data, split by date, return combined splits."""
    from run_model_benchmark import (
        extract_per_symbol_data,
        split_by_date,
        impute_market_window_splits,
    )

    dataset = run_pipeline(config)
    seq_len = config["sequence_len"]
    horizon = config["target_horizon_days"]

    # Fetch raw OHLCV for close windows
    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(
        config["symbols"], config["start"], config["end"],
    )

    all_data = extract_per_symbol_data(dataset, raw_ohlcv, seq_len=seq_len, target_horizon_days=horizon)

    # Build cross-symbol pooled news for news_scope="all"
    cross_symbol_news = _build_cross_symbol_news(all_data, seq_len)

    # Combine symbols into single arrays
    combined = {}
    for sym, sym_data in all_data.items():
        splits = split_by_date(
            sym_data, sym_data["times"],
            train_end=config["train_end"],
            val_end=config["val_end"],
            target_horizon_days=horizon,
        )
        splits = impute_market_window_splits(splits)

        # Inject cross-symbol news arrays into splits
        # Use the same purge-aware masks that split_by_date uses
        pooled_embs, pooled_masks = cross_symbol_news[sym]
        _log_news_stats(f"{sym} pooled_all", pooled_embs, pooled_masks)
        _log_news_stats(f"{sym} original_matched", sym_data["news_embs"][:, :, :768], sym_data["news_masks"])
        times = sym_data["times"]
        times_pd = pd.to_datetime(times)
        sorted_times = np.sort(np.unique(times_pd.values))  # datetime64[ns] array
        train_end_ts = pd.Timestamp(config["train_end"])
        val_end_ts = pd.Timestamp(config["val_end"])

        def _trading_day_offset(boundary, n):
            idx = np.searchsorted(sorted_times, np.datetime64(boundary), side="right") - 1
            idx = max(idx - n, 0)
            return pd.Timestamp(sorted_times[idx])

        train_end_purged = _trading_day_offset(train_end_ts, horizon)
        val_end_purged = _trading_day_offset(val_end_ts, horizon)

        train_mask = times_pd <= train_end_purged
        val_mask = (times_pd > train_end_ts) & (times_pd <= val_end_purged)
        test_mask = times_pd > val_end_ts

        for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            n_split = splits[split_name]["targets"].shape[0]
            logger.warning(
                "{} {} | raw_mask_count={} split_count={}",
                sym, split_name, int(mask.sum()), n_split
            )
            mask_indices = _assert_split_alignment(sym, split_name, times_pd, mask, n_split)

            splits[split_name]["news_embs_all"] = pooled_embs[mask_indices]
            splits[split_name]["news_masks_all"] = pooled_masks[mask_indices]

        if not combined:
            combined = splits
        else:
            for split_name in ("train", "val", "test"):
                for key in splits[split_name]:
                    assert (
                        combined[split_name][key].shape[1:]
                        ==
                        splits[split_name][key].shape[1:]
                    ), (
                        f"Shape mismatch for {key}: "
                        f"{combined[split_name][key].shape} vs "
                        f"{splits[split_name][key].shape}"
                    )                    
                    combined[split_name][key] = np.concatenate(
                        [combined[split_name][key], splits[split_name][key]], axis=0
                    )

    market_cols = list(getattr(dataset, "market_cols", []))
    return combined, market_cols

def _assert_split_alignment(
    sym: str,
    split_name: str,
    source_times: pd.Series | np.ndarray,
    mask: np.ndarray,
    n_split: int,
) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) != n_split:
        raise RuntimeError(
            f"{sym} {split_name}: raw mask count {len(idx)} != split count {n_split}. "
            "This indicates split alignment mismatch."
        )
    return idx

def _run_table(
    table: str,
    splits: dict,
    market_cols: list[str],
    horizon: int,
    device: str,
    chronos,
    hpo_params: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all cells for one table and return results DataFrame."""
    configs = generate_grid(table=table)

    rows = []
    failures = []

    with tqdm(
        total=len(configs),
        desc=f"  Cells ({table})",
        unit="cell",
        leave=False,
        ncols=100,
        position=1,
    ) as pbar:
        for cfg in configs:
            try:
                metrics = run_ablation_cell(
                    cfg, splits, market_cols, horizon=horizon, device=device, chronos=chronos,
                    seed=seed, cache_dir=Path("cache"), hpo_params=hpo_params,
                )
                row = {
                    "model_name": cfg.model_name,
                    "fusion_type": cfg.fusion_type,
                    "news_scope": cfg.news_scope,
                    "sentiment_mode": cfg.sentiment_mode,
                    "use_positional_encoding": cfg.use_positional_encoding,
                    "use_news_gate": cfg.use_news_gate,
                    "recency_gate_k": cfg.recency_gate_k,
                    "use_two_stage": cfg.use_two_stage,
                    "use_aux_loss": cfg.use_aux_loss,
                    "use_variance_reg": cfg.use_variance_reg,
                    **metrics,
                }
                rows.append(row)
            except Exception as e:
                failures.append({
                    "cell_id": getattr(cfg, "cell_id", "unknown"),
                    "model_name": cfg.model_name,
                    "fusion_type": cfg.fusion_type,
                    "news_scope": cfg.news_scope,
                    "sentiment_mode": cfg.sentiment_mode,
                    "use_positional_encoding": cfg.use_positional_encoding,
                    "use_news_gate": cfg.use_news_gate,
                    "recency_gate_k": cfg.recency_gate_k,
                    "use_two_stage": cfg.use_two_stage,
                    "use_aux_loss": cfg.use_aux_loss,
                    "use_variance_reg": cfg.use_variance_reg,
                    "seed": seed,
                    "error": repr(e),
                })
                logger.error("FAILED {}: {}", getattr(cfg, "cell_id", "unknown"), e)
            finally:
                pbar.update(1)

    df = pd.DataFrame(rows)
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_dir = Path("results/ablation_failures")
        fail_dir.mkdir(parents=True, exist_ok=True)
        fail_path = fail_dir / f"{table}__{horizon}d__seed{seed}.csv"
        fail_df.to_csv(fail_path, index=False)
        logger.warning("Saved {} failed cells to {}", len(failures), fail_path)

    return df

def _make_summary_row(table: str, df: pd.DataFrame, horizon: int) -> dict:
    """Return summary row with forecast winner and trading winner."""

    if df.empty:
        return {}

    required_forecast = {"DA%", "IC", "RMSE"}
    required_trading = {"Sharpe", "F1"}

    if not required_forecast.issubset(df.columns):
        return {}

    forecast_score = (
        0.5 * df["DA%"]
        + 0.3 * df["IC"]
        - 0.2 * df["RMSE"]
    )

    best_forecast_idx = forecast_score.idxmax()
    best_forecast = df.loc[best_forecast_idx]

    if required_trading.issubset(df.columns):
        trading_score = (
            0.7 * df["Sharpe"]
            + 0.3 * df["F1"]
        )

        best_trading_idx = trading_score.idxmax()
        best_trading = df.loc[best_trading_idx]
    else:
        best_trading = best_forecast

    setting_col = {
        "fusion": "fusion_type",
        "news_scope": "news_scope",
        "sentiment": "sentiment_mode",
        "component": "use_positional_encoding",
    }.get(table, "fusion_type")

    return {
        "table": table,
        "horizon": horizon,

        "best_forecast_model":
            best_forecast.get("model_name", ""),

        "best_forecast_setting":
            best_forecast.get(setting_col, ""),

        "best_forecast_DA":
            round(float(best_forecast.get("DA%", 0)), 2),

        "best_forecast_IC":
            round(float(best_forecast.get("IC", 0)), 4),

        "best_forecast_RMSE":
            round(float(best_forecast.get("RMSE", 0)), 5),

        "best_trading_model":
            best_trading.get("model_name", ""),

        "best_trading_setting":
            best_trading.get(setting_col, ""),

        "best_trading_sharpe":
            round(float(best_trading.get("Sharpe", 0)), 3),

        "best_trading_f1":
            round(float(best_trading.get("F1", 0)), 3),
    }
def _average_seed_dfs(dfs: list[pd.DataFrame], seeds: list[int] | None = None) -> pd.DataFrame:
    """
    Average metrics across seeds by config key, not by row order.
    Adds std columns and a seed_count column for auditability.
    """

    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        out = dfs[0].copy()
        out["seed_count"] = 1
        return out

    metric_cols = [
        "MAE",
        "RMSE",
        "DA%",
        "Sharpe",
        "IC",
        "Prec",
        "Rec",
        "F1",
        "ESS",
        "base_rate_DA%",
        "DA_skill%",
    ]

    work = []
    for i, df in enumerate(dfs):
        if df.empty:
            continue

        missing_keys = [c for c in _CONFIG_KEY_COLS if c not in df.columns]
        if missing_keys:
            raise ValueError(f"Seed DataFrame missing key columns: {missing_keys}")

        df_i = df.copy()
        seed_label = seeds[i] if seeds is not None and i < len(seeds) else i
        df_i["__seed__"] = seed_label
        work.append(df_i)

    if not work:
        return pd.DataFrame()

    long_df = pd.concat(work, axis=0, ignore_index=True)

    # Audit duplicate rows within same seed/config
    dup_mask = long_df.duplicated(subset=_CONFIG_KEY_COLS + ["__seed__"], keep=False)
    if dup_mask.any():
        dup_rows = long_df.loc[dup_mask, _CONFIG_KEY_COLS + ["__seed__"]]
        raise ValueError(
            "Duplicate config rows found within the same seed.\n"
            f"{dup_rows.to_string(index=False)}"
        )

    group = long_df.groupby(_CONFIG_KEY_COLS, dropna=False, as_index=False)

    # Start with one row per config
    out = group.size().rename(columns={"size": "seed_count"})

    # Mean/std for metric columns that actually exist
    for col in metric_cols:
        if col not in long_df.columns:
            continue

        stats = group[col].agg(["mean", "std"]).reset_index()
        stats = stats.rename(columns={
            "mean": col,
            "std": f"{col}_std",
        })
        out = out.merge(stats, on=_CONFIG_KEY_COLS, how="left")

    # Degenerate: True if any contributing seed was degenerate
    if "degenerate" in long_df.columns:
        deg = group["degenerate"].agg(lambda x: bool(np.any(x.astype(bool)))).reset_index()
        out = out.merge(deg, on=_CONFIG_KEY_COLS, how="left")

    # Optional audit: warn on incomplete seed coverage
    expected = len(work)
    incomplete = out[out["seed_count"] < expected]
    if not incomplete.empty:
        logger.warning(
            "Some configs are missing in one or more seeds: {} incomplete rows",
            len(incomplete),
        )

    return out

def _average_seed_predictions(cache_dir: Path, seeds: list[int], horizons: list[int]) -> None:
    """Merge per-seed pr    ediction .npy files into canonical files for bootstrap CI / DM test."""
    from collections import defaultdict
    pred_dir = cache_dir / "predictions"
    if not pred_dir.exists():
        return
    
    for horizon in horizons:
        seed_files = list(pred_dir.glob(f"*__seed*__{horizon}d.npy"))
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in seed_files:
            cell_id = f.stem.rsplit("__seed", 1)[0]
            groups[cell_id].append(f)
        
        with tqdm(
            total=len(groups),
            desc=f"  Merging {horizon}D predictions",
            unit="cell",
            leave=False,
            ncols=100,
        ) as pbar:
            for cell_id, files in groups.items():
                arrays = [np.load(str(f)) for f in sorted(files)]
                avg = np.mean(arrays, axis=0).astype(np.float32)
                out_path = pred_dir / f"{cell_id}__{horizon}d.npy"
                np.save(str(out_path), avg)
                pbar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation benchmark")
    parser.add_argument(
        "--table",
        choices=["fusion", "news_scope", "sentiment", "component", "all"],
        default="all",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--stage", choices=["run", "plot"], default="run")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    _ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "plot":
        logger.info("Regenerating plots...")
        for h in args.horizons:
            _regenerate_plots(h)
        logger.info("✓ Plot regeneration complete")
        return

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.warning("═════════════════════════════════════════════")
    logger.warning("  🚀 ABLATION BENCHMARK")
    logger.warning("═════════════════════════════════════════════")
    logger.warning(f"Device: {device}")
    logger.warning(f"Horizons: {args.horizons}")
    logger.warning(f"Seeds: {args.seeds}")
    logger.warning(f"Table(s): {args.table}")
    logger.warning("═════════════════════════════════════════════")

    chronos = ChronosMarketPredictor(device=device)

    # Main horizon loop with progress
    for horizon in tqdm(
        args.horizons,
        desc="Horizons",
        unit="h",
        ncols=100,
        position=0,
        leave=True,
    ):
        logger.warning(f"\n📊 Processing horizon: {horizon}D")
        config = _build_pipeline_config(horizon)
        
        # Data extraction with status
        logger.warning("  ⏳ Extracting and splitting data...")
        splits, market_cols = _extract_and_split(config)
        logger.warning(f"  ✓ Data ready")

        tables_to_run = (
            [args.table] if args.table != "all"
            else ["fusion", "news_scope", "sentiment", "component"]
        )
        hdir = _horizon_dir(horizon)
        hdir.mkdir(parents=True, exist_ok=True)
        _figures_dir(horizon).mkdir(parents=True, exist_ok=True)

        # Load baseline HPO params
        logger.warning("  ⏳ Loading baseline HPO params...")
        from src.benchmark.baseline_hpo import get_default_baseline_hpo_params
        hpo_params = get_default_baseline_hpo_params()
        logger.warning("  ✓ HPO params loaded")

        summary_rows: list[dict] = []
        
        # Table loop with progress
        for table in tqdm(
            tables_to_run,
            desc=f"  Tables ({horizon}D)",
            unit="table",
            leave=False,
            ncols=100,
            position=1,
        ):
            seed_dfs: list[pd.DataFrame] = []
            
            # Seed loop with progress
            for seed in tqdm(
                args.seeds,
                desc=f"    Seed loop ({table})",
                unit="seed",
                leave=False,
                ncols=100,
                position=2,
            ):
                set_global_seed(seed)
                logger.warning("    ▶ Running seed {}", seed)

                df_seed = _run_table(
                    table, splits, market_cols, horizon, device, chronos,
                    hpo_params=hpo_params, seed=seed
                )
                df_seed["run_seed"] = seed
                seed_dfs.append(df_seed)
            
            df = _average_seed_dfs(seed_dfs, seeds=args.seeds)
            csv_path = hdir / f"{table}.csv"
            df.to_csv(csv_path, index=False)
            logger.warning(f"  ✓ {table:15} → {csv_path.name}")

            summary_rows.append(_make_summary_row(table, df, horizon))
            _plot_table(table, df, horizon)

        if summary_rows:
            summary_df = pd.DataFrame([r for r in summary_rows if r])
            summary_path = hdir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.warning(f"  ✓ Summary → {summary_path.name}")

    # Merge per-seed prediction files
    logger.warning("\n⏳ Merging per-seed predictions...")
    _average_seed_predictions(Path("cache"), args.seeds, args.horizons)
    logger.warning("✓ Predictions merged")

    logger.warning("\n" + "═" * 45)
    logger.warning("  ✅ ABLATION BENCHMARK COMPLETE")
    logger.warning(f"     ({len(args.seeds)}-seed average)")
    logger.warning("═" * 45)


def _plot_table(table: str, df: pd.DataFrame, horizon: int) -> None:
    """Generate heatmap + delta charts for a single table."""
    if not df.empty:
        plot_table_charts(df, table, horizon, _figures_dir(horizon))


def _regenerate_plots(horizon: int) -> None:
    """Regenerate plots from existing CSVs in the horizon-scoped directory."""
    hdir = _horizon_dir(horizon)
    for table in ("fusion", "news_scope", "sentiment", "component"):
        csv_path = hdir / f"{table}.csv"
        # Also accept the old flat name for backward compat
        if not csv_path.exists():
            csv_path = _ABLATION_ROOT / f"ablation_{table}_{horizon}d.csv"
        if not csv_path.exists():
            logger.warning("⚠️  No CSV for {} {}D — skipping", table, horizon)
            continue
        df = pd.read_csv(csv_path)
        _figures_dir(horizon).mkdir(parents=True, exist_ok=True)
        _plot_table(table, df, horizon)
        logger.warning(f"  ✓ Regenerated {table:15} for {horizon}D")


if __name__ == "__main__":
    main()
