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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
from loguru import logger

from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher
from src.benchmark.chronos_encoder import ChronosMarketPredictor
from src.benchmark.ablation_config import AblationConfig, generate_grid
from src.benchmark.ablation_runner import run_ablation_cell
from src.benchmark.ablation_plots import plot_table_charts

_ABLATION_ROOT = Path("results/ablation")


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

    For each date, averages non-zero news embeddings across all symbols.
    Returns per-symbol arrays of cross-symbol pooled news windows + masks.
    """
    from collections import defaultdict

    # Step 1: Build a date→embedding map by pooling across symbols
    # Each symbol has times array of shape (N,) aligned with news_embs (N, seq_len, 768)
    # We need per-bar (date-level) embeddings, not per-window.
    # Reconstruct per-bar embeddings from the LAST bar of each window
    # (windows overlap, so bar i appears as the last bar of window i)
    date_embs: dict[str, list[np.ndarray]] = defaultdict(list)

    for sym, data in all_data.items():
        times = data["times"]  # (N_samples,)
        news_embs = data["news_embs"]  # (N_samples, seq_len, 768)
        # The last bar in each window corresponds to times[i]
        for i, t in enumerate(times):
            emb = news_embs[i, -1, :768]  # last bar of window i → bar at time t
            if np.any(emb != 0):
                date_embs[str(t)].append(emb)

    # Step 2: Pool embeddings per date (mean of non-zero)
    pooled: dict[str, np.ndarray] = {}
    for date_key, embs in date_embs.items():
        pooled[date_key] = np.mean(embs, axis=0).astype(np.float32)

    # Step 3: Rebuild per-symbol windowed arrays using pooled embeddings
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, data in all_data.items():
        times = data["times"]
        news_embs_orig = data["news_embs"]  # (N, seq_len, 768)
        N, S, D = news_embs_orig.shape
        D = min(D, 768)

        # We need the time for each bar in each window.
        # Window i covers bars [i - seq_len + 1, ..., i] in the original series.
        # But we only have times for the last bar of each window.
        # For cross-symbol scope, use the per-bar approach:
        # - For any bar that has a pooled embedding, use it; else zero.
        # Simplification: use the original window structure but replace each
        # bar's embedding with the pooled version for that bar's date.

        # Build a date→index mapping for this symbol's windows
        # Since windows are sequential, bar j in window i has time = times[i - (S-1-j)]
        # ... but times array only has one timestamp per window (the last bar).
        # Simpler approach: for each window, the last bar's time is times[i].
        # Bar at position j has time = times[i - (S - 1 - j)] IF that index exists.

        # Practical approach: rebuild the per-bar time series first
        # The first window starts at index 0 and the last bar is at index seq_len-1
        # So total unique bars = N + seq_len - 1
        # Bar k corresponds to times[k - seq_len + 1] ... but not all are in times.

        # Simplest correct approach: use the existing news_embs structure but
        # for each sample, check if ANY symbol had news at that position and use pooled.
        pooled_windows = np.zeros((N, S, D), dtype=np.float32)
        pooled_masks = np.ones((N, S), dtype=bool)  # True = no news

        for i in range(N):
            # The last bar of window i is at times[i]
            # We can infer previous bars' times from neighboring windows
            for j in range(S):
                # Bar at position j in window i = bar at position (S-1) in window (i - (S-1-j))
                ref_idx = i - (S - 1 - j)
                if 0 <= ref_idx < N:
                    bar_time = str(times[ref_idx])
                    if bar_time in pooled:
                        pooled_windows[i, j, :] = pooled[bar_time]
                        pooled_masks[i, j] = False

        result[sym] = (pooled_windows, pooled_masks)

    return result


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
            mask_indices = np.where(mask)[0][:n_split]
            if len(mask_indices) == n_split:
                splits[split_name]["news_embs_all"] = pooled_embs[mask_indices]
                splits[split_name]["news_masks_all"] = pooled_masks[mask_indices]
            else:
                # Fallback: use matched embeddings if alignment fails
                splits[split_name]["news_embs_all"] = splits[split_name]["news_embs"]
                splits[split_name]["news_masks_all"] = splits[split_name]["news_masks"]

        if not combined:
            combined = splits
        else:
            for split_name in ("train", "val", "test"):
                for key in splits[split_name]:
                    combined[split_name][key] = np.concatenate(
                        [combined[split_name][key], splits[split_name][key]], axis=0
                    )

    market_cols = list(getattr(dataset, "market_cols", []))
    return combined, market_cols


def _run_table(
    table: str,
    splits: dict,
    market_cols: list[str],
    horizon: int,
    device: str,
    chronos,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all cells for one table and return results DataFrame."""
    configs = generate_grid(table=table)
    logger.info("═══ Table '{}' — {} cells for {}D (seed={}) ═══", table, len(configs), horizon, seed)

    rows = []
    for cfg in configs:
        try:
            metrics = run_ablation_cell(
                cfg, splits, market_cols, horizon=horizon, device=device, chronos=chronos,
                seed=seed, cache_dir=Path("cache"),
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
            logger.error("FAILED {}: {}", cfg.cell_id, e)

    return pd.DataFrame(rows)


def _make_summary_row(table: str, df: pd.DataFrame, horizon: int) -> dict:
    """Return a one-row dict with the winning config for this table/horizon."""
    if df.empty or "DA%" not in df.columns:
        return {}
    best_idx = df["DA%"].idxmax()
    best = df.loc[best_idx]
    setting_col = {
        "fusion":     "fusion_type",
        "news_scope": "news_scope",
        "sentiment":  "sentiment_mode",
        "component":  "use_positional_encoding",  # informative enough
    }.get(table, "fusion_type")
    return {
        "table":       table,
        "horizon":     horizon,
        "best_model":  best.get("model_name", ""),
        "best_setting": best.get(setting_col, ""),
        "DA%":   round(float(best.get("DA%",   0)), 2),
        "Sharpe": round(float(best.get("Sharpe", 0)), 3),
        "IC":     round(float(best.get("IC",     0)), 4),
        "RMSE":   round(float(best.get("RMSE",   0)), 5),
        "F1":     round(float(best.get("F1",     0)), 3),
    }


def _average_seed_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Average metric columns across multiple seed runs; OR-combine degenerate flags."""
    if len(dfs) == 1:
        return dfs[0]
    ref = dfs[0].copy()
    metric_cols = ["DA%", "Sharpe", "IC", "RMSE", "MAE", "F1", "DA_skill%", "base_rate_DA%", "ESS"]
    for col in metric_cols:
        if col in ref.columns:
            arrays = [df[col].values for df in dfs if col in df.columns]
            ref[col] = np.mean(arrays, axis=0)
    if "degenerate" in ref.columns:
        ref["degenerate"] = np.any([df["degenerate"].values for df in dfs], axis=0)
    return ref


def _average_seed_predictions(cache_dir: Path, seeds: list[int], horizons: list[int]) -> None:
    """Merge per-seed prediction .npy files into canonical files for bootstrap CI / DM test."""
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
        for cell_id, files in groups.items():
            arrays = [np.load(str(f)) for f in sorted(files)]
            avg = np.mean(arrays, axis=0).astype(np.float32)
            out_path = pred_dir / f"{cell_id}__{horizon}d.npy"
            np.save(str(out_path), avg)
            logger.info("Averaged {} seed predictions → {}", len(files), out_path.name)


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
    args = parser.parse_args()

    _ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "plot":
        for h in args.horizons:
            _regenerate_plots(h)
        return

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)
    logger.info("Seeds: {}", args.seeds)

    chronos = ChronosMarketPredictor(device=device)

    for horizon in args.horizons:
        logger.info("═══ Horizon: {}D ═══", horizon)
        config = _build_pipeline_config(horizon)
        splits, market_cols = _extract_and_split(config)

        tables_to_run = (
            [args.table] if args.table != "all"
            else ["fusion", "news_scope", "sentiment", "component"]
        )
        hdir = _horizon_dir(horizon)
        hdir.mkdir(parents=True, exist_ok=True)
        _figures_dir(horizon).mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict] = []
        for table in tables_to_run:
            seed_dfs: list[pd.DataFrame] = []
            for seed in args.seeds:
                logger.info("─── Seed {} | Table '{}' | {}D ───", seed, table, horizon)
                df_seed = _run_table(table, splits, market_cols, horizon, device, chronos, seed=seed)
                seed_dfs.append(df_seed)
            df = _average_seed_dfs(seed_dfs)
            csv_path = hdir / f"{table}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved (avg {}-seed) → {}", len(args.seeds), csv_path)

            summary_rows.append(_make_summary_row(table, df, horizon))
            _plot_table(table, df, horizon)

        if summary_rows:
            summary_df = pd.DataFrame([r for r in summary_rows if r])
            summary_path = hdir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info("Summary → {}", summary_path)

    # Merge per-seed prediction files into canonical files for bootstrap CI / DM test
    logger.info("Merging per-seed predictions…")
    _average_seed_predictions(Path("cache"), args.seeds, args.horizons)

    logger.info("═══ Ablation benchmark complete ({}-seed avg) ═══", len(args.seeds))


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
            logger.warning("No CSV for {} {}D — skipping", table, horizon)
            continue
        df = pd.read_csv(csv_path)
        _figures_dir(horizon).mkdir(parents=True, exist_ok=True)
        _plot_table(table, df, horizon)
        logger.info("Regenerated charts for {} {}D", table, horizon)


if __name__ == "__main__":
    main()
