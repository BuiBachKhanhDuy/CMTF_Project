"""Append-only integration hooks for FinBERT Zero-Shot in run_model_benchmark.py.

This module runs a post-pass after the original benchmark ``main()`` so that
existing benchmark code stays untouched while FinBERT rows are merged into the
same CSV / plot outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.benchmark.finbert_encoder import FinBERTMarketPredictor
from src.pipeline import run_pipeline
from src.pipeline.data_fetcher import VnstockDataFetcher

FINBERT_EXPERIMENT_NAME = "FinBERT Zero-Shot"
FINBERT_CONTEXT_LENGTH = 30


def extend_protocols(protocols: dict[str, Any], protocol_meta_cls: type) -> None:
    """Register FinBERT protocol metadata (mirrors Chronos zero-shot regime)."""
    protocols[FINBERT_EXPERIMENT_NAME] = protocol_meta_cls(
        comparison_set="cross_model",
        input_regime="close_only",
        adaptation_regime="zero_shot",
        context_length=FINBERT_CONTEXT_LENGTH,
    )


def _parse_extension_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Mirror run_model_benchmark CLI flags needed for the FinBERT post-pass."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=["data", "predict", "hpo", "plot"], default=None)
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--symbols", nargs="+", type=str, default=None)
    parser.add_argument("--experiments", nargs="+", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument(
        "--skip-finbert",
        action="store_true",
        help="Skip the FinBERT zero-shot baseline.",
    )
    return parser.parse_args(argv)


def _should_run_finbert(args: argparse.Namespace) -> bool:
    if args.skip_finbert:
        return False
    if args.stage in ("data", "plot"):
        return False
    if args.experiments is not None:
        return FINBERT_EXPERIMENT_NAME in set(args.experiments)
    return True


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    from src.benchmark.baseline_models import GLOBAL_LOSS_CONFIG

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
        "sequence_len": FINBERT_CONTEXT_LENGTH,
        "horizon": 1,
        "target_horizons_days": [1, 5, 20],
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        "use_tabular_market_features": True,
        "sign_penalty_weight": GLOBAL_LOSS_CONFIG.sign_penalty_weight,
    }
    if args.horizons:
        config["target_horizons_days"] = [int(h) for h in args.horizons]
    if args.symbols:
        config["symbols"] = [str(s) for s in args.symbols]
    return config


def _use_prediction_cache(stage: str | None, no_cache: bool) -> bool:
    if no_cache:
        return False
    return stage is None or stage == "plot"


def _load_anchor_pred(cache_pred_dir: Path, sym: str, target_h: int) -> np.ndarray | None:
    matches = sorted(cache_pred_dir.glob(f"lstm_{sym}_{target_h}d_*.npy"))
    if not matches:
        return None
    return np.load(matches[0])


def _recompute_avg_row(
    df: pd.DataFrame,
    experiment_name: str,
    target_h: int,
    package_result_fn,
) -> pd.DataFrame:
    sym_rows = df[
        (df["Experiment"] == experiment_name) & (df["Symbol"] != "AVG")
    ]
    if sym_rows.empty:
        return df

    df = df[~((df["Experiment"] == experiment_name) & (df["Symbol"] == "AVG"))]

    cache_pred_dir = Path("cache/predictions")
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    a_parts: list[np.ndarray] = []

    for _, row in sym_rows.iterrows():
        sym = str(row["Symbol"])
        truth_path = cache_pred_dir / f"truth_{sym}_{target_h}d.npy"
        if not truth_path.exists():
            continue
        y_true = np.load(truth_path)
        zs_paths = sorted(cache_pred_dir.glob(f"finbert_zs_{sym}_{target_h}d_*.npy"))
        if not zs_paths:
            continue
        y_pred = np.load(zs_paths[0])
        if y_true.shape != y_pred.shape:
            continue
        y_parts.append(y_true)
        p_parts.append(y_pred)
        anchor = _load_anchor_pred(cache_pred_dir, sym, target_h)
        if anchor is not None and anchor.shape == y_pred.shape:
            a_parts.append(anchor)

    if not y_parts:
        return df

    y_concat = np.concatenate(y_parts)
    p_concat = np.concatenate(p_parts)
    a_concat = np.concatenate(a_parts) if a_parts else None
    avg_metrics = package_result_fn(
        experiment_name,
        "AVG",
        target_h,
        y_concat,
        p_concat,
        a_concat,
    )
    return pd.concat([df, pd.DataFrame([avg_metrics])], ignore_index=True)


def run_finbert_benchmark_extension(module_globals: dict[str, Any]) -> None:
    """Post-pass: compute FinBERT zero-shot preds and merge into benchmark CSVs."""
    import sys

    args = _parse_extension_args(sys.argv[1:])
    if not _should_run_finbert(args):
        logger.info("FinBERT Zero-Shot post-pass skipped.")
        return

    config = _build_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extract_per_symbol_data = module_globals["extract_per_symbol_data"]
    split_by_date = module_globals["split_by_date"]
    package_result = module_globals["package_result"]
    plot_ablation = module_globals["plot_ablation"]
    plot_per_symbol = module_globals["plot_per_symbol"]
    _split_hash = module_globals["_split_hash"]
    _save_npy = module_globals["_save_npy"]
    _load_npy = module_globals["_load_npy"]

    results_dir = module_globals["RESULTS_DIR"]
    figures_dir = module_globals["FIGURES_DIR"]
    cache_pred_dir = module_globals["CACHE_PRED_DIR"]

    from src.common import generate_walkforward_folds

    logger.info("═══ FinBERT Zero-Shot post-pass (device={}) ═══", device)
    finbert = FinBERTMarketPredictor(device=device)

    fetcher = VnstockDataFetcher()
    raw_ohlcv = fetcher.fetch_multi_symbol(config["symbols"], config["start"], config["end"])

    horizons = [int(h) for h in config.get("target_horizons_days", [1])]

    if args.folds > 1:
        all_dates = pd.date_range(config["start"], config["end"], freq="B")
        fold_pairs = generate_walkforward_folds(
            all_dates,
            n_folds=args.folds,
            test_months=6,
            min_train_months=36,
        )
    else:
        fold_pairs = [(config["train_end"], config["val_end"])]

    for fold_idx, (fold_train_end, fold_val_end) in enumerate(fold_pairs):
        fold_config = {**config, "train_end": fold_train_end, "val_end": fold_val_end}
        pipeline_cfg = {**fold_config, "target_horizon_days": horizons[0]}
        shared_dataset = run_pipeline(pipeline_cfg)

        per_symbol_by_h: dict[int, dict] = {}
        for target_h in horizons:
            per_symbol_by_h[target_h] = extract_per_symbol_data(
                shared_dataset,
                raw_ohlcv,
                seq_len=FINBERT_CONTEXT_LENGTH,
                target_horizon_days=target_h,
                close_only=True,
            )

        for target_h in horizons:
            csv_path = results_dir / f"model_benchmark_{target_h}d.csv"
            if not csv_path.exists():
                logger.warning(
                    "FinBERT post-pass: missing {} — run core benchmark first.",
                    csv_path,
                )
                continue

            results_df = pd.read_csv(csv_path)
            results_df = results_df[results_df["Experiment"] != FINBERT_EXPERIMENT_NAME]

            run_cfg = {**fold_config, "target_horizon_days": target_h}
            per_symbol = per_symbol_by_h[target_h]
            new_rows: list[dict[str, Any]] = []

            for sym, data in per_symbol.items():
                splits = split_by_date(
                    {k: v for k, v in data.items() if k != "times"},
                    data["times"],
                    run_cfg["train_end"],
                    run_cfg["val_end"],
                    target_horizon_days=target_h,
                )
                if len(splits["test"]["targets"]) == 0:
                    continue

                sh = _split_hash(splits, sym, target_h)
                cache_path = cache_pred_dir / f"finbert_zs_{sym}_{target_h}d_{sh}.npy"
                cached = (
                    _load_npy(cache_path)
                    if _use_prediction_cache(args.stage, args.no_cache)
                    else None
                )
                if cached is not None:
                    preds = cached
                else:
                    preds = finbert.zero_shot_predict(
                        splits["test"]["close_windows"],
                        splits["test"]["last_close"],
                        seed=config["seed"],
                        horizon=target_h,
                    )
                    _save_npy(cache_path, preds)

                anchor_pred = _load_anchor_pred(cache_pred_dir, sym, target_h)
                new_rows.append(
                    package_result(
                        FINBERT_EXPERIMENT_NAME,
                        sym,
                        target_h,
                        splits["test"]["targets"],
                        preds,
                        anchor_pred,
                    )
                )

            if not new_rows:
                logger.warning("FinBERT post-pass: no symbols for {}D", target_h)
                continue

            results_df = pd.concat(
                [results_df, pd.DataFrame(new_rows)],
                ignore_index=True,
            )
            results_df = _recompute_avg_row(
                results_df,
                FINBERT_EXPERIMENT_NAME,
                target_h,
                package_result,
            )

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
            results_df = results_df[col_order]
            results_df.to_csv(csv_path, index=False)
            logger.info("FinBERT results merged → {}", csv_path)

            avg_df = results_df[results_df["Symbol"] == "AVG"]
            plot_ablation(avg_df, figures_dir / f"model_{target_h}d.png")
            plot_per_symbol(results_df, figures_dir / f"per_symbol_heatmap_{target_h}d.png")

        if fold_idx < len(fold_pairs) - 1:
            logger.info("FinBERT post-pass: additional folds not yet merged into CSV.")

    logger.info("═══ FinBERT Zero-Shot post-pass complete ═══")
