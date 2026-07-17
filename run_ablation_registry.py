"""run_ablation_registry.py

CLI entry point for the config-driven CMTF component-ablation REGISTRY
(cells "0".."18p", src/benchmark/ablation_registry.py). Every cell is an
``AblationConfig`` derived from ``CMTF_CORE`` via one targeted override — there
are no hardcoded per-cell scripts, everything executes from the registry.

Every cell runs with GATED metrics (validation-calibrated fixed-coverage
confidence gate) as the primary reported metrics, across 3 seeds by default.

Usage
-----
    # Smoke test: 2 cells, 1 seed, 5D only
    .venv\\Scripts\\python.exe run_ablation_registry.py --cells 0 0p --seeds 42 --horizons 5

    # Full registry, all groups, default seeds (1, 42, 123), all horizons
    .venv\\Scripts\\python.exe run_ablation_registry.py --cells all

    # One group only
    .venv\\Scripts\\python.exe run_ablation_registry.py --cells 0 0p 18 18p --horizons 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Reuse the existing data-loading / pipeline scaffolding rather than
# duplicating it — same splits, same market_cols, same HPO defaults as
# run_ablation_benchmark.py so registry cells are directly comparable.
from run_ablation_benchmark import (
    _build_pipeline_config,
    _configure_logging,
    _extract_and_split,
    configure_determinism,
    reseed_everything,
)

from src.benchmark.ablation_registry import (
    ABLATION_CELLS,
    CELL_GROUP,
    GATE_SWEEP_CELLS,
    all_cell_ids,
    get_cell,
)
from src.benchmark.ablation_report import (
    PRIMARY_METRICS,
    generate_markdown_report,
    rank_cells,
    real_minus_placebo_table,
)
from src.benchmark.ablation_runner import run_ablation_cell
from src.benchmark.baseline_hpo import get_default_baseline_hpo_params
from src.benchmark.coverage_diagnostics import (
    coverage_deciles,
    monotonicity_report,
    plot_coverage_curves,
)

_REGISTRY_ROOT = Path("results/ablation_registry")

_METRIC_COLS = [
    "MAE", "RMSE", "DA%", "Sharpe", "IC", "Prec", "Rec", "F1", "ESS",
    "base_rate_DA%", "DA_skill%", "train_time_sec",
    "DA%_gated", "Sharpe_gated", "IC_gated", "gate_coverage", "gate_tau",
]


def _horizon_dir(horizon: int) -> Path:
    d = _REGISTRY_ROOT / f"{horizon}d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_cell_all_seeds(
    cell_id: str,
    splits: dict,
    market_cols: list[str],
    horizon: int,
    device: str,
    seeds: list[int],
    hpo_params: dict,
    use_cache: bool,
    gate_coverage: float,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    cfg = get_cell(cell_id)
    rows = []
    seed_artifacts: dict[int, dict] = {}
    for seed in seeds:
        reseed_everything(seed)
        try:
            artifacts: dict = {}
            metrics = run_ablation_cell(
                cfg,
                splits,
                market_cols,
                horizon=horizon,
                device=device,
                chronos_pipeline=None,
                seed=seed,
                cache_dir=Path("cache"),
                use_cache=use_cache,
                hpo_params=hpo_params,
                artifacts=artifacts,
                compute_gate=True,
                gate_conviction=True,
                gate_coverage=gate_coverage,
            )
            seed_artifacts[seed] = artifacts
            rows.append({"cell": cell_id, "group": CELL_GROUP.get(cell_id, ""), "seed": seed, **metrics})
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("FAILED cell={} seed={}: {}", cell_id, seed, e)
            rows.append({
                "cell": cell_id, "group": CELL_GROUP.get(cell_id, ""), "seed": seed, "error": repr(e),
            })
    return pd.DataFrame(rows), seed_artifacts


def _aggregate_per_seed(per_seed_df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Mean/std/95%% CI across seeds, grouped by registry cell id."""
    if per_seed_df.empty:
        return per_seed_df

    df = per_seed_df[per_seed_df.get("error").isna()] if "error" in per_seed_df.columns else per_seed_df
    if df.empty:
        return df

    agg_cols = [c for c in _METRIC_COLS if c in df.columns]
    grouped = df.groupby(["cell", "group"], sort=False)[agg_cols].agg(["mean", "std", "count"])

    flat_cols = []
    for base, stat in grouped.columns:
        flat_cols.append(base if stat == "mean" else f"{base}_{stat}")
    grouped.columns = flat_cols
    out = grouped.reset_index()

    # 95% CI via normal approximation on the seed mean (n is small; this is a
    # rough diagnostic CI across only 3 seeds, not a formal inference claim).
    n_seeds = len(seeds)
    for c in agg_cols:
        std_col, cnt_col, ci_col = f"{c}_std", f"{c}_count", f"{c}_ci95"
        if std_col in out.columns and cnt_col in out.columns:
            se = out[std_col] / np.sqrt(out[cnt_col].clip(lower=1))
            out[ci_col] = 1.96 * se
            out.drop(columns=[cnt_col], inplace=True)

    out.insert(2, "seed_count", df.groupby(["cell", "group"], sort=False).size().to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven CMTF ablation registry runner")
    parser.add_argument("--cells", nargs="+", default=["all"], help="Registry cell ids (e.g. 0 0p 18) or 'all'")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 123])
    parser.add_argument("--gate-coverage", type=float, default=0.25)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pipeline-sentiment", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for real-minus-placebo CI")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    cell_ids = all_cell_ids() if args.cells == ["all"] else args.cells
    unknown = [c for c in cell_ids if c not in ABLATION_CELLS]
    if unknown:
        raise SystemExit(f"Unknown cell id(s): {unknown}. Known: {all_cell_ids()}")

    reseed_everything(42)
    configure_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.warning("═════════════════════════════════════════════")
    logger.warning("  🚀 ABLATION REGISTRY RUN")
    logger.warning("═════════════════════════════════════════════")
    logger.warning("Cells: {}", cell_ids)
    logger.warning("Horizons: {}", args.horizons)
    logger.warning("Seeds: {}", args.seeds)
    logger.warning("Device: {}", device)
    logger.warning("Gate coverage (fixed): {:.0%}", args.gate_coverage)
    logger.warning("═════════════════════════════════════════════")

    hpo_params = get_default_baseline_hpo_params()
    hpo_params.setdefault("mlp_summary", {"hidden_dim": 64, "dropout": 0.2, "lr": 1e-3, "batch_size": 32})

    for horizon in tqdm(args.horizons, desc="Horizons", unit="h", ncols=100):
        logger.warning("\n📊 Processing horizon: {}D", horizon)
        base_config = _build_pipeline_config(horizon, pipeline_sentiment=args.pipeline_sentiment)
        splits, market_cols = _extract_and_split(base_config)
        hdir = _horizon_dir(horizon)

        per_seed_rows = []
        artifacts_by_cell: dict[str, dict[int, dict]] = {}
        for cell_id in tqdm(cell_ids, desc=f"  Cells ({horizon}D)", unit="cell", ncols=100, leave=False):
            df_cell, seed_artifacts = _run_cell_all_seeds(
                cell_id, splits, market_cols, horizon, device, args.seeds,
                hpo_params, use_cache=not args.no_cache, gate_coverage=args.gate_coverage,
            )
            per_seed_rows.append(df_cell)
            artifacts_by_cell[cell_id] = seed_artifacts

        per_seed_df = pd.concat(per_seed_rows, axis=0, ignore_index=True) if per_seed_rows else pd.DataFrame()
        per_seed_path = hdir / "per_seed.csv"
        per_seed_df.to_csv(per_seed_path, index=False)
        logger.warning("  ✓ per-seed metrics → {}", per_seed_path)

        agg_df = _aggregate_per_seed(per_seed_df, args.seeds)
        agg_csv = hdir / "aggregated.csv"
        agg_parquet = hdir / "aggregated.parquet"
        agg_df.to_csv(agg_csv, index=False)
        try:
            agg_df.to_parquet(agg_parquet, index=False)
        except Exception as e:
            logger.warning("Could not write parquet ({}); csv still saved.", e)
        logger.warning("  ✓ aggregated metrics → {} / {}", agg_csv.name, agg_parquet.name)

        ranked_df = rank_cells(agg_df, metrics=PRIMARY_METRICS)
        ranked_df.to_csv(hdir / "ranked.csv", index=False)

        placebo_df = real_minus_placebo_table(
            horizon, seeds=tuple(args.seeds), cache_dir=Path("cache"), n_bootstrap=args.bootstrap,
        )
        placebo_df.to_csv(hdir / "real_minus_placebo.csv", index=False)

        # Coverage-accuracy diagnostics for the gate-sweep group. Reuses the
        # val_pred/test_pred artifacts already captured during the main
        # per-seed training loop above (compute_gate=True forces a fresh pass
        # there anyway) instead of retraining a second time.
        cov_dir = hdir / "coverage"
        cov_dir.mkdir(parents=True, exist_ok=True)
        mono_rows = []
        gate_cells_requested = [c for c in cell_ids if c in GATE_SWEEP_CELLS]
        for cell_id in gate_cells_requested:
            seed0 = args.seeds[0]
            artifacts = artifacts_by_cell.get(cell_id, {}).get(seed0, {})
            val_pred = artifacts.get("val_pred")
            test_pred = artifacts.get("test_pred")
            y_val = splits["val"]["targets"]
            y_test = splits["test"]["targets"]
            if val_pred is None or test_pred is None:
                logger.warning("No val/test predictions captured for cell {} — skipping coverage diagnostics", cell_id)
                continue
            deciles_df = coverage_deciles(val_pred, y_val, test_pred, y_test, horizon=horizon)
            deciles_df.to_csv(cov_dir / f"cell{cell_id}_coverage_deciles.csv", index=False)
            plot_coverage_curves(deciles_df, cov_dir / f"cell{cell_id}_coverage_curves.png", title=f"cell {cell_id} ({horizon}D)")
            mono = monotonicity_report(deciles_df)
            mono["cell"] = cell_id
            mono_rows.append(mono)

        monotonicity_df = pd.DataFrame(mono_rows) if mono_rows else pd.DataFrame(
            columns=["cell", "spearman_DA", "spearman_Sharpe", "spearman_IC"]
        )
        monotonicity_df.to_csv(hdir / "monotonicity.csv", index=False)

        generate_markdown_report(
            horizon=horizon,
            ranked_df=ranked_df,
            placebo_df=placebo_df,
            monotonicity_df=monotonicity_df,
            out_path=hdir / "report.md",
        )
        logger.warning("  ✓ report → {}", hdir / "report.md")

    logger.warning("\n" + "═" * 45)
    logger.warning("  ✅ ABLATION REGISTRY RUN COMPLETE")
    logger.warning("═" * 45)


if __name__ == "__main__":
    main()
