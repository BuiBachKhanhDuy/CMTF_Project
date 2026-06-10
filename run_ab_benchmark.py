"""A/B Benchmark: Multiagent (with risk management) vs CMTF-only baseline.

Compares:
  A) CMTF-only: raw model predictions → sign(pred) strategy
  B) Multiagent: full pipeline (orchestrator → market+news → predict → risk → answer)
     with risk gate that can reject trades → flat when risk fails

Metrics:
  - Directional Accuracy (DA%)
  - Annualized Sharpe Ratio
  - Information Coefficient (Spearman rank-corr)
  - Precision / Recall / F1 for "up" direction
  - Win Rate (trades taken that are correct)
  - Trade Frequency (% of days with non-flat action)
  - Risk-Adjusted Return

Usage:
    python run_ab_benchmark.py                          # full run, all horizons
    python run_ab_benchmark.py --horizons 1             # 1-day only
    python run_ab_benchmark.py --symbols VCB            # VCB only
    python run_ab_benchmark.py --start 2025-01-02 --end 2025-03-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.multiagent.graph import run_graph
from src.multiagent.config import MultiAgentConfig
from src.pipeline.orchestrator import clear_prepare_cache
from src.benchmark.metrics import (
    mae,
    rmse,
    directional_accuracy,
    sharpe_ratio,
    information_coefficient,
    direction_precision,
    direction_recall,
    direction_f1,
)


def _compute_actual_returns(
    symbol: str,
    cutoffs: list[str],
    horizon: int,
) -> dict[str, float]:
    """Fetch actual forward returns for each cutoff date.

    Returns dict: cutoff_date → actual H-day log return.
    """
    from src.pipeline.data_fetcher import VnstockDataFetcher

    fetcher = VnstockDataFetcher()

    # Fetch a wide date range covering all cutoffs + horizon buffer
    start = (pd.Timestamp(min(cutoffs)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(cutoffs)) + pd.Timedelta(days=horizon * 3)).strftime("%Y-%m-%d")

    df = fetcher.fetch_ohlcv(symbol, start, end, "1D", "KBS")
    df = df.sort_index()

    actual_returns = {}
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff)
        # Find the bar at cutoff (or nearest before)
        mask = df.index <= cutoff_ts
        if mask.sum() == 0:
            continue
        cutoff_idx = df.index[mask][-1]
        cutoff_pos = df.index.get_loc(cutoff_idx)

        # Forward return: close[t+H] / close[t] - 1 (log)
        future_pos = cutoff_pos + horizon
        if future_pos >= len(df):
            continue  # not enough future data

        close_now = df.iloc[cutoff_pos]["close"]
        close_future = df.iloc[future_pos]["close"]
        if close_now > 0:
            actual_returns[cutoff] = float(np.log(close_future / close_now))

    return actual_returns


def run_multiagent_predictions(
    symbol: str,
    cutoffs: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Run multiagent system (eval mode) for all cutoffs. Returns DataFrame.

    Uses rate-limit-aware pacing to avoid vnstock API throttling.
    """
    config = MultiAgentConfig(evaluation_mode=True)
    results = []

    for i, cutoff in enumerate(cutoffs):
        # Rate limit: vnstock guest API allows 20 req/min
        # Each new cutoff needs ~2 API calls (symbol + VNINDEX OHLCV).
        # Pace at 6 cutoffs between pauses to stay well under the limit.
        if i > 0 and i % 6 == 0:
            logger.info("Rate limit pause (15s) after {} cutoffs...", i)
            time.sleep(15)
        try:
            state = run_graph(
                query_text=f"Predict {symbol} {horizon}d",
                cutoff=cutoff,
                horizon=horizon,
                symbol=symbol,
                config=config,
            )
            results.append({
                "cutoff": cutoff,
                "symbol": symbol,
                "horizon": horizon,
                "final_pred": state["final_pred"],
                "baseline_pred": state["baseline_pred"],
                "action": state["action"],
                "position_scale": state["position_scale"],
                "predict_confidence": state["predict_confidence"],
                "risk_passed": state["risk_checks"]["all_passed"],
            })
            logger.info(
                "[{}/{}] {} {} {}d → pred={:+.5f} action={} conf={:.3f}",
                i + 1, len(cutoffs), symbol, cutoff, horizon,
                state["final_pred"], state["action"], state["predict_confidence"],
            )
        except Exception as e:
            logger.error("[{}/{}] {} {} {}d — ERROR: {}", i + 1, len(cutoffs), symbol, cutoff, horizon, e)
            results.append({
                "cutoff": cutoff,
                "symbol": symbol,
                "horizon": horizon,
                "final_pred": np.nan,
                "baseline_pred": np.nan,
                "action": "error",
                "position_scale": 0.0,
                "predict_confidence": 0.0,
                "risk_passed": False,
            })

    return pd.DataFrame(results)


def compute_strategy_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    actions: np.ndarray,  # "long", "short", "flat"
    horizon: int,
) -> dict[str, float]:
    """Compute trading strategy metrics for the multiagent system."""
    # Strategy A: CMTF-only (always trade based on sign of pred)
    cmtf_returns = np.sign(pred) * actual

    # Strategy B: Multiagent (only trade when action != "flat")
    ma_returns = np.zeros_like(actual)
    for i, action in enumerate(actions):
        if action == "long":
            ma_returns[i] = actual[i]
        elif action == "short":
            ma_returns[i] = -actual[i]
        # "flat" → 0 return (no trade)

    # Trade frequency
    trades_taken = np.sum(actions != "flat")
    trade_freq = trades_taken / len(actions) if len(actions) > 0 else 0.0

    # Win rate (among trades taken)
    if trades_taken > 0:
        trade_mask = actions != "flat"
        wins = np.sum(ma_returns[trade_mask] > 0)
        win_rate = wins / trades_taken
    else:
        win_rate = 0.0

    # Annualized returns
    ann_factor = 252 / max(horizon, 1)
    cmtf_ann_ret = float(cmtf_returns.mean() * ann_factor)
    ma_ann_ret = float(ma_returns.mean() * ann_factor)

    # Max drawdown for multiagent
    cumret = np.cumsum(ma_returns)
    running_max = np.maximum.accumulate(cumret)
    drawdowns = cumret - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "cmtf_ann_return": round(cmtf_ann_ret * 100, 2),
        "ma_ann_return": round(ma_ann_ret * 100, 2),
        "trade_frequency": round(trade_freq * 100, 1),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 2),
    }


def run_ab_test(
    symbols: list[str],
    horizons: list[int],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Run full A/B comparison across symbols and horizons."""
    all_results = []

    for symbol in symbols:
        for horizon in horizons:
            logger.info("=" * 60)
            logger.info("A/B Test: {} {}d | {} → {}", symbol, horizon, start, end)
            logger.info("=" * 60)

            # Clear pipeline cache between symbol/horizon combos
            clear_prepare_cache()

            # Generate cutoff dates (business days in range)
            cutoffs = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
            logger.info("Cutoff dates: {} ({} → {})", len(cutoffs), cutoffs[0], cutoffs[-1])

            # 1. Get actual forward returns
            t0 = time.time()
            actual_returns = _compute_actual_returns(symbol, cutoffs, horizon)
            logger.info("Actual returns fetched: {}/{} valid | {:.1f}s",
                        len(actual_returns), len(cutoffs), time.time() - t0)

            # Filter to cutoffs with valid actual returns
            valid_cutoffs = [c for c in cutoffs if c in actual_returns]
            if len(valid_cutoffs) < 10:
                logger.warning("Too few valid cutoffs for {} {}d: {}", symbol, horizon, len(valid_cutoffs))
                continue

            # 2. Run multiagent predictions
            t0 = time.time()
            pred_df = run_multiagent_predictions(symbol, valid_cutoffs, horizon)
            logger.info("Multiagent predictions complete | {:.1f}s", time.time() - t0)

            # 3. Align predictions with actuals
            pred_df["actual_return"] = pred_df["cutoff"].map(actual_returns)
            pred_df = pred_df.dropna(subset=["actual_return", "final_pred"])

            if len(pred_df) < 10:
                logger.warning("Too few aligned predictions for {} {}d: {}", symbol, horizon, len(pred_df))
                continue

            actual = pred_df["actual_return"].values
            preds = pred_df["final_pred"].values
            baseline = pred_df["baseline_pred"].values
            actions = pred_df["action"].values

            # 4. Compute metrics
            # A: CMTF-only (raw predictions → sign strategy)
            cmtf_metrics = {
                "MAE": mae(actual, preds),
                "RMSE": rmse(actual, preds),
                "DA%": directional_accuracy(actual, preds),
                "Sharpe": sharpe_ratio(actual, preds, horizon=horizon),
                "IC": information_coefficient(actual, preds),
                "Prec": direction_precision(actual, preds),
                "Rec": direction_recall(actual, preds),
                "F1": direction_f1(actual, preds),
            }

            # B: Multiagent (risk-gated strategy)
            strategy_metrics = compute_strategy_metrics(actual, preds, actions, horizon)

            # Multiagent effective Sharpe (only counts traded days)
            ma_returns = np.zeros_like(actual)
            for i, action in enumerate(actions):
                if action == "long":
                    ma_returns[i] = actual[i]
                elif action == "short":
                    ma_returns[i] = -actual[i]

            if ma_returns.std() > 0:
                ma_sharpe = float((ma_returns.mean() / ma_returns.std()) * np.sqrt(252 / max(horizon, 1)))
            else:
                ma_sharpe = 0.0

            # Store results
            row = {
                "Symbol": symbol,
                "Horizon": f"{horizon}d",
                "N_Samples": len(pred_df),
                # CMTF-only metrics
                "CMTF_DA%": round(cmtf_metrics["DA%"], 1),
                "CMTF_Sharpe": round(cmtf_metrics["Sharpe"], 3),
                "CMTF_IC": round(cmtf_metrics["IC"], 4),
                "CMTF_F1": round(cmtf_metrics["F1"], 3),
                "CMTF_AnnRet%": strategy_metrics["cmtf_ann_return"],
                # Multiagent metrics
                "MA_DA%": round(cmtf_metrics["DA%"], 1),  # same preds, same DA
                "MA_Sharpe": round(ma_sharpe, 3),
                "MA_TradeFreq%": strategy_metrics["trade_frequency"],
                "MA_WinRate%": strategy_metrics["win_rate"],
                "MA_AnnRet%": strategy_metrics["ma_ann_return"],
                "MA_MaxDD%": strategy_metrics["max_drawdown"],
                # Deltas
                "Sharpe_Delta": round(ma_sharpe - cmtf_metrics["Sharpe"], 3),
                "AnnRet_Delta%": round(strategy_metrics["ma_ann_return"] - strategy_metrics["cmtf_ann_return"], 2),
            }
            all_results.append(row)

            logger.info(
                "Result: CMTF Sharpe={:.3f} vs MA Sharpe={:.3f} (Δ={:+.3f}) | "
                "TradeFreq={:.0f}% WinRate={:.0f}%",
                cmtf_metrics["Sharpe"], ma_sharpe,
                ma_sharpe - cmtf_metrics["Sharpe"],
                strategy_metrics["trade_frequency"],
                strategy_metrics["win_rate"],
            )

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="A/B Benchmark: Multiagent vs CMTF-only")
    parser.add_argument("--symbols", nargs="+", default=["VCB", "BID"], help="Stock symbols")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20], help="Forecast horizons")
    parser.add_argument("--start", default="2025-01-02", help="Test period start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-03-31", help="Test period end (YYYY-MM-DD)")
    parser.add_argument("--output", default="results/ab_test_multiagent.csv", help="Output CSV path")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    logger.info("A/B Benchmark Configuration:")
    logger.info("  Symbols: {}", args.symbols)
    logger.info("  Horizons: {}", args.horizons)
    logger.info("  Period: {} → {}", args.start, args.end)
    logger.info("  Output: {}", args.output)

    results_df = run_ab_test(args.symbols, args.horizons, args.start, args.end)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    logger.info("Results saved → {}", args.output)

    # Print summary table
    print("\n" + "=" * 80)
    print("A/B BENCHMARK RESULTS: Multiagent (risk-gated) vs CMTF-only (always-trade)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)

    # Print interpretation
    if len(results_df) > 0:
        avg_sharpe_delta = results_df["Sharpe_Delta"].mean()
        avg_trade_freq = results_df["MA_TradeFreq%"].mean()
        avg_win_rate = results_df["MA_WinRate%"].mean()
        print(f"\nSUMMARY:")
        print(f"  Avg Sharpe improvement: {avg_sharpe_delta:+.3f}")
        print(f"  Avg Trade frequency:    {avg_trade_freq:.0f}%")
        print(f"  Avg Win rate (traded):  {avg_win_rate:.0f}%")
        print(f"  Interpretation: {'Risk gate improves risk-adjusted returns' if avg_sharpe_delta > 0 else 'Risk gate too conservative'}")


if __name__ == "__main__":
    main()
