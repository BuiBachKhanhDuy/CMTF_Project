"""A/B Benchmark: Agent consensus correction + risk management evaluation.

Produces four outputs:
  1) results/ab_test_multiagent.csv    — per-cutoff raw trace (for equity curves)
  2) results/ab_agent_correction.csv   — agent correction metrics (CMTF vs Adjusted)
  3) results/benchmark_risk.csv        — risk management metrics
  4) results/figures/*.png             — waterfall, temporal, risk, equity figures

Agent correction compares CMTF-only vs per-agent-adjusted predictions:
  - CMTF only (final_pred)
  - + market_agent correction (mkt_adjusted_pred)
  - + news_agent correction (news_adjusted_pred)
  - + both agents (adjusted_pred)

Risk management compares 3 execution strategies on the SAME prediction:
  - Always-trade: sign(final_pred) every day
  - Threshold-only: trade only if |final_pred| >= buy_threshold
  - Risk-gated: multiagent action + position_scale

Usage:
    python run_ab_benchmark.py --symbols VCB --horizons 1
    python run_ab_benchmark.py --symbols VCB BID --horizons 1 5 20
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
from src.multiagent.reflection import load_policy
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
    max_drawdown,
    calmar_ratio,
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
    pause_every: int = 4,
    pause_seconds: float = 20.0,
) -> pd.DataFrame:
    """Run multiagent system (eval mode) for all cutoffs. Returns DataFrame.

    Uses rate-limit-aware pacing to avoid vnstock API throttling.
    """
    config = MultiAgentConfig(evaluation_mode=True)
    results = []

    for i, cutoff in enumerate(cutoffs):
        # Conservative pacing for Guest plan (20 req/min).
        # Each cutoff triggers about 2 OHLCV calls (symbol + VNINDEX).
        if pause_every > 0 and i > 0 and i % pause_every == 0:
            logger.info(
                "Rate limit pause ({:.1f}s) after {} cutoffs...",
                pause_seconds,
                i,
            )
            time.sleep(max(0.0, pause_seconds))
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
                "adjusted_pred": state.get("adjusted_pred", state["final_pred"]),
                "mkt_adjusted_pred": state.get("mkt_adjusted_pred", state["final_pred"]),
                "news_adjusted_pred": state.get("news_adjusted_pred", state["final_pred"]),
                "action": state["action"],
                "position_scale": state["position_scale"],
                "predict_confidence": state["predict_confidence"],
                "risk_passed": state["risk_checks"].get("tier") not in ("blocked",),
                "market_direction": state.get("market_proposal", {}).get("direction"),
                "news_direction": state.get("news_proposal", {}).get("direction"),
                "fusion_score": state.get("fusion_decision", {}).get("score"),
                "mkt_contribution_pct": state.get("fusion_decision", {}).get("mkt_contribution_pct", 0.0),
                "news_contribution_pct": state.get("fusion_decision", {}).get("news_contribution_pct", 0.0),
                "policy_version": state.get("policy_version", 1),
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
                "adjusted_pred": np.nan,
                "mkt_adjusted_pred": np.nan,
                "news_adjusted_pred": np.nan,
                "action": "error",
                "position_scale": 0.0,
                "predict_confidence": 0.0,
                "risk_passed": False,
                "market_direction": None,
                "news_direction": None,
                "fusion_score": np.nan,
                "mkt_contribution_pct": 0.0,
                "news_contribution_pct": 0.0,
                "policy_version": np.nan,
            })

    return pd.DataFrame(results)


def compute_prediction_metrics(
    actual: np.ndarray,
    baseline: np.ndarray,
    final: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """Compute prediction quality for Baseline vs Final (news-enhanced)."""
    result = {}
    for prefix, pred in [("Base", baseline), ("Final", final)]:
        result[f"{prefix}_MAE"] = round(mae(actual, pred), 6)
        result[f"{prefix}_RMSE"] = round(rmse(actual, pred), 6)
        result[f"{prefix}_DA%"] = round(directional_accuracy(actual, pred), 1)
        result[f"{prefix}_IC"] = round(information_coefficient(actual, pred), 4)
        result[f"{prefix}_Prec"] = round(direction_precision(actual, pred), 3)
        result[f"{prefix}_Rec"] = round(direction_recall(actual, pred), 3)
        result[f"{prefix}_F1"] = round(direction_f1(actual, pred), 3)
        result[f"{prefix}_Sharpe"] = round(sharpe_ratio(actual, pred, horizon=horizon), 3)
    return result


def compute_agent_correction_metrics(
    actual: np.ndarray,
    final_pred: np.ndarray,
    mkt_adjusted: np.ndarray,
    news_adjusted: np.ndarray,
    adjusted: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """Compute per-agent correction metrics: CMTF vs each agent variant."""
    result = {}
    for prefix, pred in [("CMTF", final_pred), ("Mkt", mkt_adjusted),
                         ("News", news_adjusted), ("Both", adjusted)]:
        result[f"{prefix}_DA%"] = round(directional_accuracy(actual, pred), 1)
        result[f"{prefix}_IC"] = round(information_coefficient(actual, pred), 4)
        result[f"{prefix}_MAE"] = round(mae(actual, pred), 6)
    return result


def _strategy_returns(actual: np.ndarray, pred: np.ndarray, actions: np.ndarray,
                      scales: np.ndarray, buy_threshold: float):
    """Compute returns for 3 execution strategies on the same prediction."""
    n = len(actual)
    # Always-trade: sign(pred) every day
    at_ret = np.sign(pred) * actual
    # Threshold-only: trade if |pred| >= threshold
    th_ret = np.where(np.abs(pred) >= buy_threshold, np.sign(pred) * actual, 0.0)
    # Risk-gated: multiagent action + position_scale
    rg_ret = np.zeros(n)
    for i in range(n):
        if actions[i] == "long":
            rg_ret[i] = actual[i] * scales[i]
        elif actions[i] == "short":
            rg_ret[i] = -actual[i] * scales[i]
    return at_ret, th_ret, rg_ret


def _sharpe_from_returns(returns: np.ndarray, horizon: int) -> float:
    """Annualized Sharpe directly from strategy returns."""
    if len(returns) < 3 or returns.std() == 0:
        return float("nan")
    ann = np.sqrt(252.0 / max(horizon, 1))
    return float((returns.mean() / returns.std()) * ann)


def _win_rate(returns: np.ndarray) -> float:
    """Win rate: fraction of trades with positive return (ignoring flat days)."""
    traded = returns[returns != 0]
    if len(traded) == 0:
        return 0.0
    return float((traded > 0).mean() * 100)


def _trade_freq(returns: np.ndarray) -> float:
    """Trade frequency: fraction of days with non-zero return."""
    return float((returns != 0).mean() * 100)


def compute_risk_metrics(
    actual: np.ndarray,
    final_pred: np.ndarray,
    actions: np.ndarray,
    scales: np.ndarray,
    horizon: int,
    buy_threshold: float = 0.012,
) -> dict[str, float]:
    """Compute risk management metrics for 3 execution strategies."""
    at_ret, th_ret, rg_ret = _strategy_returns(
        actual, final_pred, actions, scales, buy_threshold,
    )
    result = {}
    for prefix, ret in [("AT", at_ret), ("Thresh", th_ret), ("RG", rg_ret)]:
        result[f"{prefix}_Sharpe"] = round(_sharpe_from_returns(ret, horizon), 3)
        result[f"{prefix}_MaxDD%"] = round(max_drawdown(ret) * 100, 2)
        result[f"{prefix}_WinRate%"] = round(_win_rate(ret), 1)
        result[f"{prefix}_TradeFreq%"] = round(_trade_freq(ret), 1)
        result[f"{prefix}_Calmar"] = round(calmar_ratio(ret, horizon), 3)
        ann_factor = 252.0 / max(horizon, 1)
        result[f"{prefix}_AnnRet%"] = round(float(ret.mean() * ann_factor * 100), 2)
    return result


def run_ab_test(
    symbols: list[str],
    horizons: list[int],
    start: str,
    end: str,
    pause_every: int = 4,
    pause_seconds: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full A/B comparison. Returns (raw_trace, correction_df, risk_df)."""
    raw_rows: list[dict] = []
    corr_rows: list[dict] = []
    risk_rows: list[dict] = []

    policy = load_policy("results/multiagent_policy.json")

    for symbol in symbols:
        for horizon in horizons:
            logger.info("=" * 60)
            logger.info("A/B Test: {} {}d | {} → {}", symbol, horizon, start, end)
            logger.info("=" * 60)

            clear_prepare_cache()

            cutoffs = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
            logger.info("Cutoff dates: {} ({} → {})", len(cutoffs), cutoffs[0], cutoffs[-1])

            # 1. Actual forward returns
            t0 = time.time()
            actual_returns = _compute_actual_returns(symbol, cutoffs, horizon)
            logger.info("Actual returns: {}/{} valid | {:.1f}s",
                        len(actual_returns), len(cutoffs), time.time() - t0)

            valid_cutoffs = [c for c in cutoffs if c in actual_returns]
            if len(valid_cutoffs) < 10:
                logger.warning("Too few valid cutoffs for {} {}d: {}", symbol, horizon, len(valid_cutoffs))
                continue

            # 2. Multiagent predictions
            t0 = time.time()
            pred_df = run_multiagent_predictions(
                symbol, valid_cutoffs, horizon,
                pause_every=pause_every, pause_seconds=pause_seconds,
            )
            logger.info("Predictions complete | {:.1f}s", time.time() - t0)

            # 3. Align
            pred_df["actual_return"] = pred_df["cutoff"].map(actual_returns)
            pred_df = pred_df.dropna(subset=["actual_return", "final_pred"])

            if len(pred_df) < 10:
                logger.warning("Too few aligned for {} {}d: {}", symbol, horizon, len(pred_df))
                continue

            actual = pred_df["actual_return"].values
            final = pred_df["final_pred"].values
            mkt_adj = pred_df["mkt_adjusted_pred"].values
            news_adj = pred_df["news_adjusted_pred"].values
            adjusted = pred_df["adjusted_pred"].values
            actions = pred_df["action"].values
            scales = pred_df["position_scale"].values.astype(float)

            # 4a. Agent correction metrics
            cm = compute_agent_correction_metrics(
                actual, final, mkt_adj, news_adj, adjusted, horizon,
            )
            corr_row = {"Symbol": symbol, "Horizon": f"{horizon}d",
                        "N_Samples": len(pred_df), **cm}
            corr_rows.append(corr_row)

            # 4b. Risk management
            rm = compute_risk_metrics(actual, final, actions, scales, horizon,
                                      buy_threshold=float(policy["buy_threshold"]))
            risk_row = {"Symbol": symbol, "Horizon": f"{horizon}d",
                        "N_Samples": len(pred_df), **rm}
            risk_rows.append(risk_row)

            # Raw trace (per-cutoff)
            for _, r in pred_df.iterrows():
                raw_rows.append({
                    "Symbol": symbol, "Horizon": f"{horizon}d",
                    "Cutoff": r["cutoff"],
                    "Actual": r["actual_return"],
                    "Baseline_Pred": r["baseline_pred"],
                    "Final_Pred": r["final_pred"],
                    "Adjusted_Pred": r["adjusted_pred"],
                    "Mkt_Adjusted_Pred": r["mkt_adjusted_pred"],
                    "News_Adjusted_Pred": r["news_adjusted_pred"],
                    "Mkt_Contribution_Pct": r.get("mkt_contribution_pct", 0.0),
                    "News_Contribution_Pct": r.get("news_contribution_pct", 0.0),
                    "Action": r["action"],
                    "Position_Scale": r["position_scale"],
                })

            logger.info(
                "Result: CMTF DA={:.1f}% Adj DA={:.1f}% | "
                "AT Sharpe={:.3f} RG Sharpe={:.3f} | TradeFreq={:.0f}%",
                cm["CMTF_DA%"], cm["Both_DA%"],
                rm["AT_Sharpe"], rm["RG_Sharpe"], rm["RG_TradeFreq%"],
            )

    return pd.DataFrame(raw_rows), pd.DataFrame(corr_rows), pd.DataFrame(risk_rows)


def main():
    parser = argparse.ArgumentParser(description="A/B Benchmark: Split prediction + risk")
    parser.add_argument("--symbols", nargs="+", default=["VCB", "BID"], help="Stock symbols")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20], help="Forecast horizons")
    parser.add_argument("--start", default="2025-01-02", help="Test period start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-03-31", help="Test period end (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--pause-every", type=int, default=4, help="Pause after N cutoffs (0 disables)")
    parser.add_argument("--pause-seconds", type=float, default=20.0, help="Pause duration in seconds")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("A/B Benchmark Configuration:")
    logger.info("  Symbols: {}", args.symbols)
    logger.info("  Horizons: {}", args.horizons)
    logger.info("  Period: {} → {}", args.start, args.end)
    logger.info("  Output: {}", out_dir)

    raw_df, corr_df, risk_df = run_ab_test(
        args.symbols, args.horizons, args.start, args.end,
        pause_every=args.pause_every, pause_seconds=args.pause_seconds,
    )

    # Save CSVs
    raw_path = out_dir / "ab_test_multiagent.csv"
    corr_path = out_dir / "ab_agent_correction.csv"
    risk_path = out_dir / "benchmark_risk.csv"

    raw_df.to_csv(raw_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    risk_df.to_csv(risk_path, index=False)

    logger.info("Saved → {}", raw_path)
    logger.info("Saved → {}", corr_path)
    logger.info("Saved → {}", risk_path)

    # Print summaries
    if len(corr_df) > 0:
        print("\n" + "=" * 80)
        print("AGENT CORRECTION: CMTF vs +Market vs +News vs +Both")
        print("=" * 80)
        print(corr_df.to_string(index=False))

    if len(risk_df) > 0:
        print("\n" + "=" * 80)
        print("RISK MANAGEMENT: Always-Trade vs Threshold-Only vs Risk-Gated")
        print("=" * 80)
        print(risk_df.to_string(index=False))
        avg_rg_sharpe = risk_df["RG_Sharpe"].mean()
        avg_at_sharpe = risk_df["AT_Sharpe"].mean()
        avg_trade_freq = risk_df["RG_TradeFreq%"].mean()
        print(f"\nSUMMARY:")
        print(f"  Avg AT Sharpe:    {avg_at_sharpe:+.3f}")
        print(f"  Avg RG Sharpe:    {avg_rg_sharpe:+.3f}")
        print(f"  Avg Trade Freq:   {avg_trade_freq:.0f}%")

    # --- Generate figures ---
    if not args.no_plot:
        from src.benchmark.plots import (
            plot_agent_waterfall,
            plot_agent_temporal,
            plot_risk_benchmark,
            plot_equity_curves,
        )
        fig_dir = out_dir / "figures"
        if len(corr_df) > 0:
            plot_agent_waterfall(corr_df, fig_dir / "agent_correction_waterfall.png")
        if len(raw_df) > 0:
            plot_agent_temporal(raw_df, fig_dir / "agent_correction_temporal.png")
        if len(risk_df) > 0:
            plot_risk_benchmark(risk_df, fig_dir / "risk_benchmark.png")
        if len(raw_df) > 0:
            plot_equity_curves(raw_df, fig_dir / "equity_curves.png")


if __name__ == "__main__":
    main()
