"""Batch reflection orchestration for offline policy updates from settled decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import MultiAgentConfig
from .reflection import (
    DEFAULT_POLICY,
    load_policy,
    save_policy,
    update_policy_from_history,
)


def fetch_realized_returns(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Augment DataFrame with realized returns.
    
    Expected input columns: cutoff, horizon, action
    Adds column: realized_return (forward log return for horizon days)
    """
    from src.pipeline.data_fetcher import VnstockDataFetcher

    if df.empty:
        df = df.copy()
        df["realized_return"] = np.nan
        return df

    df = df.copy()
    fetcher = VnstockDataFetcher()

    # Fetch a wide date range
    cutoffs = df["cutoff"].unique()
    start = (pd.Timestamp(min(cutoffs)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(cutoffs)) + pd.Timedelta(days=int(df["horizon"].max()) * 3)).strftime("%Y-%m-%d")

    try:
        ohlcv = fetcher.fetch_ohlcv(symbol, start, end, "1D", "KBS")
        ohlcv = ohlcv.sort_index()
    except Exception as e:
        logger.error("Failed to fetch OHLCV for {}: {}", symbol, e)
        df["realized_return"] = np.nan
        return df

    df["realized_return"] = np.nan
    for idx, row in df.iterrows():
        cutoff_str = row["cutoff"]
        horizon = int(row["horizon"])

        cutoff_ts = pd.Timestamp(cutoff_str)
        mask = ohlcv.index <= cutoff_ts
        if mask.sum() == 0:
            continue
        cutoff_idx = ohlcv.index[mask][-1]
        cutoff_pos = ohlcv.index.get_loc(cutoff_idx)

        future_pos = cutoff_pos + horizon
        if future_pos >= len(ohlcv):
            continue

        close_now = ohlcv.iloc[cutoff_pos]["close"]
        close_future = ohlcv.iloc[future_pos]["close"]
        if close_now > 0:
            df.at[idx, "realized_return"] = float(np.log(close_future / close_now))

    return df


def settle_decisions_from_batch(
    batch_results: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Prepare settled decisions for reflection: fetch realized returns and compute outcomes.
    
    Input: batch_results from run_multiagent_predictions
    Output: DataFrame with action, fused_score, realized_return
    """
    settled = batch_results[[
        "cutoff", "horizon", "action", "fusion_score", "policy_version"
    ]].copy()

    settled = fetch_realized_returns(settled, symbol)

    # Filter out errors and flat decisions
    settled = settled[settled["action"].isin(["long", "short"])].copy()
    settled = settled[settled["realized_return"].notna()].copy()

    logger.info(
        "Settled {} decisions for {} ({} with realized returns)",
        len(batch_results), symbol, len(settled),
    )

    return settled


def apply_reflection_update(
    settled: pd.DataFrame,
    symbol: str,
    config: Optional[MultiAgentConfig] = None,
    min_samples: int = 30,
) -> dict:
    """Compute and persist policy update from settled decisions.
    
    Returns dict with: old_version, new_version, win_rate, avg_pnl, threshold_changes
    """
    if config is None:
        config = MultiAgentConfig()

    old_policy = load_policy(config.policy_store_path)
    old_version = old_policy.get("version", 1)

    # Filter trades and compute metrics
    trade_df = settled[settled["action"].isin(["long", "short"])].copy()
    if len(trade_df) > 0:
        trade_df["pnl"] = 0.0
        # For long trades: PnL = realized_return (positive if market goes up)
        # For short trades: PnL = -realized_return (positive if market goes down)
        trade_df.loc[trade_df["action"] == "long", "pnl"] = trade_df["realized_return"]
        trade_df.loc[trade_df["action"] == "short", "pnl"] = -trade_df["realized_return"]
        win_rate = float((trade_df["pnl"] > 0).mean())
        avg_pnl = float(trade_df["pnl"].mean())
    else:
        win_rate = 0.0
        avg_pnl = 0.0

    # Only update policy if we have sufficient samples
    changes = {}
    new_version = old_version
    if len(trade_df) >= min_samples:
        new_policy = update_policy_from_history(
            settled,
            min_samples=min_samples,
            base_policy=old_policy,
        )

        # Check what changed
        for key in old_policy:
            if key == "version":
                continue
            old_val = old_policy.get(key)
            new_val = new_policy.get(key)
            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}

        # Increment version only if something actually changed
        if changes:
            new_version = old_version + 1
            new_policy["version"] = new_version
            save_policy(new_policy, config.policy_store_path)
    else:
        logger.info(
            "Reflection | symbol={} | insufficient samples ({} < {}), skipping update",
            symbol, len(trade_df), min_samples,
        )

    logger.info(
        "Reflection | symbol={} | v{} → v{} | win_rate={:.2%} avg_pnl={:+.5f} | {} changes",
        symbol, old_version, new_version, win_rate, avg_pnl, len(changes),
    )

    return {
        "symbol": symbol,
        "old_version": old_version,
        "new_version": new_version,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "num_settled": len(settled),
        "threshold_changes": changes,
    }


def batch_reflection(
    batch_csv: str | Path,
    symbol: str,
    config: Optional[MultiAgentConfig] = None,
    min_samples: int = 30,
) -> dict:
    """End-to-end batch reflection: load batch results, settle outcomes, update policy.
    
    Args:
        batch_csv: Path to batch prediction results CSV
        symbol: Stock symbol for settled outcomes
        config: MultiAgentConfig (uses default if None)
        min_samples: Minimum settled trades before policy update triggers
    
    Returns:
        Dict with reflection results: symbol, version_update, win_rate, etc.
    """
    batch_csv = Path(batch_csv)
    if not batch_csv.exists():
        raise FileNotFoundError(f"Batch results not found: {batch_csv}")

    logger.info("Loading batch results from {}", batch_csv)
    batch_df = pd.read_csv(batch_csv)

    logger.info("Settling {} decisions for {}", len(batch_df), symbol)
    settled = settle_decisions_from_batch(batch_df, symbol)

    logger.info("Applying reflection update (min_samples={})", min_samples)
    result = apply_reflection_update(settled, symbol, config, min_samples)

    return result
