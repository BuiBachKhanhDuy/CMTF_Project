"""Offline reflection utilities for updating policy thresholds from settled decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POLICY: dict[str, Any] = {
    "version": 1,
    "buy_threshold": 0.012,
    "sell_threshold": -0.012,
    "weak_signal": 0.001,
    "hard_block_vol": 40.0,
    "hard_block_drawdown": 20.0,
    "hard_block_min_confidence": 0.10,
    "reduced_vol": 30.0,
    "reduced_min_confidence": 0.25,
    "min_news_coverage": 0,
    "max_staleness_frac": 1.0,
    "override_alpha": 0.3,
}


def load_policy(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_POLICY)
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_POLICY)

    merged = dict(DEFAULT_POLICY)
    merged.update(loaded)
    return merged


def save_policy(policy: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")


def update_policy_from_history(
    settled: pd.DataFrame,
    *,
    min_samples: int = 30,
    base_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simple bounded policy update from settled trade outcomes.

    Expected columns (minimum): action, fused_score, realized_return.
    """
    policy = dict(base_policy or DEFAULT_POLICY)

    required_cols = {"action", "fused_score", "realized_return"}
    if settled.empty or not required_cols.issubset(settled.columns) or len(settled) < min_samples:
        return policy

    trade_df = settled[settled["action"].isin(["long", "short"])].copy()
    if len(trade_df) < min_samples:
        return policy

    trade_df["pnl"] = 0.0
    trade_df.loc[trade_df["action"] == "long", "pnl"] = trade_df["realized_return"]
    trade_df.loc[trade_df["action"] == "short", "pnl"] = -trade_df["realized_return"]

    win_rate = float((trade_df["pnl"] > 0).mean())
    avg_pnl = float(trade_df["pnl"].mean())

    # Conservative bounded updates to avoid policy drift.
    # Adjust reduced_min_confidence (tier boundary), NOT hard_block_min_confidence (safety floor).
    if win_rate < 0.5 or avg_pnl < 0:
        policy["reduced_min_confidence"] = round(min(0.6, float(policy["reduced_min_confidence"]) + 0.02), 3)
        policy["buy_threshold"] = round(min(0.04, float(policy["buy_threshold"]) + 0.001), 4)
        policy["sell_threshold"] = round(max(-0.04, float(policy["sell_threshold"]) - 0.001), 4)
    else:
        policy["reduced_min_confidence"] = round(max(0.15, float(policy["reduced_min_confidence"]) - 0.01), 3)
        policy["buy_threshold"] = round(max(0.008, float(policy["buy_threshold"]) - 0.0005), 4)
        policy["sell_threshold"] = round(min(-0.008, float(policy["sell_threshold"]) + 0.0005), 4)

    # --- Learn override_alpha: compare DA of final_pred vs adjusted_pred ---
    alpha_cols = {"final_pred", "adjusted_pred", "realized_return"}
    if alpha_cols.issubset(settled.columns):
        eval_df = settled.dropna(subset=["final_pred", "adjusted_pred", "realized_return"])
        if len(eval_df) >= min_samples:
            actual = eval_df["realized_return"].values
            final = eval_df["final_pred"].values
            adjusted = eval_df["adjusted_pred"].values
            da_final = float(np.mean(np.sign(final) == np.sign(actual))) * 100
            da_adjusted = float(np.mean(np.sign(adjusted) == np.sign(actual))) * 100

            current_alpha = float(policy.get("override_alpha", 0.3))
            if da_adjusted > da_final + 2.0:
                # Agent correction helps → trust agents more
                current_alpha = min(0.8, current_alpha + 0.05)
            elif da_adjusted < da_final - 2.0:
                # Agent correction hurts → trust agents less
                current_alpha = max(0.05, current_alpha - 0.05)
            policy["override_alpha"] = round(current_alpha, 3)

    policy["version"] = int(policy.get("version", 1)) + 1
    return policy
