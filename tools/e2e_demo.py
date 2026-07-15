"""End-to-end demonstration of the MAS decision->narration->critic chain — the
product entry point for a historical (symbol, date) question.

Drives the real node chain (predict_agent -> gate_agent -> risk_agent ->
metalabel_agent -> narrator -> critic) over frozen predictions + REAL recent news
headlines and REAL trailing volatility/drawdown computed from the raw price data.
The orchestrator's live data-prep step (heavy PhoBERT/Chronos + network fetch via
prepare_single_cutoff) is exercised separately by ``python -m src.multiagent predict``;
this demo isolates the MAS's decision+veto+explanation+verification logic — including
the metalabel agent's real news lookup — so it runs fast and reliably while still
being backed by genuine market and news data, not placeholder constants.

Usage:
    python -m tools.e2e_demo --eval                 # deterministic, LLM-free
    python -m tools.e2e_demo                         # normal mode (real LLM narration + metalabel)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.multiagent.config import MultiAgentConfig
from src.multiagent.frozen_predictions import get_store
from src.multiagent.gate_io import load_gate_policy, policy_path
from src.multiagent.news_data import load_news_index, recent_headlines
from src.multiagent.agents.predict_agent import predict_agent_node
from src.multiagent.agents.gate_agent import gate_agent_node
from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.agents.metalabel_agent import metalabel_agent_node
from src.multiagent.agents.narrator_agent import narrator_agent_node
from src.multiagent.agents.critic_agent import critic_agent_node
from src.multiagent.trace import make_trace_record, render_step, build_manifest, write_trace_file
from src.multiagent.live_inference import resolve_price_parquet

HORIZON = 5


def _pick_rows(cfg):
    """Pick one abstain, one long, one short row for a representative demo."""
    store = get_store(HORIZON, cfg)
    pol, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, HORIZON, "VN"))
    picks = {}
    for (sym, d) in sorted(store._index, key=lambda x: (x[0], str(x[1]))):
        fp = store.get(sym, str(d))
        kind = "abstain" if abs(fp.gate_pred) < pol.tau else ("long" if fp.gate_pred > 0 else "short")
        if kind not in picks:
            picks[kind] = (sym, str(d))
        if len(picks) == 3:
            break
    return list(picks.values())


def _real_vol_and_drawdown(frame, symbol: str, cutoff: str) -> tuple[float, float]:
    """Real trailing 20d annualised volatility + max drawdown, from raw daily returns."""
    s = frame[(frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))].sort_index()
    daily = s["fwd_ret_1d"].iloc[-21:-1].to_numpy(dtype=float)
    if len(daily) < 5:
        return 0.0, 0.0
    vol_20d = float(np.nanstd(daily) * np.sqrt(252) * 100)
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max() * 100)
    return vol_20d, max_dd


def _run_row(cfg, symbol, date, frame, news_idx):
    vol_20d, max_dd = _real_vol_and_drawdown(frame, symbol, date)
    heads = recent_headlines(news_idx, symbol, date, lookback_days=5, k=15)
    real_articles = [{"title": h} for h in heads]  # feeds metalabel_agent's real news check

    state = {
        "symbol": symbol, "target_horizon_days": HORIZON, "prediction_time": date,
        "artifact_versions": {}, "node_timings": {},
        "volatility_metrics": {"vol_20d": vol_20d, "max_drawdown_pct": max_dd, "trend_pct": 0.0},
        "sentiment_metrics": {"coverage": len(heads), "staleness_frac": 0.0 if heads else 1.0,
                              "sentiment_mean": 0.0},  # scalar sentiment channel is inert (§1.4)
        "articles": real_articles,
    }
    records = []
    chain = [("predict_agent", predict_agent_node), ("gate_agent", gate_agent_node),
             ("risk_agent", risk_agent_node), ("metalabel_agent", metalabel_agent_node),
             ("narrator", narrator_agent_node), ("critic_agent", critic_agent_node)]
    for node_name, fn in chain:
        before = dict(state)
        update = fn(state, cfg)
        elapsed = update.get("node_timings", {}).get(node_name, 0.0)
        records.append(make_trace_record(node_name, elapsed, before, update))
        timings = {**state.get("node_timings", {}), **update.pop("node_timings", {})}
        state.update(update)
        state["node_timings"] = timings
    return state, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="LLM-free deterministic run")
    ap.add_argument("--out", default="results/agent_ablation/5d/e2e_demo.md")
    args = ap.parse_args()

    cfg = MultiAgentConfig(evaluation_mode=args.eval, trace_enabled=True)
    rows = _pick_rows(cfg)
    frame = pd.read_parquet(resolve_price_parquet(HORIZON))
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    news_idx = load_news_index()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg, eval_mode=args.eval, seed=0,
                              extra={"demo_rows": rows,
                                     "note": "decision->veto->narration->critic chain over frozen "
                                             "predictions + real news + real trailing vol/drawdown"})
    lines = ["# MAS end-to-end demonstration", "",
             f"Mode: {'EVAL (deterministic, LLM-free)' if args.eval else 'NORMAL (real LLM narration + metalabel)'}", ""]
    for (sym, date) in rows:
        state, records = _run_row(cfg, sym, date, frame, news_idx)
        lines.append(f"## {sym} @ {date} -> **{state['action'].upper()}** (size {state['position_scale']:+.2f})")
        lines.append("")
        for i, rec in enumerate(records, 1):
            lines.append("```text")
            lines.append(render_step(rec, i, len(records)))
            lines.append("```")
        lines.append(f"\n**Answer:** {state.get('answer_text') or '(eval mode: empty for determinism)'}")
        lines.append(f"\n**Critic:** status={state.get('critic_status')} findings={state.get('critic_findings')}")
        lines.append(f"\n**Metalabel:** flags={state.get('metalabel_flags')} vetoed={state.get('metalabel_vetoed')}")
        lines.append("")
        print(f"{sym} {date} -> {state['action']} size={state['position_scale']:+.2f} "
              f"critic={state.get('critic_status')} metalabel_vetoed={state.get('metalabel_vetoed')}")

    import json
    header = ["## Run manifest", "", "```json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str), "```", ""]
    out.write_text("\n".join(lines[:3] + header + lines[3:]), encoding="utf-8")
    print(f"\nTranscript -> {out}")


if __name__ == "__main__":
    main()
