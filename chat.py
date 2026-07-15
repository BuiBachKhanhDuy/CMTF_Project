#!/usr/bin/env python
"""Interactive Multi-Agent System chat — run ONE file, type queries, watch every node work.

Run:
    .venv/Scripts/python.exe chat.py            # fast (deterministic, grounded answer, no LLM)
    .venv/Scripts/python.exe chat.py --llm       # real LLM narration + metalabel news check (needs Ollama)

Then just type, e.g.:
    VCB                          -> latest cached date, 5d
    VCB 2025-08-13               -> that date
    BID có nên mua 5 ngày tới    -> natural Vietnamese, symbol+horizon parsed
    rank VCB,BID,CTG 2025-08-13  -> cross-sectional ranking branch
    help / symbols / quit

After every answer it prints the full node-by-node trace with each step's result, so
you can inspect exactly what the system did (predict -> gate -> risk -> metalabel ->
narrator -> critic).

Dates inside the research book (shown in the banner) are served instantly from the
frozen prediction cache. A date OUTSIDE the book triggers a real live forward pass of
the deployed champion — the pipeline fetches live OHLCV + scrapes news for the whole
universe, so it can take many minutes (cold). Predictions are always real, never faked;
if data for a live date is unavailable the system reports it and abstains.
"""

from __future__ import annotations

import argparse
import re
import sys

# Vietnamese narration/answers contain characters outside cp1252 (Windows' default
# console codepage), which crashes print() before a single query is even processed
# when stdout isn't already UTF-8 (e.g. piped/redirected). Force UTF-8 unconditionally.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# Quiet the noisy third-party import banners before anything else loads them.
from loguru import logger
logger.remove()  # chat prints its own clean step view; suppress per-node INFO spam

from src.multiagent.config import MultiAgentConfig
from src.multiagent.frozen_predictions import get_store, PredictionNotCachedError
from src.multiagent.loaders import ArtifactMissingError
from src.multiagent.gate_io import load_gate_policy, policy_path
from src.multiagent.news_data import load_news_index, recent_headlines
from src.multiagent.agents.predict_agent import predict_agent_node
from src.multiagent.agents.gate_agent import gate_agent_node
from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.agents.metalabel_agent import metalabel_agent_node
from src.multiagent.agents.narrator_agent import narrator_agent_node
from src.multiagent.agents.critic_agent import critic_agent_node
from src.multiagent.agents.rank_agent import rank_agent_node
from src.multiagent.trace import summarize_node
from src.multiagent.live_inference import resolve_price_parquet

KNOWN_SYMBOLS = ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB"]
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

DECISION_CHAIN = [
    ("predict_agent", predict_agent_node),
    ("gate_agent", gate_agent_node),
    ("risk_agent", risk_agent_node),
    ("metalabel_agent", metalabel_agent_node),
    ("narrator", narrator_agent_node),
    ("critic_agent", critic_agent_node),
]


def _symbol_dates(store):
    by = {}
    for (sym, d) in store._index:
        by.setdefault(sym, []).append(str(d))
    return {s: sorted(v) for s, v in by.items()}


def _real_vol_dd(frame, symbol, cutoff):
    """Real trailing 20d vol% + drawdown% from price history, or (None, None) if the
    date is beyond the price parquet (a live date) — never fabricated as 0.0, which the
    risk agent would read as 'calm' and skip its safety veto."""
    s = frame[(frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))].sort_index()
    daily = s["fwd_ret_1d"].iloc[-21:-1].to_numpy(dtype=float)
    if len(daily) < 5:
        return None, None
    vol = float(np.nanstd(daily) * np.sqrt(252) * 100)
    cum = np.cumsum(daily); peak = np.maximum.accumulate(cum)
    return vol, float((peak - cum).max() * 100)


def _parse(query, sym_dates, default_symbol, default_horizon):
    q = query.strip()
    up = q.upper()
    symbol = next((s for s in KNOWN_SYMBOLS if re.search(rf"\b{s}\b", up)), default_symbol)
    if re.search(r"\b1\s*(ngày|day|d)\b", q.lower()) or " 1d" in q.lower():
        horizon = 1
    elif re.search(r"\b20\s*(ngày|day|d)\b", q.lower()) or "20d" in q.lower():
        horizon = 20
    else:
        horizon = default_horizon
    m = _DATE_RE.search(q)
    date = m.group(0) if m else (sym_dates.get(symbol, [None])[-1] if symbol else None)
    return symbol, date, horizon


def _run_decision(symbol, date, horizon, cfg, frame, news_idx):
    vol, dd = _real_vol_dd(frame, symbol, date)
    heads = recent_headlines(news_idx, symbol, date, lookback_days=5, k=15)
    warnings = []
    if vol is None:
        # Live/out-of-book date: no trailing prices in the local parquet → cannot run
        # the volatility safety veto. Report it; do NOT pass 0.0 (would look 'calm').
        warnings.append("risk: no local price history for this date — safety veto disabled")
        vol_metrics = {"trend_pct": 0.0}  # vol_20d / max_drawdown_pct intentionally absent
    else:
        vol_metrics = {"vol_20d": vol, "max_drawdown_pct": dd, "trend_pct": 0.0}
    state = {
        "symbol": symbol, "target_horizon_days": horizon, "prediction_time": date,
        "artifact_versions": {}, "node_timings": {}, "warnings": warnings,
        "volatility_metrics": vol_metrics,
        "sentiment_metrics": {"coverage": len(heads), "staleness_frac": 0.0 if heads else 1.0,
                              "sentiment_mean": 0.0},
        "articles": [{"title": h} for h in heads],
    }
    steps = []
    for name, fn in DECISION_CHAIN:
        before = dict(state)
        upd = fn(state, cfg)
        steps.append((name, summarize_node(name, before, {**before, **upd})))
        timings = {**state.get("node_timings", {}), **upd.pop("node_timings", {})}
        state.update(upd); state["node_timings"] = timings
    return state, steps


def _print_steps(steps):
    print("\n  ── how the system decided (node by node) " + "─" * 30)
    for i, (name, summary) in enumerate(steps, 1):
        fields = "  ".join(f"{k}={v}" for k, v in summary.items()) if summary else "(no output)"
        print(f"  STEP {i} · {name:<15} {fields}")
    print("  " + "─" * 70)


def _handle_rank(query, cfg, store_h):
    parts = query.split()
    syms = next((p for p in parts if "," in p), None)
    m = _DATE_RE.search(query)
    if not syms or not m:
        print("  usage: rank VCB,BID,CTG 2025-08-13")
        return
    symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
    out = rank_agent_node({"target_symbols": symbols, "target_horizon_days": 5,
                           "prediction_time": m.group(0), "node_timings": {}}, cfg)
    print(f"\n  RANKING @ {m.group(0)}:")
    print(f"    LONG : {out['rank_longs']}")
    print(f"    SHORT: {out['rank_shorts']}")
    print(f"    ABSTAIN: {out['rank_abstained']}")
    if out["warnings"]:
        print(f"    ! {out['warnings']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="real LLM narration + metalabel news check (needs Ollama)")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    args = ap.parse_args()

    cfg = MultiAgentConfig(evaluation_mode=not args.llm)
    print("Loading model predictions, gate, price & news data …")
    store = get_store(args.horizon, cfg)
    sym_dates = _symbol_dates(store)
    frame = pd.read_parquet(resolve_price_parquet(args.horizon))
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    news_idx = load_news_index()
    lo = min(d for ds in sym_dates.values() for d in ds)
    hi = max(d for ds in sym_dates.values() for d in ds)

    mode = "LLM (real narration + metalabel)" if args.llm else "FAST (deterministic grounded answer, no LLM)"
    print(f"\n{'='*74}\n  Multi-Agent Stock Advisor  ·  mode: {mode}")
    print(f"  Symbols: {', '.join(KNOWN_SYMBOLS)}   Cached dates: {lo} .. {hi}")
    print(f"  Type a query (e.g. 'VCB 2025-08-13' or 'BID có nên mua 5 ngày tới').")
    print(f"  Commands: help · symbols · rank VCB,BID 2025-08-13 · quit")
    print("=" * 74)

    last_symbol = "VCB"
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); break
        if not query:
            continue
        low = query.lower()
        if low in ("quit", "exit", "q"):
            print("bye."); break
        if low in ("help", "?"):
            print("  Type: <SYMBOL> [YYYY-MM-DD] [horizon]  |  natural Vietnamese question  |  "
                  "rank A,B,C DATE  |  symbols  |  quit"); continue
        if low == "symbols":
            print(f"  {', '.join(KNOWN_SYMBOLS)}  (dates {lo}..{hi})"); continue
        if low.startswith("rank"):
            _handle_rank(query, cfg, store); continue

        symbol, date, horizon = _parse(query, sym_dates, last_symbol, args.horizon)
        if symbol is None:
            print("  Couldn't find a symbol. Try one of: " + ", ".join(KNOWN_SYMBOLS)); continue
        last_symbol = symbol
        in_book = symbol in sym_dates and date in sym_dates.get(symbol, [])
        note = ""
        if args.llm:
            note = "   (thinking, LLM calls may take ~1-2 min) …"
        elif not in_book:
            note = "   (LIVE date — fetching real OHLCV + news; this can take many minutes) …"
        else:
            note = " …"
        print(f"  → interpreting as: symbol={symbol}  date={date}  horizon={horizon}d" + note)
        try:
            state, steps = _run_decision(symbol, date, horizon, cfg, frame, news_idx)
        except PredictionNotCachedError as e:
            print(f"  ⚠ {str(e).splitlines()[0]}"); continue
        except ArtifactMissingError as e:
            print(f"  ⚠ live inference unavailable: {str(e).splitlines()[0]}"); continue
        except Exception as e:  # noqa: BLE001 — surface, never hide (R1)
            print(f"  ⚠ error: {type(e).__name__}: {e}"); continue

        action = state.get("action", "?").upper()
        size = state.get("position_scale", 0.0)
        answer = state.get("answer_text") or state.get("grounded_answer") or "(no answer)"
        src = (state.get("model_evidence") or {}).get("source", "")
        tag = "  [live forward pass]" if src == "live_inference" else ""
        print(f"\nSystem: [{action}  size {size:+.2f}]{tag}  {answer}")
        for w in state.get("warnings", []):
            print(f"  ⚠ {w}")
        _print_steps(steps)


if __name__ == "__main__":
    sys.exit(main())
