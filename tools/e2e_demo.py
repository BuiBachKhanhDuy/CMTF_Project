"""End-to-end demonstration of the MAS decision->narration->critic->reasoning chain —
the product entry point for a historical (symbol, date) question.

Drives the real, current node chain (predict_agent -> gate_agent ->
horizon_interaction_agent -> risk_agent -> metalabel_agent -> narrator ->
critic_agent -> reasoning_agent) with REAL recent news headlines and REAL trailing
volatility/drawdown computed from the raw price data. `predict_agent` always runs a
real forward pass now (never the frozen `.npy` cache — see `live_inference.py`), so
this demo also surfaces real attention/recency-gate explainability, not just the
decision+veto+explanation+verification logic. `reasoning_agent`'s widen-and-rerun is
wired exactly like `chat.py`'s `_run_decision` (widening `frame`/`news_idx` is free
here — both are already loaded in memory — so a triggered row genuinely re-runs the
whole chain once more with wider evidence, not a stub).
The orchestrator's live data-prep step (heavy PhoBERT/Chronos + network fetch via
prepare_single_cutoff) is exercised separately by ``python -m src.multiagent predict``;
this demo isolates the MAS's decision+veto+explanation+verification+reflection logic —
including the metalabel agent's real news lookup — so it runs fast and reliably while
still being backed by genuine market and news data, not placeholder constants.

Usage:
    python -m tools.e2e_demo --eval                 # deterministic, LLM-free
    python -m tools.e2e_demo                         # normal mode (real LLM narration + metalabel)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.multiagent.config import MultiAgentConfig
from src.multiagent.frozen_predictions import get_store
from src.multiagent.gate_io import load_gate_policy, policy_path
from src.multiagent.news_data import load_news_index
from src.multiagent.agents.predict_agent import predict_agent_node
from src.multiagent.agents.gate_agent import gate_agent_node
from src.multiagent.agents.horizon_interaction_agent import horizon_interaction_agent_node
from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.agents.metalabel_agent import metalabel_agent_node
from src.multiagent.agents.narrator_agent import narrator_agent_node
from src.multiagent.agents.critic_agent import critic_agent_node
from src.multiagent.agents.reasoning_agent import reasoning_agent_node
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


_CHAIN = [
    ("predict_agent", predict_agent_node), ("gate_agent", gate_agent_node),
    ("horizon_interaction_agent", horizon_interaction_agent_node), ("risk_agent", risk_agent_node),
    ("metalabel_agent", metalabel_agent_node), ("narrator", narrator_agent_node),
    ("critic_agent", critic_agent_node), ("reasoning_agent", reasoning_agent_node),
]


def _run_row(cfg, symbol, date, frame, news_idx):
    import chat  # reuse the exact same real-evidence gathering chat.py's product path uses

    evidence = chat._gather_evidence(frame, news_idx, symbol, date)
    state = {
        "symbol": symbol, "target_horizon_days": HORIZON, "prediction_time": date,
        "artifact_versions": {}, "node_timings": {}, **evidence,
    }

    def widen_and_rerun(current_state):
        """Real second look, exactly like `chat.py`'s `_run_decision` — widening the
        already-loaded `frame`/`news_idx` costs nothing, so a triggered row genuinely
        re-runs predict->...->critic_agent once more with wider evidence."""
        wider_evidence = chat._gather_evidence(
            frame, news_idx, symbol, date,
            lookback_days=cfg.reasoning_widen_lookback_days_to,
            vol_window=cfg.reasoning_widen_sequence_len_to,
        )
        rerun_state = {**current_state, **wider_evidence}
        for _, fn in _CHAIN[:7]:  # predict through critic_agent — not reasoning_agent itself
            rerun_state.update(fn(rerun_state, cfg))
        return rerun_state

    records = []
    for node_name, fn in _CHAIN:
        before = dict(state)
        update = fn(state, cfg, widen_and_rerun=widen_and_rerun) if node_name == "reasoning_agent" else fn(state, cfg)
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
                                     "note": "predict(live forward pass)->gate->horizon_interaction->"
                                             "risk->metalabel->narrator->critic->reasoning chain, real "
                                             "news + real trailing vol/drawdown, real attention/recency-"
                                             "gate explainability (no frozen-cache shortcut)"})
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
        lines.append(f"\n**Attention (top trailing days):** {state.get('attention_top_days')}")
        lines.append(f"\n**Reasoning agent:** triggered={state.get('reasoning_triggered_reasons')} "
                     f"widened={state.get('reasoning_evidence_widened')} notes={state.get('reasoning_notes')}")
        lines.append("")
        print(f"{sym} {date} -> {state['action']} size={state['position_scale']:+.2f} "
              f"critic={state.get('critic_status')} metalabel_vetoed={state.get('metalabel_vetoed')}")

    import json
    header = ["## Run manifest", "", "```json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str), "```", ""]
    out.write_text("\n".join(lines[:3] + header + lines[3:]), encoding="utf-8")
    print(f"\nTranscript -> {out}")


if __name__ == "__main__":
    main()
