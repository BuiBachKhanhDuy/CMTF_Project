"""Traceability layer (plan §9, R3).

A human can follow the whole workflow — every node, its timing, its key reads and
outputs, the gate decision, and the veto — from one command. Each node's output is
turned into one structured trace record; ``--trace`` renders them live to the
console and ``--trace-file`` writes a Markdown transcript. A run manifest (git SHA,
config, artifact versions, seeds, eval flag) makes any run reproducible.

The graph wrapper (``graph._make_node_with_config``) appends a record per node, so
individual agents stay clean — the trace is derived from the state update each node
already returns.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:+.5f}"
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]"
    return str(v)


# Per-node summary formatters: (reads shown, output keys shown). Keeps the trace
# focused on the decision-relevant fields rather than dumping the whole state.
def summarize_node(node: str, state_before: dict, update: dict) -> dict[str, str]:
    """Build a compact {label: value} summary for a node's trace record."""
    s = {**state_before, **update}
    if node == "orchestrator":
        return {"intent": _fmt(s.get("query_intent")), "symbols": _fmt(s.get("target_symbols")),
                "horizon": _fmt(s.get("target_horizon_days")), "route": _fmt(s.get("route_reason"))}
    if node == "market_agent":
        vm = s.get("volatility_metrics", {})
        return {"vol_20d": f"{vm.get('vol_20d', 0):.1f}%", "max_dd": f"{vm.get('max_drawdown_pct', 0):.1f}%",
                "trend": f"{vm.get('trend_pct', 0):+.1f}%"}
    if node == "news_agent":
        sm = s.get("sentiment_metrics", {})
        return {"coverage": _fmt(sm.get("coverage")), "staleness": f"{sm.get('staleness_frac', 0):.0%}",
                "sentiment": f"{sm.get('sentiment_mean', 0):+.3f}"}
    if node == "predict_agent":
        me = s.get("model_evidence", {})
        return {"gate_pred": _fmt(s.get("gate_pred")), "seed_mean": _fmt(s.get("final_pred")),
                "seeds": _fmt(me.get("seed_preds")), "source": _fmt(me.get("source"))}
    if node == "gate_agent":
        return {"tau": _fmt(s.get("gate_tau")), "coverage": f"{s.get('gate_coverage', 0):.2f}",
                "action": _fmt(s.get("gated_action")), "size": _fmt(s.get("position_scale")),
                "reason": _fmt(s.get("gate_reason"))}
    if node == "risk_agent":
        return {"action": _fmt(s.get("action")), "vetoed": _fmt(s.get("risk_vetoed")),
                "veto_reasons": _fmt(s.get("veto_reasons")), "size": _fmt(s.get("position_scale"))}
    if node == "metalabel_agent":
        return {"action": _fmt(s.get("action")), "flags": _fmt(s.get("metalabel_flags")),
                "vetoed": _fmt(s.get("metalabel_vetoed")), "size": _fmt(s.get("position_scale"))}
    if node == "narrator":
        txt = s.get("answer_text") or ""
        return {"chars": _fmt(len(txt))}
    if node == "critic_agent":
        return {"status": _fmt(s.get("critic_status")), "findings": _fmt(s.get("critic_findings"))}
    if node == "rank_agent":
        return {"longs": _fmt(s.get("rank_longs")), "shorts": _fmt(s.get("rank_shorts")),
                "abstained": _fmt(s.get("rank_abstained"))}
    if node == "research_agent":
        return {"docs": _fmt(s.get("retrieved_docs"))}
    return {}


def make_trace_record(node: str, elapsed: float, state_before: dict, update: dict) -> dict[str, Any]:
    return {
        "node": node,
        "elapsed_s": round(float(elapsed), 4),
        "summary": summarize_node(node, state_before, update),
    }


def render_step(rec: dict[str, Any], step: int, total: int) -> str:
    """Render one trace record as a console/markdown block (step numbered here,
    since parallel nodes have no strict pre-execution order)."""
    head = f"── STEP {step}/{total} · {rec['node']} " + "─" * 20 + f" {rec['elapsed_s']:.3f}s"
    lines = [head]
    for k, v in rec["summary"].items():
        lines.append(f"  {k:<12}: {v}")
    return "\n".join(lines)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_manifest(config, *, eval_mode: bool, seed: int | None, extra: dict | None = None) -> dict[str, Any]:
    """Reproducibility header (plan §9.3): same manifest ⇒ identical decisions."""
    manifest = {
        "git_sha": git_sha(),
        "eval_mode": bool(eval_mode),
        "seed": seed,
        "ensemble_seeds": list(config.ensemble_seeds),
        "cmtf_version": config.cmtf_version,
        "backbone_version": config.backbone_version,
        "gate_coverage": config.gate_coverage,
        "gate_on_raw_seed": config.gate_on_raw_seed,
        "news_scope_default": config.news_scope_default,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_trace_file(path: str | Path, manifest: dict[str, Any], records: list[dict[str, Any]],
                     final_answer: str | None = None) -> None:
    """Write a human-readable Markdown transcript of a run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Multi-agent run trace", "", "## Run manifest", "", "```json"]
    import json
    lines.append(json.dumps(manifest, indent=2, ensure_ascii=False))
    lines += ["```", "", "## Workflow", ""]
    total = len(records)
    for i, rec in enumerate(records, 1):
        lines.append("```text")
        lines.append(render_step(rec, i, total))
        lines.append("```")
        lines.append("")
    if final_answer:
        lines += ["## Final answer", "", final_answer, ""]
    p.write_text("\n".join(lines), encoding="utf-8")
