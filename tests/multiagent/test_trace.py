"""Tests for the traceability layer (plan §9, R3)."""

import json

from src.multiagent.config import MultiAgentConfig
from src.multiagent.trace import (
    make_trace_record,
    render_step,
    summarize_node,
    build_manifest,
    write_trace_file,
)
from src.multiagent.graph import _make_node_with_config


def test_summarize_gate_node():
    before = {}
    update = {"gate_tau": 0.018, "gate_coverage": 0.25, "gated_action": "long",
              "position_scale": 0.82, "gate_reason": "|pred|>=tau -> long"}
    s = summarize_node("gate_agent", before, update)
    assert s["action"] == "long"
    assert "0.25" in s["coverage"]


def test_render_step_has_header_and_fields():
    rec = make_trace_record("gate_agent", 0.01, {}, {"gated_action": "long", "gate_tau": 0.02,
                                                     "gate_coverage": 0.25, "position_scale": 0.8,
                                                     "gate_reason": "x"})
    out = render_step(rec, 3, 7)
    assert "STEP 3/7 · gate_agent" in out
    assert "action" in out


def test_build_manifest_fields():
    m = build_manifest(MultiAgentConfig(), eval_mode=True, seed=42)
    assert m["eval_mode"] is True
    assert m["ensemble_seeds"] == [1, 42, 123]
    assert m["gate_on_raw_seed"] is False
    assert "git_sha" in m


def test_write_trace_file(tmp_path):
    recs = [make_trace_record("predict_agent", 0.05, {}, {"gate_pred": 0.02, "final_pred": 0.02,
                                                          "model_evidence": {"seed_preds": [1, 2, 3], "source": "frozen"}})]
    p = tmp_path / "run.md"
    write_trace_file(p, build_manifest(MultiAgentConfig(), eval_mode=False, seed=None), recs, "MUA VCB")
    text = p.read_text(encoding="utf-8")
    assert "Run manifest" in text
    assert "predict_agent" in text
    assert "MUA VCB" in text


def test_wrapper_appends_trace_record_when_enabled():
    cfg = MultiAgentConfig(trace_enabled=True)

    def fake_node(state, config):
        return {"gated_action": "long", "gate_tau": 0.02, "gate_coverage": 0.25,
                "position_scale": 0.8, "gate_reason": "x", "node_timings": {"gate_agent": 0.01}}

    wrapped = _make_node_with_config(fake_node, cfg, "gate_agent")
    out = wrapped({})
    assert "trace" in out
    assert out["trace"][0]["node"] == "gate_agent"


def test_wrapper_no_trace_when_disabled():
    cfg = MultiAgentConfig(trace_enabled=False)

    def fake_node(state, config):
        return {"action": "abstain", "node_timings": {}}

    wrapped = _make_node_with_config(fake_node, cfg, "risk_agent")
    out = wrapped({})
    assert "trace" not in out
