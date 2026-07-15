"""No-fallback / no-hiding contract tests (plan §8, R1).

These encode the honesty contract as executable checks:
1. Missing GatePolicy artifact raises (no ad-hoc tau).
2. Stale GatePolicy (version mismatch) raises (no stale fallback).
3. Frozen backend never invents an out-of-book prediction.
4. Risk veto is one-way: abstain-in ⇒ abstain-out.
5. Any LLM call in eval mode is a hard error.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.gate_io import (
    load_gate_policy,
    save_gate_policy,
    policy_path,
    StalePolicyError,
    GATE_ARTIFACT_SCHEMA_VERSION,
)
from src.multiagent.loaders import ArtifactMissingError
from src.multiagent.guards import assert_llm_allowed, EvalModeLLMError
from src.benchmark.decision_policy import GatePolicy


# --- 1. Missing artifact -----------------------------------------------------
def test_missing_gate_policy_raises(tmp_path):
    with pytest.raises(ArtifactMissingError):
        load_gate_policy(policy_path(tmp_path, 5, "VN"))


def test_gate_agent_missing_policy_raises(tmp_path):
    from src.multiagent.agents.gate_agent import gate_agent_node
    cfg = MultiAgentConfig(gate_policy_dir=tmp_path)  # empty dir → no artifact
    state = {"gate_pred": 0.05, "target_horizon_days": 5, "artifact_versions": {}}
    with pytest.raises(ArtifactMissingError):
        gate_agent_node(state, cfg)


# --- 2. Stale artifact -------------------------------------------------------
def _write_policy(tmp_path, **meta_overrides):
    pol = GatePolicy(tau=0.02, conviction=True, conviction_scale=0.02, coverage=0.25, val_score=1.0)
    meta = {"symbol": "VN", "horizon": 5, "cmtf_version": "v4", "backbone_version": "v3"}
    meta.update(meta_overrides)
    p = policy_path(tmp_path, 5, "VN")
    save_gate_policy(pol, meta, p)
    return p


def test_stale_cmtf_version_raises(tmp_path):
    p = _write_policy(tmp_path, cmtf_version="v4")
    with pytest.raises(StalePolicyError):
        load_gate_policy(p, expect_cmtf_version="v99")


def test_stale_schema_version_raises(tmp_path):
    p = _write_policy(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["schema_version"] = GATE_ARTIFACT_SCHEMA_VERSION + 99
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StalePolicyError):
        load_gate_policy(p)


def test_valid_policy_loads(tmp_path):
    p = _write_policy(tmp_path)
    pol, meta = load_gate_policy(p, expect_cmtf_version="v4", expect_backbone_version="v3")
    assert pol.tau == 0.02
    assert meta["symbol"] == "VN"


# --- 3. Frozen backend honesty (covered in smoke; contract-level here) -------
def test_frozen_store_out_of_book_raises():
    store_mod = pytest.importorskip("src.multiagent.frozen_predictions")
    from src.multiagent.frozen_predictions import get_store, PredictionNotCachedError
    try:
        store = get_store(5)
    except ArtifactMissingError:
        pytest.skip("frozen predictions not cached in this environment")
    with pytest.raises(PredictionNotCachedError):
        store.get("VCB", "1980-01-01")


# --- 4. Risk veto one-way ----------------------------------------------------
def test_abstain_in_abstain_out():
    from src.multiagent.agents.risk_agent import risk_agent_node
    cfg = MultiAgentConfig()
    state = {
        "gated_action": "abstain", "position_scale": 0.0,
        "volatility_metrics": {"vol_20d": 99.0, "max_drawdown_pct": 99.0},
        "node_timings": {},
    }
    r = risk_agent_node(state, cfg)
    assert r["action"] == "abstain"
    assert r["position_scale"] == 0.0


# --- 5. Eval-mode LLM hard error ---------------------------------------------
def test_assert_llm_allowed_raises_in_eval_mode():
    with pytest.raises(EvalModeLLMError):
        assert_llm_allowed(MultiAgentConfig(evaluation_mode=True), "test")


def test_assert_llm_allowed_passes_in_normal_mode():
    assert_llm_allowed(MultiAgentConfig(evaluation_mode=False), "test")  # no raise


def test_orchestrator_eval_mode_no_llm():
    """Eval mode must take the deterministic branch, never the LLM."""
    from src.multiagent.agents.orchestrator_agent import orchestrator_node
    cfg = MultiAgentConfig(evaluation_mode=True)
    state = {"query_text": "Nên mua VCB trong 5 ngày tới không?", "node_timings": {}}
    r = orchestrator_node(state, cfg)  # must not raise EvalModeLLMError
    assert r["symbol"] == "VCB"
    assert r["target_horizon_days"] == 5
