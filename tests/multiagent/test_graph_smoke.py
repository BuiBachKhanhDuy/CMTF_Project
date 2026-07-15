"""Decision-path smoke test over the frozen-prediction backend.

The full graph's data agents (orchestrator/market/news) fetch live OHLCV+news, so a
full end-to-end smoke test would be slow and network-dependent. This test exercises
the deterministic DECISION path — predict_agent → gate_agent → risk_agent — over the
real frozen-prediction store and frozen GatePolicy, which is where the redesign's
correctness lives (runtime == research). It requires the cached predictions and a
calibrated VN_5d.json (both produced by the registry + `calibrate`).
"""

import numpy as np
import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.frozen_predictions import get_store, PredictionNotCachedError
from src.multiagent.agents.predict_agent import predict_agent_node
from src.multiagent.agents.gate_agent import gate_agent_node
from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.gate_io import policy_path

CFG = MultiAgentConfig()
HORIZON = 5
SYMBOL = "VCB"

pytestmark = pytest.mark.skipif(
    not policy_path(CFG.gate_policy_dir, HORIZON, "VN").exists(),
    reason="GatePolicy VN_5d.json not calibrated — run `python -m src.multiagent calibrate --horizon 5`",
)


def _a_cached_date(symbol=SYMBOL):
    store = get_store(HORIZON, CFG)
    dates = sorted(d for (s, d) in store._index if s == symbol)
    return str(dates[len(dates) // 2])


def _run_decision_path(symbol, date):
    state = {
        "symbol": symbol,
        "target_horizon_days": HORIZON,
        "prediction_time": date,
        "artifact_versions": {},
        "node_timings": {},
        # calm market so the veto passes the gate decision through
        "volatility_metrics": {"vol_20d": 20.0, "max_drawdown_pct": 5.0, "trend_pct": 1.0},
    }
    def _apply(update):
        # Mirror LangGraph's merge reducer for node_timings (the manual driver would
        # otherwise overwrite it each node instead of accumulating).
        timings = {**state.get("node_timings", {}), **update.pop("node_timings", {})}
        state.update(update)
        state["node_timings"] = timings

    _apply(predict_agent_node(state, CFG))
    _apply(gate_agent_node(state, CFG))
    _apply(risk_agent_node(state, CFG))
    return state


class TestDecisionPath:
    def test_end_to_end_produces_valid_action(self):
        s = _run_decision_path(SYMBOL, _a_cached_date())
        assert s["action"] in ("long", "short", "abstain")
        assert "gate_pred" in s and np.isfinite(s["gate_pred"])
        assert s["gate_coverage"] == pytest.approx(0.25, abs=1e-9)

    def test_gate_reason_and_tau_present(self):
        s = _run_decision_path(SYMBOL, _a_cached_date())
        assert s["gate_reason"]
        assert s["gate_tau"] > 0

    def test_abstain_below_tau(self):
        """A name whose |gate_pred| < tau must abstain with size 0."""
        store = get_store(HORIZON, CFG)
        # find any (symbol,date) below tau
        from src.multiagent.gate_io import load_gate_policy
        pol, _ = load_gate_policy(policy_path(CFG.gate_policy_dir, HORIZON, "VN"))
        found = None
        for (sym, d) in store._index:
            fp = store.get(sym, str(d))
            if abs(fp.gate_pred) < pol.tau:
                found = (sym, str(d))
                break
        assert found is not None, "expected at least one below-tau sample at 25% coverage"
        s = _run_decision_path(*found)
        assert s["gated_action"] == "abstain"
        assert s["action"] == "abstain"
        assert s["position_scale"] == 0.0

    def test_node_timings_recorded(self):
        s = _run_decision_path(SYMBOL, _a_cached_date())
        for node in ("predict_agent", "gate_agent", "risk_agent"):
            assert node in s["node_timings"]


class TestFrozenBackendHonesty:
    def test_out_of_book_date_raises(self):
        """R1: the frozen backend never invents a prediction for an unknown date."""
        store = get_store(HORIZON, CFG)
        with pytest.raises(PredictionNotCachedError):
            store.get(SYMBOL, "1990-01-02")

    def test_unknown_symbol_raises(self):
        store = get_store(HORIZON, CFG)
        with pytest.raises(PredictionNotCachedError):
            store.get("NOTASYMBOL", _a_cached_date())
