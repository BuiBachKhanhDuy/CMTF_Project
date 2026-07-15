"""Tests for the Risk Agent — one-way safety veto (plan §3.7).

The invariant under test: risk can ONLY downgrade a trade to abstain. It can never
turn an abstain into a trade, never up-size, never flip long↔short.
"""

from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.config import MultiAgentConfig

CFG = MultiAgentConfig()


def _state(gated_action, position_scale, vol=20.0, dd=5.0):
    return {
        "gated_action": gated_action,
        "position_scale": position_scale,
        "volatility_metrics": {"vol_20d": vol, "max_drawdown_pct": dd, "trend_pct": 2.0},
        "node_timings": {},
    }


class TestVetoPassThrough:
    def test_calm_market_passes_long(self):
        r = risk_agent_node(_state("long", 0.8), CFG)
        assert r["action"] == "long"
        assert r["position_scale"] == 0.8
        assert r["risk_vetoed"] is False

    def test_calm_market_passes_short(self):
        r = risk_agent_node(_state("short", -0.6), CFG)
        assert r["action"] == "short"
        assert r["position_scale"] == -0.6
        assert r["risk_vetoed"] is False


class TestVetoDowngrade:
    def test_high_vol_vetoes_trade(self):
        r = risk_agent_node(_state("long", 0.8, vol=45.0), CFG)
        assert r["action"] == "abstain"
        assert r["position_scale"] == 0.0
        assert r["risk_vetoed"] is True
        assert any("vol" in reason for reason in r["veto_reasons"])

    def test_high_drawdown_vetoes_trade(self):
        r = risk_agent_node(_state("short", -0.7, dd=25.0), CFG)
        assert r["action"] == "abstain"
        assert r["position_scale"] == 0.0
        assert r["risk_vetoed"] is True
        assert any("dd" in reason for reason in r["veto_reasons"])


class TestOneWayInvariant:
    def test_abstain_in_abstain_out_even_in_danger(self):
        """Abstain-in ⇒ abstain-out: risk never manufactures a trade."""
        r = risk_agent_node(_state("abstain", 0.0, vol=99.0, dd=99.0), CFG)
        assert r["action"] == "abstain"
        assert r["position_scale"] == 0.0
        # An abstain is not a trade, so there is nothing to veto.
        assert r["risk_vetoed"] is False

    def test_abstain_never_becomes_trade_in_calm_market(self):
        r = risk_agent_node(_state("abstain", 0.0), CFG)
        assert r["action"] == "abstain"
        assert r["position_scale"] == 0.0

    def test_veto_never_upsizes(self):
        """A passed-through trade keeps its exact gate size (no re-sizing)."""
        r = risk_agent_node(_state("long", 0.42), CFG)
        assert r["position_scale"] == 0.42


class TestNodeContract:
    def test_node_timings(self):
        r = risk_agent_node(_state("long", 0.8), CFG)
        assert "risk_agent" in r["node_timings"]

    def test_reasoning_present(self):
        r = risk_agent_node(_state("long", 0.8), CFG)
        assert r["decision_reasoning"]
