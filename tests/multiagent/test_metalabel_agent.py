"""Tests for the Metalabel Agent — one-way qualitative event-flag veto.

Invariant under test (same as risk_agent): can only downgrade a trade to abstain.
Never turns an abstain into a trade, never up-sizes, never flips long/short.
"""

from src.multiagent.agents.metalabel_agent import (
    metalabel_agent_node,
    _parse_flags,
    EVENT_CATEGORIES,
)
from src.multiagent.config import MultiAgentConfig

CFG_EVAL = MultiAgentConfig(evaluation_mode=True)


def _state(action, position_scale, articles=None):
    return {
        "action": action, "position_scale": position_scale,
        "symbol": "VCB", "articles": articles or [],
        "veto_reasons": [], "decision_reasoning": "", "node_timings": {},
    }


class TestParseFlags:
    def test_parses_valid_category(self):
        text = '{"flags": ["earnings_or_guidance"], "reason": "profit warning"}'
        assert _parse_flags(text) == ["earnings_or_guidance"]

    def test_rejects_unknown_category(self):
        text = '{"flags": ["made_up_category"], "reason": "x"}'
        assert _parse_flags(text) == []

    def test_empty_flags(self):
        assert _parse_flags('{"flags": [], "reason": "nothing relevant"}') == []

    def test_malformed_json_returns_empty(self):
        assert _parse_flags("not json at all") == []

    def test_multiple_valid_categories(self):
        text = '{"flags": ["earnings_or_guidance", "leadership_or_scandal"]}'
        assert set(_parse_flags(text)) == {"earnings_or_guidance", "leadership_or_scandal"}


class TestEvalModeNoLLM:
    """Eval mode must never call the LLM (byte-reproducible decision path)."""

    def test_eval_mode_no_flags_no_veto(self):
        out = metalabel_agent_node(_state("long", 0.8, articles=[{"title": "x"}]), CFG_EVAL)
        assert out["metalabel_flags"] == []
        assert out["metalabel_vetoed"] is False
        assert out["action"] == "long"
        assert out["position_scale"] == 0.8


class TestOneWayInvariant:
    def test_abstain_in_abstain_out(self):
        """Abstain-in must never become a trade, regardless of flags."""
        out = metalabel_agent_node(_state("abstain", 0.0), CFG_EVAL)
        assert out["action"] == "abstain"
        assert out["position_scale"] == 0.0
        assert out["metalabel_vetoed"] is False  # nothing to veto — not a trade

    def test_no_flags_passes_trade_through_unchanged(self):
        out = metalabel_agent_node(_state("short", -0.6), CFG_EVAL)
        assert out["action"] == "short"
        assert out["position_scale"] == -0.6
        assert out["metalabel_vetoed"] is False

    def test_veto_never_upsizes(self):
        out = metalabel_agent_node(_state("long", 0.42), CFG_EVAL)
        assert out["position_scale"] in (0.42, 0.0)  # either passthrough or full veto, never larger


class TestNodeContract:
    def test_node_timings_recorded(self):
        out = metalabel_agent_node(_state("long", 0.8), CFG_EVAL)
        assert "metalabel_agent" in out["node_timings"]

    def test_veto_reasons_preserved_and_extendable(self):
        state = _state("long", 0.8)
        state["veto_reasons"] = ["risk:vol=45%>40%"]
        out = metalabel_agent_node(state, CFG_EVAL)
        # eval mode: no new flags, so risk_agent's existing reason must survive unchanged
        assert out["veto_reasons"] == ["risk:vol=45%>40%"]

    def test_categories_are_a_small_fixed_pre_registered_set(self):
        # Guards against silent category creep after seeing results (R1).
        assert set(EVENT_CATEGORIES) == {
            "earnings_or_guidance", "ma_ownership_change",
            "regulatory_or_policy_action", "leadership_or_scandal",
            "capital_or_dividend_action",
        }
