"""Tests for reasoning_agent.py — single-pass reflection, never a loop, never a veto."""

from src.multiagent.config import MultiAgentConfig
from src.multiagent.agents.reasoning_agent import reasoning_agent_node

CFG_EVAL = MultiAgentConfig(evaluation_mode=True)


def _sufficient_state():
    return {
        "action": "long", "position_scale": 0.8, "critic_status": "ok",
        "horizon_agreement_score": 2,
        "sentiment_metrics": {"coverage": 10},
    }


class TestTriggerConditions:
    def test_no_trigger_when_evidence_sufficient(self):
        out = reasoning_agent_node(_sufficient_state(), CFG_EVAL)
        assert out["reasoning_triggered_reasons"] == []
        assert out["reasoning_notes"] is None
        assert out["reasoning_evidence_widened"] is False

    def test_critic_failed_triggers_independently(self):
        s = {**_sufficient_state(), "critic_status": "failed"}
        out = reasoning_agent_node(s, CFG_EVAL)
        assert "critic_verification_failed" in out["reasoning_triggered_reasons"]

    def test_horizon_disagreement_triggers_only_on_a_trade(self):
        s = {**_sufficient_state(), "horizon_agreement_score": 0}
        out = reasoning_agent_node(s, CFG_EVAL)
        assert "cross_horizon_disagreement" in out["reasoning_triggered_reasons"]

        abstain_s = {**s, "action": "abstain"}
        out2 = reasoning_agent_node(abstain_s, CFG_EVAL)
        assert "cross_horizon_disagreement" not in out2["reasoning_triggered_reasons"]

    def test_thin_news_coverage_triggers_only_on_a_trade(self):
        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 1}}
        out = reasoning_agent_node(s, CFG_EVAL)
        assert "thin_news_coverage" in out["reasoning_triggered_reasons"]

        abstain_s = {**s, "action": "abstain"}
        out2 = reasoning_agent_node(abstain_s, CFG_EVAL)
        assert "thin_news_coverage" not in out2["reasoning_triggered_reasons"]

    def test_multiple_reasons_can_fire_together(self):
        s = {**_sufficient_state(), "horizon_agreement_score": 0,
             "sentiment_metrics": {"coverage": 0}, "critic_status": "failed"}
        out = reasoning_agent_node(s, CFG_EVAL)
        assert set(out["reasoning_triggered_reasons"]) == {
            "critic_verification_failed", "cross_horizon_disagreement", "thin_news_coverage",
        }


class TestNoCallbackNeverChangesDecision:
    def test_triggered_without_callback_only_adds_caveat(self):
        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 0}}
        out = reasoning_agent_node(s, CFG_EVAL, widen_and_rerun=None)
        assert out["reasoning_triggered_reasons"] == ["thin_news_coverage"]
        assert out["reasoning_evidence_widened"] is False
        assert out["reasoning_notes"]  # non-empty caveat text
        # Never sets action/position_scale itself.
        assert "action" not in out
        assert "position_scale" not in out


class TestNotesAppendedPostCritic:
    """narrator/critic run BEFORE this node now, so `reasoning_notes` can only ever
    reach the user by this node appending them directly onto the already-verified
    answer_text/grounded_answer — never by narrator citing the field itself."""

    def test_appends_onto_answer_text_already_in_state(self):
        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 0},
             "answer_text": "Khuyến nghị: MUA.", "grounded_answer": "Khuyến nghị: MUA."}
        out = reasoning_agent_node(s, CFG_EVAL, widen_and_rerun=None)
        assert out["answer_text"].startswith("Khuyến nghị: MUA.")
        assert out["reasoning_notes"] in out["answer_text"]
        assert out["grounded_answer"].startswith("Khuyến nghị: MUA.")
        assert out["reasoning_notes"] in out["grounded_answer"]

    def test_appends_onto_the_rerun_pass_answer_not_the_original(self):
        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 0},
             "answer_text": "stale original", "grounded_answer": "stale original"}

        def widen_and_rerun(state):
            return {"action": "abstain", "position_scale": 0.0,
                    "answer_text": "fresh rerun answer", "grounded_answer": "fresh rerun answer"}

        out = reasoning_agent_node(s, CFG_EVAL, widen_and_rerun=widen_and_rerun)
        assert out["answer_text"].startswith("fresh rerun answer")
        assert "stale original" not in out["answer_text"]

    def test_no_trigger_leaves_answer_text_untouched(self):
        s = {**_sufficient_state(), "answer_text": "Khuyến nghị: MUA.", "grounded_answer": "Khuyến nghị: MUA."}
        out = reasoning_agent_node(s, CFG_EVAL)
        assert "answer_text" not in out  # no notes to append -> node doesn't touch it


class TestWithCallback:
    def test_callback_invoked_at_most_once_and_adopted(self):
        calls = []

        def widen_and_rerun(state):
            calls.append(state)
            return {"action": "abstain", "position_scale": 0.0, "gate_reason": "re-run says no"}

        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 0}}
        out = reasoning_agent_node(s, CFG_EVAL, widen_and_rerun=widen_and_rerun)

        assert len(calls) == 1
        assert out["reasoning_evidence_widened"] is True
        assert out["action"] == "abstain"
        assert out["position_scale"] == 0.0
        assert out["gate_reason"] == "re-run says no"

    def test_callback_not_invoked_when_no_trigger(self):
        calls = []

        def widen_and_rerun(state):
            calls.append(state)
            return {"action": "abstain", "position_scale": 0.0}

        out = reasoning_agent_node(_sufficient_state(), CFG_EVAL, widen_and_rerun=widen_and_rerun)
        assert calls == []
        assert out["reasoning_evidence_widened"] is False

    def test_callback_returning_none_falls_back_to_caveat(self):
        s = {**_sufficient_state(), "sentiment_metrics": {"coverage": 0}}
        out = reasoning_agent_node(s, CFG_EVAL, widen_and_rerun=lambda state: None)
        assert out["reasoning_evidence_widened"] is False
        assert out["reasoning_notes"]
        assert "action" not in out
