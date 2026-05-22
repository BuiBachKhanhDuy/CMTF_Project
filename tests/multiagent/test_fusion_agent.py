"""Tests for Fusion Agent — confidence modulation logic.

Verifies:
- fused_score == final_pred always (CMTF is source of truth)
- Market agreement boosts confidence, disagreement reduces it
- News trust weight modulates news bonus/penalty
- Edge cases: zero signals, missing proposals
"""

import pytest

from src.multiagent.agents.fusion_agent import fusion_agent_node, _signs_agree
from src.multiagent.config import MultiAgentConfig


_CFG = MultiAgentConfig(
    evaluation_mode=True,
    market_agree_bonus=0.15,
    market_disagree_penalty=0.10,
    news_agree_bonus=0.10,
    news_disagree_penalty=0.05,
)


def _base_state(**overrides):
    state = {
        "final_pred": 0.025,
        "predict_confidence": 0.6,
        "news_residual": 0.005,
        "market_proposal": {
            "direction": "long",
            "score": 0.03,
            "confidence": 0.8,
        },
        "news_proposal": {
            "direction": "long",
            "score": 0.3,       # raw sentiment — should NOT affect fused_score
            "confidence": 0.7,  # = trust_weight
        },
        "node_timings": {},
    }
    state.update(overrides)
    return state


class TestSignsAgree:
    def test_both_positive(self):
        assert _signs_agree(0.5, 0.1) is True

    def test_both_negative(self):
        assert _signs_agree(-0.5, -0.1) is True

    def test_disagree(self):
        assert _signs_agree(0.5, -0.1) is False

    def test_one_zero_agrees(self):
        assert _signs_agree(0.0, 0.5) is True

    def test_both_zero_agrees(self):
        assert _signs_agree(0.0, 0.0) is True


class TestFusionScore:
    """fused_score must always equal final_pred — fusion never overrides CMTF."""

    def test_score_equals_final_pred(self):
        result = fusion_agent_node(_base_state(), _CFG)
        fd = result["fusion_decision"]
        assert fd["score"] == pytest.approx(0.025, abs=1e-6)

    def test_score_unaffected_by_market_disagreement(self):
        state = _base_state(market_proposal={"score": -0.04, "confidence": 1.0})
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["score"] == pytest.approx(0.025, abs=1e-6)

    def test_score_unaffected_by_news_sentiment(self):
        """Raw news sentiment (0.9) must NOT leak into fused_score."""
        state = _base_state(news_proposal={"score": 0.9, "confidence": 1.0})
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["score"] == pytest.approx(0.025, abs=1e-6)

    def test_negative_pred_preserved(self):
        state = _base_state(final_pred=-0.03)
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["score"] == pytest.approx(-0.03, abs=1e-6)
        assert result["fusion_decision"]["direction"] == "short"


class TestConfidenceModulation:
    def test_all_agree_boosts_confidence(self):
        """CMTF long + market long + news_residual positive → confidence boosted."""
        result = fusion_agent_node(_base_state(), _CFG)
        fd = result["fusion_decision"]
        # base=0.6 + market_bonus=0.8*0.15=0.12 + news_bonus=0.7*0.10=0.07
        expected = 0.6 + 0.12 + 0.07
        assert fd["confidence"] == pytest.approx(expected, abs=0.01)
        assert fd["market_agrees"] is True
        assert fd["news_agrees"] is True

    def test_market_disagrees_reduces_confidence(self):
        """CMTF long + market short → confidence reduced."""
        state = _base_state(
            market_proposal={"score": -0.04, "confidence": 0.9},
        )
        result = fusion_agent_node(state, _CFG)
        fd = result["fusion_decision"]
        # base=0.6 - market_penalty=0.9*0.10=0.09 + news_bonus=0.7*0.10=0.07
        expected = 0.6 - 0.09 + 0.07
        assert fd["confidence"] == pytest.approx(expected, abs=0.01)
        assert fd["market_agrees"] is False

    def test_news_disagrees_reduces_confidence(self):
        """news_residual negative while CMTF positive → news penalty."""
        state = _base_state(news_residual=-0.003)
        result = fusion_agent_node(state, _CFG)
        fd = result["fusion_decision"]
        # base=0.6 + market_bonus=0.8*0.15=0.12 - news_penalty=0.7*0.05=0.035
        expected = 0.6 + 0.12 - 0.035
        assert fd["confidence"] == pytest.approx(expected, abs=0.01)
        assert fd["news_agrees"] is False

    def test_all_disagree_minimum_confidence(self):
        """Both market and news disagree → max penalty, but floor at 0."""
        state = _base_state(
            predict_confidence=0.1,
            market_proposal={"score": -0.05, "confidence": 1.0},
            news_residual=-0.01,
            news_proposal={"score": -0.5, "confidence": 1.0},
        )
        result = fusion_agent_node(state, _CFG)
        fd = result["fusion_decision"]
        # base=0.1 - 1.0*0.10 - 1.0*0.05 = 0.1 - 0.10 - 0.05 = -0.05 → clipped to 0
        assert fd["confidence"] == 0.0

    def test_confidence_capped_at_one(self):
        """High base + all agree → capped at 1.0."""
        state = _base_state(
            predict_confidence=0.9,
            market_proposal={"score": 0.05, "confidence": 1.0},
            news_proposal={"score": 0.5, "confidence": 1.0},
        )
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["confidence"] == 1.0


class TestEdgeCases:
    def test_zero_news_residual_neutral(self):
        """Zero news_residual → news agrees (neutral), gets bonus."""
        state = _base_state(news_residual=0.0)
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["news_agrees"] is True

    def test_flat_market_neutral(self):
        """Market score ~0 → agrees (neutral), gets bonus."""
        state = _base_state(market_proposal={"score": 0.0, "confidence": 0.5})
        result = fusion_agent_node(state, _CFG)
        assert result["fusion_decision"]["market_agrees"] is True

    def test_missing_proposals_graceful(self):
        """No market/news proposals → base confidence only."""
        state = {
            "final_pred": 0.02,
            "predict_confidence": 0.5,
            "news_residual": 0.0,
            "node_timings": {},
        }
        result = fusion_agent_node(state, _CFG)
        fd = result["fusion_decision"]
        assert fd["score"] == pytest.approx(0.02, abs=1e-6)
        assert fd["confidence"] == pytest.approx(0.5, abs=0.01)

    def test_decision_trace_keys(self):
        """Verify all expected keys in fusion_decision."""
        result = fusion_agent_node(_base_state(), _CFG)
        fd = result["fusion_decision"]
        required = {"direction", "score", "confidence", "base_confidence",
                     "market_agrees", "market_bonus", "news_agrees",
                     "news_trust", "news_bonus", "rationale"}
        assert required.issubset(fd.keys())
