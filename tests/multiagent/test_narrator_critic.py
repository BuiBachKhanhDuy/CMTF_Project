"""Tests for the narrator (honest disclosure) and critic (verify vs state)."""

from src.multiagent.config import MultiAgentConfig
from src.multiagent.agents.narrator_agent import grounded_template, narrator_agent_node
from src.multiagent.agents.critic_agent import verify_answer, critic_agent_node

CFG_EVAL = MultiAgentConfig(evaluation_mode=True)


def _abstain_state():
    return {
        "symbol": "VCB", "target_horizon_days": 5, "action": "abstain",
        "position_scale": 0.0, "risk_vetoed": False,
        "gate_reason": "|pred|=0.010 < tau=0.018 -> abstain", "gate_coverage": 0.25,
        "gate_tau": 0.018,
        "model_evidence": {"final_pred": 0.010, "gate_pred": 0.010},
        "volatility_metrics": {"vol_20d": 22.0, "max_drawdown_pct": 8.0, "trend_pct": 1.0},
        "sentiment_metrics": {"coverage": 5, "staleness_frac": 0.2},
        "node_timings": {},
    }


def _long_state():
    s = _abstain_state()
    s.update(action="long", position_scale=0.82,
             gate_reason="|pred|=0.030 >= tau=0.018 -> long @ size=+0.82",
             model_evidence={"final_pred": 0.030, "gate_pred": 0.030})
    return s


class TestNarrator:
    def test_abstain_template_is_honest(self):
        t = grounded_template(_abstain_state())
        assert "KHÔNG GIAO DỊCH" in t
        assert "TỪ CHỐI" in t  # calibrated abstention framing, not accuracy

    def test_eval_mode_empty_answer_but_template_present(self):
        out = narrator_agent_node(_abstain_state(), CFG_EVAL)
        assert out["answer_text"] == ""  # determinism
        assert out["grounded_answer"]  # deterministic reference still produced


class TestCriticVerification:
    def test_grounded_template_passes(self):
        s = _long_state()
        assert verify_answer(grounded_template(s), s) == []

    def test_ungrounded_number_flagged(self):
        s = _abstain_state()
        bad = "Khuyến nghị KHÔNG GIAO DỊCH. Mục tiêu giá 999999 đồng."
        findings = verify_answer(bad, s)
        assert any("ungrounded" in f for f in findings)

    def test_abstain_flip_to_trade_flagged(self):
        s = _abstain_state()
        bad = "Khuyến nghị MUA mạnh ngay bây giờ."
        findings = verify_answer(bad, s)
        assert any("tone" in f for f in findings)

    def test_empty_answer_verifies_trivially(self):
        assert verify_answer("", _abstain_state()) == []


class TestCriticNode:
    def test_eval_mode_status_ok(self):
        out = critic_agent_node({**_abstain_state(), "answer_text": "",
                                 "grounded_answer": grounded_template(_abstain_state())}, CFG_EVAL)
        assert out["critic_status"] == "ok"
        assert out["critic_findings"] == []

    def test_clean_answer_status_ok(self):
        s = _long_state()
        out = critic_agent_node({**s, "answer_text": grounded_template(s),
                                 "grounded_answer": grounded_template(s)}, CFG_EVAL)
        assert out["critic_status"] == "ok"
