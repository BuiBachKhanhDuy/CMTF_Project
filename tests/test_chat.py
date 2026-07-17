"""Regression tests for chat.py's horizon threading.

Both bugs fixed here used to hardcode the literal ``5`` regardless of what the
query asked for or what ``--horizon`` was configured, so `rank`/RESEARCH-intent
queries silently always evaluated at 5D even when the user asked for 1D/20D.
"""

import chat
from src.multiagent.config import MultiAgentConfig

CFG_EVAL = MultiAgentConfig(evaluation_mode=True)


class TestHandleRankHorizon:
    def test_uses_query_horizon_not_hardcoded_5(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            chat, "_run_rank",
            lambda symbols, date, horizon, cfg: captured.update(horizon=horizon),
        )
        chat._handle_rank("rank VCB,BID 2025-08-13 20 ngày", CFG_EVAL, default_horizon=1)
        assert captured["horizon"] == 20

    def test_falls_back_to_cli_default_not_5(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            chat, "_run_rank",
            lambda symbols, date, horizon, cfg: captured.update(horizon=horizon),
        )
        # No horizon word anywhere in the query — must fall back to the CLI's
        # configured default (1 here), never the old hardcoded literal 5.
        chat._handle_rank("rank VCB,BID 2025-08-13", CFG_EVAL, default_horizon=1)
        assert captured["horizon"] == 1

    def test_default_horizon_20_is_respected_too(self, monkeypatch):
        # Symmetric check: the old bug always forced 5 regardless of the default.
        captured = {}
        monkeypatch.setattr(
            chat, "_run_rank",
            lambda symbols, date, horizon, cfg: captured.update(horizon=horizon),
        )
        chat._handle_rank("rank VCB,BID 2025-08-13", CFG_EVAL, default_horizon=20)
        assert captured["horizon"] == 20


class TestRunResearchGapForecastHorizon:
    def test_gap_fill_forecast_uses_configured_horizon(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(
            chat, "compute_range_stats",
            lambda frame, symbol, date_start, date_end: {
                "coverage": "partial", "needs_prediction_from": "2025-08-01",
                "n_days": 3, "covered_start": "2025-07-29", "covered_end": "2025-07-31",
                "return_pct": 0.5, "volatility_pct": 12.0, "max_drawdown_pct": 2.0,
            },
        )
        monkeypatch.setattr(chat, "articles_in_range", lambda *a, **k: [])
        monkeypatch.setattr(chat, "_llm_reachable", lambda cfg: False)
        monkeypatch.setattr(
            chat, "_run_prediction_for_gap",
            lambda symbol, from_date, horizon, cfg, frame, news_idx: captured.update(horizon=horizon)
            or {"action": "abstain", "prediction_time": from_date, "gate_reason": ""},
        )

        chat._run_research("VCB", "2025-07-25", "2025-08-10", 20, CFG_EVAL, frame=None, news_idx=None)
        assert captured["horizon"] == 20

    def test_gap_fill_forecast_does_not_default_to_5(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(
            chat, "compute_range_stats",
            lambda frame, symbol, date_start, date_end: {
                "coverage": "none", "n_days": 0,
            },
        )
        monkeypatch.setattr(chat, "articles_in_range", lambda *a, **k: [])
        monkeypatch.setattr(chat, "_llm_reachable", lambda cfg: False)
        monkeypatch.setattr(
            chat, "_run_prediction_for_gap",
            lambda symbol, from_date, horizon, cfg, frame, news_idx: captured.update(horizon=horizon)
            or {"action": "abstain", "prediction_time": from_date, "gate_reason": ""},
        )

        chat._run_research("VCB", "2025-08-01", "2025-08-10", 1, CFG_EVAL, frame=None, news_idx=None)
        assert captured["horizon"] == 1
        assert captured["horizon"] != 5


def _fake_decision_chain(coverage_by_lookback):
    """A stand-in DECISION_CHAIN with no real model/data calls, so the
    reasoning-agent widen-and-rerun wiring in `_run_decision` can be tested without
    a real (slow) forward pass. `coverage_by_lookback` maps a lookback_days value to
    the "coverage" `_gather_evidence` should report for that call."""
    from src.multiagent.agents.reasoning_agent import reasoning_agent_node

    def predict_agent(state, cfg, **kw):
        return {"gate_pred": 0.01}

    def gate_agent(state, cfg, **kw):
        return {"gated_action": "long", "position_scale": 0.5, "gate_tau": 0.005, "gate_reason": "ok"}

    def horizon_interaction_agent(state, cfg, **kw):
        return {}

    def risk_agent(state, cfg, **kw):
        return {"action": state.get("gated_action", "abstain"), "position_scale": state.get("position_scale", 0.0)}

    def metalabel_agent(state, cfg, **kw):
        return {}

    def narrator(state, cfg, **kw):
        return {"answer_text": "", "grounded_answer": "template"}

    def critic_agent(state, cfg, **kw):
        return {"critic_status": "ok", "critic_findings": []}

    return [
        ("predict_agent", predict_agent), ("gate_agent", gate_agent),
        ("horizon_interaction_agent", horizon_interaction_agent), ("risk_agent", risk_agent),
        ("metalabel_agent", metalabel_agent), ("narrator", narrator),
        ("critic_agent", critic_agent), ("reasoning_agent", reasoning_agent_node),
    ]


class TestReasoningAgentWidenAndRerun:
    """Tests `_run_decision`'s wiring of reasoning_agent's widen_and_rerun callback
    with a stubbed decision chain (no real model/data calls — the reasoning_agent
    trigger logic itself is unit-tested separately in test_reasoning_agent.py)."""

    def test_thin_news_triggers_widen_and_rerun(self, monkeypatch):
        fake_chain = _fake_decision_chain({})
        monkeypatch.setattr(chat, "DECISION_CHAIN", fake_chain)
        monkeypatch.setattr(chat, "_RERUN_CHAIN", fake_chain[:7])

        call_lookbacks = []

        def fake_gather(frame, news_idx, symbol, date, lookback_days=5, vol_window=20):
            call_lookbacks.append(lookback_days)
            # Thin (0) on the initial call; the widened call reports real coverage.
            coverage = 0 if lookback_days == 5 else 5
            return {
                "warnings": [],
                "volatility_metrics": {"vol_20d": 10.0, "max_drawdown_pct": 2.0, "trend_pct": 1.0},
                "sentiment_metrics": {"coverage": coverage, "staleness_frac": 0.0, "sentiment_mean": 0.0},
                "articles": [],
            }
        monkeypatch.setattr(chat, "_gather_evidence", fake_gather)

        state, _steps = chat._run_decision("VCB", "2025-01-01", 5, CFG_EVAL, None, None)

        assert call_lookbacks == [5, CFG_EVAL.reasoning_widen_lookback_days_to]  # exactly one widen, not iterative
        assert state["reasoning_evidence_widened"] is True
        assert "thin_news_coverage" in state["reasoning_triggered_reasons"]

    def test_sufficient_evidence_never_invokes_widen(self, monkeypatch):
        fake_chain = _fake_decision_chain({})
        monkeypatch.setattr(chat, "DECISION_CHAIN", fake_chain)
        monkeypatch.setattr(chat, "_RERUN_CHAIN", fake_chain[:7])

        call_lookbacks = []

        def fake_gather(frame, news_idx, symbol, date, lookback_days=5, vol_window=20):
            call_lookbacks.append(lookback_days)
            return {
                "warnings": [],
                "volatility_metrics": {"vol_20d": 10.0, "max_drawdown_pct": 2.0, "trend_pct": 1.0},
                "sentiment_metrics": {"coverage": 10, "staleness_frac": 0.0, "sentiment_mean": 0.0},
                "articles": [],
            }
        monkeypatch.setattr(chat, "_gather_evidence", fake_gather)

        state, _steps = chat._run_decision("VCB", "2025-01-01", 5, CFG_EVAL, None, None)

        assert call_lookbacks == [5]  # widen path never invoked — no wasted recomputation
        assert state["reasoning_evidence_widened"] is False
        assert state["reasoning_triggered_reasons"] == []
