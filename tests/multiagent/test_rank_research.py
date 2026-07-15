"""Tests for the rank (cross-sectional) and research (RAG) branches."""

import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.agents.research_agent import research_agent_node, _rank_docs
from src.multiagent.loaders import ArtifactMissingError

CFG_EVAL = MultiAgentConfig(evaluation_mode=True)


class TestRankAgent:
    def test_rank_buckets_are_disjoint_and_complete(self):
        from src.multiagent.agents.rank_agent import rank_agent_node
        try:
            state = {
                "target_symbols": ["VCB", "CTG", "BID", "TCB", "MBB", "ACB", "VPB"],
                "target_horizon_days": 5, "prediction_time": "2025-06-03", "node_timings": {},
            }
            out = rank_agent_node(state, MultiAgentConfig())
        except ArtifactMissingError:
            pytest.skip("matched-scope frozen predictions not cached in this environment")
        buckets = out["rank_longs"] + out["rank_shorts"] + out["rank_abstained"]
        assert sorted(buckets) == sorted(state["target_symbols"])  # partition
        assert set(out["rank_longs"]) & set(out["rank_shorts"]) == set()
        # ranks are 1..N, sorted by signed prediction descending
        preds = [r["gate_pred"] for r in out["ranking"]]
        assert preds == sorted(preds, reverse=True)

    def test_missing_symbol_surfaced_not_dropped_silently(self):
        from src.multiagent.agents.rank_agent import rank_agent_node
        try:
            out = rank_agent_node({
                "target_symbols": ["VCB", "NOTREAL"],
                "target_horizon_days": 5, "prediction_time": "2025-06-03", "node_timings": {},
            }, MultiAgentConfig())
        except ArtifactMissingError:
            pytest.skip("matched-scope frozen predictions not cached")
        assert any("NOTREAL" in w for w in out["warnings"])
        assert "NOTREAL" not in (out["rank_longs"] + out["rank_shorts"] + out["rank_abstained"])


class TestResearchAgent:
    def _articles(self):
        return [
            {"id": "a1", "title": "VCB tăng trưởng tín dụng", "published_at": "2025-06-01", "sentiment_score": 0.4},
            {"id": "a2", "title": "Thị trường điều chỉnh", "published_at": "2025-05-20", "sentiment_score": -0.1},
        ]

    def test_no_trade_call_and_cites_ids(self):
        out = research_agent_node({"articles": self._articles(), "prediction_time": "2025-06-03",
                                   "node_timings": {}}, CFG_EVAL)
        summary = out["research_summary_vi"]
        assert "a1" in summary  # citation present
        # digest must not contain a buy/sell recommendation
        assert "MUA" not in summary and "BÁN" not in summary

    def test_empty_retrieval_is_honest(self):
        out = research_agent_node({"articles": [], "prediction_time": "2025-06-03",
                                   "node_timings": {}}, CFG_EVAL)
        assert "Không có bài báo" in out["research_summary_vi"]
        assert out["retrieved_docs"] == []

    def test_recency_ranking(self):
        docs = _rank_docs(self._articles(), "2025-06-03")
        assert docs[0]["id"] == "a1"  # most recent first
