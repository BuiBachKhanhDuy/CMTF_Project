"""Tests for the Explanation Agent — Jinja2 fallback when Ollama is unavailable."""

import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from src.multiagent.config import MultiAgentConfig
from src.multiagent.explanation_agent import explanation_node, _build_evidence_dict, _render_jinja2_fallback


def _full_state():
    """Create a fully populated state for explanation testing."""
    return {
        "symbol": "VCB",
        "prediction_time": "2025-03-31",
        "target_horizon_days": 1,
        "sequence_len": 30,
        "baseline_pred": 0.005,
        "final_pred": 0.008,
        "final_pred_adjusted": 0.007,
        "seed_preds": [0.007, 0.008, 0.009],
        "news_residual": 0.003,
        "news_residual_scale": 1.0,
        "news_weight": 0.15,
        "attn_weights": np.random.rand(30).astype(np.float32),
        "articles": [
            {"title": "VCB tăng trưởng tín dụng", "published_at": "2025-03-30",
             "bar_index": 28, "sentiment_score": 0.8},
            {"title": "Ngân hàng Nhà nước giảm lãi suất", "published_at": "2025-03-29",
             "bar_index": 27, "sentiment_score": 0.6},
        ],
        "regime_flags": {
            "vol_20d": 0.02,
            "max_drawdown": -0.03,
            "max_drawdown_abs": 0.03,
            "high_vol": False,
            "drawdown_breach": False,
        },
        "news_quality_flags": {
            "coverage": 15,
            "staleness_frac": 0.2,
            "sentiment_std": 0.3,
        },
        "disagreement_force_flat": False,
        "action": "long",
        "position_scale": 1.0,
        "node_timings": {},
        "errors": [],
        "warnings": [],
    }


class TestBuildEvidenceDict:
    """Test evidence dict construction."""

    def test_evidence_contains_required_keys(self):
        state = _full_state()
        evidence = _build_evidence_dict(state)
        assert evidence["symbol"] == "VCB"
        assert evidence["action"] == "long"
        assert evidence["baseline_pred"] == 0.005
        assert evidence["news_residual"] == 0.003
        assert "top_attended_bars" in evidence
        assert len(evidence["top_attended_bars"]) <= 3

    def test_evidence_top_bars_ordered_by_attention(self):
        state = _full_state()
        # Set clear attention pattern
        attn = np.zeros(30, dtype=np.float32)
        attn[29] = 0.9
        attn[28] = 0.7
        attn[27] = 0.5
        state["attn_weights"] = attn
        evidence = _build_evidence_dict(state)
        # Top bars should be ordered descending by attention
        bars = evidence["top_attended_bars"]
        assert bars[0]["bar_index"] == 29
        assert bars[1]["bar_index"] == 28
        assert bars[2]["bar_index"] == 27


class TestJinja2Fallback:
    """Test Jinja2 template rendering."""

    def test_fallback_produces_vietnamese_text(self):
        state = _full_state()
        evidence = _build_evidence_dict(state)
        text = _render_jinja2_fallback(evidence)
        assert isinstance(text, str)
        assert len(text) > 50
        # Should contain Vietnamese keywords
        assert "MUA" in text or "BÁN" in text or "GIỮ" in text
        assert "VCB" in text

    def test_fallback_for_flat_action(self):
        state = _full_state()
        state["action"] = "flat"
        state["position_scale"] = 0.0
        evidence = _build_evidence_dict(state)
        text = _render_jinja2_fallback(evidence)
        assert "GIỮ" in text


class TestExplanationNode:
    """Test the full explanation node with Ollama mocked."""

    @patch("src.multiagent.explanation_agent._call_ollama", return_value=None)
    def test_fallback_when_ollama_unavailable(self, mock_ollama):
        """When Ollama fails, Jinja2 fallback should produce valid output."""
        state = _full_state()
        cfg = MultiAgentConfig()
        result = explanation_node(state, config=cfg)

        assert "evidence_dict" in result
        assert "explanation_text_vi" in result
        assert len(result["explanation_text_vi"]) > 0
        assert "VCB" in result["explanation_text_vi"]
        mock_ollama.assert_called_once()

    @patch("src.multiagent.explanation_agent._call_ollama")
    def test_uses_ollama_when_available(self, mock_ollama):
        """When Ollama succeeds, use its output."""
        mock_ollama.return_value = "Đây là giải thích từ Ollama cho VCB."
        state = _full_state()
        cfg = MultiAgentConfig()
        result = explanation_node(state, config=cfg)

        assert result["explanation_text_vi"] == "Đây là giải thích từ Ollama cho VCB."

    @patch("src.multiagent.explanation_agent._call_ollama", return_value=None)
    def test_node_timings_recorded(self, mock_ollama):
        """Node timing should be recorded."""
        state = _full_state()
        result = explanation_node(state)
        assert "explanation_agent" in result["node_timings"]
        assert result["node_timings"]["explanation_agent"] >= 0
