"""Tests for the Orchestrator Agent — intent/NER extraction."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.multiagent.agents.orchestrator_agent import (
    orchestrator_node,
    _extract_symbols,
    _extract_horizon,
    _deterministic_parse,
)
from src.multiagent.config import MultiAgentConfig


class TestDeterministicParsing:
    def test_extract_symbols_known(self):
        assert _extract_symbols("Should I buy VCB?") == ["VCB"]

    def test_extract_symbols_multiple(self):
        result = _extract_symbols("VCB vs TCB which is better?")
        assert "VCB" in result
        assert "TCB" in result

    def test_extract_symbols_none(self):
        assert _extract_symbols("What is the market doing?") == []

    def test_extract_horizon_default_1d(self):
        assert _extract_horizon("buy VCB") == "1d"

    def test_extract_horizon_5d(self):
        assert _extract_horizon("VCB for 5 days") == "5d"

    def test_extract_horizon_20d(self):
        assert _extract_horizon("VCB dài hạn") == "20d"

    def test_deterministic_parse_prediction(self):
        result = _deterministic_parse("Should I buy VCB for 5 days?")
        assert result["intent"] == "PREDICTION"
        assert result["symbols"] == ["VCB"]
        assert result["horizon"] == "5d"

    def test_deterministic_parse_explanation(self):
        result = _deterministic_parse("Tại sao HPG giảm?")
        assert result["intent"] == "EXPLANATION"


class TestOrchestratorNode:
    @patch("src.pipeline.orchestrator.prepare_single_cutoff")
    def test_fast_path_cli_params(self, mock_prepare):
        """CLI-provided symbol+horizon fetches data and returns it."""
        import numpy as np
        mock_prepare.return_value = {
            "close_window": np.ones(30, dtype=np.float32),
            "market_window": np.ones((30, 23), dtype=np.float32),
            "market_tabular": np.ones(23, dtype=np.float32),
            "market_feature_cols": ["close"] * 23,
            "news_emb": np.zeros((30, 773), dtype=np.float32),
            "news_mask": np.ones(30, dtype=bool),
            "articles": [],
        }
        state = {
            "query_text": "anything",
            "symbol": "VCB",
            "prediction_time": "2025-03-31",
            "target_horizon_days": 1,
            "errors": [],
            "node_timings": {},
        }
        result = orchestrator_node(state)
        assert "orchestrator" in result["node_timings"]
        assert result["close_window"] is not None
        assert result["data_cutoff"] == "2025-03-31"
        mock_prepare.assert_called_once()

    def test_eval_mode_deterministic(self):
        """Evaluation mode uses regex, not LLM."""
        state = {"query_text": "Should I buy VCB for 5 days?", "errors": [], "node_timings": {}}
        cfg = MultiAgentConfig(evaluation_mode=True)
        result = orchestrator_node(state, config=cfg)
        assert result["symbol"] == "VCB"
        assert result["target_horizon_days"] == 5

    @patch("langchain_ollama.ChatOllama")
    def test_normal_mode_calls_llm(self, MockLLM):
        """Normal mode calls LLM orchestrator."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": "PREDICTION",
            "symbols": ["BID"],
            "horizon": "20d",
            "aspect": "price",
        })
        MockLLM.return_value.invoke.return_value = mock_response

        state = {"query_text": "Dự báo BID 20 ngày", "errors": [], "node_timings": {}}
        cfg = MultiAgentConfig(evaluation_mode=False)
        result = orchestrator_node(state, config=cfg)
        assert result["symbol"] == "BID"
        assert result["target_horizon_days"] == 20
        MockLLM.return_value.invoke.assert_called_once()
