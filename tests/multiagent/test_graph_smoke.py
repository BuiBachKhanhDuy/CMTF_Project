"""Smoke test for the full multi-agent graph — VCB 1d end-to-end.

This test uses mocked loaders to avoid requiring real model checkpoints.
It verifies the graph topology, state propagation, and that all nodes
produce their expected outputs.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.multiagent.config import MultiAgentConfig
from src.multiagent.loaders import set_loader_override, clear_overrides
from src.multiagent.state import MultiAgentState


@pytest.fixture(autouse=True)
def clean_loader_overrides():
    clear_overrides()
    yield
    clear_overrides()


def _mock_prepare_single_cutoff(symbol, cutoff, sequence_len=30, **kwargs):
    """Return deterministic fake data for testing the graph."""
    np.random.seed(42)
    market_cols = [f"feat_{i}" for i in range(23)]
    return {
        "close_window": np.linspace(50, 55, sequence_len).astype(np.float32),
        "market_window": np.random.randn(sequence_len, 23).astype(np.float32),
        "market_tabular": np.random.randn(23).astype(np.float32),
        "token_ids": np.ones((1, 512), dtype=np.int64),
        "attention_mask": np.ones((1, 512), dtype=np.int64),
        "news_emb": np.random.randn(sequence_len, 773).astype(np.float32),
        "news_mask": np.zeros(sequence_len, dtype=bool),
        "articles": [
            {"title": "VCB lợi nhuận tăng", "published_at": "2025-03-30",
             "bar_index": 28, "sentiment_score": 0.7},
            {"title": "Ngân hàng số phát triển", "published_at": "2025-03-29",
             "bar_index": 27, "sentiment_score": 0.5},
        ],
        "sentiment_features": {
            "sentiment_mean": 0.3,
            "sentiment_max_abs": 0.7,
            "sentiment_positive_ratio": 0.6,
            "sentiment_negative_ratio": 0.1,
            "sentiment_score_count": 2.0,
            "sentiment_missing_flag": 0.0,
        },
        "market_feature_cols": [f"feat_{i}" for i in range(23)],
    }


def _mock_predict_with_explanation(self, token_ids, attention_mask, news_test,
                                    tabular_test=None, market_windows_test=None,
                                    news_mask_test=None):
    """Mock predict_with_explanation for testing."""
    seq_len = news_test.shape[1] if news_test.ndim >= 2 else 30
    return {
        "baseline_pred": 0.003,
        "final_pred": 0.005,
        "news_residual": 0.002,
        "attn_weights": np.random.rand(seq_len).astype(np.float32),
        "news_weight": 0.15,
    }


class TestGraphSmoke:
    """End-to-end smoke test with mocked dependencies."""

    @patch("src.pipeline.orchestrator.prepare_single_cutoff", side_effect=_mock_prepare_single_cutoff)
    @patch("src.multiagent.explanation_agent._call_ollama", return_value=None)
    def test_full_graph_vcb_1d(self, mock_ollama, mock_prepare):
        """Run the full graph for VCB horizon=1d and verify all state keys populated."""
        from src.multiagent.graph import run_graph

        # Mock the CMTF ensemble
        mock_predictor = MagicMock()
        mock_predictor.predict_with_explanation = lambda **kw: _mock_predict_with_explanation(
            None, kw["token_ids"], kw["attention_mask"], kw["news_test"],
            kw.get("tabular_test"), kw.get("market_windows_test"), kw.get("news_mask_test")
        )
        mock_predictor.tokenize_windows = lambda x: (
            np.ones((1, 512), dtype=np.int64),
            np.ones((1, 512), dtype=np.int64),
        )
        set_loader_override("cmtf_ensemble_VCB_1d", [mock_predictor] * 3)

        cfg = MultiAgentConfig(
            buy_threshold=0.002,
            sell_threshold=0.002,
        )

        result = run_graph(symbol="VCB", cutoff="2025-03-31", horizon=1, config=cfg)

        # --- Verify all state groups are populated ---
        # Request
        assert result["symbol"] == "VCB"
        assert result["prediction_time"] == "2025-03-31"
        assert result["target_horizon_days"] == 1

        # Market
        assert result["close_window"] is not None
        assert result["market_window"] is not None

        # News
        assert result["news_emb"] is not None
        assert result["news_mask"] is not None

        # Fusion
        assert result["baseline_pred"] is not None
        assert result["final_pred"] is not None
        assert result["seed_preds"] is not None
        assert len(result["seed_preds"]) == 3
        assert result["news_residual"] is not None
        assert result["attn_weights"] is not None

        # Critics
        assert result["regime_flags"] is not None
        assert result["position_scale_regime"] is not None
        assert result["news_quality_flags"] is not None
        assert result["news_residual_scale"] is not None
        assert result["final_pred_adjusted"] is not None
        assert result["disagreement_force_flat"] is not None

        # Decision
        assert result["action"] in ("long", "short", "flat")
        assert 0 <= result["position_scale"] <= 1.0

        # Explanation
        assert result["evidence_dict"] is not None
        assert result["explanation_text_vi"] is not None
        assert len(result["explanation_text_vi"]) > 0

        # Audit
        assert result["data_cutoff"] == "2025-03-31"
        assert "market_agent" in result["node_timings"]
        assert "fusion_agent" in result["node_timings"]
        assert "decision_agent" in result["node_timings"]

    @patch("src.pipeline.orchestrator.prepare_single_cutoff", side_effect=_mock_prepare_single_cutoff)
    @patch("src.multiagent.explanation_agent._call_ollama", return_value=None)
    def test_invalid_horizon_raises(self, mock_ollama, mock_prepare):
        """Horizons other than 1, 5, 20 should raise ValueError."""
        from src.multiagent.graph import run_graph

        with pytest.raises(ValueError, match="horizon must be 1, 5, or 20"):
            run_graph(symbol="VCB", cutoff="2025-03-31", horizon=3)

    @patch("src.pipeline.orchestrator.prepare_single_cutoff")
    @patch("src.multiagent.explanation_agent._call_ollama", return_value=None)
    def test_zero_news_path(self, mock_ollama, mock_prepare):
        """When no news is available, should use baseline prediction."""
        def no_news_cutoff(symbol, cutoff, sequence_len=30, **kwargs):
            data = _mock_prepare_single_cutoff(symbol, cutoff, sequence_len)
            data["news_emb"] = np.zeros((sequence_len, 773), dtype=np.float32)
            data["news_mask"] = np.ones(sequence_len, dtype=bool)
            data["articles"] = []
            return data

        mock_prepare.side_effect = no_news_cutoff

        # Mock predictor that returns zero residual for masked news
        mock_predictor = MagicMock()
        def zero_news_predict(**kw):
            return {
                "baseline_pred": 0.003,
                "final_pred": 0.003,  # Same as baseline
                "news_residual": 0.0,
                "attn_weights": np.zeros(30, dtype=np.float32),
                "news_weight": 0.1,
            }
        mock_predictor.predict_with_explanation = lambda **kw: zero_news_predict(**kw)
        mock_predictor.tokenize_windows = lambda x: (
            np.ones((1, 512), dtype=np.int64), np.ones((1, 512), dtype=np.int64)
        )
        set_loader_override("cmtf_ensemble_VCB_1d", [mock_predictor] * 3)

        from src.multiagent.graph import run_graph
        cfg = MultiAgentConfig(buy_threshold=0.002, min_news_bars=3)
        result = run_graph(symbol="VCB", cutoff="2025-03-31", horizon=1, config=cfg)

        # News quality critic should ignore news (coverage=0 < min_news_bars=3)
        assert result["news_residual_scale"] == 0.0
        # Final adjusted = baseline + 0 * residual = baseline
        assert result["final_pred_adjusted"] == pytest.approx(result["baseline_pred"])
