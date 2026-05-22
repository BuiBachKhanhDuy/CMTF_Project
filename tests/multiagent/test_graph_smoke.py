"""Smoke test for the multi-agent graph topology.

Topology: orchestrator → [market_agent | news_agent] → predict_agent → fusion_agent → risk_agent → answer_agent → END

Orchestrator fetches data once; market/news agents are pure analytical nodes.
All external dependencies (prepare_single_cutoff, CMTF ensemble, LLM) are mocked.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.multiagent.config import MultiAgentConfig
from src.multiagent.loaders import set_loader_override, clear_overrides

# Canonical column order matching _CANONICAL_MARKET_COLS in orchestrator.py
_MARKET_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_lower", "bb_mid", "bb_upper", "atr_14",
    "vol_ratio", "log_ret",
    "vnindex_ret", "vnindex_vol_ratio",
    "sentiment_mean", "sentiment_max_abs",
    "sentiment_positive_ratio", "sentiment_negative_ratio",
    "sentiment_score_count", "sentiment_missing_flag",
]


@pytest.fixture(autouse=True)
def clean_loader_overrides():
    clear_overrides()
    yield
    clear_overrides()


def _mock_prepare_single_cutoff(symbol, cutoff, sequence_len=30, **kwargs):
    """Return deterministic fake data with realistic technical indicators."""
    np.random.seed(42)
    n_feat = len(_MARKET_COLS)

    # Build market_tabular with realistic values at known positions
    tabular = np.zeros(n_feat, dtype=np.float32)
    col_idx = {name: i for i, name in enumerate(_MARKET_COLS)}
    tabular[col_idx["close"]] = 55.0
    tabular[col_idx["rsi_14"]] = 35.0       # oversold → long bias
    tabular[col_idx["macd_hist"]] = 0.15     # positive → long bias
    tabular[col_idx["atr_14"]] = 1.0
    tabular[col_idx["bb_lower"]] = 50.0
    tabular[col_idx["bb_mid"]] = 56.0        # close < mid → long bias
    tabular[col_idx["bb_upper"]] = 62.0

    return {
        "close_window": np.linspace(50, 55, sequence_len).astype(np.float32),
        "market_window": np.random.randn(sequence_len, n_feat).astype(np.float32),
        "market_tabular": tabular,
        "market_feature_cols": _MARKET_COLS,
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
    }


def _mock_predict_with_explanation(**kw):
    """Mock CMTF predict_with_explanation with above-threshold signal."""
    return {
        "baseline_pred": 0.030,
        "final_pred": 0.035,
        "news_residual": 0.005,
        "attn_weights": np.random.rand(30).astype(np.float32),
        "news_weight": 0.15,
    }


class TestNewGraphSmoke:
    """End-to-end smoke test for new architecture with mocked dependencies."""

    @patch("src.pipeline.orchestrator.prepare_single_cutoff", side_effect=_mock_prepare_single_cutoff)
    def test_full_graph_eval_mode(self, mock_prepare):
        """Run full graph in evaluation mode (no LLM calls)."""
        from src.multiagent.graph import run_graph

        # Mock CMTF ensemble
        mock_predictor = MagicMock()
        mock_predictor.predict_with_explanation = lambda **kw: _mock_predict_with_explanation(**kw)
        mock_predictor.tokenize_windows = lambda x: (
            np.ones((1, 512), dtype=np.int64),
            np.ones((1, 512), dtype=np.int64),
        )
        set_loader_override("cmtf_ensemble_VCB_1d", [mock_predictor] * 3)

        cfg = MultiAgentConfig(evaluation_mode=True)
        result = run_graph(
            query_text="Should I buy VCB?",
            cutoff="2025-03-31",
            horizon=1,
            symbol="VCB",
            config=cfg,
        )

        # Orchestrator
        assert result["symbol"] == "VCB"
        assert result["target_horizon_days"] == 1

        # Market agent
        assert result["close_window"] is not None
        assert result["volatility_metrics"] is not None
        assert "vol_20d" in result["volatility_metrics"]

        # News agent
        assert result["news_emb"] is not None
        assert result["news_mask"] is not None
        assert result["sentiment_metrics"] is not None
        assert "sentiment_mean" in result["sentiment_metrics"]

        # Predict agent
        assert result["final_pred"] == pytest.approx(0.035, abs=1e-5)
        assert result["baseline_pred"] == pytest.approx(0.030, abs=1e-5)
        assert len(result["seed_preds"]) == 3
        assert result["predict_confidence"] is not None
        assert result["model_evidence"] is not None
        assert result["model_proposal"] is not None

        # Fusion agent
        assert result["fusion_decision"] is not None
        assert "score" in result["fusion_decision"]

        # Risk agent (final decision)
        assert result["action"] in ("long", "short", "flat")
        assert 0.0 <= result["position_scale"] <= 1.0
        assert result["risk_checks"] is not None
        assert result["decision_reasoning"] is not None
        assert result["policy_version"] >= 1

        # Answer agent (evaluation mode = empty)
        assert result["explanation_text_vi"] == ""

        # Audit
        assert result["data_cutoff"] == "2025-03-31"
        assert "orchestrator" in result["node_timings"]
        assert "market_agent" in result["node_timings"]
        assert "news_agent" in result["node_timings"]
        assert "predict_agent" in result["node_timings"]
        assert "fusion_agent" in result["node_timings"]
        assert "risk_agent" in result["node_timings"]
        assert "answer_agent" in result["node_timings"]

    @patch("src.pipeline.orchestrator.prepare_single_cutoff", side_effect=_mock_prepare_single_cutoff)
    def test_invalid_horizon_raises(self, mock_prepare):
        """Horizons other than 1, 5, 20 should raise ValueError."""
        from src.multiagent.graph import run_graph

        with pytest.raises(ValueError, match="horizon must be 1, 5, or 20"):
            run_graph(
                query_text="VCB 3 days",
                cutoff="2025-03-31",
                horizon=3,
                symbol="VCB",
                config=MultiAgentConfig(evaluation_mode=True),
            )

    @patch("src.pipeline.orchestrator.prepare_single_cutoff", side_effect=_mock_prepare_single_cutoff)
    def test_strong_signal_produces_long(self, mock_prepare):
        """pred=0.035 > 0.025 threshold → action should be long."""
        from src.multiagent.graph import run_graph

        mock_predictor = MagicMock()
        mock_predictor.predict_with_explanation = lambda **kw: _mock_predict_with_explanation(**kw)
        mock_predictor.tokenize_windows = lambda x: (
            np.ones((1, 512), dtype=np.int64),
            np.ones((1, 512), dtype=np.int64),
        )
        set_loader_override("cmtf_ensemble_VCB_1d", [mock_predictor] * 3)

        cfg = MultiAgentConfig(evaluation_mode=True)
        result = run_graph(
            cutoff="2025-03-31", horizon=1, symbol="VCB", config=cfg,
        )
        assert result["action"] == "long"
        assert result["position_scale"] > 0
