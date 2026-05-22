"""Tests for CNNLSTMCMTFPredictor.predict_with_explanation()."""

import numpy as np
import torch

from src.benchmark.cnn_lstm_cmtf import CNNLSTMCMTFPredictor


class TestPredictWithExplanation:
    """Test the deployed predictor explainability interface."""

    def _make_predictor(self, seq_len: int = 30, input_dim: int = 23, news_dim: int = 773):
        """Create a small real predictor instance for interface tests."""
        torch.manual_seed(0)
        predictor = CNNLSTMCMTFPredictor(
            input_dim=input_dim,
            news_dim=news_dim,
            hidden_dim=16,
            num_filters=16,
            num_layers=2,
            dropout=0.0,
            fusion_dim=32,
            fusion_market_dim=32,
            n_heads=4,
            seq_len=seq_len,
            device="cpu",
        )
        predictor.is_fitted = True
        return predictor

    def test_returns_correct_keys(self):
        """predict_with_explanation should return all required keys."""
        seq_len = 30
        rng = np.random.default_rng(0)
        predictor = self._make_predictor(seq_len=seq_len)

        result = predictor.predict_with_explanation(
            market_windows_test=rng.normal(size=(1, seq_len, 23)).astype(np.float32),
            news_test=rng.normal(size=(1, seq_len, 773)).astype(np.float32),
            news_mask_test=np.zeros((1, seq_len), dtype=bool),
        )

        assert "baseline_pred" in result
        assert "final_pred" in result
        assert "news_residual" in result
        assert "attn_weights" in result
        assert "quality_gate" in result
        assert "news_weight" in result
        assert isinstance(result["baseline_pred"], float)
        assert isinstance(result["final_pred"], float)
        assert isinstance(result["news_residual"], float)
        assert isinstance(result["quality_gate"], float)
        assert isinstance(result["news_weight"], float)
        assert result["attn_weights"].shape == (seq_len,)

    def test_zero_news_parity(self):
        """When all news is masked, the residual should be exactly zero."""
        seq_len = 30
        predictor = self._make_predictor(seq_len=seq_len)

        result = predictor.predict_with_explanation(
            market_windows_test=np.zeros((1, seq_len, 23), dtype=np.float32),
            news_test=np.zeros((1, seq_len, 773), dtype=np.float32),
            news_mask_test=np.ones((1, seq_len), dtype=bool),
        )

        assert abs(result["news_residual"]) < 1e-6, (
            f"Expected ~0 residual with all news masked, got {result['news_residual']}"
        )
        assert abs(result["final_pred"] - result["baseline_pred"]) < 1e-6

    def test_seed_preds_not_in_single_call(self):
        """predict_with_explanation returns single-model scalars, not ensembles."""
        seq_len = 30
        rng = np.random.default_rng(1)
        predictor = self._make_predictor(seq_len=seq_len)

        result = predictor.predict_with_explanation(
            market_windows_test=rng.normal(size=(1, seq_len, 23)).astype(np.float32),
            news_test=rng.normal(size=(1, seq_len, 773)).astype(np.float32),
            news_mask_test=np.zeros((1, seq_len), dtype=bool),
        )

        assert isinstance(result["final_pred"], float)
        assert isinstance(result["baseline_pred"], float)
