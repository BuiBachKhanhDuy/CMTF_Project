"""Tests for ChronosCMTFPredictor.predict_with_explanation() patch."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


class TestPredictWithExplanation:
    """Test the new predict_with_explanation method."""

    def _make_mock_predictor(self, seq_len=30, news_dim=773, baseline_dim=599):
        """Create a minimal mock CMTF predictor for testing."""
        from src.benchmark.chronos_cmtf import ResidualNewsFusionHead

        predictor = MagicMock()
        predictor.is_fitted = True
        predictor.device = "cpu"

        # Create a real fusion head for testing
        fusion = ResidualNewsFusionHead(
            baseline_dim=baseline_dim,
            market_dim=512,
            news_dim=news_dim,
            hidden_dim=64,
            n_heads=2,
            dropout=0.1,
            seq_len=seq_len,
        )
        predictor.fusion = fusion

        # Mock _set_market_path_mode
        predictor._set_market_path_mode = MagicMock()

        # Mock _extract_market_state to return deterministic features
        def mock_extract(token_ids, attention_mask, market_windows=None, market_tabular=None):
            B = token_ids.shape[0]
            features = torch.randn(B, baseline_dim)
            baseline_pred = torch.zeros(B)
            return features, baseline_pred

        predictor._extract_market_state = mock_extract

        return predictor

    def test_returns_correct_keys(self):
        """predict_with_explanation should return all required keys."""
        from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

        # We need to test the method exists and returns proper keys
        # Use a real instance with mocked internals
        seq_len = 30
        news_dim = 773

        mock_lora = MagicMock()
        mock_lora.market_input_dim = 23
        mock_lora.combined_feature_dim = 599
        mock_backbone = MagicMock()
        mock_backbone.d_model = 512
        mock_backbone.transformer = MagicMock()
        mock_backbone.tokenizer = MagicMock()
        mock_backbone.trainable_parameters = MagicMock(return_value=[])
        mock_backbone.trainable_parameter_names = MagicMock(return_value=[])
        mock_lora.backbone = mock_backbone

        # Mock extract_tokenized_features and regress_features
        def mock_extract_features(token_ids, attention_mask, market_windows=None, market_tabular=None):
            B = token_ids.shape[0]
            return torch.randn(B, 599)

        def mock_regress(features):
            return torch.zeros(features.shape[0])

        mock_lora.extract_tokenized_features = mock_extract_features
        mock_lora.regress_features = mock_regress

        predictor = ChronosCMTFPredictor(
            chronos_lora_predictor=mock_lora,
            news_dim=news_dim,
            fusion_dim=64,
            n_heads=2,
            dropout=0.1,
            seq_len=seq_len,
            device="cpu",
        )
        predictor.is_fitted = True

        # Create inputs
        token_ids = np.ones((1, 512), dtype=np.int64)
        attention_mask = np.ones((1, 512), dtype=np.int64)
        news = np.random.randn(1, seq_len, news_dim).astype(np.float32)
        news_mask = np.zeros((1, seq_len), dtype=bool)

        result = predictor.predict_with_explanation(
            token_ids=token_ids,
            attention_mask=attention_mask,
            news_test=news,
            news_mask_test=news_mask,
        )

        assert "baseline_pred" in result
        assert "final_pred" in result
        assert "news_residual" in result
        assert "attn_weights" in result
        assert "news_weight" in result
        assert isinstance(result["baseline_pred"], float)
        assert isinstance(result["final_pred"], float)
        assert isinstance(result["news_residual"], float)
        assert isinstance(result["news_weight"], float)
        assert result["attn_weights"].shape == (seq_len,)

    def test_zero_news_parity(self):
        """When all news is masked, news_residual should be ~0."""
        from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

        seq_len = 30
        news_dim = 773

        mock_lora = MagicMock()
        mock_lora.market_input_dim = 23
        mock_lora.combined_feature_dim = 599
        mock_backbone = MagicMock()
        mock_backbone.d_model = 512
        mock_backbone.transformer = MagicMock()
        mock_backbone.tokenizer = MagicMock()
        mock_backbone.trainable_parameters = MagicMock(return_value=[])
        mock_backbone.trainable_parameter_names = MagicMock(return_value=[])
        mock_lora.backbone = mock_backbone

        def mock_extract_features(token_ids, attention_mask, market_windows=None, market_tabular=None):
            B = token_ids.shape[0]
            return torch.randn(B, 599)

        def mock_regress(features):
            return torch.full((features.shape[0],), 0.01)

        mock_lora.extract_tokenized_features = mock_extract_features
        mock_lora.regress_features = mock_regress

        predictor = ChronosCMTFPredictor(
            chronos_lora_predictor=mock_lora,
            news_dim=news_dim,
            fusion_dim=64,
            n_heads=2,
            dropout=0.1,
            seq_len=seq_len,
            device="cpu",
        )
        predictor.is_fitted = True

        # All news masked (no news available)
        token_ids = np.ones((1, 512), dtype=np.int64)
        attention_mask = np.ones((1, 512), dtype=np.int64)
        news = np.zeros((1, seq_len, news_dim), dtype=np.float32)
        news_mask = np.ones((1, seq_len), dtype=bool)  # All True = all masked

        result = predictor.predict_with_explanation(
            token_ids=token_ids,
            attention_mask=attention_mask,
            news_test=news,
            news_mask_test=news_mask,
        )

        # Zero-news parity: residual should be exactly 0 due to has_news gate
        assert abs(result["news_residual"]) < 1e-6, (
            f"Expected ~0 residual with all news masked, got {result['news_residual']}"
        )

    def test_seed_preds_not_in_single_call(self):
        """predict_with_explanation returns per-model result, not ensemble.
        The ensemble averaging is done in fusion_agent."""
        from src.benchmark.chronos_cmtf import ChronosCMTFPredictor

        mock_lora = MagicMock()
        mock_lora.market_input_dim = 23
        mock_lora.combined_feature_dim = 599
        mock_backbone = MagicMock()
        mock_backbone.d_model = 512
        mock_backbone.transformer = MagicMock()
        mock_backbone.tokenizer = MagicMock()
        mock_backbone.trainable_parameters = MagicMock(return_value=[])
        mock_backbone.trainable_parameter_names = MagicMock(return_value=[])
        mock_lora.backbone = mock_backbone
        mock_lora.extract_tokenized_features = lambda *a, **kw: torch.randn(1, 599)
        mock_lora.regress_features = lambda f: torch.zeros(f.shape[0])

        predictor = ChronosCMTFPredictor(
            chronos_lora_predictor=mock_lora,
            news_dim=773,
            fusion_dim=64,
            n_heads=2,
            seq_len=30,
            device="cpu",
        )
        predictor.is_fitted = True

        result = predictor.predict_with_explanation(
            token_ids=np.ones((1, 512), dtype=np.int64),
            attention_mask=np.ones((1, 512), dtype=np.int64),
            news_test=np.random.randn(1, 30, 773).astype(np.float32),
            news_mask_test=np.zeros((1, 30), dtype=bool),
        )

        # Single predictor returns single values, not lists
        assert isinstance(result["final_pred"], float)
        assert isinstance(result["baseline_pred"], float)
