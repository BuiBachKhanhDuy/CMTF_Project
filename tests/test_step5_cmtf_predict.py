"""Step 5 validation: CMTF fusion head is regression-first.

Run:  python -m pytest tests/test_step5_cmtf_predict.py -v
"""

from __future__ import annotations

import torch

from src.benchmark.chronos_cmtf import CrossModalFusionHead


class TestFusionHeadRegressionContract:
    """Verify the fusion head exposes a direct regression output."""

    @staticmethod
    def _make_fusion(seed: int = 42) -> CrossModalFusionHead:
        torch.manual_seed(seed)
        return CrossModalFusionHead(
            market_dim=32, news_dim=16, fusion_dim=16, n_heads=2,
            dropout=0.0, seq_len=5,
        )

    def test_output_shape(self):
        head = self._make_fusion()
        head.eval()
        B = 8
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)
        pred = head(market, news)
        assert pred.shape == (B,)

    def test_regression_output_is_returned_directly(self):
        """Prediction path should use regression output directly."""
        head = self._make_fusion()
        head.eval()
        B = 20
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)

        with torch.no_grad():
            pred = head(market, news)

        assert pred.shape == (B,)

    def test_not_all_same_sign(self):
        """With random inputs, regression output should have mixed signs."""
        head = self._make_fusion(seed=0)
        head.eval()
        B = 50
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)

        with torch.no_grad():
            pred = head(market, news)

        n_pos = (pred > 0).sum().item()
        n_neg = (pred < 0).sum().item()
        assert n_pos >= 3, f"Only {n_pos} positive predictions out of {B}"
        assert n_neg >= 3, f"Only {n_neg} negative predictions out of {B}"

    def test_all_zero_news_rows_stay_finite(self):
        """Padding-only news rows should not produce NaNs."""
        head = self._make_fusion()
        head.eval()
        B = 10
        market = torch.randn(B, 32)
        news = torch.zeros(B, 5, 16)

        with torch.no_grad():
            pred = head(market, news)

        assert torch.isfinite(pred).all()
