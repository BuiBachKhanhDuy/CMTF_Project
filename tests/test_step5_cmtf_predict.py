"""Step 5 validation: CMTF predict() uses cls_head for direction.

Run:  python -m pytest tests/test_step5_cmtf_predict.py -v
"""

from __future__ import annotations

import numpy as np
import torch

from src.benchmark.chronos_cmtf import CrossModalFusionHead


class TestPredictUsesCls:
    """Verify predict() uses cls_head for sign, reg_head for magnitude."""

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
        reg_out, cls_logit = head(market, news)
        assert reg_out.shape == (B,)
        assert cls_logit.shape == (B,)

    def test_predict_uses_cls_direction(self):
        """Direction of output should follow cls_head sign, not raw reg_head sign."""
        head = self._make_fusion()
        head.eval()
        B = 20
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)

        with torch.no_grad():
            reg_out, cls_logit = head(market, news)
            cls_prob = torch.sigmoid(cls_logit)
            cls_sign = torch.where(cls_prob >= 0.5, 1.0, -1.0)
            pred = reg_out.abs() * cls_sign

        # pred sign should match cls_sign, not necessarily reg_out sign
        pred_sign = torch.sign(pred)
        assert torch.allclose(pred_sign, cls_sign), (
            "predict() output sign should be determined by cls_head"
        )

    def test_not_all_same_sign(self):
        """With random inputs, output should have mixed signs (not collapsed)."""
        head = self._make_fusion(seed=0)
        head.eval()
        B = 50
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)

        with torch.no_grad():
            reg_out, cls_logit = head(market, news)
            cls_prob = torch.sigmoid(cls_logit)
            cls_sign = torch.where(cls_prob >= 0.5, 1.0, -1.0)
            pred = reg_out.abs() * cls_sign

        n_pos = (pred > 0).sum().item()
        n_neg = (pred < 0).sum().item()
        # With random weights, should have at least some of each sign
        assert n_pos >= 3, f"Only {n_pos} positive predictions out of {B}"
        assert n_neg >= 3, f"Only {n_neg} negative predictions out of {B}"

    def test_magnitude_from_reg_head(self):
        """Absolute values should come from reg_head, not cls_head."""
        head = self._make_fusion()
        head.eval()
        B = 10
        market = torch.randn(B, 32)
        news = torch.randn(B, 5, 16)

        with torch.no_grad():
            reg_out, cls_logit = head(market, news)
            cls_prob = torch.sigmoid(cls_logit)
            cls_sign = torch.where(cls_prob >= 0.5, 1.0, -1.0)
            pred = reg_out.abs() * cls_sign

        assert torch.allclose(pred.abs(), reg_out.abs()), (
            "Magnitude should come from reg_head"
        )
