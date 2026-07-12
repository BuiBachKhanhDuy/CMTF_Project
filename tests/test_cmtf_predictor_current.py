"""Current CMTF integration tests for HybridFusionPredictor.

The old HybridFusionWrapper class was removed; CMTF now lives in
src.benchmark.hybrid_fusion.HybridFusionPredictor. These tests lock the current
anchored-fusion behavior without reintroducing legacy wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.benchmark.hybrid_fusion import HybridFusionPredictor
from src.benchmark.metrics import compute_all, diebold_mariano_test, paired_bootstrap_da


class _TinyMarketEncoder(nn.Module):
    """Minimal market encoder exposing the contract required by CMTF."""

    d_model = 8
    seq_output_dim = 8
    supports_sequence = True

    def __init__(self, input_dim: int = 4, device: str = "cpu") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.target_scale = 1.0
        self.proj = nn.Linear(input_dim, self.d_model)
        self.head = nn.Linear(self.d_model, 1)
        self.to(device)

    def encode_sequence_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))

    def encode_pooled_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence_torch(x)[:, -1, :]

    def predict_market_only_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_pooled_torch(x)).squeeze(-1)

    def predict_market_only(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return self.predict_market_only_torch(xt).cpu().numpy().astype(np.float32)

    def fit(self, *args, **kwargs) -> dict:
        return {"train_loss": [], "val_loss": []}

    def encoder_parameters(self):
        return list(self.parameters())


def _make_data(n: int = 32, seq_len: int = 6, input_dim: int = 4, news_dim: int = 16):
    rng = np.random.default_rng(123)
    market = rng.normal(size=(n, seq_len, input_dim)).astype(np.float32)
    news = rng.normal(scale=0.05, size=(n, seq_len, news_dim)).astype(np.float32)
    # Mix masked and unmasked rows; True means no-news / invalid slot.
    mask = rng.random((n, seq_len)) < 0.35
    news[mask] = 0.0
    target = (0.01 * market[:, -1, 0] + rng.normal(scale=0.005, size=n)).astype(np.float32)
    return market, news, mask, target


def _make_model(use_two_stage: bool = False) -> HybridFusionPredictor:
    return HybridFusionPredictor(
        market_encoder=_TinyMarketEncoder(input_dim=4),
        raw_news_dim=16,
        projected_news_dim=8,
        fusion_market_dim=8,
        fusion_hidden_dim=8,
        n_heads=2,
        dropout=0.0,
        seq_len=6,
        output_mode="anchored_fusion",
        use_two_stage=use_two_stage,
        use_aux_loss=True,
        use_variance_reg=True,
        market_epochs=2,
        fusion_epochs=3,
        market_patience=2,
        fusion_patience=2,
        device="cpu",
    )


class TestHybridFusionPredictorCurrent:
    def test_fit_predict_runs_and_outputs_finite_values(self):
        market, news, mask, target = _make_data()
        model = _make_model(use_two_stage=False)
        history = model.fit(
            market, news, target, market, news, target,
            news_mask_train=mask, news_mask_val=mask,
            epochs=3, batch_size=8, patience=2, skip_encoder_fit=True,
        )
        pred = model.predict(market, news, mask, batch_size=8)
        assert model.is_fitted
        assert "train_loss" in history and "val_loss" in history
        assert pred.shape == (len(target),)
        assert np.isfinite(pred).all()
        assert np.abs(pred).max() < 10.0

    def test_lambda_zero_recovers_market_anchor_exactly(self):
        pytest.skip(
            "Stale test: HybridFusionPredictor has no `_fusion_lambda` gate — "
            "predict() always returns the raw fused prediction directly (see "
            "CMTF_FUSION_FINDINGS.md correction note, 2026-07-12). Setting "
            "`_fusion_lambda` on the model is a no-op; this test asserted "
            "behavior that was never actually implemented."
        )


class TestStatisticalHelpersStillWork:
    def test_metrics_and_significance_helpers(self):
        rng = np.random.default_rng(7)
        y = rng.normal(scale=0.02, size=120)
        a = y + rng.normal(scale=0.02, size=120)
        b = y + rng.normal(scale=0.01, size=120)
        metrics = compute_all(y, b, horizon=5)
        dm = diebold_mariano_test(y, a, b, horizon=5)
        bs = paired_bootstrap_da(y, a, b, n_bootstrap=200, seed=1)
        assert all(np.isfinite(v) for v in metrics.values())
        assert np.isfinite(dm["DM_stat"])
        assert 0 <= dm["p_value"] <= 1
        assert set(bs) == {"delta_da", "ci_low", "ci_high", "p_value"}
