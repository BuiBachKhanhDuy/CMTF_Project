"""End-to-end integration test for the CMTF (HybridFusionWrapper) implementation.

Tests the full training + prediction pipeline with:
1. Single-branch + single-stage (DummyEncoder fallback)
2. Dual-branch + single-stage
3. Two-stage training with a real LSTMPredictor (TemporalEncoder)
4. Verify predictions are sensible (finite, bounded, zero-news invariance)
5. Verify DM test + bootstrap on actual predictions
"""

import numpy as np
import torch
import pytest

from src.benchmark.fusion_wrappers import HybridFusionWrapper
from src.benchmark.baseline_models import LSTMPredictor, _TARGET_SCALE
from src.benchmark.metrics import (
    compute_all, diebold_mariano_test, paired_bootstrap_da,
)


# ======================================================================
# Fixtures
# ======================================================================

class _DummyEncoder:
    """Non-TemporalEncoder — forces single-stage fallback."""
    d_model = 16
    supports_sequence = True

    def encode(self, mw):
        return np.ones((len(mw), self.d_model), dtype=np.float32)

    def predict_market_only(self, mw):
        return np.full(len(mw), 0.01, dtype=np.float32)


def _make_data(n=80, seq_len=30, input_dim=5, news_dim=768, news_coverage=0.3, seed=42):
    """Generate synthetic data mimicking the real pipeline structure."""
    rng = np.random.RandomState(seed)
    mw = rng.randn(n, seq_len, input_dim).astype(np.float32)
    ne = np.zeros((n, seq_len, news_dim), dtype=np.float32)
    # Sprinkle news in last few bars (realistic: only some bars have news)
    for i in range(n):
        for t in range(seq_len):
            if rng.rand() < news_coverage:
                ne[i, t, :] = rng.randn(news_dim).astype(np.float32) * 0.1
    nm = (ne.sum(axis=-1) == 0)  # True = no news
    y = rng.randn(n).astype(np.float32) * 0.02
    return mw, ne, nm, y


# ======================================================================
# Test 1: Single-branch + single-stage (non-TemporalEncoder)
# ======================================================================

class TestSingleBranchSingleStage:
    def test_training_completes(self):
        mw, ne, nm, y = _make_data(n=50)
        enc = _DummyEncoder()
        wrapper = HybridFusionWrapper(
            encoder=enc, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=True,
            use_aux_loss=True, use_variance_reg=True,
        )
        assert not wrapper._is_temporal
        history = wrapper.fit(mw, ne, y, mw, ne, y,
                              news_mask_train=nm, news_mask_val=nm,
                              epochs=5, batch_size=16, patience=10)
        assert len(history["train_loss"]) == 5
        assert all(np.isfinite(history["train_loss"]))
        assert all(np.isfinite(history["val_loss"]))

    def test_predictions_finite_and_bounded(self):
        mw, ne, nm, y = _make_data(n=50)
        enc = _DummyEncoder()
        wrapper = HybridFusionWrapper(
            encoder=enc, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=False,
        )
        wrapper.fit(mw, ne, y, mw, ne, y, news_mask_train=nm, news_mask_val=nm,
                    epochs=3, batch_size=16, patience=10)
        preds = wrapper.predict(mw, ne, nm)
        assert preds.shape == (50,)
        assert np.isfinite(preds).all()
        # Predictions should be bounded (not exploding)
        assert np.abs(preds).max() < 1.0, f"Preds too large: {np.abs(preds).max()}"

    def test_zero_news_equals_market_only(self):
        mw, ne, nm, y = _make_data(n=30)
        enc = _DummyEncoder()
        wrapper = HybridFusionWrapper(
            encoder=enc, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=False,
        )
        wrapper.fit(mw, ne, y, mw, ne, y, news_mask_train=nm, news_mask_val=nm,
                    epochs=3, batch_size=16, patience=10)
        zero_news = np.zeros_like(ne)
        all_masked = np.ones_like(nm)
        preds_zero = wrapper.predict(mw, zero_news, all_masked)
        market_only = enc.predict_market_only(mw)
        np.testing.assert_allclose(preds_zero, market_only, atol=1e-5)


# ======================================================================
# Test 2: Two-stage training with real LSTMPredictor
# ======================================================================

class TestTwoStageWithLSTM:
    def _make_lstm_and_data(self):
        """Create a trained LSTM encoder + matching data."""
        input_dim = 5
        mw, ne, nm, y = _make_data(n=80, input_dim=input_dim)
        encoder = LSTMPredictor(
            input_dim=input_dim, hidden_dim=32, num_layers=1,
            dropout=0.0, device="cpu",
        )
        # Train encoder first (as ablation_runner does)
        encoder.fit(mw, y, mw, y, epochs=5, batch_size=32)
        return encoder, mw, ne, nm, y

    def test_two_stage_trains_successfully(self):
        encoder, mw, ne, nm, y = self._make_lstm_and_data()
        wrapper = HybridFusionWrapper(
            encoder=encoder, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=True,
            use_aux_loss=True, use_variance_reg=True,
        )
        assert wrapper._is_temporal, "LSTM should be TemporalEncoder"
        history = wrapper.fit(mw, ne, y, mw, ne, y,
                              news_mask_train=nm, news_mask_val=nm,
                              batch_size=16, patience=8)
        # Should have Stage1 + Stage2 epochs
        assert len(history["train_loss"]) >= 2, "Should have at least 2 epochs total"
        assert all(np.isfinite(history["train_loss"]))
        assert all(np.isfinite(history["val_loss"]))

    def test_two_stage_predictions_sensible(self):
        encoder, mw, ne, nm, y = self._make_lstm_and_data()
        wrapper = HybridFusionWrapper(
            encoder=encoder, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=True,
            use_aux_loss=True, use_variance_reg=True,
        )
        wrapper.fit(mw, ne, y, mw, ne, y,
                    news_mask_train=nm, news_mask_val=nm,
                    batch_size=16, patience=8)
        preds = wrapper.predict(mw, ne, nm)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()
        assert np.abs(preds).max() < 1.0, f"Exploding preds: max={np.abs(preds).max()}"

    def test_encoder_weights_changed_after_stage2(self):
        """Verify encoder actually gets fine-tuned in Stage 2."""
        encoder, mw, ne, nm, y = self._make_lstm_and_data()
        # Save encoder weights before CMTF
        pre_weights = {k: v.clone() for k, v in encoder.state_dict().items()}

        wrapper = HybridFusionWrapper(
            encoder=encoder, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=True,
            use_aux_loss=True, use_variance_reg=True,
        )
        wrapper.fit(mw, ne, y, mw, ne, y,
                    news_mask_train=nm, news_mask_val=nm,
                    batch_size=16, patience=8)

        # Encoder weights should have changed
        changed = False
        for k, v in encoder.state_dict().items():
            if not torch.allclose(v, pre_weights[k], atol=1e-7):
                changed = True
                break
        assert changed, "Encoder weights should change during Stage 2 fine-tuning"

    def test_zero_news_close_to_market_only(self):
        """With zero news, CMTF prediction should be very close to market-only."""
        encoder, mw, ne, nm, y = self._make_lstm_and_data()
        wrapper = HybridFusionWrapper(
            encoder=encoder, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.0, seq_len=30, device="cpu",
            use_two_stage=True,
        )
        wrapper.fit(mw, ne, y, mw, ne, y,
                    news_mask_train=nm, news_mask_val=nm,
                    batch_size=16, patience=8)
        zero_news = np.zeros_like(ne)
        all_masked = np.ones_like(nm)
        preds_zero = wrapper.predict(mw, zero_news, all_masked)
        market_only = encoder.predict_market_only(mw)
        np.testing.assert_allclose(preds_zero, market_only, atol=1e-5)


# ======================================================================
# Test 4: Statistical tests on predictions
# ======================================================================

class TestStatisticalSignificance:
    def test_dm_test_on_real_predictions(self):
        """DM test should return finite values on realistic prediction arrays."""
        rng = np.random.RandomState(123)
        y_true = rng.randn(200) * 0.02
        preds_a = y_true + rng.randn(200) * 0.015  # baseline
        preds_b = y_true + rng.randn(200) * 0.010  # better model

        result = diebold_mariano_test(y_true, preds_a, preds_b, horizon=5)
        assert np.isfinite(result["DM_stat"])
        assert 0 <= result["p_value"] <= 1
        # Better model should produce positive DM stat (A has higher loss)
        assert result["DM_stat"] > 0

    def test_bootstrap_da_distinguishes_models(self):
        """Bootstrap should detect DA improvement when B is significantly better."""
        rng = np.random.RandomState(456)
        y_true = rng.randn(300) * 0.02
        # Model A: ~55% DA — correct sign 55% of the time
        flip_a = rng.rand(300) < 0.45  # flip sign 45% → 55% correct
        preds_a = np.where(flip_a, -np.sign(y_true), np.sign(y_true)) * np.abs(y_true)
        # Model B: ~75% DA — correct sign 75% of the time
        flip_b = rng.rand(300) < 0.25  # flip sign 25% → 75% correct
        preds_b = np.where(flip_b, -np.sign(y_true), np.sign(y_true)) * np.abs(y_true)

        result = paired_bootstrap_da(y_true, preds_a, preds_b, n_bootstrap=5000, seed=42)
        assert result["delta_da"] > 0, f"B should have higher DA, got delta={result['delta_da']}"
        assert result["ci_low"] > 0, "95% CI should exclude 0 for clear improvement"
        assert result["p_value"] < 0.05, "Should be statistically significant"


# ======================================================================
# Test 5: Full pipeline - train encoder, train CMTF, evaluate metrics
# ======================================================================

class TestFullPipeline:
    def test_end_to_end_metrics_sensible(self):
        """Full pipeline: train LSTM → CMTF → evaluate → metrics make sense."""
        input_dim = 5
        rng = np.random.RandomState(789)

        # Generate data with a learnable signal
        n_train, n_test = 120, 40
        mw_train = rng.randn(n_train, 30, input_dim).astype(np.float32)
        mw_test = rng.randn(n_test, 30, input_dim).astype(np.float32)

        # Target is correlated with last-bar feature 0
        y_train = (mw_train[:, -1, 0] * 0.01 + rng.randn(n_train) * 0.005).astype(np.float32)
        y_test = (mw_test[:, -1, 0] * 0.01 + rng.randn(n_test) * 0.005).astype(np.float32)

        # News with some signal
        ne_train = rng.randn(n_train, 30, 768).astype(np.float32) * 0.01
        ne_test = rng.randn(n_test, 30, 768).astype(np.float32) * 0.01
        ne_train[:, :20, :] = 0  # Most bars no news
        ne_test[:, :20, :] = 0
        nm_train = (ne_train.sum(axis=-1) == 0)
        nm_test = (ne_test.sum(axis=-1) == 0)

        # Step 1: Train encoder
        encoder = LSTMPredictor(
            input_dim=input_dim, hidden_dim=32, num_layers=1,
            dropout=0.0, device="cpu",
        )
        encoder.fit(mw_train, y_train, mw_test, y_test, epochs=10, batch_size=32)

        # Step 2: Market-only baseline
        market_preds = encoder.predict_market_only(mw_test)
        market_metrics = compute_all(y_test, market_preds, horizon=1)

        # Step 3: CMTF prediction
        wrapper = HybridFusionWrapper(
            encoder=encoder, news_dim=768, fusion_dim=32, fusion_market_dim=32,
            n_heads=2, dropout=0.1, seq_len=30, device="cpu",
            use_two_stage=True,
            use_aux_loss=True, use_variance_reg=True,
        )
        wrapper.fit(mw_train, ne_train, y_train, mw_test, ne_test, y_test,
                    news_mask_train=nm_train, news_mask_val=nm_test,
                    batch_size=32, patience=8)
        cmtf_preds = wrapper.predict(mw_test, ne_test, nm_test)
        cmtf_metrics = compute_all(y_test, cmtf_preds, horizon=1)

        # Step 4: Validate metrics are sensible
        print(f"\n  Market-only: DA%={market_metrics['DA%']:.1f}  RMSE={market_metrics['RMSE']:.5f}")
        print(f"  CMTF:        DA%={cmtf_metrics['DA%']:.1f}  RMSE={cmtf_metrics['RMSE']:.5f}")

        # Both should produce finite metrics
        for k, v in cmtf_metrics.items():
            assert np.isfinite(v), f"CMTF metric {k} is not finite: {v}"
        for k, v in market_metrics.items():
            assert np.isfinite(v), f"Market metric {k} is not finite: {v}"

        # DA should be > 0 (not all-wrong)
        assert cmtf_metrics["DA%"] > 0
        assert market_metrics["DA%"] > 0

        # RMSE should be positive and bounded
        assert 0 < cmtf_metrics["RMSE"] < 0.1
        assert 0 < market_metrics["RMSE"] < 0.1

        # Step 5: DM test between market-only and CMTF
        dm_result = diebold_mariano_test(y_test, market_preds, cmtf_preds, horizon=1)
        print(f"  DM stat: {dm_result['DM_stat']:.3f}  p={dm_result['p_value']:.4f}")
        assert np.isfinite(dm_result["DM_stat"])
        assert 0 <= dm_result["p_value"] <= 1

        # Step 6: Bootstrap DA
        bs_result = paired_bootstrap_da(y_test, market_preds, cmtf_preds, n_bootstrap=1000)
        print(f"  Bootstrap delta_DA: {bs_result['delta_da']:.2f}%  "
              f"CI=[{bs_result['ci_low']:.2f}, {bs_result['ci_high']:.2f}]  "
              f"p={bs_result['p_value']:.4f}")
        assert np.isfinite(bs_result["delta_da"])
