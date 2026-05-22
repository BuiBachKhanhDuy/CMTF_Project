"""Tests for src.benchmark.metrics — edge cases and known-value checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmark.metrics import (
    compute_all,
    directional_accuracy,
    information_coefficient,
    mae,
    rmse,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
)


class TestMAE:
    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0 / 3.0)

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0


class TestRMSE:
    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        assert rmse(y_true, y_pred) == pytest.approx(expected)

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0


class TestDirectionalAccuracy:
    def test_perfect(self):
        y_true = np.array([0.1, -0.2, 0.3])
        y_pred = np.array([0.5, -0.1, 0.9])
        assert directional_accuracy(y_true, y_pred) == 100.0

    def test_excludes_zeros(self):
        y_true = np.array([0.0, 0.0, 0.1, -0.2])
        y_pred = np.array([0.1, -0.1, 0.5, -0.1])
        # Only 2 non-zero y_true entries, both correct
        assert directional_accuracy(y_true, y_pred) == 100.0

    def test_all_zeros(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.1, -0.1])
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_empty(self):
        assert directional_accuracy(np.array([]), np.array([])) == 0.0


class TestSharpeRatio:
    def test_returns_nan_insufficient_data(self):
        y_true = np.array([0.01])
        y_pred = np.array([0.01])
        result = sharpe_ratio(y_true, y_pred)
        assert math.isnan(result), f"Expected NaN for <3 samples, got {result}"

    def test_returns_nan_zero_std(self):
        y_true = np.array([1.0] * 10)
        y_pred = np.array([1.0] * 10)
        # All strategy returns are identical → std == 0 → NaN
        result = sharpe_ratio(y_true, y_pred)
        assert math.isnan(result), f"Expected NaN for zero-std, got {result}"

    def test_horizon_phase_sampling(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 0.01, 100)
        y_pred = rng.normal(0, 0.01, 100)
        result = sharpe_ratio(y_true, y_pred, horizon=5)
        # Should be a finite float (enough data for phase sampling)
        assert np.isfinite(result)

    def test_horizon_gt1_insufficient_phases(self):
        y_true = np.array([0.01, 0.02, 0.03])
        y_pred = np.array([0.01, 0.01, 0.01])
        result = sharpe_ratio(y_true, y_pred, horizon=5)
        assert math.isnan(result), "Expected NaN when no phase has >=5 samples"

    def test_positive_sharpe_for_perfectly_predicted(self):
        # If we perfectly predict direction, Sharpe should be positive
        rng = np.random.default_rng(42)
        y_true = rng.normal(0.01, 0.005, 100)  # positive bias
        y_pred = y_true.copy()  # perfect prediction, always correct sign
        result = sharpe_ratio(y_true, y_pred)
        assert result > 0.0


class TestIC:
    def test_perfect_rank_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)

    def test_too_few_samples(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        assert information_coefficient(y_true, y_pred) == 0.0


class TestComputeAll:
    def test_keys(self):
        y_true = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.03, -0.01, 0.01, 0.02])
        y_pred = np.array([0.01, -0.01, 0.02, 0.02, -0.02, 0.01, 0.04, -0.02, 0.02, 0.01])
        result = compute_all(y_true, y_pred)
        assert set(result.keys()) == {"MAE", "RMSE", "DA%", "Sharpe", "IC", "Prec", "Rec", "F1"}


class TestMaxDrawdown:
    def test_empty(self):
        assert max_drawdown(np.array([])) == 0.0

    def test_all_positive_returns(self):
        # Monotonically rising wealth → no drawdown
        ret = np.array([0.01, 0.02, 0.01, 0.03])
        assert max_drawdown(ret) == 0.0

    def test_known_drawdown(self):
        # Wealth: 1.0 → 1.1 → 0.88 → 0.968
        ret = np.array([0.10, -0.20, 0.10])
        mdd = max_drawdown(ret)
        assert mdd < 0, "MaxDD should be negative"
        assert mdd == pytest.approx(-0.20, abs=0.001)

    def test_single_loss(self):
        ret = np.array([-0.05])
        assert max_drawdown(ret) == pytest.approx(-0.05, abs=0.001)


class TestCalmarRatio:
    def test_too_few(self):
        assert calmar_ratio(np.array([0.01, 0.02])) == 0.0

    def test_no_drawdown_returns_zero(self):
        # All positive → mdd ≈ 0 → Calmar = 0 (avoid div-by-zero)
        ret = np.array([0.01, 0.01, 0.01, 0.01])
        assert calmar_ratio(ret) == 0.0

    def test_positive_calmar(self):
        # Net positive return with some drawdown
        rng = np.random.default_rng(42)
        ret = rng.normal(0.002, 0.01, 50)
        c = calmar_ratio(ret)
        assert np.isfinite(c)
