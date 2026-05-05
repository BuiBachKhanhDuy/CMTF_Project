"""Tests for the retained Phase 1 metric helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.phase1.metrics import (
    compute_all,
    directional_accuracy,
    information_coefficient,
    mae,
    rmse,
    sharpe_ratio,
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
        y_true = np.array([0.01, 0.02])
        y_pred = np.array([0.01, 0.01])
        result = sharpe_ratio(y_true, y_pred)
        assert math.isnan(result), f"Expected NaN for <5 samples, got {result}"

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
