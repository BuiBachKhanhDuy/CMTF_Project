"""Tests for run_ab_benchmark split metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from run_ab_benchmark import compute_prediction_metrics, compute_risk_metrics, compute_agent_correction_metrics


class TestComputePredictionMetrics:
    """Prediction quality: Baseline vs Final on same actual returns."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.actual = rng.normal(0, 0.02, 30)
        # Baseline: noisy version of actual (mediocre predictor)
        self.baseline = self.actual + rng.normal(0, 0.01, 30)
        # Final: less noisy (better predictor)
        self.final = self.actual + rng.normal(0, 0.005, 30)

    def test_keys_present(self):
        result = compute_prediction_metrics(self.actual, self.baseline, self.final, 1)
        expected = {"Base_MAE", "Base_RMSE", "Base_DA%", "Base_IC", "Base_F1",
                    "Base_Prec", "Base_Rec", "Base_Sharpe",
                    "Final_MAE", "Final_RMSE", "Final_DA%", "Final_IC", "Final_F1",
                    "Final_Prec", "Final_Rec", "Final_Sharpe"}
        assert expected == set(result.keys())

    def test_final_better_mae(self):
        result = compute_prediction_metrics(self.actual, self.baseline, self.final, 1)
        assert result["Final_MAE"] <= result["Base_MAE"], \
            "Final (less noisy) should have lower or equal MAE than Baseline"

    def test_horizons_dont_crash(self):
        for h in [1, 5, 20]:
            result = compute_prediction_metrics(self.actual, self.baseline, self.final, h)
            assert np.isfinite(result["Final_DA%"])


class TestComputeRiskMetrics:
    """Risk management: 3 strategies on same prediction."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 30
        self.actual = rng.normal(0, 0.02, n)
        self.pred = self.actual + rng.normal(0, 0.005, n)
        # Risk-gated: some long, some flat, a few short
        self.actions = np.array(
            ["long"] * 15 + ["flat"] * 10 + ["short"] * 5
        )
        self.scales = np.where(self.actions == "flat", 0.0, rng.uniform(0.3, 1.0, n))

    def test_keys_present(self):
        result = compute_risk_metrics(self.actual, self.pred, self.actions, self.scales, 1)
        for prefix in ["AT", "Thresh", "RG"]:
            for suffix in ["Sharpe", "MaxDD%", "WinRate%", "TradeFreq%", "Calmar", "AnnRet%"]:
                assert f"{prefix}_{suffix}" in result, f"Missing {prefix}_{suffix}"

    def test_always_trade_full_frequency(self):
        result = compute_risk_metrics(self.actual, self.pred, self.actions, self.scales, 1)
        assert result["AT_TradeFreq%"] == 100.0, "Always-trade should have 100% trade freq"

    def test_risk_gated_lower_frequency(self):
        result = compute_risk_metrics(self.actual, self.pred, self.actions, self.scales, 1)
        assert result["RG_TradeFreq%"] < result["AT_TradeFreq%"], \
            "Risk-gated should trade less than always-trade"

    def test_threshold_filters_small_preds(self):
        # With high threshold, many predictions get filtered
        result = compute_risk_metrics(
            self.actual, self.pred, self.actions, self.scales, 1,
            buy_threshold=0.10,  # very high
        )
        assert result["Thresh_TradeFreq%"] < 100.0

    def test_maxdd_non_positive(self):
        result = compute_risk_metrics(self.actual, self.pred, self.actions, self.scales, 1)
        for prefix in ["AT", "Thresh", "RG"]:
            assert result[f"{prefix}_MaxDD%"] <= 0.0, \
                f"{prefix} MaxDD should be ≤ 0 (actual drawdown)"

    def test_horizons(self):
        for h in [1, 5, 20]:
            result = compute_risk_metrics(
                self.actual, self.pred, self.actions, self.scales, h
            )
            assert isinstance(result["AT_Sharpe"], float)


class TestComputeAgentCorrectionMetrics:
    """Agent correction: CMTF vs +Market vs +News vs +Both."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.actual = rng.normal(0, 0.02, 30)
        self.final = self.actual + rng.normal(0, 0.01, 30)
        # Market agent agrees with actual → should improve DA
        self.mkt_adjusted = self.actual + rng.normal(0, 0.007, 30)
        # News agent: neutral
        self.news_adjusted = self.final.copy()
        # Both: closer to actual
        self.adjusted = self.actual + rng.normal(0, 0.005, 30)

    def test_keys_present(self):
        result = compute_agent_correction_metrics(
            self.actual, self.final, self.mkt_adjusted,
            self.news_adjusted, self.adjusted, 1,
        )
        expected = {"CMTF_DA%", "CMTF_IC", "CMTF_MAE",
                    "Mkt_DA%", "Mkt_IC", "Mkt_MAE",
                    "News_DA%", "News_IC", "News_MAE",
                    "Both_DA%", "Both_IC", "Both_MAE"}
        assert expected == set(result.keys())

    def test_better_adjusted_has_lower_mae(self):
        result = compute_agent_correction_metrics(
            self.actual, self.final, self.mkt_adjusted,
            self.news_adjusted, self.adjusted, 1,
        )
        assert result["Both_MAE"] <= result["CMTF_MAE"], \
            "Adjusted (closer to actual) should have lower MAE"

    def test_values_finite(self):
        result = compute_agent_correction_metrics(
            self.actual, self.final, self.mkt_adjusted,
            self.news_adjusted, self.adjusted, 1,
        )
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
