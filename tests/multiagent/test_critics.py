"""Tests for the three deterministic critics."""

import numpy as np
import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.critics.regime_critic import regime_critic_node
from src.multiagent.critics.news_quality_critic import news_quality_node
from src.multiagent.critics.disagreement_gate import disagreement_node


def _base_state(**overrides):
    """Create a minimal state for testing."""
    state = {
        "symbol": "VCB",
        "prediction_time": "2025-03-31",
        "target_horizon_days": 1,
        "sequence_len": 30,
        "node_timings": {},
        "warnings": [],
    }
    state.update(overrides)
    return state


class TestRegimeCritic:
    """Truth-table tests for the Risk/Regime critic."""

    def test_normal_market_full_scale(self):
        """Low volatility, no drawdown → scale = 1.0."""
        # Stable uptrending close prices
        close = np.linspace(100, 105, 30).astype(np.float32)
        state = _base_state(close_window=close, market_window=np.zeros((30, 23)))
        cfg = MultiAgentConfig(vol_high_pct=0.03, dd_max_pct=0.08)

        result = regime_critic_node(state, config=cfg)

        assert result["position_scale_regime"] == 1.0
        assert result["regime_flags"]["high_vol"] is False
        assert result["regime_flags"]["drawdown_breach"] is False

    def test_high_vol_half_scale(self):
        """High volatility → scale = 0.5."""
        # Very volatile prices
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(30) * 5).astype(np.float32)
        close = np.abs(close)  # Ensure positive
        state = _base_state(close_window=close, market_window=np.zeros((30, 23)))
        cfg = MultiAgentConfig(vol_high_pct=0.01, dd_max_pct=0.90)  # Low vol threshold to trigger

        result = regime_critic_node(state, config=cfg)

        assert result["position_scale_regime"] == 0.5
        assert result["regime_flags"]["high_vol"] is True

    def test_drawdown_breach_force_flat(self):
        """Large drawdown → scale = 0.0 (force flat)."""
        # Price drops 15%
        close = np.array([100] * 15 + [85] * 15, dtype=np.float32)
        state = _base_state(close_window=close, market_window=np.zeros((30, 23)))
        cfg = MultiAgentConfig(vol_high_pct=0.50, dd_max_pct=0.10)

        result = regime_critic_node(state, config=cfg)

        assert result["position_scale_regime"] == 0.0
        assert result["regime_flags"]["drawdown_breach"] is True

    def test_drawdown_takes_priority_over_vol(self):
        """When both vol and dd breach, dd wins (scale=0)."""
        close = np.array([100] * 15 + [80] * 15, dtype=np.float32)
        state = _base_state(close_window=close, market_window=np.zeros((30, 23)))
        cfg = MultiAgentConfig(vol_high_pct=0.001, dd_max_pct=0.10)

        result = regime_critic_node(state, config=cfg)

        assert result["position_scale_regime"] == 0.0


class TestNewsQualityCritic:
    """Truth-table tests for the News-Quality critic."""

    def test_low_coverage_ignores_news(self):
        """Few bars with news → residual_scale = 0."""
        # Only 2 bars have news (mask is True = no news)
        news_mask = np.ones(30, dtype=bool)
        news_mask[0] = False
        news_mask[1] = False
        state = _base_state(
            articles=[{"title": "test", "published_at": "2025-03-30", "bar_index": 0}],
            news_mask=news_mask,
            news_residual=0.005,
            baseline_pred=0.01,
        )
        cfg = MultiAgentConfig(min_news_bars=3)

        result = news_quality_node(state, config=cfg)

        assert result["news_residual_scale"] == 0.0
        assert result["final_pred_adjusted"] == pytest.approx(0.01)  # baseline only

    def test_stale_news_discounted(self):
        """Most articles are old → residual_scale = 0.5."""
        news_mask = np.zeros(30, dtype=bool)  # All bars have news
        articles = [
            {"title": f"old_{i}", "published_at": "2025-03-01", "bar_index": i}
            for i in range(10)
        ]
        state = _base_state(
            articles=articles,
            news_mask=news_mask,
            news_residual=0.005,
            baseline_pred=0.01,
        )
        cfg = MultiAgentConfig(min_news_bars=3, max_stale_frac=0.5, staleness_days=5)

        result = news_quality_node(state, config=cfg)

        assert result["news_residual_scale"] == 0.5
        expected = 0.01 + 0.5 * 0.005
        assert result["final_pred_adjusted"] == pytest.approx(expected)

    def test_fresh_news_full_trust(self):
        """Recent articles → residual_scale = 1.0."""
        news_mask = np.zeros(30, dtype=bool)
        articles = [
            {"title": f"fresh_{i}", "published_at": "2025-03-30", "bar_index": i}
            for i in range(10)
        ]
        state = _base_state(
            articles=articles,
            news_mask=news_mask,
            news_residual=0.005,
            baseline_pred=0.01,
        )
        cfg = MultiAgentConfig(min_news_bars=3, max_stale_frac=0.7, staleness_days=5)

        result = news_quality_node(state, config=cfg)

        assert result["news_residual_scale"] == 1.0
        assert result["final_pred_adjusted"] == pytest.approx(0.015)

    def test_no_articles_ignores_news(self):
        """No articles at all → residual_scale = 0."""
        news_mask = np.ones(30, dtype=bool)
        state = _base_state(
            articles=[],
            news_mask=news_mask,
            news_residual=0.005,
            baseline_pred=0.01,
        )
        cfg = MultiAgentConfig(min_news_bars=3)

        result = news_quality_node(state, config=cfg)

        assert result["news_residual_scale"] == 0.0


class TestDisagreementGate:
    """Truth-table tests for the Ensemble Disagreement Gate."""

    def test_all_agree_positive(self):
        """All seeds positive → no disagreement."""
        state = _base_state(seed_preds=[0.01, 0.02, 0.015])
        result = disagreement_node(state)
        assert result["disagreement_force_flat"] is False

    def test_all_agree_negative(self):
        """All seeds negative → no disagreement."""
        state = _base_state(seed_preds=[-0.01, -0.02, -0.005])
        result = disagreement_node(state)
        assert result["disagreement_force_flat"] is False

    def test_mixed_signs_force_flat(self):
        """One seed disagrees → force flat."""
        state = _base_state(seed_preds=[0.01, -0.005, 0.02])
        result = disagreement_node(state)
        assert result["disagreement_force_flat"] is True

    def test_all_zero_force_flat(self):
        """All zero → treated as disagreement (ambiguous)."""
        state = _base_state(seed_preds=[0.0, 0.0, 0.0])
        result = disagreement_node(state)
        assert result["disagreement_force_flat"] is True

    def test_two_negative_one_positive(self):
        """Mean is negative but one seed is positive → force flat."""
        state = _base_state(seed_preds=[-0.01, -0.02, 0.005])
        result = disagreement_node(state)
        assert result["disagreement_force_flat"] is True
