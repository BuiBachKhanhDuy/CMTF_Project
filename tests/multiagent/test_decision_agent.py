"""Tests for the Decision Agent — all override combinations."""

import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.decision_agent import decision_node


def _base_state(**overrides):
    """Create a minimal state for decision testing."""
    state = {
        "symbol": "VCB",
        "prediction_time": "2025-03-31",
        "target_horizon_days": 1,
        "sequence_len": 30,
        "node_timings": {},
        "final_pred_adjusted": 0.0,
        "position_scale_regime": 1.0,
        "disagreement_force_flat": False,
    }
    state.update(overrides)
    return state


class TestDecisionAgent:
    """Truth-table: covers every override combination."""

    def test_strong_buy_signal(self):
        """Positive prediction above threshold → long."""
        cfg = MultiAgentConfig(buy_threshold=0.002, sell_threshold=0.002)
        state = _base_state(final_pred_adjusted=0.005, position_scale_regime=1.0)
        result = decision_node(state, config=cfg)
        assert result["action"] == "long"
        assert result["position_scale"] == 1.0

    def test_strong_sell_signal(self):
        """Negative prediction below threshold → short."""
        cfg = MultiAgentConfig(buy_threshold=0.002, sell_threshold=0.002)
        state = _base_state(final_pred_adjusted=-0.005, position_scale_regime=1.0)
        result = decision_node(state, config=cfg)
        assert result["action"] == "short"
        assert result["position_scale"] == 1.0

    def test_neutral_signal(self):
        """Prediction within thresholds → flat."""
        cfg = MultiAgentConfig(buy_threshold=0.002, sell_threshold=0.002)
        state = _base_state(final_pred_adjusted=0.001, position_scale_regime=1.0)
        result = decision_node(state, config=cfg)
        assert result["action"] == "flat"
        assert result["position_scale"] == 0.0

    def test_disagreement_forces_flat(self):
        """Disagreement override → flat regardless of prediction."""
        cfg = MultiAgentConfig(buy_threshold=0.002)
        state = _base_state(
            final_pred_adjusted=0.01,  # Strong buy
            position_scale_regime=1.0,
            disagreement_force_flat=True,
        )
        result = decision_node(state, config=cfg)
        assert result["action"] == "flat"
        assert result["position_scale"] == 0.0

    def test_regime_zero_forces_flat(self):
        """Regime critic forced scale=0 → flat regardless of prediction."""
        cfg = MultiAgentConfig(buy_threshold=0.002)
        state = _base_state(
            final_pred_adjusted=0.01,  # Strong buy
            position_scale_regime=0.0,  # Regime says no
            disagreement_force_flat=False,
        )
        result = decision_node(state, config=cfg)
        assert result["action"] == "flat"
        assert result["position_scale"] == 0.0

    def test_regime_half_scales_position(self):
        """Regime at 0.5 → long with half position."""
        cfg = MultiAgentConfig(buy_threshold=0.002)
        state = _base_state(
            final_pred_adjusted=0.005,
            position_scale_regime=0.5,
            disagreement_force_flat=False,
        )
        result = decision_node(state, config=cfg)
        assert result["action"] == "long"
        assert result["position_scale"] == 0.5

    def test_both_overrides_flat(self):
        """Both disagreement and regime zero → flat."""
        cfg = MultiAgentConfig(buy_threshold=0.002)
        state = _base_state(
            final_pred_adjusted=0.01,
            position_scale_regime=0.0,
            disagreement_force_flat=True,
        )
        result = decision_node(state, config=cfg)
        assert result["action"] == "flat"
        assert result["position_scale"] == 0.0

    def test_exact_threshold_triggers_buy(self):
        """Prediction exactly at buy_threshold → long."""
        cfg = MultiAgentConfig(buy_threshold=0.002)
        state = _base_state(final_pred_adjusted=0.002, position_scale_regime=1.0)
        result = decision_node(state, config=cfg)
        assert result["action"] == "long"

    def test_exact_negative_threshold_triggers_sell(self):
        """Prediction exactly at -sell_threshold → short."""
        cfg = MultiAgentConfig(sell_threshold=0.002)
        state = _base_state(final_pred_adjusted=-0.002, position_scale_regime=1.0)
        result = decision_node(state, config=cfg)
        assert result["action"] == "short"
