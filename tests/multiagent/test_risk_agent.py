"""Tests for the Risk Agent — tiered position sizing."""

import pytest

from src.multiagent.agents.risk_agent import (
    risk_agent_node,
    _determine_action,
    _tiered_risk,
)


_POLICY = {
    "version": 3,
    "buy_threshold": 0.012,
    "sell_threshold": -0.012,
    "weak_signal": 0.001,
    "hard_block_vol": 40.0,
    "hard_block_drawdown": 20.0,
    "hard_block_min_confidence": 0.10,
    "reduced_vol": 30.0,
    "reduced_min_confidence": 0.25,
    "min_news_coverage": 0,
    "max_staleness_frac": 1.0,
}


class TestDetermineAction:
    def test_strong_buy(self):
        assert _determine_action(0.02, 0.012, -0.012, 0.001) == "long"

    def test_strong_sell(self):
        assert _determine_action(-0.02, 0.012, -0.012, 0.001) == "short"

    def test_weak_signal_flat(self):
        assert _determine_action(0.0005, 0.012, -0.012, 0.001) == "flat"

    def test_hold_zone_flat(self):
        assert _determine_action(0.008, 0.012, -0.012, 0.001) == "flat"


class TestTieredRisk:
    """Test the three tiers: hard block, reduced, full."""

    # --- Hard block tier ---
    def test_hard_block_high_vol(self):
        r = _tiered_risk(0.03, 0.8, 45.0, 5.0, _POLICY)
        assert r["tier"] == "blocked"
        assert r["action"] == "flat"
        assert r["position_scale"] == 0.0
        assert any("vol" in s for s in r["hard_block_reasons"])

    def test_hard_block_high_drawdown(self):
        r = _tiered_risk(0.03, 0.8, 20.0, 25.0, _POLICY)
        assert r["tier"] == "blocked"
        assert r["action"] == "flat"
        assert r["position_scale"] == 0.0
        assert any("dd" in s for s in r["hard_block_reasons"])

    def test_hard_block_low_confidence(self):
        r = _tiered_risk(0.03, 0.05, 20.0, 5.0, _POLICY)
        assert r["tier"] == "blocked"
        assert r["action"] == "flat"
        assert r["position_scale"] == 0.0
        assert any("conf" in s for s in r["hard_block_reasons"])

    def test_weak_signal_is_flat(self):
        """Weak signal → flat_signal tier, not blocked."""
        r = _tiered_risk(0.0005, 0.8, 20.0, 5.0, _POLICY)
        assert r["tier"] == "flat_signal"
        assert r["action"] == "flat"
        assert r["position_scale"] == 0.0

    # --- Reduced tier ---
    def test_reduced_low_confidence(self):
        """Confidence above hard block but below reduced threshold → reduced."""
        r = _tiered_risk(0.03, 0.15, 20.0, 5.0, _POLICY)
        assert r["tier"] == "reduced"
        assert r["action"] == "long"
        assert 0.3 <= r["position_scale"] <= 0.5

    def test_reduced_moderate_vol(self):
        """Vol above reduced threshold but below hard block → reduced."""
        r = _tiered_risk(0.03, 0.5, 35.0, 5.0, _POLICY)
        assert r["tier"] == "reduced"
        assert r["action"] == "long"
        assert 0.3 <= r["position_scale"] <= 0.5

    def test_reduced_scale_increases_with_confidence(self):
        """Higher confidence within reduced tier → higher scale (closer to 0.5)."""
        r_low = _tiered_risk(0.03, 0.12, 20.0, 5.0, _POLICY)
        r_high = _tiered_risk(0.03, 0.24, 20.0, 5.0, _POLICY)
        assert r_low["tier"] == "reduced"
        assert r_high["tier"] == "reduced"
        assert r_high["position_scale"] > r_low["position_scale"]

    # --- Full tier ---
    def test_full_position_high_confidence(self):
        """High confidence + calm market → full tier, scale ∝ confidence."""
        r = _tiered_risk(0.03, 0.7, 20.0, 5.0, _POLICY)
        assert r["tier"] == "full"
        assert r["action"] == "long"
        assert r["position_scale"] == 0.7

    def test_full_position_short(self):
        r = _tiered_risk(-0.03, 0.6, 15.0, 3.0, _POLICY)
        assert r["tier"] == "full"
        assert r["action"] == "short"
        assert r["position_scale"] == 0.6

    def test_full_scale_capped_at_one(self):
        """Confidence above 1.0 edge case → scale capped at 1.0."""
        r = _tiered_risk(0.03, 1.0, 15.0, 3.0, _POLICY)
        assert r["tier"] == "full"
        assert r["position_scale"] == 1.0

    def test_full_scale_floor_at_03(self):
        """Confidence 0.25 (just at threshold) → scale = max(0.3, 0.25) = 0.3."""
        r = _tiered_risk(0.03, 0.25, 20.0, 5.0, _POLICY)
        assert r["tier"] == "full"
        assert r["position_scale"] == 0.3


class TestRiskAgentNode:
    def _base_state(self, **overrides):
        final_pred = overrides.pop("final_pred", 0.03)
        confidence = overrides.pop("confidence", 0.7)
        vol = overrides.pop("vol_20d", 20.0)
        dd = overrides.pop("max_drawdown_pct", 5.0)
        state = {
            "fusion_decision": {"score": final_pred, "confidence": confidence},
            "volatility_metrics": {"vol_20d": vol, "max_drawdown_pct": dd, "trend_pct": 2.0},
            "node_timings": {},
        }
        state.update(overrides)
        return state

    def test_full_long(self):
        result = risk_agent_node(self._base_state())
        assert result["action"] == "long"
        assert result["position_scale"] == 0.7
        assert result["risk_checks"]["tier"] == "full"
        assert "APPROVED" in result["decision_reasoning"]

    def test_hard_blocked(self):
        result = risk_agent_node(self._base_state(vol_20d=50.0))
        assert result["action"] == "flat"
        assert result["position_scale"] == 0.0
        assert "BLOCKED" in result["decision_reasoning"]

    def test_reduced_position(self):
        result = risk_agent_node(self._base_state(confidence=0.15))
        assert result["action"] == "long"
        assert 0.3 <= result["position_scale"] <= 0.5
        assert "REDUCED" in result["decision_reasoning"]

    def test_short_decision(self):
        result = risk_agent_node(self._base_state(final_pred=-0.03))
        assert result["action"] == "short"

    def test_node_timings(self):
        result = risk_agent_node(self._base_state())
        assert "risk_agent" in result["node_timings"]

    def test_policy_version_returned(self):
        result = risk_agent_node(self._base_state())
        assert result["policy_version"] >= 1
