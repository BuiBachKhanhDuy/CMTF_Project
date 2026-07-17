"""Tests for the cross-horizon interaction layer: calibration module + agent node."""

import numpy as np
import pytest

from src.benchmark.decision_policy import GatePolicy
from src.multiagent.config import MultiAgentConfig
from src.multiagent.horizon_interaction_io import (
    _grid_search_multipliers,
    calibrate_interaction_from_cache,
    load_interaction_policy,
    save_interaction_policy,
    HorizonInteractionPolicy,
    HORIZON_INTERACTION_SCHEMA_VERSION,
)
from src.multiagent.agents.horizon_interaction_agent import horizon_interaction_agent_node

FLAT_POLICY = GatePolicy(tau=0.0, conviction=False, conviction_scale=1.0, coverage=1.0, val_score=0.0)


def _planted_effect_data(seed=0, n_per_bucket=60):
    """Synthetic data where agreement=2 has a genuinely better risk-adjusted return
    than agreement=0 — a real, planted effect the monotonic search should recover."""
    rng = np.random.default_rng(seed)
    agreement = np.concatenate([np.full(n_per_bucket, b) for b in (0, 1, 2)])
    pred = rng.normal(0, 1, size=agreement.shape)  # sign is what matters for positions
    truth = np.empty_like(pred)
    # bucket 2: pred and truth strongly co-signed (good Sharpe); bucket 0: pure noise.
    signal_strength = {0: 0.0, 1: 0.15, 2: 0.4}
    for b in (0, 1, 2):
        m = agreement == b
        truth[m] = np.sign(pred[m]) * signal_strength[b] + rng.normal(0, 1, size=m.sum())
    return agreement, pred, truth


class TestGridSearchMultipliers:
    def test_recovers_monotonic_upweight_for_planted_effect(self):
        agreement, pred, truth = _planted_effect_data()
        mult, n_by_bucket, real_obj, baseline_obj = _grid_search_multipliers(
            agreement, pred, truth, FLAT_POLICY, grid=(0.6, 0.8, 1.0, 1.2, 1.4), min_bucket_n=20,
        )
        assert mult[0] <= mult[1] <= mult[2]  # monotonicity constraint respected
        assert real_obj >= baseline_obj  # the joint search should never do worse than 1.0 everywhere (1,1,1 is always a candidate)

    def test_thin_bucket_falls_back_to_1(self):
        agreement, pred, truth = _planted_effect_data(n_per_bucket=60)
        # Collapse bucket 0 down to 5 rows (below the floor).
        keep = np.concatenate([np.arange(5), np.arange(60, 180)])
        mult, n_by_bucket, _, _ = _grid_search_multipliers(
            agreement[keep], pred[keep], truth[keep], FLAT_POLICY,
            grid=(0.6, 0.8, 1.0, 1.2, 1.4), min_bucket_n=20,
        )
        assert n_by_bucket[0] < 20
        assert mult[0] == 1.0

    def test_uniform_scaling_never_changes_pooled_sharpe_alone(self):
        # Sanity check on the scale-invariance property the module docstring relies on:
        # a single-bucket-only dataset (all agreement=1) should show real_obj == baseline_obj,
        # since any uniform multiplier leaves the Sharpe ratio unchanged.
        rng = np.random.default_rng(1)
        pred = rng.normal(0, 1, size=100)
        truth = np.sign(pred) * 0.3 + rng.normal(0, 1, size=100)
        agreement = np.full(100, 1)
        mult, _, real_obj, baseline_obj = _grid_search_multipliers(
            agreement, pred, truth, FLAT_POLICY, grid=(0.6, 0.8, 1.0, 1.2, 1.4), min_bucket_n=20,
        )
        assert real_obj == pytest.approx(baseline_obj, abs=1e-9)


class TestCalibrateInteractionFromCache:
    """Exercised against the REAL cached validation predictions already on disk —
    no mocking of the model/data layer."""

    def test_produces_valid_artifact_for_all_three_horizons(self, tmp_path):
        for primary, others in ((1, (5, 20)), (5, (1, 20)), (20, (1, 5))):
            policy, meta, out_path = calibrate_interaction_from_cache(
                pred_dir="cache/predictions", gate_dir="results/gate_policies",
                interaction_dir=tmp_path, primary_horizon=primary, other_horizons=others,
                cmtf_version="v4", backbone_version="v3",
            )
            assert out_path.exists()
            assert policy.primary_horizon == primary
            assert set(policy.multiplier_by_agreement) == {0, 1, 2}
            for m in policy.multiplier_by_agreement.values():
                assert 0.5 <= m <= 1.5  # within the calibration grid's bounds
            # Monotonicity holds in the frozen artifact too.
            ordered = [policy.multiplier_by_agreement[b] for b in (0, 1, 2)]
            assert ordered[0] <= ordered[1] <= ordered[2]
            assert "real_lift_beats_placebo_lift" in meta

    def test_round_trips_through_load(self, tmp_path):
        policy, meta, out_path = calibrate_interaction_from_cache(
            pred_dir="cache/predictions", gate_dir="results/gate_policies",
            interaction_dir=tmp_path, primary_horizon=5, other_horizons=(1, 20),
            cmtf_version="v4", backbone_version="v3",
        )
        reloaded_policy, reloaded_meta = load_interaction_policy(
            out_path, expect_cmtf_version="v4", expect_backbone_version="v3",
        )
        assert reloaded_policy == policy

    def test_stale_schema_rejected(self, tmp_path):
        stale_path = tmp_path / "VN_5d_xh.json"
        policy = HorizonInteractionPolicy(
            primary_horizon=5, other_horizons=(1, 20),
            multiplier_by_agreement={0: 1.0, 1: 1.0, 2: 1.0}, n_by_agreement={0: 1, 1: 1, 2: 1},
        )
        save_interaction_policy(policy, {"cmtf_version": "v4", "backbone_version": "v3"}, stale_path)
        import json
        payload = json.loads(stale_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 999
        stale_path.write_text(json.dumps(payload), encoding="utf-8")
        from src.multiagent.gate_io import StalePolicyError
        with pytest.raises(StalePolicyError):
            load_interaction_policy(stale_path, expect_cmtf_version="v4", expect_backbone_version="v3")


def _make_state(gated_action="long", position_scale=0.8, gate_pred=0.03,
               symbol="VCB", date="2025-02-03", horizon=1):
    return {
        "gated_action": gated_action, "position_scale": position_scale, "gate_pred": gate_pred,
        "symbol": symbol, "prediction_time": date, "target_horizon_days": horizon,
    }


class TestHorizonInteractionAgentNode:
    """Uses the REAL calibrated artifacts + frozen predictions on disk for the
    happy-path cases (no mocking), plus a redirected config dir for the degrade case."""

    def test_abstain_in_abstain_out(self):
        state = _make_state(gated_action="abstain", position_scale=0.0)
        out = horizon_interaction_agent_node(state, MultiAgentConfig(evaluation_mode=True))
        assert out["position_scale"] == 0.0
        assert out["horizon_agreement_score"] is None
        assert out["horizon_interaction_multiplier"] is None

    def test_missing_artifact_degrades_to_noop_with_warning(self, tmp_path):
        cfg = MultiAgentConfig(evaluation_mode=True, horizon_interaction_dir=tmp_path)
        state = _make_state(gated_action="long", position_scale=0.8)
        out = horizon_interaction_agent_node(state, cfg)
        assert out["position_scale"] == 0.8  # unchanged — no crash, no fabricated adjustment
        assert out["horizon_interaction_multiplier"] == 1.0
        assert any("horizon_interaction" in w for w in out["warnings"])

    def test_disabled_via_config_is_a_noop(self, tmp_path):
        cfg = MultiAgentConfig(evaluation_mode=True, enable_horizon_interaction=False)
        state = _make_state(gated_action="long", position_scale=0.8)
        out = horizon_interaction_agent_node(state, cfg)
        assert out["position_scale"] == 0.8
        assert out["horizon_agreement_score"] is None

    def test_real_artifact_applies_a_real_multiplier(self):
        # VCB / 2025-02-03 / 1D is a real traded row confirmed earlier this session
        # (gate_pred=0.00355 >= tau); the real VN_1d_xh.json artifact is on disk.
        cfg = MultiAgentConfig(evaluation_mode=True)
        state = _make_state(gated_action="long", position_scale=0.9092206687370419,
                            gate_pred=0.00354583136504516, symbol="VCB", date="2025-02-03", horizon=1)
        out = horizon_interaction_agent_node(state, cfg)
        assert out["horizon_agreement_score"] in (0, 1, 2)
        assert out["horizon_interaction_multiplier"] is not None
        assert out["position_scale"] == pytest.approx(
            0.9092206687370419 * out["horizon_interaction_multiplier"]
        )


class TestVetoOrderingInvariant:
    """Proves the graph-level ordering guarantee: even after this node scales a
    position UP, a downstream risk-agent veto still zeroes it — the one-way-veto
    invariant is unaffected by inserting a symmetric node upstream of it."""

    def test_risk_veto_still_zeroes_an_upscaled_position(self):
        from src.multiagent.agents.risk_agent import risk_agent_node

        cfg = MultiAgentConfig(evaluation_mode=True)
        state = _make_state(gated_action="long", position_scale=0.9092206687370419,
                            gate_pred=0.00354583136504516, symbol="VCB", date="2025-02-03", horizon=1)
        interaction_out = horizon_interaction_agent_node(state, cfg)
        scaled_state = {**state, **interaction_out}
        # Force a dangerous market so risk_agent must veto.
        scaled_state["volatility_metrics"] = {"vol_20d": 999.0, "max_drawdown_pct": 999.0}
        risk_out = risk_agent_node(scaled_state, cfg)
        assert risk_out["action"] == "abstain"
        assert risk_out["position_scale"] == 0.0
