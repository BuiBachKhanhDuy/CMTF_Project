"""Tests for gate_io's honest per-horizon disclosure numbers (val_gated_DA%, etc.)."""

from src.multiagent.gate_io import calibrate_from_cache, load_gate_policy


def _calibrate(tmp_path, horizon):
    policy, meta, out_path = calibrate_from_cache(
        pred_dir="cache/predictions",
        gate_dir=tmp_path,
        horizon=horizon,
        coverage=0.25,
        gate_on_raw_seed=False,
        seed=1,
        cmtf_version="v4",
        backbone_version="v3",
    )
    return policy, meta, out_path


class TestDisclosureNumbers:
    def test_calibrate_from_cache_computes_real_disclosure_numbers(self, tmp_path):
        # Uses the real cached 1D validation predictions already on disk — no mocking
        # of the model/data layer, so this exercises the actual production artifacts.
        policy, meta, out_path = _calibrate(tmp_path, horizon=1)

        assert meta["val_gated_DA%"] is not None
        assert 0.0 <= meta["val_gated_DA%"] <= 100.0
        assert meta["val_base_rate_DA%"] is not None
        assert 0.0 <= meta["val_base_rate_DA%"] <= 100.0
        assert meta["val_gated_coverage"] is not None

    def test_disclosure_numbers_round_trip_through_load(self, tmp_path):
        policy, meta, out_path = _calibrate(tmp_path, horizon=1)

        _, reloaded_payload = load_gate_policy(
            out_path, expect_cmtf_version="v4", expect_backbone_version="v3",
        )
        assert reloaded_payload["val_gated_DA%"] == meta["val_gated_DA%"]
        assert reloaded_payload["val_base_rate_DA%"] == meta["val_base_rate_DA%"]

    def test_disclosure_numbers_differ_across_horizons(self, tmp_path):
        # Different horizons calibrate to different validation books — the whole point
        # of computing this per-horizon instead of hardcoding one horizon's numbers.
        _, meta_1d, _ = _calibrate(tmp_path, horizon=1)
        _, meta_5d, _ = _calibrate(tmp_path, horizon=5)
        assert (
            meta_1d["val_gated_DA%"] != meta_5d["val_gated_DA%"]
            or meta_1d["val_base_rate_DA%"] != meta_5d["val_base_rate_DA%"]
        )

    def test_schema_v1_artifact_is_rejected_as_stale(self, tmp_path):
        # An old artifact missing the disclosure keys must not be silently served.
        import json
        stale_path = tmp_path / "VN_5d.json"
        stale_path.write_text(json.dumps({
            "schema_version": 1,
            "policy": {"tau": 0.01, "conviction": True, "conviction_scale": 0.02,
                       "coverage": 0.25, "val_score": 1.0},
            "cmtf_version": "v4", "backbone_version": "v3",
        }), encoding="utf-8")

        import pytest
        from src.multiagent.gate_io import StalePolicyError
        with pytest.raises(StalePolicyError):
            load_gate_policy(stale_path, expect_cmtf_version="v4", expect_backbone_version="v3")
