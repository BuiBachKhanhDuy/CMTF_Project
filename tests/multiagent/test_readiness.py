"""Tests for the deployment-readiness check (src/multiagent/readiness.py)."""

import json

import numpy as np
import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.gate_io import calibrate_from_cache
from src.multiagent.readiness import check_horizon_readiness


def _cfg(gate_dir):
    return MultiAgentConfig(gate_policy_dir=gate_dir, cmtf_version="v4", backbone_version="v3")


def _write_deploy_checkpoint(deploy_dir, horizon, seed):
    deploy_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cmtf_lstm_{horizon}d_seed{seed}"
    (deploy_dir / f"{stem}.pt").write_bytes(b"not a real checkpoint, just a marker")
    (deploy_dir / f"{stem}.meta.json").write_text(
        json.dumps({"horizon": horizon, "seed": seed}), encoding="utf-8",
    )


class TestReadinessSubChecks:
    """Each of the four sub-checks fails independently against a deliberately
    partial fixture — proves the report doesn't just short-circuit on the first
    missing piece."""

    def test_missing_gate_policy_fails_only_that_check(self, tmp_path):
        gate_dir = tmp_path / "gate_policies"
        deploy_dir = tmp_path / "deploy_models"
        for seed in (1, 42, 123):
            _write_deploy_checkpoint(deploy_dir, 5, seed)

        report = check_horizon_readiness(
            5, _cfg(gate_dir), pred_dir="cache/predictions", deploy_dir=deploy_dir,
        )
        assert not report.gate_policy_ok
        assert report.deploy_checkpoints_ok  # independent of the missing gate policy
        assert not report.ready

    def test_missing_deploy_checkpoints_fails_only_that_check(self, tmp_path):
        gate_dir = tmp_path / "gate_policies"
        deploy_dir = tmp_path / "deploy_models"  # never populated
        calibrate_from_cache(
            pred_dir="cache/predictions", gate_dir=gate_dir, horizon=5, coverage=0.25,
            gate_on_raw_seed=False, seed=1, cmtf_version="v4", backbone_version="v3",
        )

        report = check_horizon_readiness(
            5, _cfg(gate_dir), pred_dir="cache/predictions", deploy_dir=deploy_dir,
        )
        assert report.gate_policy_ok
        assert not report.deploy_checkpoints_ok
        assert "missing seed" in report.deploy_checkpoints_detail
        assert not report.ready

    def test_deploy_checkpoint_metadata_mismatch_flagged(self, tmp_path):
        gate_dir = tmp_path / "gate_policies"
        deploy_dir = tmp_path / "deploy_models"
        for seed in (1, 42, 123):
            _write_deploy_checkpoint(deploy_dir, 5, seed)
        # Corrupt one seed's metadata to claim the wrong horizon.
        bad_meta = deploy_dir / "cmtf_lstm_5d_seed1.meta.json"
        bad_meta.write_text(json.dumps({"horizon": 20, "seed": 1}), encoding="utf-8")
        calibrate_from_cache(
            pred_dir="cache/predictions", gate_dir=gate_dir, horizon=5, coverage=0.25,
            gate_on_raw_seed=False, seed=1, cmtf_version="v4", backbone_version="v3",
        )

        report = check_horizon_readiness(
            5, _cfg(gate_dir), pred_dir="cache/predictions", deploy_dir=deploy_dir,
        )
        assert not report.deploy_checkpoints_ok
        assert "inconsistent metadata" in report.deploy_checkpoints_detail

    def test_missing_prediction_index_fails_core_and_matched(self, tmp_path):
        gate_dir = tmp_path / "gate_policies"
        pred_dir = tmp_path / "predictions"  # empty: no test_symbols__Hd.npy etc.
        deploy_dir = tmp_path / "deploy_models"
        for seed in (1, 42, 123):
            _write_deploy_checkpoint(deploy_dir, 5, seed)

        report = check_horizon_readiness(
            5, _cfg(gate_dir), pred_dir=pred_dir, deploy_dir=deploy_dir,
        )
        assert not report.core_predictions_ok
        assert not report.matched_predictions_ok
        assert len(report.problems()) >= 2

    def test_fully_ready_fixture_reports_ready(self, tmp_path):
        gate_dir = tmp_path / "gate_policies"
        deploy_dir = tmp_path / "deploy_models"
        for seed in (1, 42, 123):
            _write_deploy_checkpoint(deploy_dir, 5, seed)
        calibrate_from_cache(
            pred_dir="cache/predictions", gate_dir=gate_dir, horizon=5, coverage=0.25,
            gate_on_raw_seed=False, seed=1, cmtf_version="v4", backbone_version="v3",
        )

        report = check_horizon_readiness(
            5, _cfg(gate_dir), pred_dir="cache/predictions", deploy_dir=deploy_dir,
        )
        assert report.ready
        assert report.problems() == []


class TestReadinessAgainstRealRepo:
    """Integration sanity check against the actual repo state (no fixtures) — 5D is
    fully deployed, so this should always be ready once the recalibration step ran."""

    def test_5d_is_ready_in_real_repo(self):
        report = check_horizon_readiness(5)
        assert report.ready, report.problems()
