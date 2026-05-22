"""Tests for the artifact loader with override hooks."""

import pytest

from src.multiagent.loaders import (
    ArtifactMissingError,
    clear_overrides,
    get_cmtf_ensemble,
    get_lora_backbone,
    set_loader_override,
)
from src.multiagent.config import MultiAgentConfig


@pytest.fixture(autouse=True)
def clean_overrides():
    """Ensure overrides are cleared before and after each test."""
    clear_overrides()
    yield
    clear_overrides()


class TestLoaderOverrides:
    """Test the override hook mechanism for testing."""

    def test_set_override_lora_backbone(self):
        fake_backbone = {"backbone": True}
        set_loader_override("lora_backbone_VCB_1d", fake_backbone)
        result = get_lora_backbone("VCB", 1)
        assert result == fake_backbone

    def test_set_override_cmtf_ensemble(self):
        fake_ensemble = [1, 2, 3]
        set_loader_override("cmtf_ensemble_VCB_1d", fake_ensemble)
        result = get_cmtf_ensemble("VCB", 1)
        assert result == fake_ensemble

    def test_clear_overrides_removes_all(self):
        fake = [1, 2, 3]
        set_loader_override("cmtf_ensemble_VCB_1d", fake)
        assert get_cmtf_ensemble("VCB", 1) == fake
        clear_overrides()
        # After clearing, override is gone — real loader would fail on missing artifacts
        with pytest.raises(ArtifactMissingError):
            get_cmtf_ensemble("VCB", 1, MultiAgentConfig(cmtf_models_dir=__import__("pathlib").Path("/nonexistent")))


class TestLoaderMissingArtifacts:
    """Test that missing artifacts raise proper errors."""

    def test_missing_backbone_raises(self, tmp_path):
        cfg = MultiAgentConfig(
            cmtf_models_dir=tmp_path / "nonexistent",
            optuna_dir=tmp_path / "optuna",
        )
        # Create the optuna dir with a params file so we get past that check
        cfg.optuna_dir.mkdir(parents=True, exist_ok=True)
        (cfg.optuna_dir / "best_baseline_params_1d.json").write_text(
            '{"hidden_dim": 128, "dropout": 0.2, "tabular_dim": 23, "market_input_dim": 23, "market_hidden_dim": 64}'
        )
        with pytest.raises(ArtifactMissingError):
            get_lora_backbone("VCB", 1, cfg)

    def test_missing_cmtf_raises(self, tmp_path):
        cfg = MultiAgentConfig(
            cmtf_models_dir=tmp_path / "nonexistent",
            optuna_dir=tmp_path / "optuna",
        )
        cfg.optuna_dir.mkdir(parents=True, exist_ok=True)
        (cfg.optuna_dir / "best_params_v7_1d.json").write_text(
            '{"fusion_dim": 64, "n_heads": 2, "dropout": 0.2}'
        )
        (cfg.optuna_dir / "best_baseline_params_1d.json").write_text(
            '{"hidden_dim": 128, "dropout": 0.2, "tabular_dim": 23, "market_input_dim": 23, "market_hidden_dim": 64}'
        )
        with pytest.raises(ArtifactMissingError):
            get_cmtf_ensemble("VCB", 1, cfg)
