"""Tests for ablation config grid generation with CMTF flags."""

from __future__ import annotations

import pytest

from src.benchmark.ablation_config import AblationConfig, generate_grid


class TestAblationConfigValidity:
    def test_hybrid_rf_invalid(self):
        cfg = AblationConfig(model_name="rf", fusion_type="hybrid",
                             news_scope="matched", sentiment_mode="scalars")
        assert not cfg.is_valid()

    def test_hybrid_lstm_two_stage_valid(self):
        cfg = AblationConfig(model_name="lstm", fusion_type="hybrid",
                             news_scope="matched", sentiment_mode="scalars",
                             use_two_stage=True)
        assert cfg.is_valid()

    def test_cmtf_flags_ignored_for_non_hybrid(self):
        """CMTF flags being non-default should make non-hybrid configs invalid."""
        cfg = AblationConfig(model_name="lstm", fusion_type="late",
                             news_scope="matched", sentiment_mode="scalars",
                             use_two_stage=False)
        assert not cfg.is_valid()

    def test_full_cmtf_baseline_valid(self):
        cfg = AblationConfig(model_name="lstm", fusion_type="hybrid",
                             news_scope="matched", sentiment_mode="scalars",
                             use_two_stage=True,
                             use_aux_loss=True, use_variance_reg=True)
        assert cfg.is_valid()


class TestCellId:
    def test_cell_id_includes_cmtf_flags(self):
        cfg = AblationConfig(model_name="lstm", fusion_type="hybrid",
                             news_scope="matched", sentiment_mode="scalars",
                             use_two_stage=True)
        cid = cfg.cell_id
        assert "ts=1" in cid
        assert "aux=1" in cid
        assert "vreg=1" in cid

    def test_different_flags_different_cell_ids(self):
        base = AblationConfig(model_name="lstm", fusion_type="hybrid",
                              news_scope="matched", sentiment_mode="scalars")
        no_aux = AblationConfig(model_name="lstm", fusion_type="hybrid",
                                news_scope="matched", sentiment_mode="scalars",
                                use_aux_loss=False)
        assert base.cell_id != no_aux.cell_id


class TestGenerateGrid:
    def test_component_table_excludes_rf(self):
        """Component table should exclude rf (no latent space for cross-attention)."""
        grid = generate_grid("component")
        models = {cfg.model_name for cfg in grid}
        assert "rf" not in models
        assert "lstm" in models
        assert "cnn_lstm" in models

    def test_component_table_has_cmtf_toggles(self):
        grid = generate_grid("component")
        # Should have single-stage toggle
        single_stage = [c for c in grid if not c.use_two_stage]
        assert len(single_stage) > 0
        # Should have no-aux toggle
        no_aux = [c for c in grid if not c.use_aux_loss]
        assert len(no_aux) > 0
        # Should have no-vreg toggle
        no_vreg = [c for c in grid if not c.use_variance_reg]
        assert len(no_vreg) > 0

    def test_component_grid_size(self):
        """10 toggle combos × 2 models (lstm, cnn_lstm) = 20 cells.

        The 10th combo is the pruned CMTF: PE=N + single-stage (ts=False),
        which combines the two individually-beneficial ablations from the analysis.
        """
        grid = generate_grid("component")
        assert len(grid) == 20

    def test_no_duplicates(self):
        grid = generate_grid("all")
        cell_ids = [cfg.cell_id for cfg in grid]
        assert len(cell_ids) == len(set(cell_ids))
