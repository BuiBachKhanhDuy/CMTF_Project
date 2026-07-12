"""Tests for ablation config grid generation (fusion_comparison + component_ablation)."""


from __future__ import annotations

from src.benchmark.ablation_config import (
    AblationConfig,
    generate_grid,
    CMTF_CORE,
    CMTF_MODEL,
    BACKBONE_MODELS,
)


class TestAblationConfigValidity:
    def test_cmtf_must_use_cmtf_fusion(self):
        cfg = AblationConfig(model_name=CMTF_MODEL, fusion_type="late",
                             news_scope="matched", sentiment_mode="scalars")
        assert not cfg.is_valid()

    def test_backbone_cannot_use_cmtf_fusion(self):
        cfg = AblationConfig(model_name="lstm", fusion_type="cmtf",
                             news_scope="matched", sentiment_mode="scalars")
        assert not cfg.is_valid()

    def test_cmtf_core_is_valid(self):
        cfg = AblationConfig(model_name=CMTF_MODEL, market_encoder_name="lstm", **CMTF_CORE)
        assert cfg.is_valid()

    def test_cmtf_flags_ignored_for_non_cmtf(self):
        """Non-default CMTF-only flags should make a non-cmtf config invalid."""
        cfg = AblationConfig(model_name="lstm", fusion_type="late",
                             news_scope="matched", sentiment_mode="scalars",
                             use_two_stage=False)
        assert not cfg.is_valid()


class TestCellId:
    def test_cell_id_includes_cmtf_flags(self):
        cfg = AblationConfig(model_name=CMTF_MODEL, market_encoder_name="lstm",
                             fusion_type="cmtf", news_scope="matched",
                             sentiment_mode="scalars", use_two_stage=True)
        cid = cfg.cell_id
        assert "ts=1" in cid
        assert "aux=1" in cid
        assert "vreg=1" in cid

    def test_different_flags_different_cell_ids(self):
        base = AblationConfig(model_name=CMTF_MODEL, market_encoder_name="lstm",
                              fusion_type="cmtf", news_scope="matched",
                              sentiment_mode="scalars")
        no_aux = AblationConfig(model_name=CMTF_MODEL, market_encoder_name="lstm",
                                fusion_type="cmtf", news_scope="matched",
                                sentiment_mode="scalars", use_aux_loss=False)
        assert base.cell_id != no_aux.cell_id


class TestFusionComparison:
    def test_excludes_rf(self):
        grid = generate_grid("fusion_comparison")
        models = {c.model_name for c in grid} | {
            c.market_encoder_name for c in grid if c.market_encoder_name
        }
        assert "rf" not in models
        assert "rf" not in BACKBONE_MODELS

    def test_has_full_fusion_ladder_for_trainable_encoders(self):
        grid = generate_grid("fusion_comparison")
        for bb in ("lstm", "cnn_lstm"):
            fusions = {c.fusion_type for c in grid if c.model_name == bb}
            assert {"none", "early", "late"} <= fusions
            assert any(c.fusion_type == "cmtf" and c.market_encoder_name == bb for c in grid)

    def test_foundation_backbones_have_no_early(self):
        grid = generate_grid("fusion_comparison")
        for bb in ("gpt4ts", "chronos"):
            fusions = {c.fusion_type for c in grid if c.model_name == bb}
            assert "early" not in fusions
            assert {"none", "late"} <= fusions

    def test_has_placebo(self):
        grid = generate_grid("fusion_comparison")
        assert any(c.fusion_type == "cmtf" and c.shuffle_news for c in grid)

    def test_size(self):
        # lstm(4) + cnn_lstm(4) + gpt4ts(3) + chronos(3) + placebo(1) = 15
        assert len(generate_grid("fusion_comparison")) == 15


class TestComponentAblation:
    def test_all_cmtf_lstm(self):
        grid = generate_grid("component_ablation")
        assert all(c.fusion_type == "cmtf" for c in grid)
        assert all(c.market_encoder_name == "lstm" for c in grid)

    def test_has_component_knockouts(self):
        grid = generate_grid("component_ablation")
        assert any(not c.use_cross_attention for c in grid)
        assert any(c.recency_gate_k == 0 for c in grid)
        assert any(not c.use_news_gate for c in grid)
        assert any(not c.use_positional_encoding for c in grid)
        assert any(not c.use_aux_loss for c in grid)
        assert any(not c.use_variance_reg for c in grid)

    def test_has_all_output_modes(self):
        grid = generate_grid("component_ablation")
        modes = {c.output_mode for c in grid}
        assert modes == {
            "anchored_fusion", "encoder_residual",
            "fusion_plus_news", "market_plus_fusion",
        }

    def test_has_two_stage_variant(self):
        grid = generate_grid("component_ablation")
        assert any(c.use_two_stage for c in grid)
        assert any(not c.use_two_stage for c in grid)

    def test_core_default_is_anchored_fusion_single_stage(self):
        assert CMTF_CORE["output_mode"] == "anchored_fusion"
        assert CMTF_CORE["use_two_stage"] is False
        # news gate is load-bearing for genuine dominance (see sweep findings)
        assert CMTF_CORE["news_gate_alpha"] == 1.0


class TestGenerateGrid:
    def test_all_cells_valid(self):
        for table in ("fusion_comparison", "component_ablation", "all"):
            assert all(c.is_valid() for c in generate_grid(table))

    def test_no_duplicates(self):
        for table in ("fusion_comparison", "component_ablation", "all"):
            cell_ids = [c.cell_id for c in generate_grid(table)]
            assert len(cell_ids) == len(set(cell_ids))

    def test_all_cmtf_cells_share_core_recipe(self):
        """Every CMTF cell shares CMTF_CORE's training recipe (enables cache sharing)."""
        recipe_keys = [
            "market_epochs", "fusion_epochs", "market_patience", "fusion_patience",
            "encoder_lr_scale", "dropout", "n_heads",
        ]
        for c in (c for c in generate_grid("all") if c.fusion_type == "cmtf"):
            for k in recipe_keys:
                assert getattr(c, k) == CMTF_CORE[k]

    def test_unknown_table_is_empty(self):
        assert generate_grid("does_not_exist") == []
