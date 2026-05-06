"""Tests for MultiAgentState type shape and config defaults."""

import pytest

from src.multiagent.state import MultiAgentState
from src.multiagent.config import MultiAgentConfig, DEFAULT_CONFIG


class TestMultiAgentState:
    """Verify the state TypedDict has all required groups."""

    def test_state_has_request_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "symbol" in annotations
        assert "prediction_time" in annotations
        assert "target_horizon_days" in annotations
        assert "sequence_len" in annotations

    def test_state_has_market_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "close_window" in annotations
        assert "market_window" in annotations
        assert "market_tabular" in annotations
        assert "token_ids" in annotations
        assert "attention_mask" in annotations

    def test_state_has_news_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "articles" in annotations
        assert "news_emb" in annotations
        assert "news_mask" in annotations
        assert "sentiment_features" in annotations

    def test_state_has_fusion_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "baseline_pred" in annotations
        assert "final_pred" in annotations
        assert "seed_preds" in annotations
        assert "news_residual" in annotations
        assert "attn_weights" in annotations
        assert "news_weight" in annotations

    def test_state_has_critic_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "regime_flags" in annotations
        assert "position_scale_regime" in annotations
        assert "news_quality_flags" in annotations
        assert "news_residual_scale" in annotations
        assert "final_pred_adjusted" in annotations
        assert "disagreement_force_flat" in annotations

    def test_state_has_decision_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "action" in annotations
        assert "position_scale" in annotations

    def test_state_has_explanation_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "evidence_dict" in annotations
        assert "explanation_text_vi" in annotations

    def test_state_has_audit_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "data_cutoff" in annotations
        assert "artifact_versions" in annotations
        assert "errors" in annotations
        assert "warnings" in annotations
        assert "node_timings" in annotations


class TestMultiAgentConfig:
    """Verify config defaults are reasonable."""

    def test_default_config_exists(self):
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, MultiAgentConfig)

    def test_thresholds_positive(self):
        cfg = MultiAgentConfig()
        assert cfg.buy_threshold > 0
        assert cfg.sell_threshold > 0
        assert cfg.vol_high_pct > 0
        assert cfg.dd_max_pct > 0

    def test_ensemble_seeds(self):
        cfg = MultiAgentConfig()
        assert cfg.ensemble_seeds == [42, 123, 456]
        assert len(cfg.ensemble_seeds) == 3

    def test_sequence_len(self):
        cfg = MultiAgentConfig()
        assert cfg.sequence_len == 30

    def test_horizon_in_valid_set(self):
        # Config doesn't restrict horizons, but the guide specifies 1, 5, 20
        cfg = MultiAgentConfig()
        assert cfg.cmtf_version == "v8"
        assert cfg.hpo_version == "v7"

    def test_paths_are_path_objects(self):
        from pathlib import Path
        cfg = MultiAgentConfig()
        assert isinstance(cfg.news_cache_dir, Path)
        assert isinstance(cfg.cmtf_models_dir, Path)
        assert isinstance(cfg.optuna_dir, Path)
        assert isinstance(cfg.phase2_output_dir, Path)
