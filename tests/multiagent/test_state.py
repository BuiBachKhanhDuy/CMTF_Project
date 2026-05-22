"""Tests for MultiAgentState type shape and config defaults."""

import pytest

from src.multiagent.state import MultiAgentState
from src.multiagent.config import MultiAgentConfig, DEFAULT_CONFIG


class TestMultiAgentState:
    """Verify the state TypedDict has all required groups."""

    def test_state_has_request_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "query_text" in annotations
        assert "symbol" in annotations
        assert "prediction_time" in annotations
        assert "target_horizon_days" in annotations
        assert "sequence_len" in annotations

    def test_state_has_request_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "symbol" in annotations
        assert "target_horizon_days" in annotations

    def test_state_has_market_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "close_window" in annotations
        assert "market_window" in annotations
        assert "market_tabular" in annotations
        assert "volatility_metrics" in annotations
        assert "market_proposal" in annotations

    def test_state_has_news_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "articles" in annotations
        assert "news_emb" in annotations
        assert "news_mask" in annotations
        assert "sentiment_metrics" in annotations
        assert "news_proposal" in annotations

    def test_state_has_predict_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "baseline_pred" in annotations
        assert "final_pred" in annotations
        assert "seed_preds" in annotations
        assert "news_residual" in annotations
        assert "attn_weights" in annotations
        assert "news_weight" in annotations
        assert "predict_confidence" in annotations
        assert "model_evidence" in annotations
        assert "model_proposal" in annotations

    def test_state_has_fusion_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "fusion_decision" in annotations

    def test_state_has_risk_decision_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "action" in annotations
        assert "position_scale" in annotations
        assert "final_confidence" in annotations
        assert "risk_checks" in annotations
        assert "decision_reasoning" in annotations

    def test_state_has_answer_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "explanation_text_vi" in annotations

    def test_state_has_audit_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "data_cutoff" in annotations
        assert "artifact_versions" in annotations
        assert "errors" in annotations
        assert "warnings" in annotations
        assert "node_timings" in annotations
        assert "policy_version" in annotations
        assert "decision_id" in annotations

    def test_no_legacy_keys(self):
        annotations = MultiAgentState.__annotations__
        assert "quant_proposal" not in annotations
        assert "risk_proposal" not in annotations
        assert "debate_log" not in annotations
        assert "agent_breakdown" not in annotations
        assert "evidence_dict" not in annotations
        assert "sentiment_features" not in annotations


class TestMultiAgentConfig:
    """Verify config defaults are reasonable."""

    def test_default_config_exists(self):
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, MultiAgentConfig)

    def test_config_ollama(self):
        cfg = MultiAgentConfig()
        assert cfg.ollama_model is not None

    def test_ensemble_seeds(self):
        cfg = MultiAgentConfig()
        assert cfg.ensemble_seeds == [42, 123, 456]
        assert len(cfg.ensemble_seeds) == 3

    def test_sequence_len(self):
        cfg = MultiAgentConfig()
        assert cfg.sequence_len == 30

    def test_evaluation_mode_default_false(self):
        cfg = MultiAgentConfig()
        assert cfg.evaluation_mode is False

    def test_no_skip_llm_reasoning_field(self):
        assert not hasattr(MultiAgentConfig(), "skip_llm_reasoning")

    def test_paths_are_path_objects(self):
        from pathlib import Path
        cfg = MultiAgentConfig()
        assert isinstance(cfg.news_cache_dir, Path)
        assert isinstance(cfg.cmtf_models_dir, Path)
        assert isinstance(cfg.optuna_dir, Path)
        assert isinstance(cfg.phase2_output_dir, Path)
