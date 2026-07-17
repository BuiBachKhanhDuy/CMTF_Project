"""Tests for MultiAgentState type shape and config defaults (post-redesign)."""

from pathlib import Path

from src.multiagent.state import MultiAgentState
from src.multiagent.config import MultiAgentConfig, DEFAULT_CONFIG


class TestMultiAgentState:
    """Verify the state TypedDict has all required groups after the redesign."""

    def test_state_has_request_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"query_text", "symbol", "prediction_time",
                "target_horizon_days", "sequence_len"} <= set(ann)

    def test_state_has_routing_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"query_intent", "target_symbols", "aspect_filter", "route_reason"} <= set(ann)

    def test_state_has_market_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"close_window", "market_window", "market_tabular",
                "volatility_metrics", "market_proposal"} <= set(ann)

    def test_state_has_news_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"articles", "news_emb", "news_mask",
                "sentiment_metrics", "news_proposal"} <= set(ann)

    def test_state_has_predict_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"baseline_pred", "final_pred", "gate_pred", "seed_preds",
                "news_residual", "model_evidence"} <= set(ann)

    def test_state_has_gate_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"gated_action", "gate_tau", "gate_coverage",
                "gate_val_score", "gate_reason"} <= set(ann)

    def test_state_has_veto_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"action", "position_scale", "risk_vetoed",
                "veto_reasons", "decision_reasoning"} <= set(ann)

    def test_state_has_narrator_critic_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"answer_text", "critic_status", "critic_findings"} <= set(ann)

    def test_state_has_audit_keys(self):
        ann = MultiAgentState.__annotations__
        assert {"artifact_versions", "errors", "warnings",
                "node_timings", "trace", "decision_id"} <= set(ann)

    def test_deleted_legacy_keys_absent(self):
        """R2: the deleted fusion/tier fields must be gone from the schema."""
        ann = MultiAgentState.__annotations__
        for dead in ("fusion_decision", "adjusted_pred", "mkt_adjusted_pred",
                     "news_adjusted_pred", "final_confidence", "risk_checks",
                     "policy_version"):
            assert dead not in ann, f"legacy state key {dead!r} should be deleted"


class TestMultiAgentConfig:
    def test_default_config_exists(self):
        assert isinstance(DEFAULT_CONFIG, MultiAgentConfig)

    def test_ensemble_seeds(self):
        cfg = MultiAgentConfig()
        assert cfg.ensemble_seeds == [1, 42, 123]

    def test_gate_defaults(self):
        cfg = MultiAgentConfig()
        assert cfg.gate_coverage == 0.25
        assert cfg.use_conviction_sizing is True
        assert cfg.gate_on_raw_seed is False  # ensemble mean is the deployed default

    def test_news_scope_default_matched(self):
        assert MultiAgentConfig().news_scope_default == "matched"

    def test_veto_thresholds(self):
        cfg = MultiAgentConfig()
        assert cfg.hard_block_vol == 40.0
        assert cfg.hard_block_drawdown == 20.0

    def test_evaluation_mode_default_false(self):
        assert MultiAgentConfig().evaluation_mode is False

    def test_deleted_fusion_flags_absent(self):
        """R2: fusion/tier config flags must be gone."""
        cfg = MultiAgentConfig()
        for dead in ("override_alpha", "market_agree_bonus", "market_disagree_penalty",
                     "news_agree_bonus", "news_disagree_penalty", "policy_store_path"):
            assert not hasattr(cfg, dead), f"legacy config flag {dead!r} should be deleted"

    def test_gate_policy_dir_is_path(self):
        assert isinstance(MultiAgentConfig().gate_policy_dir, Path)
