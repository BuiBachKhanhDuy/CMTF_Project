"""Tests for raw_prediction.py — always-live prediction fetch + attention summary."""

import numpy as np
import pytest

from src.multiagent.config import MultiAgentConfig
from src.multiagent.loaders import ArtifactMissingError
from src.multiagent.raw_prediction import fetch_prediction_record, summarize_attention


class TestSummarizeAttention:
    def test_none_input_returns_none(self):
        assert summarize_attention(None) is None

    def test_empty_input_returns_none(self):
        assert summarize_attention(np.array([])) is None

    def test_top_k_ordered_by_weight_descending(self):
        attn = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
        out = summarize_attention(attn, top_k=3)
        assert [d["weight"] for d in out] == sorted((d["weight"] for d in out), reverse=True)
        assert out[0]["weight"] == pytest.approx(0.5)

    def test_days_before_cutoff_zero_for_most_recent(self):
        attn = np.array([0.1, 0.2, 0.9])  # index 2 (last) = most recent = cutoff day
        out = summarize_attention(attn, top_k=1)
        assert out[0]["days_before_cutoff"] == 0

    def test_days_before_cutoff_correct_for_oldest(self):
        attn = np.array([0.9, 0.1, 0.1])  # index 0 (first) = oldest trailing day
        out = summarize_attention(attn, top_k=1)
        assert out[0]["days_before_cutoff"] == 2  # seq_len(3) - 1 - 0


class TestFetchPredictionRecordAlwaysLive:
    """Exercised against the real deployed 1D champion — no mocking of the model."""

    def test_always_live_inference_with_real_attention(self):
        cfg = MultiAgentConfig(evaluation_mode=True)
        rec = fetch_prediction_record("VCB", "2025-02-03", 1, cfg)
        assert rec.source == "live_inference"
        # Matches the bit-for-bit value confirmed earlier against the frozen cache.
        assert rec.gate_pred == pytest.approx(0.0035458310740068555, abs=1e-9)
        assert rec.attn_weights is not None
        assert rec.attention_top_days is not None
        assert len(rec.attention_top_days) == 3
        assert all(d["days_before_cutoff"] >= 0 for d in rec.attention_top_days)

    def test_disabled_live_inference_raises(self):
        cfg = MultiAgentConfig(evaluation_mode=True, enable_live_inference=False)
        with pytest.raises(ArtifactMissingError):
            fetch_prediction_record("VCB", "2025-02-03", 1, cfg)
