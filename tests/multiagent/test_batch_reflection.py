"""Tests for batch reflection orchestration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.multiagent.batch_reflection import (
    fetch_realized_returns,
    settle_decisions_from_batch,
    apply_reflection_update,
    batch_reflection,
)
from src.multiagent.reflection import save_policy, DEFAULT_POLICY


class TestFetchRealizedReturns:
    def test_augment_with_realized_returns(self):
        """Add realized return column to batch results."""
        df = pd.DataFrame({
            "cutoff": ["2025-03-31", "2025-04-01"],
            "horizon": [1, 1],
            "action": ["long", "short"],
        })

        mock_ohlcv = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0],
        }, index=pd.date_range("2025-03-31", periods=4))

        with patch("src.pipeline.data_fetcher.VnstockDataFetcher") as MockFetcher:
            mock_fetcher = MagicMock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_ohlcv.return_value = mock_ohlcv

            result = fetch_realized_returns(df, "VCB")

            assert "realized_return" in result.columns
            assert len(result) == 2
            # Check that log returns were computed
            assert all(pd.notna(result["realized_return"]))

    def test_empty_dataframe(self):
        """Handle empty input."""
        df = pd.DataFrame({"cutoff": [], "horizon": [], "action": []})
        result = fetch_realized_returns(df, "VCB")
        assert result.empty
        assert "realized_return" in result.columns
        assert result["realized_return"].isna().all()


class TestSettleDecisions:
    def test_settle_filters_errors_and_flats(self):
        """Remove error and flat decisions from settlement."""
        batch_df = pd.DataFrame({
            "cutoff": ["2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03"],
            "horizon": [1, 1, 1, 1],
            "action": ["long", "flat", "short", "error"],
            "fusion_score": [0.05, 0.001, -0.03, np.nan],
            "policy_version": [1, 1, 1, 1],
        })

        mock_ohlcv = pd.DataFrame({
            "close": np.linspace(100, 103, 4),
        }, index=pd.date_range("2025-03-31", periods=4))

        with patch("src.pipeline.data_fetcher.VnstockDataFetcher") as MockFetcher:
            mock_fetcher = MagicMock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_ohlcv.return_value = mock_ohlcv

            settled = settle_decisions_from_batch(batch_df, "VCB")

            # Should only have long and short with realized returns
            assert all(settled["action"].isin(["long", "short"]))
            assert all(pd.notna(settled["realized_return"]))


class TestApplyReflectionUpdate:
    def test_winning_policy_loosens(self):
        """When win_rate > 50%, policy should loosen confidence requirement."""
        settled = pd.DataFrame({
            "action": ["long"] * 40,
            "fused_score": np.linspace(0.01, 0.05, 40),
            "realized_return": np.linspace(0.02, 0.06, 40),  # All positive = 100% win rate
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "test_policy.json"
            base_policy = dict(DEFAULT_POLICY)
            save_policy(base_policy, policy_path)

            with patch("src.multiagent.batch_reflection.load_policy") as mock_load:
                with patch("src.multiagent.batch_reflection.save_policy") as mock_save:
                    mock_load.return_value = base_policy

                    from src.multiagent.config import MultiAgentConfig
                    cfg = MultiAgentConfig()
                    with patch.object(cfg, "policy_store_path", policy_path):
                        result = apply_reflection_update(settled, "VCB", cfg)

                        assert result["win_rate"] == 1.0
                        assert result["old_version"] < result["new_version"]
                        # Loosening means reduced_min_confidence decreases
                        saved_policy = mock_save.call_args[0][0]
                        assert saved_policy["reduced_min_confidence"] < base_policy["reduced_min_confidence"]

    def test_losing_policy_tightens(self):
        """When win_rate < 50%, policy should tighten confidence requirement."""
        # Create trades where longs go down and shorts go up (all losing)
        settled = pd.DataFrame({
            "action": ["long"] * 20 + ["short"] * 20,
            "fused_score": np.concatenate([np.linspace(0.01, 0.05, 20), np.linspace(-0.01, -0.05, 20)]),
            # Long trades have negative returns (losing), short trades have positive returns (losing)
            "realized_return": np.concatenate([-np.linspace(0.02, 0.06, 20), np.linspace(0.02, 0.06, 20)]),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "test_policy.json"
            base_policy = dict(DEFAULT_POLICY)
            save_policy(base_policy, policy_path)

            with patch("src.multiagent.batch_reflection.load_policy") as mock_load:
                with patch("src.multiagent.batch_reflection.save_policy") as mock_save:
                    mock_load.return_value = base_policy

                    from src.multiagent.config import MultiAgentConfig
                    cfg = MultiAgentConfig()
                    with patch.object(cfg, "policy_store_path", policy_path):
                        result = apply_reflection_update(settled, "VCB", cfg)

                        assert result["win_rate"] < 0.5, f"Expected win_rate < 0.5, got {result['win_rate']}"
                        # Tightening means reduced_min_confidence increases
                        saved_policy = mock_save.call_args[0][0]
                        assert saved_policy["reduced_min_confidence"] > base_policy["reduced_min_confidence"]

    def test_insufficient_samples_skips_update(self):
        """With fewer than min_samples trades, policy is unchanged."""
        settled = pd.DataFrame({
            "action": ["long"] * 5,
            "fused_score": [0.01, 0.02, 0.03, 0.04, 0.05],
            "realized_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "test_policy.json"
            base_policy = dict(DEFAULT_POLICY)
            save_policy(base_policy, policy_path)

            from src.multiagent.config import MultiAgentConfig
            cfg = MultiAgentConfig()

            with patch.object(cfg, "policy_store_path", policy_path):
                result = apply_reflection_update(settled, "VCB", cfg, min_samples=30)

                # Version unchanged because not enough samples
                assert result["old_version"] == result["new_version"]


class TestBatchReflection:
    def test_end_to_end_batch_reflection(self):
        """Full batch reflection: load → settle → update policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            policy_path = Path(tmpdir) / "policy.json"

            # Write batch results
            batch_df = pd.DataFrame({
                "cutoff": ["2025-03-31", "2025-04-01"] * 20,
                "horizon": [1, 1] * 20,
                "action": ["long", "short"] * 20,
                "fusion_score": np.random.uniform(-0.05, 0.05, 40),
                "policy_version": [1] * 40,
            })
            batch_df.to_csv(batch_csv, index=False)

            mock_ohlcv = pd.DataFrame({
                "close": np.linspace(100, 110, 50),
            }, index=pd.date_range("2025-03-31", periods=50))

            base_policy = dict(DEFAULT_POLICY)
            save_policy(base_policy, policy_path)

            from src.multiagent.config import MultiAgentConfig
            cfg = MultiAgentConfig()

            with patch("src.multiagent.batch_reflection.fetch_realized_returns") as mock_fetch:
                with patch.object(cfg, "policy_store_path", policy_path):
                    # Mock realized returns
                    batch_df["realized_return"] = np.random.uniform(-0.05, 0.05, 40)
                    mock_fetch.return_value = batch_df[batch_df["action"].isin(["long", "short"])]

                    result = batch_reflection(batch_csv, "VCB", cfg, min_samples=10)

                    assert result["symbol"] == "VCB"
                    assert "old_version" in result
                    assert "new_version" in result
                    assert result["num_settled"] > 0

    def test_missing_batch_file(self):
        """Raise error if batch CSV not found."""
        with pytest.raises(FileNotFoundError):
            batch_reflection("/nonexistent/batch.csv", "VCB")
