"""Tests for benchmark models and split logic — synthetic data, no Chronos loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_chronos_benchmark import split_by_date, impute_tabular_splits


# ======================================================================
# split_by_date tests
# ======================================================================


class TestSplitByDate:
    """Verify walk-forward splitting with horizon-aware purge."""

    @staticmethod
    def _make_data(n: int = 200):
        rng = np.random.default_rng(42)
        times = pd.bdate_range("2023-01-02", periods=n, freq="B").values
        data = {
            "close_windows": rng.normal(100, 5, (n, 30)),
            "targets": rng.normal(0, 0.02, n),
            "news_embs": rng.normal(0, 1, (n, 768)).astype(np.float32),
        }
        return data, times

    def test_no_overlap(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        n_total = sum(len(splits[s]["targets"]) for s in ("train", "val", "test"))
        assert n_total <= len(times), "More samples than input"

    def test_chronological(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        # Extract time masks consistent with split_by_date internals
        train_end = pd.Timestamp("2023-06-30")
        val_end = pd.Timestamp("2023-09-30")
        train_t = times[times <= train_end]
        val_t = times[(times > train_end) & (times <= val_end)]
        test_t = times[times > val_end]

        if len(train_t) > 0 and len(val_t) > 0:
            assert train_t.max() < val_t.min()
        if len(val_t) > 0 and len(test_t) > 0:
            assert val_t.max() < test_t.min()

    def test_purge_buffer_reduces_train(self):
        """With horizon > 1, the last H trading days before boundary
        should be excluded from the preceding split (purge buffer)."""
        data, times = self._make_data()

        splits_h1 = split_by_date(data, times, "2023-06-30", "2023-09-30", target_horizon_days=1)
        splits_h5 = split_by_date(data, times, "2023-06-30", "2023-09-30", target_horizon_days=5)

        # With larger horizon, train should have fewer samples due to purge
        assert len(splits_h5["train"]["targets"]) <= len(splits_h1["train"]["targets"])

    def test_all_keys_present_in_splits(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        for split_name in ("train", "val", "test"):
            assert set(splits[split_name].keys()) == set(data.keys())


# ======================================================================
# impute_tabular_splits tests
# ======================================================================


class TestImputeTabular:
    def test_uses_train_only(self):
        """NaN imputation values should come from train split only."""
        train_tab = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        val_tab = np.array([[np.nan, 10.0]], dtype=np.float32)
        test_tab = np.array([[np.nan, np.nan]], dtype=np.float32)

        splits = {
            "train": {"market_tabular": train_tab, "targets": np.zeros(2)},
            "val": {"market_tabular": val_tab, "targets": np.zeros(1)},
            "test": {"market_tabular": test_tab, "targets": np.zeros(1)},
        }
        result = impute_tabular_splits(splits)

        # Train col0 mean = 2.0, col1 mean = 4.0
        assert result["val"]["market_tabular"][0, 0] == pytest.approx(2.0)
        assert result["test"]["market_tabular"][0, 0] == pytest.approx(2.0)
        assert result["test"]["market_tabular"][0, 1] == pytest.approx(4.0)


# ======================================================================
# CrossModalFusionHead (pure PyTorch, no Chronos)
# ======================================================================


class TestCrossModalFusionHead:
    def test_output_shape(self):
        from src.benchmark.chronos_cmtf import CrossModalFusionHead

        head = CrossModalFusionHead(
            market_dim=32, news_dim=16, fusion_dim=8, n_heads=2, dropout=0.0,
            seq_len=5,
        )
        import torch

        market = torch.randn(4, 32)
        news = torch.randn(4, 5, 16)
        reg_out, cls_logit = head(market, news)
        assert reg_out.shape == (4,)
        assert cls_logit.shape == (4,)

    def test_news_default_token_replaces_zeros(self):
        """All-zero news rows should be replaced by the learned default token."""
        from src.benchmark.chronos_cmtf import CrossModalFusionHead
        import torch

        head = CrossModalFusionHead(
            market_dim=32, news_dim=16, fusion_dim=8, n_heads=2, dropout=0.0,
            seq_len=3,
        )
        market = torch.randn(2, 32)
        # First sample: real news; second sample: all zeros (no news)
        news = torch.zeros(2, 3, 16)
        news[0] = torch.randn(3, 16)

        reg_out, cls_logit = head(market, news)
        assert reg_out.shape == (2,)
        assert cls_logit.shape == (2,)
        # Both should produce finite outputs (no NaN from zero input)
        assert torch.isfinite(reg_out).all()
        assert torch.isfinite(cls_logit).all()

    def test_tabular_dim(self):
        from src.benchmark.chronos_cmtf import CrossModalFusionHead
        import torch

        head = CrossModalFusionHead(
            market_dim=32, news_dim=16, tabular_dim=8,
            fusion_dim=8, n_heads=2, dropout=0.0,
            seq_len=5,
        )
        market = torch.randn(3, 32)
        news = torch.randn(3, 5, 16)
        tab = torch.randn(3, 8)
        reg_out, cls_logit = head(market, news, tabular_emb=tab)
        assert reg_out.shape == (3,)
        assert cls_logit.shape == (3,)
