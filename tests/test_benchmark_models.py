"""Tests for benchmark models and split logic — synthetic data, no Chronos loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_model_benchmark import (
    extract_per_symbol_data,
    impute_market_window_splits,
    impute_tabular_splits,
    split_by_date,
)


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


class TestImputeMarketWindows:
    def test_uses_train_only_feature_means(self):
        train_windows = np.array(
            [
                [[1.0, np.nan], [3.0, 5.0]],
                [[5.0, 7.0], [7.0, 9.0]],
            ],
            dtype=np.float32,
        )
        val_windows = np.array([[[np.nan, 11.0], [13.0, np.nan]]], dtype=np.float32)
        test_windows = np.array([[[np.nan, np.nan], [19.0, 21.0]]], dtype=np.float32)

        splits = {
            "train": {"market_windows": train_windows, "targets": np.zeros(2)},
            "val": {"market_windows": val_windows, "targets": np.zeros(1)},
            "test": {"market_windows": test_windows, "targets": np.zeros(1)},
        }
        result = impute_market_window_splits(splits)

        # Feature means are computed across train samples and timesteps only.
        assert result["val"]["market_windows"][0, 0, 0] == pytest.approx(4.0)
        assert result["val"]["market_windows"][0, 1, 1] == pytest.approx(7.0)
        assert result["test"]["market_windows"][0, 0, 0] == pytest.approx(4.0)
        assert result["test"]["market_windows"][0, 0, 1] == pytest.approx(7.0)


class TestExtractPerSymbolData:
    def test_extracts_market_windows_and_targets(self):
        times = pd.bdate_range("2024-01-02", periods=6, freq="B")
        df = pd.DataFrame(
            {
                "symbol": ["VCB"] * len(times),
                "time": times,
                "open": np.linspace(10, 15, len(times)),
                "high": np.linspace(11, 16, len(times)),
                "low": np.linspace(9, 14, len(times)),
                "close": np.linspace(10.5, 15.5, len(times)),
                "volume": np.linspace(1000, 1500, len(times)),
                "rsi_14": np.linspace(40, 60, len(times)),
                "news_emb": [np.zeros(768, dtype=np.float32) for _ in range(len(times))],
                "has_news": [False, True, False, True, True, False],
                "fwd_ret_1d": np.linspace(-0.02, 0.03, len(times)),
            }
        )
        dataset = SimpleNamespace(
            df=df,
            market_cols=["open", "high", "low", "close", "volume", "rsi_14"],
        )
        raw_ohlcv = {
            "VCB": pd.DataFrame(
                {"close": df["close"].to_numpy()},
                index=times,
            )
        }

        result = extract_per_symbol_data(dataset, raw_ohlcv, seq_len=3, target_horizon_days=1)

        assert "VCB" in result
        symbol_data = result["VCB"]
        assert symbol_data["close_windows"].shape == (4, 3)
        assert symbol_data["market_windows"].shape == (4, 3, 6)
        assert symbol_data["market_tabular"].shape == (4, 6)
        assert symbol_data["news_masks"].shape == (4, 3)
        assert symbol_data["targets"].shape == (4,)
        assert symbol_data["news_masks"][0].tolist() == [True, False, True]

    def test_extracts_hybrid_news_vectors_when_available(self):
        times = pd.bdate_range("2024-01-02", periods=6, freq="B")
        hybrid_dim = 12
        df = pd.DataFrame(
            {
                "symbol": ["VCB"] * len(times),
                "time": times,
                "open": np.linspace(10, 15, len(times)),
                "high": np.linspace(11, 16, len(times)),
                "low": np.linspace(9, 14, len(times)),
                "close": np.linspace(10.5, 15.5, len(times)),
                "volume": np.linspace(1000, 1500, len(times)),
                "rsi_14": np.linspace(40, 60, len(times)),
                "news_emb": [np.zeros(768, dtype=np.float32) for _ in range(len(times))],
                "news_hybrid_emb": [np.ones(hybrid_dim, dtype=np.float32) for _ in range(len(times))],
                "has_news": [True] * len(times),
                "fwd_ret_1d": np.linspace(-0.02, 0.03, len(times)),
            }
        )
        dataset = SimpleNamespace(
            df=df,
            market_cols=["open", "high", "low", "close", "volume", "rsi_14"],
        )
        raw_ohlcv = {
            "VCB": pd.DataFrame(
                {"close": df["close"].to_numpy()},
                index=times,
            )
        }

        result = extract_per_symbol_data(dataset, raw_ohlcv, seq_len=3, target_horizon_days=1)

        assert result["VCB"]["news_embs"].shape == (4, 3, hybrid_dim)
        assert not result["VCB"]["news_masks"].any()


class TestBaselineHpoFallback:
    def test_returns_defaults_without_running_hpo(self, tmp_path, monkeypatch):
        from src.benchmark import baseline_hpo

        monkeypatch.setattr(
            baseline_hpo,
            "run_lstm_hpo",
            lambda *args, **kwargs: pytest.fail("LSTM HPO should not run"),
        )
        monkeypatch.setattr(
            baseline_hpo,
            "run_rf_hpo",
            lambda *args, **kwargs: pytest.fail("RF HPO should not run"),
        )
        monkeypatch.setattr(
            baseline_hpo,
            "run_finetuned_chronos_hpo",
            lambda *args, **kwargs: pytest.fail("Chronos HPO should not run"),
        )

        params = baseline_hpo.load_or_run_baseline_hpo(
            tmp_path,
            market_windows_train=np.zeros((4, 3, 2), dtype=np.float32),
            targets_train=np.zeros(4, dtype=np.float32),
            market_windows_val=np.zeros((2, 3, 2), dtype=np.float32),
            targets_val=np.zeros(2, dtype=np.float32),
            chronos_predictor=object(),
            close_windows_train=np.zeros((4, 3), dtype=np.float32),
            close_windows_val=np.zeros((2, 3), dtype=np.float32),
            target_h=1,
            fallback_to_defaults=True,
        )

        assert params == baseline_hpo.get_default_baseline_hpo_params()
# ======================================================================
# ResidualNewsFusionHead (pure PyTorch)
# ======================================================================


class TestResidualNewsFusionHead:
    def test_output_shape(self):
        from src.benchmark.cnn_lstm_cmtf import ResidualNewsFusionHead

        head = ResidualNewsFusionHead(
            baseline_dim=32,
            market_dim=32,
            news_dim=16,
            hidden_dim=8,
            n_heads=2,
            dropout=0.0,
            seq_len=5,
        )
        import torch

        market = torch.randn(4, 32)
        news = torch.randn(4, 5, 16)
        pred = head(market, news)
        assert pred.shape == (4,)

    def test_all_masked_news_returns_zero(self):
        """Fully masked news should preserve exact zero residual parity."""
        from src.benchmark.cnn_lstm_cmtf import ResidualNewsFusionHead
        import torch

        head = ResidualNewsFusionHead(
            baseline_dim=32,
            market_dim=32,
            news_dim=16,
            hidden_dim=8,
            n_heads=2,
            dropout=0.0,
            seq_len=3,
        )
        market = torch.randn(2, 32)
        news = torch.zeros(2, 3, 16)
        news_mask = torch.ones(2, 3, dtype=torch.bool)

        pred = head(market, news, news_mask=news_mask)
        assert pred.shape == (2,)
        assert torch.allclose(pred, torch.zeros_like(pred), atol=1e-6, rtol=0.0)

    def test_zero_news_rows_stay_finite(self):
        from src.benchmark.cnn_lstm_cmtf import ResidualNewsFusionHead
        import torch

        head = ResidualNewsFusionHead(
            baseline_dim=32,
            market_dim=32,
            news_dim=16,
            hidden_dim=8,
            n_heads=2,
            dropout=0.0,
            seq_len=3,
        )
        market = torch.randn(2, 32)
        news = torch.zeros(2, 3, 16)
        news[0] = torch.randn(3, 16)

        pred = head(market, news)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
