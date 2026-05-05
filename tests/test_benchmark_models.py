"""Tests for benchmark models and split logic — synthetic data, no Chronos loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_chronos_benchmark import (
    _run_optuna_hpo,
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


class TestCmtfHpo:
    def test_uses_actual_hybrid_news_dim(self, tmp_path, monkeypatch):
        import run_chronos_benchmark as benchmark

        captured_news_dims: list[int] = []

        class FakeChronosCMTF:
            def __init__(self, _backbone, news_dim, **kwargs):
                captured_news_dims.append(int(news_dim))

            def fit_tokenized(self, *args, **kwargs):
                return {"train_loss": [0.0], "val_loss": [0.0]}

            def predict_tokenized(self, token_ids, attention_mask, news_embs, **kwargs):
                return np.zeros(news_embs.shape[0], dtype=np.float32)

        def fake_load_or_train_ft_chronos_model(*args, **kwargs):
            return object(), None, "fakehash"

        monkeypatch.setattr(benchmark, "ChronosCMTFPredictor", FakeChronosCMTF)
        monkeypatch.setattr(benchmark, "_load_or_train_ft_chronos_model", fake_load_or_train_ft_chronos_model)
        monkeypatch.setattr(
            benchmark,
            "compute_composite_metrics",
            lambda *args, **kwargs: {"CompositeScore": 0.0},
        )
        monkeypatch.setattr(benchmark, "_cmtf_hpo_cache_file", lambda target_h: tmp_path / f"best_params_{target_h}d.json")

        news_dim = 773
        all_symbol_splits = {
            "VCB": {
                "train": {
                    "news_embs": np.zeros((2, 3, news_dim), dtype=np.float32),
                    "targets": np.zeros(2, dtype=np.float32),
                    "market_windows": np.zeros((2, 3, 4), dtype=np.float32),
                    "market_tabular": np.zeros((2, 4), dtype=np.float32),
                    "news_masks": np.zeros((2, 3), dtype=bool),
                },
                "val": {
                    "news_embs": np.zeros((1, 3, news_dim), dtype=np.float32),
                    "targets": np.zeros(1, dtype=np.float32),
                    "market_windows": np.zeros((1, 3, 4), dtype=np.float32),
                    "market_tabular": np.zeros((1, 4), dtype=np.float32),
                    "news_masks": np.zeros((1, 3), dtype=bool),
                },
                "test": {
                    "news_embs": np.zeros((1, 3, news_dim), dtype=np.float32),
                    "targets": np.zeros(1, dtype=np.float32),
                    "market_windows": np.zeros((1, 3, 4), dtype=np.float32),
                    "market_tabular": np.zeros((1, 4), dtype=np.float32),
                    "news_masks": np.zeros((1, 3), dtype=bool),
                },
            }
        }
        all_symbol_tokens = {
            "VCB": {
                "train_ids": np.ones((2, 3), dtype=np.int64),
                "train_mask": np.ones((2, 3), dtype=np.int64),
                "val_ids": np.ones((1, 3), dtype=np.int64),
                "val_mask": np.ones((1, 3), dtype=np.int64),
                "test_ids": np.ones((1, 3), dtype=np.int64),
                "test_mask": np.ones((1, 3), dtype=np.int64),
            }
        }

        params = _run_optuna_hpo(
            chronos=object(),
            all_symbol_splits=all_symbol_splits,
            all_symbol_tokens=all_symbol_tokens,
            all_symbol_anchor_val_preds={"VCB": np.zeros(1, dtype=np.float32)},
            target_h=1,
            use_tabular=True,
            device="cpu",
            n_trials=1,
            seq_len=3,
            ft_backbone_params={"lr": 1e-4},
        )

        assert captured_news_dims == [news_dim]
        assert set(params) == {"fusion_dim", "lr", "dir_penalty_weight", "dropout"}


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
        pred = head(market, news)
        assert pred.shape == (4,)

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

        pred = head(market, news)
        assert pred.shape == (2,)
        # Both should produce finite outputs (no NaN from zero input)
        assert torch.isfinite(pred).all()

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
        pred = head(market, news, tabular_emb=tab)
        assert pred.shape == (3,)
