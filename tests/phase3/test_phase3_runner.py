from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_phase3_temporal_fusion import main


class _FakeFetcher:
    def fetch_multi_symbol(self, symbols, start, end, interval="1D", source="KBS"):
        index = pd.date_range("2024-06-01", periods=6, freq="B")
        frame = pd.DataFrame(
            {
                "open": np.linspace(10, 15, len(index)),
                "high": np.linspace(11, 16, len(index)),
                "low": np.linspace(9, 14, len(index)),
                "close": np.linspace(10, 15, len(index)),
                "volume": np.linspace(100, 200, len(index)),
            },
            index=index,
        )
        return {symbol: frame for symbol in symbols}

    def fetch_news_multi_source(self, symbol, start, end):
        return pd.DataFrame(
            {
                "published_date": ["2024-06-03 08:00:00", "2024-06-04 08:00:00"],
                "title": [f"{symbol} news 1", f"{symbol} news 2"],
            }
        )

    def fetch_news(self, symbol, source="VCI"):
        return self.fetch_news_multi_source(symbol, start="", end="")


class _FailingNewsFetcher(_FakeFetcher):
    def fetch_news(self, symbol, source="VCI"):
        raise KeyError("data")


class _FallbackNewsFetcher(_FakeFetcher):
    def fetch_news(self, symbol, source="VCI"):
        if source == "VCI":
            raise KeyError("data")
        return pd.DataFrame(
            {
                "published_date": ["2024-10-01 08:00:00"],
                "title": [f"{symbol} kbs news"],
                "article_id": [f"{symbol}-1"],
            }
        )


class _RecordingFetcher(_FakeFetcher):
    def __init__(self):
        self.market_calls: list[dict[str, str]] = []

    def fetch_multi_symbol(self, symbols, start, end, interval="1D", source="KBS"):
        self.market_calls.append(
            {
                "start": start,
                "end": end,
                "interval": interval,
                "source": source,
            }
        )
        return super().fetch_multi_symbol(symbols, start, end, interval=interval, source=source)


def test_phase3_runner_smoke(tmp_path, monkeypatch):
    fake_market_samples = {
        "VCB": {
            "close_windows": np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64),
            "market_windows": np.asarray(
                [
                    [[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]],
                    [[2.0, 0.5], [3.0, 1.0], [4.0, 1.5]],
                ],
                dtype=np.float32,
            ),
            "last_close": np.asarray([3.0, 4.0], dtype=np.float64),
            "targets": np.asarray([0.1, 0.2], dtype=np.float32),
            "times": pd.to_datetime(["2024-06-03 09:00:00", "2024-10-01 09:00:00"]).to_numpy(),
        }
    }

    monkeypatch.setattr("run_phase3_temporal_fusion.VnstockDataFetcher", _FakeFetcher)
    monkeypatch.setattr(
        "run_phase3_temporal_fusion.extract_market_only_per_symbol_data",
        lambda raw_ohlcv, seq_len, target_horizon_days, train_end: fake_market_samples,
    )

    output_dir = tmp_path / "phase3_outputs"
    main(
        [
            "--symbols",
            "VCB",
            "--horizons",
            "1",
            "--train-end",
            "2024-06-30",
            "--val-end",
            "2024-09-30",
            "--output-dir",
            str(output_dir),
            "--allow-empty-news",
            "--prefer-scraped-news",
        ]
    )

    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "horizon_summary.csv").exists()
    assert (output_dir / "data_overview.csv").exists()
    assert (output_dir / "1d" / "dataset_manifest.json").exists()
    assert (output_dir / "1d" / "phase3_samples.csv").exists()
    assert (output_dir / "figures" / "data_overview.png").exists()

    manifest = json.loads((output_dir / "1d" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2


def test_phase3_runner_continues_when_news_fetch_fails(tmp_path, monkeypatch):
    fake_market_samples = {
        "VCB": {
            "close_windows": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            "market_windows": np.asarray([[[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]]], dtype=np.float32),
            "last_close": np.asarray([3.0], dtype=np.float64),
            "targets": np.asarray([0.1], dtype=np.float32),
            "times": pd.to_datetime(["2024-10-01 09:00:00"]).to_numpy(),
        }
    }

    monkeypatch.setattr("run_phase3_temporal_fusion.VnstockDataFetcher", _FailingNewsFetcher)
    monkeypatch.setattr(
        "run_phase3_temporal_fusion.extract_market_only_per_symbol_data",
        lambda raw_ohlcv, seq_len, target_horizon_days, train_end: fake_market_samples,
    )

    output_dir = tmp_path / "phase3_outputs_empty_news"
    main(
        [
            "--symbols",
            "VCB",
            "--horizons",
            "1",
            "--train-end",
            "2024-06-30",
            "--val-end",
            "2024-09-30",
            "--output-dir",
            str(output_dir),
            "--allow-empty-news",
        ]
    )

    manifest = json.loads((output_dir / "1d" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert manifest["samples_without_news"] == 1


def test_phase3_runner_falls_back_to_kbs_news(tmp_path, monkeypatch):
    fake_market_samples = {
        "VCB": {
            "close_windows": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            "market_windows": np.asarray([[[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]]], dtype=np.float32),
            "last_close": np.asarray([3.0], dtype=np.float64),
            "targets": np.asarray([0.1], dtype=np.float32),
            "times": pd.to_datetime(["2024-10-01 09:00:00"]).to_numpy(),
        }
    }

    monkeypatch.setattr("run_phase3_temporal_fusion.VnstockDataFetcher", _FallbackNewsFetcher)
    monkeypatch.setattr(
        "run_phase3_temporal_fusion.extract_market_only_per_symbol_data",
        lambda raw_ohlcv, seq_len, target_horizon_days, train_end: fake_market_samples,
    )

    output_dir = tmp_path / "phase3_outputs_kbs_fallback"
    main(
        [
            "--symbols",
            "VCB",
            "--horizons",
            "1",
            "--train-end",
            "2024-06-30",
            "--val-end",
            "2024-09-30",
            "--output-dir",
            str(output_dir),
            "--allow-empty-news",
        ]
    )

    manifest = json.loads((output_dir / "1d" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert manifest["samples_without_news"] == 0
    assert manifest["news_article_counts"]["VCB"] == 1


def test_phase3_runner_pads_market_history_and_clips_samples(tmp_path, monkeypatch):
    recording_fetcher = _RecordingFetcher()
    fake_market_samples = {
        "VCB": {
            "close_windows": np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64),
            "market_windows": np.asarray(
                [
                    [[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]],
                    [[2.0, 0.5], [3.0, 1.0], [4.0, 1.5]],
                ],
                dtype=np.float32,
            ),
            "last_close": np.asarray([3.0, 4.0], dtype=np.float64),
            "targets": np.asarray([0.1, 0.2], dtype=np.float32),
            "times": pd.to_datetime(["2024-05-20 09:00:00", "2024-06-03 09:00:00"]).to_numpy(),
        }
    }

    monkeypatch.setattr("run_phase3_temporal_fusion.VnstockDataFetcher", lambda: recording_fetcher)
    monkeypatch.setattr(
        "run_phase3_temporal_fusion.extract_market_only_per_symbol_data",
        lambda raw_ohlcv, seq_len, target_horizon_days, train_end: fake_market_samples,
    )

    output_dir = tmp_path / "phase3_outputs_warmup"
    main(
        [
            "--symbols",
            "VCB",
            "--horizons",
            "1",
            "--start",
            "2024-06-01",
            "--end",
            "2024-06-30",
            "--train-end",
            "2024-06-30",
            "--val-end",
            "2024-09-30",
            "--output-dir",
            str(output_dir),
            "--allow-empty-news",
            "--prefer-scraped-news",
        ]
    )

    manifest = json.loads((output_dir / "1d" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert recording_fetcher.market_calls
    assert pd.Timestamp(recording_fetcher.market_calls[0]["start"]) < pd.Timestamp("2024-06-01")


def test_phase3_runner_fails_when_news_coverage_is_zero(tmp_path, monkeypatch):
    fake_market_samples = {
        "VCB": {
            "close_windows": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            "market_windows": np.asarray([[[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]]], dtype=np.float32),
            "last_close": np.asarray([3.0], dtype=np.float64),
            "targets": np.asarray([0.1], dtype=np.float32),
            "times": pd.to_datetime(["2024-10-01 09:00:00"]).to_numpy(),
        }
    }

    monkeypatch.setattr("run_phase3_temporal_fusion.VnstockDataFetcher", _FailingNewsFetcher)
    monkeypatch.setattr(
        "run_phase3_temporal_fusion.extract_market_only_per_symbol_data",
        lambda raw_ohlcv, seq_len, target_horizon_days, train_end: fake_market_samples,
    )

    output_dir = tmp_path / "phase3_outputs_zero_news_guard"
    try:
        main(
            [
                "--symbols",
                "VCB",
                "--horizons",
                "1",
                "--train-end",
                "2024-06-30",
                "--val-end",
                "2024-09-30",
                "--output-dir",
                str(output_dir),
            ]
        )
    except RuntimeError as exc:
        assert "news coverage" in str(exc)
    else:
        raise AssertionError("Expected Phase 3 runner to fail on zero news coverage")