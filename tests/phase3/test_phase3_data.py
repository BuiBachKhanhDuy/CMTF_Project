from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase3 import Phase3DatasetConfig, build_phase3_dataset, normalize_news_frame


def _make_market_samples() -> dict[str, dict[str, np.ndarray]]:
    times = pd.to_datetime(
        [
            "2024-06-24 09:00:00",
            "2024-06-25 09:00:00",
            "2024-07-01 09:00:00",
            "2024-07-05 09:00:00",
            "2024-10-01 09:00:00",
        ]
    ).to_numpy()
    close_windows = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)
    market_windows = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    return {
        "VCB": {
            "close_windows": close_windows,
            "market_windows": market_windows,
            "last_close": np.asarray([10, 11, 12, 13, 14], dtype=np.float64),
            "targets": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            "times": times,
        }
    }


def test_normalize_news_frame_keeps_titles_and_timestamps():
    news_df = pd.DataFrame(
        {
            "published_date": ["2024-06-24 08:00:00", "2024-06-24 08:30:00"],
            "title": ["Tin 1", "Tin 2"],
        }
    )

    normalized = normalize_news_frame(news_df, symbol="VCB")

    assert list(normalized.columns) == ["article_id", "symbol", "published_at", "title_raw"]
    assert normalized["symbol"].tolist() == ["VCB", "VCB"]
    assert normalized["title_raw"].tolist() == ["Tin 1", "Tin 2"]


def test_build_phase3_dataset_excludes_future_news_and_assigns_splits():
    market_samples = _make_market_samples()
    news_df = pd.DataFrame(
        {
            "published_date": [
                "2024-06-17 08:00:00",
                "2024-06-24 08:30:00",
                "2024-06-24 10:00:00",
                "2024-06-26 08:30:00",
                "2024-07-04 08:30:00",
                "2024-10-01 08:00:00",
                "2024-10-01 10:00:00",
            ],
            "title": [
                "Qua cua so 5 ngay",
                "Hop le truoc cat off",
                "Sau cut off phai loai",
                "Tin giua tuan",
                "Tin gan test",
                "Tin truoc test cutoff",
                "Tin sau test cutoff",
            ],
        }
    )

    bundle = build_phase3_dataset(
        market_samples,
        {"VCB": news_df},
        Phase3DatasetConfig(
            train_end="2024-06-30",
            val_end="2024-09-30",
            target_horizon_days=1,
            trailing_window_days=5,
        ),
    )

    frame = bundle.dataframe
    assert frame["split"].tolist() == ["train", "val", "test"]

    first_sample = frame.iloc[0]
    assert first_sample["news_titles_raw"] == ["Hop le truoc cat off"]

    second_sample = frame.iloc[1]
    assert second_sample["news_titles_raw"] == ["Tin giua tuan"]

    test_sample = frame.iloc[2]
    assert test_sample["news_titles_raw"] == ["Tin truoc test cutoff"]

    assert bundle.manifest["purged_samples"] == 2
    assert bundle.manifest["samples_without_news"] == 0
    assert bundle.manifest["news_coverage_ratio"] == 1.0


def test_build_phase3_dataset_marks_zero_news_samples():
    market_samples = _make_market_samples()

    bundle = build_phase3_dataset(
        market_samples,
        {"VCB": pd.DataFrame(columns=["published_date", "title"])},
        Phase3DatasetConfig(
            train_end="2024-06-30",
            val_end="2024-09-30",
            target_horizon_days=1,
            trailing_window_days=5,
        ),
    )

    assert bundle.dataframe["has_news"].eq(False).all()
    assert bundle.dataframe["news_count"].eq(0).all()
    assert bundle.manifest["samples_without_news"] == len(bundle.dataframe)
    assert bundle.manifest["news_coverage_ratio"] == 0.0


def test_build_phase3_dataset_handles_empty_market_samples():
    market_samples = {
        "VCB": {
            "close_windows": np.empty((0, 3), dtype=np.float64),
            "market_windows": np.empty((0, 3, 2), dtype=np.float32),
            "last_close": np.empty((0,), dtype=np.float64),
            "targets": np.empty((0,), dtype=np.float32),
            "times": pd.to_datetime([]).to_numpy(),
        }
    }

    bundle = build_phase3_dataset(
        market_samples,
        {"VCB": pd.DataFrame(columns=["published_date", "title"])},
        Phase3DatasetConfig(
            train_end="2024-06-30",
            val_end="2024-09-30",
            target_horizon_days=1,
            trailing_window_days=5,
        ),
    )

    assert bundle.dataframe.empty
    assert bundle.manifest["row_count"] == 0
    assert bundle.manifest["purged_samples"] == 0