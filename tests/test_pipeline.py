"""Unit tests for the CMTF data pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.pipeline.temporal_aligner import TemporalAligner
from src.pipeline.news_encoder import NewsEncoder
from src.pipeline.dataset_builder import CMTFDataset


# ======================================================================
# Fixtures
# ======================================================================

def _make_daily_ohlcv(n_days: int = 20, start: str = "2024-01-02") -> pd.DataFrame:
    """Create a toy daily OHLCV DataFrame."""
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    close = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n_days),
            "high": close + rng.uniform(0, 2, n_days),
            "low": close - rng.uniform(0, 2, n_days),
            "close": close,
            "volume": rng.integers(1_000, 10_000, n_days),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )
    return df


def _make_news(articles: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Create a toy news DataFrame from list of (date_str, title, content)."""
    rows = []
    for dt_str, title, content in articles:
        rows.append(
            {
                "published_date": pd.Timestamp(dt_str),
                "title": title,
                "content": content,
            }
        )
    return pd.DataFrame(rows)


# ======================================================================
# Test: Temporal alignment — no data leakage
# ======================================================================

class TestAlignerNoLeakage:
    """Verify that news published on day T NEVER appears in bar T."""

    def test_same_day_news_not_in_same_bar(self):
        """News published during trading on day T → bar T+1, NOT bar T."""
        df_ohlcv = _make_daily_ohlcv(5, start="2024-01-02")

        # News published at 10:00 on 2024-01-02 (Tuesday, during trading)
        df_news = _make_news([
            ("2024-01-02 10:00:00", "Midday headline", "Some content"),
        ])

        aligned = TemporalAligner.assign_news_to_bars(df_ohlcv, df_news)

        # Bar on 2024-01-02 must NOT have this article
        jan2 = pd.Timestamp("2024-01-02")
        bar_jan2 = aligned.loc[aligned.index.normalize() == jan2]
        assert bar_jan2["news_count"].iloc[0] == 0, "Leakage! Same-day news appeared in bar T"

        # It should appear in bar 2024-01-03
        jan3 = pd.Timestamp("2024-01-03")
        bar_jan3 = aligned.loc[aligned.index.normalize() == jan3]
        assert bar_jan3["news_count"].iloc[0] == 1

    def test_premarket_news_in_same_bar(self):
        """News published before 09:00 on day T → bar T."""
        df_ohlcv = _make_daily_ohlcv(5, start="2024-01-02")

        df_news = _make_news([
            ("2024-01-03 07:30:00", "Early morning headline", "Content"),
        ])

        aligned = TemporalAligner.assign_news_to_bars(df_ohlcv, df_news)

        jan3 = pd.Timestamp("2024-01-03")
        bar_jan3 = aligned.loc[aligned.index.normalize() == jan3]
        assert bar_jan3["news_count"].iloc[0] == 1

    def test_weekend_news_next_trading_bar(self):
        """Weekend news → next trading day (Monday)."""
        # 2024-01-06 is Saturday, 2024-01-08 is Monday
        df_ohlcv = _make_daily_ohlcv(10, start="2024-01-02")

        df_news = _make_news([
            ("2024-01-06 14:00:00", "Weekend headline", "Weekend content"),
        ])

        aligned = TemporalAligner.assign_news_to_bars(df_ohlcv, df_news)

        # Should appear on Monday 2024-01-08
        jan8 = pd.Timestamp("2024-01-08")
        bar_jan8 = aligned.loc[aligned.index.normalize() == jan8]
        assert bar_jan8["news_count"].iloc[0] == 1

    def test_after_hours_news_next_bar(self):
        """News after 15:00 on day T → bar T+1."""
        df_ohlcv = _make_daily_ohlcv(5, start="2024-01-02")

        df_news = _make_news([
            ("2024-01-02 17:00:00", "After-hours headline", "Content"),
        ])

        aligned = TemporalAligner.assign_news_to_bars(df_ohlcv, df_news)

        jan2 = pd.Timestamp("2024-01-02")
        bar_jan2 = aligned.loc[aligned.index.normalize() == jan2]
        assert bar_jan2["news_count"].iloc[0] == 0

        jan3 = pd.Timestamp("2024-01-03")
        bar_jan3 = aligned.loc[aligned.index.normalize() == jan3]
        assert bar_jan3["news_count"].iloc[0] == 1


# ======================================================================
# Test: NewsEncoder null mask
# ======================================================================

class TestEncoderNullMask:
    """Verify null-news encoding produces zero vectors with has_news=False."""

    def test_empty_texts_returns_zero_vec(self):
        encoder = NewsEncoder()
        result = encoder.encode_window(texts=[], null_mask=False)
        assert result["has_news"] is False
        assert result["embedding"].shape == (768,)
        assert np.allclose(result["embedding"], 0.0)

    def test_null_mask_flag_returns_zero_vec(self):
        encoder = NewsEncoder()
        result = encoder.encode_window(texts=["Some real text"], null_mask=True)
        assert result["has_news"] is False
        assert np.allclose(result["embedding"], 0.0)

    def test_whitespace_only_texts_returns_zero_vec(self):
        encoder = NewsEncoder()
        result = encoder.encode_window(texts=["   ", "  \n  "], null_mask=False)
        assert result["has_news"] is False
        assert np.allclose(result["embedding"], 0.0)


# ======================================================================
# Test: CMTFDataset temporal split
# ======================================================================

class TestDatasetTemporalSplit:
    """Verify walk-forward splits preserve chronological order."""

    def _build_dataset(self) -> CMTFDataset:
        n = 100
        dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
        rng = np.random.default_rng(1)

        df = pd.DataFrame(
            {
                "open": rng.normal(100, 5, n),
                "high": rng.normal(102, 5, n),
                "low": rng.normal(98, 5, n),
                "close": rng.normal(100, 5, n),
                "volume": rng.integers(1000, 9000, n),
                "rsi_14": rng.normal(50, 10, n),
                "log_ret": rng.normal(0, 0.02, n),
                "fwd_ret_1d": rng.normal(0, 0.02, n),
                "news_emb": [rng.normal(0, 1, 768).astype(np.float32) for _ in range(n)],
                "has_news": rng.choice([True, False], n),
            },
            index=pd.DatetimeIndex(dates, name="time"),
        )
        return CMTFDataset(df, sequence_len=5, horizon=1)

    def test_no_overlap_between_splits(self):
        ds = self._build_dataset()
        train, val, test = ds.create_splits("2023-03-31", "2023-04-30")

        all_indices = set(train.indices) | set(val.indices) | set(test.indices)
        assert len(all_indices) == len(train.indices) + len(val.indices) + len(test.indices), \
            "Splits overlap!"

    def test_train_before_val_before_test(self):
        ds = self._build_dataset()
        train, val, test = ds.create_splits("2023-03-31", "2023-04-30")

        if train.indices and val.indices:
            assert max(train.indices) < min(val.indices)
        if val.indices and test.indices:
            assert max(val.indices) < min(test.indices)

    def test_target_excluded_from_market_features(self):
        ds = self._build_dataset()
        assert "fwd_ret_1d" not in ds.market_cols, "Target column leaked into market features!"

    def test_sample_shapes(self):
        ds = self._build_dataset()
        sample = ds[0]
        assert sample["market"].shape == (5, len(ds.market_cols))
        assert sample["news"].shape == (5, 768)
        assert sample["mask"].shape == (5,)
        assert sample["target"].shape == (1,)


# ======================================================================
# Test: NewsScraper helpers
# ======================================================================

from src.pipeline.news_scraper import (
    NewsScraper,
    _SUPPORTED_BANK_SYMBOLS,
    _normalise_title,
    _dedup_articles,
)


class TestNewsScraperHelpers:
    """Verify scraper helper functions (no network calls)."""

    def test_supported_bank_symbols(self):
        """Banking-only mode supports exactly VCB and MBB."""
        assert set(_SUPPORTED_BANK_SYMBOLS) == {"VCB", "MBB"}

    def test_deduplication_removes_identical_titles(self):
        articles = [
            {"title": "Vietcombank lãi kỉ lục", "content": "A", "source": "cafef"},
            {"title": "Vietcombank lãi kỉ lục", "content": "B", "source": "vietstock"},
            {"title": "MBB tăng trưởng tín dụng", "content": "C", "source": "cafef_banking"},
        ]
        result, dup_rows = _dedup_articles(articles)
        assert len(result) == 2
        assert len(dup_rows) == 1
        titles = [r["title"] for r in result]
        assert "Vietcombank lãi kỉ lục" in titles
        assert "MBB tăng trưởng tín dụng" in titles

    def test_dedup_normalises_whitespace_and_case(self):
        articles = [
            {"title": "  Hello  World! ", "content": "A"},
            {"title": "hello world", "content": "B"},
        ]
        result, _ = _dedup_articles(articles)
        assert len(result) == 1

    def test_dedup_similarity_not_exact_match(self):
        articles = [
            {"title": "Vietcombank loi nhuan tang manh quy 1", "content": "A"},
            {"title": "Loi nhuan Vietcombank tang manh trong quy 1", "content": "B"},
        ]
        result, dup_rows = _dedup_articles(articles, similarity_threshold=85.0)
        assert len(result) == 1
        assert len(dup_rows) == 1
        assert dup_rows[0]["filter_reason"] == "duplicate"

    def test_date_filtering_in_articles_to_dataframe(self):
        articles = [
            {"title": "Old", "content": "old article", "published_date": "2021-06-01"},
            {"title": "In range", "content": "good article", "published_date": "2023-06-15"},
            {"title": "Future", "content": "future article", "published_date": "2025-01-01"},
        ]
        df = NewsScraper._articles_to_dataframe(articles, "2022-01-01", "2024-12-31")
        assert len(df) == 1
        assert df.iloc[0]["title"] == "In range"

    def test_output_schema(self):
        articles = [
            {"title": "Test", "content": "body text here", "published_date": "2023-03-01"},
        ]
        df = NewsScraper._articles_to_dataframe(articles, "2023-01-01", "2023-12-31")
        assert list(df.columns) == [
            "published_date",
            "title",
            "content",
            "source",
            "source_url",
            "article_id",
            "filter_reason",
        ]
        assert pd.api.types.is_datetime64_any_dtype(df["published_date"])
        assert (df["filter_reason"] == "kept").all()

    def test_empty_articles_returns_empty_df(self):
        df = NewsScraper._articles_to_dataframe([], "2023-01-01", "2023-12-31")
        assert df.empty
        assert list(df.columns) == [
            "published_date",
            "title",
            "content",
            "source",
            "source_url",
            "article_id",
            "filter_reason",
        ]

    def test_normalise_title(self):
        assert _normalise_title("  Hello, World!  ") == "hello world"
        assert _normalise_title("VCB: Lãi kỉ lục!") == "vcb lãi kỉ lục"
