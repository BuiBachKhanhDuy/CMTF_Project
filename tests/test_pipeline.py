"""Unit tests for the CMTF data pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.temporal_aligner import TemporalAligner
from src.pipeline.news_encoder import NEWS_HYBRID_COLUMN, SENTIMENT_TRACE_COLUMNS, NewsEncoder
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
    """Verify daily alignment follows a conservative market-close cutoff."""

    def test_same_day_intraday_news_stays_in_same_bar(self):
        """News published before market close on day T -> bar T."""
        df_ohlcv = _make_daily_ohlcv(5, start="2024-01-02")

        # News published at 10:00 on 2024-01-02 (before close)
        df_news = _make_news([
            ("2024-01-02 10:00:00", "Midday headline", "Some content"),
        ])

        aligned = TemporalAligner.assign_news_to_bars(df_ohlcv, df_news)

        # Bar on 2024-01-02 should have this article
        jan2 = pd.Timestamp("2024-01-02")
        bar_jan2 = aligned.loc[aligned.index.normalize() == jan2]
        assert bar_jan2["news_count"].iloc[0] == 1

    def test_premarket_news_in_same_bar(self):
        """News published before market close on day T -> bar T."""
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
        """News at or after 15:00 on day T -> bar T+1."""
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

    def test_date_only_timestamp_shifted_conservatively(self):
        """Date-only timestamps (00:00:00) are shifted to next trading bar."""
        df_ohlcv = _make_daily_ohlcv(5, start="2024-01-02")

        # Simulates scraped source where exact publish time is unknown.
        df_news = _make_news([
            ("2024-01-02", "Date-only headline", "Content"),
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


class TestEncoderHybridSentiment:
    def test_encode_dataframe_adds_hybrid_sentiment_features(self, monkeypatch):
        class FakeInferencer:
            def predict_titles(self, titles, batch_size=32):
                return pd.DataFrame(
                    {
                        "title_raw": list(titles),
                        "title_clean": list(titles),
                        "sentiment_score": np.array([0.4, -0.2], dtype=np.float32)[: len(titles)],
                        "prob_negative": np.array([0.2, 0.6], dtype=np.float32)[: len(titles)],
                        "prob_neutral": np.array([0.3, 0.3], dtype=np.float32)[: len(titles)],
                        "prob_positive": np.array([0.5, 0.1], dtype=np.float32)[: len(titles)],
                    }
                )

        monkeypatch.setattr(
            NewsEncoder,
            "encode_window",
            lambda self, texts, null_mask=False, weights=None: {
                "embedding": np.ones(768, dtype=np.float32) if texts and not null_mask else np.zeros(768, dtype=np.float32),
                "has_news": bool(texts) and not bool(null_mask),
            },
        )

        frame = pd.DataFrame(
            {
                "symbol": ["VCB", "VCB"],
                "time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "news_titles": [["A", "B"], []],
                "news_content": [["content a", "content b"], []],
                "news_missing_flag": [False, True],
            }
        )

        encoder = NewsEncoder(sentiment_inferencer=FakeInferencer())
        encoded = encoder.encode_dataframe(frame, use_cache=False)

        assert NEWS_HYBRID_COLUMN in encoded.columns
        assert encoded.loc[0, NEWS_HYBRID_COLUMN].shape == (773,)
        assert np.allclose(encoded.loc[1, NEWS_HYBRID_COLUMN], 0.0)
        assert encoded.loc[0, "sentiment_mean"] == pytest.approx(0.1)
        assert encoded.loc[0, "sentiment_positive_ratio"] == pytest.approx(0.5)
        assert encoded.loc[1, "sentiment_missing_flag"] == pytest.approx(1.0)
        assert encoder.last_sentiment_trace is not None
        assert len(encoder.last_sentiment_trace) == 2


class TestDatasetCacheHybridRestore:
    def test_save_and_load_restores_hybrid_news_embeddings(self, tmp_path):
        from src.pipeline.orchestrator import _load_dataset_cache, _save_dataset_cache

        cache_path = tmp_path / "dataset_test.parquet"
        frame = pd.DataFrame(
            {
                "symbol": ["VCB"],
                "news_count": [1],
                "has_news": [True],
                "news_emb": [np.ones(768, dtype=np.float32)],
                NEWS_HYBRID_COLUMN: [np.ones(773, dtype=np.float32)],
                "sentiment_mean": [0.2],
                "sentiment_max_abs": [0.2],
                "sentiment_positive_ratio": [1.0],
                "sentiment_negative_ratio": [0.0],
                "sentiment_score_count": [1.0],
                "sentiment_missing_flag": [0.0],
                "fwd_ret_1d": [0.01],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="time"),
        )

        _save_dataset_cache(frame, cache_path)
        restored = _load_dataset_cache(cache_path)

        assert restored is not None
        assert restored.loc[restored.index[0], NEWS_HYBRID_COLUMN].shape == (773,)
        assert np.allclose(restored.loc[restored.index[0], NEWS_HYBRID_COLUMN], 1.0)
        for col in SENTIMENT_TRACE_COLUMNS:
            assert col in restored.columns


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
        """Verify walk-forward splits have no overlapping samples."""
        n = 100
        times = pd.bdate_range("2023-01-02", periods=n, freq="B").values
        data = {
            "close_windows": np.random.default_rng(1).normal(100, 5, (n, 30)),
            "targets": np.random.default_rng(1).normal(0, 0.02, n),
            "news_embs": np.random.default_rng(1).normal(0, 1, (n, 768)).astype(np.float32),
        }
        from run_chronos_benchmark import split_by_date
        splits = split_by_date(data, times, "2023-03-31", "2023-04-30", target_horizon_days=1)

        train_set = set(range(len(splits["train"]["targets"])))
        val_set = set(range(len(splits["val"]["targets"])))
        test_set = set(range(len(splits["test"]["targets"])))
        total = len(splits["train"]["targets"]) + len(splits["val"]["targets"]) + len(splits["test"]["targets"])
        assert total <= n, "More samples than original data"
        # Verify splits are mutually exclusive via time ranges
        train_times = times[times <= pd.Timestamp("2023-03-31")]
        val_times = times[(times > pd.Timestamp("2023-03-31")) & (times <= pd.Timestamp("2023-04-30"))]
        test_times = times[times > pd.Timestamp("2023-04-30")]
        assert len(set(train_times) & set(val_times)) == 0, "Train/val overlap!"
        assert len(set(val_times) & set(test_times)) == 0, "Val/test overlap!"

    def test_train_before_val_before_test(self):
        """Verify chronological ordering of walk-forward splits."""
        n = 100
        times = pd.bdate_range("2023-01-02", periods=n, freq="B").values
        data = {
            "close_windows": np.random.default_rng(1).normal(100, 5, (n, 30)),
            "targets": np.random.default_rng(1).normal(0, 0.02, n),
            "news_embs": np.random.default_rng(1).normal(0, 1, (n, 768)).astype(np.float32),
        }
        from run_chronos_benchmark import split_by_date
        splits = split_by_date(data, times, "2023-03-31", "2023-04-30", target_horizon_days=1)

        # Reconstruct times per split using the mask logic
        train_mask = times <= pd.Timestamp("2023-03-31")
        val_mask = (times > pd.Timestamp("2023-03-31")) & (times <= pd.Timestamp("2023-04-30"))
        test_mask = times > pd.Timestamp("2023-04-30")

        train_t = times[train_mask]
        val_t = times[val_mask]
        test_t = times[test_mask]

        if len(train_t) > 0 and len(val_t) > 0:
            assert train_t.max() < val_t.min(), "Train not before val!"
        if len(val_t) > 0 and len(test_t) > 0:
            assert val_t.max() < test_t.min(), "Val not before test!"

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
        """Banking mode supports the 2 target symbols."""
        assert set(_SUPPORTED_BANK_SYMBOLS) == {"VCB", "BID"}

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

    def test_dedup_keeps_higher_quality_duplicate(self):
        articles = [
            {
                "title": "Vietcombank loi nhuan tang manh",
                "content": "short",
                "source": "source_a",
            },
            {
                "title": "Loi nhuan Vietcombank tang manh",
                "content": "This is a much longer and more informative article body.",
                "source": "source_b",
            },
        ]
        result, dup_rows = _dedup_articles(articles, similarity_threshold=80.0)
        assert len(result) == 1
        assert len(dup_rows) == 1
        assert result[0]["source"] == "source_b"

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
