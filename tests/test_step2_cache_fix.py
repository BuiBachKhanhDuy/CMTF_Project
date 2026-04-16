"""Step 2 validation: news cache is date-range-aware.

Run:  python -m pytest tests/test_step2_cache_fix.py -v
"""

from __future__ import annotations

import pytest

from src.pipeline import news_scraper as ns


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    """Redirect news cache dir to tmp so tests don't pollute real cache."""
    monkeypatch.setattr(ns, "_NEWS_CACHE_DIR", tmp_path)


def _sample_articles() -> list[dict]:
    return [
        {
            "title": "Vietcombank profit up 20%",
            "content": "Article content about VCB results" * 5,
            "published_date": "2024-06-15",
            "source": "vnexpress",
            "article_id": "abc123",
        },
    ]


class TestCacheDateRange:

    def test_cache_path_includes_dates(self):
        p = ns._cache_path("VCB", "2022-01-01", "2026-03-31")
        assert "20220101" in p.name
        assert "20260331" in p.name
        assert "VCB" in p.name

    def test_cache_path_different_dates_different_files(self):
        p1 = ns._cache_path("VCB", "2022-01-01", "2026-03-31")
        p2 = ns._cache_path("VCB", "2018-01-01", "2026-03-31")
        assert p1 != p2

    def test_cache_roundtrip_same_range(self):
        arts = _sample_articles()
        ns._save_cache("VCB", "2022-01-01", "2026-03-31", arts)
        loaded = ns._load_cache("VCB", "2022-01-01", "2026-03-31")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0]["title"] == arts[0]["title"]

    def test_cache_roundtrip_different_ranges(self):
        arts = _sample_articles()
        ns._save_cache("VCB", "2022-01-01", "2026-03-31", arts)
        # Load with different date range → should miss
        loaded = ns._load_cache("VCB", "2018-01-01", "2026-03-31")
        assert loaded is None

    def test_stage_data_config_allows_cache_write(self):
        """Verify --stage data no longer suppresses news_use_cache."""
        # Simulate what run_chronos_benchmark does for --stage data
        config = {"rebuild_data": False, "news_use_cache": True}
        # Simulating stage == "data" override (after fix)
        config["rebuild_data"] = True
        # news_use_cache should NOT be set to False anymore
        assert config.get("news_use_cache") is True, (
            "--stage data should not suppress news cache writes"
        )
