"""Step 1 validation: news scraper volume settings.

Run:  pytest tests/test_step1_scraper_volume.py -v
"""

from __future__ import annotations

import os

import pytest

from src.pipeline import news_scraper as ns


class TestScraperConfig:
    """Verify scraper caps and keyword config are sufficient."""

    def test_max_articles_cap_raised(self):
        assert ns._MAX_ARTICLES_PER_SOURCE >= 500, (
            f"_MAX_ARTICLES_PER_SOURCE={ns._MAX_ARTICLES_PER_SOURCE}, expected ≥500"
        )

    @pytest.mark.parametrize("symbol", ["VCB", "BID"])
    def test_vnexpress_keywords_include_ticker(self, symbol: str):
        keywords = ns._VNEXPRESS_KEYWORDS[symbol]
        tickers_in_kw = [kw for kw in keywords if symbol in kw]
        assert tickers_in_kw, (
            f"{symbol}: ticker code not found in VNExpress keywords: {keywords}"
        )

    @pytest.mark.parametrize("symbol", ["VCB", "BID"])
    def test_vnexpress_keywords_count(self, symbol: str):
        keywords = ns._VNEXPRESS_KEYWORDS[symbol]
        assert len(keywords) >= 4, (
            f"{symbol}: only {len(keywords)} keywords, expected ≥4: {keywords}"
        )

    def test_vnexpress_pagination_limit(self):
        """VNExpress max_pages should be ≥50 for 4+ year date ranges."""
        import inspect
        sig = inspect.signature(ns._scrape_vnexpress)
        default = sig.parameters["max_pages"].default
        assert default >= 50, f"VNExpress max_pages={default}, expected ≥50"

    def test_vietstock_pagination_limit(self):
        """Vietstock max_pages should be ≥120 for 4+ year date ranges."""
        import inspect
        sig = inspect.signature(ns._scrape_vietstock)
        default = sig.parameters["max_pages"].default
        assert default >= 120, f"Vietstock max_pages={default}, expected ≥120"


@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCRAPER", "0") != "1",
    reason="Set RUN_LIVE_SCRAPER=1 to run live scraper volume test.",
)
def test_live_scrape_volume():
    """Smoke: scrape VCB for a 3-month window, expect ≥30 kept articles."""
    scraper = ns.NewsScraper()
    df = scraper.fetch_news(
        symbol="VCB",
        start="2025-01-01",
        end="2025-03-31",
        sources=("vnexpress", "cafef_banking", "vietstock"),
        use_cache=False,
        export_trace=False,
    )
    assert len(df) >= 30, f"Only {len(df)} articles for VCB in 3 months, expected ≥30"
