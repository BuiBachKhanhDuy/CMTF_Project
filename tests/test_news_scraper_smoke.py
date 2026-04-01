"""Optional live smoke tests for crawler availability.

Run explicitly:
    pytest -m smoke tests/test_news_scraper_smoke.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.pipeline.news_scraper import NewsScraper


pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def scraper() -> NewsScraper:
    return NewsScraper()


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCRAPER", "0") != "1",
    reason="Set RUN_LIVE_SCRAPER=1 to run live smoke tests.",
)
def test_cafef_banking_live(scraper: NewsScraper):
    df = scraper.fetch_news(
        symbol="VCB",
        start="2025-04-01",
        end="2025-06-30",
        sources=("cafef_banking",),
        use_cache=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert "published_date" in df.columns


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCRAPER", "0") != "1",
    reason="Set RUN_LIVE_SCRAPER=1 to run live smoke tests.",
)
def test_vietstock_live(scraper: NewsScraper):
    df = scraper.fetch_news(
        symbol="MBB",
        start="2025-04-01",
        end="2025-06-30",
        sources=("vietstock",),
        use_cache=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert "published_date" in df.columns


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCRAPER", "0") != "1",
    reason="Set RUN_LIVE_SCRAPER=1 to run live smoke tests.",
)
def test_reject_non_bank_symbol(scraper: NewsScraper):
    with pytest.raises(ValueError):
        scraper.fetch_news(
            symbol="VIC",
            start="2025-04-01",
            end="2025-06-30",
            sources=("cafef_banking",),
            use_cache=False,
        )
