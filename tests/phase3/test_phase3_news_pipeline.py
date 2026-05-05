from __future__ import annotations

import pandas as pd

from src.phase3 import (
    RawNewsArticle,
    classify_news_articles,
    drop_duplicate_news_articles,
    filter_retained_news_articles,
    mark_duplicate_news_articles,
    normalize_news_articles,
    raw_articles_to_frame,
)
from src.pipeline.news_scraper import _extract_vietstock_article_urls, _iter_month_windows


def test_normalize_news_articles_uses_publisher_time_and_stable_ids():
    raw = raw_articles_to_frame(
        [
            RawNewsArticle(
                source="cafef",
                published_at="2024-06-03T08:05:00+07:00",
                scraped_at="2024-06-03T08:20:00+07:00",
                title="  Vietcombank tang truong loi nhuan  ",
                url="https://cafef.vn/article?id=1&utm_source=newsletter#top",
                publisher_article_id="ART-1",
            ),
            RawNewsArticle(
                source="cafef",
                published_at=None,
                title="Ban tin khong co publisher timestamp",
                url="https://cafef.vn/article?id=2",
                publisher_article_id="ART-2",
            ),
        ]
    )

    normalized = normalize_news_articles(raw, source="cafef")
    normalized_again = normalize_news_articles(raw, source="cafef")

    assert normalized.loc[0, "canonical_url"] == "https://cafef.vn/article?id=1"
    assert normalized.loc[0, "published_at"] == pd.Timestamp("2024-06-03 08:05:00")
    assert normalized.loc[0, "scraped_at"] == pd.Timestamp("2024-06-03 08:20:00")
    assert normalized.loc[0, "title_clean"] == "Vietcombank tang truong loi nhuan"
    assert normalized.loc[1, "rejection_reason"] == "missing_published_at"
    assert normalized["article_id"].tolist() == normalized_again["article_id"].tolist()


def test_mark_duplicate_news_articles_flags_exact_and_near_duplicates():
    raw = pd.DataFrame(
        {
            "published_at": [
                "2024-06-03 08:05:00",
                "2024-06-03 08:06:00",
                "2024-06-03 08:20:00",
            ],
            "title": [
                "VN-Index tang manh phien sang",
                "VN-Index tang manh phien sang",
                "Thanh khoan tang tro lai",
            ],
            "url": [
                "https://vietstock.vn/a?id=1",
                "https://vietstock.vn/b?id=2",
                "https://vietstock.vn/c?id=3",
            ],
        }
    )

    normalized = filter_retained_news_articles(normalize_news_articles(raw, source="vietstock"))
    marked = mark_duplicate_news_articles(normalized)
    retained = drop_duplicate_news_articles(marked)

    assert marked["is_duplicate"].tolist() == [False, True, False]
    assert marked.loc[1, "duplicate_reason"] == "duplicate_title_time_bucket"
    assert retained["article_id"].tolist() == [marked.loc[0, "article_id"], marked.loc[2, "article_id"]]


def test_classify_news_articles_splits_symbol_and_market_channels():
    raw = pd.DataFrame(
        {
            "published_at": [
                "2024-06-03 08:05:00",
                "2024-06-03 09:00:00",
                "2024-06-03 10:00:00",
            ],
            "title": [
                "Vietcombank mo rong tin dung",
                "VN-Index cai thien liquidity toan thi truong",
                "BID va VN-Index cung tang",
            ],
            "content": ["", "", ""],
        }
    )

    normalized = filter_retained_news_articles(normalize_news_articles(raw, source="cafef"))
    classified = classify_news_articles(
        normalized,
        symbol_aliases={"VCB": ["vietcombank"], "BID": ["bidv"]},
        market_keywords=["vn-index", "liquidity"],
    )

    assert classified.loc[0, "classification"] == "symbol_linked"
    assert classified.loc[0, "linked_symbols"] == ("VCB",)

    assert classified.loc[1, "classification"] == "market_wide"
    assert bool(classified.loc[1, "is_market_wide"]) is True

    assert classified.loc[2, "classification"] == "mixed"
    assert classified.loc[2, "linked_symbols"] == ("BID",)


def test_vietstock_helpers_extract_article_urls_and_month_windows():
    html = """
    <div>
      <a href="/2022/01/bai-viet-a-830-1001.htm">A</a>
      <a href="https://vietstock.vn/2022/01/bai-viet-b-830-1002.htm">B</a>
      <a href="/chu-de/1-2/moi-cap-nhat.htm">ignore</a>
      <a href="/2022/01/bai-viet-a-830-1001.htm">dup</a>
    </div>
    """

    urls = _extract_vietstock_article_urls(html)
    assert urls == [
        "https://vietstock.vn/2022/01/bai-viet-a-830-1001.htm",
        "https://vietstock.vn/2022/01/bai-viet-b-830-1002.htm",
    ]

    windows = _iter_month_windows("2022-01-15", "2022-03-03")
    assert windows == [
        ("2022-01-15", "2022-01-31"),
        ("2022-02-01", "2022-02-28"),
        ("2022-03-01", "2022-03-03"),
    ]