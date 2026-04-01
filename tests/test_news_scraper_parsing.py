"""Parser and selector fallback tests for news scraper (mocked network)."""

from __future__ import annotations

import pandas as pd

from src.pipeline import news_scraper as ns


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def test_cafef_banking_listing_and_article_parsing(monkeypatch):
    listing_html = """
    <html><body>
      <a class='box-category-link-title' href='/cafef-1.chn'>Ngân hàng tăng trưởng mạnh</a>
    </body></html>
    """
    article_html = """
    <html>
      <head>
        <meta property='article:published_time' content='2024-03-01T10:30:00+07:00'/>
      </head>
      <body>
        <div class='detail-content'>
          Noi dung bai viet du dai hon 100 ky tu. Noi dung bai viet du dai hon 100 ky tu.
          Noi dung bai viet du dai hon 100 ky tu. Noi dung bai viet du dai hon 100 ky tu.
        </div>
      </body>
    </html>
    """

    def fake_get(url: str, **_: object):
      if "tai-chinh-ngan-hang" in url:
        return _Resp(listing_html)
      if "cafef-1.chn" in url:
        return _Resp(article_html)
      return _Resp("<html></html>")

    monkeypatch.setattr(ns, "_http_get", fake_get)

    rows = ns._scrape_cafef_banking(
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-12-31"),
        max_pages=1,
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "cafef_banking"
    assert rows[0]["source_url"].startswith("https://cafef.vn/")
    assert rows[0]["article_id"]


def test_vietstock_listing_and_article_parsing(monkeypatch):
    listing_html = """
    <html><body>
      <h3><a href='https://vietstock.vn/VCB/vietstock-1.htm'>VCB mở rộng tín dụng</a></h3>
    </body></html>
    """
    article_html = """
    <html>
      <head><meta property='article:published_time' content='2024-02-15T08:00:00+07:00'/></head>
      <body>
        <div class='pContent'>
          Noi dung dai hon 100 ky tu. Noi dung dai hon 100 ky tu. Noi dung dai hon 100 ky tu.
          Noi dung dai hon 100 ky tu. Noi dung dai hon 100 ky tu. Noi dung dai hon 100 ky tu.
        </div>
      </body>
    </html>
    """

    def fake_get(url: str, **_: object):
        if "vietstock.vn/VCB/tin-tuc.htm" in url:
            return _Resp(listing_html)
        if "vietstock-1.htm" in url:
            return _Resp(article_html)
        return _Resp("<html></html>")

    monkeypatch.setattr(ns, "_http_get", fake_get)

    rows = ns._scrape_vietstock(
        symbol="VCB",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-12-31"),
        max_pages=1,
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "vietstock"
    assert rows[0]["source_url"].startswith("https://vietstock.vn/")
    assert rows[0]["article_id"]


def test_fetch_news_rejects_non_bank_symbol():
    scraper = ns.NewsScraper()
    try:
        scraper.fetch_news(
            symbol="VIC",
            start="2024-01-01",
            end="2024-12-31",
            use_cache=False,
            export_trace=False,
        )
        assert False, "Expected ValueError for unsupported symbol"
    except ValueError:
        assert True
