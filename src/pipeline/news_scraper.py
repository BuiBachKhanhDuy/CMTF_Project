"""CMTF Data Pipeline — Web News Scraper module.

Scrapes Vietnamese financial news from CafeF and VnExpress to provide
dense news coverage for each symbol across the study period.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEWS_CACHE_DIR = Path("./cache/news")

# Map ticker symbols to Vietnamese company names for search queries.
# Using company names produces far more relevant financial articles than
# ticker symbols (which often match product names like "VCB Digibank").
SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "VCB": ["Vietcombank", "Ngân hàng Ngoại thương"],
    "VIC": ["Vingroup", "Tập đoàn Vingroup"],
    "VHM": ["Vinhomes"],
}

_REQUEST_DELAY = 0.5  # seconds between HTTP requests (rate limiting)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_title(title: str) -> str:
    """Lowercase, strip whitespace and punctuation for dedup comparison."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()


def _dedup_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate articles by normalised title."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for art in articles:
        key = _normalise_title(str(art.get("title", "")))
        if key and key not in seen:
            seen.add(key)
            unique.append(art)
    return unique


def _cache_path(symbol: str) -> Path:
    """Return the JSON cache file path for a symbol."""
    _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _NEWS_CACHE_DIR / f"{symbol}_news.json"


def _load_cache(symbol: str) -> list[dict[str, Any]] | None:
    """Load cached articles for *symbol*, or None if cache miss."""
    p = _cache_path(symbol)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            logger.info("News cache hit for {} — {} articles", symbol, len(data))
            return data
        except Exception:
            logger.warning("Corrupt news cache for {} — ignoring", symbol)
    return None


def _save_cache(symbol: str, articles: list[dict[str, Any]]) -> None:
    """Persist scraped articles to disk cache."""
    p = _cache_path(symbol)
    p.write_text(json.dumps(articles, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("News cache saved for {} — {} articles", symbol, len(articles))


# ---------------------------------------------------------------------------
# CafeF scraper
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _http_get(url: str, **kwargs: Any) -> "requests.Response":
    """GET with retry, rate-limiting, and standard headers."""
    import requests

    headers = kwargs.pop("headers", {})
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    headers.setdefault("Accept-Language", "vi-VN,vi;q=0.9,en;q=0.8")
    resp = requests.get(url, headers=headers, timeout=30, **kwargs)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY)
    return resp


_MAX_ARTICLES_PER_SOURCE = 30  # cap individual article fetches per source


def _scrape_cafef_listing(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    keywords: list[str] | None = None,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    """Scrape article listing from CafeF search.

    Uses the search endpoint ``https://cafef.vn/tim-kiem.chn?keywords=...``
    since the old ``/ma-co-phieu/{SYMBOL}.chn`` tag pages no longer exist.
    """
    from bs4 import BeautifulSoup

    # Build search terms: use company keywords if provided, else ticker
    search_terms = keywords or [symbol]
    articles: list[dict[str, Any]] = []

    for keyword in search_terms:
        for page in range(1, max_pages + 1):
            url = f"https://cafef.vn/tim-kiem.chn?keywords={keyword}&page={page}"
            logger.debug("CafeF search page {} for '{}'", page, keyword)
            try:
                resp = _http_get(url)
            except Exception:
                logger.warning("CafeF page {} failed for '{}' — stopping", page, keyword)
                break

            soup = BeautifulSoup(resp.text, "lxml")

            # CafeF search results: h3 > a.box-category-link-title, or generic h3 a
            items = soup.select("li.tlitem, div.tlitem, li.news-item, div.news-item")
            if not items:
                items = soup.select("h3 a[href]")
                if not items:
                    logger.debug("No more items on CafeF page {} for '{}'", page, keyword)
                    break

            for item in items:
                link_tag = item if item.name == "a" else item.select_one("a[href]")
                if link_tag is None:
                    continue

                href = link_tag.get("href", "")
                if not href:
                    continue
                if href.startswith("/"):
                    href = "https://cafef.vn" + href

                title = link_tag.get_text(strip=True)

                articles.append({
                    "title": title,
                    "url": href,
                    "published_date": None,
                    "source": "cafef",
                })

    logger.info("CafeF listing: {} candidate articles for {}", len(articles), symbol)

    # Fetch full content + date for each article (capped)
    enriched: list[dict[str, Any]] = []
    for art in articles[:_MAX_ARTICLES_PER_SOURCE]:
        try:
            content, pub_date = _scrape_cafef_article(art["url"])
            if pub_date is not None:
                art["published_date"] = str(pub_date)
                # Filter by date range
                if pub_date < start_date or pub_date > end_date:
                    continue
            art["content"] = content
            if content and len(content) >= 100:
                enriched.append(art)
        except Exception:
            logger.debug("Failed to fetch CafeF article: {}", art.get("url", ""))

    return enriched


def _scrape_cafef_article(url: str) -> tuple[str, pd.Timestamp | None]:
    """Extract full article text and published date from a CafeF article page."""
    from bs4 import BeautifulSoup

    resp = _http_get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Content extraction — CafeF uses several div classes
    content_div = (
        soup.select_one("div.detail-content")
        or soup.select_one("div.contentdetail")
        or soup.select_one("div#mainContent")
        or soup.select_one("article")
    )
    content = ""
    if content_div:
        # Remove script/style tags
        for tag in content_div.find_all(["script", "style"]):
            tag.decompose()
        content = content_div.get_text(separator=" ", strip=True)

    # Date extraction — ordered by reliability:
    # 1. meta article:published_time (CafeF provides this)
    # 2. JSON-LD datePublished
    # 3. span.pdate ("30-03-2026 - 13:57 PM")
    # 4. span.dateandcate fallback
    pub_date = None

    meta_date = soup.select_one("meta[property='article:published_time']")
    if meta_date:
        try:
            pub_date = pd.to_datetime(meta_date["content"], errors="coerce")
            if pub_date is not None and pub_date.tzinfo is not None:
                pub_date = pub_date.tz_localize(None)
        except Exception:
            pass

    if pub_date is None:
        for script in soup.find_all("script", type="application/ld+json"):
            txt = script.string or ""
            match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', txt)
            if match:
                try:
                    pub_date = pd.to_datetime(match.group(1), errors="coerce")
                    if pub_date is not None and pub_date.tzinfo is not None:
                        pub_date = pub_date.tz_localize(None)
                    break
                except Exception:
                    pass

    if pub_date is None:
        date_tag = (
            soup.select_one("span.pdate")
            or soup.select_one("span.dateandcate")
        )
        if date_tag:
            try:
                pub_date = pd.to_datetime(
                    date_tag.get_text(strip=True),
                    dayfirst=True,
                    errors="coerce",
                )
            except Exception:
                pass

    return content, pub_date


# ---------------------------------------------------------------------------
# VnExpress scraper
# ---------------------------------------------------------------------------

def _scrape_vnexpress(
    keywords: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int = 8,
) -> list[dict[str, Any]]:
    """Scrape articles from VnExpress search API with date range.

    VnExpress supports ``fromdate`` and ``todate`` as Unix timestamps.
    """
    from bs4 import BeautifulSoup

    articles: list[dict[str, Any]] = []
    from_epoch = int(start_date.timestamp())
    to_epoch = int(end_date.timestamp())

    for keyword in keywords:
        for page in range(1, max_pages + 1):
            url = (
                f"https://timkiem.vnexpress.net/?q={keyword}"
                f"&media_type=all"
                f"&fromdate={from_epoch}"
                f"&todate={to_epoch}"
                f"&latest="
                f"&cate_code=kinhdoanh"  # Business section — more relevant
                f"&page={page}"
            )
            logger.debug("VnExpress search page {} for keyword '{}'", page, keyword)
            try:
                resp = _http_get(url)
            except Exception:
                logger.warning("VnExpress page {} failed for '{}' — stopping", page, keyword)
                break

            soup = BeautifulSoup(resp.text, "lxml")

            # VnExpress search results
            result_items = soup.select("article.item-news, div.item-news")
            if not result_items:
                # Try title links
                result_items = soup.select("h3.title-news a[href]")
                if not result_items:
                    break

            for item in result_items:
                link_tag = item if item.name == "a" else item.select_one("h3.title-news a, a.title-news, h2 a")
                if link_tag is None:
                    continue

                href = link_tag.get("href", "")
                if not href or not href.startswith("http"):
                    continue

                title = link_tag.get_text(strip=True)

                # Extract description/snippet
                desc_tag = item.select_one("p.description")
                description = desc_tag.get_text(strip=True) if desc_tag else ""

                # Try date from listing
                date_tag = item.select_one("span.time-ago, span.date, span.time-public")
                pub_date = None
                if date_tag:
                    date_str = date_tag.get("datetime") or date_tag.get_text(strip=True)
                    try:
                        pub_date = pd.to_datetime(date_str, errors="coerce")
                    except Exception:
                        pass

                articles.append({
                    "title": title,
                    "url": href,
                    "published_date": str(pub_date) if pub_date else None,
                    "description": description,
                    "source": "vnexpress",
                })

    logger.info("VnExpress search: {} candidate articles", len(articles))

    # Fetch full content for each article (capped)
    enriched: list[dict[str, Any]] = []
    for art in articles[:_MAX_ARTICLES_PER_SOURCE]:
        try:
            content, pub_date = _scrape_vnexpress_article(art["url"])
            if pub_date is not None:
                art["published_date"] = str(pub_date)
            art["content"] = content or art.get("description", "")
            if art["content"] and len(art["content"]) >= 100:
                enriched.append(art)
        except Exception:
            # Use description as fallback content
            if art.get("description") and len(art["description"]) >= 100:
                art["content"] = art["description"]
                enriched.append(art)

    return enriched


def _scrape_vnexpress_article(url: str) -> tuple[str, pd.Timestamp | None]:
    """Extract full article text and published date from a VnExpress article page."""
    from bs4 import BeautifulSoup

    resp = _http_get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Content extraction
    content_div = (
        soup.select_one("article.fck_detail")
        or soup.select_one("div.fck_detail")
        or soup.select_one("div.content_detail")
        or soup.select_one("article")
    )
    content = ""
    if content_div:
        for tag in content_div.find_all(["script", "style", "figure"]):
            tag.decompose()
        content = content_div.get_text(separator=" ", strip=True)

    # Date extraction — ordered by reliability:
    # 1. meta[name='pubdate'] (VnExpress primary; ISO 8601 with TZ)
    # 2. JSON-LD datePublished
    # 3. span.date (human-readable Vietnamese format)
    pub_date = None

    meta_pubdate = soup.select_one("meta[name='pubdate']")
    if meta_pubdate:
        try:
            pub_date = pd.to_datetime(meta_pubdate["content"], errors="coerce")
            if pub_date is not None and pub_date.tzinfo is not None:
                pub_date = pub_date.tz_localize(None)
        except Exception:
            pass

    if pub_date is None:
        for script in soup.find_all("script", type="application/ld+json"):
            txt = script.string or ""
            match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', txt)
            if match:
                try:
                    pub_date = pd.to_datetime(match.group(1), errors="coerce")
                    if pub_date is not None and pub_date.tzinfo is not None:
                        pub_date = pub_date.tz_localize(None)
                    break
                except Exception:
                    pass

    if pub_date is None:
        date_tag = soup.select_one("span.date")
        if date_tag:
            # Format: "Thứ sáu, 6/12/2024, 09:54 (GMT+7)"
            raw = date_tag.get_text(strip=True)
            # Strip Vietnamese day prefix and timezone suffix
            raw = re.sub(r"^[^,]+,\s*", "", raw)       # remove "Thứ ..., "
            raw = re.sub(r"\s*\(.*\)\s*$", "", raw)  # remove "(GMT+7)"
            try:
                pub_date = pd.to_datetime(raw, dayfirst=True, errors="coerce")
            except Exception:
                pass

    return content, pub_date


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NewsScraper:
    """Scrapes Vietnamese financial news from CafeF and VnExpress.

    Provides dense news coverage to replace the sparse vnstock VCI API
    (which returns only ~10 articles per symbol with date mismatches).

    Attributes:
        symbol_keywords: Mapping of ticker → company name search queries.
        cache_dir: Path to the JSON cache directory.
    """

    def __init__(
        self,
        symbol_keywords: dict[str, list[str]] | None = None,
        cache_dir: Path = _NEWS_CACHE_DIR,
    ) -> None:
        self.symbol_keywords = symbol_keywords or SYMBOL_KEYWORDS
        self.cache_dir = cache_dir

    def fetch_news(
        self,
        symbol: str,
        start: str,
        end: str,
        sources: tuple[str, ...] = ("cafef", "vnexpress"),
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch news articles for *symbol* from web sources.

        Args:
            symbol: Ticker symbol (e.g. ``'VCB'``).
            start: Start date ``'YYYY-MM-DD'``.
            end: End date ``'YYYY-MM-DD'``.
            sources: Which backends to query.
            use_cache: Whether to use/populate the disk cache.

        Returns:
            DataFrame with columns ``[published_date, title, content]``.
        """
        # Check cache first
        if use_cache:
            cached = _load_cache(symbol)
            if cached is not None:
                df = self._articles_to_dataframe(cached, start, end)
                if not df.empty:
                    return df

        keywords = self.symbol_keywords.get(symbol, [symbol])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        all_articles: list[dict[str, Any]] = []

        # --- CafeF ---
        if "cafef" in sources:
            try:
                logger.info("Scraping CafeF for {} …", symbol)
                cafef_arts = _scrape_cafef_listing(
                    symbol, start_ts, end_ts, keywords=keywords,
                )
                all_articles.extend(cafef_arts)
                logger.info("CafeF: {} articles for {}", len(cafef_arts), symbol)
            except Exception:
                logger.warning("CafeF scraping failed for {} — continuing", symbol)

        # --- VnExpress ---
        if "vnexpress" in sources:
            try:
                logger.info("Scraping VnExpress for {} (keywords: {}) …", symbol, keywords)
                vne_arts = _scrape_vnexpress(keywords, start_ts, end_ts)
                all_articles.extend(vne_arts)
                logger.info("VnExpress: {} articles for {}", len(vne_arts), symbol)
            except Exception:
                logger.warning("VnExpress scraping failed for {} — continuing", symbol)

        # Deduplicate
        all_articles = _dedup_articles(all_articles)
        logger.info(
            "Total unique articles for {}: {} (after dedup)",
            symbol,
            len(all_articles),
        )

        # Cache results
        if use_cache and all_articles:
            _save_cache(symbol, all_articles)

        return self._articles_to_dataframe(all_articles, start, end)

    @staticmethod
    def _articles_to_dataframe(
        articles: list[dict[str, Any]],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Convert article dicts to the standard news DataFrame schema."""
        if not articles:
            return pd.DataFrame(columns=["published_date", "title", "content"])

        df = pd.DataFrame(articles)

        # Ensure published_date is datetime
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        if df["published_date"].dt.tz is not None:
            df["published_date"] = df["published_date"].dt.tz_localize(None)

        # Ensure title and content columns exist
        if "title" not in df.columns:
            df["title"] = ""
        if "content" not in df.columns:
            df["content"] = ""

        # Drop rows with no valid date
        df = df.dropna(subset=["published_date"])

        # Filter to date range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df["published_date"] >= start_ts) & (df["published_date"] <= end_ts)]

        # Return standard schema
        df = df[["published_date", "title", "content"]].copy()
        df = df.sort_values("published_date").reset_index(drop=True)
        return df
