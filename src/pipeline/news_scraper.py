"""CMTF Data Pipeline — Web News Scraper module.

Simplified banking-only scraper:
- Symbols: VCB, MBB
- Sources: CafeF banking category + Vietstock symbol news
- No keyword-based filtering/search
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    import requests


_NEWS_CACHE_DIR = Path("./cache/news")
_NEWS_TRACE_DIR = Path("./artifacts/news_trace")
_SUPPORTED_BANK_SYMBOLS = ("VCB", "MBB")
_REQUEST_DELAY = 1.5
_MAX_ARTICLES_PER_SOURCE = 150


def _normalise_title(title: str) -> str:
    """Lowercase and strip punctuation/extra spaces for title matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()


def _token_set_similarity(a: str, b: str) -> float:
    """Return fuzzy token/sequence similarity in [0, 100]."""
    na = _normalise_title(a)
    nb = _normalise_title(b)
    if not na or not nb:
        return 0.0

    ta = set(na.split())
    tb = set(nb.split())
    if not ta or not tb:
        return 0.0

    overlap = 2.0 * len(ta & tb) / (len(ta) + len(tb))
    seq = SequenceMatcher(None, na, nb).ratio()
    return max(overlap, seq) * 100.0


def _build_article_id(article: dict[str, Any]) -> str:
    """Build stable short id for traceability and dedup reporting."""
    base = "|".join(
        [
            str(article.get("source", "")),
            str(article.get("url", "")),
            _normalise_title(str(article.get("title", ""))),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]


def _dedup_articles(
    articles: list[dict[str, Any]], similarity_threshold: float = 85.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove near-duplicate articles by title similarity."""
    unique: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    for art in articles:
        title = str(art.get("title", ""))
        if not title:
            continue

        best_score = -1.0
        best_match: dict[str, Any] | None = None
        for kept in unique:
            score = _token_set_similarity(title, str(kept.get("title", "")))
            if score > best_score:
                best_score = score
                best_match = kept

        if best_match is not None and best_score >= similarity_threshold:
            duplicate_rows.append(
                {
                    "article_id": art.get("article_id", ""),
                    "source": art.get("source", ""),
                    "source_url": art.get("source_url", art.get("url", "")),
                    "title": title,
                    "published_date": art.get("published_date"),
                    "filter_reason": "duplicate",
                    "matched_article_id": best_match.get("article_id", ""),
                    "dedup_score": round(best_score, 2),
                }
            )
            continue

        unique.append(art)

    return unique, duplicate_rows


def _cache_path(symbol: str) -> Path:
    _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _NEWS_CACHE_DIR / f"{symbol}_news.json"


def _load_cache(symbol: str) -> list[dict[str, Any]] | None:
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
    p = _cache_path(symbol)
    p.write_text(json.dumps(articles, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("News cache saved for {} — {} articles", symbol, len(articles))


def _trace_path(symbol: str, start: str, end: str) -> Path:
    _NEWS_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return _NEWS_TRACE_DIR / f"{symbol}_{safe_start}_{safe_end}_{ts}.csv"


def _export_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "article_id",
        "source",
        "source_url",
        "title",
        "published_date",
        "filter_reason",
        "matched_article_id",
        "dedup_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _http_get(url: str, **kwargs: Any) -> "requests.Response":
    """GET with retry and browser-like headers."""
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


def _cafef_banking_page_candidates(page: int) -> list[str]:
    if page == 1:
        return ["https://cafef.vn/tai-chinh-ngan-hang.chn"]
    return [
        f"https://cafef.vn/tai-chinh-ngan-hang/p{page}.chn",
        f"https://cafef.vn/tai-chinh-ngan-hang/trang-{page}.chn",
    ]


def _scrape_cafef_banking(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int = 30,
) -> list[dict[str, Any]]:
    """Scrape CafeF banking category pages (no keyword filtering)."""
    from bs4 import BeautifulSoup

    candidates: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        page_links: list[str] = []
        for url in _cafef_banking_page_candidates(page):
            try:
                resp = _http_get(url)
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception:
                continue

            links = soup.select("a.box-category-link-title[href], h3 a[href]")
            extracted: list[str] = []
            for link in links:
                href = str(link.get("href", "")).strip()
                if not href:
                    continue
                if href.startswith("/"):
                    href = "https://cafef.vn" + href
                if not href.startswith("http"):
                    continue
                if not href.endswith(".chn"):
                    continue
                if "tai-chinh-ngan-hang" in href and ("p" in href or "trang" in href):
                    continue
                extracted.append(href)

            if extracted:
                page_links = extracted
                break

        if not page_links:
            if page > 1:
                break
            continue

        for href in page_links:
            title = href.rsplit("/", 1)[-1].replace(".chn", "")
            candidates.append(
                {
                    "article_id": _build_article_id(
                        {"source": "cafef_banking", "url": href, "title": title}
                    ),
                    "title": title,
                    "url": href,
                    "source_url": href,
                    "published_date": None,
                    "source": "cafef_banking",
                }
            )

    logger.info("CafeF banking listing: {} candidate articles", len(candidates))

    enriched: list[dict[str, Any]] = []
    for art in candidates[:_MAX_ARTICLES_PER_SOURCE]:
        try:
            content, pub_date = _scrape_cafef_article(art["url"])
            if pub_date is not None:
                art["published_date"] = str(pub_date)
                if pub_date < start_date or pub_date > end_date:
                    continue
            art["content"] = content
            if content and len(content) >= 100:
                enriched.append(art)
        except Exception:
            logger.debug("Failed to fetch CafeF banking article: {}", art.get("url", ""))

    return enriched


def _scrape_cafef_article(url: str) -> tuple[str, pd.Timestamp | None]:
    """Extract full text/date from CafeF article page."""
    from bs4 import BeautifulSoup

    resp = _http_get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    content_div = (
        soup.select_one("div.detail-content")
        or soup.select_one("div.contentdetail")
        or soup.select_one("div#mainContent")
        or soup.select_one("article")
    )
    content = ""
    if content_div:
        for tag in content_div.find_all(["script", "style"]):
            tag.decompose()
        content = content_div.get_text(separator=" ", strip=True)

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
            if not match:
                continue
            try:
                pub_date = pd.to_datetime(match.group(1), errors="coerce")
                if pub_date is not None and pub_date.tzinfo is not None:
                    pub_date = pub_date.tz_localize(None)
                break
            except Exception:
                pass

    if pub_date is None:
        date_tag = soup.select_one("span.pdate") or soup.select_one("span.dateandcate")
        if date_tag:
            try:
                pub_date = pd.to_datetime(date_tag.get_text(strip=True), dayfirst=True, errors="coerce")
            except Exception:
                pass

    return content, pub_date


def _vietstock_page_candidates(symbol: str, page: int) -> list[str]:
    if page == 1:
        return [f"https://vietstock.vn/{symbol}/tin-tuc.htm"]
    return [
        f"https://vietstock.vn/{symbol}/p{page}-tin-tuc.htm",
        f"https://vietstock.vn/{symbol}/tin-tuc/trang-{page}.htm",
    ]


def _scrape_vietstock(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int = 25,
) -> list[dict[str, Any]]:
    """Scrape Vietstock symbol news pages (no keyword filtering)."""
    from bs4 import BeautifulSoup

    candidates: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        page_links: list[str] = []
        for url in _vietstock_page_candidates(symbol, page):
            try:
                resp = _http_get(url)
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception:
                continue

            links = soup.select("a[href$='.htm'][href*='vietstock.vn'], h3 a[href$='.htm'], a.news-title[href]")
            extracted: list[str] = []
            for link in links:
                href = str(link.get("href", "")).strip()
                if not href:
                    continue
                if href.startswith("/"):
                    href = "https://vietstock.vn" + href
                if not href.startswith("http"):
                    continue
                if "/tin-tuc" in href and href.endswith(".htm") and f"/{symbol}/" in href:
                    # keep article urls, skip listing pages
                    if re.search(r"/p\d+-tin-tuc\.htm$", href):
                        continue
                extracted.append(href)

            extracted = [h for h in extracted if not re.search(r"/p\d+-tin-tuc\.htm$", h)]
            extracted = [h for h in extracted if "tin-tuc.htm" not in h or "/tin-tuc/" in h]
            if extracted:
                page_links = extracted
                break

        if not page_links:
            if page > 1:
                break
            continue

        for href in page_links:
            title = href.rsplit("/", 1)[-1].replace(".htm", "")
            candidates.append(
                {
                    "article_id": _build_article_id(
                        {"source": "vietstock", "url": href, "title": title}
                    ),
                    "title": title,
                    "url": href,
                    "source_url": href,
                    "published_date": None,
                    "source": "vietstock",
                }
            )

    logger.info("Vietstock listing [{}]: {} candidate articles", symbol, len(candidates))

    enriched: list[dict[str, Any]] = []
    for art in candidates[:_MAX_ARTICLES_PER_SOURCE]:
        try:
            content, pub_date = _scrape_vietstock_article(art["url"])
            if pub_date is not None:
                art["published_date"] = str(pub_date)
                if pub_date < start_date or pub_date > end_date:
                    continue
            art["content"] = content
            if content and len(content) >= 100:
                enriched.append(art)
        except Exception:
            logger.debug("Failed to fetch Vietstock article: {}", art.get("url", ""))

    return enriched


def _scrape_vietstock_article(url: str) -> tuple[str, pd.Timestamp | None]:
    """Extract full text/date from Vietstock article page."""
    from bs4 import BeautifulSoup

    resp = _http_get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    content_div = (
        soup.select_one("div.pContent")
        or soup.select_one("div#vst_detail")
        or soup.select_one("div.m-content")
        or soup.select_one("article")
    )
    content = ""
    if content_div:
        for tag in content_div.find_all(["script", "style", "figure"]):
            tag.decompose()
        content = content_div.get_text(separator=" ", strip=True)

    pub_date = None

    meta_date = soup.select_one("meta[property='article:published_time']")
    if meta_date:
        try:
            pub_date = pd.to_datetime(meta_date.get("content", ""), errors="coerce")
            if pub_date is not None and pub_date.tzinfo is not None:
                pub_date = pub_date.tz_localize(None)
        except Exception:
            pass

    if pub_date is None:
        time_tag = soup.select_one("time[datetime]")
        if time_tag:
            try:
                pub_date = pd.to_datetime(time_tag.get("datetime", ""), errors="coerce")
                if pub_date is not None and pub_date.tzinfo is not None:
                    pub_date = pub_date.tz_localize(None)
            except Exception:
                pass

    if pub_date is None:
        raw_text = soup.get_text(" ", strip=True)
        match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})", raw_text)
        if match:
            try:
                pub_date = pd.to_datetime(match.group(1), dayfirst=True, errors="coerce")
            except Exception:
                pass

    return content, pub_date


class NewsScraper:
    """Banking-only web news scraper for VCB and MBB."""

    def __init__(self, cache_dir: Path = _NEWS_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def fetch_news(
        self,
        symbol: str,
        start: str,
        end: str,
        sources: tuple[str, ...] = ("cafef_banking", "vietstock"),
        use_cache: bool = True,
        export_trace: bool = True,
        similarity_threshold: float = 85.0,
    ) -> pd.DataFrame:
        """Fetch banking news for one supported bank symbol."""
        symbol = str(symbol).upper().strip()
        if symbol not in _SUPPORTED_BANK_SYMBOLS:
            raise ValueError(
                f"Unsupported symbol '{symbol}'. Supported symbols: {_SUPPORTED_BANK_SYMBOLS}"
            )

        if use_cache:
            cached = _load_cache(symbol)
            if cached is not None:
                df = self._articles_to_dataframe(cached, start, end)
                if not df.empty:
                    return df

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        all_articles: list[dict[str, Any]] = []

        if "cafef_banking" in sources:
            try:
                logger.info("Scraping CafeF banking source for {} ...", symbol)
                all_articles.extend(_scrape_cafef_banking(start_ts, end_ts))
            except Exception:
                logger.warning("CafeF banking scraping failed for {} — continuing", symbol)

        if "vietstock" in sources:
            try:
                logger.info("Scraping Vietstock source for {} ...", symbol)
                all_articles.extend(_scrape_vietstock(symbol, start_ts, end_ts))
            except Exception:
                logger.warning("Vietstock scraping failed for {} — continuing", symbol)

        for art in all_articles:
            art["source_url"] = str(art.get("source_url", art.get("url", "")))
            if not art.get("article_id"):
                art["article_id"] = _build_article_id(art)

        candidate_count = len(all_articles)
        all_articles, duplicate_rows = _dedup_articles(
            all_articles, similarity_threshold=similarity_threshold
        )
        df = self._articles_to_dataframe(all_articles, start, end)

        if use_cache and all_articles:
            _save_cache(symbol, all_articles)

        if export_trace:
            trace_rows: list[dict[str, Any]] = []
            for art in all_articles:
                trace_rows.append(
                    {
                        "article_id": art.get("article_id", ""),
                        "source": art.get("source", ""),
                        "source_url": art.get("source_url", art.get("url", "")),
                        "title": art.get("title", ""),
                        "published_date": art.get("published_date", ""),
                        "filter_reason": "kept",
                        "matched_article_id": "",
                        "dedup_score": "",
                    }
                )
            trace_rows.extend(duplicate_rows)
            if trace_rows:
                trace_file = _trace_path(symbol, start, end)
                _export_trace_csv(trace_file, trace_rows)
                logger.info(
                    "News trace CSV exported: {} (candidates={}, kept={}, duplicates={})",
                    trace_file,
                    candidate_count,
                    len(all_articles),
                    len(duplicate_rows),
                )

        return df

    @staticmethod
    def _articles_to_dataframe(
        articles: list[dict[str, Any]], start: str, end: str
    ) -> pd.DataFrame:
        if not articles:
            return pd.DataFrame(
                columns=[
                    "published_date",
                    "title",
                    "content",
                    "source",
                    "source_url",
                    "article_id",
                    "filter_reason",
                ]
            )

        df = pd.DataFrame(articles)
        df["published_date"] = pd.to_datetime(df.get("published_date"), errors="coerce")
        if df["published_date"].dt.tz is not None:
            df["published_date"] = df["published_date"].dt.tz_localize(None)

        if "title" not in df.columns:
            df["title"] = ""
        if "content" not in df.columns:
            df["content"] = ""
        if "source" not in df.columns:
            df["source"] = ""
        if "source_url" not in df.columns:
            df["source_url"] = ""
        if "article_id" not in df.columns:
            df["article_id"] = ""

        df["filter_reason"] = "kept"

        df = df.dropna(subset=["published_date"])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df["published_date"] >= start_ts) & (df["published_date"] <= end_ts)]

        df = df[
            [
                "published_date",
                "title",
                "content",
                "source",
                "source_url",
                "article_id",
                "filter_reason",
            ]
        ].copy()
        df = df.sort_values("published_date").reset_index(drop=True)
        return df
