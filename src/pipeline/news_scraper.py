"""CMTF Data Pipeline — Web News Scraper module.

Simplified banking-only scraper:
- Symbols: VCB, BID
- Sources: CafeF banking category + VnExpress + Vietstock symbol news
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
_SUPPORTED_BANK_SYMBOLS = ("VCB", "BID")
_REQUEST_DELAY = 1.5
_MAX_ARTICLES_PER_SOURCE = 500
_MARKET_CLOSE_HOUR = 15

# VNExpress search keywords per symbol
_VNEXPRESS_KEYWORDS: dict[str, list[str]] = {
    "VCB": ["Vietcombank", "Ngan+hang+Vietcombank", "tin+tuc+Vietcombank", "VCB", "Ngan+hang+Ngoai+thuong"],
    "BID": ["BIDV", "Ngan+hang+BIDV", "tin+tuc+BIDV", "BID", "Ngan+hang+Dau+tu", "Dau+tu+va+Phat+trien", "co+phieu+BIDV", "BIDV+Viet+Nam"],
}

# Sector-wide / macro keywords that affect ALL banking stocks
_VNEXPRESS_SECTOR_KEYWORDS: list[str] = [
    "ngan+hang+nha+nuoc",          # State Bank of Vietnam (SBV)
    "lai+suat+ngan+hang",          # bank interest rates
    "chinh+sach+tien+te",          # monetary policy
    "tin+dung+ngan+hang",          # bank credit
    "co+phieu+ngan+hang",          # banking stocks
    "tang+truong+tin+dung",        # credit growth
    "no+xau+ngan+hang",           # bank bad debt / NPL
    "ty+gia+ngoai+te",            # foreign exchange rates
]

# Brand names / aliases for each symbol (used to filter generic banking news)
_SYMBOL_BRAND_NAMES: dict[str, list[str]] = {
    "VCB": ["Vietcombank", "VCB", "Ngoại thương", "Ngoai thuong"],
    "BID": ["BIDV", "BID", "Đầu tư", "Dau tu", "Ngân hàng Đầu tư"],
}

# Sector keywords in readable form (for CafeF relevance filtering)
_SECTOR_FILTER_KEYWORDS: list[str] = [
    "ngân hàng nhà nước", "lãi suất", "chính sách tiền tệ",
    "tín dụng", "tăng trưởng tín dụng", "nợ xấu",
    "tỷ giá", "ngoại tệ", "cổ phiếu ngân hàng",
    "ngan hang nha nuoc", "lai suat", "tin dung",
    "no xau", "ty gia", "co phieu ngan hang",
]

# Module-level cache so sector news is scraped once per process
_sector_news_cache: dict[str, list[dict[str, Any]]] = {}
# Module-level cache so CafeF banking is scraped once per process
_cafef_banking_cache: dict[str, list[dict[str, Any]]] = {}


def _filter_cafef_by_relevance(
    articles: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    """Keep CafeF banking articles relevant to *symbol* or to sector macro.

    An article is kept if its title or content contains:
    - Any brand name / alias for the target symbol, OR
    - Any sector-wide macro keyword (interest rates, SBV, credit, etc.)
    """
    brand_names = _SYMBOL_BRAND_NAMES.get(symbol.upper(), [])
    kept: list[dict[str, Any]] = []
    for art in articles:
        text = (
            (art.get("title", "") + " " + art.get("content", ""))
            .lower()
        )
        # Check symbol brand names
        if any(bn.lower() in text for bn in brand_names):
            kept.append(art)
            continue
        # Check sector keywords
        if any(kw in text for kw in _SECTOR_FILTER_KEYWORDS):
            kept.append(art)
            continue
    logger.info(
        "CafeF relevance filter for {}: {} → {} articles",
        symbol, len(articles), len(kept),
    )
    return kept


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


def _article_quality_score(article: dict[str, Any]) -> tuple[int, int, int, str]:
    """Return deterministic quality score for dedup keep/drop decision.

    Higher is better. We prefer richer content, then informative title, then
    presence of a timestamp. Final string tiebreak keeps ordering deterministic.
    """
    content_len = len(str(article.get("content", "")).strip())
    title_len = len(str(article.get("title", "")).strip())
    has_timestamp = 1 if pd.notna(pd.to_datetime(article.get("published_date"), errors="coerce")) else 0
    stable_key = f"{article.get('source', '')}|{article.get('url', '')}|{_normalise_title(str(article.get('title', '')))}"
    return (content_len, title_len, has_timestamp, stable_key)


def _effective_trading_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Map publication timestamp to leakage-safe effective trading timestamp.

    Rule:
    - timestamp at or after market close (15:00) -> next calendar day
    - timestamp at 00:00:00 is treated as unknown-time and shifted to next day
    - otherwise, keep same timestamp
    """
    if pd.isna(ts):
        return ts

    ts = pd.Timestamp(ts)
    has_midnight_time = (
        ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0
    )
    if has_midnight_time or ts.hour >= _MARKET_CLOSE_HOUR:
        return (ts.normalize() + pd.Timedelta(days=1))
    return ts


def _dedup_articles(
    articles: list[dict[str, Any]], similarity_threshold: float = 85.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove near-duplicate articles by title similarity.

    The keep/drop decision is quality-aware and deterministic: higher-quality
    articles are processed first so duplicates map to the best available copy.

    Strategy:
    1. Exact URL dedup (O(n)) — fast pass removes obvious duplicates.
    2. Fuzzy title dedup (O(n²)) — capped at _FUZZY_DEDUP_CAP articles to
       avoid quadratic blow-up when thousands of shared macro articles are
       combined with symbol-specific ones.
    """
    _FUZZY_DEDUP_CAP = 3000  # max articles for O(n²) fuzzy pass

    # --- Pass 1: exact URL / article_id dedup (O(n)) ---
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    url_unique: list[dict[str, Any]] = []
    url_dups: list[dict[str, Any]] = []
    for art in articles:
        url = str(art.get("source_url", art.get("url", "")))
        aid = str(art.get("article_id", ""))
        key = url or aid
        if key and key in seen_urls:
            url_dups.append({
                "article_id": aid,
                "source": art.get("source", ""),
                "source_url": url,
                "title": art.get("title", ""),
                "published_date": art.get("published_date"),
                "filter_reason": "duplicate_url",
                "matched_article_id": "",
                "dedup_score": 100.0,
            })
            continue
        if key:
            seen_urls.add(key)
        if aid and aid not in seen_ids:
            seen_ids.add(aid)
        url_unique.append(art)

    # --- Pass 2: fuzzy title dedup (O(n²), capped) ---
    if len(url_unique) > _FUZZY_DEDUP_CAP:
        # Too many articles for fuzzy pass — skip it, return URL-deduped list
        logger.debug(
            "Skipping fuzzy dedup ({} articles > cap {}); URL dedup only",
            len(url_unique), _FUZZY_DEDUP_CAP,
        )
        return url_unique, url_dups

    ranked_articles = sorted(url_unique, key=_article_quality_score, reverse=True)

    unique: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = list(url_dups)

    for art in ranked_articles:
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


def _cache_path(symbol: str, start: str = "", end: str = "") -> Path:
    _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return _NEWS_CACHE_DIR / f"{symbol}_{safe_start}_{safe_end}_news.json"


def _load_cache(symbol: str, start: str = "", end: str = "") -> list[dict[str, Any]] | None:
    p = _cache_path(symbol, start, end)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            logger.info("News cache hit for {} — {} articles", symbol, len(data))
            return data
        except Exception:
            logger.warning("Corrupt news cache for {} — ignoring", symbol)
    return None


def _save_cache(symbol: str, start: str, end: str, articles: list[dict[str, Any]]) -> None:
    p = _cache_path(symbol, start, end)
    p.write_text(
        json.dumps(articles, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
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
    max_pages: int = 100,
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
    max_pages: int = 120,
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


def _scrape_vnexpress(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int = 50,
) -> list[dict[str, Any]]:
    """Scrape VNExpress search results for a symbol using time-bounded query."""
    from bs4 import BeautifulSoup

    keywords = _VNEXPRESS_KEYWORDS.get(symbol.upper(), [symbol])
    fromdate = int(start_date.timestamp())
    todate = int(end_date.timestamp())

    seen_urls: set[str] = set()
    all_articles: list[dict[str, Any]] = []

    for keyword in keywords:
        for page in range(1, max_pages + 1):
            url = (
                f"https://timkiem.vnexpress.net/?q={keyword}"
                f"&media=1&fromdate={fromdate}&todate={todate}&page={page}"
            )
            try:
                resp = _http_get(url)
            except Exception:
                logger.debug("VNExpress search failed for keyword: {} page: {}", keyword, page)
                break

            soup = BeautifulSoup(resp.text, "lxml")
            items = soup.select("article.item-news")
            logger.info("VNExpress search [{}] page {}: {} candidates", keyword, page, len(items))
            if not items:
                break

            for item in items:
                ts_str = item.get("data-publishtime")
                article_url = str(item.get("data-url", ""))
                if not ts_str or not article_url or article_url in seen_urls:
                    continue

                try:
                    pub_date = pd.to_datetime(int(ts_str), unit="s")
                except Exception:
                    continue
                if pub_date < start_date or pub_date > end_date:
                    continue

                title_el = item.select_one("h3.title-news a, h2 a")
                title = title_el.get_text(strip=True) if title_el else ""
                desc_el = item.select_one("p.description a, p.description, p")
                snippet = desc_el.get_text(strip=True) if desc_el else ""
                content = (title + " " + snippet).strip() if snippet else title

                if not title:
                    continue

                seen_urls.add(article_url)
                all_articles.append(
                    {
                        "article_id": _build_article_id(
                            {"source": "vnexpress", "url": article_url, "title": title}
                        ),
                        "title": title,
                        "url": article_url,
                        "source_url": article_url,
                        "published_date": str(pub_date),
                        "content": content,
                        "source": "vnexpress",
                    }
                )

    logger.info("VNExpress scraped {} articles for {}", len(all_articles), symbol)
    return all_articles


def _scrape_vnexpress_sector(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int = 30,
) -> list[dict[str, Any]]:
    """Scrape VNExpress for sector-wide banking/macro news (shared across all symbols)."""
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _sector_news_cache:
        logger.info("VNExpress sector: returning {} cached articles", len(_sector_news_cache[cache_key]))
        return list(_sector_news_cache[cache_key])

    from bs4 import BeautifulSoup

    fromdate = int(start_date.timestamp())
    todate = int(end_date.timestamp())
    seen_urls: set[str] = set()
    all_articles: list[dict[str, Any]] = []

    for keyword in _VNEXPRESS_SECTOR_KEYWORDS:
        for page in range(1, max_pages + 1):
            url = (
                f"https://timkiem.vnexpress.net/?q={keyword}"
                f"&media=1&fromdate={fromdate}&todate={todate}&page={page}"
            )
            try:
                resp = _http_get(url)
            except Exception:
                logger.debug("VNExpress sector search failed: {} page {}", keyword, page)
                break

            soup = BeautifulSoup(resp.text, "lxml")
            items = soup.select("article.item-news")
            logger.info("VNExpress sector [{}] page {}: {} candidates", keyword, page, len(items))
            if not items:
                break

            for item in items:
                ts_str = item.get("data-publishtime")
                article_url = str(item.get("data-url", ""))
                if not ts_str or not article_url or article_url in seen_urls:
                    continue
                try:
                    pub_date = pd.to_datetime(int(ts_str), unit="s")
                except Exception:
                    continue
                if pub_date < start_date or pub_date > end_date:
                    continue

                title_el = item.select_one("h3.title-news a, h2 a")
                title = title_el.get_text(strip=True) if title_el else ""
                desc_el = item.select_one("p.description a, p.description, p")
                snippet = desc_el.get_text(strip=True) if desc_el else ""
                content = (title + " " + snippet).strip() if snippet else title
                if not title:
                    continue

                seen_urls.add(article_url)
                all_articles.append(
                    {
                        "article_id": _build_article_id(
                            {"source": "vnexpress_sector", "url": article_url, "title": title}
                        ),
                        "title": title,
                        "url": article_url,
                        "source_url": article_url,
                        "published_date": str(pub_date),
                        "content": content,
                        "source": "vnexpress_sector",
                    }
                )

    logger.info("VNExpress sector scraped {} macro/banking articles", len(all_articles))
    _sector_news_cache[cache_key] = all_articles
    return list(all_articles)


# ------------------------------------------------------------------
# Google News RSS scraper — broad macro / geopolitical coverage
# ------------------------------------------------------------------

# Symbol-specific queries for Google News
_GOOGLE_NEWS_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "VCB": ["Vietcombank cổ phiếu", "VCB ngân hàng kết quả"],
    "BID": ["BIDV cổ phiếu", "BID ngân hàng đầu tư kết quả"],
}

# Macro / sector / geopolitical queries shared across all symbols.
# These capture events that broadly affect Vietnamese banking stocks:
# SBV rate decisions, inflation, GDP, US Fed, war/trade disruptions,
# government policy — all classes of news the user requested.
_GOOGLE_NEWS_MACRO_KEYWORDS: list[str] = [
    "ngân hàng nhà nước lãi suất Việt Nam",     # SBV rate decisions
    "tỷ giá USD VND Việt Nam",                   # USD/VND exchange rate
    "lạm phát CPI Việt Nam kinh tế",             # Inflation / CPI
    "GDP kinh tế Việt Nam tăng trưởng",          # Economic growth
    "Fed tăng lãi suất ngân hàng thế giới",      # US Fed — affects EM markets
    "chứng khoán ngân hàng Việt Nam VN-Index",   # Banking stocks / market
    "chính phủ Việt Nam chính sách tài chính",   # Govt fiscal policy
    "chiến tranh Nga Ukraine kinh tế thế giới",  # War / commodity shock
    "xuất khẩu nhập khẩu thương mại Việt Nam",  # Trade balance
    "tín dụng nợ xấu ngân hàng Việt Nam",       # Credit / NPL sector
]

# Module-level cache so shared macro queries are fetched once per process
_google_news_macro_cache: dict[str, list[dict[str, Any]]] = {}

# Shorter delay for Google News RSS (public API, no HTML parsing)
_GOOGLE_NEWS_REQUEST_DELAY = 0.5


def _generate_quarterly_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (chunk_start, chunk_end) pairs in 3-month (quarterly) increments."""
    import calendar

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        # Three months ahead
        adv_month = cur.month + 2
        adv_year = cur.year
        while adv_month > 12:
            adv_month -= 12
            adv_year += 1
        last_day = calendar.monthrange(adv_year, adv_month)[1]
        chunk_end = min(
            pd.Timestamp(year=adv_year, month=adv_month, day=last_day, hour=23, minute=59),
            end,
        )
        chunks.append((cur, chunk_end))
        # Advance by 3 months
        next_month = cur.month + 3
        next_year = cur.year
        while next_month > 12:
            next_month -= 12
            next_year += 1
        cur = pd.Timestamp(year=next_year, month=next_month, day=1)
    return chunks


def _fetch_google_news_rss_keyword(
    keyword: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    seen_urls: set[str],
) -> list[dict[str, Any]]:
    """Fetch one Google News RSS query for a single keyword + date chunk."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    from urllib.parse import quote
    import requests as _requests

    after = chunk_start.strftime("%Y-%m-%d")
    before = chunk_end.strftime("%Y-%m-%d")
    query = f"{keyword} after:{after} before:{before}"
    url = (
        f"https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=vi&gl=VN&ceid=VN:vi"
    )

    articles: list[dict[str, Any]] = []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml",
        }
        resp = _requests.get(url, headers=headers, timeout=20)
        time.sleep(_GOOGLE_NEWS_REQUEST_DELAY)
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item"):
            title_raw = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date_str = item.findtext("pubDate", "")
            description = item.findtext("description", "") or ""

            if not title_raw or not link or link in seen_urls:
                continue

            # Google News appends " - Source Name" to titles; strip it
            title = re.sub(r"\s*-\s*[^-]{2,40}$", "", title_raw).strip() or title_raw

            try:
                pub_dt = parsedate_to_datetime(pub_date_str)
                pub_date = pd.Timestamp(pub_dt.replace(tzinfo=None))
            except Exception:
                continue

            if pub_date < chunk_start or pub_date > chunk_end:
                continue

            # Strip HTML tags from description for plain-text content
            content = (title + " " + re.sub(r"<[^>]+>", "", description)).strip()

            seen_urls.add(link)
            articles.append(
                {
                    "article_id": _build_article_id(
                        {"source": "google_news", "url": link, "title": title}
                    ),
                    "title": title,
                    "url": link,
                    "source_url": link,
                    "published_date": str(pub_date),
                    "content": content,
                    "source": "google_news",
                }
            )
    except Exception as exc:
        logger.debug(
            "Google News RSS failed | keyword='{}' chunk={}/{}: {}",
            keyword, after, before, exc,
        )
    return articles


def _scrape_google_news_rss(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Scrape Google News RSS for *symbol*-specific + broad macro/geopolitical news.

    Uses quarterly date chunks so each query returns at most ~100 articles
    and historical coverage reaches back to 2022.  The shared macro queries
    are cached module-level so only the first symbol incurs the fetch cost.
    """
    chunks = _generate_quarterly_chunks(start_date, end_date)
    cache_key = f"{start_date.date()}_{end_date.date()}"

    # --- Shared macro queries (fetched once, reused for every symbol) ---
    if cache_key not in _google_news_macro_cache:
        logger.info("Google News RSS: fetching {} macro keywords × {} quarterly chunks …",
                    len(_GOOGLE_NEWS_MACRO_KEYWORDS), len(chunks))
        seen: set[str] = set()
        macro_arts: list[dict[str, Any]] = []
        for kw in _GOOGLE_NEWS_MACRO_KEYWORDS:
            for cs, ce in chunks:
                macro_arts.extend(_fetch_google_news_rss_keyword(kw, cs, ce, seen))
        _google_news_macro_cache[cache_key] = macro_arts
        logger.info("Google News RSS macro cache: {} articles", len(macro_arts))

    shared = list(_google_news_macro_cache[cache_key])

    # --- Symbol-specific queries ---
    sym_keywords = _GOOGLE_NEWS_SYMBOL_KEYWORDS.get(symbol.upper(), [symbol])
    seen_sym: set[str] = set()
    sym_arts: list[dict[str, Any]] = []
    for kw in sym_keywords:
        for cs, ce in chunks:
            sym_arts.extend(_fetch_google_news_rss_keyword(kw, cs, ce, seen_sym))

    all_articles = sym_arts + shared
    logger.info(
        "Google News RSS for {}: {} symbol-specific + {} macro = {} total",
        symbol, len(sym_arts), len(shared), len(all_articles),
    )
    return all_articles


class NewsScraper:
    """Banking web news scraper with symbol-specific + sector-wide macro coverage."""

    def __init__(self) -> None:
        pass

    def fetch_news(
        self,
        symbol: str,
        start: str,
        end: str,
        sources: tuple[str, ...] = ("vnexpress", "cafef_banking", "vietstock"),
        use_cache: bool = True,
        export_trace: bool = True,
        similarity_threshold: float = 85.0,
        news_filter_by_symbol: bool = True,
    ) -> pd.DataFrame:
        """Fetch banking news for one supported bank symbol."""
        symbol = str(symbol).upper().strip()
        if symbol not in _SUPPORTED_BANK_SYMBOLS:
            raise ValueError(
                f"Unsupported symbol '{symbol}'. Supported symbols: {_SUPPORTED_BANK_SYMBOLS}"
            )

        if use_cache:
            cached = _load_cache(symbol, start, end)
            if cached is not None:
                df = self._articles_to_dataframe(cached, start, end)
                if not df.empty:
                    return df

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        all_articles: list[dict[str, Any]] = []

        if "vnexpress" in sources:
            try:
                logger.info("Scraping VNExpress for {} ...", symbol)
                all_articles.extend(_scrape_vnexpress(symbol, start_ts, end_ts))
            except Exception:
                logger.warning("VNExpress scraping failed for {} — continuing", symbol)

        if "cafef_banking" in sources:
            try:
                cafef_key = f"{start_ts}_{end_ts}"
                if cafef_key in _cafef_banking_cache:
                    logger.info("CafeF banking: returning {} cached articles for {}", len(_cafef_banking_cache[cafef_key]), symbol)
                    cafef_raw = list(_cafef_banking_cache[cafef_key])
                else:
                    logger.info("Scraping CafeF banking source for {} ...", symbol)
                    cafef_raw = _scrape_cafef_banking(start_ts, end_ts)
                    _cafef_banking_cache[cafef_key] = cafef_raw
                cafef_filtered = _filter_cafef_by_relevance(cafef_raw, symbol) if news_filter_by_symbol else cafef_raw
                all_articles.extend(cafef_filtered)
            except Exception:
                logger.warning("CafeF banking scraping failed for {} — continuing", symbol)

        if "vietstock" in sources:
            try:
                logger.info("Scraping Vietstock source for {} ...", symbol)
                all_articles.extend(_scrape_vietstock(symbol, start_ts, end_ts))
            except Exception:
                logger.warning("Vietstock scraping failed for {} — continuing", symbol)

        if "google_news" in sources:
            try:
                logger.info("Scraping Google News RSS for {} ...", symbol)
                all_articles.extend(_scrape_google_news_rss(symbol, start_ts, end_ts))
            except Exception:
                logger.warning("Google News RSS scraping failed for {} — continuing", symbol)

        # Add sector-wide banking/macro news (shared across all symbols)
        if "vnexpress" in sources:
            try:
                sector_articles = _scrape_vnexpress_sector(start_ts, end_ts, max_pages=10)
                logger.info("VNExpress sector: {} macro/banking articles for {}", len(sector_articles), symbol)
                all_articles.extend(sector_articles)
            except Exception:
                logger.warning("VNExpress sector scraping failed — continuing")

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
            _save_cache(symbol, start, end, all_articles)

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
        df["published_date"] = df["published_date"].apply(_effective_trading_timestamp)

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
