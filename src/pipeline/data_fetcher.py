"""CMTF Data Pipeline — Data Fetcher module.

Fetches OHLCV market data and company news from Vietnamese exchanges
using the vnstock library (v3.x).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception_type,
)

# ---------------------------------------------------------------------------
# Disk cache (joblib)
# ---------------------------------------------------------------------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OHLCV overlap-aware disk cache — mirrors news_scraper.py's `_find_overlapping_cache`
# (same problem: every live-inference query uses a slightly different `end`, e.g.
# today's date, so an exact-filename cache almost never hits past the first call,
# forcing a full multi-year network re-fetch, per symbol, on every single live
# query). Unlike news, OHLCV had no local cache at all — this was the dominant
# source of live-query latency (8 symbols incl. VNINDEX x full 2020-> history,
# rate-limited to ~16 req/min). Historical closes are immutable once the trading
# day has settled, so a full-cover cache hit needs no network call at all; a
# partial cover only re-fetches the small incremental tail (plus a short buffer to
# absorb late same-day revisions to the most recent bars).
# ---------------------------------------------------------------------------
_OHLCV_CACHE_DIR = CACHE_DIR / "ohlcv"
_OHLCV_CACHE_FILE_RE = re.compile(r"^([A-Z0-9]+)_([A-Za-z]+)_([A-Za-z0-9]+)_(\d{8})_(\d{8})_ohlcv\.parquet$")


def _ohlcv_cache_path(symbol: str, source: str, interval: str, start: str, end: str) -> Path:
    _OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return _OHLCV_CACHE_DIR / f"{symbol}_{source}_{interval}_{safe_start}_{safe_end}_ohlcv.parquet"


# In-process memoization keyed by the exact (symbol, start, end, interval, source,
# use_cache) call — separate from the on-disk overlap cache above. For a genuinely
# live `end` (e.g. "today"), no on-disk cache can ever be a true "full cover" (the
# market hasn't published that day's bar yet), so every call — including the SAME
# logical request made twice in one process, e.g. `run_pipeline`'s internal fetch
# and `_extract_and_split`'s separate `fetch_multi_symbol` raw-OHLCV fetch moments
# later — would otherwise each hit the network's freshest-tail boundary
# independently. If the real-time feed's "latest available bar" shifts by even one
# row between those two calls (a new bar just posted, or one still settling), the
# two results silently disagree — this is what caused the real
# "N timestamps missing after merge with raw OHLCV" crash. Memoizing the exact call
# guarantees identical repeat calls within one process return the IDENTICAL
# DataFrame, never a second network round trip that could observe a different
# "now".
#
# Bounded (not a plain dict): a single long-running `chat.py --llm` session can
# query many distinct dates over its lifetime, each pulling in a full per-symbol
# OHLCV frame — an unbounded cache here was one contributor to a real MemoryError
# seen in a long session (on top of `_pipeline_splits`'s own bounded lru_cache).
# Small entries (raw OHLCV, not embeddings), so a modest cap is enough headroom.
_INPROCESS_OHLCV_CACHE_MAXSIZE = 32
_INPROCESS_OHLCV_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()


def _ohlcv_memo_get(key: tuple) -> pd.DataFrame | None:
    if key not in _INPROCESS_OHLCV_CACHE:
        return None
    _INPROCESS_OHLCV_CACHE.move_to_end(key)
    return _INPROCESS_OHLCV_CACHE[key]


def _ohlcv_memo_put(key: tuple, value: pd.DataFrame) -> None:
    _INPROCESS_OHLCV_CACHE[key] = value
    _INPROCESS_OHLCV_CACHE.move_to_end(key)
    while len(_INPROCESS_OHLCV_CACHE) > _INPROCESS_OHLCV_CACHE_MAXSIZE:
        _INPROCESS_OHLCV_CACHE.popitem(last=False)


def _find_overlapping_ohlcv_cache(
    symbol: str, source: str, interval: str, start: str,
) -> tuple[Path, pd.Timestamp, pd.Timestamp] | None:
    """Find the on-disk OHLCV cache for (symbol, source, interval) whose cached
    start is <= the requested start, preferring the one with the largest cached
    end (most coverage) — same selection rule as news_scraper's overlap finder."""
    start_ts = pd.Timestamp(start)
    best: tuple[Path, pd.Timestamp, pd.Timestamp] | None = None
    for p in _OHLCV_CACHE_DIR.glob(f"{symbol}_{source}_{interval}_*_ohlcv.parquet"):
        m = _OHLCV_CACHE_FILE_RE.match(p.name)
        if not m:
            continue
        cached_start = pd.to_datetime(m.group(4), format="%Y%m%d")
        cached_end = pd.to_datetime(m.group(5), format="%Y%m%d")
        if cached_start > start_ts:
            continue  # doesn't cover the requested start
        if best is None or cached_end > best[2]:
            best = (p, cached_start, cached_end)
    return best


# ---------------------------------------------------------------------------
# Global vnstock rate limiter (process-wide)
# ---------------------------------------------------------------------------
_RATE_LOCK = threading.Lock()
_REQUEST_TIMESTAMPS: deque[float] = deque()
_RATE_LIMIT_PER_MIN = max(1, int(os.getenv("VNSTOCK_RATE_LIMIT_PER_MIN", "16")))

def _normalize_datetime_index_to_naive(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Normalize a dataframe time column/index to timezone-naive datetime index."""
    out = df.copy()

    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(out[time_col]):
            out[time_col] = out[time_col].dt.tz_localize(None)
        out = out.set_index(time_col)
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)

    out.index.name = "time"
    out = out.sort_index()
    return out

def _throttle_vnstock_requests() -> None:
    """Block until issuing another vnstock request is within minute budget.

    Defaults to 16 req/min (below Guest 20 req/min) for a safety margin.
    Override via environment variable ``VNSTOCK_RATE_LIMIT_PER_MIN``.
    """
    while True:
        now = time.monotonic()
        with _RATE_LOCK:
            # Drop events outside the rolling 60-second window.
            while _REQUEST_TIMESTAMPS and now - _REQUEST_TIMESTAMPS[0] >= 60.0:
                _REQUEST_TIMESTAMPS.popleft()

            if len(_REQUEST_TIMESTAMPS) < _RATE_LIMIT_PER_MIN:
                _REQUEST_TIMESTAMPS.append(now)
                return

            wait_for = 60.0 - (now - _REQUEST_TIMESTAMPS[0])

        # Sleep outside lock so other threads can continue processing.
        if wait_for > 0:
            logger.info(
                "vnstock throttle: {:.1f}s wait ({} req/min limit)",
                wait_for,
                _RATE_LIMIT_PER_MIN,
            )
            time.sleep(wait_for)


class VnstockDataFetcher:
    """Fetches OHLCV and news data from Vietnamese exchanges via vnstock v3.x."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        pass

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _fetch_ohlcv_raw(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1D",
        source: str = "KBS",
    ) -> pd.DataFrame:
        """Low-level fetch with retry logic."""
        from vnstock import Quote

        _throttle_vnstock_requests()
        quote = Quote(symbol=symbol, source=source)
        df = quote.history(start=start, end=end, interval=interval)
        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1D",
        source: str = "KBS",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. ``'VCB'``).
            start: Start date ``'YYYY-MM-DD'``.
            end: End date ``'YYYY-MM-DD'``.
            interval: Bar interval (``'1D'``, ``'1H'``, etc.).
            source: Exchange data source (default ``'KBS'``).
            use_cache: Reuse the on-disk overlap-aware OHLCV cache (see
                `_find_overlapping_ohlcv_cache`) — a full cover skips the network
                fetch entirely; a partial cover only re-fetches the small
                incremental tail (never a redundant multi-year re-fetch just
                because `end` moved forward, e.g. every live-inference query).

        Returns:
            DataFrame indexed by ``time`` (datetime64) with columns
            ``[open, high, low, close, volume]``.
        """
        memo_key = (symbol, start, end, interval, source)
        if use_cache:
            memoized = _ohlcv_memo_get(memo_key)
            if memoized is not None:
                return memoized.copy()

        prior_df: pd.DataFrame | None = None
        fetch_start = start
        if use_cache:
            overlap = _find_overlapping_ohlcv_cache(symbol, source, interval, start)
            if overlap is not None:
                cache_path, cached_start, cached_end = overlap
                end_ts = pd.Timestamp(end)
                try:
                    prior_df = pd.read_parquet(cache_path)
                except Exception:
                    logger.warning("Corrupt OHLCV cache {} — ignoring", cache_path.name)
                    prior_df = None
                if prior_df is not None:
                    if cached_end >= end_ts:
                        logger.info(
                            "OHLCV cache FULL COVER for {} from {} (needed {}..{}) — "
                            "no fetch needed", symbol, cache_path.name, start, end,
                        )
                        out = prior_df[(prior_df.index >= pd.Timestamp(start)) & (prior_df.index <= end_ts)]
                        out = out.sort_index()
                        if use_cache:
                            _ohlcv_memo_put(memo_key, out)
                        return out.copy()
                    # Partial cover: re-fetch a short buffer plus the new tail —
                    # guards against late same-day revisions to the most recently
                    # cached bars (a live day's close can be revised intraday).
                    buffer_days = 5
                    fetch_start = str((cached_end - pd.Timedelta(days=buffer_days)).date())
                    logger.info(
                        "OHLCV cache PARTIAL COVER for {} from {} ({}..{}) — incremental "
                        "fetch only for {}..{} ({} prior rows reused)",
                        symbol, cache_path.name, cached_start.date(), cached_end.date(),
                        fetch_start, end, len(prior_df),
                    )

        logger.info("Fetching OHLCV | {} | {} → {} | {}", symbol, fetch_start, end, interval)
        df = self._fetch_ohlcv_raw(symbol, fetch_start, end, interval, source)

        # Ensure datetime index
        df = _normalize_datetime_index_to_naive(df, time_col="time")

        # Keep only required columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            if col not in df.columns:
                logger.warning("Missing column '{}' for {}", col, symbol)
        df = df[[c for c in expected_cols if c in df.columns]]
        df = df.sort_index()

        if prior_df is not None:
            # Rows before the re-fetched buffer window are untouched; the buffer
            # window onward comes from the fresh fetch (supersedes any prior bars
            # in that overlap, absorbing late revisions).
            df = pd.concat([prior_df[prior_df.index < pd.Timestamp(fetch_start)], df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        if use_cache and not df.empty:
            combined_start = str(df.index[0].date())
            combined_end = str(df.index[-1].date())
            try:
                df.to_parquet(_ohlcv_cache_path(symbol, source, interval, combined_start, combined_end))
            except Exception:
                logger.warning("Failed to write OHLCV cache for {} — continuing without it", symbol)

        out = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

        # Validate missing bars over the actually REQUESTED range (not the
        # possibly-wider fetch_start used for the incremental buffer re-fetch).
        if interval == "1D":
            trading_days = pd.bdate_range(start=start, end=end)
            missing = trading_days.difference(out.index.normalize())
            if len(missing) > 0:
                logger.warning(
                    "{} — {} missing trading days detected (of {} expected)",
                    symbol,
                    len(missing),
                    len(trading_days),
                )

        out = out.sort_index()
        logger.info("OHLCV fetched | {} | {} rows", symbol, len(out))
        if use_cache:
            _ohlcv_memo_put(memo_key, out)
        return out.copy()

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _fetch_news_raw(self, symbol: str, source: str = "VCI") -> pd.DataFrame:
        """Low-level news fetch with retry."""
        from vnstock import Company

        _throttle_vnstock_requests()
        company = Company(symbol=symbol, source=source)
        df = company.news()
        return df

    def fetch_news(
        self, symbol: str, source: str = "VCI"
    ) -> pd.DataFrame:
        """Fetch all available news for a single symbol.

        Args:
            symbol: Ticker symbol.
            source: News source (``'VCI'`` recommended for richer data).

        Returns:
            DataFrame with columns ``[published_date, title, content]``.
            ``published_date`` is timezone-naive datetime64.
        """
        logger.info("Fetching news | {} | source={}", symbol, source)
        df = self._fetch_news_raw(symbol, source)

        if df is None or df.empty:
            logger.warning("No news returned for {}", symbol)
            return pd.DataFrame(columns=["published_date", "title", "content"])

        # --- Normalise published_date ---------------------------------
        date_col_candidates = ["public_date", "published_date", "created_at", "updated_at", "publishedDate", "date", "publish_date"]
        date_col = None
        for candidate in date_col_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            logger.warning("No date column found in news for {}. Columns: {}", symbol, list(df.columns))
            return pd.DataFrame(columns=["published_date", "title", "content"])

        raw_date = df[date_col]
        if pd.api.types.is_numeric_dtype(raw_date):
            # Unix millisecond timestamps (e.g. VCI returns ms since epoch)
            df["published_date"] = pd.to_datetime(raw_date, unit="ms", errors="coerce")
        else:
            df["published_date"] = pd.to_datetime(raw_date, errors="coerce")
        # Strip timezone if present
        if df["published_date"].dt.tz is not None:
            df["published_date"] = df["published_date"].dt.tz_localize(None)

        # --- Normalise title ------------------------------------------
        title_candidates = ["title", "Title", "headline", "news_title"]
        title_col = next((c for c in title_candidates if c in df.columns), None)
        if title_col and title_col != "title":
            df = df.rename(columns={title_col: "title"})

        # --- Normalise content ----------------------------------------
        content_candidates = ["content", "Content", "body", "description", "news_full_content", "news_short_content"]
        content_col = next((c for c in content_candidates if c in df.columns), None)
        if content_col is not None and content_col != "content":
            df = df.rename(columns={content_col: "content"})

        if "content" not in df.columns:
            # Fallback: concatenate title + description if available
            desc_col = next((c for c in ["description", "Description", "summary"] if c in df.columns), None)
            if desc_col is not None:
                df["content"] = df.get("title", "").astype(str) + " " + df[desc_col].astype(str)
            else:
                df["content"] = df.get("title", "").astype(str)
            logger.warning("No 'content' column for {}; using fallback", symbol)

        df = df[["published_date", "title", "content"]].dropna(subset=["published_date"])
        df = df.sort_values("published_date").reset_index(drop=True)
        logger.info("News fetched | {} | {} articles", symbol, len(df))
        return df

    # ------------------------------------------------------------------
    # Multi-source news (web scraping + VCI fallback)
    # ------------------------------------------------------------------
    def fetch_news_multi_source(
        self,
        symbol: str,
        start: str,
        end: str,
        sources: tuple[str, ...] = ("vnexpress", "cafef_banking", "vietstock"),
        use_cache: bool = True,
        export_trace: bool = True,
        similarity_threshold: float = 85.0,
    ) -> pd.DataFrame:
        """Fetch news from banking web sources, with VCI fallback.

        Args:
            symbol: Ticker symbol.
            start: Start date ``'YYYY-MM-DD'``.
            end: End date ``'YYYY-MM-DD'``.
            sources: Web scraping backends to try.
            use_cache: Whether to reuse disk cache in the web scraper.
            export_trace: Whether to export row-level trace CSV.
            similarity_threshold: Near-duplicate title threshold in [0, 100].

        Returns:
            DataFrame with columns ``[published_date, title, content]``.
        """
        from .news_scraper import NewsScraper

        scraper = NewsScraper()
        try:
            df = scraper.fetch_news(
                symbol,
                start,
                end,
                sources=sources,
                use_cache=use_cache,
                export_trace=export_trace,
                similarity_threshold=similarity_threshold,
            )
            if not df.empty:
                logger.info(
                    "Web scraping returned {} articles for {}",
                    len(df),
                    symbol,
                )
                return df
        except Exception:
            logger.warning(
                "Web scraping failed for {} — falling back to VCI", symbol
            )

        # Fallback to vnstock VCI API
        logger.info("Falling back to vnstock VCI news for {}", symbol)
        df_vci = self.fetch_news(symbol, source="VCI")
        if not df_vci.empty:
            df_vci = df_vci[
                (df_vci["published_date"] >= pd.Timestamp(start))
                & (df_vci["published_date"] <= pd.Timestamp(end))
            ]
        return df_vci

    # ------------------------------------------------------------------
    # Multi-symbol
    # ------------------------------------------------------------------
    def fetch_multi_symbol(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1D",
        source: str = "KBS",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            start: Start date.
            end: End date.
            interval: Bar interval.
            source: Data source.

        Returns:
            Dict mapping symbol → OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch_ohlcv(sym, start, end, interval, source)
            except Exception:
                logger.exception("Failed to fetch OHLCV for {}", sym)
        return results
