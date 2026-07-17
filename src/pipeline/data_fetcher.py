"""CMTF Data Pipeline — Data Fetcher module.

Fetches OHLCV market data and company news from Vietnamese exchanges
using the vnstock library (v3.x).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
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
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. ``'VCB'``).
            start: Start date ``'YYYY-MM-DD'``.
            end: End date ``'YYYY-MM-DD'``.
            interval: Bar interval (``'1D'``, ``'1H'``, etc.).
            source: Exchange data source (default ``'KBS'``).

        Returns:
            DataFrame indexed by ``time`` (datetime64) with columns
            ``[open, high, low, close, volume]``.
        """
        logger.info("Fetching OHLCV | {} | {} → {} | {}", symbol, start, end, interval)
        df = self._fetch_ohlcv_raw(symbol, start, end, interval, source)

        # Ensure datetime index
        df = _normalize_datetime_index_to_naive(df, time_col="time")

        # Keep only required columns
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            if col not in df.columns:
                logger.warning("Missing column '{}' for {}", col, symbol)
        df = df[[c for c in expected_cols if c in df.columns]]

        # Validate missing bars
        if interval == "1D":
            trading_days = pd.bdate_range(start=start, end=end)
            missing = trading_days.difference(df.index.normalize())
            if len(missing) > 0:
                logger.warning(
                    "{} — {} missing trading days detected (of {} expected)",
                    symbol,
                    len(missing),
                    len(trading_days),
                )

        df = df.sort_index()
        logger.info("OHLCV fetched | {} | {} rows", symbol, len(df))
        return df

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
