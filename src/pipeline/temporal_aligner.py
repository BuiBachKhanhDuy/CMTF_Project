"""CMTF Data Pipeline — Temporal Aligner module.

Assigns news articles to OHLCV bars with STRICT leakage prevention.
News published on day T is only visible to bar T+1 or later.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Vietnam trading session (ICT / UTC+7)
_MARKET_OPEN_HOUR = 9   # 09:00
_MARKET_CLOSE_HOUR = 15  # 15:00


class TemporalAligner:
    """Aligns news articles to OHLCV bars without look-ahead leakage.

    Rules:
        * Daily bars: news on day T-1 or earlier → bar T.
          News on day T (same day, during/after market) → bar T+1.
          Pre-market news (before 09:00 on day T) → bar T.
        * Intraday bars: strict ``news_ts < bar_open_ts``.
        * Weekend / holiday news → next available trading bar.
    """

    @staticmethod
    def assign_news_to_bars(
        df_ohlcv: pd.DataFrame,
        df_news: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign news articles to OHLCV bars without leakage.

        Args:
            df_ohlcv: OHLCV DataFrame indexed by ``time`` (datetime64).
            df_news: News DataFrame with columns
                ``[published_date, title, content]``.

        Returns:
            DataFrame with the same index as *df_ohlcv* plus new columns:
            ``news_count``, ``news_titles``, ``news_content``, ``has_news``.
        """
        ohlcv = df_ohlcv.copy()
        ohlcv.index = pd.to_datetime(ohlcv.index)

        # Initialise output columns
        ohlcv["news_count"] = 0
        ohlcv["news_titles"] = [[] for _ in range(len(ohlcv))]
        ohlcv["news_content"] = [[] for _ in range(len(ohlcv))]
        ohlcv["has_news"] = False

        if df_news is None or df_news.empty:
            logger.info("No news to align — returning empty news columns")
            return ohlcv

        news = df_news.copy()
        news["published_date"] = pd.to_datetime(news["published_date"])
        news = news.sort_values("published_date")

        bar_times = ohlcv.index.sort_values()
        is_daily = TemporalAligner._is_daily(bar_times)

        if is_daily:
            ohlcv = TemporalAligner._assign_daily(ohlcv, news, bar_times)
        else:
            ohlcv = TemporalAligner._assign_intraday(ohlcv, news, bar_times)

        assigned = int(ohlcv["has_news"].sum())
        logger.info(
            "News alignment complete | {} / {} bars have news",
            assigned,
            len(ohlcv),
        )
        return ohlcv

    # ------------------------------------------------------------------
    # Daily assignment
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_daily(
        ohlcv: pd.DataFrame,
        news: pd.DataFrame,
        bar_times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Assign news to daily bars respecting Vietnam trading hours.

        Pre-market news (before 09:00 on day T) → bar T.
        Same-day news at or after 09:00 on day T → bar T+1.
        Weekend / holiday news → next trading bar.
        """
        # Build a mapping: for each news article determine the eligible bar
        bar_dates = bar_times.normalize().unique().sort_values()

        for _, article in news.iterrows():
            pub = article["published_date"]
            pub_date = pd.Timestamp(pub).normalize()
            pub_hour = pd.Timestamp(pub).hour

            if pub_hour < _MARKET_OPEN_HOUR:
                # Pre-market → same-day bar
                eligible_date = pub_date
            else:
                # During or after market → next bar
                eligible_date = pub_date + pd.Timedelta(days=1)

            # Find the next available trading bar on or after eligible_date
            future_bars = bar_dates[bar_dates >= eligible_date]
            if future_bars.empty:
                continue  # no eligible bar in dataset

            target_date = future_bars[0]

            # Match to ohlcv rows whose date equals target_date
            mask = ohlcv.index.normalize() == target_date
            idxs = ohlcv.index[mask]
            if idxs.empty:
                continue

            idx = idxs[0]
            pos = ohlcv.index.get_loc(idx)
            if isinstance(pos, slice):
                pos = pos.start  # type: ignore[assignment]

            ohlcv.at[idx, "news_count"] += 1
            ohlcv.at[idx, "news_titles"] = ohlcv.at[idx, "news_titles"] + [
                str(article.get("title", ""))
            ]
            ohlcv.at[idx, "news_content"] = ohlcv.at[idx, "news_content"] + [
                str(article.get("content", ""))
            ]
            ohlcv.at[idx, "has_news"] = True

        return ohlcv

    # ------------------------------------------------------------------
    # Intraday assignment
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_intraday(
        ohlcv: pd.DataFrame,
        news: pd.DataFrame,
        bar_times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Assign news to intraday bars: strictly news_ts < bar_open_ts."""
        sorted_bars = bar_times.sort_values()

        for _, article in news.iterrows():
            pub = pd.Timestamp(article["published_date"])

            # Find the first bar whose open time is strictly after pub
            future = sorted_bars[sorted_bars > pub]
            if future.empty:
                continue

            target_bar = future[0]
            idx = target_bar

            ohlcv.at[idx, "news_count"] += 1
            ohlcv.at[idx, "news_titles"] = ohlcv.at[idx, "news_titles"] + [
                str(article.get("title", ""))
            ]
            ohlcv.at[idx, "news_content"] = ohlcv.at[idx, "news_content"] + [
                str(article.get("content", ""))
            ]
            ohlcv.at[idx, "has_news"] = True

        return ohlcv

    # ------------------------------------------------------------------
    # Null-news mask
    # ------------------------------------------------------------------
    @staticmethod
    def add_null_mask(df_aligned: pd.DataFrame) -> pd.DataFrame:
        """Add ``news_missing_flag`` column for [NO_NEWS] token injection.

        Args:
            df_aligned: Output of :meth:`assign_news_to_bars`.

        Returns:
            Same DataFrame with an additional boolean column
            ``news_missing_flag`` (True where ``news_count == 0``).
        """
        df = df_aligned.copy()
        df["news_missing_flag"] = df["news_count"] == 0
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_daily(bar_times: pd.DatetimeIndex) -> bool:
        """Heuristic: if median bar gap ≥ 20 hours, treat as daily."""
        if len(bar_times) < 2:
            return True
        diffs = bar_times.sort_values().diff().dropna()
        median_gap = diffs.median()
        return median_gap >= pd.Timedelta(hours=20)
