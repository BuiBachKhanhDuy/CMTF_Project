"""CMTF Data Pipeline — Temporal Aligner module.

Assigns news articles to OHLCV bars with conservative leakage prevention.
News at/after market close on day T is only visible to bar T+1 or later.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

# Vietnam trading session (ICT / UTC+7)
_MARKET_CLOSE_HOUR = 15  # 15:00


class TemporalAligner:
    """Aligns news articles to OHLCV bars without look-ahead leakage.

    Rules:
                * Daily bars: market-close cutoff.
                    News before 15:00 on day T → bar T.
                    News at/after 15:00 on day T → bar T+1.
                    Date-only timestamps (00:00:00) are treated as unknown-time and
                    shifted to T+1 conservatively.
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

        if ohlcv.index.tz is None:
            ohlcv.index = ohlcv.index.tz_localize("Asia/Ho_Chi_Minh")
        else:
            ohlcv.index = ohlcv.index.tz_convert("Asia/Ho_Chi_Minh")

        # Initialise output columns
        ohlcv["news_count"] = 0
        ohlcv["news_titles"] = [[] for _ in range(len(ohlcv))]
        ohlcv["news_content"] = [[] for _ in range(len(ohlcv))]
        ohlcv["has_news"] = False

        if df_news is None or df_news.empty:
            logger.info("No news to align — returning empty news columns")
            return ohlcv

        news = df_news.copy()
        news["published_date"] = pd.to_datetime(news["published_date"], utc=True)
        news["published_date"] = news["published_date"].dt.tz_convert("Asia/Ho_Chi_Minh")
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

        Market-close cutoff:
        - before 15:00 on day T -> bar T
        - at/after 15:00 on day T -> bar T+1
        - date-only timestamps at 00:00 are shifted to T+1 conservatively
        Weekend / holiday news → next trading bar.

        Vectorized equivalent of the original per-article Python loop (which
        recomputed ``ohlcv.index.normalize()`` — an O(bars) scan — once per
        article, i.e. O(articles x bars); with ~18k articles x ~1.6k bars per
        symbol that was ~10-13s of the live-inference latency on its own). Same
        market-close/leakage rules, same "first eligible bar" tie-break, same
        chronological append order within a bar — verified against
        ``TestAlignerNoLeakage`` (same-day, premarket, weekend, after-hours,
        date-only cases).
        """
        bar_dates = bar_times.normalize().unique().sort_values()
        if bar_dates.empty or news.empty:
            return ohlcv

        pub = pd.to_datetime(news["published_date"])
        valid_pub = pub.notna()
        pub_date = pub.dt.normalize()
        is_midnight = (
            (pub.dt.hour == 0) & (pub.dt.minute == 0) & (pub.dt.second == 0) & (pub.dt.microsecond == 0)
        )
        after_close = pub.dt.hour >= _MARKET_CLOSE_HOUR
        eligible_date = pub_date.where(~(is_midnight | after_close), pub_date + pd.Timedelta(days=1))

        # First bar_date >= eligible_date, for every article at once (bar_dates is
        # sorted, so this is the vectorized form of `bar_dates[bar_dates >= x][0]`).
        positions = bar_dates.searchsorted(eligible_date.to_numpy(), side="left")
        has_target = valid_pub.to_numpy() & (positions < len(bar_dates))
        if not has_target.any():
            return ohlcv

        target_dates = bar_dates[positions[has_target]]

        # First ohlcv row for each calendar date — mirrors the original's
        # `ohlcv.index[ohlcv.index.normalize() == target_date][0]`.
        date_to_idx = pd.Series(ohlcv.index, index=ohlcv.index.normalize()).groupby(level=0).first()
        target_idx = date_to_idx.reindex(target_dates).to_numpy()

        assigned = news.loc[has_target].copy()
        assigned["_bar_idx"] = target_idx
        # `news` was already sorted by published_date before this call, and
        # groupby preserves within-group row order, so titles/content land in
        # the same chronological order the original one-at-a-time loop produced.
        for idx, group in assigned.groupby("_bar_idx", sort=False):
            titles = group["title"].astype(str).tolist() if "title" in group else [""] * len(group)
            contents = group["content"].astype(str).tolist() if "content" in group else [""] * len(group)
            ohlcv.at[idx, "news_count"] += len(group)
            ohlcv.at[idx, "news_titles"] = ohlcv.at[idx, "news_titles"] + titles
            ohlcv.at[idx, "news_content"] = ohlcv.at[idx, "news_content"] + contents
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
