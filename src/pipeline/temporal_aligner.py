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

        print("=" * 80)
        print("Index unique:", ohlcv.index.is_unique)
        print("Duplicated:", ohlcv.index.duplicated().sum())

        if "symbol" in ohlcv.columns:
            print(ohlcv[["symbol"]].head(20))
        else:
            print("No symbol column")

        print(ohlcv.head())
        print("=" * 80)

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
        news["published_date"] = news["published_date"].dt.tz_convert(
            "Asia/Ho_Chi_Minh"
        )
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
        """
        # Build a mapping: for each news article determine the eligible bar
        bar_dates = bar_times.normalize().unique().sort_values()

        for _, article in news.iterrows():
            pub = pd.Timestamp(article["published_date"])
            pub_date = pub.normalize()

            has_midnight_time = (
                pub.hour == 0
                and pub.minute == 0
                and pub.second == 0
                and pub.microsecond == 0
            )
            if has_midnight_time or pub.hour >= _MARKET_CLOSE_HOUR:
                # Unknown exact time or after close -> next bar conservatively
                eligible_date = pub_date + pd.Timedelta(days=1)
            else:
                eligible_date = pub_date

            # Find the next available trading bar on or after eligible_date
            future_bars = bar_dates[bar_dates >= eligible_date]
            if future_bars.empty:
                continue  # no eligible bar in dataset

            target_date = future_bars[0]

            # Match to ohlcv rows whose date equals target_date
            # mask = ohlcv.index.normalize() == target_date
            # idxs = ohlcv.index[mask]
            # if idxs.empty:
            #     continue

            # idx = idxs[0]
            positions = (ohlcv.index.normalize() == target_date).nonzero()[0]

            if len(positions) == 0:
                continue

            pos = positions[0]

            # idx = idxs[0]
            # pos = ohlcv.index.get_loc(idx)
            # if isinstance(pos, slice):
            #     pos = pos.start  # type: ignore[assignment]

            # ohlcv.at[idx, "news_count"] += 1
            # ohlcv.at[idx, "news_titles"] = ohlcv.at[idx, "news_titles"] + [
            #     str(article.get("title", ""))
            # ]
            # ohlcv.at[idx, "news_content"] = ohlcv.at[idx, "news_content"] + [
            #     str(article.get("content", ""))
            # ]
            # ohlcv.at[idx, "has_news"] = True

            # luôn lấy vị trí integer
            # pos = ohlcv.index.get_indexer([idx])[0]

            # ohlcv.iat[pos, ohlcv.columns.get_loc("news_count")] += 1

            # titles = list(ohlcv.iat[pos, ohlcv.columns.get_loc("news_titles")])
            # titles.append(str(article.get("title", "")))
            # ohlcv.iat[pos, ohlcv.columns.get_loc("news_titles")] = titles

            # contents = list(ohlcv.iat[pos, ohlcv.columns.get_loc("news_content")])
            # contents.append(str(article.get("content", "")))
            # ohlcv.iat[pos, ohlcv.columns.get_loc("news_content")] = contents

            # ohlcv.iat[pos, ohlcv.columns.get_loc("has_news")] = True

            news_col = ohlcv.columns.get_loc("news_titles")
            content_col = ohlcv.columns.get_loc("news_content")
            count_col = ohlcv.columns.get_loc("news_count")
            has_col = ohlcv.columns.get_loc("has_news")

            ohlcv.iat[pos, count_col] += 1

            titles = list(ohlcv.iat[pos, news_col])
            titles.append(str(article.get("title", "")))
            ohlcv.iat[pos, news_col] = titles

            contents = list(ohlcv.iat[pos, content_col])
            contents.append(str(article.get("content", "")))
            ohlcv.iat[pos, content_col] = contents

            ohlcv.iat[pos, has_col] = True

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
