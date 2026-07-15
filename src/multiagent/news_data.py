"""Shared real-news loading (used by the metalabel agent and its backtest harness).

Reads the raw article text/titles from ``cache/news/{SYMBOL}_*.json`` — never a
standardized/z-scored column (that bug crippled the H3 forecaster brief once
already; see h3_faithfulness.py's correction note). Headlines are filtered to a
lookback window strictly at/before the cutoff (no look-ahead).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

_NEWS_DIR = Path("cache/news")


def load_news_index() -> dict:
    """symbol -> DataFrame(title, pub) from the widest news file per symbol."""
    import pandas as pd
    idx = {}
    for f in sorted(glob.glob(str(_NEWS_DIR / "*_news.json"))):
        sym = Path(f).name.split("_")[0]
        if sym in idx:
            continue  # first (widest date range) wins
        arts = json.load(open(f, encoding="utf-8"))
        df = pd.DataFrame(arts)
        if "published_date" not in df or "title" not in df:
            continue
        df["pub"] = pd.to_datetime(df["published_date"], errors="coerce")
        idx[sym] = df.dropna(subset=["pub"]).sort_values("pub")
    return idx


def recent_headlines(news_index: dict, symbol: str, cutoff: str,
                     lookback_days: int = 5, k: int = 15) -> list[str]:
    """Most recent k headlines published at/before cutoff, within lookback_days."""
    import pandas as pd
    df = news_index.get(symbol)
    if df is None or df.empty:
        return []
    c = pd.Timestamp(cutoff)
    win = df[(df["pub"] <= c) & (df["pub"] > c - pd.Timedelta(days=lookback_days))]
    win = win.tail(k)
    return [f"({r.pub.date()}) {str(r.title)[:140]}" for r in win.itertuples()]
