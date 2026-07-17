"""Shared real-news loading (used by the metalabel agent and its backtest harness).

Reads the raw article text/titles from ``cache/news/{SYMBOL}_*.json`` — never a
standardized/z-scored column (that bug crippled the H3 forecaster brief once
already; see h3_faithfulness.py's correction note). Headlines are filtered to a
lookback window strictly at/before the cutoff (no look-ahead).
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

_NEWS_DIR = Path("cache/news")
_NEWS_FILE_RE = re.compile(r"^([A-Z]+)_(\d{8})_(\d{8})_news\.json$")


def load_news_index() -> dict:
    """symbol -> DataFrame(title, pub) from the widest news file per symbol.

    Multiple cache files can exist per symbol (e.g. VCB_..._20250813_news.json AND
    VCB_..._20260410_news.json, from different runs' end dates — see
    news_scraper.py's overlap-aware cache). Picking by ``sorted(glob(...))`` order
    (the previous behaviour) actually picks the alphabetically-first match, which is
    the SMALLEST end-date, not the widest range, despite the old comment's claim —
    confirmed: it was silently serving 2025-08 as VCB's "latest" news while a
    2026-04 file sat right next to it. Parse each filename's own end-date and pick
    the true maximum instead.
    """
    import pandas as pd
    best_end: dict[str, str] = {}
    best_path: dict[str, str] = {}
    for f in glob.glob(str(_NEWS_DIR / "*_news.json")):
        m = _NEWS_FILE_RE.match(Path(f).name)
        sym, end = (m.group(1), m.group(3)) if m else (Path(f).name.split("_")[0], "")
        if sym not in best_end or end > best_end[sym]:
            best_end[sym] = end
            best_path[sym] = f

    idx = {}
    for sym, f in best_path.items():
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


def recent_articles(news_index: dict, symbol: str, cutoff: str,
                    lookback_days: int = 14, k: int = 30) -> list[dict]:
    """Structured (title, published_at) dicts for the RESEARCH branch's article
    retrieval — same filtering as ``recent_headlines`` but structured, not
    pre-formatted into a display string, since ``research_agent_node`` ranks and
    cites by published date rather than just printing titles."""
    import pandas as pd
    df = news_index.get(symbol)
    if df is None or df.empty:
        return []
    c = pd.Timestamp(cutoff)
    win = df[(df["pub"] <= c) & (df["pub"] > c - pd.Timedelta(days=lookback_days))]
    win = win.tail(k)
    return [
        {"id": f"{symbol}-{i}", "title": str(r.title), "published_at": str(r.pub.date())}
        for i, r in enumerate(win.itertuples())
    ]


def articles_in_range(news_index: dict, symbol: str, date_start: str, date_end: str,
                      k: int = 40) -> list[dict]:
    """Structured (title, published_at) dicts published within an EXPLICIT calendar
    range — the query's own [date_start, date_end], not a fixed lookback window
    counted backward from a single cutoff. This is what lets "phân tích tháng 3"
    retrieve March's actual news instead of whatever fell inside some fixed N-day
    lookback from wherever the date resolver happened to land.

    Two things beyond a raw filter, both needed for a usable range summary:
      1. De-duplicate by title — the same wire story is often syndicated under
         near-identical headlines on the same day (e.g. two identical "VN-Index
         tạo đáy" rows), which would otherwise crowd out distinct stories.
      2. Spread the sample EVENLY across the range instead of ``.tail(k)`` — a
         busy month can have 800+ articles; taking the last k grabbed only the
         final day or two, so "analyze March" silently became "analyze March 30".
         Even sampling keeps early-, mid-, and late-month coverage.
    """
    import numpy as np
    import pandas as pd
    df = news_index.get(symbol)
    if df is None or df.empty:
        return []
    lo, hi = pd.Timestamp(date_start), pd.Timestamp(date_end)
    win = df[(df["pub"] >= lo) & (df["pub"] <= hi)].drop_duplicates(subset=["title"])
    n = len(win)
    if n == 0:
        return []
    if n > k:
        sel = np.linspace(0, n - 1, k).round().astype(int)
        win = win.iloc[np.unique(sel)]
    return [
        {"id": f"{symbol}-{i}", "title": str(r.title), "published_at": str(r.pub.date())}
        for i, r in enumerate(win.itertuples())
    ]
