#!/usr/bin/env python
"""Interactive Multi-Agent System chat — run ONE file, type queries, watch every node work.

Run:
    .venv/Scripts/python.exe chat.py            # fast (deterministic, grounded answer, no LLM)
    .venv/Scripts/python.exe chat.py --llm       # real LLM narration + metalabel news check (needs Ollama)
    streamlit run chat.py                        # web interface (Streamlit UI)

Then just type, e.g.:
    VCB                          -> latest cached date, 5d
    VCB 2025-08-13               -> that date
    BID có nên mua 5 ngày tới    -> natural Vietnamese, symbol+horizon parsed
    rank VCB,BID,CTG 2025-08-13  -> cross-sectional ranking branch
    help / symbols / quit

After every answer it prints the full node-by-node trace with each step's result, so
you can inspect exactly what the system did (predict -> gate -> horizon_interaction ->
risk -> metalabel -> narrator -> critic -> reasoning).

Dates inside the research book (shown in the banner) are served instantly from the
frozen prediction cache. A date OUTSIDE the book triggers a real live forward pass of
the deployed champion — the pipeline fetches live OHLCV + scrapes news for the whole
universe, so it can take many minutes (cold). Predictions are always real, never faked;
if data for a live date is unavailable the system reports it and abstains.
"""

from __future__ import annotations

import traceback
import argparse
import contextlib
import io
import re
import sys
import urllib.request
import json
from dataclasses import replace as _dc_replace
from datetime import datetime

# Vietnamese narration/answers contain characters outside cp1252 (Windows' default
# console codepage), which crashes print() before a single query is even processed
# when stdout isn't already UTF-8 (e.g. piped/redirected). Force UTF-8 unconditionally.
# stdin needs the same fix: Windows' default console codepage mangles Vietnamese
# diacritics on the way IN too, silently corrupting the query text before any intent
# classifier or keyword matcher ever sees it — a plain-ASCII English query works by
# accident (ASCII bytes are identical across codepages), which is exactly why this
# went unnoticed until a Vietnamese query was tried.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
# pytest (and some other embedding harnesses) replace stdin with an object that
# doesn't support reconfigure() — only touch it when the real interactive stream
# supports it, so chat.py stays importable for testing.
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# Web imports (only needed for web mode)
try:
    import tornado.ioloop
    import tornado.web
    import tornado.httpserver
    import tornado.websocket
    import tornado.escape

    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False

import numpy as np
import pandas as pd

# Quiet the noisy third-party import banners before anything else loads them.
from loguru import logger

logger.remove()  # chat prints its own clean step view; suppress per-node INFO spam

from src.multiagent.config import MultiAgentConfig
from src.multiagent.frozen_predictions import get_store, PredictionNotCachedError
from src.multiagent.loaders import ArtifactMissingError
from src.multiagent.news_data import (
    load_news_index,
    recent_headlines,
    articles_in_range,
)
from src.multiagent.agents.orchestrator_agent import orchestrator_node
from src.multiagent.agents.predict_agent import predict_agent_node
from src.multiagent.agents.gate_agent import gate_agent_node
from src.multiagent.agents.horizon_interaction_agent import (
    horizon_interaction_agent_node,
)
from src.multiagent.agents.risk_agent import risk_agent_node
from src.multiagent.agents.metalabel_agent import metalabel_agent_node
from src.multiagent.agents.reasoning_agent import reasoning_agent_node
from src.multiagent.agents.narrator_agent import narrator_agent_node
from src.multiagent.agents.critic_agent import critic_agent_node
from src.multiagent.agents.rank_agent import rank_agent_node
from src.multiagent.agents.market_agent import compute_range_stats
from src.multiagent.trace import summarize_node
from src.multiagent.live_inference import resolve_price_parquet

KNOWN_SYMBOLS = ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB"]
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

DECISION_CHAIN = [
    ("predict_agent", predict_agent_node),
    ("gate_agent", gate_agent_node),
    ("horizon_interaction_agent", horizon_interaction_agent_node),
    ("risk_agent", risk_agent_node),
    ("metalabel_agent", metalabel_agent_node),
    ("narrator", narrator_agent_node),
    ("critic_agent", critic_agent_node),
    ("reasoning_agent", reasoning_agent_node),
]
# reasoning_agent runs LAST (needs the real critic_status — a node before critic_agent
# can never see it, since it hasn't run yet). Its widen-and-rerun callback therefore
# re-runs EVERYTHING except reasoning_agent itself — predict through critic_agent, a
# real second pass including fresh narration + verification, not a partial one.
_RERUN_CHAIN = DECISION_CHAIN[:7]
# The decision core alone (predict -> gate -> horizon_interaction -> risk ->
# metalabel), no narration/verification — used by RESEARCH's gap-fill forecast
# (`_run_prediction_for_gap`), which only needs the trade decision itself, never a
# user-facing narrated answer. Deliberately a SEPARATE slice from `_RERUN_CHAIN`
# (which now includes narrator/critic for reasoning_agent's different purpose).
_DECISION_ONLY_CHAIN = DECISION_CHAIN[:5]


def _symbol_dates(store):
    by = {}
    for sym, d in store._index:
        by.setdefault(sym, []).append(str(d))
    return {s: sorted(v) for s, v in by.items()}


def _real_vol_dd(frame, symbol, cutoff, window=20):
    """Real trailing `window`-day vol% + drawdown% + trend% from price history, or
    (None, None, None) if the date is beyond the price parquet (a live date) — never
    fabricated as 0.0, which the risk agent would read as 'calm' and skip its safety
    veto, and which used to make every answer report a fake flat 0.0% trend.

    `window` is overridable (not just the default 20) — `reasoning_agent`'s
    widen-and-rerun path re-derives this from a WIDER already-loaded slice of the
    same in-memory `frame` (no new fetch) when the initial evidence looked thin."""

    s = frame[
        (frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))
    ].sort_index()

    # ==== THÊM ĐOẠN SỬA LỖI NÀY (BẮT BUỘC) ====
    # Ensure fwd_ret_1d contains ONLY scalar floats (handle lists/non-numeric from live data)
    def _to_scalar_float(x):
        # Handle NaN/None values
        if pd.isna(x) or x is None:
            return np.nan

        # Handle lists/tuples - recursively extract first valid scalar
        if isinstance(x, (list, tuple)):
            # Recursively process each element until we find a scalar
            for item in x:
                result = _to_scalar_float(item)
                if not pd.isna(result):  # Return first non-NaN result
                    return result
            return np.nan  # Return NaN if all elements were NaN/empty

        # Handle numpy arrays - convert to scalar if possible
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            if hasattr(x, "size") and x.size == 1:  # Scalar-like array
                try:
                    return float(x.item())
                except (ValueError, TypeError):
                    return np.nan
            elif len(x) > 0:  # Array-like with multiple elements
                for item in x.flat if hasattr(x, "flat") else x:
                    result = _to_scalar_float(item)
                    if not pd.isna(result):
                        return result
                return np.nan
            else:
                return np.nan

        # Handle scalar values - try to convert to float
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    # Apply cleaning to the column with robust error handling
    s_clean = s.copy()
    try:
        # Apply conversion
        converted_series = s["fwd_ret_1d"].apply(_to_scalar_float)

        # Convert to numpy array, handling any remaining object types
        if hasattr(converted_series, "values"):
            raw_values = converted_series.values
        else:
            raw_values = np.array(list(converted_series))

        # Ensure we have a 1D array of float64
        if raw_values.dtype == object:
            # Convert object array to float64, replacing non-convertible items with NaN
            converted_values = []
            for item in raw_values.flat if hasattr(raw_values, "flat") else raw_values:
                try:
                    if pd.isna(item):
                        converted_values.append(np.nan)
                    else:
                        converted_values.append(float(item))
                except (ValueError, TypeError):
                    converted_values.append(np.nan)
            values_array = np.array(converted_values, dtype=np.float64)
        else:
            values_array = raw_values.astype(np.float64)

        # Verify dimensions match
        if len(values_array) != len(s):
            raise ValueError(
                f"Length mismatch: got {len(values_array)} values, expected {len(s)}"
            )

        # Direct assignment - this should work now
        s_clean["fwd_ret_1d"] = values_array

    except Exception as e:
        # Fallback: if conversion fails, set all to NaN to avoid crashing
        print(f"Warning: Failed to convert fwd_ret_1d column: {e}")
        s_clean["fwd_ret_1d"] = np.full(len(s), np.nan, dtype=np.float64)
    # ==== KẾT THÚC ĐOẠN SỬA ====

    # daily = s["fwd_ret_1d"].iloc[-(window + 1) : -1].to_numpy(dtype=float)
    daily = s_clean["fwd_ret_1d"].iloc[-(window + 1) : -1].to_numpy(dtype=float)

    if len(daily) < 5:
        return None, None, None
    # fwd_ret_1d is a LOG return: compound by exp(sum), not prod(1+r) (which is for
    # simple returns); drawdown from the real price-relative curve, not log space.
    vol = float(np.nanstd(daily) * np.sqrt(252) * 100)
    curve = np.exp(np.cumsum(daily))
    peak = np.maximum.accumulate(curve)
    dd = float(((peak - curve) / peak).max() * 100)
    trend = float(
        (np.exp(np.sum(daily)) - 1.0) * 100
    )  # compounded log return over the window
    return vol, dd, trend


def _real_market_window_range(frame, symbol, cutoff, sequence_len):
    """Real calendar (start, end) of the trailing `sequence_len`-day window the
    model actually consumes for this prediction — NOT `vol_window` (a separate,
    risk-agent-only lookback that can differ from the model's own input length).
    None, None if there isn't enough local history (mirrors `_real_vol_dd`'s
    honesty rule: never fabricate a window that wasn't actually there)."""
    s = frame[
        (frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))
    ].sort_index()
    window = s.iloc[-sequence_len:]
    if len(window) < 5:
        return None, None
    return str(window.index[0].date()), str(window.index[-1].date())


def _real_news_window_range(news_idx, symbol, cutoff, lookback_days):
    """Real calendar (start, end) the news lookback covers — a fixed
    (cutoff - lookback_days, cutoff) span, disclosed as such regardless of whether
    any article actually fell inside it (an empty window is still a real window,
    not a missing one)."""
    end = pd.Timestamp(cutoff)
    start = end - pd.Timedelta(days=lookback_days)
    return str(start.date()), str(end.date())


def _resolve_attention_dates(frame, symbol, cutoff, top_days):
    """Enrich `attention_top_days` (from `raw_prediction.summarize_attention` —
    `{"days_before_cutoff": int, "weight": float}`) with the REAL calendar date
    each trailing-day offset corresponds to, using the same trading-calendar slice
    `_real_market_window_range` uses — "3 ngày trước" becomes "2026-03-19", a
    concrete date a user can actually look up, not just an abstract offset.
    Returns the list unchanged (still real, still usable) if the local price
    history needed to resolve dates isn't available for this cutoff."""
    if not top_days:
        return top_days
    s = frame[
        (frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))
    ].sort_index()
    if len(s) < 2:
        return top_days
    enriched = []
    for d in top_days:
        offset = int(d["days_before_cutoff"])
        date_str = str(s.index[-(offset + 1)].date()) if offset + 1 <= len(s) else None
        enriched.append({**d, "date": date_str})
    return enriched


_HORIZON_DAYS = {"1d": 1, "5d": 5, "20d": 20}


def _extract_horizon(query, cfg, default_horizon, parsed=None):
    """Classify the query's horizon via the REAL orchestrator_agent, falling back to
    ``default_horizon`` (the CLI's ``--horizon``) when the query names none.

    orchestrator_agent's own horizon extractor defaults to "1d" when the query has no
    horizon keyword at all — it has no concept of chat.py's --horizon CLI default, so
    blindly trusting it would silently override --horizon 5/20 on every query that
    doesn't happen to mention a horizon. Only accept its result when the query
    actually contains a horizon-indicating word; otherwise keep the CLI's default.
    Shared by every call site that needs a query's horizon (`_classify`, the `rank`
    command, RESEARCH's gap-fill forecast) so none of them can drift back to a
    hardcoded literal.

    ``parsed``: an already-computed ``orchestrator_node`` output, when the caller
    (``_classify``) already ran it for intent/symbols — avoids a second real LLM
    call (in --llm mode) just to re-derive the horizon from the same query.
    """
    out = (
        parsed
        if parsed is not None
        else orchestrator_node({"query_text": query, "node_timings": {}}, cfg)
    )
    mentions_horizon = any(
        k in query.lower()
        for k in (
            "1d",
            "5d",
            "20d",
            "ngày",
            "tuần",
            "tháng",
            "day",
            "week",
            "month",
            "dài hạn",
            "trung hạn",
            "ngắn hạn",
        )
    )
    return (
        _HORIZON_DAYS.get(out.get("target_horizon", ""), default_horizon)
        if mentions_horizon
        else default_horizon
    )


def _classify(query, cfg, default_symbol, default_horizon, sym_dates):
    """Route through the REAL orchestrator_agent — the actual first layer of the
    multi-agent system (intent classification + entity/symbol recognition) — rather
    than a hand-rolled keyword matcher in chat.py. In eval mode this runs its
    deterministic regex parser; with --llm it runs the real LLM-based parse. This
    also gets the larger, real symbol vocabulary and route_reason for free, instead
    of chat.py silently guessing.

    NOTE: `state["symbol"]`/`state["target_horizon_days"]` are deliberately left
    UNSET here — orchestrator_node has a "fast path" that triggers a full live data
    fetch (`prepare_single_cutoff`) when both are already present, which is a
    different data-loading route than chat.py's (pre-loaded frame/news_idx/store).
    Passing only query_text keeps it on the classification-only branch.
    """
    out = orchestrator_node({"query_text": query, "node_timings": {}}, cfg)
    intent = out.get("query_intent", "PREDICTION")
    route_reason = out.get("route_reason", "")

    symbols = [s for s in out.get("target_symbols", []) if s in KNOWN_SYMBOLS]
    symbol = symbols[0] if symbols else default_symbol

    horizon = _extract_horizon(query, cfg, default_horizon, parsed=out)

    # date_start/date_end: whatever the orchestrator's own language understanding
    # extracted (a real calendar range like "March 2026") — never re-derived by a
    # second, duplicate regex pass in chat.py. Falls back to a single resolved date
    # only when the query named no range at all.
    date_start = out.get("date_start")
    date_end = out.get("date_end")
    m = _DATE_RE.search(query)
    if m:
        date = m.group(0)
    else:
        # The query named NO date/range at all (e.g. "dự đoán BID 1 tuần tới" with
        # no absolute date) — anchor on the REAL wall-clock date ("hôm nay"), never
        # the last date that happens to be cached in the frozen prediction store.
        # That cached last-date is an implementation detail of the offline research
        # book, not "today" — silently substituting it made a live "predict from
        # now" query quietly answer a stale historical date instead (e.g. landing
        # on 2026-03-24 when it's actually already 2026-07-17). `in_book` below
        # naturally falls through to the LIVE-date fetch path when today isn't in
        # the local parquet yet, which is exactly what a real "now" prediction needs.
        date = datetime.now().strftime("%Y-%m-%d")
    if not date_start or not date_end:
        date_start = date_end = date
    return intent, symbol, symbols, date, date_start, date_end, horizon, route_reason


def _gather_evidence(
    frame, news_idx, symbol, date, lookback_days=5, vol_window=20, sequence_len=30
):
    """Build the volatility/sentiment/articles evidence bundle for (symbol, date) —
    factored out so `reasoning_agent`'s widen-and-rerun closure can call it again with
    wider `lookback_days`/`vol_window` values. Both slice the ALREADY-loaded
    `frame`/`news_idx` (the full research book, preloaded once at chat.py startup) —
    widening never triggers a new fetch.

    `sequence_len` (default matches `MultiAgentConfig.sequence_len`) is the model's
    OWN input window length — deliberately separate from `vol_window` (the
    risk-agent's lookback) — used only to disclose the real calendar range of data
    that actually fed the prediction (Section: narrator's data-window disclosure)."""
    vol, dd, trend = _real_vol_dd(frame, symbol, date, window=vol_window)
    heads = recent_headlines(news_idx, symbol, date, lookback_days=lookback_days, k=15)
    warnings = []
    if vol is None:
        # Live/out-of-book date: no trailing prices in the local parquet → cannot run
        # the volatility safety veto. Report it; do NOT pass 0.0 (would look 'calm').
        warnings.append(
            "risk: no local price history for this date — safety veto disabled"
        )
        vol_metrics = {}  # vol_20d / max_drawdown_pct / trend_pct intentionally absent
    else:
        vol_metrics = {"vol_20d": vol, "max_drawdown_pct": dd, "trend_pct": trend}
    market_window_start, market_window_end = _real_market_window_range(
        frame, symbol, date, sequence_len
    )
    news_window_start, news_window_end = _real_news_window_range(
        news_idx, symbol, date, lookback_days
    )
    # Real gap between the requested cutoff and the latest trading day actually
    # found. A weekend/holiday produces a small (1-3 day) gap; a LARGE one means the
    # real market data source has no trading days past `market_window_end` yet as of
    # this fetch (e.g. the query's "today" is ahead of what the market has actually
    # published) — surfaced honestly rather than silently predicting off stale data
    # while claiming it's current.
    staleness_days = None
    if market_window_end:
        staleness_days = (pd.Timestamp(date) - pd.Timestamp(market_window_end)).days
        if staleness_days > 5:
            warnings.append(
                f"data lag: dữ liệu giá gần nhất chỉ có đến {market_window_end}, "
                f"cách ngày dự đoán ({date}) {staleness_days} ngày — thị trường thực "
                f"chưa có dữ liệu mới hơn tại thời điểm truy vấn, không phải lỗi hệ thống"
            )
    return {
        "warnings": warnings,
        "volatility_metrics": vol_metrics,
        "sentiment_metrics": {
            "coverage": len(heads),
            "staleness_frac": 0.0 if heads else 1.0,
            "sentiment_mean": 0.0,
        },
        "articles": [{"title": h} for h in heads],
        "market_window_start": market_window_start,
        "market_window_end": market_window_end,
        "news_window_start": news_window_start,
        "news_window_end": news_window_end,
        "market_data_staleness_days": staleness_days,
    }


def _run_decision(symbol, date, horizon, cfg, frame, news_idx):
    evidence = _gather_evidence(
        frame, news_idx, symbol, date, sequence_len=cfg.sequence_len
    )
    state = {
        "symbol": symbol,
        "target_horizon_days": horizon,
        "prediction_time": date,
        "artifact_versions": {},
        "node_timings": {},
        **evidence,
    }

    def _widen_and_rerun(current_state):
        """Real second look: re-derive evidence from a WIDER already-loaded slice
        (no fetch), then re-run predict->gate->interaction->risk->metalabel->
        narrator->critic once — a full fresh pass, including re-verification, since
        reasoning_agent (the only caller of this closure) runs after critic_agent
        and its whole point is to check whether that verification held up."""
        wider_evidence = _gather_evidence(
            frame,
            news_idx,
            symbol,
            date,
            lookback_days=cfg.reasoning_widen_lookback_days_to,
            vol_window=cfg.reasoning_widen_sequence_len_to,
            sequence_len=cfg.sequence_len,  # the model's own input length never widens, only the risk check does
        )
        rerun_state = {**current_state, **wider_evidence}
        for name, fn in _RERUN_CHAIN:
            upd = fn(rerun_state, cfg)
            if name == "predict_agent":
                upd["attention_top_days"] = _resolve_attention_dates(
                    frame, symbol, date, upd.get("attention_top_days")
                )
            rerun_state.update(upd)
        return rerun_state

    steps = []
    for name, fn in DECISION_CHAIN:
        before = dict(state)
        upd = (
            fn(state, cfg, widen_and_rerun=_widen_and_rerun)
            if name == "reasoning_agent"
            else fn(state, cfg)
        )
        if name == "predict_agent":
            # real calendar date per attended trailing day — narrator/critic cite the
            # date, not just an abstract offset (state.py's attention_top_days schema)
            upd["attention_top_days"] = _resolve_attention_dates(
                frame, symbol, date, upd.get("attention_top_days")
            )
        steps.append((name, summarize_node(name, before, {**before, **upd})))
        timings = {**state.get("node_timings", {}), **upd.pop("node_timings", {})}
        state.update(upd)
        state["node_timings"] = timings
    return state, steps


def _print_steps(steps):
    print("\n  ── how the system decided (node by node) " + "─" * 30)
    for i, (name, summary) in enumerate(steps, 1):
        fields = (
            "  ".join(f"{k}={v}" for k, v in summary.items())
            if summary
            else "(no output)"
        )
        print(f"  STEP {i} · {name:<15} {fields}")
    print("  " + "─" * 70)


def _run_rank(symbols, date, horizon, cfg):
    out = rank_agent_node(
        {
            "target_symbols": symbols,
            "target_horizon_days": horizon,
            "prediction_time": date,
            "node_timings": {},
        },
        cfg,
    )
    print(f"\n  RANKING @ {date}:")
    print(f"    LONG : {out['rank_longs']}")
    print(f"    SHORT: {out['rank_shorts']}")
    print(f"    ABSTAIN: {out['rank_abstained']}")
    if out["warnings"]:
        print(f"    ! {out['warnings']}")
    return out


def _handle_rank(query, cfg, default_horizon):
    parts = query.split()
    syms = next((p for p in parts if "," in p), None)
    m = _DATE_RE.search(query)
    if not syms or not m:
        print("  usage: rank VCB,BID,CTG 2025-08-13")
        return None
    symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
    horizon = _extract_horizon(query, cfg, default_horizon)
    return _run_rank(symbols, m.group(0), horizon, cfg)


def _run_prediction_for_gap(symbol, from_date, horizon, cfg, frame, news_idx):
    """The REAL calibrated prediction chain (predict -> gate -> risk -> metalabel),
    anchored at ``from_date`` — used only for whatever portion of a requested range
    is NOT yet knowable from real data. The synthesis step narrates this; it never
    invents it — the number and trade/abstain action always come from here."""
    evidence = _gather_evidence(
        frame, news_idx, symbol, from_date, sequence_len=cfg.sequence_len
    )
    state = {
        "symbol": symbol,
        "target_horizon_days": horizon,
        "prediction_time": from_date,
        "artifact_versions": {},
        "node_timings": {},
        **evidence,
    }
    for (
        _,
        fn,
    ) in (
        _DECISION_ONLY_CHAIN
    ):  # predict, gate, horizon_interaction, risk, metalabel — no narrator/critic needed here
        state.update(fn(state, cfg))
    state["prediction_time"] = from_date
    return state


@contextlib.contextmanager
def _collecting(message: str):
    """Show one clean status line, then swallow the noisy third-party output that
    the data layer emits underneath (vnstock/vnai promo banners, tqdm encoding
    bars, HTTP chatter). Those are library-level prints to stdout/stderr the user
    should never see in a chat transcript — replaced by a single 'đang thu thập…'
    line. Exceptions still propagate (and are handled by the caller with stdout
    already restored)."""
    print(f"  🔍 {message}", flush=True)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        pass


def _llm_reachable(cfg) -> bool:
    """Quick check whether a local LLM is actually up, so RESEARCH can auto-use it
    for a real written analysis without the user needing --llm, and fall back
    cleanly (never crash, never hang) when it isn't."""
    from src.multiagent.guards import ensure_local_no_proxy

    try:
        ensure_local_no_proxy(cfg.ollama_base_url)
        urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


_NEWS_THEMES = {
    "tỷ giá / USD": ("tỷ giá", "usd", "ngoại tệ", "dxy"),
    "lãi suất / chính sách tiền tệ": (
        "lãi suất",
        "ngân hàng nhà nước",
        "sbv",
        "tiền tệ",
        "tín dụng",
    ),
    "VN-Index / thị trường chung": (
        "vn-index",
        "vnindex",
        "chứng khoán",
        "cổ phiếu",
        "thị trường",
    ),
    "Fed / vĩ mô quốc tế": ("fed", "powell", "mỹ", "gdp", "lạm phát"),
    "nội tại VCB / ngân hàng": (
        "vietcombank",
        "vcb",
        "lợi nhuận",
        "cổ tức",
        "chi nhánh",
    ),
}


def _news_theme_counts(articles: list[dict]) -> list[tuple[str, int]]:
    """Bucket headlines into a few finance themes so the summary reports WHAT the
    news was about, not a raw scroll of headlines the user can already read."""
    counts = {k: 0 for k in _NEWS_THEMES}
    for a in articles:
        t = a["title"].lower()
        for theme, kws in _NEWS_THEMES.items():
            if any(kw in t for kw in kws):
                counts[theme] += 1
    return [(k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0]


def _deterministic_conclusion(
    symbol, date_start, date_end, stats, articles, prediction
) -> str:
    """A CLEAN analytical conclusion built without an LLM — interprets the real
    numbers into sentences (trend direction, volatility regime, drawdown) and
    summarizes news by theme, instead of dumping raw headlines. Used when no LLM
    is reachable; the --llm path replaces this with a real written analysis."""
    parts: list[str] = []
    if stats["coverage"] == "none":
        parts.append(
            f"Không có dữ liệu giá thật của {symbol} trong khoảng {date_start}..{date_end}."
        )
    else:
        r = stats["return_pct"]
        v = stats["volatility_pct"]
        dd = stats["max_drawdown_pct"]
        direction = "tăng" if r > 1 else ("giảm" if r < -1 else "đi ngang")
        vol_regime = "thấp" if v < 20 else ("trung bình" if v < 35 else "cao")
        parts.append(
            f"Trong {stats['n_days']} ngày giao dịch ({stats['covered_start']}..{stats['covered_end']}), "
            f"{symbol} {direction} {r:+.1f}%, biến động {vol_regime} ({v:.0f}%/năm), "
            f"sụt giảm tối đa {dd:.1f}%."
        )
    themes = _news_theme_counts(articles)
    if themes:
        top = ", ".join(f"{name} ({n})" for name, n in themes[:3])
        parts.append(
            f"Dòng tin ({len(articles)} bài, trải đều khoảng) tập trung vào: {top}."
        )
    if prediction is not None:
        act = {
            "long": "khả năng tăng",
            "short": "khả năng giảm",
            "abstain": "không đủ độ tin cậy để khuyến nghị",
        }.get(prediction.get("action"), prediction.get("action"))
        parts.append(
            f"Phần chưa xảy ra (từ {prediction['prediction_time']}): mô hình dự báo {act} "
            f"— đây là DỰ BÁO, không phải dữ kiện thật."
        )
    return " ".join(parts)


def _llm_synthesize_research(
    symbol, date_start, date_end, stats, articles, prediction, cfg
):
    """LLM writes the natural-language analysis; every number it's ALLOWED to cite
    comes from ``stats``/``prediction`` (real market data / the real calibrated
    forecast) — same grounding discipline as narrator_agent's system prompt. The
    LLM only narrates; it is explicitly told a forecast is a forecast, not fact."""
    from src.multiagent.guards import assert_llm_allowed, ensure_local_no_proxy

    assert_llm_allowed(cfg, "chat.research_synthesis")
    ensure_local_no_proxy(cfg.ollama_base_url)
    from langchain_ollama import ChatOllama

    fact_lines = [
        f"Mã: {symbol}  Khoảng thời gian được hỏi: {date_start} đến {date_end}",
        f"Phạm vi dữ liệu thật có sẵn: {stats['coverage']} ({stats.get('n_days', 0)} ngày giao dịch)",
    ]
    if stats["coverage"] != "none":
        fact_lines.append(
            f"Dữ kiện THẬT (đã xảy ra) từ {stats['covered_start']} đến {stats['covered_end']}: "
            f"lợi nhuận {stats['return_pct']:+.2f}%  |  biến động {stats['volatility_pct']:.2f}%  |  "
            f"sụt giảm tối đa {stats['max_drawdown_pct']:.2f}%"
        )
    if prediction is not None:
        fact_lines.append(
            f"DỰ BÁO của mô hình (CHƯA xảy ra, không phải dữ kiện thật) từ "
            f"{prediction['prediction_time']}: hành động={prediction.get('action')}  "
            f"lý do={prediction.get('gate_reason', '')}"
        )
    if articles:
        fact_lines.append("Tin tức THẬT trong khoảng:")
        fact_lines += [
            f"  [{a['id']}] ({a['published_at']}) {a['title']}" for a in articles[:15]
        ]
    else:
        fact_lines.append("Không có tin tức nào được tìm thấy trong khoảng này.")

    system = (
        "Bạn là chuyên gia phân tích tài chính Việt Nam. Viết phân tích xu hướng "
        "(<250 từ) bằng tiếng Việt CHỈ dựa trên các con số/tin tức trong bảng dữ kiện "
        "dưới đây. TUYỆT ĐỐI không bịa số liệu nào khác. Nếu có phần DỰ BÁO, PHẢI nói "
        "rõ đó là dự báo của mô hình (chưa xảy ra), không được trình bày như dữ kiện thật."
    )
    llm = ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.2,
        timeout=cfg.ollama_timeout,
    )
    return llm.invoke(
        [("system", system), ("human", "\n".join(fact_lines))]
    ).content.strip()


def _run_research(symbol, date_start, date_end, horizon, cfg, frame, news_idx):
    """RESEARCH intent: real market stats + real news over the EXACT calendar range
    the query named (never a hardcoded lookback window). If the range reaches past
    what's knowable from data, the uncovered tail is answered by a REAL calibrated
    forecast (predict -> gate -> risk -> metalabel), at the query's own classified
    horizon (never a hardcoded one) — the LLM (in --llm mode) only narrates that
    forecast, it never invents the number or the decision itself.
    """
    with _collecting(
        f"Đang thu thập dữ liệu thị trường & tin tức {symbol} ({date_start}..{date_end})…"
    ):
        stats = compute_range_stats(frame, symbol, date_start, date_end)
        articles = articles_in_range(news_idx, symbol, date_start, date_end, k=40)
        prediction = None
        if stats["coverage"] in ("partial", "none"):
            anchor = stats.get("needs_prediction_from") or date_start
            prediction = _run_prediction_for_gap(
                symbol, anchor, horizon, cfg, frame, news_idx
            )

    # Compact factual header (real numbers only).
    print(
        f"\n  📊 {symbol}  {date_start} .. {date_end}  (dữ liệu thật: {stats['coverage']})"
    )
    if stats["coverage"] != "none":
        print(
            f"     {stats['covered_start']}..{stats['covered_end']} ({stats['n_days']} ngày): "
            f"lợi nhuận {stats['return_pct']:+.2f}%  ·  biến động {stats['volatility_pct']:.1f}%  ·  "
            f"sụt giảm tối đa {stats['max_drawdown_pct']:.1f}%  ·  {len(articles)} bài tin"
        )

    # The ANSWER: a written analytical conclusion — LLM if one is reachable (real
    # synthesis), otherwise a clean interpreted conclusion. Never a raw news dump.
    print()
    if _llm_reachable(cfg):
        try:
            llm_cfg = _dc_replace(
                cfg, evaluation_mode=False
            )  # research is narration, not the deterministic decision path
            with _collecting("Đang phân tích…"):
                answer = _llm_synthesize_research(
                    symbol, date_start, date_end, stats, articles, prediction, llm_cfg
                )
            print(f"  💬 {answer}")
            return {
                "stats": stats,
                "articles": articles,
                "prediction": prediction,
                "answer": answer,
            }
        except Exception as e:  # noqa: BLE001 — surface, fall back, never hide (R1)
            print(
                f"  (LLM tạm không khả dụng: {type(e).__name__}; dùng phân tích xác định thay thế)"
            )
    det_ans = _deterministic_conclusion(
        symbol, date_start, date_end, stats, articles, prediction
    )
    print(f"  💬 {det_ans}")
    return {
        "stats": stats,
        "articles": articles,
        "prediction": prediction,
        "answer": det_ans,
    }


def run_streamlit_gui():
    """Renders the Streamlit Web Interface wrapping the exact same agent logic."""
    import streamlit as st

    st.set_page_config(
        page_title="Multi-Agent Stock Advisor", layout="wide", page_icon="📊"
    )
    st.title("📊 Multi-Agent Stock Advisor Demonstration")
    st.markdown("---")

    # Mode and Config Selection in Sidebar
    st.sidebar.header("⚙️ Configuration")
    use_llm = st.sidebar.checkbox(
        "Enable LLM Mode (--llm)",
        value=False,
        help="Real LLM narration + metalabel news check",
    )
    default_horizon = st.sidebar.selectbox(
        "Default Horizon (days)", options=[1, 5, 20], index=1
    )

    # Initialize shared variables or caches inside Streamlit session state
    if "loaded_data" not in st.session_state:
        with st.spinner("Loading model predictions, gate, price & news data ..."):
            cfg = MultiAgentConfig(evaluation_mode=not use_llm)
            store = get_store(default_horizon, cfg)
            sym_dates = _symbol_dates(store)
            frame = pd.read_parquet(
                resolve_price_parquet(default_horizon, allow_missing_target=True)
            )
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            news_idx = load_news_index()
            lo = min(d for ds in sym_dates.values() for d in ds)
            hi = max(d for ds in sym_dates.values() for d in ds)

            st.session_state["loaded_data"] = {
                "cfg": cfg,
                "store": store,
                "sym_dates": sym_dates,
                "frame": frame,
                "news_idx": news_idx,
                "lo": lo,
                "hi": hi,
            }

    data = st.session_state["loaded_data"]
    # Dynamic config updates based on checkbox
    data["cfg"].evaluation_mode = not use_llm

    # System Status Banner
    mode_str = (
        "🧠 LLM (real narration)" if use_llm else "⚡ FAST (deterministic, no LLM)"
    )
    st.sidebar.info(f"**Current Mode:** {mode_str}")
    st.sidebar.markdown(f"**Universe:** {', '.join(KNOWN_SYMBOLS)}")
    st.sidebar.markdown(f"**Cached Range:** `{data['lo']}` to `{data['hi']}`")

    # Sample Queries Helper
    st.markdown("### 💡 Quick Examples")
    cols = st.columns(4)
    examples = [
        "VCB 2025-08-13",
        "BID có nên mua 5 ngày tới",
        "rank VCB,BID,CTG 2025-08-13",
        "Phân tích xu hướng TCB",
    ]
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state["query_input"] = ex

    # Main Chat Input
    query = st.text_input(
        "**Enter your financial query or command:**",
        key="query_input",
        placeholder="e.g., VCB 2025-08-13",
    )

    if query:
        query_strip = query.strip()
        low = query_strip.lower()

        if low in ("help", "?"):
            st.info(
                "Type: `<SYMBOL> [YYYY-MM-DD] [horizon]` | natural question (EN/VI) | `rank A,B,C DATE` | `symbols`"
            )
        elif low == "symbols":
            st.success(
                f"Supported Symbols: {', '.join(KNOWN_SYMBOLS)} (Available book dates: {data['lo']} .. {data['hi']})"
            )
        elif low.startswith("rank"):
            with st.spinner("Running Multi-Agent Ranking..."):
                parts = query_strip.split()
                syms = next((p for p in parts if "," in p), None)
                m = _DATE_RE.search(query_strip)
                if not syms or not m:
                    st.error(
                        "Usage syntax error. Example: `rank VCB,BID,CTG 2025-08-13`"
                    )
                else:
                    symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
                    horizon = _extract_horizon(
                        query_strip, data["cfg"], default_horizon
                    )
                    out = rank_agent_node(
                        {
                            "target_symbols": symbols,
                            "target_horizon_days": horizon,
                            "prediction_time": m.group(0),
                            "node_timings": {},
                        },
                        data["cfg"],
                    )

                    st.markdown(f"### 📈 Cross-Sectional Ranking @ **{m.group(0)}**")
                    st.json(
                        {
                            "LONG Positions": out["rank_longs"],
                            "SHORT Positions": out["rank_shorts"],
                            "ABSTAINED Nodes": out["rank_abstained"],
                            "Warnings": out["warnings"],
                        }
                    )
        else:
            # Standard Route Extraction via original _classify function
            (
                intent,
                symbol,
                symbols,
                date,
                date_start,
                date_end,
                horizon,
                route_reason,
            ) = _classify(
                query_strip, data["cfg"], "VCB", default_horizon, data["sym_dates"]
            )

            if symbol is None:
                st.error(
                    f"Couldn't identify any known symbol. Supported: {', '.join(KNOWN_SYMBOLS)}"
                )
                return

            st.write(
                f"🗺️ *Orchestrator routing:* `intent={intent}`, `symbol={symbol}`, `date={date}`, `horizon={horizon}d` ({route_reason})"
            )

            if intent == "COMPARISON":
                if len(symbols) >= 2 and date:
                    with st.spinner("Processing asset comparison..."):
                        out = rank_agent_node(
                            {
                                "target_symbols": symbols,
                                "target_horizon_days": horizon,
                                "prediction_time": date,
                                "node_timings": {},
                            },
                            data["cfg"],
                        )
                        st.json(out)
                else:
                    st.warning(
                        "Comparison needs 2+ known symbols and a concrete date. E.g., `VCB vs BID 2025-08-13`"
                    )

            elif intent == "RESEARCH":
                with st.spinner(
                    f"Gathering historical news & market stats for {symbol}..."
                ):
                    try:
                        res = _run_research(
                            symbol,
                            date_start,
                            date_end,
                            horizon,
                            data["cfg"],
                            data["frame"],
                            data["news_idx"],
                        )
                        if res:
                            st.markdown("### 💬 System Analysis Output")
                            st.info(res["answer"])

                            st.markdown("#### 📊 Real Data Summary Metrics")
                            st.columns(3)[0].metric(
                                "Return %", f"{res['stats']['return_pct']:+.2f}%"
                            )
                            st.columns(3)[1].metric(
                                "Volatility %", f"{res['stats']['volatility_pct']:.2f}%"
                            )
                            st.columns(3)[2].metric(
                                "Max Drawdown %",
                                f"{res['stats']['max_drawdown_pct']:.2f}%",
                            )

                            if res["articles"]:
                                with st.expander(
                                    f"📰 Trailing News Corpus ({len(res['articles'])} articles found)"
                                ):
                                    for a in res["articles"][:15]:
                                        st.caption(f"- {a.get('title')}")
                    except Exception as e:
                        st.exception(e)

            else:
                # Core Prediction & Reasoning Pipeline Flow
                with st.spinner(
                    "Executing active Multi-Agent pipeline layer calculations..."
                ):
                    try:
                        state, steps = _run_decision(
                            symbol,
                            date,
                            horizon,
                            data["cfg"],
                            data["frame"],
                            data["news_idx"],
                        )

                        action = state.get("action", "?").upper()
                        size = state.get("position_scale", 0.0)
                        answer = (
                            state.get("answer_text")
                            or state.get("grounded_answer")
                            or "(no answer)"
                        )

                        # System Output Card
                        st.markdown(
                            f"### 🤖 System Response: `{action}` (Size: `{size:+.2f}`)"
                        )
                        if action == "LONG":
                            st.success(answer)
                        elif action == "SHORT":
                            st.error(answer)
                        else:
                            st.warning(answer)

                        for w in state.get("warnings", []):
                            st.warning(f"⚠️ {w}")

                        # Trace Steps Rendering Accordion
                        with st.expander(
                            "⛓️ Inspect Trace Stack Summary (Node-by-Node Architecture)"
                        ):
                            for idx, (name, summary) in enumerate(steps, 1):
                                st.markdown(f"**Step {idx} · {name}**")
                                st.json(summary)

                    except Exception as e:
                        st.exception(e)


def main():
    # If the execution context is triggered via Streamlit, load the UI directly
    if sys.argv[0].endswith("streamlit") or "streamlit" in sys.modules:
        run_streamlit_gui()
        return

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm",
        action="store_true",
        help="real LLM narration + metalabel news check (needs Ollama)",
    )
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    # Add optional port arg to prevent breaking scripts but route cleanly to CLI default execution
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    cfg = MultiAgentConfig(evaluation_mode=not args.llm)
    print("Loading model predictions, gate, price & news data …")
    store = get_store(args.horizon, cfg)
    sym_dates = _symbol_dates(store)
    # allow_missing_target=True: real market-range analysis (compute_range_stats)
    # only needs fwd_ret_1d, not this horizon's own longer-range target, so a
    # recent-but-real day must not disappear just because that target is NaN yet.
    frame = pd.read_parquet(
        resolve_price_parquet(args.horizon, allow_missing_target=True)
    )
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    news_idx = load_news_index()
    lo = min(d for ds in sym_dates.values() for d in ds)
    hi = max(d for ds in sym_dates.values() for d in ds)

    mode = (
        "LLM (real narration + metalabel)"
        if args.llm
        else "FAST (deterministic grounded answer, no LLM)"
    )
    print(f"\n{'='*74}\n  Multi-Agent Stock Advisor  ·  mode: {mode}")
    print(f"  Symbols: {', '.join(KNOWN_SYMBOLS)}   Cached dates: {lo} .. {hi}")
    print(f"  Type a query (e.g. 'VCB 2025-08-13' or 'BID có nên mua 5 ngày tới').")
    print(f"  Commands: help · symbols · rank VCB,BID 2025-08-13 · quit")
    print("=" * 74)

    last_symbol = "VCB"
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not query:
            continue
        low = query.lower()
        if low in ("quit", "exit", "q"):
            print("bye.")
            break
        if low in ("help", "?"):
            print(
                "  Type: <SYMBOL> [YYYY-MM-DD] [horizon]  |  natural question (EN/VI)  |  "
                "rank A,B,C DATE  |  symbols  |  quit\n"
                "  Intent is auto-detected: 'trend'/'phân tích' -> research, "
                "'vs'/'so sánh' -> comparison, 'why'/'tại sao' -> explanation, "
                "else -> prediction."
            )
            continue
        if low == "symbols":
            print(f"  {', '.join(KNOWN_SYMBOLS)}  (dates {lo}..{hi})")
            continue
        if low.startswith("rank"):
            _handle_rank(query, cfg, args.horizon)
            continue

        # Route by intent BEFORE assuming every query is a single-day trade/abstain
        # question. Without this, "analyze the trend of VCB" and "should I buy VCB"
        # produced the identical gate-decision template — the query's actual intent
        # was never looked at. This calls the REAL orchestrator_agent (the MAS's
        # actual first layer), not a hand-rolled duplicate.
        intent, symbol, symbols, date, date_start, date_end, horizon, route_reason = (
            _classify(query, cfg, last_symbol, args.horizon, sym_dates)
        )
        if symbol is None:
            print("  Couldn't find a symbol. Try one of: " + ", ".join(KNOWN_SYMBOLS))
            continue
        last_symbol = symbol

        if intent == "COMPARISON":
            if len(symbols) >= 2 and date:
                print(
                    f"  → orchestrator: intent={intent} symbols={symbols} @ {date}  ({route_reason})"
                )
                _run_rank(symbols, date, horizon, cfg)
                continue
            print(f"  → orchestrator: intent={intent}  ({route_reason})")
            print(
                "  Comparison needs 2+ known symbols and a date, e.g. "
                "'rank VCB,BID,CTG 2025-08-13' or 'VCB vs BID 2025-08-13'."
            )
            continue

        if intent == "RESEARCH":
            print(
                f"  → orchestrator: intent={intent} symbol={symbol} range={date_start}..{date_end}  ({route_reason})"
            )
            try:
                _run_research(
                    symbol, date_start, date_end, horizon, cfg, frame, news_idx
                )
            except Exception as e:  # noqa: BLE001 — surface, never hide (R1)
                print(f"  ⚠ error: {type(e).__name__}: {e}")
                # traceback.print_exc()
            continue

        # EXPLANATION falls through to the decision chain below — the gate's own
        # reason string + narrator already answer "why", which is what this intent
        # is asking for; RESEARCH is the one that needs different (trend/news)
        # content instead of a trade decision.

        in_book = symbol in sym_dates and date in sym_dates.get(symbol, [])
        note = ""
        if args.llm:
            note = "   (thinking, LLM calls may take ~1-2 min) …"
        elif not in_book:
            note = "   (LIVE date — fetching real OHLCV + news; this can take many minutes) …"
        else:
            note = " …"
        print(
            f"  → interpreting as: symbol={symbol}  date={date}  horizon={horizon}d"
            + note
        )
        try:
            with _collecting(
                f"Đang thu thập dữ liệu thị trường & tin tức {symbol} ({date})…"
            ):
                state, steps = _run_decision(
                    symbol, date, horizon, cfg, frame, news_idx
                )
        except PredictionNotCachedError as e:
            print(f"  ⚠ {str(e).splitlines()[0]}")
            continue
        except ArtifactMissingError as e:
            print(f"  ⚠ live inference unavailable: {str(e).splitlines()[0]}")
            continue
        except Exception as e:  # noqa: BLE001 — surface, never hide (R1)
            print(f"  ⚠ error: {type(e).__name__}: {e}")
            continue
        # except Exception as e:
        #     traceback.print_exc()
        #     continue

        action = state.get("action", "?").upper()
        size = state.get("position_scale", 0.0)
        answer = (
            state.get("answer_text") or state.get("grounded_answer") or "(no answer)"
        )
        src = (state.get("model_evidence") or {}).get("source", "")
        tag = "  [live forward pass]" if src == "live_inference" else ""
        print(f"\nSystem: [{action}  size {size:+.2f}]{tag}  {answer}")
        for w in state.get("warnings", []):
            print(f"  ⚠ {w}")
        _print_steps(steps)


if __name__ == "__main__":
    # Auto-detect if invoked via Streamlit CLI executor string wrap
    if sys.argv[0].endswith("streamlit") or "streamlit" in sys.modules:
        run_streamlit_gui()
    else:
        sys.exit(main())
