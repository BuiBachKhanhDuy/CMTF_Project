"""Orchestrator Agent — LLM-based intent/NER/time-range analysis brain.

Mandatory LLM in normal mode. In evaluation mode, uses deterministic regex parsing.
"""

from __future__ import annotations

import calendar
import json
import re
import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_NAMES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_KNOWN_SYMBOLS = {
    "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "TPB", "STB", "HDB",
    "HPG", "HSG", "NKG", "SMC",
    "FPT", "CMG", "VNG",
    "VNM", "MSN", "MCH",
    "MWG", "PNJ", "DGW",
    "VHM", "VRE", "NVL", "KDH", "DXG", "PDR",
    "SSI", "VCI", "HCM", "VND",
    "GAS", "PLX", "PVD", "PVS",
    "VIC", "SAB", "REE", "POW", "GVR",
}

_SYSTEM_PROMPT = """\
Bạn là bộ phân tích ý định cho hệ thống dự báo cổ phiếu Việt Nam.
Nhiệm vụ: phân loại câu hỏi và trích xuất thông tin.

Phân loại ý định (intent):
1. "PREDICTION" — dự báo/khuyến nghị mua bán
2. "EXPLANATION" — giải thích biến động
3. "RESEARCH" — phân tích xu hướng / ngành / thị trường trong một khoảng thời gian
4. "COMPARISON" — so sánh 2+ cổ phiếu

Trả lời CHÍNH XÁC JSON:
{
  "intent": "PREDICTION" | "EXPLANATION" | "RESEARCH" | "COMPARISON",
  "symbols": [<mã cổ phiếu, tối đa 5>],
  "horizon": "1d" | "5d" | "20d",
  "aspect": "price" | "news" | "risk" | "general",
  "date_start": "<YYYY-MM-DD hoặc null>",
  "date_end": "<YYYY-MM-DD hoặc null>"
}

Quy tắc:
- Mã cổ phiếu VN: 3 chữ hoa (VCB, BID, TCB...)
- "ngắn hạn"/"1 ngày" → "1d"; "trung hạn"/"tuần" → "5d"; "dài hạn"/"tháng" → "20d"
- Mặc định "1d" nếu không rõ horizon
- Mặc định "PREDICTION" nếu có mã, "RESEARCH" nếu không
- Nếu câu hỏi nêu MỘT KHOẢNG THỜI GIAN cụ thể (ví dụ "tháng 3 2026", "March 2026",
  một ngày cụ thể), tính chính xác date_start/date_end (ngày đầu và cuối tháng đó,
  hoặc chính ngày đó cho cả hai trường). NGÀY và NĂM có thể không nằm liền kề trong
  câu (ví dụ "tháng 3 cổ phiếu VCB 2026" nghĩa là tháng 3 năm 2026).
- Nếu KHÔNG có khoảng thời gian rõ ràng nào được nêu, trả về null cho cả hai
  trường — TUYỆT ĐỐI không tự đặt ra một khoảng ngày mặc định.
"""


# ── Deterministic parsing (evaluation mode) ────────────────────────────

def _extract_symbols(text: str) -> list[str]:
    """Extract stock symbols from text via regex."""
    candidates = re.findall(r"\b([A-Z]{3})\b", text.upper())
    seen = set()
    symbols = []
    for c in candidates:
        if c not in seen and c in _KNOWN_SYMBOLS:
            symbols.append(c)
            seen.add(c)
    return symbols[:5]


def _extract_horizon(text: str) -> str:
    """Extract horizon from text via keyword matching."""
    lowered = text.lower()
    if any(k in lowered for k in ("20d", "20 day", "20 ngày", "dài hạn", "tháng", "month")):
        return "20d"
    if any(k in lowered for k in ("5d", "5 day", "5 ngày", "trung hạn", "tuần", "week")):
        return "5d"
    return "1d"


def _extract_date_range(text: str) -> tuple[str | None, str | None]:
    """Extract an explicit calendar date range mentioned in the query — never a
    hardcoded lookback window. Supports:
      - explicit ISO date(s): "2025-08-13", "2025-08-01 to 2025-08-31"
      - Vietnamese month+year: "tháng 3 2026" (month and year need not be
        adjacent in the sentence — "tháng 3 cổ phiếu VCB 2026" is valid Vietnamese
        word order, so this searches the whole text for a plausible year
        independently of where "tháng N" appears)
      - English month+year: "march 2026", "in March 2026"
    Returns (None, None) when nothing explicit is found — the caller decides the
    fallback (e.g. "no range stated → use the latest known date"), this function
    never invents one.
    """
    dates = sorted(_DATE_RE.findall(text))
    if len(dates) >= 2:
        return dates[0], dates[-1]
    if len(dates) == 1:
        return dates[0], dates[0]

    lowered = text.lower()
    year_m = re.search(r"\b(20\d{2})\b", text)
    year = int(year_m.group(1)) if year_m else None
    if year is None:
        return None, None

    month = None
    vi_m = re.search(r"tháng\s*(\d{1,2})\b", lowered)
    if vi_m and 1 <= int(vi_m.group(1)) <= 12:
        month = int(vi_m.group(1))
    else:
        for name, num in _MONTH_NAMES_EN.items():
            if name in lowered:
                month = num
                break

    if month is None:
        return None, None
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


def _deterministic_parse(query_text: str) -> dict[str, Any]:
    """Parse query deterministically without LLM."""
    symbols = _extract_symbols(query_text)
    horizon = _extract_horizon(query_text)
    date_start, date_end = _extract_date_range(query_text)
    lowered = query_text.lower()

    if any(k in lowered for k in ("why", "tại sao", "explain", "giải thích")):
        intent = "EXPLANATION"
    elif any(k in lowered for k in ("vs", "so sánh", "compare")):
        intent = "COMPARISON"
    elif any(k in lowered for k in ("sector", "ngành", "trend", "xu hướng", "phân tích", "analyze", "analysis")):
        intent = "RESEARCH"
    else:
        intent = "PREDICTION"

    return {
        "intent": intent, "symbols": symbols, "horizon": horizon, "aspect": "general",
        "date_start": date_start, "date_end": date_end,
    }


# ── LLM-based parsing (normal mode) ───────────────────────────────────

def _llm_parse(query_text: str, config: MultiAgentConfig) -> dict[str, Any]:
    """Parse query via LLM orchestrator."""
    from ..guards import assert_llm_allowed, ensure_local_no_proxy

    assert_llm_allowed(config, "orchestrator._llm_parse")
    ensure_local_no_proxy(config.ollama_base_url)
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=0.1,
        timeout=config.ollama_timeout,
    )
    response = llm.invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", query_text),
    ])
    content = response.content.strip()

    # Extract JSON from potential markdown blocks
    if "```" in content:
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break

    result = json.loads(content)

    intent = str(result.get("intent", "PREDICTION")).upper()
    if intent not in ("PREDICTION", "EXPLANATION", "RESEARCH", "COMPARISON"):
        intent = "PREDICTION"

    symbols = result.get("symbols", [])
    if not isinstance(symbols, list):
        symbols = [symbols] if symbols else []
    symbols = [str(s).upper().strip() for s in symbols if s]
    symbols = [s for s in symbols if 2 <= len(s) <= 5][:5]

    horizon = str(result.get("horizon", "1d")).lower()
    if horizon not in ("1d", "5d", "20d"):
        horizon = "1d"

    aspect = str(result.get("aspect", "general")).lower()
    if aspect not in ("price", "news", "risk", "general"):
        aspect = "general"

    # Trust the LLM's own date-range understanding directly — it already reasons
    # about calendar months, relative time expressions, and word order far better
    # than a hand-written regex could; only validate the shape, don't re-derive it.
    date_start = result.get("date_start")
    date_end = result.get("date_end")
    date_start = date_start if isinstance(date_start, str) and _DATE_RE.fullmatch(date_start) else None
    date_end = date_end if isinstance(date_end, str) and _DATE_RE.fullmatch(date_end) else None

    return {
        "intent": intent, "symbols": symbols, "horizon": horizon, "aspect": aspect,
        "date_start": date_start, "date_end": date_end,
    }


# ── LangGraph node ─────────────────────────────────────────────────────

def orchestrator_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Orchestrator — analyze query intent, extract entities.

    Normal mode: mandatory LLM analysis.
    Evaluation mode: deterministic regex parsing.

    Reads: query_text, (optional) symbol, target_horizon_days
    Writes: query_intent, target_symbols, target_horizon, aspect_filter,
            symbol, target_horizon_days, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    query_text = state.get("query_text", "")

    # Fast path: CLI already provides symbol + horizon
    if state.get("symbol") and state.get("target_horizon_days"):
        # Fetch all data once — market/news agents read from state
        from src.pipeline.orchestrator import prepare_single_cutoff

        cutoff = state.get("prediction_time", "")
        seq_len = state.get("sequence_len", cfg.sequence_len)
        data = prepare_single_cutoff(
            symbol=state["symbol"],
            cutoff=cutoff,
            sequence_len=seq_len,
            news_cache_dir=str(cfg.news_cache_dir),
            sentiment_output_dir=str(cfg.sentiment_output_dir),
        )
        elapsed = time.time() - t0
        return {
            # Explicit route (never silently defaulted — R1): CLI supplied symbol+horizon.
            "query_intent": "PREDICTION",
            "target_symbols": [state["symbol"]],
            "target_horizon": f"{state['target_horizon_days']}d",
            "aspect_filter": "general",
            "route_reason": "symbol+horizon supplied by CLI → PREDICTION",
            "close_window": data["close_window"],
            "market_window": data["market_window"],
            "market_tabular": data["market_tabular"],
            "market_feature_cols": data.get("market_feature_cols", []),
            "news_emb": data["news_emb"],
            "news_mask": data["news_mask"],
            "articles": data.get("articles", []),
            "data_cutoff": cutoff,
            "node_timings": {"orchestrator": elapsed},
        }

    # Parse query
    if cfg.evaluation_mode:
        parsed = _deterministic_parse(query_text)
    else:
        parsed = _llm_parse(query_text, cfg)

    # Ensure symbols are populated
    if not parsed["symbols"]:
        parsed["symbols"] = _extract_symbols(query_text)

    # Resolve primary symbol
    symbol = parsed["symbols"][0] if parsed["symbols"] else state.get("symbol", "")

    # Horizon to days
    horizon_map = {"1d": 1, "5d": 5, "20d": 20}
    horizon_days = horizon_map.get(parsed["horizon"], 1)

    # Route reason is always recorded; a PREDICTION fallback (no symbol) is logged, not
    # silently taken (R1).
    intent = parsed["intent"]
    if not parsed["symbols"] and intent in ("PREDICTION", "COMPARISON", "EXPLANATION"):
        route_reason = f"no symbol parsed → fallback from {intent} (orchestrator.fallback=true)"
        logger.warning("Orchestrator fallback: {} intent with no symbol", intent)
    else:
        route_reason = f"parsed intent={intent} symbols={parsed['symbols']}"

    elapsed = time.time() - t0
    logger.info(
        "Orchestrator | intent={} symbol={} horizon={} | {:.2f}s",
        intent, symbol, parsed["horizon"], elapsed,
    )

    return {
        "query_intent": intent,
        "target_symbols": parsed["symbols"] or ([symbol] if symbol else []),
        "target_horizon": parsed["horizon"],
        "aspect_filter": parsed["aspect"],
        "route_reason": route_reason,
        "symbol": symbol,
        "target_horizon_days": horizon_days,
        # Explicit calendar range the query itself named (LLM-understood in normal
        # mode; regex fallback in eval mode) — None/None when no range was stated,
        # never a made-up default. Downstream (market/news data layer) uses this
        # directly instead of a fixed lookback window.
        "date_start": parsed.get("date_start"),
        "date_end": parsed.get("date_end"),
        "node_timings": {"orchestrator": elapsed},
    }
