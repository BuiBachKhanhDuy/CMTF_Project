"""Metalabel Agent — qualitative event-flag veto (meta-labeling, not forecasting).

Distinct from every LLM-signal experiment that failed this project (news-sentiment
fusion, LLM/model consensus voting): this agent NEVER outputs a number that gets
fused into the prediction. It answers a bounded text-comprehension question — does
recent real news mention one of a small, PRE-REGISTERED set of known confound
events? — and, like risk_agent, can only DOWNGRADE a trade to abstain. It can never
turn an abstain into a trade, never up-size, never flip long/short.

This is a meta-label in the López de Prado sense: a secondary signal that decides
whether to ACT on the primary model's bet, using side-information the primary model
never sees (raw news text), not a competing forecast.

Categories are frozen (see docstring in config) — chosen before any test-data results
were examined, and not tuned against them afterward.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

# Pre-registered event categories — fixed before touching any test-row outcome.
# Standard finance confounders for short-horizon single-name return prediction.
EVENT_CATEGORIES = {
    "earnings_or_guidance": "earnings release, profit warning, or guidance revision",
    "ma_ownership_change": "M&A, stake acquisition/divestment, or major shareholder change",
    "regulatory_or_policy_action": "central bank rate decision, regulatory sanction, or policy action affecting the bank",
    "leadership_or_scandal": "management change, investigation, scandal, or compliance issue",
    "capital_or_dividend_action": "dividend announcement, share issuance, buyback, or stock split",
}

_SYSTEM = (
    "Bạn là chuyên gia phân tích rủi ro tin tức tài chính. Dựa trên các tiêu đề tin tức "
    "gần đây, hãy xác định xem có tin nào thuộc các loại sự kiện sau không:\n"
    + "\n".join(f"- {k}: {v}" for k, v in EVENT_CATEGORIES.items()) + "\n\n"
    'Trả lời DUY NHẤT JSON: {"flags": [<danh sách các loại khớp, có thể rỗng>], '
    '"reason": "<ngắn>"}. Chỉ liệt kê loại THỰC SỰ khớp với tin tức được cung cấp; '
    "không suy đoán. Nếu không có tin tức nào khớp, trả về flags rỗng."
)


def _parse_flags(text: str) -> list[str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        flags = obj.get("flags", [])
        if not isinstance(flags, list):
            return []
        return [f for f in flags if f in EVENT_CATEGORIES]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def metalabel_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: qualitative event-flag veto (one-way, meta-label only).

    Reads: action (post risk-veto), symbol, prediction_time, articles
    Writes: action, position_scale, metalabel_flags, metalabel_vetoed,
            veto_reasons (appended), decision_reasoning (appended), node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    action = state["action"]
    position_scale = float(state.get("position_scale", 0.0))
    is_trade = action in ("long", "short")

    flags: list[str] = []
    if is_trade and not cfg.evaluation_mode:
        from ..guards import assert_llm_allowed, ensure_local_no_proxy
        assert_llm_allowed(cfg, "metalabel_agent")
        ensure_local_no_proxy(cfg.ollama_base_url)
        from langchain_ollama import ChatOllama

        articles = state.get("articles", [])
        headlines = [a.get("title", "") for a in articles[-15:]] if articles else []
        if headlines:
            llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url,
                             temperature=0.1, timeout=cfg.ollama_timeout)
            msg = f"Mã: {state.get('symbol')}\nTin tức:\n" + "\n".join(f"- {h}" for h in headlines)
            flags = _parse_flags(llm.invoke([("system", _SYSTEM), ("human", msg)]).content)

    metalabel_vetoed = is_trade and bool(flags)
    veto_reasons = list(state.get("veto_reasons", []))
    decision_reasoning = state.get("decision_reasoning", "")

    if metalabel_vetoed:
        action = "abstain"
        position_scale = 0.0
        veto_reasons = veto_reasons + [f"metalabel:{','.join(flags)}"]
        decision_reasoning = (
            f"{decision_reasoning} Metalabel VETO ({','.join(flags)}): →abstain."
        )

    elapsed = time.time() - t0
    logger.info("MetalabelAgent | flags={} vetoed={} | {:.3f}s", flags, metalabel_vetoed, elapsed)

    return {
        "action": action,
        "position_scale": position_scale,
        "metalabel_flags": flags,
        "metalabel_vetoed": metalabel_vetoed,
        "veto_reasons": veto_reasons,
        "decision_reasoning": decision_reasoning,
        "node_timings": {"metalabel_agent": elapsed},
    }
