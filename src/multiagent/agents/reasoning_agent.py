"""Reasoning Agent — a single-pass reflection AFTER narrating+verifying (plan:
"reasoning brain"). NOT a free LLM debate, and NOT a loop — a deterministic,
pre-registered trigger (same style as `metalabel_agent`'s pre-registered event
categories) decides whether the evidence gathered so far looks thin or conflicting.

**Runs LAST, after `critic_agent`, not before `narrator` — this is a deliberate fix,
not the original design.** One of the three trigger conditions checks
`critic_status == "failed"`; a node positioned before `critic_agent` can never see
that field (it hasn't run yet), which would make the condition permanently
unreachable dead code. Placing this node after critic means the whole re-run
(when triggered) must go through predict -> gate -> horizon_interaction -> risk ->
metalabel -> narrator -> critic again with wider evidence — a real second pass,
not a partial one — so the re-verified `answer_text`/`critic_status` this produces
is what actually gets used.

Where a real second look is genuinely free (the caller passes `widen_and_rerun` —
`chat.py` already has the full research-book data in memory, so widening the
trailing window costs nothing), it takes exactly ONE extra full pass and adopts
THAT pass's result. Where it isn't free (`graph.py`'s node-based path —
`market_agent`'s data fetch is scoped per-request, not free), a trigger only adds a
disclosed caveat; nothing is re-fetched.

This node NEVER sets `action`/`position_scale` itself — those only ever come from
`gate_agent`/`risk_agent`/`metalabel_agent`, whether from the original pass or the
one `widen_and_rerun` produced. It DOES append its own disclosure sentence directly
onto the final `answer_text`/`grounded_answer` post-critic (not critic-re-verified) —
safe by construction, not by trust: both `_deterministic_note` and
`generate_reasoning_note` are restricted to a fixed vocabulary of pre-registered
reason labels and are structurally unable to state a number or claim about the
trade itself, so there is nothing for a grounding check to catch.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

_REASON_LABELS_VI = {
    "critic_verification_failed": "câu trả lời trước chưa được xác minh đầy đủ",
    "cross_horizon_disagreement": "hai khung thời gian khác không đồng thuận với tín hiệu này",
    "thin_news_coverage": "lượng tin tức liên quan còn ít",
}


def _evaluate_triggers(state: MultiAgentState, cfg: MultiAgentConfig) -> list[str]:
    """Pre-registered, deterministic conditions — never LLM-decided."""
    reasons: list[str] = []
    if state.get("critic_status") == "failed":
        reasons.append("critic_verification_failed")

    is_trade = state.get("action", "abstain") in ("long", "short")
    if is_trade and state.get("horizon_agreement_score") == 0:
        reasons.append("cross_horizon_disagreement")
    if is_trade:
        coverage = (state.get("sentiment_metrics") or {}).get("coverage", 0)
        if coverage < cfg.reasoning_min_news_coverage:
            reasons.append("thin_news_coverage")
    return reasons


def _deterministic_note(reasons: list[str], widened: bool) -> str:
    cites = ", ".join(_REASON_LABELS_VI.get(r, r) for r in reasons)
    if widened:
        return (f"Lưu ý: {cites} — hệ thống đã xem xét lại với cửa sổ dữ liệu rộng hơn "
                f"trước khi trả lời.")
    return (f"Lưu ý: {cites} — không thể mở rộng thêm dữ liệu cho lần này, "
            f"kết quả nên được xem xét thận trọng.")


def generate_reasoning_note(
    reasons: list[str], widened: bool, config: MultiAgentConfig,
) -> str:
    """LLM version of the disclosure (normal mode only), guarded exactly like
    `narrator.generate_answer` — strictly grounded in the pre-registered reason
    labels, never allowed to invent a new reason or a number."""
    from ..guards import assert_llm_allowed, ensure_local_no_proxy

    assert_llm_allowed(config, "reasoning_agent.generate_reasoning_note")
    ensure_local_no_proxy(config.ollama_base_url)
    from langchain_ollama import ChatOllama

    system = (
        "Bạn là trợ lý phân tích tài chính. Viết MỘT câu ngắn (<40 từ) bằng tiếng Việt "
        "giải thích lý do hệ thống xem xét lại trước khi trả lời, CHỈ dựa trên danh sách "
        "lý do được cung cấp. TUYỆT ĐỐI không thêm lý do, số liệu, hay khuyến nghị nào khác."
    )
    human = f"Lý do (đã xác định trước): {reasons}\nĐã mở rộng dữ liệu xem xét: {widened}"
    llm = ChatOllama(model=config.ollama_model, base_url=config.ollama_base_url,
                     temperature=0.1, timeout=config.ollama_timeout)
    return llm.invoke([("system", system), ("human", human)]).content.strip()


def reasoning_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
    widen_and_rerun: Callable[[MultiAgentState], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """LangGraph node: the LAST node before END — reflect after narrating+verifying.

    Reads: critic_status, action, horizon_agreement_score, sentiment_metrics,
           answer_text, grounded_answer
    Writes: reasoning_triggered_reasons, reasoning_notes, reasoning_evidence_widened,
            node_timings — plus, ONLY if `widen_and_rerun` fires, the FULL fresh
            state that real second pass produced (predict through critic again —
            never set directly by this node), with `reasoning_notes` appended onto
            its `answer_text`/`grounded_answer`.
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    reasons = _evaluate_triggers(state, cfg)
    widened = False
    update: dict[str, Any] = {}

    if reasons and widen_and_rerun is not None:
        rerun_result = widen_and_rerun(state)
        if rerun_result is not None:
            widened = True
            update.update(rerun_result)

    notes = ""
    if reasons:
        if cfg.evaluation_mode:
            notes = _deterministic_note(reasons, widened)
        else:
            try:
                notes = generate_reasoning_note(reasons, widened, cfg)
            except Exception as e:  # noqa: BLE001 — disclosure text, degrade not crash
                logger.warning("ReasoningAgent | LLM note generation failed ({}), using template", type(e).__name__)
                notes = _deterministic_note(reasons, widened)

    elapsed = time.time() - t0
    logger.info(
        "ReasoningAgent | triggered={} widened={} | {:.4f}s",
        reasons, widened, elapsed,
    )

    if notes:
        # Append post-critic — safe by construction (see module docstring): both note
        # generators are restricted to the fixed reason-label vocabulary and cannot
        # state a number or a claim about the trade, so there is nothing for a
        # grounding check to catch by appending here instead of re-verifying.
        base_answer = update.get("answer_text", state.get("answer_text", "")) or ""
        base_template = update.get("grounded_answer", state.get("grounded_answer", "")) or ""
        update["answer_text"] = (base_answer + " " + notes).strip() if base_answer else base_answer
        update["grounded_answer"] = (base_template + " " + notes).strip()

    update.update({
        "reasoning_triggered_reasons": reasons,
        "reasoning_notes": notes or None,
        "reasoning_evidence_widened": widened,
        "node_timings": {"reasoning_agent": elapsed},
    })
    return update
