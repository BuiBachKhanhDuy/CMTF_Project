"""Narrator Agent — honest Vietnamese explanation (plan §3.10; was answer_agent).

Maps the final action to Vietnamese and explains using ONLY evidence in state.
Never inflates confidence: on abstain it says the model lacks conviction and
discloses the operating point *with its CI*; on veto it states the safety reason.
Eval mode returns "" (determinism, §1.6).

Two products:
- ``generate_answer`` — the LLM narration (normal mode only), guarded so it can
  never run in eval mode.
- ``grounded_template`` — a deterministic, state-only answer used both as the
  critic's fallback and as the grounded reference for faithfulness checks.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

_ACTION_VI = {"long": "MUA", "short": "BÁN", "abstain": "KHÔNG GIAO DỊCH", "flat": "GIỮ"}

# Honest, universe-limited operating-point disclosure (plan §0/§3.10): the single-name
# gated accuracy is within CI of the base rate, so the calibrated ABSTENTION — not an
# accuracy number — is the product. Never headline an inflated figure.
#
# Computed PER HORIZON from state (gate_agent surfaces it from the frozen
# VN_{H}d.json's own validation-set numbers — see gate_io.calibrate_from_cache) rather
# than a fixed literal: 1D/5D/20D each calibrate to a different accuracy/coverage/base
# rate, so a single hardcoded string would misreport the operating point for two of the
# three horizons. Falls back to a number-free (but still honest) disclosure when the
# loaded policy predates schema v2 and has no disclosure numbers yet.
def _disclosure_note(state: MultiAgentState) -> str:
    da = state.get("gate_disclosure_da_pct")
    base = state.get("gate_disclosure_base_rate_pct")
    cov = state.get("gate_coverage")
    if da is None or base is None or cov is None:
        return (
            "Điểm vận hành: chưa có số liệu hiệu chuẩn công khai cho horizon này — "
            "giá trị của hệ thống là việc TỪ CHỐI giao dịch có hiệu chuẩn, không phải độ chính xác."
        )
    return (
        f"Điểm vận hành: ~{da:.0f}% độ chính xác hướng ở mức bao phủ ~{cov * 100:.0f}% "
        f"(khoảng tin cậy còn chồng lấn tỷ lệ nền ~{base:.1f}% trên tập kiểm định — "
        f"giá trị của hệ thống là việc TỪ CHỐI giao dịch có hiệu chuẩn, không phải độ chính xác)."
    )


def _attention_note(state: MultiAgentState) -> str:
    """Cite the single most-attended trailing day — a real internal signal from the
    model's cross-attention (see raw_prediction.summarize_attention), not an LLM's
    guess at "why". Prefers the real calendar date (resolved by
    chat.py::_resolve_attention_dates / graph.py's equivalent) over the abstract
    day-offset when available. Empty string (nothing to cite) when unavailable,
    never fabricated."""
    top_days = state.get("attention_top_days")
    if not top_days:
        return ""
    top = top_days[0]
    days_before, weight = top["days_before_cutoff"], top.get("weight")
    date = top.get("date")
    if date:
        when = f"ngày {date}"
    else:
        when = "ngày gần nhất" if days_before == 0 else f"{days_before} ngày trước"
    return f" Mô hình tập trung chú ý nhiều nhất vào dữ liệu {when} (trọng số {weight:.0%})."


def _window_note(state: MultiAgentState) -> str:
    """Disclose the REAL calendar date range of the market/news data that fed this
    prediction — not just the horizon. `market_window_start/end` and
    `news_window_start/end` are state-derived (chat.py::_gather_evidence /
    market_agent + news_agent), never invented by the model. Omits a side whose
    range isn't available rather than fabricating one."""
    ms, me = state.get("market_window_start"), state.get("market_window_end")
    ns, ne = state.get("news_window_start"), state.get("news_window_end")
    parts = []
    if ms and me:
        parts.append(f"dữ liệu giá {ms} → {me}")
    if ns and ne:
        parts.append(f"tin tức {ns} → {ne}")
    if not parts:
        return ""
    note = f" Dữ liệu dùng để dự đoán: {', '.join(parts)}."
    # Same >5-day threshold as chat.py::_gather_evidence's staleness warning — an
    # ordinary weekend/holiday gap (1-3 days) isn't worth calling out, but a real
    # lag between "hôm nay" and the latest data the market has actually published
    # must be stated plainly, not left for the reader to compute from two dates.
    staleness = state.get("market_data_staleness_days")
    if isinstance(staleness, (int, float)) and staleness > 5:
        note += (
            f" Lưu ý: dữ liệu giá thực tế mới nhất chỉ có đến {me}, cách thời điểm "
            f"dự đoán {int(staleness)} ngày — thị trường thực chưa công bố dữ liệu mới "
            f"hơn tại thời điểm này, đây không phải lỗi hệ thống."
        )
    return note


_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích tài chính Việt Nam. Viết một đoạn giải thích ngắn gọn "
    "(<200 từ) bằng tiếng Việt cho khuyến nghị giao dịch, CHỈ dùng dữ liệu trong bằng chứng. "
    "KHÔNG bịa số. Nếu khuyến nghị là KHÔNG GIAO DỊCH, nói rõ mô hình thiếu độ tin cậy và "
    "nêu điểm vận hành kèm khoảng tin cậy; KHÔNG dùng giọng điệu tự tin mua/bán. "
    "BẮT BUỘC phải nêu rõ trong câu trả lời: (1) khoảng thời gian thực (ngày bắt đầu → "
    "ngày kết thúc) của dữ liệu giá và dữ liệu tin tức đã dùng để dự đoán, lấy đúng từ "
    "phần 'Khoảng dữ liệu' trong bằng chứng — KHÔNG tự suy ra từ horizon; (2) ngày cụ thể "
    "mà mô hình tập trung chú ý (attention) nhiều nhất, lấy đúng từ phần 'Chú ý mô hình' "
    "trong bằng chứng; (3) NẾU bằng chứng có dòng 'Độ trễ dữ liệu' (nghĩa là dữ liệu giá "
    "thực không có đến sát ngày dự đoán), PHẢI nêu rõ con số ngày trễ đó và giải thích đây "
    "là do thị trường thực chưa công bố dữ liệu mới hơn, KHÔNG phải lỗi hệ thống. Nếu bằng "
    "chứng không có một trong các mục trên thì bỏ qua mục đó, KHÔNG bịa ra ngày tháng hay "
    "con số nào."
)


def grounded_template(state: MultiAgentState) -> str:
    """Deterministic answer built only from verified state fields (no LLM)."""
    action = state.get("action", "abstain")
    action_vi = _ACTION_VI.get(action, "KHÔNG GIAO DỊCH")
    symbol = state.get("symbol", "?")
    horizon = state.get("target_horizon_days", "?")
    size = state.get("position_scale", 0.0)
    vm = state.get("volatility_metrics", {})

    parts = [f"Khuyến nghị cho {symbol} ({horizon} ngày): {action_vi}."]
    if action == "abstain":
        if state.get("risk_vetoed"):
            parts.append(f"Bị chặn bởi kiểm soát rủi ro ({', '.join(state.get('veto_reasons', []))}).")
        else:
            parts.append(f"Mô hình thiếu độ tin cậy: {state.get('gate_reason', '')}.")
        parts.append(_disclosure_note(state))
    else:
        parts.append(f"Kích thước vị thế: {size:+.2f}. {state.get('gate_reason', '')}.")
        parts.append(_disclosure_note(state))
    parts.append(
        f"Bối cảnh thị trường: biến động 20 ngày {vm.get('vol_20d', 0):.1f}%, "
        f"sụt giảm tối đa {vm.get('max_drawdown_pct', 0):.1f}%."
        + _window_note(state)
        + _attention_note(state)
    )
    # NOTE: reasoning_agent runs AFTER critic_agent (it needs a real critic_status to
    # check), so `reasoning_notes` is never present yet when narrator/critic run —
    # reasoning_agent appends its own disclosure directly onto the final answer_text/
    # grounded_answer post-critic instead (see reasoning_agent.py's module docstring
    # for why that's safe without a second grounding pass).
    return " ".join(parts)


def _evidence_prompt(state: MultiAgentState) -> str:
    me = state.get("model_evidence", {})
    vm = state.get("volatility_metrics", {})
    sm = state.get("sentiment_metrics", {})
    action_vi = _ACTION_VI.get(state.get("action", "abstain"), "KHÔNG GIAO DỊCH")
    ms, mend = state.get("market_window_start"), state.get("market_window_end")
    ns, nend = state.get("news_window_start"), state.get("news_window_end")
    market_range = f"{ms} → {mend}" if ms and mend else "không có"
    news_range = f"{ns} → {nend}" if ns and nend else "không có"
    top_days = state.get("attention_top_days") or []
    top = top_days[0] if top_days else None
    if top and top.get("date"):
        attention_str = f"ngày {top['date']} (trọng số {top['weight']:.0%})"
    elif top:
        attention_str = f"{top['days_before_cutoff']} ngày trước cutoff (trọng số {top['weight']:.0%})"
    else:
        attention_str = "không có"
    staleness = state.get("market_data_staleness_days")
    lines = [
        f"Mã: {state.get('symbol')}  Horizon: {state.get('target_horizon_days')}d",
        f"Khuyến nghị: {action_vi}  |  Kích thước: {state.get('position_scale', 0):+.2f}",
        f"Gate: {state.get('gate_reason', '')}  (bao phủ {state.get('gate_coverage', 0):.0%})",
        f"Rủi ro: vetoed={state.get('risk_vetoed', False)} {state.get('veto_reasons', [])}",
        f"Dự báo (ensemble): {me.get('final_pred', 0):.5f}  gate_pred: {me.get('gate_pred', 0):.5f}",
        f"Biến động 20d: {vm.get('vol_20d', 0):.1f}%  Sụt giảm: {vm.get('max_drawdown_pct', 0):.1f}%",
        f"Tin tức: coverage={sm.get('coverage', 0)} staleness={sm.get('staleness_frac', 0):.0%}",
        f"Khoảng dữ liệu: giá {market_range}  |  tin tức {news_range}",
        f"Chú ý mô hình (attention) — ngày tập trung nhiều nhất: {attention_str}",
    ]
    if isinstance(staleness, (int, float)) and staleness > 5:
        lines.append(
            f"Độ trễ dữ liệu: dữ liệu giá thực mới nhất cách ngày dự đoán {int(staleness)} "
            f"ngày (thị trường thực chưa công bố dữ liệu mới hơn, không phải lỗi hệ thống)"
        )
    lines.append(f"LƯU Ý bắt buộc: {_disclosure_note(state)}")
    return "\n".join(lines)


def generate_answer(state: MultiAgentState, config: MultiAgentConfig, strict: bool = False) -> str:
    """LLM narration (normal mode only). Guarded against eval mode."""
    from ..guards import assert_llm_allowed, ensure_local_no_proxy

    assert_llm_allowed(config, "narrator.generate_answer")
    ensure_local_no_proxy(config.ollama_base_url)
    from langchain_ollama import ChatOllama

    system = _SYSTEM_PROMPT
    if strict:
        system += (" TUYỆT ĐỐI chỉ dùng các con số xuất hiện trong bằng chứng; "
                   "không thêm bất kỳ con số nào khác.")
    llm = ChatOllama(model=config.ollama_model, base_url=config.ollama_base_url,
                     temperature=0.1 if strict else 0.2, timeout=config.ollama_timeout)
    resp = llm.invoke([("system", system), ("human", _evidence_prompt(state))])
    return resp.content.strip()


def narrator_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: produce the first-attempt answer + the grounded template.

    Reads: action, position_scale, gate_*, risk_*, model_evidence, volatility_metrics,
           sentiment_metrics
    Writes: answer_text, grounded_answer, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    template = grounded_template(state)
    if cfg.evaluation_mode:
        answer = ""  # determinism: no LLM narration in eval
    else:
        answer = generate_answer(state, cfg)

    elapsed = time.time() - t0
    logger.info("Narrator | action={} chars={} | {:.3f}s",
                state.get("action"), len(answer), elapsed)
    return {
        "answer_text": answer,
        "grounded_answer": template,
        "node_timings": {"narrator": elapsed},
    }
