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
_COVERAGE_CI_NOTE = (
    "Điểm vận hành: ~54% độ chính xác hướng ở mức bao phủ ~25% "
    "(khoảng tin cậy còn chồng lấn tỷ lệ nền ~53.8% trên một mã — "
    "giá trị của hệ thống là việc TỪ CHỐI giao dịch có hiệu chuẩn, không phải độ chính xác)."
)

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích tài chính Việt Nam. Viết một đoạn giải thích ngắn gọn "
    "(<200 từ) bằng tiếng Việt cho khuyến nghị giao dịch, CHỈ dùng dữ liệu trong bằng chứng. "
    "KHÔNG bịa số. Nếu khuyến nghị là KHÔNG GIAO DỊCH, nói rõ mô hình thiếu độ tin cậy và "
    "nêu điểm vận hành kèm khoảng tin cậy; KHÔNG dùng giọng điệu tự tin mua/bán."
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
        parts.append(_COVERAGE_CI_NOTE)
    else:
        parts.append(f"Kích thước vị thế: {size:+.2f}. {state.get('gate_reason', '')}.")
        parts.append(_COVERAGE_CI_NOTE)
    parts.append(
        f"Bối cảnh thị trường: biến động 20 ngày {vm.get('vol_20d', 0):.1f}%, "
        f"sụt giảm tối đa {vm.get('max_drawdown_pct', 0):.1f}%."
    )
    return " ".join(parts)


def _evidence_prompt(state: MultiAgentState) -> str:
    me = state.get("model_evidence", {})
    vm = state.get("volatility_metrics", {})
    sm = state.get("sentiment_metrics", {})
    action_vi = _ACTION_VI.get(state.get("action", "abstain"), "KHÔNG GIAO DỊCH")
    return "\n".join([
        f"Mã: {state.get('symbol')}  Horizon: {state.get('target_horizon_days')}d",
        f"Khuyến nghị: {action_vi}  |  Kích thước: {state.get('position_scale', 0):+.2f}",
        f"Gate: {state.get('gate_reason', '')}  (bao phủ {state.get('gate_coverage', 0):.0%})",
        f"Rủi ro: vetoed={state.get('risk_vetoed', False)} {state.get('veto_reasons', [])}",
        f"Dự báo (ensemble): {me.get('final_pred', 0):.5f}  gate_pred: {me.get('gate_pred', 0):.5f}",
        f"Biến động 20d: {vm.get('vol_20d', 0):.1f}%  Sụt giảm: {vm.get('max_drawdown_pct', 0):.1f}%",
        f"Tin tức: coverage={sm.get('coverage', 0)} staleness={sm.get('staleness_frac', 0):.0%}",
        f"LƯU Ý bắt buộc: {_COVERAGE_CI_NOTE}",
    ])


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
