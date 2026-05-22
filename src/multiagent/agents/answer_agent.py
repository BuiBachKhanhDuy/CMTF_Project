"""Answer Agent — generates Vietnamese explanation from final decision evidence.

Cannot alter action or position_scale. In evaluation mode, outputs empty string.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

_ACTION_VI = {"long": "MUA", "short": "BÁN", "flat": "GIỮ"}

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích tài chính Việt Nam. "
    "Dựa trên dữ liệu bằng chứng được cung cấp, viết một đoạn giải thích ngắn gọn "
    "bằng tiếng Việt về khuyến nghị giao dịch. "
    "KHÔNG được bịa đặt thông tin. Chỉ sử dụng dữ liệu trong bằng chứng. "
    "Giữ câu trả lời dưới 200 từ."
)


def _format_evidence_prompt(state: MultiAgentState) -> str:
    """Format state into evidence prompt for LLM."""
    action_vi = _ACTION_VI.get(state.get("action", "flat"), "GIỮ")
    model_ev = state.get("model_evidence", {})
    vol_metrics = state.get("volatility_metrics", {})
    sentiment = state.get("sentiment_metrics", {})
    risk_checks = state.get("risk_checks", {})
    market_proposal = state.get("market_proposal", {})
    news_proposal = state.get("news_proposal", {})
    fusion = state.get("fusion_decision", {})

    parts = [
        f"Mã cổ phiếu: {state.get('symbol')}",
        f"Thời điểm: {state.get('prediction_time')}",
        f"Horizon: {state.get('target_horizon_days')}d",
        f"Khuyến nghị: {action_vi}",
        f"Position scale: {state.get('position_scale', 0):.2f}",
        f"Confidence: {state.get('final_confidence', 0):.3f}",
        f"Policy version: {state.get('policy_version', 1)}",
        "",
        "=== DỰ BÁO ===",
        f"Baseline (chỉ thị trường): {model_ev.get('baseline_pred', 0):.5f}",
        f"Final (có tin tức): {model_ev.get('final_pred', 0):.5f}",
        f"News residual: {model_ev.get('news_residual', 0):.5f}",
        f"Seed agreement: {model_ev.get('all_same_sign', '?')}",
        f"Predict confidence: {model_ev.get('predict_confidence', 0):.3f}",
        f"Fusion score: {fusion.get('score', 0):+.5f}",
        f"Fusion confidence: {fusion.get('confidence', 0):.3f}",
        "",
        "=== ĐÓNG GÓP AGENT ===",
        f"Market proposal: {market_proposal.get('direction', '?')} ({market_proposal.get('score', 0):+.5f})",
        f"News proposal: {news_proposal.get('direction', '?')} ({news_proposal.get('score', 0):+.5f})",
        f"Fusion rationale: {fusion.get('rationale', '')}",
        "",
        "=== RỦI RO ===",
        f"Vol 20d: {vol_metrics.get('vol_20d', 0):.1f}%",
        f"Max drawdown: {vol_metrics.get('max_drawdown_pct', 0):.1f}%",
        f"Trend: {vol_metrics.get('trend_pct', 0):+.1f}%",
        f"Risk tier: {risk_checks.get('tier', '?')}",
        "",
        "=== TIN TỨC ===",
        f"Sentiment mean: {sentiment.get('sentiment_mean', 0):.3f}",
        f"Coverage: {sentiment.get('coverage', 0)} bars",
        f"Staleness: {sentiment.get('staleness_frac', 0):.0%}",
        "",
        f"=== QUYẾT ĐỊNH ===",
        f"{state.get('decision_reasoning', '')}",
    ]

    return "\n".join(parts)


def answer_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: Answer Agent — generate explanation only.

    Cannot alter action or position_scale.
    In evaluation mode: outputs empty string.

    Reads: action, position_scale, final_confidence, model_evidence,
           volatility_metrics, sentiment_metrics, risk_checks, decision_reasoning
    Writes: explanation_text_vi, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    # Evaluation mode: no LLM, empty explanation
    if cfg.evaluation_mode:
        elapsed = time.time() - t0
        return {
            "explanation_text_vi": "",
            "node_timings": {"answer_agent": elapsed},
        }

    # Normal mode: LLM generates explanation
    from langchain_ollama import ChatOllama

    evidence_prompt = _format_evidence_prompt(state)

    llm = ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.2,
        timeout=cfg.ollama_timeout,
    )
    response = llm.invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", evidence_prompt),
    ])
    explanation = response.content

    elapsed = time.time() - t0
    logger.info("AnswerAgent | {} chars | {:.2f}s", len(explanation), elapsed)

    return {
        "explanation_text_vi": explanation,
        "node_timings": {"answer_agent": elapsed},
    }
