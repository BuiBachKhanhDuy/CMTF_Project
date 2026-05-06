"""Explanation Agent — builds evidence dict and generates Vietnamese explanation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .state import MultiAgentState

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _build_evidence_dict(state: MultiAgentState) -> dict[str, Any]:
    """Build a structured evidence dict from the full agent state."""
    # Top-3 attended bars
    attn_weights = state.get("attn_weights")
    articles = state.get("articles", [])

    top_bars: list[dict] = []
    if attn_weights is not None and len(attn_weights) > 0:
        top_indices = np.argsort(attn_weights)[-3:][::-1]
        for idx in top_indices:
            bar_articles = [a for a in articles if a.get("bar_index") == int(idx)]
            top_bars.append({
                "bar_index": int(idx),
                "attention_weight": float(attn_weights[idx]),
                "articles": bar_articles[:3],  # Max 3 articles per bar
            })

    return {
        "symbol": state.get("symbol"),
        "prediction_time": state.get("prediction_time"),
        "horizon": state.get("target_horizon_days"),
        "baseline_pred": state.get("baseline_pred"),
        "final_pred": state.get("final_pred"),
        "news_residual": state.get("news_residual"),
        "news_residual_scale": state.get("news_residual_scale"),
        "final_pred_adjusted": state.get("final_pred_adjusted"),
        "action": state.get("action"),
        "position_scale": state.get("position_scale"),
        "regime_flags": state.get("regime_flags"),
        "news_quality_flags": state.get("news_quality_flags"),
        "disagreement_force_flat": state.get("disagreement_force_flat"),
        "news_weight": state.get("news_weight"),
        "top_attended_bars": top_bars,
        "seed_preds": state.get("seed_preds"),
    }


def _render_jinja2_fallback(evidence: dict[str, Any]) -> str:
    """Render Vietnamese explanation using Jinja2 template."""
    try:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=False,
        )
        template = env.get_template("explanation_vi.j2")
        return template.render(**evidence)
    except Exception as e:
        logger.warning("Jinja2 rendering failed: {} — using minimal fallback", e)
        return _minimal_fallback(evidence)


def _minimal_fallback(evidence: dict[str, Any]) -> str:
    """Minimal Vietnamese explanation when Jinja2 is also unavailable."""
    action_vi = {"long": "MUA", "short": "BÁN", "flat": "GIỮ"}.get(
        evidence.get("action", "flat"), "GIỮ"
    )
    return (
        f"Khuyến nghị {action_vi} cho {evidence.get('symbol', '?')} "
        f"(horizon {evidence.get('horizon', '?')}d). "
        f"Dự báo: {evidence.get('final_pred_adjusted', 0):.5f}. "
        f"Mức độ vị thế: {evidence.get('position_scale', 0):.0%}."
    )


def _call_ollama(evidence: dict[str, Any], config: MultiAgentConfig) -> str | None:
    """Call Ollama for Vietnamese explanation generation. Returns None on failure."""
    try:
        from langchain_ollama import ChatOllama

        system_prompt = (
            "Bạn là chuyên gia phân tích tài chính Việt Nam. "
            "Dựa trên dữ liệu bằng chứng được cung cấp, viết một đoạn giải thích ngắn gọn "
            "bằng tiếng Việt về khuyến nghị giao dịch. "
            "KHÔNG được bịa đặt thông tin. Chỉ sử dụng dữ liệu trong bằng chứng. "
            "Giữ câu trả lời dưới 200 từ."
        )

        # Format evidence as a concise string for the LLM
        action_vi = {"long": "MUA", "short": "BÁN", "flat": "GIỮ"}.get(
            evidence.get("action", "flat"), "GIỮ"
        )
        user_content = (
            f"Mã cổ phiếu: {evidence.get('symbol')}\n"
            f"Thời điểm: {evidence.get('prediction_time')}\n"
            f"Horizon: {evidence.get('horizon')}d\n"
            f"Khuyến nghị: {action_vi}\n"
            f"Dự báo baseline (chỉ thị trường): {evidence.get('baseline_pred', 0):.5f}\n"
            f"Dự báo cuối (có tin tức): {evidence.get('final_pred_adjusted', 0):.5f}\n"
            f"Đóng góp tin tức: {evidence.get('news_residual', 0):.5f} × {evidence.get('news_residual_scale', 0):.1f}\n"
            f"Mức vị thế: {evidence.get('position_scale', 0):.0%}\n"
        )

        # Add regime info
        regime = evidence.get("regime_flags", {})
        if regime:
            user_content += (
                f"Biến động 20d: {regime.get('vol_20d', 0):.4f}\n"
                f"Drawdown: {regime.get('max_drawdown_abs', 0):.4f}\n"
            )

        # Add top news
        top_bars = evidence.get("top_attended_bars", [])
        if top_bars:
            user_content += "\nTin tức quan trọng nhất:\n"
            for bar in top_bars[:3]:
                for art in bar.get("articles", [])[:2]:
                    user_content += f"  - {art.get('title', '?')}\n"

        if evidence.get("disagreement_force_flat"):
            user_content += "\nLưu ý: Các mô hình không đồng thuận → khuyến nghị GIỮ.\n"

        llm = ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=0.2,
            timeout=config.ollama_timeout,
        )
        messages = [
            ("system", system_prompt),
            ("human", user_content),
        ]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.warning("Ollama call failed: {} — falling back to Jinja2", e)
        return None


def explanation_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: generate Vietnamese explanation from evidence.

    Attempts Ollama (qwen2.5:7b) first, falls back to Jinja2 template on failure.

    Reads: all state keys (builds evidence from full state)
    Writes: evidence_dict, explanation_text_vi, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    evidence = _build_evidence_dict(state)

    # Try Ollama first
    explanation = _call_ollama(evidence, cfg)

    # Fallback to Jinja2
    if explanation is None:
        explanation = _render_jinja2_fallback(evidence)

    elapsed = time.time() - t0
    logger.info("ExplanationAgent | {} chars | {:.2f}s", len(explanation), elapsed)

    timings = dict(state.get("node_timings", {}))
    timings["explanation_agent"] = elapsed

    return {
        "evidence_dict": evidence,
        "explanation_text_vi": explanation,
        "node_timings": timings,
    }
