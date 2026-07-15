"""Research Agent — grounded news RAG (plan §3.9).

Retrieves the news articles already loaded into state (by the orchestrator), ranks
them by recency, and summarises them. It makes NO trade call and never touches the
gate. Every sentence in an LLM summary must cite a retrieved article id; if
retrieval returns nothing, it says so plainly (R1). In eval mode (LLM-free) it
returns a deterministic, citation-only digest instead of a generated summary.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState

_SYSTEM_PROMPT = (
    "Bạn là trợ lý nghiên cứu tin tức tài chính Việt Nam. Tóm tắt các bài báo được cung cấp "
    "bằng tiếng Việt (<200 từ). MỖI câu phải trích dẫn id bài báo dạng [id]. KHÔNG bịa thông tin, "
    "KHÔNG đưa ra khuyến nghị mua/bán. Nếu không có bài báo, nói rõ là không có dữ liệu."
)


def _rank_docs(articles: list[dict], cutoff: str, top_k: int = 8) -> list[dict]:
    """Rank articles by recency (most recent first); attach a stable id."""
    cutoff_ts = pd.Timestamp(cutoff) if cutoff else None
    scored = []
    for i, a in enumerate(articles):
        pub = a.get("published_at")
        try:
            age = (cutoff_ts - pd.Timestamp(pub)).days if (cutoff_ts and pub) else 10_000
        except (ValueError, TypeError):
            age = 10_000
        scored.append((age, {
            "id": a.get("id") or a.get("url") or f"art{i}",
            "title": a.get("title", "(no title)"),
            "published_at": str(pub) if pub else "unknown",
            "sentiment_score": a.get("sentiment_score"),
            "age_days": age,
        }))
    scored.sort(key=lambda x: x[0])
    return [d for _, d in scored[:top_k]]


def _deterministic_digest(docs: list[dict]) -> str:
    if not docs:
        return "Không có bài báo nào được truy xuất cho truy vấn này."
    return "\n".join(f"[{d['id']}] ({d['published_at']}) {d['title']}" for d in docs)


def research_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: retrieve + summarise news. Makes no trade decision.

    Reads: articles, prediction_time, aspect_filter
    Writes: retrieved_docs, research_summary_vi, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    articles = state.get("articles", [])
    docs = _rank_docs(articles, state.get("prediction_time", ""))

    if not docs:
        summary = "Không có bài báo nào được truy xuất cho truy vấn này."
    elif cfg.evaluation_mode:
        summary = _deterministic_digest(docs)  # LLM-free digest with citations
    else:
        from ..guards import assert_llm_allowed, ensure_local_no_proxy
        assert_llm_allowed(cfg, "research_agent")
        ensure_local_no_proxy(cfg.ollama_base_url)
        from langchain_ollama import ChatOllama
        corpus = "\n".join(
            f"[{d['id']}] {d['published_at']} — {d['title']} (sentiment={d['sentiment_score']})"
            for d in docs
        )
        llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url,
                         temperature=0.2, timeout=cfg.ollama_timeout)
        summary = llm.invoke([("system", _SYSTEM_PROMPT), ("human", corpus)]).content.strip()

    elapsed = time.time() - t0
    logger.info("ResearchAgent | {} docs retrieved | {} chars | {:.3f}s",
                len(docs), len(summary), elapsed)
    return {
        "retrieved_docs": docs,
        "research_summary_vi": summary,
        "node_timings": {"research_agent": elapsed},
    }
