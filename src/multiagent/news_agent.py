"""News Agent — validates and enriches news data from the market preparation step."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .state import MultiAgentState


def news_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: validate news data prepared by the market agent.

    The heavy lifting (news loading, filtering, encoding) is already done by
    prepare_single_cutoff() called in market_node(). This node validates the
    results and ensures consistent state keys.

    Reads: news_emb, news_mask, articles, sentiment_features (from market_node output)
    Writes: node_timings (updates)
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    import numpy as np

    news_emb = state.get("news_emb")
    news_mask = state.get("news_mask")
    articles = state.get("articles", [])
    sentiment_features = state.get("sentiment_features", {})

    # Validate shapes
    seq_len = state.get("sequence_len", cfg.sequence_len)
    warnings: list[str] = list(state.get("warnings", []))

    if news_emb is None:
        news_emb = np.zeros((seq_len, 773), dtype=np.float32)
        news_mask = np.ones(seq_len, dtype=bool)
        warnings.append("NewsAgent: no news_emb in state — defaulting to zeros")

    if news_mask is None:
        news_mask = news_emb.sum(axis=-1) == 0

    # Count coverage
    bars_with_news = int((~news_mask).sum())
    total_articles = len(articles)
    logger.info(
        "NewsAgent | {} bars with news, {} articles total",
        bars_with_news,
        total_articles,
    )

    elapsed = time.time() - t0
    timings = dict(state.get("node_timings", {}))
    timings["news_agent"] = elapsed

    return {
        "news_emb": news_emb,
        "news_mask": news_mask,
        "articles": articles,
        "sentiment_features": sentiment_features,
        "warnings": warnings,
        "node_timings": timings,
    }
