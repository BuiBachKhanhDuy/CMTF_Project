"""Market Agent — prepares market data for a single cutoff."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .state import MultiAgentState


def market_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: fetch and prepare market data for one (symbol, cutoff) window.

    Reads: symbol, prediction_time, sequence_len
    Writes: close_window, market_window, market_tabular, token_ids, attention_mask,
            news_emb, news_mask, articles, sentiment_features, data_cutoff, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    symbol = state["symbol"]
    cutoff = state["prediction_time"]
    seq_len = state.get("sequence_len", cfg.sequence_len)

    from src.pipeline.orchestrator import prepare_single_cutoff

    result = prepare_single_cutoff(
        symbol=symbol,
        cutoff=cutoff,
        sequence_len=seq_len,
        news_cache_dir=str(cfg.news_cache_dir),
        phase2_output_dir=str(cfg.phase2_output_dir),
    )

    elapsed = time.time() - t0
    logger.info("MarketAgent | {} cutoff={} | {:.2f}s", symbol, cutoff, elapsed)

    timings = dict(state.get("node_timings", {}))
    timings["market_agent"] = elapsed

    return {
        "close_window": result["close_window"],
        "market_window": result["market_window"],
        "market_tabular": result["market_tabular"],
        "token_ids": result["token_ids"],
        "attention_mask": result["attention_mask"],
        "news_emb": result["news_emb"],
        "news_mask": result["news_mask"],
        "articles": result["articles"],
        "sentiment_features": result["sentiment_features"],
        "data_cutoff": cutoff,
        "node_timings": timings,
    }
