"""News-Quality Critic — deterministic coverage and staleness checker."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def news_quality_node(state: MultiAgentState, config: MultiAgentConfig | None = None) -> dict[str, Any]:
    """LangGraph node: assess news quality and compute adjusted prediction.

    Deterministic rules based on news coverage, article staleness, and
    sentiment dispersion. Scales the news residual accordingly.

    Reads: articles, news_mask, news_residual, baseline_pred, prediction_time
    Writes: news_quality_flags, news_residual_scale, final_pred_adjusted, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    articles = state.get("articles", [])
    news_mask = state["news_mask"]
    news_residual = state["news_residual"]
    baseline_pred = state["baseline_pred"]
    cutoff = state["prediction_time"]
    cutoff_ts = pd.Timestamp(cutoff)

    # --- Compute quality indicators ---

    # 1. Coverage: number of bars with at least one article
    coverage = int((~news_mask).sum())

    # 2. Staleness: fraction of articles older than staleness_days
    stale_count = 0
    total_count = len(articles)
    if total_count > 0:
        for article in articles:
            pub = article.get("published_at")
            if pub:
                try:
                    pub_ts = pd.Timestamp(pub)
                    age_days = (cutoff_ts - pub_ts).days
                    if age_days > cfg.staleness_days:
                        stale_count += 1
                except (ValueError, TypeError):
                    stale_count += 1  # Conservative: treat unparseable as stale
        staleness_frac = stale_count / total_count
    else:
        staleness_frac = 1.0  # No articles = maximally stale

    # 3. Sentiment dispersion: std of article sentiment scores
    sentiment_scores = [
        a["sentiment_score"]
        for a in articles
        if a.get("sentiment_score") is not None
    ]
    sentiment_std = float(np.std(sentiment_scores)) if len(sentiment_scores) >= 2 else 0.0

    # --- Apply rules ---
    if coverage < cfg.min_news_bars:
        news_residual_scale = 0.0  # Ignore news entirely
    elif staleness_frac > cfg.max_stale_frac:
        news_residual_scale = 0.5  # Discount stale news
    else:
        news_residual_scale = 1.0  # Trust news fully

    # Compute adjusted prediction
    final_pred_adjusted = baseline_pred + news_residual_scale * news_residual

    news_quality_flags = {
        "coverage": coverage,
        "total_articles": total_count,
        "stale_count": stale_count,
        "staleness_frac": staleness_frac,
        "sentiment_std": sentiment_std,
        "min_news_bars_threshold": cfg.min_news_bars,
        "max_stale_frac_threshold": cfg.max_stale_frac,
    }

    elapsed = time.time() - t0
    logger.info(
        "NewsQualityCritic | coverage={} staleness={:.2f} | scale={:.1f} adj_pred={:.5f} | {:.3f}s",
        coverage, staleness_frac, news_residual_scale, final_pred_adjusted, elapsed,
    )

    timings = dict(state.get("node_timings", {}))
    timings["news_quality_critic"] = elapsed

    return {
        "news_quality_flags": news_quality_flags,
        "news_residual_scale": news_residual_scale,
        "final_pred_adjusted": final_pred_adjusted,
        "node_timings": timings,
    }
