"""News Agent — computes sentiment metrics and news quality assessment.

Pure analytical node: reads news data from state (fetched by orchestrator).
The news_proposal provides a quality/trust signal for fusion, not a directional vote.
The actual ML-learned news signal is news_residual (computed by predict_agent).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def _compute_sentiment_metrics(
    articles: list[dict],
    news_mask: np.ndarray,
    cutoff: str,
) -> dict[str, Any]:
    """Compute news quality and sentiment metrics."""
    cutoff_ts = pd.Timestamp(cutoff)
    coverage = int((~news_mask).sum())
    total_articles = len(articles)

    stale_count = 0
    if total_articles > 0:
        for article in articles:
            pub = article.get("published_at")
            if pub:
                try:
                    age_days = (cutoff_ts - pd.Timestamp(pub)).days
                    if age_days > 14:
                        stale_count += 1
                except (ValueError, TypeError):
                    stale_count += 1
            else:
                stale_count += 1
        staleness_frac = stale_count / total_articles
    else:
        staleness_frac = 1.0

    scores = [a["sentiment_score"] for a in articles if a.get("sentiment_score") is not None]
    sentiment_mean = float(np.mean(scores)) if scores else 0.0
    sentiment_std = float(np.std(scores)) if len(scores) >= 2 else 0.0

    return {
        "coverage": coverage,
        "total_articles": total_articles,
        "staleness_frac": round(staleness_frac, 2),
        "sentiment_mean": round(sentiment_mean, 3),
        "sentiment_std": round(sentiment_std, 3),
    }


def _build_news_proposal(sentiment_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build news quality assessment — trust weight for fusion, not a directional vote.

    The directional news signal comes from news_residual (predict_agent),
    NOT from this proposal's score. This proposal only provides:
    - quality/trust_weight: how much to trust the ML news signal
    - raw sentiment_mean: for explainability/logging
    """
    sentiment_mean = float(sentiment_metrics.get("sentiment_mean", 0.0))
    coverage = float(sentiment_metrics.get("coverage", 0.0))
    staleness_frac = float(sentiment_metrics.get("staleness_frac", 1.0))

    # Trust weight: high coverage + fresh articles → trust the ML news signal
    trust_weight = max(0.0, min(1.0, (coverage / 10.0) * (1.0 - staleness_frac)))

    # Direction from raw sentiment (for display only, not used in fusion math)
    direction = "flat" if abs(sentiment_mean) < 0.05 else ("long" if sentiment_mean > 0 else "short")

    return {
        "direction": direction,
        "score": round(sentiment_mean, 6),  # raw sentiment, NOT scaled
        "confidence": round(trust_weight, 3),  # = quality trust weight
        "rationale": (
            f"sentiment_mean={sentiment_mean:+.3f} coverage={coverage:.0f} "
            f"staleness={staleness_frac:.2f} trust={trust_weight:.3f}"
        ),
        "quality": {
            "coverage": int(coverage),
            "staleness_frac": round(staleness_frac, 2),
            "trust_weight": round(trust_weight, 3),
        },
    }


def news_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: News Agent — compute sentiment metrics and quality assessment.

    Reads from state (populated by orchestrator):
        news_emb, news_mask, articles, prediction_time
    Writes: sentiment_metrics, news_proposal, node_timings
    """
    t0 = time.time()

    cutoff = state.get("prediction_time", "")
    news_mask = state["news_mask"]
    articles = state.get("articles", [])

    sentiment_metrics = _compute_sentiment_metrics(articles, news_mask, cutoff)
    news_proposal = _build_news_proposal(sentiment_metrics)

    elapsed = time.time() - t0
    logger.info(
        "NewsAgent | articles={} coverage={} sent={:.3f} trust={:.3f} | {:.2f}s",
        sentiment_metrics["total_articles"],
        sentiment_metrics["coverage"],
        sentiment_metrics["sentiment_mean"],
        news_proposal["confidence"],
        elapsed,
    )

    return {
        "sentiment_metrics": sentiment_metrics,
        "news_proposal": news_proposal,
        "node_timings": {"news_agent": elapsed},
    }
