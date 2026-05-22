"""Fusion Agent — confidence modulation + agent consensus correction.

CMTF final_pred is the base prediction (score). Market and news proposals
modulate confidence via agreement/disagreement bonuses.  Additionally, agent
consensus can scale the prediction via ``adjusted_pred`` so that downstream
risk management and evaluation can measure agent contribution.
"""

from __future__ import annotations

import math
import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _signs_agree(a: float, b: float, eps: float = 1e-4) -> bool:
    """True when both signals point the same way, or one is ~zero."""
    if abs(a) < eps or abs(b) < eps:
        return True  # neutral → no disagreement
    return math.copysign(1, a) == math.copysign(1, b)


def fusion_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: confidence-modulated fusion.

    Score = final_pred (CMTF is source of truth, never overridden).
    Confidence = predict_confidence ± market/news agreement bonuses.
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    # --- score comes solely from CMTF ---
    final_pred = float(state.get("final_pred", 0.0))
    fused_score = final_pred

    # --- base confidence from predict_agent ---
    base_confidence = float(state.get("predict_confidence", 0.0))

    # --- market agreement ---
    market_prop = state.get("market_proposal", {})
    market_score = float(market_prop.get("score", 0.0))
    market_conf = float(market_prop.get("confidence", 0.0))

    market_agrees = _signs_agree(market_score, final_pred)
    if market_agrees:
        market_bonus = market_conf * cfg.market_agree_bonus
    else:
        market_bonus = -market_conf * cfg.market_disagree_penalty

    # --- news agreement via news_residual (ML-learned signal) ---
    news_residual = float(state.get("news_residual", 0.0))
    news_prop = state.get("news_proposal", {})
    news_trust = float(news_prop.get("confidence", 0.0))  # = trust_weight

    news_agrees = _signs_agree(news_residual, final_pred)
    if news_agrees:
        news_bonus = news_trust * cfg.news_agree_bonus
    else:
        news_bonus = -news_trust * cfg.news_disagree_penalty

    # --- final fused confidence ---
    fused_confidence = _clip(base_confidence + market_bonus + news_bonus, 0.0, 1.0)

    direction = "flat" if abs(fused_score) < 0.001 else ("long" if fused_score > 0 else "short")

    # --- agent consensus correction (per-agent attribution) ---
    cmtf_dir = math.copysign(1, final_pred) if abs(final_pred) > 1e-6 else 0.0
    mkt_dir = math.copysign(1, market_score) if abs(market_score) > 1e-6 else 0.0
    news_dir = math.copysign(1, news_residual) if abs(news_residual) > 1e-6 else 0.0

    # Per-agent agreement: +1 agrees, -1 contradicts, 0 neutral
    mkt_agreement = mkt_dir * cmtf_dir  # in {-1, 0, +1}
    news_agreement = news_dir * cmtf_dir

    # Combined agreement score ∈ [-1, +1]
    agreement_score = (mkt_agreement + news_agreement) / 2.0

    alpha = cfg.override_alpha

    # Per-agent individual corrections
    mkt_scale = _clip(1.0 + alpha * mkt_agreement, 0.3, 1.7)
    news_scale = _clip(1.0 + alpha * news_agreement, 0.3, 1.7)
    both_scale = _clip(1.0 + alpha * agreement_score, 0.3, 1.7)

    adjusted_pred = round(final_pred * both_scale, 8)
    mkt_adjusted_pred = round(final_pred * mkt_scale, 8)
    news_adjusted_pred = round(final_pred * news_scale, 8)

    # Contribution percentages (how much each agent shifted the prediction)
    mkt_contribution_pct = round((mkt_scale - 1.0) * 100, 2)
    news_contribution_pct = round((news_scale - 1.0) * 100, 2)

    fused_decision = {
        "direction": direction,
        "score": round(fused_score, 6),
        "confidence": round(fused_confidence, 3),
        "base_confidence": round(base_confidence, 3),
        "market_agrees": market_agrees,
        "market_bonus": round(market_bonus, 4),
        "news_agrees": news_agrees,
        "news_trust": round(news_trust, 3),
        "news_bonus": round(news_bonus, 4),
        "agreement_score": round(agreement_score, 3),
        "override_alpha": round(alpha, 4),
        "scale_factor": round(both_scale, 4),
        "mkt_contribution_pct": mkt_contribution_pct,
        "news_contribution_pct": news_contribution_pct,
        "rationale": (
            f"score={final_pred:+.5f} (CMTF) | conf={base_confidence:.3f}"
            f"{market_bonus:+.4f}(mkt){news_bonus:+.4f}(news)={fused_confidence:.3f}"
            f" | adj={adjusted_pred:+.5f} (α={alpha:.2f} agree={agreement_score:+.2f})"
        ),
    }

    elapsed = time.time() - t0
    logger.info(
        "FusionAgent | score={:+.5f} adj={:+.5f} conf={:.3f} "
        "mkt_agree={} news_agree={} α={:.2f} | {:.3f}s",
        fused_score, adjusted_pred, fused_confidence,
        market_agrees, news_agrees, alpha, elapsed,
    )

    return {
        "fusion_decision": fused_decision,
        "adjusted_pred": adjusted_pred,
        "mkt_adjusted_pred": mkt_adjusted_pred,
        "news_adjusted_pred": news_adjusted_pred,
        "node_timings": {"fusion_agent": elapsed},
    }
