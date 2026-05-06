"""LangGraph StateGraph wiring for the multi-agent inference system."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END
from loguru import logger

from .state import MultiAgentState
from .config import DEFAULT_CONFIG, MultiAgentConfig
from .market_agent import market_node
from .news_agent import news_node
from .fusion_agent import fusion_node
from .critics.regime_critic import regime_critic_node
from .critics.news_quality_critic import news_quality_node
from .critics.disagreement_gate import disagreement_node
from .decision_agent import decision_node
from .explanation_agent import explanation_node


def _make_node_with_config(node_fn, config: MultiAgentConfig):
    """Wrap a node function to inject config."""
    def wrapper(state: MultiAgentState) -> dict[str, Any]:
        return node_fn(state, config=config)
    wrapper.__name__ = node_fn.__name__
    return wrapper


def build_graph(config: MultiAgentConfig | None = None) -> StateGraph:
    """Build and compile the multi-agent inference graph.

    Topology (linear with sequential critics):
        market → news → fusion → regime_critic → news_quality → disagreement → decision → explanation
    """
    cfg = config or DEFAULT_CONFIG

    graph = StateGraph(MultiAgentState)

    # Add nodes (each wrapped with config)
    graph.add_node("market", _make_node_with_config(market_node, cfg))
    graph.add_node("news", _make_node_with_config(news_node, cfg))
    graph.add_node("fusion", _make_node_with_config(fusion_node, cfg))
    graph.add_node("regime_critic", _make_node_with_config(regime_critic_node, cfg))
    graph.add_node("news_quality", _make_node_with_config(news_quality_node, cfg))
    graph.add_node("disagreement", _make_node_with_config(disagreement_node, cfg))
    graph.add_node("decision", _make_node_with_config(decision_node, cfg))
    graph.add_node("explanation", _make_node_with_config(explanation_node, cfg))

    # Wire edges (sequential)
    graph.set_entry_point("market")
    graph.add_edge("market", "news")
    graph.add_edge("news", "fusion")
    graph.add_edge("fusion", "regime_critic")
    graph.add_edge("regime_critic", "news_quality")
    graph.add_edge("news_quality", "disagreement")
    graph.add_edge("disagreement", "decision")
    graph.add_edge("decision", "explanation")
    graph.add_edge("explanation", END)

    return graph.compile()


def run_graph(
    symbol: str,
    cutoff: str,
    horizon: int,
    config: MultiAgentConfig | None = None,
) -> MultiAgentState:
    """Run the full multi-agent graph for one prediction request.

    Args:
        symbol: Stock ticker (e.g. "VCB", "BID")
        cutoff: ISO date string (e.g. "2025-03-31")
        horizon: Prediction horizon in days (1, 5, or 20)
        config: Optional config override

    Returns:
        Fully populated MultiAgentState dict.
    """
    cfg = config or DEFAULT_CONFIG

    if horizon not in (1, 5, 20):
        raise ValueError(f"horizon must be 1, 5, or 20, got {horizon}")

    initial_state: MultiAgentState = {
        "symbol": symbol,
        "prediction_time": cutoff,
        "target_horizon_days": horizon,
        "sequence_len": cfg.sequence_len,
        "errors": [],
        "warnings": [],
        "node_timings": {},
        "artifact_versions": {},
    }

    compiled_graph = build_graph(cfg)
    result = compiled_graph.invoke(initial_state)

    logger.info(
        "Graph complete | {} {} {}d | action={} scale={:.2f}",
        symbol, cutoff, horizon,
        result.get("action", "?"),
        result.get("position_scale", 0),
    )

    return result
