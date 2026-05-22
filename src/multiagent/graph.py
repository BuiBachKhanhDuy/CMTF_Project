"""LangGraph StateGraph wiring for the multi-agent inference system.

Topology:
    orchestrator → [market_agent | news_agent] (parallel)
    → predict_agent → fusion_agent → risk_agent → answer_agent → END
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from langgraph.graph import StateGraph, END
from loguru import logger

from .state import MultiAgentState
from .config import DEFAULT_CONFIG, MultiAgentConfig
from .agents.orchestrator_agent import orchestrator_node
from .agents.market_agent import market_agent_node
from .agents.news_agent import news_agent_node
from .agents.predict_agent import predict_agent_node
from .agents.fusion_agent import fusion_agent_node
from .agents.risk_agent import risk_agent_node
from .agents.answer_agent import answer_agent_node


def _make_node_with_config(node_fn, config: MultiAgentConfig):
    """Wrap a node function to inject config."""
    def wrapper(state: MultiAgentState) -> dict[str, Any]:
        return node_fn(state, config=config)
    wrapper.__name__ = node_fn.__name__
    return wrapper


def build_graph(config: MultiAgentConfig | None = None) -> StateGraph:
    """Build and compile the multi-agent inference graph.

    Topology:
        orchestrator → [market_agent | news_agent] (parallel fan-out)
        → predict_agent (fan-in)
        → fusion_agent (proposal synthesis)
        → risk_agent (policy execution)
        → answer_agent (explanation only)
        → END
    """
    cfg = config or DEFAULT_CONFIG

    graph = StateGraph(MultiAgentState)

    graph.add_node("orchestrator", _make_node_with_config(orchestrator_node, cfg))
    graph.add_node("market_agent", _make_node_with_config(market_agent_node, cfg))
    graph.add_node("news_agent", _make_node_with_config(news_agent_node, cfg))
    graph.add_node("predict_agent", _make_node_with_config(predict_agent_node, cfg))
    graph.add_node("fusion_agent", _make_node_with_config(fusion_agent_node, cfg))
    graph.add_node("risk_agent", _make_node_with_config(risk_agent_node, cfg))
    graph.add_node("answer_agent", _make_node_with_config(answer_agent_node, cfg))

    # Wire edges
    graph.set_entry_point("orchestrator")

    # Parallel fan-out: orchestrator → market + news
    graph.add_edge("orchestrator", "market_agent")
    graph.add_edge("orchestrator", "news_agent")

    # Fan-in: both data agents → predict
    graph.add_edge("market_agent", "predict_agent")
    graph.add_edge("news_agent", "predict_agent")

    # Sequential: predict → fusion → risk → answer → END
    graph.add_edge("predict_agent", "fusion_agent")
    graph.add_edge("fusion_agent", "risk_agent")
    graph.add_edge("risk_agent", "answer_agent")
    graph.add_edge("answer_agent", END)

    return graph.compile()


def run_graph(
    query_text: str | None = None,
    cutoff: str | None = None,
    horizon: int | None = None,
    config: MultiAgentConfig | None = None,
    *,
    symbol: str | None = None,
) -> MultiAgentState:
    """Run the full multi-agent graph for one prediction request.

    Args:
        query_text: Raw query text (classified by orchestrator)
        cutoff: Optional ISO date string (defaults to yesterday)
        horizon: Optional horizon override (1, 5, or 20)
        config: Optional config override
        symbol: Optional symbol override

    Returns:
        Fully populated MultiAgentState dict.
    """
    cfg = config or DEFAULT_CONFIG

    resolved_cutoff = cutoff or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    resolved_query = query_text or ""

    if horizon is not None and horizon not in (1, 5, 20):
        raise ValueError(f"horizon must be 1, 5, or 20, got {horizon}")

    initial_state: MultiAgentState = {
        "query_text": resolved_query,
        "prediction_time": resolved_cutoff,
        "sequence_len": cfg.sequence_len,
        "errors": [],
        "node_timings": {},
        "artifact_versions": {},
    }

    if symbol:
        initial_state["symbol"] = symbol
        initial_state["target_symbols"] = [symbol]
    if horizon is not None:
        initial_state["target_horizon_days"] = horizon
        initial_state["target_horizon"] = f"{horizon}d"

    compiled_graph = build_graph(cfg)
    result = compiled_graph.invoke(initial_state)

    if not result.get("symbol"):
        raise ValueError("Could not resolve symbol from query or CLI args; provide --symbol explicitly")

    logger.info(
        "Graph complete | {} {} {}d | action={} scale={:.2f}",
        result.get("symbol", "?"), resolved_cutoff, result.get("target_horizon_days", "?"),
        result.get("action", "?"),
        result.get("position_scale", 0),
    )

    return result
