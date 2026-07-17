"""LangGraph StateGraph wiring for the multi-agent inference system.

Topology:
    orchestrator → [market_agent | news_agent] (parallel)
    → predict_agent → gate_agent → horizon_interaction_agent (symmetric size adjustment)
    → risk_agent (veto) → metalabel_agent (veto) → narrator → critic
    → reasoning_agent (single-pass reflection, runs LAST so it can see the real
    critic_status — see reasoning_agent.py) → END
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
from .agents.gate_agent import gate_agent_node
from .agents.horizon_interaction_agent import horizon_interaction_agent_node
from .agents.risk_agent import risk_agent_node
from .agents.metalabel_agent import metalabel_agent_node
from .agents.reasoning_agent import reasoning_agent_node
from .agents.narrator_agent import narrator_agent_node
from .agents.critic_agent import critic_agent_node


def _make_node_with_config(node_fn, config: MultiAgentConfig, node_name: str):
    """Wrap a node to inject config and, when tracing is on, record a trace step.

    Keeping the trace in the wrapper (derived from each node's returned update) lets
    the agents stay clean while still giving R3 a legible per-node record.
    """
    import time as _time

    from .trace import make_trace_record, render_step

    def wrapper(state: MultiAgentState) -> dict[str, Any]:
        if not config.trace_enabled:
            return node_fn(state, config=config)
        t0 = _time.time()
        update = node_fn(state, config=config) or {}
        elapsed = update.get("node_timings", {}).get(node_name, _time.time() - t0)
        rec = make_trace_record(node_name, elapsed, dict(state), update)
        # Live console rendering (step numbers are finalised in the transcript).
        logger.info("\n{}", render_step(rec, 0, 0).replace("STEP 0/0 · ", ""))
        merged = dict(update)
        merged["trace"] = [rec]
        return merged

    wrapper.__name__ = node_fn.__name__
    return wrapper


def build_graph(config: MultiAgentConfig | None = None) -> StateGraph:
    """Build and compile the multi-agent inference graph.

    Topology:
        orchestrator → [market_agent | news_agent] (parallel fan-out)
        → predict_agent (fan-in)
        → gate_agent (frozen GatePolicy — the only place trade/abstain is set)
        → horizon_interaction_agent (symmetric conviction adjustment from the OTHER
          two horizons' agreement — NOT a veto; runs before the vetoes so they can
          still zero out whatever size this layer produced)
        → risk_agent (one-way safety veto: vol/drawdown)
        → metalabel_agent (one-way qualitative veto: pre-registered news event flags)
        → narrator (honest Vietnamese disclosure)
        → critic (verify vs state; regenerate/template fallback)
        → reasoning_agent (single-pass reflection, runs LAST — needs the real
          critic_status, which doesn't exist before critic_agent runs. No
          `widen_and_rerun` callback on this path, since `market_agent`'s data fetch
          here is not free; a trigger only adds a disclosed caveat onto the final
          answer, it never attempts to re-fetch)
        → END
    """
    cfg = config or DEFAULT_CONFIG

    graph = StateGraph(MultiAgentState)

    graph.add_node("orchestrator", _make_node_with_config(orchestrator_node, cfg, "orchestrator"))
    graph.add_node("market_agent", _make_node_with_config(market_agent_node, cfg, "market_agent"))
    graph.add_node("news_agent", _make_node_with_config(news_agent_node, cfg, "news_agent"))
    graph.add_node("predict_agent", _make_node_with_config(predict_agent_node, cfg, "predict_agent"))
    graph.add_node("gate_agent", _make_node_with_config(gate_agent_node, cfg, "gate_agent"))
    graph.add_node("horizon_interaction_agent",
                   _make_node_with_config(horizon_interaction_agent_node, cfg, "horizon_interaction_agent"))
    graph.add_node("risk_agent", _make_node_with_config(risk_agent_node, cfg, "risk_agent"))
    graph.add_node("metalabel_agent", _make_node_with_config(metalabel_agent_node, cfg, "metalabel_agent"))
    graph.add_node("reasoning_agent", _make_node_with_config(reasoning_agent_node, cfg, "reasoning_agent"))
    graph.add_node("narrator", _make_node_with_config(narrator_agent_node, cfg, "narrator"))
    graph.add_node("critic_agent", _make_node_with_config(critic_agent_node, cfg, "critic_agent"))

    # Wire edges
    graph.set_entry_point("orchestrator")

    # Parallel fan-out: orchestrator → market + news
    graph.add_edge("orchestrator", "market_agent")
    graph.add_edge("orchestrator", "news_agent")

    # Fan-in: both data agents → predict
    graph.add_edge("market_agent", "predict_agent")
    graph.add_edge("news_agent", "predict_agent")

    # Sequential: predict → gate → horizon_interaction (adjust) → risk (veto)
    # → metalabel (veto) → narrator → critic → reasoning_agent (reflects on the
    # real critic_status) → END
    graph.add_edge("predict_agent", "gate_agent")
    graph.add_edge("gate_agent", "horizon_interaction_agent")
    graph.add_edge("horizon_interaction_agent", "risk_agent")
    graph.add_edge("risk_agent", "metalabel_agent")
    graph.add_edge("metalabel_agent", "narrator")
    graph.add_edge("narrator", "critic_agent")
    graph.add_edge("critic_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", END)

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
