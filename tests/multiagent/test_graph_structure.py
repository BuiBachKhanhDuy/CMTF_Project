"""Structural test for graph.py's compiled topology — confirms the reasoning-agent
insertion stayed a plain sequential node (no LangGraph conditional edges/cycles were
added, per the deliberate "no loop-back" design decision), and that it runs LAST
(after critic_agent), which is what makes its `critic_status == "failed"` trigger
reachable at all — see reasoning_agent.py's module docstring."""

from src.multiagent.config import MultiAgentConfig
from langgraph.graph import END
from src.multiagent.graph import build_graph


def test_reasoning_agent_is_a_plain_sequential_node():
    graph = build_graph(MultiAgentConfig(evaluation_mode=True))
    graph_repr = graph.get_graph()

    assert "reasoning_agent" in graph_repr.nodes

    edges = {(e.source, e.target): e.conditional for e in graph_repr.edges}
    assert edges[("critic_agent", "reasoning_agent")] is False
    assert edges[("reasoning_agent", END)] is False

    # No conditional edges ANYWHERE in the compiled graph — the whole point of the
    # simplified design is that this stays a pure DAG, not a cycle.
    assert not any(e.conditional for e in graph_repr.edges)

    # reasoning_agent has exactly one outgoing edge (to END) — never a loop-back
    # target for itself or an earlier node.
    outgoing = [e for e in graph_repr.edges if e.source == "reasoning_agent"]
    assert len(outgoing) == 1
    assert outgoing[0].target == END

    # It must run AFTER narrator/critic_agent (not before) so that critic_status
    # is real, not permanently absent.
    incoming = [e for e in graph_repr.edges if e.target == "reasoning_agent"]
    assert len(incoming) == 1
    assert incoming[0].source == "critic_agent"
