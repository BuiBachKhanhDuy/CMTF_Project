"""Agent panel for the multi-agent inference system."""

from .orchestrator_agent import orchestrator_node
from .market_agent import market_agent_node
from .news_agent import news_agent_node
from .predict_agent import predict_agent_node
from .gate_agent import gate_agent_node
from .risk_agent import risk_agent_node
from .metalabel_agent import metalabel_agent_node
from .narrator_agent import narrator_agent_node
from .critic_agent import critic_agent_node
from .rank_agent import rank_agent_node
from .research_agent import research_agent_node

__all__ = [
    "orchestrator_node",
    "market_agent_node",
    "news_agent_node",
    "predict_agent_node",
    "gate_agent_node",
    "risk_agent_node",
    "metalabel_agent_node",
    "narrator_agent_node",
    "critic_agent_node",
    "rank_agent_node",
    "research_agent_node",
]
