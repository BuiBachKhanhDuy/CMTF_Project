from src.multiagent.agents.orchestrator_agent import orchestrator_node
from src.multiagent.config import DEFAULT_CONFIG
from dataclasses import replace
cfg = replace(DEFAULT_CONFIG, evaluation_mode=False)
with open("logs/test_date_parse_out.txt", "w", encoding="utf-8") as f:
    for q in ['dự báo xu hướng cổ phiếu VCB ngắn hạn sau 16/7 sắp tới', 'dự báo xu hướng cổ phiếu VCB ngắn hạn sau 15/7']:
        out = orchestrator_node({'query_text': q, 'node_timings': {}}, cfg)
        f.write(f"{q}\n")
        f.write(f"  intent={out.get('query_intent')} date_start={out.get('date_start')} date_end={out.get('date_end')} route_reason={out.get('route_reason')} horizon={out.get('target_horizon')}\n")
