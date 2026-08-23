"""Evaluate the metalabel veto against the frozen multi-agent sample.

The evaluation compares the base gate, the metalabel-enhanced gate, and stored
independent-forecaster records on the same symbol-date rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .agents.metalabel_agent import _parse_flags, _SYSTEM, EVENT_CATEGORIES
from .config import DEFAULT_CONFIG, MultiAgentConfig
from .gate_io import load_gate_policy, policy_path
from .news_data import load_news_index, recent_headlines


def _battery(sign, sized, mask, truth, cost_bps=25):
    n = len(truth)
    nt = int(mask.sum())
    if nt == 0:
        return {"coverage": 0.0, "n_trades": 0, "DA": float("nan"), "IC": float("nan"),
                "Sharpe": float("nan"), "profit_factor": float("nan"), "max_drawdown": float("nan"),
                "net_PnL@25bps": 0.0}
    da = float((np.sign(sign[mask]) == np.sign(truth[mask])).mean()) * 100
    ic, _ = stats.spearmanr(sign[mask], truth[mask]) if nt >= 3 else (float("nan"), 0)
    pnl_all = sized * truth  # zero where not traded
    traded_pnl = pnl_all[mask]
    sharpe = float(traded_pnl.mean() / (traded_pnl.std() + 1e-9) * np.sqrt(252 / 5))
    wins = traded_pnl[traded_pnl > 0]; losses = traded_pnl[traded_pnl < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    cum = np.cumsum(pnl_all); peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max())
    net = float(traded_pnl.sum() - cost_bps / 1e4 * nt)
    return {"coverage": round(nt / n, 3), "n_trades": nt, "DA": round(da, 2),
            "IC": round(float(ic), 4), "Sharpe": round(sharpe, 3),
            "profit_factor": round(pf, 3), "max_drawdown": round(max_dd, 5),
            "net_PnL@25bps": round(net, 4)}


def run_metalabel_eval(horizon: int = 5, config: MultiAgentConfig | None = None,
                       llm_sample_path: str | None = None) -> dict:
    from loguru import logger
    from .guards import ensure_local_no_proxy

    cfg = config or DEFAULT_CONFIG
    ensure_local_no_proxy(cfg.ollama_base_url)
    from langchain_ollama import ChatOllama

    sample_path = Path(llm_sample_path or f"results/agent_ablation/{horizon}d/h3_forecaster.json")
    d = json.load(open(sample_path, encoding="utf-8"))
    recs = d["records"]
    n = len(recs)

    policy, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, horizon, "VN"),
                                 expect_cmtf_version=cfg.cmtf_version,
                                 expect_backbone_version=cfg.backbone_version)
    news_idx = load_news_index()
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url,
                     temperature=0.1, timeout=cfg.ollama_timeout)

    truth = np.array([r["truth"] for r in recs], float)
    gate_pred = np.array([r["gate_pred"] for r in recs], float)
    a1_sign = np.array([r["a1_sign"] for r in recs], float)
    a1_conf = np.array([r["a1_conf"] for r in recs], float)

    mas_trade = np.abs(gate_pred) >= policy.tau
    from src.benchmark.decision_policy import apply_positions
    mas_pos = apply_positions(gate_pred, policy)

    ckpt = sample_path.parent / "metalabel_partial.json"
    flags_all: list[list[str]] = []
    for i, r in enumerate(recs, 1):
        if mas_trade[i - 1]:  # only classify rows the MAS would actually trade (cheaper, and the
            # only rows where a veto could matter — consistent with the one-way invariant)
            heads = recent_headlines(news_idx, r["symbol"], r["date"], lookback_days=5, k=15)
            flags = []
            if heads:
                msg = f"Mã: {r['symbol']}\nTin tức:\n" + "\n".join(f"- {h}" for h in heads)
                flags = _parse_flags(llm.invoke([("system", _SYSTEM), ("human", msg)]).content)
        else:
            flags = []
        flags_all.append(flags)
        if i % 20 == 0 or i == n:
            logger.info("metalabel_eval {}/{} | vetoes so far: {}", i, n,
                        sum(1 for f in flags_all if f))
            ckpt.write_text(json.dumps({"done": i, "total": n, "flags": flags_all}, ensure_ascii=False),
                           encoding="utf-8")

    metalabel_vetoed = np.array([bool(f) and mas_trade[i] for i, f in enumerate(flags_all)])
    mas_ml_trade = mas_trade & ~metalabel_vetoed
    mas_ml_pos = np.where(mas_ml_trade, mas_pos, 0.0)

    llm_trade = a1_sign != 0
    llm_pos = a1_sign  # flat-unit, matching the earlier reported "LLM (flat unit)" convention

    out = {
        "n_rows": n, "n_source_sample": str(sample_path),
        "pre_registered_categories": list(EVENT_CATEGORIES),
        "n_mas_trades_classified": int(mas_trade.sum()),
        "n_metalabel_vetoed": int(metalabel_vetoed.sum()),
        "MAS_baseline": _battery(np.sign(gate_pred), mas_pos, mas_trade, truth),
        "MAS_plus_metalabel": _battery(np.sign(gate_pred), mas_ml_pos, mas_ml_trade, truth),
        "Plain_LLM": _battery(a1_sign, llm_pos, llm_trade, truth),
    }
    out_path = sample_path.parent / "metalabel_eval.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["out_path"] = str(out_path)
    return out
