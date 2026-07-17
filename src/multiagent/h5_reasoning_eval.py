"""H5 — reasoning-agent (single-pass reflection) evaluation.

Does the reasoning agent's widen-and-rerun path (triggered on thin news coverage /
cross-horizon disagreement / a failed critic verification) actually improve
decisions on real historical data, or is it a wash — or net negative? Unlike H4
(which reused a frozen, static LLM sample with no attached price/news windows),
this genuinely runs the decision chain twice per row — the reasoning agent's whole
mechanism depends on access to a WIDER slice of real price/news data than a static
sample can provide, so there is no shortcut around real (cheap, in-book) forward
passes here.

Compares, on the same real sampled (symbol, date) rows at horizon=`horizon`:
- MAS baseline: predict -> gate -> horizon_interaction -> risk -> metalabel ->
  narrator -> critic_agent (no reasoning agent at all). narrator/critic_agent run
  too — not because their text feeds the DA/Sharpe metrics below (it doesn't), but
  because reasoning_agent needs a REAL critic_status to evaluate its
  `critic_verification_failed` trigger, exactly like the production chat.py path.
- MAS + reasoning: the same chain, plus the reasoning agent's single-pass
  reflection (may widen-and-rerun on the pre-registered trigger conditions).

Reports DA/Sharpe for both arms on (a) all sampled rows, (b) the subset where the
reasoning agent triggered at all, and (c) the subset where it *actually changed*
the decision (evidence_widened=True and the action/size differ from baseline) —
this last one is the only subset that can show a difference, since a triggered-
but-unchanged row is identical between arms by construction, and pooling those in
would dilute a real effect into a wall of no-op comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .h3_faithfulness import _sample_rows
from .agents.reasoning_agent import reasoning_agent_node


def _battery(sign, sized, mask, truth, cost_bps=25):
    """Same shape/formulas as `metalabel_eval.py`/`h4_interaction_eval.py::_battery` —
    consistent metrics across every H-hypothesis in this project."""
    n = len(truth)
    nt = int(mask.sum())
    if nt == 0:
        return {"coverage": 0.0, "n_trades": 0, "DA": float("nan"), "IC": float("nan"),
                "Sharpe": float("nan"), "profit_factor": float("nan"), "max_drawdown": float("nan"),
                "net_PnL@25bps": 0.0}
    da = float((np.sign(sign[mask]) == np.sign(truth[mask])).mean()) * 100
    ic, _ = stats.spearmanr(sign[mask], truth[mask]) if nt >= 3 else (float("nan"), 0)
    pnl_all = sized * truth
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


def _run_baseline_chain(symbol, date, horizon, cfg, frame, news_idx):
    import chat
    evidence = chat._gather_evidence(frame, news_idx, symbol, date)
    state = {
        "symbol": symbol, "target_horizon_days": horizon, "prediction_time": date,
        "artifact_versions": {}, "node_timings": {}, **evidence,
    }
    # Full predict->...->critic_agent chain (not just the 5-node decision core) —
    # reasoning_agent_node needs a REAL critic_status to evaluate its
    # `critic_verification_failed` trigger, exactly like the real chat.py path.
    for _, fn in chat._RERUN_CHAIN:
        state.update(fn(state, cfg))
    return state


def run_reasoning_eval(horizon: int = 5, n: int = 60, seed: int = 0,
                       config: MultiAgentConfig | None = None) -> dict:
    import chat
    from .frozen_predictions import get_store
    from .live_inference import resolve_price_parquet
    from .news_data import load_news_index
    from loguru import logger

    cfg = config or DEFAULT_CONFIG
    store = get_store(horizon, cfg)
    rows = _sample_rows(store, n, seed)

    frame = pd.read_parquet(resolve_price_parquet(horizon, allow_missing_target=True))
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    news_idx = load_news_index()

    records = []
    dropped = []
    for i, (symbol, date) in enumerate(rows, 1):
        state = _run_baseline_chain(symbol, date, horizon, cfg, frame, news_idx)
        truth = (state.get("model_evidence") or {}).get("truth")
        if truth is None:
            dropped.append((symbol, date))
            continue

        baseline_action = state.get("action")
        baseline_size = float(state.get("position_scale", 0.0))

        def widen_and_rerun(current_state, _symbol=symbol, _date=date):
            wider_evidence = chat._gather_evidence(
                frame, news_idx, _symbol, _date,
                lookback_days=cfg.reasoning_widen_lookback_days_to,
                vol_window=cfg.reasoning_widen_sequence_len_to,
            )
            rerun_state = {**current_state, **wider_evidence}
            for _, fn in chat._RERUN_CHAIN:  # same full chain, mirrors chat.py's _widen_and_rerun
                rerun_state.update(fn(rerun_state, cfg))
            return rerun_state

        reasoning_update = reasoning_agent_node(dict(state), cfg, widen_and_rerun=widen_and_rerun)
        reasoning_action = reasoning_update.get("action", baseline_action)
        reasoning_size = float(reasoning_update.get("position_scale", baseline_size))

        records.append({
            "symbol": symbol, "date": date, "truth": truth,
            "baseline_action": baseline_action, "baseline_size": baseline_size,
            "reasoning_action": reasoning_action, "reasoning_size": reasoning_size,
            "triggered_reasons": reasoning_update.get("reasoning_triggered_reasons", []),
            "evidence_widened": reasoning_update.get("reasoning_evidence_widened", False),
        })
        if i % 10 == 0 or i == len(rows):
            logger.info("h5_reasoning_eval {}/{} | triggered so far: {}", i, len(rows),
                       sum(1 for r in records if r["triggered_reasons"]))

    n_rows = len(records)
    truth = np.array([r["truth"] for r in records], float)

    def _sign_size(action_key, size_key):
        sign = np.array([{"long": 1.0, "short": -1.0, "abstain": 0.0}[r[action_key]] for r in records])
        size = np.array([r[size_key] for r in records], float)
        mask = sign != 0.0
        return sign, size, mask

    b_sign, b_size, b_mask = _sign_size("baseline_action", "baseline_size")
    r_sign, r_size, r_mask = _sign_size("reasoning_action", "reasoning_size")

    triggered_idx = np.array([bool(r["triggered_reasons"]) for r in records])
    changed_idx = np.array([
        r["baseline_action"] != r["reasoning_action"] or abs(r["baseline_size"] - r["reasoning_size"]) > 1e-9
        for r in records
    ])

    def _subset_battery(idx):
        if idx.sum() == 0:
            return {"n_rows": 0, "baseline": None, "reasoning": None}
        return {
            "n_rows": int(idx.sum()),
            "baseline": _battery(b_sign[idx], b_size[idx], b_mask[idx], truth[idx]),
            "reasoning": _battery(r_sign[idx], r_size[idx], r_mask[idx], truth[idx]),
        }

    out = {
        "hypothesis": "H5",
        "description": ("Does the reasoning agent's single-pass widen-and-rerun reflection "
                        "improve decisions vs the MAS baseline (no reasoning agent), on real "
                        "historical (symbol,date) rows?"),
        "horizon": horizon, "n_requested": n, "n_run": n_rows, "n_dropped": len(dropped),
        "sample_seed": seed,
        "n_triggered": int(triggered_idx.sum()),
        "n_changed_decision": int(changed_idx.sum()),
        "trigger_rate": round(float(triggered_idx.mean()), 4) if n_rows else None,
        "all_rows": _subset_battery(np.ones(n_rows, dtype=bool)),
        "triggered_rows": _subset_battery(triggered_idx),
        "changed_decision_rows": _subset_battery(changed_idx),
    }
    out_path = Path("results/agent_ablation") / f"{horizon}d" / "h5_reasoning_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**out, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    out["out_path"] = str(out_path)
    return out
