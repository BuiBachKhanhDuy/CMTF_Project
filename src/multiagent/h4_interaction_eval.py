"""H4 — cross-horizon interaction evaluation. Reuses the EXISTING 280-row LLM sample
(``h3_forecaster.json``, no re-run, no new LLM calls) — the same convention
`metalabel_eval.py` uses, so results are directly comparable and no new sample cost is
incurred.

Compares, on the identical rows:
- **MAS (baseline):** the deployed gate's decision on the frozen champion prediction.
- **MAS + horizon interaction:** the same gate decision, with position size scaled by
  the frozen `HorizonInteractionPolicy` (`VN_{H}d_xh.json`) using the OTHER two
  horizons' TEST predictions for the same (symbol, date) — already cached, no new
  inference needed.
- **Placebo:** the interaction applied with the other-horizon predictions permuted
  across rows (breaks the true correspondence), evaluated identically — this is the
  same permutation-placebo idea used at calibration time, now checked out-of-sample.

H4, like every other pre-registered hypothesis in this project, is reported honestly
either way: a null result here is a correct, useful outcome, not a failure to hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .frozen_predictions import PredictionNotCachedError, get_store
from .gate_io import load_gate_policy, policy_path
from .horizon_interaction_io import interaction_policy_path, load_interaction_policy

_OTHER_HORIZONS = {1: (5, 20), 5: (1, 20), 20: (1, 5)}


def _battery(sign, sized, mask, truth, cost_bps=25):
    """Identical shape/formulas to `metalabel_eval.py::_battery` — same metrics,
    same conventions, so H4 numbers are directly comparable to the metalabel result."""
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


def _fetch_other_horizon_preds(recs: list[dict], horizon: int, config: MultiAgentConfig) -> np.ndarray:
    """TEST predictions for `horizon` at the same (symbol,date) as each record in
    `recs`. NaN (never fabricated) where the row isn't cached at that horizon."""
    store = get_store(horizon, config)
    out = np.full(len(recs), np.nan)
    n_missing = 0
    for i, r in enumerate(recs):
        try:
            out[i] = store.get(r["symbol"], r["date"]).gate_pred
        except PredictionNotCachedError:
            n_missing += 1
    if n_missing:
        from loguru import logger
        logger.warning("h4_interaction_eval: {} / {} rows missing a cached {}d prediction "
                       "(treated as non-agreeing)", n_missing, len(recs), horizon)
    return out


def run_h4_interaction_eval(horizon: int = 5, config: MultiAgentConfig | None = None,
                            llm_sample_path: str | None = None, placebo_seed: int = 0) -> dict:
    cfg = config or DEFAULT_CONFIG

    sample_path = Path(llm_sample_path or f"results/agent_ablation/{horizon}d/h3_forecaster.json")
    d = json.load(open(sample_path, encoding="utf-8"))
    recs = d["records"]
    n = len(recs)

    from src.benchmark.decision_policy import apply_positions

    policy, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, horizon, "VN"),
                                 expect_cmtf_version=cfg.cmtf_version,
                                 expect_backbone_version=cfg.backbone_version)
    xh_policy, xh_meta = load_interaction_policy(
        interaction_policy_path(cfg.horizon_interaction_dir, horizon, "VN"),
        expect_cmtf_version=cfg.cmtf_version, expect_backbone_version=cfg.backbone_version,
    )

    truth = np.array([r["truth"] for r in recs], float)
    gate_pred = np.array([r["gate_pred"] for r in recs], float)
    mas_trade = np.abs(gate_pred) >= policy.tau
    mas_pos = apply_positions(gate_pred, policy)
    primary_sign = np.sign(gate_pred)

    other_preds = [_fetch_other_horizon_preds(recs, h, cfg) for h in xh_policy.other_horizons]

    def _agreement(preds_list):
        return sum((np.sign(p) == primary_sign).astype(int) for p in preds_list)

    agreement = _agreement(other_preds)
    multiplier = np.array([xh_policy.multiplier_by_agreement[int(a)] for a in agreement])
    xh_pos = np.where(mas_trade, mas_pos * multiplier, 0.0)

    # Placebo: permute which row's other-horizon predictions are consulted (test-set
    # version of the same check performed at calibration time), averaged over several
    # draws for a stable comparison.
    rng = np.random.default_rng(placebo_seed)
    placebo_batteries = []
    for _ in range(10):
        perm = rng.permutation(n)
        placebo_other = [p[perm] for p in other_preds]
        placebo_agreement = _agreement(placebo_other)
        placebo_mult = np.array([xh_policy.multiplier_by_agreement[int(a)] for a in placebo_agreement])
        placebo_pos = np.where(mas_trade, mas_pos * placebo_mult, 0.0)
        placebo_batteries.append(_battery(primary_sign, placebo_pos, mas_trade, truth))
    placebo_sharpe_mean = float(np.mean([b["Sharpe"] for b in placebo_batteries]))

    baseline_battery = _battery(primary_sign, mas_pos, mas_trade, truth)
    xh_battery = _battery(primary_sign, xh_pos, mas_trade, truth)

    out = {
        "hypothesis": "H4",
        "description": ("Does the cross-horizon interaction layer (VN_{H}d_xh.json) "
                        "improve on the MAS baseline, out-of-sample (TEST), on the "
                        "same rows used for H3?"),
        "horizon": horizon, "primary_horizon": horizon,
        "other_horizons": list(xh_policy.other_horizons),
        "n_rows": n, "n_source_sample": str(sample_path),
        "n_mas_trades": int(mas_trade.sum()),
        "calibration_summary": {
            "multiplier_by_agreement": xh_policy.multiplier_by_agreement,
            "real_lift_over_baseline_at_calibration": xh_meta.get("real_lift_over_baseline"),
            "beat_placebo_at_calibration": xh_meta.get("real_lift_beats_placebo_lift"),
        },
        "MAS_baseline": baseline_battery,
        "MAS_plus_horizon_interaction": xh_battery,
        "MAS_plus_interaction_placebo_mean": {"Sharpe": round(placebo_sharpe_mean, 3)},
        "sharpe_delta_xh_minus_baseline": round(xh_battery["Sharpe"] - baseline_battery["Sharpe"], 4)
                                          if np.isfinite(xh_battery["Sharpe"]) and np.isfinite(baseline_battery["Sharpe"])
                                          else None,
        "sharpe_delta_beats_placebo": bool(
            xh_battery["Sharpe"] - baseline_battery["Sharpe"] > placebo_sharpe_mean - baseline_battery["Sharpe"]
        ) if np.isfinite(xh_battery["Sharpe"]) and np.isfinite(baseline_battery["Sharpe"]) else None,
    }
    out_path = sample_path.parent / "h4_interaction_eval.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["out_path"] = str(out_path)
    return out
