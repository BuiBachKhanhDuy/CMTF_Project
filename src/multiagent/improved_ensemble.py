"""Improved MAS decision layer — de-biased, validation-selected backbone fusion.

Findings that motivate this (all measured, see the results report):
- Model-level fusion of the diverse CMTF backbones beats the single champion.
- The weak backbones carry a directional BIAS; centering them (predict relative to
  their own mean) lifts standalone DA ~+7 pts.
- Leave-one-out: cnn_lstm is the least valuable member (marginally hurts).

This module builds the improved predictor **leak-free without retraining**, using a
time-split *within the test book*: everything that must be "learned" — the per-model
centering statistics, the best backbone subset, and the gate threshold — is fit on the
EARLY portion (calib) and evaluated on the LATE portion (eval). No look-ahead.

It exposes per-(symbol, date) improved-MAS predictions so the MAS-vs-LLM comparison
(``compare_mas_vs_llm``) can be recomputed on the exact rows the LLM forecaster saw.
"""

from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .config import DEFAULT_CONFIG, MultiAgentConfig

PD = Path("cache/predictions")
BACKBONES = {"lstm": "60f71e5ce5", "cnn_lstm": "436c539bab",
             "gpt4ts": "224ea408d2", "chronos": "def461a1ee"}


def _load(horizon):
    truth = np.load(PD / f"truth__{horizon}d.npy").astype(float)
    days = np.asarray(np.load(PD / f"test_times__{horizon}d.npy", allow_pickle=True)).astype("datetime64[D]")
    syms = np.load(PD / f"test_symbols__{horizon}d.npy", allow_pickle=True)
    preds = {}
    for name, hh in BACKBONES.items():
        fs = [f for f in sorted(glob.glob(str(PD / f"{hh}__seed*__{horizon}d.npy"))) if "__val__" not in f]
        preds[name] = np.mean([np.load(f).astype(float) for f in fs], axis=0)
    return preds, truth, days, syms


def _gated_da_ic(sig, truth, cov=0.25):
    n = len(sig); k = max(1, int(np.ceil(cov * n)))
    o = np.argsort(-np.abs(sig))[:k]
    da = float((np.sign(sig[o]) == np.sign(truth[o])).mean()) * 100
    ic, _ = stats.spearmanr(sig[o], truth[o])
    return da, float(ic)


def build_improved_mas(horizon: int = 5, calib_frac: float = 0.6,
                       coverage: float = 0.25, config: MultiAgentConfig | None = None) -> dict:
    """Fit centering + subset + gate on the early (calib) test slice; return eval preds.

    Returns per-row improved-MAS prediction (centered z-mean of the selected subset),
    the champion (lstm) prediction, the eval mask, and the fitted choices — all leak-free.
    """
    preds, truth, days, syms = _load(horizon)
    uniq = np.sort(np.unique(days))
    cut = uniq[int(len(uniq) * calib_frac)]
    calib = days < cut
    evalm = days >= cut

    # 1. De-bias: per-model mean/std fit on CALIB only, applied everywhere (leak-free).
    zc = {}
    for name, p in preds.items():
        mu, sd = float(p[calib].mean()), float(p[calib].std() + 1e-9)
        zc[name] = (p - mu) / sd

    # 2. Subset selection: best gated DA on CALIB (leak-free — uses calib truth only).
    names = list(BACKBONES)
    best, best_da = None, -1.0
    for r in range(1, len(names) + 1):
        for cmb in itertools.combinations(names, r):
            ens = np.mean([zc[k] for k in cmb], axis=0)
            da, _ = _gated_da_ic(ens[calib], truth[calib], coverage)
            if da > best_da:
                best_da, best = da, cmb

    ens_all = np.mean([zc[k] for k in best], axis=0)
    # 3. Gate threshold: (1-coverage) quantile of |ens| on CALIB (leak-free).
    tau = float(np.quantile(np.abs(ens_all[calib]), 1.0 - coverage))

    return {
        "ens": ens_all, "champion": zc["lstm"], "eval_mask": evalm, "calib_mask": calib,
        "truth": truth, "days": days, "syms": syms, "tau": tau,
        "selected_subset": list(best), "cut_date": str(cut), "coverage": coverage,
    }


def _battery(sig, truth, tau=None, cov=0.25, cost_bps=25):
    """Full metric panel for a signal on a set of rows."""
    n = len(sig)
    if tau is None:
        k = max(1, int(np.ceil(cov * n))); trade = np.argsort(-np.abs(sig))[:k]
        mask = np.zeros(n, bool); mask[trade] = True
    else:
        mask = np.abs(sig) >= tau
    nt = int(mask.sum())
    if nt == 0:
        return {"coverage": 0.0, "n_trades": 0, "DA": float("nan")}
    da = float((np.sign(sig[mask]) == np.sign(truth[mask])).mean()) * 100
    ic, _ = stats.spearmanr(sig[mask], truth[mask])
    pnl = np.sign(sig[mask]) * truth[mask]
    sharpe = float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252.0 / 5))
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    net = float(pnl.sum() - cost_bps / 1e4 * nt)
    return {"coverage": round(nt / n, 3), "n_trades": nt, "DA": round(da, 2),
            "IC": round(float(ic), 4), "Sharpe": round(sharpe, 3),
            "profit_factor": round(pf, 3), "net_PnL@25bps": round(net, 4)}


def h1_h2_battery(horizon: int = 5, config: MultiAgentConfig | None = None) -> dict:
    """H1 (fusion IC) + H2 (gate / AURC / selective prediction) for the MULTI-CMTF MAS.

    Evaluated leak-free via time-split: de-bias + subset + gate calibrated on the early
    test slice, all metrics measured on the late slice. Compares the multi-model ensemble
    to the single champion (lstm) on identical eval rows. (Official validation-split
    calibration would need backbone val predictions, which require a retrain; the time-split
    is a valid leak-free stand-in — labelled as such.)
    """
    from src.benchmark.calibration import aurc, paired_bootstrap_aurc, selective_da_at_coverage

    imp = build_improved_mas(horizon, config=config)
    ens, champ, evalm, truth = imp["ens"], imp["champion"], imp["eval_mask"], imp["truth"]
    cov = imp["coverage"]
    e_ens, e_champ, e_tru = ens[evalm], champ[evalm], truth[evalm]
    n = len(e_tru)
    rng = np.random.default_rng(0)

    def battery(sig, tau):
        conf = np.abs(sig)
        # H2 calibration: AURC vs no-skill (shuffled confidence) + selective curve
        a_gate = aurc(sig, e_tru, conf)
        a_ns = aurc(sig, e_tru, conf[rng.permutation(n)])
        boot = paired_bootstrap_aurc(sig, conf, sig, conf[rng.permutation(n)], e_tru, n_boot=3000)
        full_da = float((np.sign(sig) == np.sign(e_tru)).mean()) * 100
        sel = selective_da_at_coverage(sig, e_tru, conf, cov)
        # gated (tau from calib): H1 gated IC + gated DA + coverage
        m = conf >= tau; nt = int(m.sum())
        g_da = float((np.sign(sig[m]) == np.sign(e_tru[m])).mean()) * 100 if nt else float("nan")
        g_ic, _ = stats.spearmanr(sig[m], e_tru[m]) if nt >= 3 else (float("nan"), 0)
        full_ic, _ = stats.spearmanr(sig, e_tru)
        return {
            "full_book_DA": round(full_da, 2), "full_book_IC": round(float(full_ic), 4),
            "selective_DA@cov": round(sel["DA%"], 2), "gated_DA": round(g_da, 2),
            "gated_IC": round(float(g_ic), 4), "gate_coverage": round(nt / n, 3),
            "AURC_gate": round(a_gate, 4), "AURC_noskill": round(a_ns, 4),
            "delta_AURC": round(boot["delta_aurc"], 4),
            "delta_AURC_ci": [round(boot["ci_low"], 4), round(boot["ci_high"], 4)],
            "delta_AURC_significant": boot["significant"],
        }

    # gate thresholds from calib slice (leak-free)
    cens = ens[imp["calib_mask"]]; cch = champ[imp["calib_mask"]]
    tau_ens = float(np.quantile(np.abs(cens), 1 - cov))
    tau_ch = float(np.quantile(np.abs(cch), 1 - cov))

    out = {
        "model": "multi-CMTF ensemble MAS vs single champion",
        "calibration": "leak-free time-split (calib=early test, eval=late test)",
        "eval_window": f">= {imp['cut_date']}", "n_eval": n,
        "selected_subset": imp["selected_subset"],
        "H1_H2_multi_model_MAS": battery(e_ens, tau_ens),
        "H1_H2_single_champion": battery(e_champ, tau_ch),
        "H1b_cross_sectional": ("not computable for the multi-CMTF ensemble — the backbones are "
                                "all-scope; matched-scope variants of every backbone are not cached "
                                "(only lstm-matched exists). Cross-sectional IC needs matched scope."),
    }
    o = Path("results/agent_ablation") / f"{horizon}d" / "h1_h2_multimodel.json"
    o.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["out_path"] = str(o)
    return out


def compare_mas_vs_llm(horizon: int = 5, config: MultiAgentConfig | None = None) -> dict:
    """Improved MAS vs the LLM forecaster, on the shared eval rows, full battery."""
    imp = build_improved_mas(horizon, config=config)
    ens, champ, evalm, truth, days, syms = (imp["ens"], imp["champion"], imp["eval_mask"],
                                            imp["truth"], imp["days"], imp["syms"])
    # (symbol, date) -> row index, restricted to eval window
    idx = {(str(s), d): i for i, (s, d) in enumerate(zip(syms, days))}

    # Load the LLM forecaster records (real-input run).
    llm = json.loads((Path("results/agent_ablation") / f"{horizon}d" / "h3_forecaster.json").read_text(encoding="utf-8"))
    rows = llm["records"]
    common = []
    for r in rows:
        key = (r["symbol"], np.datetime64(r["date"], "D"))
        i = idx.get(key)
        if i is not None and evalm[i]:
            common.append((i, r))
    if not common:
        return {"error": "no overlap between LLM rows and eval window"}

    ci = np.array([i for i, _ in common])
    tr = truth[ci]
    mas_sig = ens[ci]
    champ_sig = champ[ci]
    llm_sig = np.array([r["a1_sign"] * max(r["a1_conf"], 1e-3) for _, r in common])  # signed by conf

    out = {
        "eval_window": f">= {imp['cut_date']}",
        "selected_subset": imp["selected_subset"],
        "n_common_rows": len(common),
        "note": ("Leak-free: de-bias + subset + gate fit on early test slice; evaluated on "
                 "late slice ∩ LLM rows. MAS gated by top-25% |ensemble|; LLM by its own confidence."),
        "improved_MAS": _battery(mas_sig, tr),
        "champion_lstm": _battery(champ_sig, tr),
        "LLM_plain_call": _battery(llm_sig, tr),
    }
    o = Path("results/agent_ablation") / f"{horizon}d" / "improved_mas_vs_llm.json"
    o.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["out_path"] = str(o)
    return out
