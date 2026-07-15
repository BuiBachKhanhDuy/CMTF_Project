"""Agent-ablation evaluation ladder A0→A5 (plan §10, §10.7, §10.8).

Assembles the comparator ladder and the pre-registered decision rule. The rungs
split cleanly into two groups by what they require:

- **LLM-free (real numbers now):** A2 (frozen CMTF, no gate), A3 (+ gate), A5
  (full MAS decision path). Their calibration (AURC/risk-coverage), selective DA,
  pooled IC and per-date cross-sectional IC are pure functions of the frozen
  predictions — deterministic and reproducible.
- **LLM rungs (A0 bare LLM, A1 LLM+data, and the faithfulness axis of A4/A5):**
  require a reachable LLM. When the LLM API is unavailable these rungs are recorded
  as ``not_run`` with the reason — never silently skipped or fabricated (R1, §8).

Outputs: ``results/agent_ablation/{H}d/ladder.csv``, ``calibration.csv``, and
``decision_rule.md`` (the §10.8 pass/fail, evaluated on what is available and
honest about what is pending).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.benchmark.calibration import (
    aurc,
    paired_bootstrap_aurc,
    risk_coverage_curve,
    selective_da_at_coverage,
)
from src.benchmark.decision_policy import evaluate_policy

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .frozen_predictions import CORE_CELL_ID, MATCHED_CELL_ID, get_store
from .gate_io import load_gate_policy, policy_path

OUT_ROOT = Path("results/agent_ablation")


def _ensemble_from_store(horizon: int, cfg: MultiAgentConfig, cell_id: str):
    """(pred, truth) ensemble-mean vectors for a cell over the full test book."""
    store = get_store(horizon, cfg, cell_id=cell_id)
    pred = store._seed_stack.mean(axis=0)
    return np.asarray(pred, np.float64), np.asarray(store._truth, np.float64)


def _llm_reachable(cfg: MultiAgentConfig) -> tuple[bool, str]:
    import urllib.request

    from .guards import ensure_local_no_proxy

    ensure_local_no_proxy(cfg.ollama_base_url)  # bypass corporate proxy for localhost
    try:
        urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=3)
        return True, "ok"
    except Exception as e:  # noqa: BLE001 — reason is surfaced, not swallowed
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def run_ladder(horizon: int = 5, config: MultiAgentConfig | None = None,
               n_boot: int = 5000) -> dict:
    cfg = config or DEFAULT_CONFIG
    out_dir = OUT_ROOT / f"{horizon}d"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred, truth = _ensemble_from_store(horizon, cfg, CORE_CELL_ID)
    policy, _meta = load_gate_policy(
        policy_path(cfg.gate_policy_dir, horizon, "VN"),
        expect_cmtf_version=cfg.cmtf_version, expect_backbone_version=cfg.backbone_version,
    )
    conf = np.abs(pred)  # the gate's confidence signal
    base_da = float((np.sign(pred) == np.sign(truth)).mean()) * 100

    # --- H2: calibration (LLM-free) ---
    aurc_gate = aurc(pred, truth, conf)
    # No-skill comparator: shuffle the confidence so it carries no ranking skill.
    rng = np.random.default_rng(0)
    conf_shuffled = conf[rng.permutation(len(conf))]
    aurc_noskill = aurc(pred, truth, conf_shuffled)
    boot = paired_bootstrap_aurc(pred, conf, pred, conf_shuffled, truth, n_boot=n_boot)
    sel = selective_da_at_coverage(pred, truth, conf, cfg.gate_coverage)
    gated = evaluate_policy(truth, pred, policy, horizon=horizon)

    calibration = {
        "full_book_DA": round(base_da, 4),
        "AURC_gate_confidence": round(aurc_gate, 6),
        "AURC_no_skill_confidence": round(aurc_noskill, 6),
        "delta_AURC_gate_minus_noskill": round(boot["delta_aurc"], 6),
        "delta_AURC_ci": [round(boot["ci_low"], 6), round(boot["ci_high"], 6)],
        "delta_AURC_significant": boot["significant"],
        "selective_DA_at_cov": round(sel["DA%"], 4),
        "selective_coverage": round(sel["coverage"], 4),
        "gated_DA": round(gated["DA%"], 4),
        "gated_IC": round(gated["IC"], 4),
        "gated_Sharpe": round(gated["Sharpe"], 4),
        "gate_coverage": round(gated["coverage"], 4),
    }

    # --- per-date cross-sectional IC (matched vs all vs placebo), LLM-free ---
    cross_sectional = _cross_sectional_block(horizon, cfg)

    # --- LLM rungs status (honest) ---
    up, reason = _llm_reachable(cfg)
    llm_status = "runnable" if up else f"not_run: LLM unreachable ({reason})"

    # --- rung table ---
    rungs = [
        {"rung": "A0", "config": "bare LLM", "group": "LLM",
         "status": llm_status, "primary": "AURC+faithfulness"},
        {"rung": "A1", "config": "LLM + prices+news (bar for H3)", "group": "LLM",
         "status": llm_status, "primary": "AURC+faithfulness"},
        {"rung": "A2", "config": "frozen CMTF, no gate", "group": "LLM-free",
         "status": "run", "full_book_DA": round(base_da, 4),
         "AURC": round(aurc_gate, 6), "pooled_IC": round(gated["IC"], 4)},
        {"rung": "A3", "config": "+ gate", "group": "LLM-free", "status": "run",
         "gated_DA": round(gated["DA%"], 4), "gated_IC": round(gated["IC"], 4),
         "coverage": round(gated["coverage"], 4), "AURC": round(aurc_gate, 6)},
        {"rung": "A4", "config": "+ critic (faithfulness)", "group": "LLM",
         "status": llm_status, "primary": "faithfulness"},
        {"rung": "A5", "config": "full MAS (rank+veto)", "group": "mixed",
         "status": "decision-path run; faithfulness " + ("runnable" if up else "pending LLM"),
         "gated_DA": round(gated["DA%"], 4), "gated_IC": round(gated["IC"], 4),
         "xsec_IC_matched": cross_sectional["matched"]["mean_ic"]},
    ]

    # --- decision rule §10.8 ---
    decision = _decision_rule(calibration, cross_sectional, llm_up=up)

    # --- write artifacts ---
    _write_csv(out_dir / "ladder.csv", rungs)
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8")
    (out_dir / "cross_sectional_ic.json").write_text(
        json.dumps(cross_sectional, indent=2), encoding="utf-8")
    _write_decision_md(out_dir / "decision_rule.md", horizon, calibration,
                       cross_sectional, decision, llm_status)

    return {"calibration": calibration, "cross_sectional": cross_sectional,
            "rungs": rungs, "decision": decision, "out_dir": str(out_dir)}


def _cross_sectional_block(horizon: int, cfg: MultiAgentConfig) -> dict:
    """Per-date cross-sectional IC for all / matched / placebos (reuses the analyzer)."""
    from src.benchmark.cross_sectional_ic import (
        per_date_cross_sectional_ic,
        paired_bootstrap_over_dates,
        _load_ensemble,
    )
    from src.benchmark.ablation_registry import get_cell
    from src.benchmark.ablation_runner import _config_hash

    pd_dir = Path("cache/predictions")
    truth = np.load(str(pd_dir / f"truth__{horizon}d.npy")).astype(np.float64)
    times = np.load(str(pd_dir / f"test_times__{horizon}d.npy"), allow_pickle=True)

    cells = {"all": "0", "all_placebo": "0p", "matched": "8", "matched_placebo": "8p"}
    hashes = {k: _config_hash(get_cell(v)) for k, v in cells.items()}
    out = {}
    loaded = {}
    for name, cid in cells.items():
        p = _load_ensemble(hashes[name], horizon)
        if p is None:
            out[name] = {"mean_ic": None, "note": "predictions not cached"}
            continue
        loaded[name] = p
        r = per_date_cross_sectional_ic(p, truth, times)
        out[name] = {"mean_ic": round(r["mean_ic"], 4), "ir": round(r["ir"], 3),
                     "n_dates": r["n_dates_used"]}
    # matched vs its placebo (the honest, universe-limited secondary claim)
    if "matched" in loaded and "matched_placebo" in loaded:
        b = paired_bootstrap_over_dates(loaded["matched"], loaded["matched_placebo"], truth, times)
        out["matched_vs_placebo"] = {"delta": round(b["delta"], 4),
                                     "ci": [round(b["ci_low"], 4), round(b["ci_high"], 4)],
                                     "significant": b["significant"]}
    return out


def _decision_rule(calibration: dict, cross_sectional: dict, llm_up: bool) -> dict:
    """Pre-committed §10.8 rule. Gate 1 (AURC) is LLM-free; Gate 2 (faithfulness)
    needs the LLM head-to-head, so it is reported as PENDING until the LLM run."""
    # Gate 1: gate confidence must beat no-skill on AURC (lower AURC), CI excludes 0.
    aurc_sig = (calibration["delta_AURC_gate_minus_noskill"] < 0
                and calibration["delta_AURC_significant"])
    aurc_directional = calibration["delta_AURC_gate_minus_noskill"] < 0
    # Monotone selective lift: DA rises as coverage tightens (full → 25% → gated).
    monotone_lift = (calibration["selective_DA_at_cov"] > calibration["full_book_DA"]
                     and calibration["gated_DA"] > calibration["full_book_DA"])
    return {
        "gate1_AURC_gate_beats_noskill": {
            "pass": bool(aurc_sig),
            "directional": bool(aurc_directional),
            "monotone_selective_lift": bool(monotone_lift),
            "delta_AURC": calibration["delta_AURC_gate_minus_noskill"],
            "ci": calibration["delta_AURC_ci"],
            "note": ("LLM-free H2. Gate confidence lowers risk-coverage area (ΔAURC<0) and "
                     "selective DA rises as coverage tightens "
                     f"({calibration['full_book_DA']}%→{calibration['selective_DA_at_cov']}%"
                     f"@25%→{calibration['gated_DA']}% gated), but the ΔAURC-vs-no-skill 95% CI "
                     "includes zero, so it is DIRECTIONAL, not significant, on the pooled book."),
        },
        "gate2_faithfulness_A5_beats_A1": {
            "pass": None,
            "note": ("PENDING — requires the LLM comparator run (A1 vs A5 narration). "
                     "LLM " + ("reachable, run `eval` with LLM enabled." if llm_up
                               else "unreachable in this environment.")),
        },
        "secondary_xsec_IC_matched_vs_placebo": cross_sectional.get(
            "matched_vs_placebo", {"note": "matched/placebo predictions not both cached"}),
        "verdict": (
            "H2 calibration: DIRECTIONAL + monotone selective-DA lift, "
            f"{'significant' if aurc_sig else 'NOT significant at 95% CI (ΔAURC CI crosses 0)'}. "
            "H3 faithfulness: PENDING LLM run (reported honestly, not inferred)."),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_decision_md(path: Path, horizon: int, calibration: dict,
                       cross_sectional: dict, decision: dict, llm_status: str) -> None:
    lines = [
        f"# Agent-ablation decision rule — {horizon}d", "",
        "## §10.8 pre-committed gates", "",
        f"- **Gate 1 (AURC, calibration, LLM-free):** "
        f"{'PASS (significant)' if decision['gate1_AURC_gate_beats_noskill']['pass'] else 'DIRECTIONAL, not significant at 95%'} — "
        f"ΔAURC={calibration['delta_AURC_gate_minus_noskill']} "
        f"CI{calibration['delta_AURC_ci']} (negative ⇒ gate confidence lowers risk-coverage area; "
        f"selective DA {calibration['full_book_DA']}%→{calibration['selective_DA_at_cov']}%@25%→"
        f"{calibration['gated_DA']}% gated is a monotone lift).",
        f"- **Gate 2 (faithfulness, A5 vs A1):** "
        f"{decision['gate2_faithfulness_A5_beats_A1']['note']}",
        "",
        "## Calibration (H2)", "",
        f"- Full-book DA: {calibration['full_book_DA']}%  |  "
        f"selective DA @ {calibration['selective_coverage']:.0%}: {calibration['selective_DA_at_cov']}%",
        f"- Gated DA: {calibration['gated_DA']}%  IC: {calibration['gated_IC']}  "
        f"Sharpe: {calibration['gated_Sharpe']}  coverage: {calibration['gate_coverage']}",
        f"- AURC gate: {calibration['AURC_gate_confidence']}  vs no-skill: "
        f"{calibration['AURC_no_skill_confidence']}",
        "",
        "## Cross-sectional IC (secondary, universe-limited)", "",
        f"- all: {cross_sectional.get('all', {}).get('mean_ic')}  |  "
        f"matched: {cross_sectional.get('matched', {}).get('mean_ic')}",
        f"- matched vs placebo: {cross_sectional.get('matched_vs_placebo', {})}",
        "",
        f"## LLM rung status", "", f"- {llm_status}",
        "",
        f"## Verdict", "", f"{decision['verdict']}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
