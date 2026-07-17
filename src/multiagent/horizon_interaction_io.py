"""Frozen HorizonInteraction artifact I/O + offline calibration.

The horizon-interaction layer scales a trade's position size up when the OTHER two
horizons corroborate the primary horizon's direction, and down when they conflict —
a symmetric adjustment, not a veto (contrast `gate_io.py`'s `GatePolicy`, which only
ever sets a hard trade/no-trade boundary).

Important calibration subtlety this module gets right: `selection_score` (and any
Sharpe-like mean/std ratio) is scale-invariant to a UNIFORM positive multiplier applied
within one bucket in isolation — multiplying every prediction in a bucket by the same
constant does not change that bucket's own DA/Sharpe/IC at all (verified empirically,
not just assumed). So per-bucket multipliers cannot be chosen by optimizing each
bucket's own score independently — there is no signal to select from. What DOES respond
to relative bucket weighting is the *pooled* book's realized Sharpe across all traded
rows: up-weighting a bucket with a better risk-adjusted return relative to the others
changes the POOLED mean/std ratio, even though it leaves each bucket's own ratio
unchanged. This module therefore grid-searches the three bucket multipliers JOINTLY
against the pooled validation Sharpe, not independently.

R1-consistent rules:
- A missing/stale artifact is the CALLER's decision how to treat (unlike GatePolicy,
  this layer is an enhancement, not a decision boundary — see horizon_interaction_agent.py).
- Calibration reads VALIDATION predictions only, aligned by (symbol, date) across all
  three horizons; TEST predictions never enter (leak-free, matching gate_io.py).
- A placebo control (randomly permuting which row's OTHER-horizon predictions pair
  with which primary row) is calibrated identically and stored alongside the real
  result, so a genuine cross-horizon effect can be distinguished from grid-search
  overfitting on a small validation book.
"""

from __future__ import annotations

import glob
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.decision_policy import GatePolicy, apply_positions

from .gate_io import StalePolicyError, core_cell_for, load_gate_policy, policy_path
from .loaders import ArtifactMissingError

HORIZON_INTERACTION_SCHEMA_VERSION = 1

DEFAULT_MULTIPLIER_GRID: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
DEFAULT_MIN_BUCKET_N = 20


@dataclass(frozen=True)
class HorizonInteractionPolicy:
    """Calibrated multiplier-by-agreement-bucket table for one primary horizon."""

    primary_horizon: int
    other_horizons: tuple[int, int]
    multiplier_by_agreement: dict[int, float]  # keys 0, 1, 2
    n_by_agreement: dict[int, int]


def interaction_policy_path(interaction_dir: str | Path, horizon: int, symbol: str = "VN") -> Path:
    return Path(interaction_dir) / f"{symbol}_{horizon}d_xh.json"


def save_interaction_policy(
    policy: HorizonInteractionPolicy, meta: dict[str, Any], path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HORIZON_INTERACTION_SCHEMA_VERSION,
        "policy": {
            "primary_horizon": policy.primary_horizon,
            "other_horizons": list(policy.other_horizons),
            "multiplier_by_agreement": {str(k): v for k, v in policy.multiplier_by_agreement.items()},
            "n_by_agreement": {str(k): v for k, v in policy.n_by_agreement.items()},
        },
        **meta,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_interaction_policy(
    path: str | Path,
    *,
    expect_cmtf_version: str | None = None,
    expect_backbone_version: str | None = None,
) -> tuple[HorizonInteractionPolicy, dict[str, Any]]:
    """Load a frozen HorizonInteractionPolicy. Raises the same error types as
    `gate_io.load_gate_policy` — the CALLER decides fatal-vs-degrade (see module
    docstring); this function itself does not soften anything."""
    p = Path(path)
    if not p.exists():
        raise ArtifactMissingError(
            f"HorizonInteractionPolicy artifact not found: {p} — run `python -m "
            f"src.multiagent calibrate-interaction --horizon <H>` first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))

    schema = payload.get("schema_version")
    if schema != HORIZON_INTERACTION_SCHEMA_VERSION:
        raise StalePolicyError(
            f"HorizonInteractionPolicy {p} has schema_version={schema}, runtime expects "
            f"{HORIZON_INTERACTION_SCHEMA_VERSION} — recalibrate."
        )
    if expect_cmtf_version is not None and payload.get("cmtf_version") != expect_cmtf_version:
        raise StalePolicyError(
            f"HorizonInteractionPolicy {p} calibrated on cmtf_version="
            f"{payload.get('cmtf_version')} but runtime loaded {expect_cmtf_version} — recalibrate."
        )
    if expect_backbone_version is not None and payload.get("backbone_version") != expect_backbone_version:
        raise StalePolicyError(
            f"HorizonInteractionPolicy {p} calibrated on backbone_version="
            f"{payload.get('backbone_version')} but runtime loaded {expect_backbone_version} — recalibrate."
        )

    pol = payload["policy"]
    policy = HorizonInteractionPolicy(
        primary_horizon=int(pol["primary_horizon"]),
        other_horizons=tuple(int(h) for h in pol["other_horizons"]),
        multiplier_by_agreement={int(k): float(v) for k, v in pol["multiplier_by_agreement"].items()},
        n_by_agreement={int(k): int(v) for k, v in pol["n_by_agreement"].items()},
    )
    return policy, payload


def _load_val_index(pred_dir: Path, config_hash: str, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (keys, pred, truth) for one horizon's validation book, ``keys`` being an
    array of ``"SYMBOL|YYYY-MM-DD"`` strings aligned 1:1 with ``pred``/``truth``."""
    sym_f = pred_dir / f"val_symbols__{horizon}d.npy"
    tim_f = pred_dir / f"val_times__{horizon}d.npy"
    tru_f = pred_dir / f"val_truth__{horizon}d.npy"
    for f in (sym_f, tim_f, tru_f):
        if not f.exists():
            raise ArtifactMissingError(
                f"Missing validation index artifact {f} — re-run the registry "
                f"(cell {core_cell_for(horizon)}) so val_symbols/val_times/val_truth are cached."
            )
    symbols = np.load(str(sym_f), allow_pickle=True)
    days = np.asarray(np.load(str(tim_f), allow_pickle=True)).astype("datetime64[D]")
    truth = np.load(str(tru_f)).astype(np.float64)

    seed_files = sorted(glob.glob(str(pred_dir / f"{config_hash}__seed*__val__{horizon}d.npy")))
    if not seed_files:
        raise ArtifactMissingError(
            f"No cached validation predictions for {config_hash} at {horizon}d in {pred_dir}."
        )
    pred = np.mean([np.load(s).astype(np.float64) for s in seed_files], axis=0)

    keys = np.array([f"{s}|{d}" for s, d in zip(symbols, days)])
    return keys, pred, truth


def _pooled_sharpe(positions: np.ndarray, truth: np.ndarray) -> float:
    """Plain (non-overlap-adjusted) mean/std Sharpe proxy of the traded book — the
    quantity that DOES respond to relative bucket weighting (see module docstring)."""
    pnl = positions * truth
    if pnl.size < 3 or np.std(pnl) < 1e-12:
        return float("-inf")
    return float(np.mean(pnl) / np.std(pnl))


def _grid_search_multipliers(
    agreement: np.ndarray,
    primary_pred_traded: np.ndarray,
    truth_traded: np.ndarray,
    primary_policy: GatePolicy,
    grid: tuple[float, ...],
    min_bucket_n: int,
) -> tuple[dict[int, float], dict[int, int], float, float]:
    """Joint grid search over (m0, m1, m2). Returns
    (multiplier_by_bucket, n_by_bucket, objective_at_solution, objective_at_baseline_111)."""
    base_positions = apply_positions(primary_pred_traded, primary_policy)
    n_by_bucket = {b: int(np.sum(agreement == b)) for b in (0, 1, 2)}
    searchable = [b for b in (0, 1, 2) if n_by_bucket[b] >= min_bucket_n]
    fixed = [b for b in (0, 1, 2) if b not in searchable]

    baseline_positions = base_positions.copy()
    baseline_obj = _pooled_sharpe(baseline_positions, truth_traded)

    if not searchable:
        return {b: 1.0 for b in (0, 1, 2)}, n_by_bucket, baseline_obj, baseline_obj

    best_obj = float("-inf")
    best_mult: dict[int, float] = {b: 1.0 for b in fixed}
    for combo in itertools.product(grid, repeat=len(searchable)):
        mult = {**{b: 1.0 for b in fixed}, **dict(zip(searchable, combo))}
        # Pre-registered structural prior, not a free 125-way search: more agreement
        # must never size DOWN relative to less agreement. With only ~dozens of rows
        # per bucket, an unconstrained joint search reliably fits noise (verified: an
        # early unconstrained run on this exact cache picked agreement=0 > agreement=2,
        # backwards from the hypothesis, off a 21-row bucket) — monotonicity is the
        # same kind of honesty constraint the confidence gate itself relies on
        # (verified monotonic skill-by-confidence decile, Phase 2 §6.1, before trusting it).
        ordered = [mult[b] for b in (0, 1, 2)]
        if any(ordered[i] > ordered[i + 1] for i in range(2)):
            continue
        positions = base_positions.copy()
        for b, m in mult.items():
            positions = np.where(agreement == b, base_positions * m, positions)
        obj = _pooled_sharpe(positions, truth_traded)
        if obj > best_obj:
            best_obj = obj
            best_mult = mult

    return best_mult, n_by_bucket, best_obj, baseline_obj


def calibrate_interaction_from_cache(
    *,
    pred_dir: str | Path,
    gate_dir: str | Path,
    interaction_dir: str | Path,
    primary_horizon: int,
    other_horizons: tuple[int, int],
    cmtf_version: str,
    backbone_version: str,
    multiplier_grid: tuple[float, ...] = DEFAULT_MULTIPLIER_GRID,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
    placebo_seed: int = 0,
) -> tuple[HorizonInteractionPolicy, dict[str, Any], Path]:
    """Freeze a `HorizonInteractionPolicy` from cached VALIDATION predictions, aligned
    by (symbol, date) across all three horizons. Never touches TEST (leak-free)."""
    from src.benchmark.ablation_registry import get_cell
    from src.benchmark.ablation_runner import _config_hash

    pred_dir = Path(pred_dir)
    # Each horizon uses ITS OWN validated champion cell (core_cell_for) — cell 13 for
    # 5D/20D, cell 0 for 1D — not a single shared cell across the whole calibration,
    # since the two "other" horizons consulted for agreement may be a different
    # champion cell than the primary horizon being calibrated.
    primary_config_hash = _config_hash(get_cell(core_cell_for(primary_horizon)))

    primary_keys, primary_pred, primary_truth = _load_val_index(pred_dir, primary_config_hash, primary_horizon)
    other_data = [
        _load_val_index(pred_dir, _config_hash(get_cell(core_cell_for(h))), h) for h in other_horizons
    ]

    primary_policy, _ = load_gate_policy(
        policy_path(gate_dir, primary_horizon, symbol="VN"),
        expect_cmtf_version=cmtf_version, expect_backbone_version=backbone_version,
    )

    # Inner-join on (symbol,date) across primary + both other horizons.
    common = set(primary_keys)
    other_maps = []
    for keys, pred, _truth in other_data:
        m = dict(zip(keys, pred))
        other_maps.append(m)
        common &= set(keys)
    common = sorted(common)

    primary_map = dict(zip(primary_keys, primary_pred))
    truth_map = dict(zip(primary_keys, primary_truth))
    p_pred = np.array([primary_map[k] for k in common])
    p_truth = np.array([truth_map[k] for k in common])
    o_preds = [np.array([m[k] for k in common]) for m in other_maps]

    traded_mask = np.abs(p_pred) >= primary_policy.tau
    p_pred_t, p_truth_t = p_pred[traded_mask], p_truth[traded_mask]
    o_preds_t = [o[traded_mask] for o in o_preds]

    def _agreement(o_list: list[np.ndarray]) -> np.ndarray:
        primary_sign = np.sign(p_pred_t)
        return sum((np.sign(o) == primary_sign).astype(int) for o in o_list)

    agreement = _agreement(o_preds_t)
    mult, n_by_bucket, real_obj, baseline_obj = _grid_search_multipliers(
        agreement, p_pred_t, p_truth_t, primary_policy, multiplier_grid, min_bucket_n,
    )

    # Placebo: permute which row's OTHER-horizon predictions are consulted, breaking
    # the true (symbol,date) correspondence while preserving each variable's own
    # marginal distribution. Averaged over several independent permutations (not one)
    # so the placebo baseline itself isn't just a single noisy draw — a lone
    # permutation can look artificially flat OR artificially strong by chance.
    rng = np.random.default_rng(placebo_seed)
    n_placebo_draws = 10
    placebo_objs, placebo_mults = [], []
    for _ in range(n_placebo_draws):
        perm = rng.permutation(len(p_pred_t))
        o_preds_placebo = [o[perm] for o in o_preds_t]
        agreement_placebo = _agreement(o_preds_placebo)
        pm, _, p_obj, placebo_baseline_obj = _grid_search_multipliers(
            agreement_placebo, p_pred_t, p_truth_t, primary_policy, multiplier_grid, min_bucket_n,
        )
        placebo_objs.append(p_obj)
        placebo_mults.append(pm)
    placebo_obj = float(np.mean(placebo_objs))
    placebo_obj_std = float(np.std(placebo_objs))
    placebo_mult = placebo_mults[0]  # representative single draw, for inspection only

    policy = HorizonInteractionPolicy(
        primary_horizon=primary_horizon, other_horizons=other_horizons,
        multiplier_by_agreement=mult, n_by_agreement=n_by_bucket,
    )

    meta = {
        "symbol": "VN",
        "cell_id": core_cell_for(primary_horizon),
        "config_hash": primary_config_hash,
        "cmtf_version": cmtf_version,
        "backbone_version": backbone_version,
        "calibrated_on": "validation",
        "n_common_val_rows": len(common),
        "n_traded": int(traded_mask.sum()),
        "multiplier_grid": list(multiplier_grid),
        "min_bucket_n": min_bucket_n,
        "pooled_sharpe_real_optimized": round(float(real_obj), 4),
        "pooled_sharpe_baseline_111": round(float(baseline_obj), 4),
        "real_lift_over_baseline": round(float(real_obj - baseline_obj), 4),
        "placebo_multiplier_by_agreement_example": {str(k): v for k, v in placebo_mult.items()},
        "pooled_sharpe_placebo_optimized_mean": round(placebo_obj, 4),
        "pooled_sharpe_placebo_optimized_std": round(placebo_obj_std, 4),
        "pooled_sharpe_placebo_baseline_111": round(float(placebo_baseline_obj), 4),
        "placebo_lift_over_baseline_mean": round(placebo_obj - float(placebo_baseline_obj), 4),
        "n_placebo_draws": n_placebo_draws,
        "placebo_seed": placebo_seed,
        "real_lift_beats_placebo_lift": bool(
            (real_obj - baseline_obj) > (placebo_obj - float(placebo_baseline_obj))
        ),
        "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = interaction_policy_path(interaction_dir, primary_horizon, symbol="VN")
    save_interaction_policy(policy, meta, out_path)
    return policy, meta, out_path
