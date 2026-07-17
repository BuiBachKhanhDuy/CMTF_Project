"""Per-date cross-sectional IC — the primary rank metric (plan §10.3).

The pooled IC in ``decision_policy.evaluate_policy`` mixes cross-sectional and
time-series variation. The ranking claim is *cross-sectional*: on each date,
rank the names traded that day and correlate with realised returns. This module
computes that metric from the frozen prediction cache + the per-row ``times``
index saved next to the predictions (``test_times__{H}d.npy``).

Cell → config_hash mapping uses the registry's own hasher, so there is no
ambiguity about which cached vector belongs to which registry cell.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from scipy import stats

PRED_DIR = Path("cache/predictions")


def _load_ensemble(config_hash: str, horizon: int) -> np.ndarray | None:
    """Ensemble prediction = mean over the per-seed vectors (the atomic outputs).

    We deliberately average the ``__seed*__`` files rather than reading a cached
    ``__ensemble__`` file, so a stale ensemble sidecar can never be silently mixed
    with freshly-written seeds (R1).
    """
    seeds = sorted(glob.glob(str(PRED_DIR / f"{config_hash}__seed*__{horizon}d.npy")))
    # Exclude the validation sidecars (``__seed*__val__{H}d.npy``), which this glob
    # also matches but have a different length than the test predictions.
    seeds = [s for s in seeds if "__val__" not in Path(s).name]
    if not seeds:
        return None
    return np.mean([np.load(s).astype(np.float64) for s in seeds], axis=0)


def per_date_cross_sectional_ic(
    pred: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    n_min: int = 3,
) -> dict:
    """Mean per-date cross-sectional Spearman IC + information ratio.

    For each unique date, compute Spearman(pred, truth) across the names present
    that day (requires >= ``n_min`` names, else the date is skipped and counted).
    Returns mean IC, IR = mean/std over dates, the per-date IC array, and the
    number of dates used vs skipped (skips are surfaced, never hidden — R1).
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    days = np.asarray(times).astype("datetime64[D]")
    if not (len(pred) == len(truth) == len(days)):
        raise ValueError(f"length mismatch: pred={len(pred)} truth={len(truth)} times={len(days)}")

    ics: list[float] = []
    skipped = 0
    for d in np.unique(days):
        m = days == d
        if m.sum() < n_min:
            skipped += 1
            continue
        p, t = pred[m], truth[m]
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            skipped += 1
            continue
        c, _ = stats.spearmanr(p, t)
        if np.isfinite(c):
            ics.append(float(c))
    ics_arr = np.asarray(ics, dtype=np.float64)
    mean_ic = float(ics_arr.mean()) if ics_arr.size else float("nan")
    std_ic = float(ics_arr.std(ddof=1)) if ics_arr.size > 1 else float("nan")
    ir = mean_ic / std_ic if std_ic and std_ic > 1e-12 else float("nan")
    return {
        "mean_ic": mean_ic,
        "ir": ir,
        "n_dates_used": int(ics_arr.size),
        "n_dates_skipped": int(skipped),
        "per_date_ic": ics_arr,
    }


def paired_bootstrap_over_dates(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    n_boot: int = 5000,
    n_min: int = 3,
    seed: int = 0,
) -> dict:
    """95% CI on Δ(mean per-date IC) = A − B, resampling *dates* (paired)."""
    days = np.asarray(times).astype("datetime64[D]")
    uniq = np.unique(days)
    # Precompute per-date IC for both series (same date set, paired).
    def _by_date(pred):
        pred = np.asarray(pred, dtype=np.float64).ravel()
        t = np.asarray(truth, dtype=np.float64).ravel()
        out = {}
        for d in uniq:
            m = days == d
            if m.sum() < n_min or np.std(pred[m]) < 1e-12 or np.std(t[m]) < 1e-12:
                continue
            c, _ = stats.spearmanr(pred[m], t[m])
            if np.isfinite(c):
                out[d] = c
        return out

    a, b = _by_date(pred_a), _by_date(pred_b)
    common = [d for d in uniq if d in a and d in b]
    da = np.array([a[d] for d in common])
    db = np.array([b[d] for d in common])
    point = float(da.mean() - db.mean())
    rng = np.random.default_rng(seed)
    n = len(common)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = da[idx].mean() - db[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_dates": n, "significant": bool(lo * hi > 0)}


def _main() -> None:
    from src.benchmark.ablation_registry import get_cell
    from src.benchmark.ablation_runner import _config_hash

    horizon = 5
    truth = np.load(str(PRED_DIR / f"truth__{horizon}d.npy")).astype(np.float64)
    times_file = PRED_DIR / f"test_times__{horizon}d.npy"
    if not times_file.exists():
        raise SystemExit(f"missing {times_file} — re-run the registry so the date index is saved")
    times = np.load(str(times_file), allow_pickle=True)

    cells = {"0": "core (all)", "0p": "core placebo", "8": "matched", "8p": "matched placebo"}
    hashes = {cid: _config_hash(get_cell(cid)) for cid in cells}

    print(f"n={len(truth)}  dates={len(np.unique(np.asarray(times).astype('datetime64[D]')))}\n")
    print(f"{'cell':<16}{'hash':<12}{'mean_ic':>9}{'IR':>7}{'dates':>7}{'skip':>6}")
    loaded = {}
    for cid, label in cells.items():
        p = _load_ensemble(hashes[cid], horizon)
        if p is None:
            print(f"{label:<16}{hashes[cid]:<12}   (no cached preds)")
            continue
        loaded[cid] = p
        r = per_date_cross_sectional_ic(p, truth, times)
        print(f"{label:<16}{hashes[cid]:<12}{r['mean_ic']:>9.4f}{r['ir']:>7.2f}"
              f"{r['n_dates_used']:>7}{r['n_dates_skipped']:>6}")

    print("\nPaired bootstrap over dates (delta mean per-date IC):")
    def cmp(a, b, name):
        if a in loaded and b in loaded:
            r = paired_bootstrap_over_dates(loaded[a], loaded[b], truth, times)
            tag = "SIGNIFICANT" if r["significant"] else "ns"
            print(f"  {name:<26} {r['delta']:+.4f}  95%CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]  {tag}")
    cmp("8", "0", "matched - all")
    cmp("0", "0p", "all real - placebo")
    cmp("8", "8p", "matched real - placebo")


if __name__ == "__main__":
    _main()
