"""Deep analysis: raw data → predictions → results.

Covers:
  1. Target distribution per horizon (mean, std, skew, kurtosis, up-fraction)
  2. Seed variance per cell (std of DA% across 3 seeds)
  3. Prediction distribution per model (mean, std, sign-bias, unique-value check)
  4. Prediction vs truth: residual analysis, hit/miss breakdown by return quintile
  5. Cross-seed prediction correlation (are seeds learning the same thing?)
  6. Cumulative P&L curves (sign-weighted, unit-sized positions)
  7. Error by return magnitude quintile (where does each model fail?)
  8. Aggregate summary table
"""

from __future__ import annotations
import glob
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PRED_DIR = Path("cache/predictions")
HORIZONS = [1, 5, 20]
SEEDS = [42, 123, 456]

# ── helpers ───────────────────────────────────────────────────────────────────

def load(name: str) -> np.ndarray:
    return np.load(str(PRED_DIR / name))

def truth(h: int) -> np.ndarray:
    return load(f"truth__{h}d.npy")

def da(y_true, y_pred):
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))

def sharpe(y_true, y_pred):
    pnl = np.sign(y_pred) * y_true
    return float(pnl.mean() / (pnl.std() + 1e-9))

def ic(y_true, y_pred):
    return float(stats.spearmanr(y_pred, y_true).statistic)

def mae(y_true, y_pred):
    return float(np.abs(y_true - y_pred).mean())

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# ── 1. TARGET DISTRIBUTION ────────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 1: TARGET (GROUND TRUTH) DISTRIBUTION PER HORIZON")
print("="*80)
for h in HORIZONS:
    y = truth(h)
    up = (y > 0).mean()
    print(f"\n  {h}D  n={len(y)}")
    print(f"    mean   = {y.mean()*100:+.4f}%")
    print(f"    median = {np.median(y)*100:+.4f}%")
    print(f"    std    = {y.std()*100:.4f}%")
    print(f"    skew   = {stats.skew(y):.4f}")
    print(f"    kurt   = {stats.kurtosis(y):.4f}  (excess kurtosis)")
    print(f"    up%    = {up*100:.2f}%  down%={100-up*100:.2f}%")
    print(f"    min    = {y.min()*100:+.4f}%  max={y.max()*100:+.4f}%")
    print(f"    p5     = {np.percentile(y,5)*100:+.4f}%  p95={np.percentile(y,95)*100:+.4f}%")
    print(f"    p25    = {np.percentile(y,25)*100:+.4f}%  p75={np.percentile(y,75)*100:+.4f}%")
    # autocorrelation lag-1
    ac1 = np.corrcoef(y[:-1], y[1:])[0,1]
    print(f"    autocorr(lag-1) = {ac1:.4f}")
    # Jarque-Bera normality test
    jb, jbp = stats.jarque_bera(y)
    print(f"    Jarque-Bera p   = {jbp:.4f}  ({'NOT normal' if jbp<0.05 else 'cannot reject normal'})")

# ── 2. SEED VARIANCE ──────────────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 2: SEED VARIANCE ANALYSIS (per cell, per horizon)")
print("="*80)

key_cells_20d = {
    "lstm_none":         "lstm__none__none__none",
    "lstm_hybrid_full":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "lstm_hybrid_vr0":   "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0",
    "lstm_hybrid_ts0":   "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=0__aux=1__vreg=1",
    "lstm_hybrid_k10":   "lstm__hybrid__matched__scalars__pe=1__k=10__gate=1__ts=1__aux=1__vreg=1",
    "cnn_lstm_hybrid":   "cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "cnn_lstm_early":    "cnn_lstm__early__matched__scalars",
    "rf_late":           "rf__late__matched__scalars",
}

for h in HORIZONS:
    y = truth(h)
    print(f"\n  --- {h}D ---")
    print(f"  {'Cell':<30} {'seed42':>8} {'seed123':>8} {'seed456':>8} {'std':>8} {'range':>8}")
    for label, cell in key_cells_20d.items():
        das = []
        for s in SEEDS:
            fname = f"{cell}__seed{s}__{h}d.npy"
            if (PRED_DIR / fname).exists():
                p = load(fname)
                das.append(da(y, p) * 100)
            else:
                das.append(float("nan"))
        das_arr = [x for x in das if not np.isnan(x)]
        std_val = np.std(das_arr) if len(das_arr) > 1 else float("nan")
        rng_val = (max(das_arr) - min(das_arr)) if len(das_arr) > 1 else float("nan")
        print(f"  {label:<30} {das[0]:>8.2f} {das[1]:>8.2f} {das[2]:>8.2f} {std_val:>8.2f} {rng_val:>8.2f}")

# ── 3. PREDICTION DISTRIBUTION ────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 3: PREDICTION DISTRIBUTION (averaged predictions)")
print("="*80)

avg_cells = {
    "lstm_none":        "lstm__none__none__none",
    "lstm_hybrid_full": "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "lstm_hybrid_vr0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0",
    "lstm_hybrid_ts0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=0__aux=1__vreg=1",
    "cnn_lstm_hybrid":  "cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "cnn_lstm_early":   "cnn_lstm__early__matched__scalars",
    "rf_late":          "rf__late__matched__scalars",
}

for h in HORIZONS:
    y = truth(h)
    print(f"\n  --- {h}D ---")
    hdr = f"  {'Cell':<28} {'pred_mean':>10} {'pred_std':>10} {'up_bias%':>10} {'unique':>8} {'corr_w_truth':>14}"
    print(hdr)
    for label, cell in avg_cells.items():
        fname = f"{cell}__{h}d.npy"
        if not (PRED_DIR / fname).exists():
            continue
        p = load(fname)
        n_unique = len(np.unique(np.round(p, 8)))
        corr = np.corrcoef(p, y)[0, 1]
        up_b = (p > 0).mean() * 100
        print(f"  {label:<28} {p.mean()*100:>+10.4f} {p.std()*100:>10.4f} {up_b:>10.2f} {n_unique:>8} {corr:>14.4f}")

# ── 4. RESIDUAL ANALYSIS BY RETURN QUINTILE ───────────────────────────────────

print("\n" + "="*80)
print("SECTION 4: ERROR AND HIT-RATE BY RETURN QUINTILE")
print("(Does the model fail on big moves or small moves?)")
print("="*80)

focus_20d = {
    "lstm_none":        "lstm__none__none__none__20d.npy",
    "lstm_hybrid_vr0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0__20d.npy",
    "lstm_hybrid_full": "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__20d.npy",
    "lstm_hybrid_ts0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=0__aux=1__vreg=1__20d.npy",
    "cnn_lstm_hybrid":  "cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__20d.npy",
    "cnn_lstm_early":   "cnn_lstm__early__matched__scalars__20d.npy",
    "rf_late":          "rf__late__matched__scalars__20d.npy",
}

y20 = truth(20)
quintile_labels = ["Q1 (most neg)", "Q2", "Q3 (mid)", "Q4", "Q5 (most pos)"]
q_edges = np.percentile(y20, [0, 20, 40, 60, 80, 100])

print(f"\n  Return quintile boundaries (20D):")
for i in range(5):
    print(f"    {quintile_labels[i]}: [{q_edges[i]*100:+.3f}%, {q_edges[i+1]*100:+.3f}%]  "
          f"actual_up={((y20 >= q_edges[i]) & (y20 < q_edges[i+1]) & (y20 > 0)).sum()}"
          f"/{((y20 >= q_edges[i]) & (y20 < q_edges[i+1])).sum()} days")

for label, fname in focus_20d.items():
    if not (PRED_DIR / fname).exists():
        continue
    p = load(fname)
    print(f"\n  [{label}]")
    print(f"  {'Quintile':<16} {'n':>4} {'DA%':>7} {'MAE%':>8} {'mean_err%':>10} {'pred_mean%':>11}")
    for i in range(5):
        mask = (y20 >= q_edges[i]) & (y20 < q_edges[i+1] + (1e-9 if i==4 else 0))
        yq = y20[mask]; pq = p[mask]
        if len(yq) == 0:
            continue
        dq = (np.sign(pq) == np.sign(yq)).mean() * 100
        maeq = np.abs(yq - pq).mean() * 100
        mean_err = (pq - yq).mean() * 100
        print(f"  {quintile_labels[i]:<16} {len(yq):>4} {dq:>7.1f} {maeq:>8.4f} {mean_err:>+10.4f} {pq.mean()*100:>+11.4f}")

# ── 5. CROSS-SEED PREDICTION CORRELATION ──────────────────────────────────────

print("\n" + "="*80)
print("SECTION 5: CROSS-SEED PREDICTION CORRELATION (20D)")
print("(Are the 3 seeds learning the same signal or different ones?)")
print("="*80)

corr_cells = {
    "lstm_hybrid_full": "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "lstm_hybrid_vr0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0",
    "cnn_lstm_hybrid":  "cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1",
    "cnn_lstm_early":   "cnn_lstm__early__matched__scalars",
    "rf_late":          "rf__late__matched__scalars",
}

y20 = truth(20)
for label, cell in corr_cells.items():
    preds = []
    for s in SEEDS:
        fname = f"{cell}__seed{s}__20d.npy"
        if (PRED_DIR / fname).exists():
            preds.append(load(fname))
    if len(preds) < 2:
        continue
    c01 = np.corrcoef(preds[0], preds[1])[0,1]
    c02 = np.corrcoef(preds[0], preds[2])[0,1]
    c12 = np.corrcoef(preds[1], preds[2])[0,1]
    da_each = [da(y20, p)*100 for p in preds]
    # Disagreement: fraction of days where seeds disagree on direction
    signs = np.array([np.sign(p) for p in preds])
    disagree = (signs.std(axis=0) > 0).mean() * 100
    print(f"\n  {label}")
    print(f"    DA per seed: {da_each[0]:.1f}% / {da_each[1]:.1f}% / {da_each[2]:.1f}%")
    print(f"    Pearson corr: s42-s123={c01:.3f}  s42-s456={c02:.3f}  s123-s456={c12:.3f}")
    print(f"    Mean inter-seed corr: {np.mean([c01,c02,c12]):.3f}")
    print(f"    Days seeds DISAGREE on direction: {disagree:.1f}%")

# ── 6. CUMULATIVE P&L CURVES (key cells) ─────────────────────────────────────

print("\n" + "="*80)
print("SECTION 6: CUMULATIVE P&L STATISTICS (sign-weighted unit position)")
print("="*80)

pnl_cells_20d = {
    "lstm_none":        "lstm__none__none__none__20d.npy",
    "lstm_hybrid_full": "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__20d.npy",
    "lstm_hybrid_vr0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0__20d.npy",
    "lstm_hybrid_ts0":  "lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=0__aux=1__vreg=1__20d.npy",
    "cnn_lstm_hybrid":  "cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__20d.npy",
    "cnn_lstm_early":   "cnn_lstm__early__matched__scalars__20d.npy",
    "rf_late":          "rf__late__matched__scalars__20d.npy",
}

y20 = truth(20)
print(f"\n  {'Model':<28} {'tot_ret%':>9} {'sharpe':>8} {'max_dd%':>9} {'win_streak':>11} {'loss_streak':>12}")
for label, fname in pnl_cells_20d.items():
    if not (PRED_DIR / fname).exists():
        continue
    p = load(fname)
    pnl = np.sign(p) * y20
    cum = np.cumsum(pnl)
    tot = cum[-1] * 100
    sh = pnl.mean() / (pnl.std() + 1e-9)
    # max drawdown
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    max_dd = drawdown.max() * 100
    # streak analysis
    wins = (pnl > 0).astype(int)
    max_win = max_loss = cur = 0
    for w in wins:
        if w == 1:
            cur = cur + 1 if cur > 0 else 1
            max_win = max(max_win, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_loss = max(max_loss, -cur)
    print(f"  {label:<28} {tot:>+9.3f} {sh:>8.3f} {max_dd:>9.4f} {max_win:>11} {max_loss:>12}")

# ── 7. PREDICTION BIAS: OVER/UNDER-PREDICTION BY SIGN ────────────────────────

print("\n" + "="*80)
print("SECTION 7: SIGNED PREDICTION BIAS (does model over-predict up or down?)")
print("="*80)

for h in HORIZONS:
    y = truth(h)
    up_days   = y > 0
    down_days = y <= 0
    print(f"\n  --- {h}D: truth up={up_days.sum()} days, down={down_days.sum()} days ---")
    cells_h = {
        "lstm_none":        f"lstm__none__none__none__{h}d.npy",
        "lstm_hybrid_full": f"lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__{h}d.npy",
        "lstm_hybrid_vr0":  f"lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=0__{h}d.npy",
        "cnn_lstm_hybrid":  f"cnn_lstm__hybrid__matched__scalars__pe=1__k=5__gate=1__ts=1__aux=1__vreg=1__{h}d.npy",
        "rf_late":          f"rf__late__matched__scalars__{h}d.npy",
    }
    print(f"  {'Model':<28} {'pred_up%':>9} {'hit_up%':>9} {'hit_dn%':>9} {'bias':>10}")
    for label, fname in cells_h.items():
        if not (PRED_DIR / fname).exists():
            continue
        p = load(fname)
        pred_up_pct = (p > 0).mean() * 100
        hit_up = (np.sign(p[up_days]) == 1).mean() * 100
        hit_dn = (np.sign(p[down_days]) == -1).mean() * 100
        bias = "up-biased" if pred_up_pct > 60 else ("down-biased" if pred_up_pct < 40 else "balanced")
        print(f"  {label:<28} {pred_up_pct:>9.1f} {hit_up:>9.1f} {hit_dn:>9.1f} {bias:>10}")

# ── 8. AGGREGATE RANKING TABLE ────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 8: AGGREGATE RANKING (20D averaged predictions, all fusion types)")
print("="*80)

all_20d = [f for f in os.listdir(PRED_DIR) if f.endswith("__20d.npy") and "seed" not in f and "truth" not in f]
y20 = truth(20)
rows = []
for fname in sorted(all_20d):
    p = load(fname)
    rows.append({
        "cell": fname.replace("__20d.npy",""),
        "DA%":   da(y20, p)*100,
        "Sharpe": sharpe(y20, p),
        "IC":    ic(y20, p),
        "MAE":   mae(y20, p)*100,
        "RMSE":  rmse(y20, p)*100,
        "up_bias": (p>0).mean()*100,
        "unique": len(np.unique(np.round(p,8))),
    })

df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n\nTop 5 by Sharpe (20D):")
print(df.head(5)[["cell","DA%","Sharpe","IC","MAE","RMSE"]].to_string(index=False))
print("\nTop 5 by IC (20D):")
print(df.sort_values("IC",ascending=False).head(5)[["cell","DA%","Sharpe","IC","MAE","RMSE"]].to_string(index=False))
print("\nTop 5 by DA% (20D):")
print(df.sort_values("DA%",ascending=False).head(5)[["cell","DA%","Sharpe","IC","MAE","RMSE"]].to_string(index=False))
print("\nBottom 5 by Sharpe (20D):")
print(df.tail(5)[["cell","DA%","Sharpe","IC","MAE","RMSE"]].to_string(index=False))

print("\n\n=== ANALYSIS COMPLETE ===\n")
