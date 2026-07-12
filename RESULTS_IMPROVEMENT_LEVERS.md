# Improving DA & Sharpe — Results (no retraining)

**Data:** frozen 5D test predictions in `cache/predictions/` (n=2114), placebo-controlled.
**Cells (LSTM):** `cmtf_real` 60f71e5ce5 · `cmtf_placebo` 436e06e40e (shuffled news) ·
`market_only` bf03634e54 · `late` 5b85788ae7. All levers are pure post-hoc transforms
of saved predictions — no model was retrained.

> The one-off script that generated this analysis (`experiments/
> apply_improvement_levers.py`) has been retired as part of a workspace cleanup
> (2026-07-12) now that its findings are fully captured below and its output
> CSVs are frozen in `results/improvement_levers/`. The production replacement
> for Levers 1/2 (confidence gate + conviction sizing) now lives in
> `src/benchmark/decision_policy.py`, wired into `run_ablation_benchmark.py
> --gate` (see the "Gate wired into `fusion_comparison`" section below).

The bar: a lever only counts as *news-driven* if it helps **real news more than the
shuffled-news placebo**. Otherwise it is a generic decision-layer trick.

---

## Headline

**Yes — DA and Sharpe can be improved substantially, and the improvement is real
(news-driven, beats placebo). The catch: the signal lives almost entirely in
CMTF's high-confidence (high-|pred|) predictions. On the full book it is drowned
by a low-confidence majority that is pure noise (and actively anti-skill).**

The earlier "news-blind, λ=0" conclusion was a *full-book* statement. Conditioned on
the model's own confidence, news carries directional signal.

---

## Baseline (trade everything — the misleading view)

| cell | DA%_ens | Sharpe_ens | IC_ens |
|---|---|---|---|
| cmtf_real | 44.85 | −0.099 | **+0.063** |
| cmtf_placebo | 48.63 | 0.447 | −0.051 |
| market_only | 49.26 | 0.564 | −0.095 |
| late | 49.00 | 0.778 | −0.099 |

Classic decoupling: CMTF-real has the **only positive IC** but the **worst DA and
Sharpe** on the full book. This is the pattern we already diagnosed.

---

## Lever 1 — confidence "no-trade zone" (targets DA + Sharpe)

Trade only `|pred| ≥ q-quantile`; metrics on the traded subset.

| q (skip below) | coverage | cmtf_real DA% | cmtf_real Sharpe | placebo DA% | market DA% |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 44.85 | −0.099 | 48.63 | 49.26 |
| 0.3 | 0.70 | 52.01 | 0.131 | 51.54 | 52.42 |
| 0.5 | 0.50 | 54.78 | 0.469 | 49.95 | 49.90 |
| 0.7 | 0.30 | **57.92** | **0.763** | 50.70 | 45.31 |
| 0.8 | 0.20 | **60.15** | **1.025** | 53.79 | 41.99 |

**CMTF-real DA and Sharpe rise monotonically with confidence** (44.9%→60.2%,
−0.10→+1.03). Placebo and market-only do **not** — market-only actually *inverts*
(large-magnitude market predictions are anti-informative: DA 42%, Sharpe −1.15).

Real-minus-placebo at matched q flips decisively positive from q≥0.3 and peaks at
**q=0.7: +7.2pp DA, +0.91 Sharpe**. → passes the placebo test in the confident regime.

---

## Lever 1 — OUT-OF-SAMPLE (threshold picked on a CAL half, applied to a disjoint DEPLOY half)

| cell | deploy DA% full → gated | deploy Sharpe full → gated | deploy coverage |
|---|---|---|---|
| **cmtf_real** | 44.99 → **57.48** (+12.5) | −0.076 → **+0.435** | 0.30 |
| cmtf_placebo | 48.02 → 49.14 (+1.1) | 0.334 → 0.180 | 0.18 |
| market_only | 51.04 → 54.56 (+3.5) | 0.905 → 0.892 | 0.86 |

The dead-band **transfers out of sample** for CMTF-real (+12.5pp DA, +0.51 Sharpe on
a held-out half) and **does not** for the placebo. This is the strongest single
piece of evidence that the news signal is real, not in-sample overfitting.

---

## Lever 2 — conviction-weighted sizing (targets Sharpe)

`pos = sign(pred) · (|pred| conviction weight)` vs flat sign book.

| cell | Sharpe_flat | Sharpe_conviction | Δ |
|---|---|---|---|
| **cmtf_real** | −0.099 | **+0.341** | **+0.44** |
| cmtf_placebo | 0.447 | 0.262 | −0.19 |
| market_only | 0.564 | 0.165 | −0.40 |

Sizing by |pred| **helps real news (+0.44 Sharpe) and hurts placebo/market** — again
confirming |pred| is a genuine confidence signal *only* when the news is real.
(True vol-scaling needs per-name time alignment the concatenated test vector lacks;
conviction weighting is the order-independent analog and isolates the same effect.)

---

## Lever 4 — seed ensembling — **does NOT help CMTF-real**

| cell | dDA% (ens−seedmean) | dSharpe |
|---|---|---|
| cmtf_real | **−0.95** | **−0.121** |
| cmtf_placebo | +0.56 | +0.114 |
| market_only | +1.33 | +0.077 |
| late | +0.39 | +0.271 |

Averaging seeds shrinks predictions toward zero, **eroding the tail conviction** that
carries CMTF-real's signal. Recommendation: **do not ensemble CMTF-real by naive mean**;
if ensembling, ensemble the *decisions after* confidence gating, or use a
magnitude-preserving combiner.

---

## Diagnostic — CMTF-real by |pred| decile (the mechanism)

| decile | n | DA% | IC | signflip vs market |
|---|---|---|---|---|
| 1 (smallest \|pred\|) | 212 | 42.6 | −0.033 | **0.92** |
| 2 | 211 | 42.7 | −0.154 | 0.57 |
| 3 | 211 | 41.7 | −0.070 | 0.61 |
| 4 | 212 | 41.1 | −0.176 | 0.69 |
| 5 | 211 | 48.2 | +0.074 | 0.58 |
| … | | | | |
| 8 | 211 | 54.1 | +0.131 | 0.56 |
| 9 | 211 | 52.8 | +0.136 | 0.46 |
| 10 (largest \|pred\|) | 212 | **66.8** | **+0.248** | **0.41** |

Exactly the predicted structure:
- **Low-|pred| bins** (1–4): DA **below 50** (anti-skill), IC negative, and the model
  flips away from the market anchor up to **92%** of the time — pure noise churn.
- **High-|pred| bins** (8–10): DA 54–67%, IC positive up to **+0.25**, fewer flips.

CMTF's positive full-book IC and its news signal are **concentrated in the tail**; the
sign flips that destroy full-book DA are **concentrated near zero**.

---

## What this changes about the project's conclusion

- Previous honest statement (still true): *on the full book, a leak-free selector
  picks λ=0 — news is net-neutral-to-harmful if you trade every prediction.*
- New, stronger statement: *the full book hides a real, placebo-beating, out-of-sample
  news signal that is only expressed through prediction magnitude. The correct use of
  this model is **selective / conviction-weighted**, not trade-everything.*

Deployment recipe (no retraining): **gate at a validation-calibrated |pred| threshold
(~top 30% by confidence) and size by conviction; do not naive-seed-ensemble.**
Expected out-of-sample: **DA ≈ 57%+, Sharpe ≈ 0.4–0.8**, vs ≈45% / ≈0 on the full book.

---

## Lever 3 (retraining) — TESTED, NO WIN (2026-07-12)

**(Historical) script:** `experiments/cmtf_lever3_sweep.py` — sweeps
`(sign_penalty_weight, sharpe_surrogate_weight) ∈ {0.01,0.05,0.2} × {0.0,0.1}` on the
real 1D LSTM `anchored_fusion` cell, 3 seeds (42/123/456), real vs. shuffled-news
placebo. `sharpe_surrogate_weight` adds a differentiable `-Sharpe` term
(`_sharpe_surrogate` in `src/benchmark/hybrid_fusion.py`) on `tanh(k·pred)` soft
positions, active only post-warmup (`direction_warmup_epochs=5`), same gating as the
existing sign penalty. Retired after producing the frozen CSV below (workspace
cleanup, 2026-07-12) — the `sharpe_surrogate_weight`/`sign_penalty_weight` knobs
themselves remain live in `ablation_config.py`/`hybrid_fusion.py` if this needs
re-running.

| spw | shw | DA%_real | DA%_placebo | Sharpe_real | Sharpe_placebo | vs placebo | vs baseline |
|---|---|---|---|---|---|---|---|
| 0.01 (baseline) | 0.0 | 45.81 | 46.50 | 0.428 | 0.533 | — | — |
| 0.05 | 0.0 | 45.37 | 46.05 | −0.024 | 0.342 | ✗ | ✗ |
| 0.20 | 0.0 | 45.55 | 46.02 | −0.009 | 0.356 | ✗ | ✗ |
| 0.01 | 0.1 | 46.00 | 46.31 | 0.388 | 0.262 | ✗ (DA) | ok |
| 0.05 | 0.1 | 45.96 | 46.64 | 0.454 | 0.549 | ✗ (DA) | ok |
| 0.20 | 0.1 | 46.12 | 46.22 | 0.495 | −0.014 | ✗ (DA, by 0.1pp) | ok |

**Verdict: NO OBJECTIVE-DRIVEN WIN.** No knob setting beats its own shuffled-news
placebo on *both* DA% and Sharpe (the closest miss, `spw=0.2|shw=0.1`, wins Sharpe
decisively (0.495 vs −0.014) but loses DA by 0.1pp — noise-level, but still a miss
under the pre-registered strict bar).

**Root cause (verified, corrected 2026-07-12):** my first hypothesis — that a
CMTF λ-blend gate discards the retrained fusion head — was **wrong** and has been
retracted (see `CMTF_FUSION_FINDINGS.md` §5): `HybridFusionPredictor.predict()`
has no lambda blend at all; it always returns the raw fused prediction directly.
Instrumenting the actual training loop (per-epoch `val_ic`) found the real cause:
`direction_warmup_epochs=5` keeps the sign penalty and Sharpe surrogate inert for
the first 5 epochs, and validation-based checkpoint selection can lock onto a
best epoch that falls *inside* that window. For seed=42, epochs 0–4 produced
byte-identical `val_ic` histories (`[0.0689, 0.076, 0.0646, 0.0751, -0.0013]`)
regardless of knob — as expected, since the knob has no effect during warmup —
and the two knob settings ended up selecting the same pre-warmup checkpoint
(confirmed: their epoch 5+ histories diverge sharply, e.g. `0.0892` vs `0.0334`,
yet the final DEPLOYED metrics were still byte-identical). So for any seed whose
best validation checkpoint lands in the first 5 epochs, the knob literally cannot
change the deployed weights — no matter how strong the objective change is.

**Implication:** the sweep as run is a **confounded, not a clean, test** of the
Lever 3 hypothesis for those seeds. A fair re-test would need `direction_warmup_epochs`
lowered (e.g. to 0–2) or training extended so more seeds select a post-warmup
checkpoint before concluding the objective genuinely doesn't help — but note this
still doesn't change the observed verdict (no seed's *raw* Sharpe/DA reliably beat
its placebo even where the knob **did** take effect, e.g. seeds 123/456 in the
table above), so the practical recommendation is unchanged.

**Decision: do not adopt.** `CMTF_CORE` keeps `sign_penalty_weight=0.01`,
`sharpe_surrogate_weight=0.0` (no-op default). The knobs stay in
`ablation_config.py`/`hybrid_fusion.py` as opt-in, cache-neutral research
parameters (default value keeps existing `cell_id`s and prediction caches valid)
— useful if the sweep is ever re-run with `direction_warmup_epochs` lowered to
rule out the checkpoint-selection confound above.

**Reproduce:** frozen results are in `results/cmtf_lever3_sweep_1d.csv` and
`results/improvement_levers/` (generating scripts retired, see notes above) —
the actual deployment recipe remains the **no-retrain** one above (confidence
gate + conviction sizing), since Lever 3 did not move the full-book numbers.

---

## Gate wired into `fusion_comparison` — confirmed at full-table scale (2026-07-12)

The confidence-gate + conviction-sizing decision policy (`src/benchmark/
decision_policy.py`) is now a first-class, opt-in layer in the production
ablation runner: `run_ablation_benchmark.py --gate` calibrates it on VALIDATION
(leak-free) and applies it to every cell's frozen TEST predictions — CMTF **and**
every baseline (late/early/none, all 4 backbones) under the *identical* policy,
adding `DA%_gated` / `Sharpe_gated` / `IC_gated` / `gate_coverage` columns
alongside the existing raw metrics. No retraining; predictions are unchanged.

**Bug found and fixed while wiring this up:** `late` fusion had a pre-existing
indentation bug in `ablation_runner.py` — `preds = wrapper.predict(...)` was
nested inside `if cached_oof is None: if oof_to_cache is not None:`, so whenever
the OOF sub-cache was warm (the common case), `preds` was never reassigned and
silently stayed `None` (→ a 0-d array → a crash in `compute_all`). Fixed; also
fixed `early` fusion not exposing validation predictions to the gate.

**Full `fusion_comparison` @ 5D, 3 seeds (42/123/456), sorted by gated Sharpe:**

| cell | DA% | Sharpe | IC | DA%_gated | Sharpe_gated | IC_gated | coverage |
|---|---|---|---|---|---|---|---|
| **cmtf(lstm)** | 45.80 | 0.011 | 0.046 | **57.37** | **0.424** | 0.183 | 0.25 |
| gpt4ts::late | 45.90 | 0.221 | 0.010 | 52.35 | 0.393 | 0.038 | 0.33 |
| cmtf(cnn_lstm) | 46.10 | 0.308 | 0.071 | 55.21 | 0.291 | 0.075 | 0.32 |
| chronos::none | 47.81 | 0.451 | 0.025 | 53.47 | 0.272 | 0.089 | 0.18 |
| chronos::late | 47.79 | 0.457 | 0.025 | 53.39 | 0.266 | 0.093 | 0.21 |
| cmtf(chronos) | 44.75 | −0.152 | 0.067 | 51.84 | 0.259 | 0.139 | 0.41 |
| gpt4ts::none | 48.14 | 0.660 | 0.020 | 51.05 | 0.221 | −0.066 | 0.17 |
| cmtf(gpt4ts) | 44.52 | −0.288 | 0.071 | 50.54 | 0.121 | 0.137 | 0.60 |
| **cmtf(lstm), placebo** | 48.07 | 0.343 | −0.036 | 50.41 | 0.120 | −0.026 | 0.13 |
| cnn_lstm::late | 48.53 | 0.566 | −0.050 | 48.90 | −0.050 | −0.006 | 0.34 |
| cnn_lstm::early | 48.70 | 0.048 | −0.046 | 48.36 | −0.055 | −0.056 | 0.44 |
| lstm::late | 47.30 | 0.224 | −0.043 | 46.04 | −0.174 | −0.044 | 0.27 |
| cnn_lstm::none | 47.88 | 0.625 | −0.051 | 47.21 | −0.177 | −0.087 | 0.31 |
| lstm::none | 47.93 | 0.543 | −0.061 | 45.25 | −0.203 | −0.075 | 0.19 |
| lstm::early | 43.03 | −0.402 | −0.054 | 49.46 | −0.331 | −0.038 | 0.38 |

**This reproduces the Lever 1/2 story at full-table scale, on fresh 3-seed
predictions, independent of the earlier single-cell `apply_improvement_levers.py`
analysis:**

- **`cmtf(lstm)` is the #1 cell in the whole table under gating** — both by
  gated DA% (57.4%) and gated Sharpe (0.42) — despite being mediocre on the raw,
  trade-everything numbers (DA 45.8%, Sharpe ≈0). Exactly the "signal in the
  tail, drowned in the bulk" pattern.
- **Placebo test passes at table scale:** `cmtf(lstm)` gains **+11.6pp DA / +0.41
  Sharpe** from gating; its shuffled-news twin `cmtf(lstm), placebo` gains only
  **+2.3pp DA and *loses* Sharpe (0.34→0.12)**. Real news benefits from
  confidence-gating far more than the placebo — the gate is picking up a genuine
  news-conditioned signal, not just "trade less, more Sharpe" noise reduction.
- **News-blind baselines often get *worse* under the same gate**: `cnn_lstm::late`
  (0.57→−0.05), `cnn_lstm::none` (0.62→−0.18), `lstm::none` (0.54→−0.20). Their
  highest-magnitude predictions are anti-informative, so confidence-gating (which
  trades the *largest*-|pred| subset) actively hurts them — the mirror image of
  CMTF's genuine confidence signal.
- Other CMTF encoders (`cnn_lstm`, `chronos`) also gain from gating but less than
  `lstm`; `cmtf(gpt4ts)` needs 60% coverage to break even (its confidence signal
  is weaker). `lstm` remains the strongest CMTF backbone.

**Deployment recommendation, confirmed at table scale: ship `cmtf(lstm)` gated**
(top ~25% by confidence, validation-calibrated) as the production cell — it is
simultaneously the best forecaster (DA/IC) and the best trading cell
(Sharpe) in the entire `fusion_comparison` grid once gated, and it is the one
row whose gated lift demonstrably beats its own placebo.

**Reproduce:**
`.venv\Scripts\python.exe run_ablation_benchmark.py --table fusion_comparison --horizons 5 --seeds 42 123 456 --gate`
(writes `DA%_gated`/`Sharpe_gated`/`IC_gated`/`gate_coverage` into
`results/ablation/5d/fusion_comparison.csv`).
