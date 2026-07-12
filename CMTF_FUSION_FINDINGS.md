# CMTF Fusion — Diagnosis, Fixes & Evidence

Senior-engineer investigation of the Cross-Modal Temporal Fusion (CMTF /
`HybridFusionPredictor`) model, run end-to-end on **real** VN market + news data.
Every claim is backed by a **fair, leak-free, placebo-controlled** protocol: for
each `(horizon, seed)` ONE LSTM market encoder is trained once and deep-copied
into every fusion variant, so all deltas are attributable to the *fusion design*,
not to a different backbone. The market-only anchor is that same encoder.

Metrics (priority order): **DA** directional accuracy → **Sharpe** (sign-based) →
**IC** Spearman rank corr → **RMSE**. Δ is vs. the shared market-only LSTM.

> **Effective sample size (ESS) matters.** Targets are overlapping forward
> returns, so ESS ≈ n/h. The **5D** horizon (ESS ≈ n/5) gives far more trustworthy
> DA/Sharpe/IC estimates than **20D** (ESS ≈ n/20 ≈ 38 on validation). We treat 5D
> as the **primary** horizon and distrust single-seed 20D point estimates.

> **⚠ CORRECTION (2026-07-12):** §0–§5 below describe an experimental
> `select_additive_lambda` blend (`final = anchor + λ·(fusion_pred − anchor)`,
> validation-gated by a block-stability guard) as if it were the live CMTF
> deployment path. It never was — that function was **dead code**, called only
> from its own tests, and has now been **removed** (see `fusion_selection.py`).
> `HybridFusionPredictor.predict()` has always unconditionally returned the raw
> `fusion_pred + news_residual` value with no post-hoc blend or gate ("no
> lambda guard" per its own docstring). Read §0.2/§0.4, §1's "Fix", §4's closing
> paragraph, and all of §5 as a **historical account of an offline analysis**
> (run over cached predictions via a one-off probe script, since retired —
> frozen output in `results/fusion_probe_1d.csv`), not as a description of what
> the shipped model does. The real, still-true takeaway from that analysis is
> §0.3: a moderate news blend genuinely beats market-only and survives a
> placebo — CMTF's own `anchored_fusion` head is the production attempt to
> capture that lift directly, without any gate.

---

## 0. TL;DR

1. **Root-cause bug fixed:** the additive-λ fusion selector chose news weights on
   a single overlap-inflated validation point estimate, so at 20D it admitted
   overfit news that *reversed* on test (val +0.11 skill → test −8 DA). Fixed with
   a **block-stability guard** (`src/benchmark/fusion_selection.py`): a positive λ
   is admitted only if it improves DA-skill in *every* contiguous validation block.
   Shared by late fusion **and** CMTF.
2. **[SUPERSEDED, see correction above]** An offline analysis explored a
   validation-selected λ blend `final = anchor + λ·(fusion_pred + news_residual
   − anchor)`. This was never implemented as a live deployment gate. The
   production `output_mode="anchored_fusion"` trains the fusion head to predict
   the full news-using target and **deploys that prediction directly** — no λ,
   no anchor blend, no guard.
3. **Genuine news value is real but modest.** In the (offline, cached-
   prediction) placebo-controlled blend sweep, a moderate weight (w≈0.35)
   **dominates market-only on DA/Sharpe/IC/RMSE at both 5D and 20D**, and the
   lift **survives a shuffled-news placebo** → it is genuine news, not an
   ensemble artifact. This motivated shipping CMTF's news branch at all.
4. **[SUPERSEDED, see correction above]** There is no λ-gate in the shipped
   model, so there is no "picks λ=0 / falls back to market-only" behaviour to
   report. The live model always uses its full fused prediction; whether that
   is good or bad is measured directly by `fusion_comparison`/`component_ablation`
   (see `results/ablation/`), not by a gate's admit/reject decision.

---

## 1. The root-cause bug and the fix

The additive weight λ for `market + λ·news` was selected on validation by a single
scalar score over the whole set. At long horizons the validation targets overlap
heavily (a 20D return shares 19/20 of its window with its neighbour), collapsing
ESS to ~38. A positive λ could then post a large **point-estimate** gain that is
pure overfitting and **reverses on test**:

| cell | val ΔDA-skill (old score) | test ΔDA (old λ>0) |
|---|---|---|
| seed42 / 20D late fusion | **+0.112** (looks great) | **−8.06** (disaster) |

No scalar threshold separates this from a genuine signal, because the *false* 20D
gain (0.11) is larger than the *genuine* 5D gain (~0.02).

**[SUPERSEDED]** An offline `fusion_selection.select_additive_lambda` (block-
stability guard: split validation into `N_BLOCKS=5` contiguous blocks; admit λ>0
only if it improves DA-skill, without a material Sharpe regression, in at least
`MIN_BLOCK_FRAC` of them) was prototyped and unit-tested to study this failure
mode offline. It was **never called from any production path** (not
`HybridFusionPredictor`, not `LateFusionWrapper`) and has since been **removed**
as dead code, along with its tests. The finding it produced — a genuine,
above-noise directional signal must hold *across* the validation span, not just
as a single overlap-inflated point estimate — remains valid methodological
guidance for anyone adding a future gate, it just isn't wired into anything today.

---

## 2. Multi-seed design sweep (3 seeds × {5D, 20D}, placebo-controlled)

14 CMTF formulations vs the shared anchor. Mean over 3 seeds. Key rows:

### 5D — PRIMARY (high ESS). Anchor: DA 45.8, Sharpe 0.251, IC −0.069
| variant | ΔDA | ΔSharpe | ΔIC | note |
|---|---|---|---|---|
| `encoder_residual` (old core) | +0.00 | +0.00 | +0.00 | **λ=0 → market-only (news-blind)** |
| `two_stage` | +1.93 | +0.25 | +0.05 | gain mostly *encoder retraining*, not news (see §3) |
| `fusion_plus_news` (gate 0.3) | −0.70 | −0.11 | +0.12 | news buys IC, **loses DA anchor** |

### 20D — low ESS. Anchor: DA 57.5, Sharpe 0.83, IC 0.187
- With all 3 seeds, the seed-42 "win" for `fusion_plus_news` (single-seed DA +3.6)
  **collapses to DA −0.59** — a textbook low-ESS mirage. This *validates* the guard.
- `encoder_residual` again reverts to **λ=0** (news-blind) across all seeds.

**Takeaways:** (a) `encoder_residual` is downside-safe but news-blind on this data;
(b) plain `fusion_plus_news` uses news but sacrifices the DA anchor; (c) neither
dominates → motivates `anchored_fusion`.

---

## 3. Placebo control — separating news from re-modeling

news-effect = REAL lift − PLACEBO(shuffled-news) lift. A genuine effect needs the
placebo lift ≈ 0.

| config | horizon | REAL ΔDA | PLACEBO ΔDA | **genuine news ΔDA** |
|---|---|---|---|---|
| `two_stage` | 5D | +1.93 | **+1.37** | +0.56 (mostly *not* news!) |
| `fusion_plus_news` | 20D | −0.59 | −1.93 | +1.34 (genuine, but abs DA still <mkt) |
| `fpn + news_gate=1.0` | 20D | −0.28 | −2.21 | +1.93 (genuine) |

`two_stage`'s headline 5D DA gain is **mostly encoder fine-tuning** (placebo alone
recovers +1.37 of +1.93) — declaring it a "news win" would be wrong. This is why
we keep `use_two_stage=False`.

---

## 4. The (offline) motivating result — market+news blend beats market-only

Training-free blend of the cached anchor and the news-gated `fusion_plus_news`
prediction, `final = (1−w)·anchor + w·fpn`, with the placebo twin blended
identically — computed offline over cached predictions, not a live model path.
`news_gate_alpha=1.0` (full gate) is **load-bearing** — the softened 0.3 gate
never achieves genuine dominance.

| w | 5D REAL (dom?) | 5D PLACEBO | 20D REAL (dom?) | genuine? |
|---|---|---|---|---|
| 0.35 | DA +2.54, Sh +0.32, IC +0.11, RMSE↓ ✅ | DA +0.35, IC +0.03 | DA +0.17, Sh +0.03, IC +0.13, RMSE↓ ✅ | **YES (REAL≫PLACEBO)** |

A single moderate w≈0.35 **dominates market-only on all four metrics at both
horizons**, and the real blend beats its placebo by ~+2.2 DA / +0.08 IC at 5D →
the extra lift is genuine news, not variance-reduction ensembling.

This offline sweep (over cached predictions, not a live model path) motivated
shipping CMTF's own `output_mode="anchored_fusion"` in `hybrid_fusion.py`: train
the fusion head to predict the full news-using target directly
(`final_pred = fusion_pred + news_residual`) and **deploy that value with no
post-hoc blend, gate, or lambda** — `predict()` always returns it as-is. There is
no `w` knob at inference; whatever DA/Sharpe/IC the trained head achieves on the
test set is exactly what gets reported in `fusion_comparison`/`component_ablation`.

---

## 5. [REMOVED] Stale λ=0 deployment caveat

This section previously claimed the deployed `anchored_fusion` model "falls back
to market-only" because a validation-selected λ picks 0. That mechanism does not
exist in the shipped model (see the correction banner at the top of this file) —
there is nothing to fall back from. Whatever gap exists between the offline
blend-sweep numbers in §4 and CMTF's actual `fusion_comparison` numbers is a
question of how well the *trained fused prediction* matches the *offline blend*,
not a gate admitting or rejecting news.

**The real, verified confound for retraining attempts** (e.g. `sign_penalty_weight`
/ `sharpe_surrogate_weight` sweeps) is different and unrelated to any lambda gate:
`direction_warmup_epochs=5` means the sign penalty / Sharpe surrogate are inert
for the first 5 training epochs. If validation-based early stopping selects its
best checkpoint from within that warmup window (confirmed by instrumentation:
identical per-epoch `val_ic` for epochs 0–4 regardless of the knob, for a seed
where the deployed metrics also came out byte-identical across knobs), the knob
never gets a chance to shape the deployed weights. See `RESULTS_IMPROVEMENT_LEVERS.md`
"Lever 3" for the full, corrected root-cause writeup and reproduction steps.

---

## 6. Changes made to the codebase

- **`src/benchmark/fusion_selection.py`** — DA/Sharpe-first blended
  `selection_score` used for CMTF epoch selection and decision-gate calibration.
  (The `select_additive_lambda` block-stability guard that used to live here was
  dead code — never called from any production path — and was removed 2026-07-12.)
- **`src/benchmark/hybrid_fusion.py`** — `output_mode="anchored_fusion"` (the
  class default) trains the fusion head on the full news-using target and
  deploys it directly, no blend. `market_plus_fusion` still warns (harmful);
  `select_by_ic`/`fusion_weight_decay` remain research-only knobs.
- **`src/benchmark/ablation_config.py`** — `CMTF_CORE` → `output_mode=
  "anchored_fusion"`, `news_gate_alpha=1.0`, `use_two_stage=False`. Component
  ablation now covers all four output modes.
- **Tests** — `tests/test_fusion_selection.py` covers `selection_score`/
  `da_fraction`/`rank_ic`; `tests/test_ablation_config.py` covers the canonical
  design.

---

## 7. Recommendations for the full benchmark

1. **Primary CMTF cell:** `anchored_fusion` + `news_gate_alpha=1.0`, single-stage.
   No gate — whatever the trained head does on test is what gets reported.
2. Report **5D as the primary horizon** (high ESS); treat single-seed 20D numbers
   as low-confidence and always average ≥3 seeds.
3. Keep `fusion_plus_news` and `encoder_residual` as documented ablations; never
   ship `market_plus_fusion`.
4. Do **not** claim a news win from a point-estimate — require it to (a) beat a
   shuffled-news placebo and (b) hold across seeds.
5. If a future retraining attempt (stronger directional objective, new loss term,
   etc.) shows no effect across seeds, check whether `direction_warmup_epochs`
   is masking it via early-checkpoint selection before concluding the objective
   doesn't work (see §5).
