# Phase 1 — Model Training and Results

## Abstract

This document covers the second half of Phase 1: the market-only baseline models trained on the
dataset described in `01_market_data_overview.md`, the metrics used to evaluate them, their
architectures, and their training procedures. **The results table and charts in Section 6 are
intentionally left as a pending placeholder.** A repository-wide history check (see Section 6 for
detail) found no committed benchmark output for the current 7-symbol/2020–2026 configuration —
the only prior numbers are a 2-symbol, pre-correctness-fix run already flagged
`docs/reference/phase2_benchmark_report_HISTORICAL.md`, "do not cite." Rather than reuse those or
fabricate numbers, this document specifies exactly what will be reported and how to generate it,
so the results section can be filled in from a real run without changing the document's
structure.

## 1. Scope

This document covers the **market-only baseline roster** — the models that consume only the 19
canonical market features (Section 4.2–4.3 of `01_market_data_overview.md`), with no news/sentiment
input. Fusion variants (CMTF, early/late fusion) belong to Phase 2 and are documented separately.
As in the data document, every model and metric here is evaluated across the same **three
horizons — 1D, 5D, 20D** — and the same **7 symbols** (VCB, BID, CTG, TCB, MBB, ACB, VPB).

## 2. Evaluation Metrics — Definitions and Interpretation

All metrics are implemented in `src/benchmark/metrics.py`. A recurring design point in this module
(documented in its own header as a "fix pass") is that directional metrics used to derive their
active/inactive dead-zone threshold from `y_true`'s scale and apply it to `y_pred`; this
systematically penalized models whose predictions were smaller in magnitude than actual returns
even when correctly signed. Every directional metric below now derives its threshold from **each
series' own distribution** unless a fixed `eps` is supplied.

### 2.1 Regression error metrics

- **MAE** — $\frac{1}{n}\sum_i |y_i - \hat{y}_i|$. Mean absolute log-return error, in the same
  units as the forward return target.
- **RMSE** — $\sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$. Penalizes large errors more than MAE;
  the primary regression metric used for model ranking.

### 2.2 Directional metrics

A dead-zone threshold `thr` maps each series to $\{-1, 0, +1\}$ (`_signed_labels`): values with
$|v| \le \text{thr}$ are labeled "no direction" (0). By default `thr` is adaptive — half of the
series' own 20th-percentile absolute magnitude, floored at $10^{-6}$ — so both a model's
predictions and the true returns get a threshold sized to their own scale.

- **DA% (Directional Accuracy)** — percentage of "active" true-return days (true label ≠ 0) where
  the predicted sign matches the true sign. Computed only over the active subset, not all samples.
- **base_rate_DA%** — the accuracy of always predicting the majority sign (whichever of up/down is
  more frequent in the active subset). This is the naive baseline DA% must beat to show any skill.
- **DA_skill% = DA% − base_rate_DA%** — the actual directional edge over always guessing the
  majority class. This is the number that should be compared across models and horizons, not raw
  DA% alone, since base rates shift with horizon and symbol.
- **Precision / Recall / F1 (direction)** — computed symmetrically over both classes ({-1, +1}),
  then averaged, rather than being biased toward "predicting up." For class $c \in \{-1,+1\}$:
  $\text{Prec}_c = TP_c / (TP_c+FP_c)$, $\text{Rec}_c = TP_c/(TP_c+FN_c)$,
  $F1_c = 2\cdot\text{Prec}_c\cdot\text{Rec}_c/(\text{Prec}_c+\text{Rec}_c)$; the reported value is
  the mean over $c$.
- **DA_ind% / Prec_ind / Rec_ind / F1_ind** — the same directional metrics computed on
  **non-overlapping, phase-averaged subsamples** (stride = horizon). A horizon-$h$ forward return
  target overlaps its neighbors by $h{-}1$ steps, which inflates apparent sample size and can
  overstate significance; these `_ind` variants average the metric over $h$ phase offsets of
  strided (non-overlapping) samples, giving an effective-sample-size-corrected view. For $h=1$
  they equal the plain directional metrics.
- **ESS (Effective Sample Size)** — $n$ for $h=1$; $\lfloor n/h \rfloor$ for $h>1$, i.e., the
  sample count after removing horizon-induced overlap. Reported alongside every horizon so that
  apparent significance isn't read off the raw (overlapping) row count.

### 2.3 Trading / rank metrics

- **Sharpe** — annualized Sharpe ratio of a sign-based long/short strategy: go long when
  $\hat{y} > \text{thr}$, short when $\hat{y} < -\text{thr}$, flat otherwise (threshold sized from
  $\hat{y}$'s own distribution), realize the true return $y$ on that position. Annualization factor
  is $\sqrt{252}$ for $h=1$; for $h>1$, Sharpe is computed separately on each of the $h$
  non-overlapping phase offsets (annualized by $\sqrt{252/h}$) and averaged, for the same
  overlap-correction reason as `DA_ind%`.
- **IC (Information Coefficient)** — Spearman rank correlation between $\hat{y}$ and $y$. Captures
  whether the model's *ranking* of predicted returns tracks the true ranking, independent of
  calibration/scale — the standard cross-sectional signal-quality metric in quant finance.

### 2.4 Composite diagnostics

- **ModalDisagreement** — fraction of sign disagreements between a candidate model's predictions
  and a designated "anchor" prediction (e.g., a market-only baseline used as the reference when
  scoring a fusion variant in later phases). Not meaningful for a baseline compared against itself;
  relevant once Phase 2 fusion models are scored against their market-only counterpart.
- **TemporalLag** — best-lag phase-shifted correlation penalty: searches lags in
  $[-\min(h,10), +\min(h,10)]$ for the lag at which $\hat{y}$ correlates most strongly with $y$,
  and returns $|\text{best\_lag}| / \text{max\_lag}$. A value near 0 means the model's predictions
  are already time-aligned with the truth; a value near 1 means the model's best correlation with
  reality is at a large lag — i.e., it's tracking a delayed/lagged version of the signal rather
  than predicting ahead of it.
- **CompositeScore** — a single lower-is-better diagnostic combining normalized RMSE/MAE (divided
  by the true-return standard deviation), DA penalty ($1 - DA\%/100$), F1 penalty, IC penalty
  (mapped from $[-1,1]$ to $[1,0]$), TemporalLag, and ModalDisagreement, weighted
  $0.28/0.18/0.16/0.14/0.12/0.07/0.05$ respectively. The module docstring is explicit that this is
  **diagnostic only** and should not be used as the sole model-selection criterion — it exists to
  give one number that roughly tracks "is this model good across several axes at once," not a
  formally justified utility function.
- **flag_degenerate** — not a metric but a collapse detector: flags a run as degenerate if
  predictions have ~zero variance, IC is exactly 0 (constant output → Spearman `nan` → 0), recall
  pins to one class ($>98\%$ or $<2\%$), or DA% sits within 0.25 points of the base rate with
  F1 < 0.40. Used to catch models that "cheat" the composite score by predicting a near-constant
  value rather than genuinely forecasting direction.

### 2.5 Risk metrics

- **max_drawdown** — largest peak-to-trough decline of the cumulative strategy return series.
- **calmar_ratio** — annualized return ÷ |max drawdown|; a return-per-unit-of-worst-drawdown risk
  metric, complementary to Sharpe (which normalizes by volatility, not tail loss).

### 2.6 Statistical comparison tools

- **diebold_mariano_test** — tests whether two models' loss series (squared or absolute error) have
  equal predictive accuracy, using a Newey–West-style long-run variance estimate with bandwidth
  equal to the horizon (to account for the same overlap-induced autocorrelation as the `_ind`
  metrics). Returns a DM statistic and p-value.
- **paired_bootstrap_da** — bootstraps the *difference* in DA% between two models (10,000
  resamples by default) to get a confidence interval and p-value on `delta_da`, rather than
  comparing two DA% point estimates directly.

These two are the tools intended for claiming "Model A is significantly better than Model B" —
comparing point-estimate metrics across a table is not by itself a significance claim.

## 3. Models Evaluated (Phase 1 Baseline Roster)

The baseline experiment set actually wired into `run_model_benchmark.py` (`experiments = [...]`,
excluding fusion/CMTF variants, which belong to Phase 2) is:

| Model | Family | Input | News? |
|---|---|---|---|
| Chronos Zero-Shot | Pretrained foundation model, no training | Raw close-price sequence | No |
| LSTM Baseline | Recurrent, trained from scratch | Market feature sequence | No |
| LSTM Hybrid | Recurrent + tabular summary | Sequence + engineered summary | No |
| Random Forest Baseline | Tree ensemble | Engineered summary features | No |
| Linear Summary Baseline | Ridge regression | Engineered summary features | No |
| MLP Summary Baseline | Feedforward network | Engineered summary features | No |
| CNN-LSTM | Dilated causal conv + recurrent | Market feature sequence | No |
| CNN-LSTM Hybrid | CNN-LSTM + tabular summary | Sequence + engineered summary | No |
| GPT4TS Baseline | Pretrained GPT-2 backbone, partially fine-tuned | Patched market sequence | No |
| GPT4TS Hybrid | GPT4TS + tabular summary | Sequence + engineered summary | No |

This is a materially larger and different roster than the 4-model set (Chronos Zero-Shot, LSTM,
Random Forest, "Chronos Fine-Tuned/LoRA") described in older repository documents
(`README.md`'s original Key Results table, `docs/reference/phase2_benchmark_report_HISTORICAL.md`).
Two concrete changes worth flagging explicitly rather than silently carrying forward:

1. **"Chronos Fine-Tuned (LoRA)" no longer describes the current code.** `peft`/LoRA is not a
   dependency in `requirements.txt`, and no `LoraConfig`/adapter-injection code exists in
   `src/benchmark/`. The current Chronos-adaptation class is `ChronosAdapter`
   (`src/benchmark/chronos_encoder.py`): a frozen Chronos T5 encoder, a trainable (or optionally
   frozen, for memoization) linear `input_projection` into the encoder's embedding space, and a
   trainable regression head — an "apple-to-apples" linear-probe/adapter design, not LoRA. It is
   also **not currently in `run_model_benchmark.py`'s `experiments` list**, so it is documented
   here (Section 4.1) for completeness but is not part of the pending results run unless
   re-enabled.
2. **The roster has grown**: `Linear Summary Baseline`, `MLP Summary Baseline`, `CNN-LSTM`
   (+Hybrid), and `GPT4TS` (+Hybrid) are new additions not reflected in the older docs at all.

`extract_market_summary_features()` (`src/benchmark/baseline_models.py`) is the shared tabular
featurization used by every "Summary"/Random-Forest/Hybrid model: for each of the 19 market
channels it computes 8 window-level statistics (last value, mean, std, min, max, trend =
last − first, and the mean/std of the most recent 5 steps), giving $19 \times 8 = 152$ engineered
features from the same 30-step window the sequence models consume directly.

## 4. Model Architectures

### 4.1 Chronos Zero-Shot

`ChronosMarketPredictor.zero_shot_predict()` (`src/benchmark/chronos_encoder.py`). No training.
The 30-day close-price window (for Chronos Zero-Shot specifically, the extraction path actually
pulls a longer 365-day close-only window, `per_symbol_365`, to give the foundation model more
context) is converted to log-returns and fed to `amazon/chronos-t5-small` via
`ChronosPipeline.predict()`, sampling 5 future paths, summing each sampled path to the target
horizon, and taking the median across samples as the point forecast. Zero-shot means the pretrained
weights are used as-is — no gradient updates, no market-specific adaptation.

*(`ChronosAdapter` / `ChronosHybridAdapter` — frozen encoder + trainable projection/regressor head
— exist in the same file and are documented here for completeness per Section 3, but are not
currently wired into the `experiments` list.)*

### 4.2 LSTM Baseline / LSTM Hybrid

`LSTMPredictor` (`src/benchmark/baseline_models.py`): a standard multi-layer `nn.LSTM`
(`hidden_dim=64`, `num_layers=2` by default, dropout between layers when `num_layers>1`) consuming
the full 30-step, 19-channel market window; the final layer's hidden state across all LSTM layers
is flattened (`num_layers × hidden_dim`) and passed through a 2-layer MLP head
(`Linear → ReLU → Linear(1)`) to a scalar return prediction.

`LSTMHybridPredictor` adds a second branch: the same summary-statistics tabular vector (152-dim)
used by Random Forest, passed through its own small MLP, `LayerNorm`-normalized alongside the
(also `LayerNorm`-normalized) LSTM hidden state, concatenated, and passed through a combined head.

### 4.3 CNN-LSTM / CNN-LSTM Hybrid

`CNNLSTMPredictor`: the market window is linearly projected to `num_filters=64` channels, passed
through a stack of causal dilated 1D-convolution residual blocks (`_CausalDilatedBlock`, kernel
size 3, dilations `(1, 2, 4)` — each block left-pads by `(kernel_size−1)×dilation` so no future
information leaks into a given timestep, applies two causal convolutions with `GroupNorm` + ReLU +
dropout, and adds a residual connection), then fed into the same 2-layer LSTM + MLP head as
Section 4.2. The dilated stack gives the model a wider effective receptive field than a plain
Conv1D before the recurrent layer sees the sequence. `CNN-LSTM Hybrid` adds the tabular branch
exactly as in Section 4.2.

### 4.4 Random Forest Baseline

`RandomForestRegressor_Wrapper` (`sklearn.ensemble.RandomForestRegressor`) over the 152-dim
engineered summary features (Section 3), with the tabular input `StandardScaler`-normalized before
fitting. Default hyperparameters: 200 trees, `max_depth=9`, `min_samples_split=3`,
`max_features="log2"` (Section 5.3 covers HPO-searched alternatives). Has no latent embedding space
(`encode()` raises `NotImplementedError`) — it is a pure tabular regressor, included as the
non-sequential control: if a tree ensemble over hand-engineered summaries recovers most of the
signal, that bounds how much credit sequence models deserve for "seeing the whole window."

### 4.5 Linear Summary Baseline

`LinearSummaryRegressor_Wrapper`: `sklearn.linear_model.Ridge` ($\alpha=1.0$) over the same 152-dim
scaled summary features. The simplest possible control — a linear model with no capacity for
interaction effects — used to bound how much of the Random Forest's / neural baselines' apparent
skill is attributable to nonlinearity versus the summary features themselves already carrying most
of the signal.

### 4.6 MLP Summary Baseline

`MLPSummaryPredictor`: a 3-layer feedforward network (`Linear(152→64) → ReLU → Dropout →
Linear(64→32) → ReLU → Dropout → Linear(32→1)`) over the same scaled summary features, trained with
the shared scheduled sign-aware Huber loss (Section 5.1). Sits between Random Forest/Ridge
(tabular, non-sequential) and the LSTM/CNN-LSTM family (sequential): same input as the tabular
models, same loss/training regime as the sequence models.

### 4.7 GPT4TS Baseline / GPT4TS Hybrid

`src/benchmark/gpt4ts_encoder.py`. Adapts a pretrained GPT-2 backbone (`transformers.GPT2Model`,
6 layers) to market forecasting via time-series **patching**: the 30-step window is split into
overlapping patches (`patch_length=3`, `patch_stride=1` for 1D/5D — fine-grained; `patch_length=6`,
`stride=3` for 20D — coarser, roughly 3× fewer tokens, applied by the benchmark runner as a
horizon-adaptive choice) and each patch is embedded via a `Conv1d` + `LayerNorm` patch embedder
into GPT-2's input space. Adaptation is deliberately partial/lightweight: only the **top 1** GPT-2
block plus the final `LayerNorm` are unfrozen (`unfreeze_top_k_blocks=1`); the remaining 5 blocks
stay frozen. An attention-pooling head reduces the patch-token sequence to a single vector, followed
by a hidden-dim-128 regression head. Training uses **differential learning rates** — backbone
`5e-5`, head `3e-4` — via separate optimizer parameter groups, the standard recipe for adapting a
much-larger-than-the-data pretrained backbone without catastrophic forgetting. `GPT4TS Hybrid`
concatenates the tabular summary branch as in Sections 4.2/4.3.

## 5. Training Strategy

### 5.1 Shared Loss Function

Every trainable torch model (all except Chronos Zero-Shot and Random Forest/Ridge) shares one loss,
`sign_aware_huber_loss` (`src/benchmark/baseline_models.py`), wrapped in a scheduled variant,
`_scheduled_sign_aware_loss`:

- **Regression term**: Huber loss (`nn.HuberLoss`), with `delta` set per-run from the *training
  target distribution* itself (`compute_huber_delta`: the midpoint of the 25th/75th percentile of
  $|y_{\text{train}}|$, floored at 0.01) rather than a fixed constant — so the loss's
  quadratic/linear transition point adapts to how large a "typical" return is for that horizon.
- **Direction term**: a hinge-style penalty, `relu(margin − pred·sign(target))`, applied only to
  "active" samples ($|target| > \text{direction\_epsilon}=0.1$ in target-scale units), with a
  per-sample margin proportional to the target's own magnitude (floored at a minimum margin). This
  directly penalizes the model for getting the *sign* wrong (or right but with too little
  confidence), on top of the regression loss.
- **Class-balanced direction weighting** (`class_balance_dir=True` by default): the direction
  term is inverse-frequency reweighted between the positive/negative active samples (capped at 5×)
  so a class-imbalanced horizon doesn't let the direction loss collapse toward always predicting
  the majority sign.
- **Warmup + ramp schedule**: the direction penalty's weight is 0 for `warmup_epochs` epochs, then
  linearly ramps to its full weight (`sign_penalty_weight`, default 0.3, model-specific overrides
  as low as 0.02 — see Section 5.3 defaults) over `direction_ramp_epochs=3` further epochs. This
  lets the regression loss stabilize the model first before the (higher-variance) directional
  signal is introduced. Critically, **early stopping always compares the "clean" loss** (direction
  term evaluated as if warmup had already completed, `epoch=999`) so the stopping decision is not
  itself biased by where a given epoch falls in the warmup ramp.
- **Variance regularizer** (opt-in, default weight 0.0): an anti-collapse penalty that discourages
  near-constant predictions when enabled; the module's own comment records that a global 0.5
  weight was A/B tested and did not reliably help DA%/IC on a real signal-bearing cell, so it is
  off by default and only enabled per-cell with small, horizon-scaled weights when a specific
  model is observed to collapse.

### 5.2 Per-Model Optimizer / Schedule / Early Stopping

| Model | Optimizer | LR schedule | Epochs (max) | Patience | Batch size |
|---|---|---|---|---|---|
| LSTM / LSTM Hybrid | AdamW (`weight_decay=1e-5`) | `ReduceLROnPlateau` (factor 0.5, patience 5) | 100 | 15 | 32 (HPO-searched) |
| CNN-LSTM / CNN-LSTM Hybrid | AdamW (`weight_decay=1e-4`) | `ReduceLROnPlateau` | 100 | 15 | 32 |
| MLP Summary | AdamW (`weight_decay=1e-5`) | `ReduceLROnPlateau` | 100 | 15 | 32 |
| GPT4TS / GPT4TS Hybrid | AdamW, 2 param groups (backbone `5e-5`, head `3e-4`) | — | 50 | 10 | 32 |
| Random Forest | — (closed-form tree fit) | — | — | — | — |
| Linear (Ridge) | — (closed-form solve) | — | — | — | — |
| Chronos Zero-Shot | — (no training) | — | — | — | — |

Early stopping (`train_with_early_stopping`, `src/benchmark/training_utils.py`) restores the model
weights from the epoch with the best "clean" validation loss (not necessarily the final epoch),
with gradient-norm clipping at 1.0 on every step for every torch model.

### 5.3 Hyperparameter Search

Optuna-based HPO exists for two model families (`src/benchmark/baseline_hpo.py`), cached to disk
so repeated runs reuse the search result rather than re-searching:

- **LSTM** (`run_lstm_hpo`, 30 trials): `hidden_dim ∈ [32,256]` (step 32), `num_layers ∈ [1,4]`,
  `dropout ∈ [0,0.5]`, `lr ∈ [1e-4,1e-2]` (log scale), `batch_size ∈ [16,64]` (step 16).
- **Random Forest** (`run_rf_hpo`, 20 trials): `n_estimators ∈ [50,300]` (step 50),
  `max_depth ∈ [5,30]`, `min_samples_split ∈ [2,20]`, `max_features ∈ {sqrt, log2}`.

All other models (CNN-LSTM, MLP Summary, GPT4TS) use **fixed, non-searched defaults**
(`DEFAULT_BASELINE_PARAMS`, e.g. CNN-LSTM: `hidden_dim=64, num_layers=2, dropout=0.3, lr=1e-3,
batch_size=32, sign_penalty_weight=0.02`) rather than a dedicated Optuna search — this is a
resource/scope trade-off recorded here rather than left implicit, since it means the LSTM/RF rows
in the eventual results table reflect a tuned configuration while CNN-LSTM/MLP/GPT4TS reflect
untuned defaults, which is a relevant caveat when comparing their relative ranking.

### 5.4 Reproducibility

- **Baseline comparison protocol is single-seed.** `run_model_benchmark.py` calls
  `set_global_seed(42)` once at startup and every baseline model in this document's roster is
  fitted exactly once (no multi-seed ensembling) — this is a different protocol from the
  three-seed robustness set (`{1, 42, 123}`) used for the Phase 2/3 fusion/ablation comparisons
  (`src/benchmark/ablation_report.py`), and the two should not be conflated. Per-seed variance for
  these baseline models has not been characterized; if that is desired later, it would require
  re-running this benchmark under multiple seeds, which is out of scope for the current pending
  run.
- Every model, symbol, and horizon shares the exact train/val/test split described in
  `01_market_data_overview.md` Section 5 (chronological, horizon-aware purge).

## 6. Results — Pending

**No results are reported here yet.** Before writing this section, the repository's entire git
history was searched for a committed baseline-comparison output matching the current 7-symbol,
2020–2026 configuration; none exists. The only committed benchmark artifacts
(`results/chronos_benchmark_*.csv`, last updated in commit `45e8c0a`) are a 2-symbol (VCB, BID),
2022–2026 run that predates the `unitstd_v2` correctness fix, and are already marked "do not cite"
in `docs/reference/phase2_benchmark_report_HISTORICAL.md`. Reusing or extrapolating from that run
here would misrepresent the current pipeline's performance, so this section is left as an explicit
placeholder rather than filled with numbers that cannot be attributed to the current codebase.

### 6.1 How to generate the real results

```powershell
# Full baseline sweep, all 7 symbols, all 3 horizons (single seed=42, per Section 5.4)
python run_model_benchmark.py --stage predict

# Or one horizon at a time (faster iteration):
python run_model_benchmark.py --stage predict --horizons 1
python run_model_benchmark.py --stage predict --horizons 5
python run_model_benchmark.py --stage predict --horizons 20

# Regenerate the summary charts from the resulting CSVs:
python run_model_benchmark.py --stage plot
```

This produces `results/model_benchmark_{1,5,20}d.csv` (one row per symbol per model, plus an
`AVG` row aggregating across all 7 symbols — the "single seed, averaged across symbols" view
requested for this document) and, from the `plot` stage,
`results/figures/model_{h}d.png` (bar chart of the `AVG` row across models, all primary metrics)
and `results/figures/per_symbol_heatmap_{h}d.png` (per-symbol × per-model heatmap, surfacing
whether a model's edge is broad-based or concentrated in one or two symbols).

### 6.2 Table to be filled in

Once generated, this section should report, per horizon (1D/5D/20D), one table with one row per
model in Section 3 and the following columns pulled directly from `compute_all()`
(Section 2): **MAE, RMSE, DA%, DA_skill%, Sharpe, IC, Prec, Rec, F1, ESS**, plus **CompositeScore**
from `compute_composite_metrics()`. Both the per-symbol breakdown and the `AVG`-across-symbols row
should be retained (not just the average), since Section 6.1's heatmap figure is only informative
if the underlying per-symbol numbers are also in the document.

### 6.3 Charts to be embedded

- `model_{h}d.png` (one per horizon) — copied into this folder and embedded, per the project's
  "figures live beside the document" convention.
- `per_symbol_heatmap_{h}d.png` (one per horizon) — same treatment.

## 7. Analysis Framework (to be completed once results land)

This section cannot draw conclusions without real numbers, but the specific questions the analysis
should answer, given the metric definitions in Section 2, are recorded here so the eventual
analysis is not written ad hoc:

1. **Does any baseline beat its base rate?** Report DA_skill%, not raw DA%, per model per horizon —
   a model can show DA% > 50 purely from a favorable class imbalance in a given horizon/symbol.
2. **Does skill hold up under overlap correction?** Compare `DA%`/`F1` against `DA_ind%`/`F1_ind` —
   a large gap between the two indicates the plain metric's apparent skill is partly an artifact of
   overlapping-window inflation (Section 2.2), especially relevant at the 20D horizon where overlap
   is largest.
3. **Does Chronos Zero-Shot's pretrained prior transfer to VN bank equities at all**, relative to
   the from-scratch-trained baselines (LSTM/CNN-LSTM/RF/MLP/Linear)? This is the specific
   "foundation model transfer" question Phase 1 is meant to establish before Phase 2 builds fusion
   on top of the strongest backbone.
4. **How much of the trained baselines' apparent skill is attributable to nonlinearity** — compare
   Linear Summary Baseline (no interaction terms) against Random Forest / MLP Summary (same input
   features, nonlinear) to bound this.
5. **Is GPT4TS's partial fine-tuning (Section 4.7) worth its extra training cost** relative to the
   fully-trained-from-scratch LSTM/CNN-LSTM family, given they share the same market input?
6. **Statistical significance**, not just point-estimate ranking: for the top 2 models by
   CompositeScore per horizon, run `diebold_mariano_test` and/or `paired_bootstrap_da`
   (Section 2.6) before claiming one is better than the other.
7. **Per-symbol consistency**: does the best model by `AVG` win broadly (per the heatmap figure) or
   is its average propped up by one or two symbols?

## 8. Summary

| Item | Value |
|---|---|
| Baseline models evaluated | 10 (Section 3) — market-only, no news |
| Shared training loss | Scheduled sign-aware Huber (regression + class-balanced direction penalty) |
| HPO coverage | LSTM (30-trial Optuna), Random Forest (20-trial Optuna); others fixed defaults |
| Seed protocol | Single seed (42) for this baseline comparison; `{1,42,123}` reserved for Phase 2/3 fusion robustness checks |
| Primary ranking metric | RMSE, cross-checked against DA_skill%, IC, and CompositeScore (diagnostic only) |
| Results status | **Pending** — no real run exists yet for the current 7-symbol config; Section 6.1 has the exact reproduction command |

## References

- `src/benchmark/metrics.py` — every metric definition in Section 2
- `src/benchmark/baseline_models.py` — LSTM/CNN-LSTM/RF/Ridge/MLP architectures, shared loss, `BaseTorchMarketPredictor`/`BaseTorchHybridPredictor`
- `src/benchmark/chronos_encoder.py` — Chronos Zero-Shot, `ChronosAdapter`/`ChronosHybridAdapter`
- `src/benchmark/gpt4ts_encoder.py` — GPT4TS patch embedding and partial fine-tuning
- `src/benchmark/training_utils.py` — shared early-stopping training loop
- `src/benchmark/baseline_hpo.py` — LSTM/RF Optuna search spaces and fixed defaults for other models
- `run_model_benchmark.py` — experiment roster, per-model training call sites, reproduction commands
- `src/benchmark/ablation_report.py` — multi-seed robustness set `{1, 42, 123}` (Phase 2/3 only)
- `docs/reference/phase2_benchmark_report_HISTORICAL.md` — superseded 2-symbol run; not cited for numbers
