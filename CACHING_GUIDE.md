# Ablation Benchmark Caching — Architecture, Correctness & Rerun Guide

This document describes the layered caching system used by the Phase-2 ablation
harness (`run_ablation_benchmark.py` → `src/benchmark/ablation_runner.py`) after
the **`unitstd_v2`** correctness pass. It covers what is cached, how reuse is
kept sound, what was intentionally *not* built, and how to (re)run.

---

## 1. Why cache at all

One ablation run trains many market encoders (LSTM / CNN-LSTM / GPT4TS / Chronos)
across four fusion strategies (`none` / `early` / `late` / `cmtf`), over
multiple horizons and seeds. (Random Forest was removed from the ablation study.)
The **market encoder is the expensive part**, and
the *same* backbone (same data, seed, params, recipe) is re-used by the
`none`, `late`, and `cmtf` cells as well as by the shared market-only *anchor*
used for disagreement diagnostics. Caching lets each backbone train **once** per
`(data, seed, params, recipe)` and be reused everywhere it is valid.

---

## 2. The three cache layers

| Layer | What it stores | Keyed by | Reused across |
|-------|----------------|----------|---------------|
| **L1 — Encoder cache** | trained torch `state_dict` (`cache/encoders/*.pt`) + in-memory `_ENCODER_CACHE` | `_encoder_cache_key(...)` | `none` / `late` / `cmtf` stage-1 + anchor (NOT `early`) |
| **L2 — Anchor cache** | market-only test predictions (`cache/anchors/*.npy`) + `_ANCHOR_PRED_CACHE` | `_anchor_cache_key(...)` | every non-`none` cell sharing the same backbone/data/recipe |
| **L3 — Cell prediction cache** | the fully-trained cell's test predictions (`cache/predictions/*.npy` + `*.json` provenance) | `cell_id` + `seed` + `horizon` + `data_sig` + `news_sig` + `scaling_version` | exact re-runs of the same cell |

- **L1/L2** deliver the *within-run* multiplier (share one backbone across fusion
  types).
- **L3** delivers the *re-run* multiplier — a cell that was already computed for
  the current data + scaling + news config skips **all** training/inference.
- **`early`** fusion intentionally never reuses the market-only encoder (its
  encoder is trained on concatenated market+news inputs, so its weights are a
  different object entirely).

---

## 3. Correctness invariants (the `unitstd_v2` pass)

The cache is only useful if reuse can **never change results**. Two safeguards:

### 3a. Cache keys encode *everything* that affects the artifact
`_encoder_cache_key` / `_anchor_cache_key` fold in: scaling version, encoder
name, horizon, seed, sentiment mode, a feature-column hash, an
encoder-hyperparameter hash, a split-shape/time-bounds hash, **and a
`recipe_sig`** (new). L3 additionally validates a JSON provenance sidecar
(`scaling_version`, `cell_id`, `data_sig`, `news_sig`) before reuse — any
mismatch or missing sidecar is a **miss** (recompute), never a silent reuse.

### 3b. One canonical encoder training recipe
Previously the three encoder-training paths disagreed:

| Path | Old recipe |
|------|-----------|
| `none` | model defaults (`epochs=50`, `patience=5`) |
| `cmtf` stage-1 | `cfg.market_epochs` / `cfg.market_patience` |
| `late` | **loaded the cached encoder then retrained it anyway** with defaults — the cache load was wasted |

Because the encoder key excludes `fusion_type` (that's what enables sharing) but
*omitted the recipe*, whichever cell ran first could poison the shared entry —
an unsound, order-dependent result.

**Fix:** a single `_encoder_recipe(cfg, params, enc_name, horizon)` returns the
canonical fit-kwargs and is applied **identically** in `none`, `late`, and
`cmtf` stage-1:

```python
{ "epochs": cfg.market_epochs,
  "batch_size": p.get("batch_size", 32),
  "learning_rate": p.get("lr", 1e-3),
  "patience": cfg.market_patience,
  "warmup_epochs": min(horizon, 10) }
```

> Since the grid redesign, **every CMTF cell derives from a single `CMTF_CORE`
> config**, so all CMTF cells share one training recipe — maximising L1 encoder
> reuse within a backbone/seed.

- The recipe is folded into the encoder/anchor cache keys (`recipe_sig`), so
  **matching recipes share** (fast) and **mismatched recipes never share**
  (correct). Example: a `cmtf` config with `market_epochs=80` will *not* share a
  backbone with a `none` cell at the default `market_epochs=50` — different
  training, different weights, different key.
- `late` now passes the recipe to `wrapper.fit(...)` and, on an encoder cache
  hit, passes `skip_encoder_fit=True` so the **cached weights are preserved**
  (the leakage-free OOF phase still runs, since residual targets require it).
  This is the new `skip_encoder_fit` parameter on `LateFusionWrapper.fit`.

### Behavior change (expected)
`none` and `late` (torch) baselines now train with `cfg.market_patience`
(default **8**) instead of the model default **5**, and `late`'s encoder now
trains with the full recipe on both OOF folds and the final fit. This shifts
those numbers slightly. **`_SCALING_VERSION` was bumped `unitstd_v1 → unitstd_v2`**,
which invalidates every on-disk encoder/anchor/prediction cache from the old
scheme automatically (no manual cache wipe needed for correctness).

---

## 4. What was intentionally NOT built (deferred)

To keep the harness in a correct, runnable state we deliberately stopped after
L1–L3. The following were considered and deferred with rationale:

| Idea | Decision | Rationale |
|------|----------|-----------|
| **Full-model cache** (persist the entire fusion model) | **Skip** | Subsumed by L3: for re-runs we only need the *predictions*, which L3 already stores and validates. Persisting whole fusion models adds large artifacts and version-fragility for no extra speedup. |
| **Frozen-embedding cache** for Chronos/GPT4TS backbones | **Defer** | This *changes results* (freezing a backbone ≠ fine-tuning it), so it is an opt-in modeling choice that needs its own validation, not a transparent cache. L1 already ensures each backbone trains once per `(data, seed, params, recipe)` and is reused across fusion types, which is the main multiplier; a frozen-embedding cache is a further optimization on top, to be added behind an explicit flag when needed. |

---

## 5. Running / rerunning

### Normal run (caching ON)
```powershell
& "$PWD\.venv\Scripts\python.exe" run_ablation_benchmark.py
```
Defaults: `--table all --horizons 1 5 20 --seeds 42 123 456` (tables
`fusion_comparison` + `component_ablation`). On the **first**
`unitstd_v2` run, all L1/L2/L3 caches are cold and will be (re)computed; the
banner logs `Prediction cache: ON`.

### Force full recompute (caching OFF)
Use this to reproduce numbers from scratch or when validating the pipeline:
```powershell
& "$PWD\.venv\Scripts\python.exe" run_ablation_benchmark.py --no-cache
```
`--no-cache` disables the **L3** prediction-cache *load* (banner logs
`Prediction cache: OFF (--no-cache)`), so every cell is retrained. L1/L2 are
still written for within-run sharing.

### Fast smoke test (one cell family)
```powershell
& "$PWD\.venv\Scripts\python.exe" run_ablation_benchmark.py --table fusion_comparison --horizons 1 --seeds 42 --model lstm
```

### Note on the version bump
Because `_SCALING_VERSION = "unitstd_v2"`, the **next run recomputes** encoder,
anchor, and prediction caches regardless of `--no-cache` — old `unitstd_v1`
artifacts are ignored by key mismatch, not deleted. To reclaim disk, you may
delete `cache/encoders`, `cache/anchors`, `cache/predictions` directly; it is
not required for correctness.

---

## 6. Quick verification checklist

```powershell
# 1. Both modules parse & import
& "$PWD\.venv\Scripts\python.exe" -c "import src.benchmark.ablation_runner, src.benchmark.fusion_wrappers; print('IMPORT OK')"

# 2. Recipe is identical across none/late for the same backbone, and
#    recipe_sig makes matching recipes share a key / differing recipes not.
#    (See the smoke test in the PR description / commit body.)

# 3. --no-cache is available
& "$PWD\.venv\Scripts\python.exe" run_ablation_benchmark.py --help | Select-String no-cache
```

**Cache-safety rules** (also enforced via the auto-attached project rule
"Ablation Prediction Cache"):
1. Always validate the provenance sidecar before reusing an L3 prediction.
2. `data_sig` / `news_sig` must capture cols, split shapes, targets, scaling
   version, news scope and the full test news tensor.
3. Bump `_SCALING_VERSION` on **any** scheme or recipe change.
4. The encoder key excludes `fusion_type` (to allow sharing) but includes the
   training `recipe` (to keep sharing sound).
5. `early` fusion must never reuse the market-only encoder.
6. Always provide `--no-cache` for reproducibility.
