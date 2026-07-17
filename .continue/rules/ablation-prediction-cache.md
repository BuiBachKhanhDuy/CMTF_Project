---
globs: src/benchmark/ablation_runner.py,run_ablation_benchmark.py
description: Rules for the ablation benchmark's layered caching (encoder cache,
  anchor cache, and the cell-level prediction cache) in
  src/benchmark/ablation_runner.py and run_ablation_benchmark.py.
---

The ablation benchmark uses a layered cache. Preserve these invariants when editing caching code:
1. Any cell-level prediction reuse MUST validate the on-disk provenance sidecar (`scaling_version`, `cell_id`, `data_sig`, `news_sig`) against the current run; treat any mismatch or missing sidecar as a cache MISS and recompute. Never reuse a prediction whose provenance does not match.
2. `data_sig` must capture market cols + window shapes + y_train/y_val/y_test + `_SCALING_VERSION` + `target_scale`. `news_sig` must capture news_scope, shuffle_news, and the news tensors/shapes.
3. Bump `_SCALING_VERSION` in ablation_runner.py whenever the target-scaling or encoder-training scheme changes, to invalidate stale encoder/anchor/prediction caches.
4. The encoder cache key must NOT depend on fusion_type (so none/late/cmtf share a market-only backbone), but MUST encode every hyperparameter that defines the encoder, including its training recipe (epochs/patience/lr/batch_size) once unified.
5. Early fusion must never reuse a market-only cached encoder (news enters before the encoder).
6. Always provide a `--no-cache` escape hatch that forces fresh retraining/inference.