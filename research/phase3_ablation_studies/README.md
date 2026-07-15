# Phase 3 — Ablation Studies

Status: **pending** — not yet written.

Planned scope:
- Component-ablation registry design (one change per cell vs. the base configuration)
- Placebo-controlled comparisons (real vs. placebo news) to separate genuine signal from
  capacity/overfitting effects
- Monotonicity and coverage diagnostics across horizons
- Ranked findings and which components survive multi-seed confirmation

Source material to draw from: `src/benchmark/ablation_config.py`, `src/benchmark/ablation_runner.py`,
`src/benchmark/ablation_report.py`, `results/ablation_registry/`, `run_ablation_benchmark.py`,
`run_ablation_registry.py`, `docs/reference/CACHING_GUIDE.md`.
