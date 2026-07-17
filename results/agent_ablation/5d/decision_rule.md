# Agent-ablation decision rule — 5d

## §10.8 pre-committed gates

- **Gate 1 (AURC, calibration, LLM-free):** PASS (significant) — ΔAURC=-0.035838 CI[-0.063605, -0.009082] (negative ⇒ gate confidence lowers risk-coverage area; selective DA 51.2299%→59.3573%@25%→58.2895% gated is a monotone lift).
- **Gate 2 (faithfulness, A5 vs A1):** PENDING — requires the LLM comparator run (A1 vs A5 narration). LLM reachable, run `eval` with LLM enabled.

## Calibration (H2)

- Full-book DA: 51.2299%  |  selective DA @ 25%: 59.3573%
- Gated DA: 58.2895%  IC: 0.2081  Sharpe: 0.5194  coverage: 0.3959
- AURC gate: 0.448736  vs no-skill: 0.484574

## Cross-sectional IC (secondary, universe-limited)

- all: -0.0004  |  matched: 0.0402
- matched vs placebo: {'delta': 0.0422, 'ci': [-0.0071, 0.0914], 'significant': False}

## LLM rung status

- runnable

## Verdict

H2 calibration: DIRECTIONAL + monotone selective-DA lift, significant. H3 faithfulness: PENDING LLM run (reported honestly, not inferred).