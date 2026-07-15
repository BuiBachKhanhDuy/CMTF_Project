# Agent-ablation decision rule — 5d

## §10.8 pre-committed gates

- **Gate 1 (AURC, calibration, LLM-free):** DIRECTIONAL, not significant at 95% — ΔAURC=-0.021294 CI[-0.04723, 0.005458] (negative ⇒ gate confidence lowers risk-coverage area; selective DA 47.6348%→52.741%@25%→54.3689% gated is a monotone lift).
- **Gate 2 (faithfulness, A5 vs A1):** PENDING — requires the LLM comparator run (A1 vs A5 narration). LLM reachable, run `eval` with LLM enabled.

## Calibration (H2)

- Full-book DA: 47.6348%  |  selective DA @ 25%: 52.741%
- Gated DA: 54.3689%  IC: 0.1285  Sharpe: 0.2516  coverage: 0.2668
- AURC gate: 0.496293  vs no-skill: 0.517587

## Cross-sectional IC (secondary, universe-limited)

- all: 0.0004  |  matched: 0.0402
- matched vs placebo: {'delta': 0.0413, 'ci': [-0.0077, 0.0906], 'significant': False}

## LLM rung status

- runnable

## Verdict

H2 calibration: DIRECTIONAL + monotone selective-DA lift, NOT significant at 95% CI (ΔAURC CI crosses 0). H3 faithfulness: PENDING LLM run (reported honestly, not inferred).