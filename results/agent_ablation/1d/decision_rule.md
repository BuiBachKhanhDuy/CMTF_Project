# Agent-ablation decision rule — 1d

## §10.8 pre-committed gates

- **Gate 1 (AURC, calibration, LLM-free):** PASS (significant) — ΔAURC=-0.031529 CI[-0.057723, -0.004287] (negative ⇒ gate confidence lowers risk-coverage area; selective DA 46.5453%→49.4403%@25%→62.4% gated is a monotone lift).
- **Gate 2 (faithfulness, A5 vs A1):** PENDING — requires the LLM comparator run (A1 vs A5 narration). LLM reachable, run `eval` with LLM enabled.

## Calibration (H2)

- Full-book DA: 46.5453%  |  selective DA @ 25%: 49.4403%
- Gated DA: 62.4%  IC: 0.0971  Sharpe: 0.7587  coverage: 0.1307
- AURC gate: 0.512737  vs no-skill: 0.544266

## Cross-sectional IC (secondary, universe-limited)

- all: 0.0213  |  matched: 0.0296
- matched vs placebo: {'delta': 0.0459, 'ci': [0.0054, 0.0866], 'significant': True}

## LLM rung status

- runnable

## Verdict

H2 calibration: DIRECTIONAL + monotone selective-DA lift, significant. H3 faithfulness: PENDING LLM run (reported honestly, not inferred).