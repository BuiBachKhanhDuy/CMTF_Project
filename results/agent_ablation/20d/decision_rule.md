# Agent-ablation decision rule — 20d

## §10.8 pre-committed gates

- **Gate 1 (AURC, calibration, LLM-free):** PASS (significant) — ΔAURC=-0.104891 CI[-0.130469, -0.079575] (negative ⇒ gate confidence lowers risk-coverage area; selective DA 59.4326%→78.9264%@25%→83.6134% gated is a monotone lift).
- **Gate 2 (faithfulness, A5 vs A1):** PENDING — requires the LLM comparator run (A1 vs A5 narration). LLM reachable, run `eval` with LLM enabled.

## Calibration (H2)

- Full-book DA: 59.4326%  |  selective DA @ 25%: 78.9264%
- Gated DA: 83.6134%  IC: 0.4078  Sharpe: 1.1329  coverage: 0.2663
- AURC gate: 0.292872  vs no-skill: 0.397763

## Cross-sectional IC (secondary, universe-limited)

- all: 0.0509  |  matched: 0.1239
- matched vs placebo: {'delta': 0.0282, 'ci': [-0.0321, 0.0877], 'significant': False}

## LLM rung status

- runnable

## Verdict

H2 calibration: DIRECTIONAL + monotone selective-DA lift, significant. H3 faithfulness: PENDING LLM run (reported honestly, not inferred).