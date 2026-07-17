# CMTF Component Ablation Registry — 20D Report

Reproducible, config-driven ablation of the CMTF(LSTM) champion. Every cell derives from `CMTF_CORE` via a single-field override (src/benchmark/ablation_registry.py); GATED metrics (`DA%_gated`, `Sharpe_gated`, `IC_gated`) are the PRIMARY reported metrics, computed by a validation-calibrated fixed-coverage confidence gate layered on top of each cell's frozen predictions.

## Ranking (by gated DA%, then gated Sharpe, then gated IC)

| rank | cell | group | DA%_gated | Sharpe_gated | IC_gated | DA% | Sharpe | IC | RMSE | gate_coverage | seed_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 14 | 4 — Output formulation | 85.2531 | 1.0605 | 0.2369 | 59.8234 | 1.1492 | 0.3522 | 0.0904 | 0.2379 | 3 |
| 2 | 13 | 3 — Gate sweep | 83.6976 | 0.9906 | 0.3024 | 56.8617 | 0.8169 | 0.3483 | 0.0936 | 0.2454 | 3 |
| 3 | 11 | 3 — Gate sweep | 83.2842 | 1.1455 | 0.3525 | 56.5121 | 1.0309 | 0.3633 | 0.0897 | 0.2872 | 3 |
| 4 | 3 | 1 — Component knockouts | 82.8598 | 1.1489 | 0.3860 | 57.0640 | 1.0924 | 0.3507 | 0.0902 | 0.2754 | 3 |
| 5 | 10 | 3 — Gate sweep | 81.3205 | 1.0360 | 0.2667 | 54.5070 | 0.7679 | 0.2824 | 0.0943 | 0.2510 | 3 |
| 6 | 17 | 4 — Output formulation | 80.8517 | 0.9702 | 0.4261 | 53.9183 | 0.8699 | 0.2896 | 0.0914 | 0.2227 | 3 |
| 7 | 16 | 4 — Output formulation | 80.5146 | 1.0346 | 0.3621 | 54.5070 | 0.8669 | 0.3088 | 0.0913 | 0.2597 | 3 |
| 8 | 4 | 1 — Component knockouts | 80.4996 | 1.0415 | 0.3222 | 56.6961 | 0.9298 | 0.3358 | 0.0935 | 0.2832 | 3 |
| 9 | 0 | 0 — Reference | 80.0595 | 1.0247 | 0.2641 | 54.1207 | 0.7721 | 0.2813 | 0.0927 | 0.2977 | 3 |
| 10 | 6 | 1 — Component knockouts | 79.8436 | 1.0822 | 0.2530 | 51.8948 | 0.5971 | 0.2554 | 0.0944 | 0.3008 | 3 |
| 11 | 5 | 1 — Component knockouts | 78.6801 | 1.0526 | 0.3415 | 56.4386 | 1.0007 | 0.3060 | 0.0921 | 0.2930 | 3 |
| 12 | 15 | 4 — Output formulation | 78.5962 | 1.0248 | 0.3435 | 55.7395 | 0.8932 | 0.2922 | 0.0922 | 0.2560 | 3 |
| 13 | 8 | 2 — News side | 75.3494 | 0.9036 | 0.3282 | 53.8999 | 0.7493 | 0.2365 | 0.0952 | 0.2417 | 3 |
| 14 | 9 | 2 — News side | 75.2612 | 0.9124 | 0.3939 | 53.0721 | 0.6336 | 0.2129 | 0.0938 | 0.2517 | 3 |
| 15 | 12 | 3 — Gate sweep | 74.9825 | 0.8064 | 0.3162 | 54.2311 | 0.7417 | 0.2321 | 0.0969 | 0.2130 | 3 |
| 16 | 18 | 5 — Learned gate | 74.8261 | 0.8917 | 0.3987 | 54.0103 | 0.6588 | 0.2401 | 0.0941 | 0.2786 | 3 |
| 17 | 1 | 1 — Component knockouts | 73.5152 | 0.8200 | 0.4089 | 53.7160 | 0.7079 | 0.2562 | 0.0945 | 0.2371 | 3 |
| 18 | 8p | 2 — News side | 73.0728 | 0.8733 | 0.2593 | 59.1244 | 0.9356 | 0.2348 | 0.0935 | 0.2852 | 3 |
| 19 | 7 | 1 — Component knockouts | 72.6957 | 0.8199 | 0.2807 | 50.8278 | 0.6035 | 0.2013 | 0.0954 | 0.2651 | 3 |
| 20 | 18p | 5 — Learned gate | 66.4177 | 0.7171 | 0.2556 | 56.0338 | 0.8609 | 0.1568 | 0.0944 | 0.2938 | 3 |
| 21 | 0p | 0 — Reference | 66.0992 | 0.7210 | 0.0968 | 55.9419 | 0.8415 | 0.1849 | 0.0953 | 0.2895 | 3 |
| 22 | 2 | 1 — Component knockouts | 62.5297 | 0.5064 | 0.2522 | 51.6924 | 0.3956 | 0.1449 | 0.0992 | 0.2935 | 3 |

## Real-minus-placebo comparisons (bootstrap 95% CI)

Positive `real_minus_placebo_*` means the REAL-news cell beats its shuffled-news placebo twin — evidence of genuine news signal rather than a generic decision-layer artifact. CI computed via paired bootstrap (2000 resamples) on one seed's frozen test predictions (see `bootstrap_seed` column); point estimates average the delta across every seed with both predictions cached.

| real_cell | placebo_cell | real_minus_placebo_DA | real_minus_placebo_Sharpe | real_minus_placebo_IC | n_seeds_available | bootstrap_seed | DA_ci_low | DA_ci_high | Sharpe_ci_low | Sharpe_ci_high | IC_ci_low | IC_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0p | -7.6854 | -0.3489 | -0.0040 | 3 | 1 | -9.9280 | -5.3903 | -0.5024 | -0.1957 | -0.0544 | 0.0445 |
| 8 | 8p | -2.5121 | -0.2824 | -0.0452 | 3 | 1 | -5.3153 | 0.1676 | -0.4769 | -0.0804 | -0.0982 | 0.0057 |
| 18 | 18p | -2.9826 | -0.4094 | 0.0360 | 3 | 1 | -5.3276 | -0.7148 | -0.5596 | -0.2610 | -0.0167 | 0.0920 |

## Gate monotonicity (coverage-accuracy diagnostics)

Spearman correlation between confidence rank (decile 1 = full book, decile 10 = most-confident 10%) and test-set performance. Strong positive correlation means the gate ranks conviction correctly — trading only the most confident predictions should not hurt.

| spearman_DA | spearman_Sharpe | spearman_IC | cell |
|---|---|---|---|
| 1.000 | 1.000 | 0.745 | 0 |
| 1.000 | 0.176 | -0.552 | 10 |
| 1.000 | -0.333 | -0.345 | 11 |
| 0.988 | -1.000 | 0.576 | 12 |
| 1.000 | -0.903 | 1.000 | 13 |

## Cell documentation

| cell | group | research question |
|---|---|---|
| 0 | 0 — Reference | Reference cell. Unmodified CMTF_CORE (LSTM champion). Every other registry cell's deltas are measured against this anchor. |
| 0p | 0 — Reference | Placebo twin of cell 0 (shuffle_news=True). Establishes the noise floor: if news carries no genuine signal, this should collapse toward the market-only baseline. Anchors every real-minus-placebo comparison in the registry. |
| 1 | 1 — Component knockouts | Cross-modal attention knockout (use_cross_attention=False). Research question: does letting market queries attend over news tokens beat a simpler pooled-news fallback? |
| 2 | 1 — Component knockouts | Recency gating knockout (recency_gate_k=0, disables the exponential recency decay entirely). Research question: does down-weighting stale news tokens by recency matter, or is plain relevance gating sufficient? |
| 3 | 1 — Component knockouts | News gate knockout (use_news_gate=False). Research question: does the learned market-conditioned sigmoid gate on the news branch add value over ungated attention output? |
| 4 | 1 — Component knockouts | Auxiliary market-anchor loss knockout (use_aux_loss=False). Research question: does anchoring training to the encoder's own scalar prediction help keep the fusion head close to a known-good backbone? |
| 5 | 1 — Component knockouts | Variance-regularisation knockout (use_variance_reg=False). Research question: does the attention-collapse guard prevent degenerate (near-constant) fused predictions? |
| 6 | 1 — Component knockouts | Sentiment contribution (sentiment_mode='none' strips the scalar sentiment features). Research question: how much of CMTF's edge is sentiment vs raw news embeddings? |
| 7 | 1 — Component knockouts | News positional encoding (use_positional_encoding=True; CMTF_CORE default is False). Research question: does explicit within-window recency position embedding help once recency gating already exists, or is it redundant/harmful? |
| 8 | 2 — News side | Matched-only news scope (news_scope='matched') vs CMTF_CORE's cross-symbol 'all' scope. Research question: does pooling news across the whole market beat restricting to each symbol's own matched news? |
| 8p | 2 — News side | Placebo twin of cell 8 (shuffle_news=True). Isolates whether the matched-scope result is genuine news signal or a decision-layer artifact. |
| 9 | 2 — News side | Handcrafted cross-modal interaction features (fusion_style='handcrafted' + all 5 interaction toggles ON) vs CMTF_CORE's minimal learned core ([market_latent, attn_out] only). Research question: do explicit interaction/cosine/context-product terms add signal over letting the fusion head learn its own interactions? |
| 10 | 3 — Gate sweep | News-gate softening: news_gate_alpha=0.3 (mostly pass-through, gate barely applied) vs CMTF_CORE's alpha=1.0 (gate fully applied). Research question: is a softer news gate better? |
| 11 | 3 — Gate sweep | News-gate softening: news_gate_alpha=0.5 (half-strength gate) vs CMTF_CORE's alpha=1.0. Midpoint of the alpha sweep. |
| 12 | 3 — Gate sweep | Tighter recency window: recency_gate_k=1 (very fast decay, only the most recent news matters) vs CMTF_CORE's k=3. |
| 13 | 3 — Gate sweep | Wider recency window: recency_gate_k=5 (slower decay, more of the news history retained) vs CMTF_CORE's k=3. |
| 14 | 4 — Output formulation | output_mode='encoder_residual': encoder's trained scalar pred + news residual, vs CMTF_CORE's 'anchored_fusion'. Reconfirms whether a fixed-weight additive news correction on top of the encoder's own head is competitive. |
| 15 | 4 — Output formulation | output_mode='fusion_plus_news': fusion_pred + news_residual with no DA anchor guard. Reconfirms the DA-vs-IC tradeoff documented in CMTF_FUSION_FINDINGS.md. |
| 16 | 4 — Output formulation | output_mode='market_plus_fusion' (DEPRECATED / previously shown harmful): re-predicts the market from a projection instead of using the encoder's trained head. Kept as a negative-control reconfirmation row. |
| 17 | 4 — Output formulation | use_two_stage=True: end-to-end encoder fine-tuning during fusion training, vs CMTF_CORE's frozen single-stage encoder. Reconfirms whether the two-stage gain (if any) is genuine news signal or mostly encoder fine-tuning. |
| 18 | 5 — Learned gate | gate_mode='learned': the fixed news_gate_alpha scalar is replaced by a lightweight trainable head (Linear->GELU->Linear->Sigmoid) predicting a per-sample mixing coefficient from [market_emb, pooled_news]. Research question: should the news-gate mixing strength be learned per-sample rather than fixed? |
| 18p | 5 — Learned gate | Placebo twin of cell 18 (shuffle_news=True). Confirms the learned gate does not simply learn to fit placebo noise (i.e. its gain, if any, should collapse here too). |
