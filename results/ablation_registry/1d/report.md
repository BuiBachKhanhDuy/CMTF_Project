# CMTF Component Ablation Registry — 1D Report

Reproducible, config-driven ablation of the CMTF(LSTM) champion. Every cell derives from `CMTF_CORE` via a single-field override (src/benchmark/ablation_registry.py); GATED metrics (`DA%_gated`, `Sharpe_gated`, `IC_gated`) are the PRIMARY reported metrics, computed by a validation-calibrated fixed-coverage confidence gate layered on top of each cell's frozen predictions.

## Ranking (by gated DA%, then gated Sharpe, then gated IC)

| rank | cell | group | DA%_gated | Sharpe_gated | IC_gated | DA% | Sharpe | IC | RMSE | gate_coverage | seed_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17 | 4 — Output formulation | 56.4132 | 0.8581 | 0.1261 | 46.3466 | 0.5128 | 0.0499 | 0.0201 | 0.2597 | 3 |
| 2 | 4 | 1 — Component knockouts | 56.1557 | 0.7999 | 0.1136 | 46.6423 | 0.3463 | 0.0470 | 0.0203 | 0.3021 | 3 |
| 3 | 18 | 5 — Learned gate | 55.4255 | 0.7993 | 0.1075 | 46.0856 | 0.3683 | 0.0374 | 0.0200 | 0.2742 | 3 |
| 4 | 8p | 2 — News side | 55.2052 | 0.6058 | 0.1117 | 46.6945 | 0.3134 | 0.0240 | 0.0199 | 0.2557 | 3 |
| 5 | 13 | 3 — Gate sweep | 55.0831 | 0.8759 | 0.1484 | 46.1204 | 0.4087 | 0.0378 | 0.0201 | 0.2840 | 3 |
| 6 | 9 | 2 — News side | 54.6264 | 0.5400 | 0.0756 | 47.8253 | 0.3236 | 0.0371 | 0.0202 | 0.2888 | 3 |
| 7 | 11 | 3 — Gate sweep | 54.4680 | 0.6576 | 0.1030 | 46.5205 | 0.2640 | 0.0250 | 0.0199 | 0.2691 | 3 |
| 8 | 0 | 0 — Reference | 54.4474 | 0.4453 | 0.0847 | 45.8594 | 0.4235 | 0.0262 | 0.0199 | 0.2754 | 3 |
| 9 | 8 | 2 — News side | 54.2960 | 0.6626 | 0.0865 | 46.1204 | 0.6229 | 0.0310 | 0.0199 | 0.2417 | 3 |
| 10 | 18p | 5 — Learned gate | 54.0965 | 0.6105 | 0.0822 | 48.1559 | 0.6081 | 0.0215 | 0.0200 | 0.2474 | 3 |
| 11 | 2 | 1 — Component knockouts | 53.6784 | 0.5268 | 0.0863 | 47.3556 | 0.7319 | 0.0326 | 0.0199 | 0.2589 | 3 |
| 12 | 12 | 3 — Gate sweep | 53.5993 | 0.7134 | 0.1105 | 46.4335 | 0.6950 | 0.0365 | 0.0202 | 0.2993 | 3 |
| 13 | 3 | 1 — Component knockouts | 53.4890 | 0.4227 | 0.0927 | 47.7209 | 0.7294 | 0.0383 | 0.0200 | 0.2840 | 3 |
| 14 | 0p | 0 — Reference | 53.2515 | 0.5454 | 0.0836 | 46.5031 | 0.5814 | 0.0322 | 0.0200 | 0.2593 | 3 |
| 15 | 7 | 1 — Component knockouts | 53.0472 | 0.4547 | 0.0722 | 46.3814 | 0.5382 | 0.0189 | 0.0199 | 0.2631 | 3 |
| 16 | 1 | 1 — Component knockouts | 53.0291 | 0.4852 | 0.0728 | 46.0508 | 0.2365 | 0.0234 | 0.0199 | 0.2644 | 3 |
| 17 | 5 | 1 — Component knockouts | 52.6569 | 0.3321 | 0.0306 | 47.1468 | 0.6218 | 0.0381 | 0.0201 | 0.2134 | 3 |
| 18 | 6 | 1 — Component knockouts | 52.6273 | 0.5456 | 0.0704 | 45.8072 | 0.2737 | 0.0343 | 0.0208 | 0.3570 | 3 |
| 19 | 15 | 4 — Output formulation | 52.5972 | 0.5399 | 0.0213 | 45.1113 | 0.2578 | 0.0227 | 0.0201 | 0.3514 | 3 |
| 20 | 10 | 3 — Gate sweep | 52.4384 | 0.4035 | 0.0391 | 46.5379 | 0.5165 | 0.0183 | 0.0205 | 0.3052 | 3 |
| 21 | 16 | 4 — Output formulation | 50.9664 | 0.3624 | 0.0324 | 45.1287 | 0.1420 | 0.0064 | 0.0203 | 0.2193 | 3 |
| 22 | 14 | 4 — Output formulation | 50.3266 | 0.3469 | 0.0369 | 46.3292 | 0.6326 | 0.0277 | 0.0198 | 0.2706 | 3 |

## Real-minus-placebo comparisons (bootstrap 95% CI)

Positive `real_minus_placebo_*` means the REAL-news cell beats its shuffled-news placebo twin — evidence of genuine news signal rather than a generic decision-layer artifact. CI computed via paired bootstrap (2000 resamples) on one seed's frozen test predictions (see `bootstrap_seed` column); point estimates average the delta across every seed with both predictions cached.

| real_cell | placebo_cell | real_minus_placebo_DA | real_minus_placebo_Sharpe | real_minus_placebo_IC | n_seeds_available | bootstrap_seed | DA_ci_low | DA_ci_high | Sharpe_ci_low | Sharpe_ci_high | IC_ci_low | IC_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0p | 0.2538 | 0.2742 | 0.0050 | 3 | 1 | -1.4114 | 1.9681 | -0.1966 | 0.7534 | -0.0311 | 0.0416 |
| 8 | 8p | -1.1983 | -0.0521 | 0.0043 | 3 | 1 | -3.1724 | 0.7235 | -0.6262 | 0.5701 | -0.0313 | 0.0394 |
| 18 | 18p | -0.7648 | -0.1668 | 0.0314 | 3 | 1 | -3.2935 | 1.7171 | -0.9280 | 0.6303 | -0.0114 | 0.0709 |

## Gate monotonicity (coverage-accuracy diagnostics)

Spearman correlation between confidence rank (decile 1 = full book, decile 10 = most-confident 10%) and test-set performance. Strong positive correlation means the gate ranks conviction correctly — trading only the most confident predictions should not hurt.

| spearman_DA | spearman_Sharpe | spearman_IC | cell |
|---|---|---|---|
| 0.733 | -0.879 | 0.539 | 0 |
| -0.152 | -0.782 | -0.818 | 10 |
| 0.964 | 0.745 | 0.721 | 11 |
| 0.952 | 0.661 | 0.891 | 12 |
| 1.000 | 1.000 | 0.867 | 13 |

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
