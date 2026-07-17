# CMTF Component Ablation Registry — 5D Report

Reproducible, config-driven ablation of the CMTF(LSTM) champion. Every cell derives from `CMTF_CORE` via a single-field override (src/benchmark/ablation_registry.py); GATED metrics (`DA%_gated`, `Sharpe_gated`, `IC_gated`) are the PRIMARY reported metrics, computed by a validation-calibrated fixed-coverage confidence gate layered on top of each cell's frozen predictions.

## Ranking (by gated DA%, then gated Sharpe, then gated IC)

| rank | cell | group | DA%_gated | Sharpe_gated | IC_gated | DA% | Sharpe | IC | RMSE | gate_coverage | seed_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13 | 3 — Gate sweep | 57.5025 | 0.3965 | 0.1533 | 48.7220 | 0.3612 | 0.1012 | 0.0481 | 0.3141 | 3 |
| 2 | 3 | 1 — Component knockouts | 56.6727 | 0.4059 | 0.1877 | 46.4461 | 0.0987 | 0.0775 | 0.0486 | 0.2968 | 3 |
| 3 | 1 | 1 — Component knockouts | 56.2054 | 0.3944 | 0.1677 | 45.6758 | 0.1060 | 0.0766 | 0.0494 | 0.3661 | 3 |
| 4 | 12 | 3 — Gate sweep | 55.9569 | 0.5275 | 0.0966 | 48.6345 | 0.4442 | 0.0560 | 0.0467 | 0.2378 | 3 |
| 5 | 6 | 1 — Component knockouts | 55.7838 | 0.3503 | 0.0986 | 46.1660 | -0.0116 | 0.0193 | 0.0482 | 0.2718 | 3 |
| 6 | 10 | 3 — Gate sweep | 55.7202 | 0.3428 | 0.1311 | 46.8137 | 0.1427 | 0.0702 | 0.0493 | 0.3478 | 3 |
| 7 | 9 | 2 — News side | 54.8729 | 0.2751 | 0.1342 | 46.8312 | 0.1489 | 0.0631 | 0.0488 | 0.3466 | 3 |
| 8 | 5 | 1 — Component knockouts | 54.7115 | 0.3805 | 0.1696 | 44.9405 | -0.1088 | 0.0530 | 0.0491 | 0.3398 | 3 |
| 9 | 2 | 1 — Component knockouts | 53.6565 | 0.1212 | 0.0707 | 47.6541 | 0.0861 | 0.0344 | 0.0492 | 0.2532 | 3 |
| 10 | 0 | 0 — Reference | 53.1569 | 0.2335 | 0.0918 | 44.2577 | -0.2266 | 0.0097 | 0.0487 | 0.2876 | 3 |
| 11 | 4 | 1 — Component knockouts | 53.1569 | 0.2335 | 0.0918 | 44.2577 | -0.2266 | 0.0097 | 0.0487 | 0.2876 | 3 |
| 12 | 18 | 5 — Learned gate | 51.9879 | 0.2169 | 0.0577 | 44.1352 | -0.3336 | -0.0077 | 0.0474 | 0.2616 | 3 |
| 13 | 14 | 4 — Output formulation | 51.6285 | -0.0700 | 0.0848 | 44.6779 | -0.2563 | 0.0232 | 0.0506 | 0.3351 | 3 |
| 14 | 8 | 2 — News side | 51.4692 | 0.2101 | 0.1171 | 45.7458 | 0.1425 | 0.0490 | 0.0486 | 0.3277 | 3 |
| 15 | 11 | 3 — Gate sweep | 51.3916 | 0.3095 | 0.1093 | 44.3102 | -0.2239 | 0.0072 | 0.0495 | 0.3053 | 3 |
| 16 | 0p | 0 — Reference | 51.3376 | 0.2302 | 0.0566 | 48.4769 | 0.4895 | -0.0186 | 0.0465 | 0.2327 | 3 |
| 17 | 16 | 4 — Output formulation | 50.4909 | -0.2651 | -0.0342 | 45.0280 | -0.1745 | -0.0187 | 0.0522 | 0.2799 | 3 |
| 18 | 15 | 4 — Output formulation | 50.0528 | -0.2234 | 0.0638 | 44.2927 | -0.4655 | 0.0012 | 0.0514 | 0.3400 | 3 |
| 19 | 7 | 1 — Component knockouts | 49.9694 | -0.1308 | 0.0001 | 44.8179 | -0.0362 | 0.0248 | 0.0485 | 0.3668 | 3 |
| 20 | 18p | 5 — Learned gate | 49.5478 | 0.1471 | 0.0278 | 46.0959 | 0.1982 | -0.0322 | 0.0466 | 0.2146 | 3 |
| 21 | 17 | 4 — Output formulation | 47.8971 | -0.2462 | 0.0487 | 43.6625 | -0.4206 | 0.0044 | 0.0513 | 0.4089 | 3 |
| 22 | 8p | 2 — News side | 46.4292 | -0.2252 | -0.1009 | 46.4111 | 0.1545 | -0.0382 | 0.0478 | 0.2155 | 3 |

## Real-minus-placebo comparisons (bootstrap 95% CI)

Positive `real_minus_placebo_*` means the REAL-news cell beats its shuffled-news placebo twin — evidence of genuine news signal rather than a generic decision-layer artifact. CI computed via paired bootstrap (2000 resamples) on one seed's frozen test predictions (see `bootstrap_seed` column); point estimates average the delta across every seed with both predictions cached.

| real_cell | placebo_cell | real_minus_placebo_DA | real_minus_placebo_Sharpe | real_minus_placebo_IC | n_seeds_available | bootstrap_seed | DA_ci_low | DA_ci_high | Sharpe_ci_low | Sharpe_ci_high | IC_ci_low | IC_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0p | -5.8718 | -0.8705 | -0.0345 | 3 | 1 | -9.1149 | -2.5894 | -1.2804 | -0.4472 | -0.0874 | 0.0210 |
| 8 | 8p | 1.1327 | 0.1685 | 0.0555 | 3 | 1 | -1.8338 | 4.0259 | -0.2285 | 0.5737 | 0.0063 | 0.1063 |
| 18 | 18p | -0.6509 | -0.2292 | 0.0268 | 3 | 1 | -3.7273 | 2.2955 | -0.6422 | 0.1862 | -0.0310 | 0.0832 |

## Gate monotonicity (coverage-accuracy diagnostics)

Spearman correlation between confidence rank (decile 1 = full book, decile 10 = most-confident 10%) and test-set performance. Strong positive correlation means the gate ranks conviction correctly — trading only the most confident predictions should not hurt.

| spearman_DA | spearman_Sharpe | spearman_IC | cell |
|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 0 |
| 1.000 | 0.879 | 0.333 | 10 |
| 0.988 | 0.903 | 0.927 | 11 |
| 0.879 | -0.236 | 0.503 | 12 |
| 0.891 | 0.600 | 1.000 | 13 |

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
