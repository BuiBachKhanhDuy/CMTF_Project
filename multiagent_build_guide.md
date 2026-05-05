# Multi-Agent Build Guide for the Next Step

## 1. Purpose of This Document

This document is the implementation plan for the next step of the project. It is not a generic architecture note. It is a code-grounded guide for turning the current repository into:

1. a LangGraph-based multi-agent system
2. a separate backtesting and evaluation layer

The plan below is intentionally constrained by the code that already exists in the repo.

## 2. Current Code Reality Checked

This plan is based on the current repository state, not on a clean-room redesign.

- `src/pipeline/orchestrator.py` already performs leakage-safe data fetching, temporal alignment, market feature engineering, VN-Index enrichment, news encoding, and optional Phase 2 PhoBERT sentiment integration.
- `src/phase2/handoff.py` and `src/phase2/inference.py` already expose the trained PhoBERT branch as reusable inference artifacts for downstream use. PhoBERT checkpoint and tokenizer are resolved from `outputs/phase2/latest/`.
- `src/pipeline/news_encoder.py` uses `dangvantuan/vietnamese-embedding` (768-dim sentence transformer) for text encoding and PhoBERT for title-level sentiment. It now supports **weighted pooling** (articles weighted by absolute sentiment score) and combines the 768-dim embedding with 5 PhoBERT sentiment features into `news_hybrid_emb` (773-dim total). Six sentiment scalar columns (`sentiment_mean`, `sentiment_max_abs`, `sentiment_positive_ratio`, `sentiment_negative_ratio`, `sentiment_score_count`, `sentiment_missing_flag`) are also propagated as market features.
- `src/benchmark/baseline_models.py` provides `ChronosLoRAPredictor` as the fine-tuned Chronos market backbone. It includes a market LSTM encoder (input_dim=23) and regression head. `combined_feature_dim = d_model(512) + market_hidden_dim(64) + tabular_dim`.
- `src/benchmark/chronos_cmtf.py` provides `ChronosCMTFPredictor` (currently **v8**). Architecture: `ResidualNewsFusionHead` with learnable positional encoding over the 30-bar news window, multi-head cross-attention (n_heads configurable via HPO), a learnable `news_weight` gate, and zero-news parity (output is exactly zero when all news slots are masked). Training uses **differential learning rate**: fusion head at `lr`, backbone at `lr × 0.1`. Ensemble of 3 seeds [42, 123, 456] at inference.
- `run_chronos_benchmark.py` builds sequence windows (seq_len=30), derives `news_masks`, applies purge-aware walk-forward splits (train ≤ 2024-06-30, val ≤ 2024-12-31, test > 2024-12-31), and evaluates prediction metrics across horizons [1, 5, 20] days.

### Current Model Dimensions (After Recent Fixes)

| Component | Dimension | Source |
|---|---|---|
| Market features | 23 | 17 OHLCV+TA + 6 sentiment scalars |
| News embedding (hybrid) | 773 | 768 vietnamese-embedding + 5 PhoBERT sentiment |
| Chronos d_model | 512 | amazon/chronos-t5-small |
| Market LSTM hidden | 64 | baseline HPO |
| Combined baseline feature | 599 | 512 + 64 + 23 (d_model + market_hidden + tabular) |
| Fusion hidden dim | 64 or 128 | HPO result per horizon |
| Cross-attention heads | 1 or 2 | HPO result per horizon |

### Current HPO Best Parameters

| Horizon | fusion_dim | n_heads | lr | dir_penalty | dropout |
|---|---|---|---|---|---|
| 1D | 64 | 2 | 4.73e-4 | 0.052 | 0.227 |
| 5D | 128 | 1 | 2.74e-4 | 0.087 | 0.289 |
| 20D | 128 | 2 | 1.20e-4 | 0.065 | 0.174 |

### Current Artifact Paths

| Artifact | Path Pattern |
|---|---|
| Dataset cache | `cache/dataset/dataset_{hash}.parquet` |
| Chronos tokens | `cache/chronos_emb/chronos_lora_tokens_{sym}_{h}d_{split_hash}_*.npy` |
| LoRA backbone ckpt | `cache/cmtf_models/ft_chronos_lora_backbone_v3_{sym}_{h}d_{param_hash}_{split_hash}.pt` |
| CMTF ensemble ckpt | `cache/cmtf_models/cmtf_lora_v8_{sym}_{h}d_seed{s}_{hpo_hash}_{ft_hash}_{sh}.pt` |
| HPO params | `cache/optuna/best_params_v7_{h}d.json` |
| Baseline HPO params | `cache/optuna/best_baseline_params_{h}d.json` |
| PhoBERT handoff | `outputs/phase2/latest/phase3_phobert_handoff.json` |

### Important Implications

- The current modeling contract is forward-return regression, not 3-class trade classification.
- Trading action should be a policy layer above the model, not the model target itself.
- The hybrid news tensor is **773-dim** (768 embedding + 5 sentiment). `news_dim` must always be derived from actual data rather than hardcoded.
- Explanation can reuse: (a) `fusion.last_attn_weights` — cross-attention over the 30-bar news window, (b) `fusion.news_weight` — the learned scalar showing overall news contribution magnitude, (c) article-level PhoBERT sentiment traces stored in `artifacts/hybrid_sentiment/`, (d) the `baseline_pred` vs `final_pred` delta showing how much news shifted the forecast.
- The repo does not yet contain LangChain or LangGraph orchestration.
- The repo does not yet contain a broker-style backtesting engine with commission, slippage, and equity-curve accounting.

## 3. Non-Goals for This Step

- Do not retrain PhoBERT from scratch.
- Do not replace Chronos LoRA with a new market model.
- Do not build a second fusion architecture unless the current CMTF interface proves insufficient.
- Do not make action classification the primary prediction target.
- Do not mix forecast evaluation and portfolio backtesting into one report.

## 4. Phase 1: Build the Multi-Agent System

### 4.1 Objective

Create a LangGraph-based orchestration layer that consumes the existing pipeline and model artifacts to answer three questions for a symbol at time $t$:

1. What is the predicted forward return?
2. What trading action follows from the policy threshold?
3. Why did the system make that call?

### 4.2 Why LangGraph Here

Use LangGraph as the control plane because it fits the missing part of the project:

- stateful orchestration across multiple steps
- durable execution and resumable runs
- traceable node execution
- optional human-in-the-loop checkpoints

Use plain LangChain components only inside nodes where a chat model or tool wrapper is actually needed. The market, news, fusion, and policy nodes should stay deterministic Python components.

### 4.3 Proposed Agent Graph

1. **Supervisor Node**

Validates `symbol`, `prediction_time`, `target_horizon_days`, `sequence_len`, and execution mode. Creates the shared graph state and enforces the time cutoff so downstream nodes only see information available at time $t$.

2. **Market Agent**

Reuses the existing market pipeline contract. Its job is to produce the latest close window, market feature window, and tabular feature row using the same feature engineering logic already used in the benchmark path.

3. **News Agent**

Reuses temporal alignment and Phase 2 PhoBERT inference. Its job is to collect only articles available before the cutoff, score titles with PhoBERT, aggregate sentiment per bar, and return `news_hybrid_emb` plus `news_mask`.

4. **Fusion Agent**

Loads the current Chronos LoRA backbone and CMTF v8 head. Its job is to compute the baseline market prediction and the final fused prediction using the current regression contract. Must load the **ensemble of 3 seed checkpoints** (seeds 42, 123, 456) and average their predictions for robustness. Exposes: `baseline_pred`, `news_residual` (= final − baseline), `final_pred`, `fusion.last_attn_weights` (30-bar attention), and `fusion.news_weight` (learned scalar gate ≈ 0.04–0.10 for good models).

5. **Decision Agent**

Maps the predicted return to `long`, `short`, or `flat` using an explicit threshold or threshold band. This keeps action selection outside the model and makes backtesting rules adjustable without retraining.

6. **Explanation Agent**

Builds a structured explanation object from hard evidence: recent market context, PhoBERT sentiment scores, aggregated sentiment features, attention weights, predicted return, and final action. If a chat model is used here, it should only rewrite the structured evidence into natural language and must not invent extra reasoning.

### 4.4 Recommended Shared State

The graph state should contain:

- request fields: `symbol`, `prediction_time`, `target_horizon_days`, `sequence_len` (default 30)
- market inputs: `close_window` (30,), `market_window` (30, 23), `market_tabular` (23,), `token_ids` (30, T), `attention_mask` (30, T)
- news inputs: `articles` (list of dicts), `title_scores` (list of floats), `sentiment_features` (6 scalars), `news_emb` (30, 773), `news_mask` (30,) bool
- model outputs: `baseline_pred` (float), `news_residual` (float), `final_pred` (float), `action` (str: "long"|"short"|"flat")
- explanation fields: `attn_weights` (30,), `news_weight_scalar` (float — learned gate), `top_news_items` (list), `explanation_payload` (dict)
- audit fields: `data_cutoff`, `artifact_versions` (dict with CMTF_VERSION="v8", backbone_ckpt hash, HPO hash), `ensemble_seed_preds` (list of 3 floats), `errors`, `warnings`

### 4.5 Mandatory Reuse of Current Project Assets

The first implementation must consume current assets instead of replacing them:

- PhoBERT inference from `src/phase2/handoff.py` and `src/phase2/inference.py`
- leakage-safe data preparation from `src/pipeline/orchestrator.py`
- hybrid news encoding from `src/pipeline/news_encoder.py`
- Chronos LoRA from `src/benchmark/baseline_models.py`
- CMTF fusion from `src/benchmark/chronos_cmtf.py`
- window and mask contract from `run_chronos_benchmark.py`

### 4.6 Implementation Order

1. Create a new `src/multiagent/` package with `state.py`, `graph.py`, `loaders.py`, and one file per graph node.
2. Add a model-loader layer that loads stable PhoBERT, Chronos LoRA, and CMTF v8 artifacts from current cache paths. Must load all 3 ensemble checkpoints per (symbol, horizon) pair from `cache/cmtf_models/cmtf_lora_v8_*.pt` and the backbone from `cache/cmtf_models/ft_chronos_lora_backbone_v3_*.pt`.
3. Add a `prepare_context` utility that reproduces the benchmark window contract for one symbol and one cutoff time: 30-bar sliding window with 23 market features, Chronos tokenization, and 773-dim hybrid news encoding with masks.
4. Implement Market Agent and News Agent first, because they define the state contract consumed by the fusion node.
5. Implement Fusion Agent as a thin wrapper around existing predictors. Add a `predict_with_explanation()` method to `ChronosCMTFPredictor` that returns a dict: `{"baseline_pred": float, "final_pred": float, "news_residual": float, "attn_weights": ndarray(30,), "news_weight": float}`. Run all 3 ensemble seeds and average.
6. Implement Decision Agent with configurable thresholds, for example `flat` when $|pred| < \epsilon$. Load thresholds from a config file so they can be tuned independently of the model.
7. Implement Explanation Agent with a deterministic template first, then optionally add an LLM rewriter through LangChain. The template should format Vietnamese-language explanations referencing specific article titles and dates.
8. Add a CLI or service entry point for one-shot inference and batch inference.

### 4.7 Explanation Design

The explanation layer is required, but it must be evidence-bounded.

The explanation should report:

- the predicted return from the market-only backbone (`baseline_pred`)
- the change introduced by news-aware fusion (`news_residual = final_pred - baseline_pred`)
- the learned `news_weight` gate value (shows how much the model trusts news in general)
- the strongest recent news bars according to cross-attention weights (with bar indices mapped to dates)
- the most influential titles and their PhoBERT sentiment scores from the sentiment trace
- whether the final action came from strong signal or threshold crossing
- ensemble agreement: whether all 3 seed models agree on direction

The explanation should not:

- claim causal certainty
- expose hidden chain-of-thought
- describe signals that were not present in the state
- hardcode sentiment importance if the attention or residual path disagrees

### 4.8 Acceptance Criteria for Phase 1

Phase 1 is done when:

- one graph call can produce a prediction for one symbol and one cutoff time
- the graph reuses current trained artifacts rather than retraining models
- zero-news samples still fall back to the baseline behavior
- the explanation includes structured evidence from PhoBERT traces and CMTF attention
- LangSmith or an equivalent trace layer can inspect each node output

## 5. Phase 2: Backtesting and Evaluation

### 5.1 Objective

Build a true broker-style backtesting layer on top of the Phase 1 graph. This phase must evaluate both forecast quality and trading usefulness, but those must remain separate reports.

### 5.2 What Exists Already

The current repo already supports:

- purge-aware walk-forward splitting
- prediction-level metrics such as MAE, RMSE, DA%, Sharpe, IC, Precision, Recall, and F1
- per-symbol and pooled benchmark reporting

The current repo does not yet support:

- portfolio ledger simulation
- explicit position management
- execution price assumptions
- commission and slippage modeling
- equity-curve-based risk metrics such as max drawdown and Sortino

### 5.3 Backtest Engine Design

1. **Input Stream**

Use the same leakage-safe chronological windows as the benchmark path. For each bar $t$, the graph only receives information available up to $t$.

2. **Signal Generation**

Run the multi-agent graph to produce `final_pred` and `action`.

3. **Execution Policy**

Define explicit rules for when the trade is filled. Example: generate the signal at bar close $t$, execute at next open $t+1$.

4. **Friction Model**

Apply commission, slippage, and optional short borrow or interest costs as separate parameters. Keep them configurable by market regime or instrument type.

5. **Portfolio Ledger**

Track cash, exposure, position direction, position size, entry price, realized PnL, unrealized PnL, and equity curve.

6. **Audit Trail**

Store the model explanation payload together with each executed order or daily state so later analysis can link trades back to evidence.

### 5.4 Policy Layer

Because the current model predicts returns, not discrete classes, the backtester needs an explicit policy such as:

- `long` if `pred >= buy_threshold`
- `short` if `pred <= -sell_threshold`
- `flat` otherwise

This policy must be versioned separately from model checkpoints so threshold tuning does not silently alter reported model performance.

### 5.5 Evaluation Outputs

Keep two evaluation families:

1. **Forecast Evaluation**

Use the existing benchmark metrics to compare predictive quality.

2. **Portfolio Evaluation**

Add cumulative return, annualized return if the sample is long enough, Sharpe, Sortino, max drawdown, turnover, win rate, average trade PnL, average holding period, and cost drag.

This split is important because a model can improve RMSE while still producing poor trade economics after costs.

### 5.6 Validation Sequence

1. Reproduce the current benchmark predictions without trading costs.
2. Run the backtester with zero commission and zero slippage and confirm the action path behaves as expected.
3. Turn on commission only and confirm PnL decreases.
4. Turn on slippage and confirm fills worsen or some trades fail under the chosen rules.
5. Compare zero-news cases against the baseline path to confirm explanation and execution remain stable.

### 5.7 Recommended Outputs

Store the new artifacts separately from existing benchmark CSVs:

- `results/backtests/` for summary tables and equity curves
- `artifacts/backtests/` for trade logs and per-step explanation payloads
- `cache/backtests/` for reusable simulation intermediates

### 5.8 Acceptance Criteria for Phase 2

Phase 2 is done when:

- the engine runs chronologically with no leakage
- trade rules and execution timing are explicit
- commission and slippage are configurable and tested
- forecast metrics and portfolio metrics are reported separately
- each trade can be traced back to the multi-agent explanation payload

## 6. Suggested Code Layout for the Next Step

```
src/multiagent/
    __init__.py
    state.py              # TypedDict graph state definition
    graph.py              # LangGraph graph construction and compilation
    loaders.py            # Model artifact loaders (CMTF, LoRA backbone, PhoBERT)
    config.py             # Decision thresholds, artifact paths, symbol registry
    market_agent.py       # Market data preparation node
    news_agent.py         # News retrieval and encoding node
    fusion_agent.py       # CMTF ensemble inference node
    decision_agent.py     # Threshold-based action mapping node
    explanation_agent.py  # Evidence-bounded explanation generation node
    cli.py                # Command-line entry point
src/backtesting/
    __init__.py
    engine.py             # Chronological simulation loop
    policy.py             # Threshold policy with configurable bands
    ledger.py             # Position and equity tracking
    metrics.py            # Portfolio-level evaluation metrics
tests/
    test_multiagent_graph.py
    test_backtesting_engine.py
    test_explanations.py
```

## 7. Immediate Next Actions

1. Install `langgraph`, `langchain-core`, and `langsmith` into the project `.venv`.
2. Build `src/multiagent/state.py` with the TypedDict state definition matching Section 4.4 dimensions.
3. Build `src/multiagent/loaders.py` — a module that resolves and loads CMTF v8 ensemble checkpoints, LoRA backbone, and PhoBERT bundle from their cache paths. Must handle missing-checkpoint errors gracefully.
4. Build `src/multiagent/market_agent.py` — reuses orchestrator pipeline for feature engineering and Chronos tokenization for a single (symbol, cutoff) pair.
5. Build `src/multiagent/news_agent.py` — reuses `NewsEncoder.encode_window()` and PhoBERT sentiment scoring for the 30-bar context window.
6. Build `src/multiagent/fusion_agent.py` — wraps CMTF `predict_with_explanation()` across 3 ensemble seeds, averages predictions, merges attention weights.
7. Add one-symbol smoke tests (VCB, horizon=1d) before attempting full backtests.
8. After inference graph is stable, implement the backtesting engine as a separate layer, not inside the benchmark runner.

## 8. References Used for This Plan

| Source | Type | What is referenced here |
| --- | --- | --- |
| Ansari et al., *Chronos: Learning the Language of Time Series* (2024) | Paper | Tokenized time-series pretrained backbone; used to justify keeping Chronos as the market foundation rather than replacing it with a new model family. |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | Paper | Parameter-efficient adaptation; used to justify consuming the existing Chronos LoRA fine-tune path instead of full-model retraining. |
| Nguyen and Nguyen, *PhoBERT: Pre-trained language models for Vietnamese* (2020) | Paper | Vietnamese-language representation; used to justify keeping PhoBERT as the sentiment branch for Vietnamese financial text. |
| `dangvantuan/vietnamese-embedding` | Model card | 768-dim Vietnamese sentence embeddings (PhoBERT-based); the actual encoder used in `news_encoder.py` for the text embedding component of the 773-dim hybrid. |
| LangGraph overview documentation | Production documentation | Durable, stateful, traceable orchestration for the new multi-agent control plane. |
| LangSmith documentation | Production documentation | Tracing, debugging, and evaluation of node outputs and explanation payloads. |
| Backtrader commission documentation | Production documentation | Commission, margin, and broker-accounting concepts for future backtesting. |
| Backtrader slippage documentation | Production documentation | Slippage rules, fill behavior, and execution-price realism for future backtesting. |

## 9. Note on Referenced Material

This guide references architecture ideas from papers and production systems, but the implementation target is the current repository. The next step should adapt those references to the existing code paths above, not copy an external architecture verbatim.