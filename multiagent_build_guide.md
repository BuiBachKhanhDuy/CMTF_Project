# Multi-Agent Build Guide for the Next Step

## 1. Purpose of This Document

This document is the implementation plan for the next step of the project. It is not a generic architecture note. It is a code-grounded guide for turning the current repository into:

1. a LangGraph-based multi-agent system whose **deterministic critic agents** are allowed to *modify* the action and position size produced by the base CMTF v8 predictor, so the multi-agent layer has a real lever to outperform plain CMTF
2. an **A/B evaluation harness** that compares CMTF-only vs CMTF + multi-agent policy on the existing test split using Sharpe of a simple long/short policy
3. a **broker-style backtesting layer** with commission, slippage, and equity-curve accounting, reusing the same multi-agent graph as its signal source

The plan below is intentionally constrained by the code that already exists in the repo and by these decisions:

- **LLM provider**: local Ollama running `qwen2.5:7b`. Used only inside the Explanation Agent to rewrite a structured evidence dict into Vietnamese natural language. Critics are deterministic Python.
- **News source at inference time**: cached parquet only, filtered by `published_at <= cutoff`. No live scraping, no injected article lists in the first cut.
- **Smoke target**: VCB, horizon=1d. Full benchmark parity (all symbols × {1d, 5d, 20d}) only after the smoke graph is green.
- **Performance claim**: multi-agent layer outperforms plain CMTF on Sharpe via three deterministic levers — Risk/Regime Critic, News-Quality Critic, Ensemble-Disagreement Gate. No LLM-driven debate.

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

- Do not retrain PhoBERT, Chronos LoRA, or CMTF v8.
- Do not introduce a second fusion architecture.
- Do not run live news scraping at inference time.
- Do not let the LLM modify predictions, actions, or position sizes. The LLM only rewrites a deterministic evidence dict into Vietnamese prose.
- Do not mix forecast metrics and portfolio metrics into one report.
- Do not collapse the A/B harness and the broker backtester into one module. They have different goals (hypothesis testing vs realistic execution simulation) and different friction assumptions.

## 4. Phase 1: Multi-Agent Inference Graph

### 4.1 Objective

Produce, for one `(symbol, prediction_time, horizon)` request, a structured object containing:

1. `final_pred` — forward-return forecast from CMTF v8 ensemble
2. `action` — `long`, `short`, or `flat` after critic overrides
3. `position_scale` — float in `[0, 1]` set by the Risk Critic
4. `explanation` — Vietnamese natural-language rationale grounded in attention weights, sentiment trace, and any critic overrides

### 4.2 Graph Topology

The graph is **linear with a critic fan-in** before the decision step. Critic ordering matters: critics run after Fusion (so they can see `baseline_pred`, `news_residual`, ensemble seed predictions) and before Decision (so they can override action and scale).

```
Supervisor
    ↓
Market Agent
    ↓
News Agent
    ↓
Fusion Agent  (loads CMTF v8 ensemble × 3 seeds, returns baseline_pred, final_pred, attn, news_weight, seed_preds)
    ↓
┌───────────────────────────────────────────────────────────┐
│  Risk/Regime Critic   News-Quality Critic   Disagreement  │
│  (deterministic)      (deterministic)       Gate          │
└───────────────────────────────────────────────────────────┘
    ↓
Decision Agent  (applies threshold policy AFTER critics; combines overrides)
    ↓
Explanation Agent  (deterministic evidence dict → Ollama qwen2.5:7b → Vietnamese prose)
```

The three critics may run sequentially in the first implementation. LangGraph parallel branches can be added later if profiling shows they matter.

### 4.3 Agent Contracts

**Supervisor.** Validates `symbol`, `prediction_time`, `target_horizon_days ∈ {1, 5, 20}`, `sequence_len=30`. Sets `data_cutoff = prediction_time`. Initializes empty critic-override fields.

**Market Agent.** Reuses `src/pipeline/orchestrator.py` to produce `close_window (30,)`, `market_window (30, 23)`, `market_tabular (23,)`, `token_ids`, `attention_mask` for one cutoff. Must not call the orchestrator's full multi-symbol path; expose a single-symbol single-cutoff helper in `src/pipeline/orchestrator.py` if one does not exist.

**News Agent.** Reads cached news parquet only (path resolved via `src/multiagent/config.py`). Filters by `published_at <= cutoff` and by symbol. Calls existing PhoBERT inference from `src/phase2/inference.py` and existing `NewsEncoder.encode_window()`. Returns `news_emb (30, 773)`, `news_mask (30,)`, `articles` (list of dicts with `title`, `published_at`, `sentiment_score`, `bar_index`), and the 6 sentiment scalar features.

**Fusion Agent.** Loads the LoRA backbone once and the 3 CMTF v8 ensemble checkpoints once via the lazy loader. Calls a new method `ChronosCMTFPredictor.predict_with_explanation(...)` that returns:

```python
{
    "baseline_pred": float,        # market-only path, news_mask all False
    "final_pred": float,           # mean over 3 seeds
    "seed_preds": list[float],     # 3 floats for the disagreement gate
    "news_residual": float,        # final_pred - baseline_pred
    "attn_weights": np.ndarray,    # shape (30,), mean over 3 seeds
    "news_weight": float,          # mean of fusion.news_weight over 3 seeds
}
```

**Risk/Regime Critic (deterministic).** Inputs: `market_window`, `close_window`, optional VN-Index features already in the tabular row. Computes:

- realized 20-day volatility from `close_window`
- maximum drawdown over the window
- VN-Index z-score (if available in the tabular features)

Output written to state: `regime_flags` dict and `position_scale_regime ∈ [0, 1]`. Rules (configurable in `config.py`):

- if 20d vol > `vol_high_pct` → `position_scale_regime = 0.5`
- if drawdown > `dd_max_pct` → `position_scale_regime = 0.0` (force flat)
- else `position_scale_regime = 1.0`

**News-Quality Critic (deterministic).** Inputs: `articles`, `news_mask`, `news_residual`, `news_weight`. Computes:

- staleness: fraction of articles older than `staleness_days`
- coverage: number of bars with at least one article
- sentiment dispersion: std of PhoBERT title scores

Output: `news_quality_flags` dict and `news_residual_scale ∈ [0, 1]`. Rules:

- if coverage < `min_news_bars` (e.g. 3) → `news_residual_scale = 0.0` (ignore news contribution; effectively use baseline)
- if staleness > `max_stale_frac` → `news_residual_scale = 0.5`
- else `news_residual_scale = 1.0`

The agent computes `final_pred_adjusted = baseline_pred + news_residual_scale × news_residual` and writes it to state alongside the original `final_pred`. It does **not** overwrite `final_pred`.

**Ensemble-Disagreement Gate (deterministic).** Inputs: `seed_preds`. If the sign of any of the 3 seed predictions differs from the mean sign → `disagreement_force_flat = True`. Otherwise False.

**Decision Agent.** Reads `final_pred_adjusted`, `position_scale_regime`, `disagreement_force_flat`, plus thresholds from `config.py`. Logic:

```
if disagreement_force_flat or position_scale_regime == 0.0:
    action = "flat"; position_scale = 0.0
elif final_pred_adjusted >=  buy_threshold:
    action = "long";  position_scale = position_scale_regime
elif final_pred_adjusted <= -sell_threshold:
    action = "short"; position_scale = position_scale_regime
else:
    action = "flat"; position_scale = 0.0
```

Thresholds and friction-relevant constants live in `config.py`, not in checkpoints, so the policy can be retuned without retraining.

**Explanation Agent.** Builds a deterministic Vietnamese evidence dict from state: top-3 attended bars (date + dominant article titles + PhoBERT scores), `baseline_pred`, `news_residual`, applied `news_residual_scale`, `regime_flags`, `disagreement_force_flat`, final `action`, `position_scale`. The Ollama call uses a fixed system prompt that forbids invented facts, with `temperature=0.2`. If Ollama is unreachable, fall back to a Jinja2 Vietnamese template — never block prediction on the LLM.

### 4.4 Shared State (`state.py`)

Single `TypedDict` named `MultiAgentState`:

| Group | Keys |
|---|---|
| Request | `symbol`, `prediction_time`, `target_horizon_days`, `sequence_len` |
| Market | `close_window`, `market_window`, `market_tabular`, `token_ids`, `attention_mask` |
| News | `articles`, `title_scores`, `sentiment_features`, `news_emb`, `news_mask` |
| Fusion | `baseline_pred`, `final_pred`, `seed_preds`, `news_residual`, `attn_weights`, `news_weight` |
| Critics | `regime_flags`, `position_scale_regime`, `news_quality_flags`, `news_residual_scale`, `final_pred_adjusted`, `disagreement_force_flat` |
| Decision | `action`, `position_scale` |
| Explanation | `evidence_dict`, `explanation_text_vi` |
| Audit | `data_cutoff`, `artifact_versions`, `errors`, `warnings`, `node_timings` |

### 4.5 Lazy Artifact Loader (`loaders.py`)

Single module-level cache keyed by `(symbol, horizon)`. Public API:

```python
get_phobert_bundle() -> PhoBERTBundle              # singleton, no key
get_lora_backbone(symbol, horizon) -> nn.Module    # cached
get_cmtf_ensemble(symbol, horizon) -> list[ChronosCMTFPredictor]  # 3 seeds
get_news_encoder() -> NewsEncoder                  # singleton
```

All loaders raise `ArtifactMissingError` with the resolved cache path on failure. Tests inject a fake loader through a `set_loader_override()` hook so node tests do not hit disk.

### 4.6 Implementation Order (one-shot friendly)

Build files in this exact order. Each step's tests must pass before moving on.

1. `src/multiagent/__init__.py`, `state.py`, `config.py`. Tests: state shape, config keys present.
2. `src/multiagent/loaders.py` with the override hook. Tests: override hook returns fake objects; missing artifact raises `ArtifactMissingError`.
3. **Patch** `src/benchmark/chronos_cmtf.py` to add `ChronosCMTFPredictor.predict_with_explanation()`. Tests: shape and zero-news parity (when `news_mask` all False, `news_residual ≈ 0`).
4. **Patch** `src/pipeline/orchestrator.py` to expose `prepare_single_cutoff(symbol, cutoff, sequence_len=30)` returning the market dict plus tokenized inputs. Tests: deterministic output for fixed cutoff; cutoff strictly enforced.
5. `market_agent.py`, `news_agent.py`. Tests with override loader: state keys populated with correct shapes; news with zero rows yields `news_mask` all False.
6. `fusion_agent.py`. Tests: ensemble averaging, `seed_preds` length 3, attention shape `(30,)`.
7. `critics/regime_critic.py`, `critics/news_quality_critic.py`, `critics/disagreement_gate.py`. Pure-function tests with synthetic state.
8. `decision_agent.py`. Truth-table tests covering every override combination.
9. `explanation_agent.py` with Ollama disabled in tests; Jinja2 fallback path tested.
10. `graph.py` wiring all nodes. Smoke test: VCB, cutoff = last available bar in test split, horizon=1d, returns a populated state.
11. `cli.py` with `predict` and `batch-predict` subcommands.

### 4.7 Acceptance Criteria for Phase 1

- One graph call returns a fully populated `MultiAgentState` for VCB, horizon=1d.
- All 3 ensemble seed checkpoints loaded once and reused across multiple cutoffs in the same process.
- Zero-news cutoffs produce `final_pred ≈ baseline_pred` and `news_residual_scale = 0` from the news-quality critic.
- Ollama unreachable does not break inference; Jinja2 fallback emits a Vietnamese explanation.
- Each node writes to disjoint state keys (no node overwrites another's outputs).
- `python -m src.multiagent.cli predict --symbol VCB --cutoff 2025-03-31 --horizon 1` succeeds end to end.

## 5. Phase 2a: A/B Evaluation Harness (lightweight, no broker)

### 5.1 Objective

Test the thesis claim *"multi-agent layer improves performance over plain CMTF"* on the existing walk-forward test split, using a simple long/short policy. No commission, no slippage, no portfolio ledger — those belong to Phase 2b.

### 5.2 Comparison Arms

| Arm | Signal source | Action policy |
|---|---|---|
| `cmtf_only` | `final_pred` from CMTF v8 ensemble (no critics) | Threshold from `config.py`, `position_scale = 1` always |
| `multiagent` | Multi-agent graph (`final_pred_adjusted`, critic overrides) | Decision Agent output (`action`, `position_scale`) |

Both arms run on identical `(symbol, cutoff, horizon)` inputs and the same realized forward returns from the test split.

### 5.3 Metric

Primary: **Sharpe of the daily PnL stream** of a 1-bar holding-period long/short policy:

```
pnl_t = position_scale_t × sign_action_t × realized_return_{t→t+horizon}
sharpe = mean(pnl) / std(pnl) × sqrt(252 / horizon)
```

Secondary (reported but not used to declare a winner): DA%, hit rate, mean trade PnL, fraction of bars set to flat by each critic.

### 5.4 Module Layout

```
src/multiagent/eval/
    __init__.py
    ab_runner.py       # iterates test cutoffs, runs both arms, writes per-arm CSV
    ab_metrics.py      # Sharpe + secondary metrics from per-arm CSV
    ab_report.py       # CLI: prints comparison table; writes results/ab/{run_id}.json
```

### 5.5 Acceptance Criteria

- Both arms run on the same cutoff list, asserted equal.
- `results/ab/{run_id}.json` contains per-arm Sharpe, DA%, and per-critic activation counts.
- Smoke run: VCB × 1d. Full run: all symbols × {1d, 5d, 20d}.
- Multi-agent arm Sharpe ≥ CMTF-only Sharpe in at least one (symbol, horizon) cell — otherwise the thesis claim must be revisited before Phase 2b.

## 6. Phase 2b: Broker-Style Backtesting

Phase 2b is built only after Phase 2a shows a credible signal. It reuses the same multi-agent graph as the signal source.

### 6.1 Engine Design

1. **Input stream**: same chronological cutoffs as Phase 2a; signal generated at bar close $t$, executed at next open $t+1$.
2. **Friction**: commission and slippage as separate configurable parameters per (symbol, side); optional short borrow cost.
3. **Portfolio ledger**: cash, exposure, position direction, position size (driven by `position_scale` from the graph), entry price, realized PnL, unrealized PnL, equity curve.
4. **Audit trail**: each executed order stores the `evidence_dict` and `explanation_text_vi` from the graph.

### 6.2 Module Layout

```
src/backtesting/
    __init__.py
    engine.py          # chronological loop; pulls signals from src.multiagent.graph
    policy.py          # threshold + position_scale wiring (re-uses Decision Agent output)
    ledger.py          # positions, cash, equity curve
    friction.py        # commission and slippage models
    metrics.py         # Sharpe, Sortino, max drawdown, turnover, hit rate, cost drag
```

### 6.3 Validation Sequence

1. Reproduce Phase 2a Sharpe with zero commission and zero slippage. The two numbers must match within numerical tolerance — if not, there is a leakage or accounting bug.
2. Turn commission on, confirm PnL drops monotonically with commission rate.
3. Turn slippage on, confirm fills worsen.
4. Confirm zero-news cutoffs follow the baseline path with no critic activations from `news_quality_critic`.

### 6.4 Output Locations

| Type | Path |
|---|---|
| Equity curves and summary tables | `results/backtests/{run_id}/` |
| Trade logs with explanation payloads | `artifacts/backtests/{run_id}/` |
| Reusable simulation intermediates | `cache/backtests/{run_id}/` |

### 6.5 Acceptance Criteria

- No leakage: the engine never reads bars at index `> t` when generating the signal for $t$.
- Phase 2a and Phase 2b agree on Sharpe under zero friction.
- Commission and slippage are configurable, tested in isolation, and reported in the summary.
- Forecast metrics and portfolio metrics live in separate JSON files under `results/backtests/{run_id}/`.
- Every executed order can be traced to a `MultiAgentState` snapshot.

## 7. Suggested Code Layout

```
src/multiagent/
    __init__.py
    state.py
    config.py                # thresholds, critic params, ollama settings, news cache path
    loaders.py               # lazy artifact cache + test override hook
    graph.py                 # LangGraph wiring
    cli.py                   # predict / batch-predict
    market_agent.py
    news_agent.py
    fusion_agent.py
    decision_agent.py
    explanation_agent.py
    critics/
        __init__.py
        regime_critic.py
        news_quality_critic.py
        disagreement_gate.py
    eval/
        __init__.py
        ab_runner.py
        ab_metrics.py
        ab_report.py
src/backtesting/
    __init__.py
    engine.py
    policy.py
    ledger.py
    friction.py
    metrics.py
tests/multiagent/
    test_state.py
    test_loaders.py
    test_predict_with_explanation.py     # patches chronos_cmtf
    test_market_agent.py
    test_news_agent.py
    test_fusion_agent.py
    test_critics.py
    test_decision_agent.py
    test_explanation_agent.py            # ollama mocked
    test_graph_smoke.py                  # VCB 1d end-to-end
    test_ab_runner.py
tests/backtesting/
    test_engine_no_friction.py
    test_engine_with_commission.py
    test_engine_with_slippage.py
```

## 8. Immediate Next Actions

1. Add to `requirements.txt`: `langgraph`, `langchain-core`, `langchain-ollama`, `langsmith`, `jinja2`. Install `qwen2.5:7b` locally via `ollama pull qwen2.5:7b`.
2. Build `src/multiagent/state.py` and `config.py` (Section 4.4).
3. Build `src/multiagent/loaders.py` with the test override hook (Section 4.5).
4. Patch `ChronosCMTFPredictor.predict_with_explanation()` in `src/benchmark/chronos_cmtf.py` (Section 4.6 step 3).
5. Add `prepare_single_cutoff()` to `src/pipeline/orchestrator.py` (Section 4.6 step 4).
6. Build agents in the order listed in Section 4.6.
7. Wire `graph.py` and run the VCB 1d smoke test.
8. Build the A/B harness in `src/multiagent/eval/` and run Phase 2a on VCB 1d, then full grid.
9. Only after Phase 2a is green, build `src/backtesting/` and run Phase 2b validation.

## 9. References Used for This Plan

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

## 10. Note on Referenced Material

This guide references architecture ideas from papers and production systems, but the implementation target is the current repository. The next step should adapt those references to the existing code paths above, not copy an external architecture verbatim.