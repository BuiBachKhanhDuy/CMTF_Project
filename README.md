# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with **Vietnamese financial news embeddings** through a FiLM-conditioned fusion architecture, benchmarked against Amazon Chronos foundation model baselines.

## Prerequisites

Install these on the target machine **before** cloning:

| Requirement | Why | Notes |
|---|---|---|
| **Python 3.11** (3.11.x) | Matches the pinned toolchain (`torch`, `chronos-forecasting`, `sentence-transformers` wheels). | 3.12+/3.14 may fail to resolve wheels. Verify with `python --version`. |
| **Git + Git LFS** | Sentiment checkpoints (`phobert.pt` ≈ 542 MB) are stored via LFS. | `git lfs install` once per machine. |
| **~4 GB free disk** | Pip packages + runtime caches (`cache/` grows to several GB). | |
| **Network access** | The data pipeline fetches live OHLCV (vnstock/VCI) and scrapes Vietnamese news (CafeF/VnExpress/Vietstock/Google News). | Vietnam-facing endpoints; a proxy may block them. |
| **Ollama + `qwen2.5:7b-instruct`** (optional) | Only for real-LLM narration in `chat.py --llm` and the H3/H4/H5 LLM evals. | Server at `http://localhost:11434`. The default (LLM-free) path needs none. |

> Reproducibility note: a small **offline demo bundle** IS committed (deploy checkpoints,
> frozen predictions, news index, and the 6 current-champion dataset parquets under
> `cache/`), so the real-time multi-agent chatbot (`python chat.py`) runs immediately after
> Setup — no rebuild, no network. Reproducing the *full* research (all benchmarks/ablations)
> still rebuilds the heavy caches from the pipeline (Steps 1–6), which re-fetches live data,
> so regenerated numbers track — but are not bit-for-bit identical with — the committed
> `results/` tables. See [Reproducibility & Caches](#reproducibility--caches).

## Setup

```powershell
# 1. Enable Git LFS (once per machine), then clone
git lfs install
git clone <repo-url>
cd ChatbotThesis
git lfs pull                       # pulls phobert.pt / custom_transformer.pt

# 2. Create & activate a Python 3.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the install (fast, offline — mocked HTTP)
pytest -v
```

## Run the Live Demo Immediately (no rebuild, no network)

The offline demo bundle is committed, so straight after Setup you can run the real-time
multi-agent chatbot on the cached research book:

```powershell
python -m src.multiagent check-deploy --all   # expect 1D/5D/20D READY
python chat.py                                 # interactive; type: VCB  then  VCB 2025-08-13
```

`check-deploy` verifies the gate policies, frozen predictions, and deploy checkpoints are all
present. In `chat.py`, an in-book date answers instantly from a real forward pass of the
committed champion checkpoints. For real-LLM narration, start Ollama first
(`ollama pull qwen2.5:7b-instruct`) and run `python chat.py --llm`. The full end-to-end
research run below is only needed to regenerate benchmarks/ablations from scratch.

## End-to-End Run

Run the phases in order. Each step writes caches consumed by the next, so a first full run
is sequential; later runs reuse `cache/` and are fast. Every command assumes the activated
`.venv`. On Windows, prefix env-vars with `$env:` (shown inline).

### Step 1 — Sentiment encoder (Phase 1/2)

Trains the Vietnamese sentiment models. Checkpoints already ship under
`outputs/sentiment/latest/` (via LFS), so this step is optional unless you want to retrain.

```powershell
python run_sentiment_benchmark.py --variant both      # PhoBERT + custom transformer
python run_sentiment_benchmark.py --variant phobert --skip-dataset-download
```

### Step 2 — Data pipeline (Phase 1)

Fetches OHLCV + news for the configured symbols/date range and builds the feature dataset
into `cache/dataset/`. **Requires network.** Throttle vnstock with
`$env:VNSTOCK_RATE_LIMIT_PER_MIN="16"` if the API rate-limits you.

```powershell
python pipeline.py
```

### Step 3 — Baseline + CMTF benchmark (Phase 2)

Trains/evaluates Chronos, LSTM, RF, CNN-LSTM, and CMTF; writes `results/*.csv` + figures.

```powershell
python run_model_benchmark.py                          # all stages, all horizons
python run_model_benchmark.py --horizons 1 5 20        # pick horizons
python run_model_benchmark.py --stage hpo              # Optuna HPO only
python run_model_benchmark.py --stage predict          # train/predict only
python run_model_benchmark.py --stage plot             # regenerate plots only
python run_model_benchmark.py --skip-chronos           # skip the slow zero-shot baseline
# valid --stage values: data | predict | hpo | plot
```

### Step 4 — Fusion ablations (Phase 3)

```powershell
# Fusion-strategy comparison table + gated decision policy
python run_ablation_benchmark.py --horizons 1 5 20 --gate --gate-coverage 0.25
python run_ablation_benchmark.py --stage plot          # regenerate ablation figures

# Config-driven component-ablation registry (all cells, 3 seeds)
python run_ablation_registry.py --cells all --horizons 1 5 20 --seeds 1 42 123
```

### Step 5 — Train + persist deploy checkpoints (required for live inference)

The multi-agent system serves predictions from `cache/deploy_models/`. Persist the champion
checkpoints for every horizon (set `SAVE_DEPLOY_MODEL=1`):

```powershell
$env:SAVE_DEPLOY_MODEL="1"
python run_ablation_registry.py --cells 0 --horizons 1 5 20 --seeds 1 42 123
$env:SAVE_DEPLOY_MODEL=""

# Freeze the validation-calibrated gate policy per horizon → results/gate_policies/VN_{H}d.json
python -m src.multiagent calibrate --horizon 1
python -m src.multiagent calibrate --horizon 5
python -m src.multiagent calibrate --horizon 20

# Verify the graph has everything it needs (gate policy + frozen preds + deploy checkpoints)
python -m src.multiagent check-deploy --all
```

### Step 6 — Multi-agent evaluation & A/B tests (Phase 4)

```powershell
python run_ab_benchmark.py --symbols VCB BID --horizons 1 5 20   # MAS vs CMTF-only A/B
python -m src.multiagent eval --horizon 5                        # A0–A5 agent ladder
python -m src.multiagent h3 --mode forecaster --horizon 5        # MAS vs plain LLM
python -m src.multiagent metalabel-eval --horizon 5
python -m src.multiagent h4-interaction-eval --horizon 5
python -m src.multiagent h5-reasoning-eval --horizon 5
```

### Step 7 — Real-time chatbot & live prediction (Phase 5)

```powershell
# Interactive CLI (deterministic, grounded, no LLM required)
python chat.py

# Interactive CLI with real LLM narration (needs Ollama running + model pulled)
#   ollama serve   &&   ollama pull qwen2.5:7b-instruct
python chat.py --llm

# One-shot orchestrator-routed live prediction, with a full node trace
python -m src.multiagent predict --symbol VCB --cutoff 2025-08-13 --horizon 5 --trace
python -m src.multiagent rank --symbols VCB,BID,CTG --horizon 5 --cutoff 2025-08-13
python -m src.multiagent research --symbol VCB --cutoff 2025-08-13

# End-to-end product demo report
python -m tools.e2e_demo
```

Inside `chat.py`, type e.g. `VCB`, `VCB 2025-08-13`, `BID có nên mua 5 ngày tới`,
`rank VCB,BID,CTG 2025-08-13`, or `help` / `symbols` / `quit`. Dates inside the cached
research book answer instantly; a date outside it triggers a real live forward pass
(fetches OHLCV + news for the universe — can take minutes on a cold process).

## Reproducibility & Caches

- **Git-tracked (survive a clone):** all source (`src/`, `run_*.py`, `chat.py`), `tests/`,
  `results/` tables + figures + all gate policies (`VN_{1,5,20}d.json`), sentiment
  checkpoints (`outputs/sentiment/latest/`, via LFS), and the **offline demo bundle** under
  `cache/`: `deploy_models/` (champion checkpoints, all 3 horizons), `predictions/` (frozen
  book), `news/` (news index), and the 6 current-champion `dataset/*.parquet` files.
- **Git-ignored (rebuilt locally):** the heavy regenerable caches — `cache/encoders/`,
  `cache/dataset_splits/`, `cache/embeddings/`, stale `cache/dataset/*.parquet`,
  `artifacts/`, `data/external/`, and `*.zip`. These are only needed to regenerate the full
  benchmarks/ablations (Steps 1–6); the live demo does not need them.
- **Determinism:** benchmarks seed all RNGs (`--seeds`), but the data pipeline fetches
  live web data, so re-scraped news changes over time — regenerated metrics track the
  committed `results/` closely rather than reproducing them bit-for-bit.

## Tests

```powershell
pytest -v                                              # full suite (mocked HTTP, offline)
pytest tests/test_news_scraper_parsing.py -v           # parser-only crawler tests
$env:RUN_LIVE_SCRAPER="1"; pytest -m smoke tests/test_news_scraper_smoke.py -v  # live crawler
```

## Project Structure

```
├── pipeline.py                  # Entry point (Phase 1): data ingestion pipeline
├── run_model_benchmark.py       # Entry point (Phase 2): model training & evaluation
├── run_sentiment_benchmark.py   # Entry point (Phase 1/2): PhoBERT sentiment training
├── run_ablation_benchmark.py    # Entry point (Phase 3): fusion-strategy ablation
├── run_ablation_registry.py     # Entry point (Phase 3): component-ablation registry
├── run_ab_benchmark.py          # Entry point (Phase 4): multiagent A/B test + plots
├── chat.py                      # Entry point (Phase 5): interactive chatbot CLI
├── src/
│   ├── pipeline/                # Data ingestion & preprocessing
│   │   ├── orchestrator.py      # End-to-end pipeline orchestration
│   │   ├── data_fetcher.py      # vnstock API + multi-source news
│   │   ├── news_scraper.py      # CafeF + VnExpress + Vietstock scraping
│   │   ├── temporal_aligner.py  # Leakage-free news → bar assignment
│   │   ├── feature_engineer.py  # 15 technical indicators + normalization
│   │   ├── news_encoder.py      # PhoBERT → 768-dim embeddings
│   │   └── dataset_builder.py   # PyTorch Dataset with walk-forward splits
│   ├── benchmark/               # Model implementations & evaluation
│   │   ├── metrics.py           # MAE, RMSE, DA%, Sharpe, IC, F1 + composite
│   │   ├── baseline_models.py   # LSTM, RF, Chronos, CNN-LSTM, CNN-LSTM CMTF
│   │   ├── baseline_hpo.py      # Optuna HPO for baseline models
│   │   ├── chronos_encoder.py   # Zero-shot + Chronos embeddings
│   │   ├── fusion_wrappers.py   # News fusion heads (early/late/hybrid/residual)
│   │   ├── ablation_runner.py   # Phase 3 ablation harness
│   │   ├── calibration.py       # Prediction calibration checks
│   │   ├── cross_sectional_ic.py # Cross-sectional IC evaluation
│   │   └── plots.py             # A/B benchmark visualizations
│   ├── sentiment/               # Vietnamese news sentiment (PhoBERT)
│   │   ├── modeling.py          # PhoBERT + Custom Transformer architectures
│   │   ├── training.py          # Sentiment model training loop
│   │   ├── inference.py         # Deployed sentiment scorer
│   │   └── handoff.py           # Sentiment encoder → pipeline artifact handoff
│   └── multiagent/              # LangGraph multi-agent inference system (Phase 4/5)
│       ├── graph.py             # StateGraph topology + run_graph()
│       ├── config.py            # All tunable parameters
│       ├── state.py             # Typed shared state definition
│       ├── loaders.py           # Lazy model artifact loading
│       ├── live_inference.py    # Real-time inference for the chatbot
│       ├── frozen_predictions.py, gate_io.py, guards.py, trace.py, news_data.py
│       ├── eval_ladder.py, metalabel_eval.py, h3_faithfulness.py, improved_ensemble.py
│       └── agents/              # 11 specialized agent nodes (market, news, predict,
│                                 # research, rank, risk, gate, metalabel, critic,
│                                 # narrator, orchestrator)
├── tools/
│   └── e2e_demo.py              # Phase 5 end-to-end product demo/report generator
├── tests/                       # Unit tests
├── results/                     # Benchmark outputs (CSV + figures)
└── docs/reference/              # Stable reference docs (see below)
```

### Documentation

Thesis-facing writeups live in [research/](research/) (tracked, one folder per phase) — start
there on a fresh clone. The `docs/` reference notes below are **git-ignored** (local only):

- [research/](research/) — phase-by-phase research documents (Phases 1–5), **tracked**
- [docs/reference/CMTF_FUSION_FINDINGS.md](docs/reference/CMTF_FUSION_FINDINGS.md) — Phase 2/3 fusion research findings *(local)*
- [docs/reference/RESULTS_IMPROVEMENT_LEVERS.md](docs/reference/RESULTS_IMPROVEMENT_LEVERS.md) — Phase 3 improvement-lever experiments *(local)*
- [docs/reference/CACHING_GUIDE.md](docs/reference/CACHING_GUIDE.md) — cache layout for ablation/benchmark runs *(local)*
- [docs/reference/RELATED_WORK_AND_RESEARCH_PLAN.md](docs/reference/RELATED_WORK_AND_RESEARCH_PLAN.md) — literature review & research agenda *(local)*
- [docs/reference/MULTIAGENT_SYSTEM.md](docs/reference/MULTIAGENT_SYSTEM.md) — Phase 4/5 multiagent architecture, workflow & results *(local)*
- [docs/reference/MULTIAGENT_REDESIGN_PLAN.md](docs/reference/MULTIAGENT_REDESIGN_PLAN.md) — Phase 4 design rationale / decision log *(local)*
- [docs/report.md](docs/report.md) — full thesis document (LaTeX source, local/untracked)

## Stack

- **Python 3.11** · PyTorch · Amazon Chronos T5-Small · PhoBERT
- **Data:** vnstock v3.x (OHLCV) · CafeF + VnExpress + Vietstock (news)
- **Symbols:** VCB, BID — Vietnamese banking large-caps
- **Date Range:** 2022-01-01 to 2026-03-31

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Chronos Zero-Shot | Foundation model predicts directly (no training) |
| 2 | CMTF | LoRA-tuned Chronos backbone with FiLM + GRN fusion of market + news embeddings |
| 3 | LSTM Baseline | Sequence model baseline on close windows |
| 4 | LSTM + CMTF | LSTM embeddings fused with CMTF (residual-gated) |
| 5 | Random Forest Baseline | Tabular baseline over engineered market features |
| 6 | Chronos Fine-Tuned (LoRA) | Market-only Chronos baseline fine-tuned with LoRA |

## Key Results (20-Day Horizon, VCB + BID)

| Model | DA% | Sharpe | F1 | IC |
|-------|-----|--------|----|----|
| Chronos Zero-Shot | 46.8 | -0.32 | 0.48 | -0.13 |
| **CMTF** | **68.7** | **1.19** | **0.61** | **0.46** |
| LSTM Baseline | 55.7 | 0.32 | 0.63 | 0.28 |
| LSTM + CMTF | 57.8 | 0.81 | 0.65 | 0.28 |
| Random Forest Baseline | 52.9 | 0.02 | 0.46 | 0.01 |
| Chronos Fine-Tuned (LoRA) | 47.5 | 0.03 | 0.64 | -0.15 |

CMTF is the strongest model on the 20-day horizon by RMSE, DA%, Sharpe, IC, and composite score.

See [docs/reference/CMTF_FUSION_FINDINGS.md](docs/reference/CMTF_FUSION_FINDINGS.md) for current methodology and results, and [docs/report.md](docs/report.md) for the full thesis writeup.

## License

Academic use only. See report for citations.
