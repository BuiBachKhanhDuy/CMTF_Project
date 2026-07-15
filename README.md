# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with **Vietnamese financial news embeddings** through a FiLM-conditioned fusion architecture, benchmarked against Amazon Chronos foundation model baselines.

## Clone Requirements

This repository includes sentiment-encoder model checkpoints under `outputs/sentiment/latest/`.
To clone and pull these files successfully, install and enable Git LFS first:

```powershell
git lfs install
git clone <repo-url>
cd ChatbotThesis
git lfs pull
```

## Quick Start

```powershell
# 1. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest -v

# 3a. Run parser-only crawler tests (mocked HTTP)
pytest tests/test_news_scraper_parsing.py -v

# 3b. Optional live crawler smoke tests
# (PowerShell) $env:RUN_LIVE_SCRAPER="1"
pytest -m smoke tests/test_news_scraper_smoke.py -v

# 4. Run the data pipeline
python pipeline.py

# 5. Run the benchmark
python run_model_benchmark.py

# 5a. Run specific stages
python run_model_benchmark.py --stage hpo    # Optuna HPO only
python run_model_benchmark.py --stage cmtf   # Retrain CMTF only
python run_model_benchmark.py --stage plot   # Regenerate plots
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

- [docs/reference/CMTF_FUSION_FINDINGS.md](docs/reference/CMTF_FUSION_FINDINGS.md) — Phase 2/3 fusion research findings
- [docs/reference/RESULTS_IMPROVEMENT_LEVERS.md](docs/reference/RESULTS_IMPROVEMENT_LEVERS.md) — Phase 3 improvement-lever experiments (evidence trail)
- [docs/reference/CACHING_GUIDE.md](docs/reference/CACHING_GUIDE.md) — cache layout for ablation/benchmark runs
- [docs/reference/RELATED_WORK_AND_RESEARCH_PLAN.md](docs/reference/RELATED_WORK_AND_RESEARCH_PLAN.md) — literature review & research agenda
- [docs/reference/MULTIAGENT_SYSTEM.md](docs/reference/MULTIAGENT_SYSTEM.md) — Phase 4/5 multiagent architecture, workflow & results (canonical)
- [docs/reference/MULTIAGENT_REDESIGN_PLAN.md](docs/reference/MULTIAGENT_REDESIGN_PLAN.md) — Phase 4 design rationale / decision log
- [docs/reference/phase2_benchmark_report_HISTORICAL.md](docs/reference/phase2_benchmark_report_HISTORICAL.md) — superseded Phase 2 snapshot, kept for audit trail only
- [docs/report.md](docs/report.md) — full thesis document (LaTeX source, local/untracked)

## Stack

- **Python 3.14** · PyTorch · Amazon Chronos T5-Small · PhoBERT
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
