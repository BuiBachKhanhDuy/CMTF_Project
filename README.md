# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with **Vietnamese financial news embeddings** through a FiLM-conditioned fusion architecture, benchmarked against Amazon Chronos foundation model baselines.

## Clone Requirements

This repository includes Phase 2 model checkpoints under `outputs/phase2/latest/`.
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
├── pipeline.py                  # Entry point: data ingestion pipeline
├── run_model_benchmark.py       # Entry point: model training & evaluation
├── run_ab_benchmark.py          # Entry point: multiagent A/B test + plots
├── run_sentiment_benchmark.py   # Entry point: PhoBERT sentiment training
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
│   │   ├── baseline_models.py   # LSTM, RF, Fine-tuned Chronos baselines
│   │   ├── baseline_hpo.py      # Optuna HPO for baseline models
│   │   ├── chronos_encoder.py   # Zero-shot + Chronos embeddings
│   │   ├── cnn_lstm_cmtf.py     # CNN-LSTM + cross-modal news fusion
│   │   └── plots.py             # A/B benchmark visualizations
│   ├── sentiment/               # Vietnamese news sentiment (PhoBERT)
│   │   ├── modeling.py          # PhoBERT + Custom Transformer architectures
│   │   ├── training.py          # Sentiment model training loop
│   │   ├── inference.py         # Deployed sentiment scorer
│   │   └── handoff.py           # Phase 2 → pipeline artifact handoff
│   └── multiagent/              # LangGraph multi-agent inference system
│       ├── graph.py             # StateGraph topology + run_graph()
│       ├── config.py            # All tunable parameters
│       ├── state.py             # Typed shared state definition
│       ├── loaders.py           # Lazy model artifact loading
│       ├── reflection.py        # Offline policy updates
│       └── agents/              # 7 specialized agent nodes
├── tests/                       # 85 unit tests (4 skipped smoke tests)
├── results/                     # Benchmark outputs (CSV + figures)
└── report.md                    # Full technical report
```

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

See [report.md](report.md) for full methodology, results, and references.

## License

Academic use only. See report for citations.
