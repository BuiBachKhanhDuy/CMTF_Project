# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with **Vietnamese financial news embeddings** through a FiLM-conditioned fusion architecture, benchmarked against Amazon Chronos foundation model baselines.

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
python run_chronos_benchmark.py

# 5a. Run specific stages
python run_chronos_benchmark.py --stage hpo    # Optuna HPO only
python run_chronos_benchmark.py --stage cmtf   # Retrain CMTF only
python run_chronos_benchmark.py --stage plot   # Regenerate plots
```

## Project Structure

```
├── pipeline.py                  # Entry point: data ingestion pipeline
├── run_chronos_benchmark.py     # Entry point: benchmark experiments
├── src/
│   ├── pipeline/                # Data ingestion & preprocessing
│   │   ├── orchestrator.py      # End-to-end pipeline orchestration
│   │   ├── data_fetcher.py      # vnstock API + multi-source news
│   │   ├── news_scraper.py      # CafeF + VnExpress + Vietstock scraping
│   │   ├── temporal_aligner.py  # Leakage-free news → bar assignment
│   │   ├── feature_engineer.py  # 15 technical indicators + normalization
│   │   ├── news_encoder.py      # PhoBERT → 768-dim embeddings
│   │   └── dataset_builder.py   # PyTorch Dataset with walk-forward splits
│   └── benchmark/               # Chronos experiments & evaluation
│       ├── metrics.py           # MAE, RMSE, DA%, Sharpe, IC, F1
│       ├── chronos_market.py    # Zero-shot + linear probe
│       └── chronos_cmtf.py      # FiLM + GRN fusion head
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
| 2 | Chronos Linear-Probe | Ridge regression on Chronos embeddings + tabular features |
| 3 | Chronos + CMTF | FiLM + GRN fusion of market + news embeddings |

## Key Results (20-Day Horizon, VCB + BID)

| Model | DA% | Sharpe | F1 | IC |
|-------|-----|--------|----|----|
| Chronos Zero-Shot | 46.8 | −0.32 | 0.48 | −0.13 |
| Chronos Linear-Probe | 56.6 | 1.10 | 0.55 | 0.25 |
| **Chronos + CMTF** | **62.9** | **0.82** | **0.52** | **0.48** |

CMTF achieves the target ordering: ZeroShot < LinearProbe < CMTF across DA% and IC at the 20-day horizon.

See [report.md](report.md) for full methodology, results, and references.

## License

Academic use only. See report for citations.
