# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with **Vietnamese news embeddings** through a cross-attention architecture, benchmarked against Amazon Chronos foundation model baselines.

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

# 5. Run the benchmark (3 experiments × 3 symbols)
python run_chronos_benchmark.py
```

## Project Structure

```
├── pipeline.py                  # Entry point: data ingestion pipeline
├── run_chronos_benchmark.py     # Entry point: benchmark experiments
├── src/
│   ├── pipeline/                # Data ingestion & preprocessing
│   │   ├── orchestrator.py      # End-to-end pipeline orchestration
│   │   ├── data_fetcher.py      # vnstock API + multi-source news
│   │   ├── news_scraper.py      # CafeF + VnExpress web scraping
│   │   ├── temporal_aligner.py  # Leakage-free news → bar assignment
│   │   ├── feature_engineer.py  # 15 technical indicators + normalization
│   │   ├── news_encoder.py      # PhoBERT → 768-dim embeddings
│   │   └── dataset_builder.py   # PyTorch Dataset with walk-forward splits
│   └── benchmark/               # Chronos experiments & evaluation
│       ├── metrics.py           # MAE, RMSE, DA%, Sharpe, IC
│       ├── chronos_market.py    # Zero-shot + linear probe
│       └── chronos_cmtf.py      # Cross-modal fusion head
├── tests/
│   └── test_pipeline.py         # 18 unit tests
├── results/                     # Benchmark outputs (CSV + figures)
└── report.md                    # Full technical report
```

## Stack

- **Python 3.14** · PyTorch · Amazon Chronos T5-Small · PhoBERT
- **Data:** vnstock v3.x (OHLCV) · CafeF + VnExpress (news scraping)
- **Symbols:** VCB, VIC, VHM — Vietnamese large-caps (2022–2024)

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Chronos Zero-Shot | Foundation model predicts directly (no training) |
| 2 | Chronos Linear-Probe | Ridge regression on Chronos encoder embeddings |
| 3 | Chronos + CMTF | Cross-attention fusion of market + news embeddings |

See [report.md](report.md) for full results, analysis, and methodology.

## License

Academic use only. See report for citations.
