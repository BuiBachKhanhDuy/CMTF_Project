# Cross-Modal Temporal Fusion for Vietnamese Bank Stocks

This repository contains an end-to-end research and inference system for forecasting forward returns of Vietnamese bank stocks. It combines OHLCV-derived market features, time-aligned Vietnamese financial news, and sentiment features, then evaluates market-only and news-aware models across 1-, 5-, and 20-trading-day horizons.

The project has five connected parts:

1. Collect and align market data and Vietnamese news.
2. Train a Vietnamese sentiment encoder (optional when using the included artifacts).
3. Benchmark market-only and hybrid forecasting models.
4. Run fusion and component ablations, including validation-calibrated decision gates.
5. Serve grounded single-stock, ranking, and research requests through a LangGraph multi-agent workflow and CLI chatbot.

> This is research software, not investment advice. Forecasts, rankings, and trade-style labels are experimental outputs.

## What the code supports

- **Universe:** `VCB`, `BID`, `CTG`, `TCB`, `MBB`, `ACB`, and `VPB`
- **Horizons:** 1, 5, and 20 trading days
- **Inputs:** daily OHLCV features, technical indicators, scraped financial news, news embeddings, and optional sentiment features
- **Forecast models:** Chronos zero-shot, LSTM, CNN-LSTM, GPT4TS, Random Forest, linear/MLP summary baselines, their hybrid variants, and CMTF configurations
- **Evaluation:** MAE, RMSE, directional accuracy, Sharpe, information coefficient, classification metrics, effective sample size, calibration, coverage diagnostics, and placebo tests
- **Product layer:** deterministic or Ollama-narrated multi-agent predictions, rankings, grounded research digests, trace files, and deployment-readiness checks

## Requirements

- Python 3.11 recommended
- Git and Git LFS when cloning artifacts stored with LFS
- Network access for a fresh market/news build and for downloading model dependencies
- Optional: Ollama with `qwen2.5:7b-instruct` for LLM narration and the LLM-dependent evaluations

The pipeline can use CUDA when PyTorch detects it; CPU execution is supported but full benchmark and ablation runs can be slow.

## Setup

From the repository root:

```powershell
git lfs install
git lfs pull

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not on your PATH, use `.\.venv\Scripts\python.exe` in the commands below. Verify the environment with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Quick start: use the included product artifacts

The repository is configured to retain an offline demo bundle under `cache/` (prediction stores, deploy checkpoints, news index, and selected datasets). Check that all required 1D, 5D, and 20D artifacts are available:

```powershell
.\.venv\Scripts\python.exe -m src.multiagent check-deploy --all
```

Start the interactive chatbot:

```powershell
.\.venv\Scripts\python.exe chat.py
```

Useful prompts inside the chatbot:

```text
VCB
VCB 2025-08-13
rank VCB,BID,CTG 2025-08-13
BID có nên mua 5 ngày tới
symbols
help
quit
```

For LLM narration, start Ollama separately and use:

```powershell
ollama pull qwen2.5:7b-instruct
ollama serve
.\.venv\Scripts\python.exe chat.py --llm
```

## End-to-end workflow

The commands below are ordered for a full regeneration. Dataset construction fetches market data and web news, so reruns are not expected to reproduce historical metrics bit-for-bit. Most scripts reuse compatible artifacts in `cache/` unless `--no-cache` is supplied.

### 1. Train or refresh the sentiment model (optional)

The sentiment pipeline can train the custom Transformer, PhoBERT, or both from the data under `data/sentiment/`.

```powershell
.\.venv\Scripts\python.exe run_sentiment_benchmark.py --variant both
.\.venv\Scripts\python.exe run_sentiment_benchmark.py --variant phobert --skip-dataset-download
```

### 2. Build a dataset

`pipeline.py` is the basic data-ingestion entry point. Its checked-in configuration currently uses `VCB` and `BID`, a 2022-01-01 to 2026-03-31 range, 30-step sequences, and 1-day labels.

```powershell
.\.venv\Scripts\python.exe pipeline.py
```

For the full seven-stock, three-horizon research configuration, the benchmark runners build their own pipeline configuration in code.

### 3. Run the model benchmark

The model benchmark uses the seven-stock universe, 2020-01-01 to 2026-03-31 data, a 30-day context, training through 2024-06-30, validation through 2024-12-31, and test data afterward. It writes per-horizon CSVs and figures to `results/`.

```powershell
# Full benchmark for 1D, 5D, and 20D
.\.venv\Scripts\python.exe run_model_benchmark.py

# Smaller targeted runs
.\.venv\Scripts\python.exe run_model_benchmark.py --horizons 5 --symbols VCB BID
.\.venv\Scripts\python.exe run_model_benchmark.py --horizons 5 --experiments "LSTM Baseline" "CNN-LSTM"
.\.venv\Scripts\python.exe run_model_benchmark.py --stage plot
.\.venv\Scripts\python.exe run_model_benchmark.py --skip-chronos
```

Valid stages are `data`, `predict`, `hpo`, and `plot`. Use `--folds N` for walk-forward evaluation and `--no-cache` only when a fresh inference/training run is intended.

### 4. Compare fusion strategies

The fusion benchmark compares backbone/fusion combinations (`none`, `early`, `late`, and `cmtf`), including shuffled-news placebo controls. Results are written below `results/ablation/`.

```powershell
.\.venv\Scripts\python.exe run_ablation_benchmark.py --horizons 1 5 20 --gate
.\.venv\Scripts\python.exe run_ablation_benchmark.py --horizons 5 --model cmtf --seeds 42
.\.venv\Scripts\python.exe run_ablation_benchmark.py --stage plot
```

### 5. Run the component-ablation registry

The registry is the authoritative runner for component-level CMTF studies. It evaluates pre-registered cells over default seeds `1 42 123`, writes aggregated tables and reports below `results/ablation_registry/<horizon>d/`, and computes gated metrics at fixed coverage.

```powershell
# Fast smoke run
.\.venv\Scripts\python.exe run_ablation_registry.py --cells 0 0p --seeds 42 --horizons 5

# Full registry
.\.venv\Scripts\python.exe run_ablation_registry.py --cells all --horizons 1 5 20 --seeds 1 42 123
```

### 6. Prepare deployable multi-agent artifacts

The live prediction path requires a three-seed CMTF checkpoint ensemble, frozen prediction stores, and a gate policy calibrated on validation predictions. Persist checkpoints during the core-cell run, then calibrate each horizon.

```powershell
$env:SAVE_DEPLOY_MODEL = "1"
.\.venv\Scripts\python.exe run_ablation_registry.py --cells 0 --horizons 1 5 20 --seeds 1 42 123
Remove-Item Env:SAVE_DEPLOY_MODEL

.\.venv\Scripts\python.exe -m src.multiagent calibrate --horizon 1
.\.venv\Scripts\python.exe -m src.multiagent calibrate --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent calibrate --horizon 20

.\.venv\Scripts\python.exe -m src.multiagent calibrate-interaction --horizon 1
.\.venv\Scripts\python.exe -m src.multiagent calibrate-interaction --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent calibrate-interaction --horizon 20

.\.venv\Scripts\python.exe -m src.multiagent check-deploy --all
```

### 7. Evaluate and use the multi-agent system

The graph routes requests through an orchestrator, parallel market/news evidence collection, prediction, a calibrated decision gate, cross-horizon interaction, risk and metalabel vetoes, narration, critic verification, and a reasoning reflection pass.

```powershell
# A single live, traced prediction
.\.venv\Scripts\python.exe -m src.multiagent predict --symbol VCB --cutoff 2025-08-13 --horizon 5 --trace

# Ranking and grounded research branches
.\.venv\Scripts\python.exe -m src.multiagent rank --symbols VCB,BID,CTG --cutoff 2025-08-13 --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent research --symbol VCB --cutoff 2025-08-13 --eval

# Evaluation studies
.\.venv\Scripts\python.exe run_ab_benchmark.py --symbols VCB BID --horizons 1 5 20
.\.venv\Scripts\python.exe -m src.multiagent eval --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent h3 --mode forecaster --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent metalabel-eval --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent h4-interaction-eval --horizon 5
.\.venv\Scripts\python.exe -m src.multiagent h5-reasoning-eval --horizon 5 --eval-mode

# Product demonstration report; --eval avoids LLM calls
.\.venv\Scripts\python.exe -m tools.e2e_demo --eval
```

Run `.\.venv\Scripts\python.exe -m src.multiagent --help` for the full command reference, including `batch-predict`, output files, JSON records, and trace transcripts.

## Outputs and artifact layout

| Location | Contents |
|---|---|
| `results/model_benchmark_<horizon>d.csv` | Forecast-model benchmark metrics |
| `results/figures/` | Benchmark charts and per-symbol heatmaps |
| `results/ablation/` | Fusion-comparison outputs |
| `results/ablation_registry/<horizon>d/` | Registry tables, ranking, placebo analysis, coverage diagnostics, and Markdown report |
| `results/gate_policies/` | Validation-calibrated gate policies |
| `results/horizon_interaction/` | Cross-horizon interaction policies |
| `results/agent_ablation/` | Multi-agent evaluation outputs and demonstration reports |
| `cache/predictions/` | Cached prediction stores used by evaluation/product flows |
| `cache/deploy_models/` | Deployable CMTF checkpoint ensemble and metadata |
| `cache/dataset/` | Selected retained dataset artifacts; other datasets are regenerated locally |
| `research/` | Thesis-facing methodology and result documents by project phase |

## Project layout

```text
pipeline.py                     Basic ingestion entry point
run_sentiment_benchmark.py      Vietnamese sentiment training/evaluation
run_model_benchmark.py          Forecast-model benchmark
run_ablation_benchmark.py       Fusion-strategy benchmark
run_ablation_registry.py        Config-driven CMTF component ablations
run_ab_benchmark.py             Multi-agent versus CMTF-only A/B benchmark
chat.py                         Interactive Vietnamese chatbot
tools/e2e_demo.py               Product-path demonstration report
src/pipeline/                   Fetching, scraping, alignment, features, datasets
src/sentiment/                  Sentiment data, models, training, inference, handoff
src/benchmark/                  Models, fusion, metrics, calibration, ablations, plots
src/multiagent/                 LangGraph workflow, live inference, policies, evaluations
tests/                          Unit and integration-style tests
research/                       Phase-by-phase research documentation
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pytest tests/test_news_scraper_parsing.py -v
```

The live scraper smoke test requires external access:

```powershell
$env:RUN_LIVE_SCRAPER = "1"
.\.venv\Scripts\python.exe -m pytest -m smoke tests/test_news_scraper_smoke.py -v
Remove-Item Env:RUN_LIVE_SCRAPER
```

## Research documents

Start with [research/README.md](research/README.md). It links the project’s five phase directories: data/baselines, CMTF fusion, ablation studies, multi-agent system, and the realtime chatbot.

## License

Academic and research use only.
