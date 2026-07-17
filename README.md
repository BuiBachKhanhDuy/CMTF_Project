# Cross-Modal Temporal Fusion (CMTF) — Vietnamese Stock Prediction

A multimodal time-series forecasting system that fuses **OHLCV market data** with
**Vietnamese financial news embeddings** through a FiLM/GRN-conditioned cross-modal
temporal fusion (CMTF) architecture, benchmarked against Amazon Chronos and other
time-series baselines, then wrapped in a **LangGraph multi-agent decision system** with a
real-time chatbot front end.

The project is organized into five phases, each with a tracked writeup in [research/](research/):

| Phase | Scope | Entry point |
|---|---|---|
| **1 — Data & baselines** | OHLCV + Vietnamese news ingestion, feature engineering, walk-forward dataset, baseline models | `pipeline.py`, `run_model_benchmark.py` |
| **2 — CMTF fusion** | Cross-modal temporal fusion of market + news; confidence-gate mechanism | `run_ablation_benchmark.py` |
| **3 — Component ablations** | 22-cell registry isolating each CMTF design choice; horizon-specific champion selection | `run_ablation_registry.py` |
| **4 — Multi-agent system** | LangGraph decision pipeline (gate/risk/metalabel/critic/…) vs. base-LLM baselines | `run_ab_benchmark.py`, `python -m src.multiagent` |
| **5 — Real-time chatbot** | Live inference + interactive Vietnamese advisor | `chat.py` |

## Prerequisites

Install these on the target machine **before** cloning:

| Requirement | Why | Notes |
|---|---|---|
| **Python 3.11** (3.11.x) | Matches the pinned toolchain (`torch`, `chronos-forecasting`, `sentence-transformers` wheels). | 3.12+/3.14 may fail to resolve wheels. Verify with `python --version`. |
| **Git + Git LFS** | Sentiment checkpoints (`phobert.pt` ≈ 542 MB) are stored via LFS. | `git lfs install` once per machine. |
| **~15 GB free disk** | Pip deps (~5 GB incl. torch) + HuggingFace model weights (~3 GB) + runtime `cache/` (grows to several GB) + optional Ollama model (~5 GB). | The offline demo alone needs only ~2 GB. |
| **Network access** | (a) HuggingFace model weights auto-download on first run (see [Models & Downloads](#models--downloads-first-run)); (b) the data pipeline fetches live OHLCV (vnstock/VCI) and scrapes Vietnamese news (CafeF/VnExpress/Vietstock/Google News). | Vietnam-facing endpoints + huggingface.co; a proxy may block either. |
| **HuggingFace models** (auto) | `dangvantuan/vietnamese-embedding`, `vinai/phobert-base-v2`, `amazon/chronos-t5-small`, `gpt2` — pulled on first use by the pipeline/benchmarks. | Downloaded automatically; pre-fetch optional (below). Not needed for the offline demo. |
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

## Models & Downloads (first run)

Model weights are **not** vendored in git (except the committed sentiment checkpoints via
LFS). They download automatically the first time each component runs:

| Model | Downloaded by | Needed for |
|---|---|---|
| `dangvantuan/vietnamese-embedding` | `src/pipeline/news_encoder.py` (SentenceTransformer) | any pipeline data build (Steps 2–6) |
| `vinai/phobert-base-v2` | sentiment training/inference | sentiment step + hybrid news sentiment |
| `amazon/chronos-t5-small` | `chronos-forecasting` | Chronos baseline + fusion (Step 3/4) |
| `gpt2` | `transformers` (`GPT2Model`) | GPT4TS baseline (Step 3) |
| `qwen2.5:7b-instruct` | **Ollama** (not HuggingFace) | only `chat.py --llm` / LLM evals |

The **offline demo path needs none of these** — `chat.py` and `check-deploy` read the
committed parquet (embeddings already baked in) and deploy checkpoints. The downloads only
matter when you rebuild the pipeline/benchmarks (Steps 1–6).

Optional — pre-fetch the HuggingFace weights up front (otherwise they pull lazily on first use):

```powershell
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("dangvantuan/vietnamese-embedding")
from transformers import AutoModel, AutoTokenizer, GPT2Model
AutoTokenizer.from_pretrained("vinai/phobert-base-v2"); AutoModel.from_pretrained("vinai/phobert-base-v2")
GPT2Model.from_pretrained("gpt2")
from chronos import ChronosPipeline
ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
print("All HuggingFace weights cached.")
PY
```

For the real-LLM chatbot, install [Ollama](https://ollama.com/) and pull the model once:

```powershell
ollama pull qwen2.5:7b-instruct     # ~4.7 GB; then `ollama serve` must be running
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

### Step 3 — Baseline benchmark (Phase 2)

Trains/evaluates the 10 baseline experiments (Chronos, LSTM/CNN-LSTM/GPT4TS ± news, RF,
Linear/MLP summary); writes `results/*.csv` + figures. This runner is **baseline-only** —
the CMTF champion is produced by the ablation registry in Step 4.

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

## Understanding the Benchmarks & Ablations

Three distinct runners, each answering a different question:

- **`run_model_benchmark.py` (Phase 2, baseline-only)** — how do standard time-series models
  (Chronos, LSTM, CNN-LSTM, GPT4TS, RF, summary baselines) and their news-fusion `Hybrid`
  variants compare on the shared walk-forward test set? Stages: `data | predict | hpo | plot`.
- **`run_ablation_benchmark.py` (Phase 2/3, fusion strategy)** — how do fusion *mechanisms*
  (none / early / late / CMTF) compare, and what does the **confidence gate** add? `--gate
  --gate-coverage 0.25` layers the validation-calibrated gate + conviction sizing on top of
  each cell and adds `DA%_gated / Sharpe_gated / IC_gated / gate_coverage` columns.
- **`run_ablation_registry.py` (Phase 3, component ablation)** — which *single design choice*
  drives CMTF's behavior? A config-driven registry of **22 cells**, each a one-field override
  of `CMTF_CORE`, in 6 groups: `0` reference (+ `0p` placebo), `1`–`7` component knockouts,
  `8`/`8p`/`9` news side, `10`–`13` gate sweep, `14`–`17` output formulation, `18`/`18p`
  learned gate. **Gated metrics are primary.** Outputs land in
  `results/ablation_registry/{1,5,20}d/` (`ranked.csv`, `real_minus_placebo.csv`,
  `monotonicity.csv`, `report.md`, coverage curves).

```powershell
python run_ablation_registry.py --cells all --horizons 1 5 20 --seeds 1 42 123   # full registry
python run_ablation_registry.py --cells 0 13 --horizons 5 --seeds 1 42 123        # just the champion vs reference
python run_ablation_registry.py --cells all --gate-coverage 0.25 --bootstrap 2000 # tune gate / CI resamples
```

Key caveat baked into the results: **no single component wins across all three horizons** —
read [research/phase3_ablation_studies/01_component_ablation_registry.md](research/phase3_ablation_studies/01_component_ablation_registry.md) before citing any cell.

## Understanding the Multi-Agent System (Phase 4/5)

A LangGraph `StateGraph` turns a raw CMTF prediction into a disclosed, risk-checked trade
decision. Two entry surfaces share the same 13 agent implementations:

- **`chat.py`** — interactive advisor; preloads the research book once, so in-book queries
  are instant. Chain: `predict → gate → horizon_interaction → risk → metalabel → narrator
  → critic → reasoning`.
- **`python -m src.multiagent`** — scriptable CLI over the full compiled graph:
  `orchestrator → [market ‖ news] → predict → gate → horizon_interaction → risk →
  metalabel → narrator → critic → reasoning`.

**How the decision is made:** `predict_agent` runs a real forward pass of the 3-seed champion
ensemble. `gate_agent` is the **only** node that turns the prediction into long/short/abstain
— it abstains when `|prediction| < tau` (the confidence threshold; only the top ~25% most
confident predictions trade). `risk_agent` and `metalabel_agent` can **veto** a trade
(volatility, adverse news) but never create one; `critic_agent` checks every number/date in
the answer is grounded in state; `reasoning_agent` can append a caveat or trigger a
widen-and-rerun.

**Intents** (auto-classified by `orchestrator_agent`, EN/VI): `PREDICTION`, `RESEARCH`
(trend/analysis over a date range), `COMPARISON` (rank N symbols), `EXPLANATION`.

**In-book vs live:** a query date already inside the cached range answers from cache (fast,
bit-exact); a date beyond it triggers a real live OHLCV + news fetch (`[live forward pass]`,
minutes on a cold process). Undated queries anchor on the real current date.

**Evaluations** (`python -m src.multiagent ...`, results in `results/agent_ablation/{H}d/`):
`eval` (A0–A5 agent ladder), `h3` (MAS vs plain LLM), `metalabel-eval`, `h4-interaction-eval`,
`h5-reasoning-eval`. `check-deploy --all` verifies gate policy + frozen predictions + deploy
checkpoints per horizon. Canonical writeups:
[research/phase4_multiagent_system/](research/phase4_multiagent_system/) and
[research/phase5_realtime_chatbot/](research/phase5_realtime_chatbot/).

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
│       └── agents/              # 13 specialized agent nodes: orchestrator, market, news,
│                                 # predict, gate, horizon_interaction, risk, metalabel,
│                                 # narrator, critic, reasoning, rank, research
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

## Stack & Data

- **Python 3.11** · PyTorch · Amazon Chronos (T5-Small) · PhoBERT · LangGraph + Ollama
- **Market data:** vnstock v3.x, OHLCV source `KBS`, `1D` bars
- **News:** VnExpress · CafeF (banking) · Vietstock · Google News (leakage-free alignment)
- **Symbols (7 VN banking large-caps):** `VCB, BID, CTG, TCB, MBB, ACB, VPB`
- **Fetch range:** **2020-01-01 → 2026-03-31**
- **Walk-forward splits:** train ≤ **2024-06-30** · validation ≤ **2024-12-31** · test **2025-01-01 → 2026-03-31**
- **Model inputs:** 30-day trailing window · ~23-dim market feature vector · 768-dim PhoBERT
  news embeddings (773-dim hybrid when the sentiment handoff is enabled)
- **Horizons:** 1D, 5D, 20D forward log-return

> The narrower range in `pipeline.py`'s `__main__` block is only a quick demo config; the
> research/multi-agent path uses the full range above via
> `run_ablation_benchmark._build_pipeline_config`.

## Models Benchmarked (Phase 2, `run_model_benchmark.py`)

Ten experiments on the shared walk-forward test set (`--experiments` filters the list;
`--skip-chronos` drops the slow zero-shot run). `Hybrid` = the news-fusion (CMTF) variant
of that backbone; the plain name = market-only baseline.

| Experiment | Type |
|---|---|
| Chronos Zero-Shot | Foundation model, no training |
| LSTM Baseline / **LSTM Hybrid** | Sequence model, market-only / + news fusion |
| CNN-LSTM / **CNN-LSTM Hybrid** | Conv-sequence model, market-only / + news fusion |
| GPT4TS Baseline / **GPT4TS Hybrid** | Frozen-GPT2 time-series model, market-only / + news fusion |
| Random Forest Baseline | Tabular baseline over engineered features |
| Linear Summary Baseline | Linear model over summary features |
| MLP Summary Baseline | MLP over summary features |

Metrics per model: **MAE, RMSE, DA%, Sharpe, IC, F1**, plus a composite diagnostic
(`src/benchmark/metrics.py`). The **CMTF champion** itself is not one of these rows — it is
selected and deployed via the Phase 3 ablation registry (below).

## Deployed CMTF Champion (Phase 3 → 4)

The production model is horizon-specific, chosen by the ablation registry and confirmed
out-of-sample (`core_cell_for()` in `src/multiagent/gate_io.py`):

| Horizon | Champion cell | Out-of-sample gated result |
|---|---|---|
| 1D | cell 0 (`CMTF_CORE`) | reference; cell 13 hurt 1D, so not adopted |
| 5D | cell 13 (`recency_gate_k=5`) | DA 54.4% → **58.3%**, Sharpe 0.25 → **0.52**, IC 0.13 → **0.21** |
| 20D | cell 13 (`recency_gate_k=5`) | DA 75.4% → **83.6%**, Sharpe 0.99 → **1.13**, IC ≈ 0.41 |

All metrics are **gated** (top ~25% most-confident predictions, the deployed operating
point). Full evidence: [research/phase3_ablation_studies/](research/phase3_ablation_studies/)
and `results/ablation_registry/{1,5,20}d/`. Treat any single-cell/single-horizon number as a
hypothesis until checked across horizons — horizon-dependence is Phase 3's headline finding.

## License

Academic use only. See the thesis writeups under [research/](research/) for citations.
