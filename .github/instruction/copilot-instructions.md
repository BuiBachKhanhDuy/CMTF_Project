# GitHub Copilot — Repository Instructions
# Vietnamese Financial Chatbot | Multimodal Price Prediction Thesis

## Project Identity
This is a Django-based Vietnamese financial chatbot that integrates multimodal
machine learning (OHLCV time-series, PhoBERT Vietnamese NLP, FinBERT English NLP,
macro indicators, company report parsing) for stock price prediction on the
Vietnamese HOSE/HNX markets. This is an academic thesis project.

## Monorepo Layout
```
vn-fin-chatbot/
├── .github/
│   ├── agents/              ← Copilot agent definitions (.agent.md)
│   ├── instructions/        ← This file and supplementary instructions
│   └── copilot/             ← copilot-setup-steps.yml
├── config/                  ← Django settings (base, dev, test, prod)
├── src/
│   ├── data/                ← ETL, scrapers, transformers (data-pipeline agent)
│   ├── models/              ← ML architectures, training, evaluation (ml-trainer agent)
│   ├── chatbot/             ← RAG pipeline, LLM client, prompts (chatbot-api agent)
│   ├── api/                 ← DRF views, serializers, URLs (chatbot-api agent)
│   └── integrations/        ← Vector store, prediction service clients
├── tests/                   ← All tests (test-evaluator agent)
├── docs/                    ← All documentation (documentation agent)
├── experiments/             ← ML experiment configs and results
├── notebooks/               ← Jupyter EDA notebooks
├── pipelines/               ← Celery task definitions
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── manage.py
```

## Global Coding Standards (all agents must follow)

### Python
- Python 3.11+; use PEP 604 union types (`X | None` not `Optional[X]`)
- Type-hint all function signatures; use `from __future__ import annotations` at top
- Google-style docstrings on all public classes and functions
- Max line length: 100 characters (configured in `pyproject.toml`)
- Formatter: Black; Linter: Ruff; Type checker: mypy (strict mode for new files)
- Imports: stdlib → third-party → local (isort enforced)

### Django
- Django 5.x patterns only — no deprecated `url()`, no class-less views
- Always use `select_related` / `prefetch_related` to prevent N+1 queries
- Database: PostgreSQL 16 (never SQLite, even in dev — use Docker)
- Migrations: one migration per logical change; never squash without team discussion
- Settings: use `django-environ`; never hardcode secrets or URLs

### Git / PR
- Branch naming: `feature/<agent>/<short-desc>` e.g. `feature/data-pipeline/ohlcv-vnstock`
- Commit messages: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`)
- PR must pass: pytest, mypy, ruff, black --check before merge
- No force-push to `main` or `develop`

### Security
- Secrets: environment variables only (`django-environ`)
- No API keys, tokens, or passwords in code, comments, or git history
- All user input sanitised before use in prompts, queries, or file paths
- Vietnamese text: always handle Unicode correctly (NFC normalisation)

## Agent Routing Guide
When GitHub Copilot receives a task, use this to pick the right agent:

| Task involves... | Use agent |
|-----------------|-----------|
| `src/data/`, scrapers, ETL, Celery data tasks | `data-pipeline` |
| `src/models/`, ML training, evaluation, ONNX export | `ml-trainer` |
| `src/chatbot/`, `src/api/`, LLM, RAG, endpoints | `chatbot-api` |
| `tests/`, evaluation scripts, coverage | `test-evaluator` |
| `docs/`, docstrings, ADRs, thesis writing | `documentation` |
| Ambiguous / cross-cutting | Ask the user to clarify scope |

## Vietnamese Language Handling
- All Vietnamese text must be NFC-normalised: `unicodedata.normalize('NFC', text)`
- Support both Northern and Southern Vietnamese diacritics
- Never assume ASCII-only input — all text fields are `TextField` in Django
- PhoBERT tokeniser handles Vietnamese segmentation; do not use NLTK for VN text
- Log Vietnamese strings as UTF-8; configure `PYTHONIOENCODING=utf-8` in Docker

## Financial Domain Rules
- Never generate trading signals presented as financial advice
- Always attach confidence scores and data staleness warnings to predictions
- Price predictions are for research purposes only — include disclaimer in API responses
- Vietnamese market hours: 9:00–11:30 and 13:00–15:00 ICT (UTC+7), Mon–Fri
- Ticker format: 3-letter uppercase (e.g. VNM, HPG, FPT, VIC)