# prepare_kaggle_bundle.ps1
# Copies all files needed to run the full model benchmark (baselines + CMTF) on Kaggle.
# After running, zip kaggle_bundle/ and upload to Kaggle as a dataset.

$ErrorActionPreference = "Stop"
$bundle = "kaggle_bundle"

# Clean previous bundle
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
New-Item -ItemType Directory -Path $bundle | Out-Null

Write-Host "=== Copying source code ===" -ForegroundColor Cyan

# Core scripts
Copy-Item "run_model_benchmark.py" "$bundle/"
Copy-Item "pipeline.py" "$bundle/" -ErrorAction SilentlyContinue
Copy-Item "requirements.txt" "$bundle/"
Copy-Item "pytest.ini" "$bundle/" -ErrorAction SilentlyContinue

# src/ (exclude __pycache__)
robocopy "src" "$bundle/src" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NC /NS | Out-Null

Write-Host "=== Copying cached datasets (skip vnstock API) ===" -ForegroundColor Cyan

# cache/dataset/ — parquet files (skip live data fetch)
New-Item -ItemType Directory -Path "$bundle/cache/dataset" -Force | Out-Null
Copy-Item "cache/dataset/*.parquet" "$bundle/cache/dataset/"

Write-Host "=== Copying cached news (skip web scraping) ===" -ForegroundColor Cyan

# cache/news/ — pre-scraped news JSON
New-Item -ItemType Directory -Path "$bundle/cache/news" -Force | Out-Null
Copy-Item "cache/news/*.json" "$bundle/cache/news/"

Write-Host "=== Copying cached embeddings (skip sentence-transformer encode) ===" -ForegroundColor Cyan

# cache/embeddings/ — news embedding npz files
New-Item -ItemType Directory -Path "$bundle/cache/embeddings" -Force | Out-Null
Copy-Item "cache/embeddings/*.npz" "$bundle/cache/embeddings/"

Write-Host "=== Copying Chronos token cache (skip re-tokenization) ===" -ForegroundColor Cyan

# cache/chronos_emb/ — tokenized close windows for LoRA
New-Item -ItemType Directory -Path "$bundle/cache/chronos_emb" -Force | Out-Null
Copy-Item "cache/chronos_emb/*.npy" "$bundle/cache/chronos_emb/"

Write-Host "=== Copying pre-trained model checkpoints ===" -ForegroundColor Cyan

# cache/cmtf_models/ — LSTM/CNN-LSTM/LoRA checkpoints (optional, saves retraining)
New-Item -ItemType Directory -Path "$bundle/cache/cmtf_models" -Force | Out-Null
Copy-Item "cache/cmtf_models/*.pt" "$bundle/cache/cmtf_models/" -ErrorAction SilentlyContinue

Write-Host "=== Copying cached predictions ===" -ForegroundColor Cyan

# cache/predictions/ — cached model outputs
New-Item -ItemType Directory -Path "$bundle/cache/predictions" -Force | Out-Null
Copy-Item "cache/predictions/*.npy" "$bundle/cache/predictions/" -ErrorAction SilentlyContinue

Write-Host "=== Copying Optuna HPO cache ===" -ForegroundColor Cyan

# cache/optuna/ — HPO results
New-Item -ItemType Directory -Path "$bundle/cache/optuna" -Force | Out-Null
Copy-Item "cache/optuna/*" "$bundle/cache/optuna/" -ErrorAction SilentlyContinue

Write-Host "=== Creating output directories ===" -ForegroundColor Cyan

# Pre-create output dirs so the script doesn't fail
New-Item -ItemType Directory -Path "$bundle/results/figures" -Force | Out-Null
New-Item -ItemType Directory -Path "$bundle/outputs/phase2" -Force | Out-Null

Write-Host "=== Pre-caching raw OHLCV as CSV (avoid vnstock API on Kaggle) ===" -ForegroundColor Cyan

# Save raw OHLCV locally so Kaggle doesn't need vnstock API
$pythonScript = @"
import sys
sys.path.insert(0, '.')
from src.pipeline.data_fetcher import VnstockDataFetcher
from pathlib import Path
import pandas as pd

fetcher = VnstockDataFetcher()
symbols = ['VCB', 'BID']
start, end = '2022-01-01', '2026-03-31'

out_dir = Path('kaggle_bundle/cache/raw_ohlcv')
out_dir.mkdir(parents=True, exist_ok=True)

for sym in symbols:
    df = fetcher.fetch_ohlcv(sym, start, end)
    path = out_dir / f'{sym}_{start}_{end}_ohlcv.csv'
    df.to_csv(path)
    print(f'  Saved {path} ({len(df)} rows)')
"@
& .\.venv\Scripts\python.exe -c $pythonScript

Write-Host "=== Copying Kaggle notebook ===" -ForegroundColor Cyan
Copy-Item "kaggle_notebook.py" "$bundle/" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Bundle complete! ===" -ForegroundColor Green
Write-Host "Location: $bundle/"

# Show size
$size = (Get-ChildItem $bundle -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total size: {0:N1} MB" -f $size)
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. zip kaggle_bundle/ folder"
Write-Host "  2. Push to git OR upload directly to Kaggle as a Dataset"
Write-Host "  3. Create a Kaggle Notebook (GPU P100), use kaggle_notebook.py as template"
