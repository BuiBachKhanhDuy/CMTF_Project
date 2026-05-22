"""Kaggle Notebook Launcher for Model Benchmark.

Upload the kaggle_bundle/ folder as a Kaggle Dataset, then paste this
into a Kaggle Notebook (GPU P100 or T4) and run all cells.

Dataset should be mounted at: /kaggle/input/chatbot-thesis/
"""

# %% [markdown]
# # Model Benchmark — Full Run (Baselines + CMTF)

# %% Install dependencies
# fmt: off
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "vnstock>=3.2.0", "pandas-ta>=0.3.14b1", "sentence-transformers>=2.2.0",
    "chronos-forecasting>=1.3.0", "optuna>=3.5.0", "loguru>=0.7.0",
    "tenacity>=8.2.0", "tqdm>=4.65.0", "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0", "beautifulsoup4>=4.12.0", "lxml>=4.9.0",
    "langgraph>=0.2.0", "langchain-core>=0.3.0", "langchain-ollama>=0.2.0",
])
# fmt: on

# %% Setup working directory
import os, shutil
from pathlib import Path

# Kaggle mounts dataset here
INPUT_DIR = Path("/kaggle/input/chatbot-thesis")
WORK_DIR = Path("/kaggle/working/benchmark")

# Copy source code to writable location
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
shutil.copytree(INPUT_DIR, WORK_DIR)
os.chdir(WORK_DIR)

# Ensure cache directories are writable (input is read-only on Kaggle)
for d in ["cache/predictions", "cache/cmtf_models", "cache/optuna",
           "cache/chronos_emb", "results/figures", "outputs/phase2"]:
    Path(d).mkdir(parents=True, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"GPU available: {__import__('torch').cuda.is_available()}")
if __import__('torch').cuda.is_available():
    print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")

# %% Patch VnstockDataFetcher to use cached CSV when API unavailable
import pandas as pd
import sys
sys.path.insert(0, str(WORK_DIR))

from src.pipeline.data_fetcher import VnstockDataFetcher

_original_fetch_ohlcv = VnstockDataFetcher.fetch_ohlcv

def _patched_fetch_ohlcv(self, symbol, start, end, interval="1D", source="KBS"):
    """Try cached CSV first, fall back to vnstock API."""
    csv_path = Path(f"cache/raw_ohlcv/{symbol}_{start}_{end}_ohlcv.csv")
    if csv_path.exists():
        print(f"  [CACHE HIT] Loading raw OHLCV from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "time"
        return df
    print(f"  [CACHE MISS] Fetching {symbol} from vnstock API...")
    return _original_fetch_ohlcv(self, symbol, start, end, interval, source)

VnstockDataFetcher.fetch_ohlcv = _patched_fetch_ohlcv
print("VnstockDataFetcher patched to use CSV cache.")

# %% Run the full benchmark (all baselines + CMTF)
os.environ["PYTHONPATH"] = str(WORK_DIR)

# Option 1: Run via subprocess (captures full log)
result = subprocess.run(
    [sys.executable, "run_model_benchmark.py", "--comparison-set", "full", "--horizons", "1", "5", "20"],
    capture_output=False,
    text=True,
    cwd=str(WORK_DIR),
)
print(f"\nBenchmark exit code: {result.returncode}")

# %% Show results
results_dir = WORK_DIR / "results"
for csv_file in sorted(results_dir.glob("chronos_benchmark_*.csv")):
    print(f"\n{'='*60}")
    print(f"  {csv_file.name}")
    print(f"{'='*60}")
    df = pd.read_csv(csv_file)
    print(df.to_string(index=False))

# %% Display figures
from IPython.display import Image, display
figures_dir = results_dir / "figures"
for img in sorted(figures_dir.glob("*.png")):
    print(f"\n{img.name}")
    display(Image(filename=str(img)))
