"""CMTF Data Pipeline — CLI entry point.

Thin wrapper that delegates to ``src.pipeline.orchestrator``.
"""

from src.pipeline import run_pipeline  # noqa: F401

# ------------------------------------------------------------------
# Convenience: run from CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "symbols": ["VCB", "VIC", "VHM"],
        "start": "2022-01-01",
        "end": "2024-12-31",
        "interval": "1D",
        "ohlcv_source": "KBS",
        "news_source": "VCI",
        "sequence_len": 30,
        "horizon": 1,
        "train_end": "2023-12-31",
        "val_end": "2024-06-30",
        "normalize_method": "zscore",
    }

    dataset = run_pipeline(config)

    # Quick sanity check
    sample = dataset[0]
    for k, v in sample.items():
        print(f"{k:8s} → {v.shape}  dtype={v.dtype}")
