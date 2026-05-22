"""CMTF Data Pipeline — CLI entry point.

Thin wrapper that delegates to ``src.pipeline.orchestrator``.
"""

from src.pipeline import run_pipeline  # noqa: F401

# ------------------------------------------------------------------
# Convenience: run from CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "symbols": ["VCB", "BID"],
        "start": "2022-01-01",
        "end": "2026-03-31",
        "interval": "1D",
        "ohlcv_source": "KBS",
        "news_source": "web",
        "news_sources": ("vnexpress", "cafef_banking", "vietstock"),
        "news_use_cache": False,
        "news_export_trace": True,
        "news_similarity_threshold": 85.0,
        "log_news_coverage": True,
        "sequence_len": 30,
        "horizon": 1,
        "train_end": "2024-06-30",
        "val_end": "2024-12-31",
        "normalize_method": "zscore",
        # Optional: stability-based feature selection (disabled by default)
        "stability_selection_enabled": True,
        "stability_corr_threshold": 0.95,
        "stability_lasso_alpha": 0.001,
        "stability_n_folds": 5,
        "stability_threshold": 0.6,
        "stability_min_train_rows": 120,
    }

    dataset = run_pipeline(config)

    # Quick sanity check
    sample = dataset[0]
    for k, v in sample.items():
        print(f"{k:8s} → {v.shape}  dtype={v.dtype}")
