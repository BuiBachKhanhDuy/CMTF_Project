"""CMTF Data Pipeline — End-to-end orchestration.

Wires together data fetching, temporal alignment, feature engineering,
news encoding, and dataset construction.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from loguru import logger

from .data_fetcher import VnstockDataFetcher
from .temporal_aligner import TemporalAligner
from .feature_engineer import FeatureEngineer
from .news_encoder import NewsEncoder
from .dataset_builder import CMTFDataset


def run_pipeline(config: dict[str, Any]) -> CMTFDataset:
    """Execute the full CMTF data-ingestion pipeline.

    Steps:
        1. Fetch OHLCV per symbol
        2. Fetch news per symbol
        3. Temporal alignment (leakage-safe)
        4. Technical indicators
        5. Concatenate multi-symbol DataFrames
        6. Encode news → 768-dim embeddings
        7. Normalise market features (fit on train only)
        8. Build and return ``CMTFDataset``

    Args:
        config: Pipeline configuration dict.  Required keys::

            symbols, start, end, interval, ohlcv_source, news_source,
            sequence_len, horizon, train_end, val_end, normalize_method

    Returns:
        A ready-to-use :class:`CMTFDataset` instance.
    """
    symbols: list[str] = config["symbols"]
    start: str = config["start"]
    end: str = config["end"]
    interval: str = config.get("interval", "1D")
    ohlcv_source: str = config.get("ohlcv_source", "KBS")
    news_source: str = config.get("news_source", "VCI")
    news_sources: tuple[str, ...] = tuple(
        config.get("news_sources", ("cafef_banking", "vietstock"))
    )
    news_use_cache: bool = bool(config.get("news_use_cache", True))
    news_export_trace: bool = bool(config.get("news_export_trace", True))
    news_similarity_threshold: float = float(config.get("news_similarity_threshold", 85.0))
    log_news_coverage: bool = bool(config.get("log_news_coverage", True))
    seq_len: int = config.get("sequence_len", 30)
    horizon: int = config.get("horizon", 1)
    target_horizon_days: int = int(config.get("target_horizon_days", 1))
    target_col = f"fwd_ret_{target_horizon_days}d"
    train_end: str = config["train_end"]
    val_end: str = config["val_end"]
    norm_method: str = config.get("normalize_method", "zscore")

    fetcher = VnstockDataFetcher()
    aligner = TemporalAligner()
    engineer = FeatureEngineer()
    encoder = NewsEncoder()

    all_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        logger.info("━━━ Processing {} ━━━", symbol)

        # 1. OHLCV
        try:
            df_ohlcv = fetcher.fetch_ohlcv(symbol, start, end, interval, ohlcv_source)
        except Exception:
            logger.exception("Skipping {} — OHLCV fetch failed", symbol)
            continue

        # 2. News (source controlled by config)
        try:
            mode = str(news_source).strip().lower()
            if mode in {"vci", "api"}:
                df_news = fetcher.fetch_news(symbol, source="VCI")
                if not df_news.empty:
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    df_news = df_news[
                        (df_news["published_date"] >= start_ts)
                        & (df_news["published_date"] <= end_ts)
                    ]
            else:
                df_news = fetcher.fetch_news_multi_source(
                    symbol,
                    start,
                    end,
                    sources=news_sources,
                    use_cache=news_use_cache,
                    export_trace=news_export_trace,
                    similarity_threshold=news_similarity_threshold,
                )
        except Exception:
            logger.warning("News fetch failed for {} — proceeding without news", symbol)
            df_news = pd.DataFrame(columns=["published_date", "title", "content"])

        # 3. Temporal alignment
        df_aligned = aligner.assign_news_to_bars(df_ohlcv, df_news)
        df_aligned = aligner.add_null_mask(df_aligned)
        if log_news_coverage:
            bars_with_news = int(df_aligned["has_news"].sum()) if "has_news" in df_aligned else 0
            total_bars = len(df_aligned)
            pct = (100.0 * bars_with_news / total_bars) if total_bars else 0.0
            logger.info(
                "News coverage | {} | {} / {} bars ({:.2f}%)",
                symbol,
                bars_with_news,
                total_bars,
                pct,
            )

        # 4. Technical indicators
        df_featured = engineer.compute_technical(df_aligned)

        # Add symbol column
        df_featured["symbol"] = symbol

        all_frames.append(df_featured)

    if not all_frames:
        raise ValueError("No data fetched for any symbol — aborting pipeline")

    # 5. Concatenate
    df_all = pd.concat(all_frames, axis=0)
    df_all = df_all.sort_index()
    logger.info("Combined DataFrame: {} rows × {} cols", *df_all.shape)

    # 6. Encode news
    df_all = encoder.encode_dataframe(df_all, text_col="news_content")
    if log_news_coverage and "has_news" in df_all.columns:
        has_news_count = int(df_all["has_news"].sum())
        total_rows = len(df_all)
        pct = (100.0 * has_news_count / total_rows) if total_rows else 0.0
        logger.info(
            "Encoded news coverage | {} / {} rows ({:.2f}%) have non-zero news",
            has_news_count,
            total_rows,
            pct,
        )

    # 7. Normalise market features (fit on train split only)
    market_feature_cols = [
        c
        for c in df_all.columns
        if not re.match(r"^fwd_ret_\d+d$", str(c))
        and c not in {
            "news_emb", "has_news", "news_count",
            "news_titles", "news_content", "news_missing_flag", "symbol",
        }
        and df_all[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    df_all = engineer.normalize(
        df_all,
        feature_cols=market_feature_cols,
        method=norm_method,  # type: ignore[arg-type]
        split_date=train_end,
        symbol="combined",
    )

    # Drop rows with NaN target (first / last rows from indicator warm-up)
    df_all = df_all.dropna(subset=[target_col])

    # 8. Build dataset
    dataset = CMTFDataset(
        df_featured=df_all,
        sequence_len=seq_len,
        horizon=horizon,
        target_horizon_days=target_horizon_days,
    )

    logger.info("Pipeline complete | dataset length = {}", len(dataset))
    return dataset
