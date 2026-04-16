"""CMTF Data Pipeline — End-to-end orchestration.

Wires together data fetching, temporal alignment, feature engineering,
news encoding, and dataset construction.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .data_fetcher import VnstockDataFetcher
from .temporal_aligner import TemporalAligner
from .feature_engineer import FeatureEngineer
from .news_encoder import NewsEncoder
from .dataset_builder import CMTFDataset

_DATASET_CACHE_DIR = Path("./cache/dataset")


def _config_hash(config: dict[str, Any]) -> str:
    """Compute a short hash of config keys that affect the dataset."""
    keys = [
        "symbols", "start", "end", "interval", "ohlcv_source",
        "news_source", "news_sources", "news_similarity_threshold",
        "sequence_len", "horizon", "target_horizon_days",
        "train_end", "val_end", "normalize_method",
    ]
    h = hashlib.sha256()
    for k in keys:
        h.update(f"{k}={config.get(k)}".encode())
    return h.hexdigest()[:16]


def _save_dataset_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save the processed DataFrame to parquet (news_emb as bytes)."""
    df_out = df.copy()
    # Convert numpy arrays in news_emb to bytes for parquet compatibility
    if "news_emb" in df_out.columns:
        df_out["news_emb"] = df_out["news_emb"].apply(
            lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
        )
    # Drop list/object columns that parquet can't handle natively
    drop_cols = [c for c in ("news_titles", "news_content") if c in df_out.columns]
    # Also drop any remaining columns with dtype 'object' that aren't
    # already handled (except news_emb which is now bytes, and symbol)
    for c in df_out.columns:
        if df_out[c].dtype == object and c not in ("news_emb", "symbol"):
            drop_cols.append(c)
    if drop_cols:
        df_out = df_out.drop(columns=list(set(drop_cols)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(cache_path, index=True)
    logger.info("Dataset cached → {} ({} rows)", cache_path.name, len(df_out))


def _load_dataset_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached DataFrame and restore news_emb from bytes."""
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        # Restore news_emb from bytes → np.ndarray
        if "news_emb" in df.columns:
            df["news_emb"] = df["news_emb"].apply(
                lambda b: np.frombuffer(b, dtype=np.float32).copy()
                if isinstance(b, (bytes, bytearray)) else b
            )
        # Recreate empty list columns so downstream code doesn't break
        if "news_titles" not in df.columns:
            df["news_titles"] = [[] for _ in range(len(df))]
        if "news_content" not in df.columns:
            df["news_content"] = [[] for _ in range(len(df))]
        logger.info("Dataset loaded from cache: {} ({} rows)", cache_path.name, len(df))
        return df
    except Exception:
        logger.warning("Corrupt dataset cache {} — rebuilding", cache_path.name)
        return None


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
    rebuild_data: bool = bool(config.get("rebuild_data", False))

    # When rebuild_data is True, bypass ALL caches (news, embeddings, dataset)
    if rebuild_data:
        news_use_cache = False
        logger.info("rebuild_data=True → all caches bypassed")

    # --- Try loading from dataset cache ---
    cfg_hash = _config_hash(config)
    cache_path = _DATASET_CACHE_DIR / f"dataset_{cfg_hash}.parquet"
    if not rebuild_data:
        cached_df = _load_dataset_cache(cache_path)
        if cached_df is not None:
            cached_df = cached_df.dropna(subset=[target_col])
            dataset = CMTFDataset(
                df_featured=cached_df,
                sequence_len=seq_len,
                horizon=horizon,
                target_horizon_days=target_horizon_days,
            )
            logger.info("Pipeline complete (from cache) | dataset length = {}", len(dataset))
            return dataset

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

    # 5b. VN-Index macro features (RCSAN / TFT exogenous covariate approach)
    # Fetch VN-Index OHLCV and compute market-wide features that capture
    # macro events (war, policy, FX shocks) already priced into the index.
    try:
        df_vnindex = fetcher.fetch_ohlcv("VNINDEX", start, end, interval, ohlcv_source)
        vnidx_close = df_vnindex["close"]
        vnidx_features = pd.DataFrame(index=df_vnindex.index)
        vnidx_features["vnindex_ret"] = np.log(vnidx_close / vnidx_close.shift(1))
        vnidx_vol_ma = df_vnindex["volume"].rolling(window=20, min_periods=1).mean()
        vnidx_features["vnindex_vol_ratio"] = df_vnindex["volume"] / vnidx_vol_ma
        # Merge into each bar by date (left join preserves all stock rows)
        df_all = df_all.merge(
            vnidx_features, left_index=True, right_index=True, how="left",
        )
        # Forward-fill then back-fill small gaps (holidays differ between
        # individual stocks and the index)
        for col in ("vnindex_ret", "vnindex_vol_ratio"):
            df_all[col] = df_all[col].ffill().bfill().fillna(0.0)
        logger.info("VN-Index macro features added (vnindex_ret, vnindex_vol_ratio)")
    except Exception:
        logger.warning("VN-Index fetch failed — proceeding without macro features")
        df_all["vnindex_ret"] = 0.0
        df_all["vnindex_vol_ratio"] = 1.0

    # 6. Encode news
    df_all = encoder.encode_dataframe(
        df_all, text_col="news_content", use_cache=(not rebuild_data),
    )
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

    # 7. Normalise market features (fit on train split only, per symbol)
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

    # Ensure all market feature columns are float before normalization
    # (e.g. volume is int64 but becomes float after z-score)
    for col in market_feature_cols:
        if df_all[col].dtype in ("int64", "int32"):
            df_all[col] = df_all[col].astype("float64")

    for sym in df_all["symbol"].unique():
        sym_mask = df_all["symbol"] == sym
        sym_df = df_all.loc[sym_mask].copy()
        sym_df = engineer.normalize(
            sym_df,
            feature_cols=market_feature_cols,
            method=norm_method,  # type: ignore[arg-type]
            split_date=train_end,
            symbol=sym,
        )
        df_all.loc[sym_mask, market_feature_cols] = sym_df[market_feature_cols]

    # Drop rows with NaN target (first / last rows from indicator warm-up)
    df_all = df_all.dropna(subset=[target_col])

    # 8. Build dataset
    dataset = CMTFDataset(
        df_featured=df_all,
        sequence_len=seq_len,
        horizon=horizon,
        target_horizon_days=target_horizon_days,
    )

    # 9. Save processed dataset to cache for reuse
    _save_dataset_cache(df_all, cache_path)

    logger.info("Pipeline complete | dataset length = {}", len(dataset))
    return dataset
