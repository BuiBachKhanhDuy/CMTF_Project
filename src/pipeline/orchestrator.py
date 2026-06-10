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

from src.sentiment import Phase2PhoBERTInferencer, load_phase2_phobert_inference_bundle

from .data_fetcher import VnstockDataFetcher
from .temporal_aligner import TemporalAligner
from .feature_engineer import FeatureEngineer
from .news_encoder import NEWS_HYBRID_COLUMN, SENTIMENT_TRACE_COLUMNS, NewsEncoder
from .dataset_builder import CMTFDataset

_DATASET_CACHE_DIR = Path("./cache/dataset")
_SENTIMENT_EXPORT_DIR = Path("./artifacts/hybrid_sentiment")


def _config_hash(config: dict[str, Any]) -> str:
    """Compute a short hash of config keys that affect the dataset."""
    keys = [
        "symbols", "start", "end", "interval", "ohlcv_source",
        "news_source", "news_sources", "news_similarity_threshold",
        "sequence_len", "horizon", "target_horizon_days",
        "train_end", "val_end", "normalize_method",
        "news_sentiment_enabled", "phase2_output_dir", "news_sentiment_device",
    ]
    h = hashlib.sha256()
    for k in keys:
        h.update(f"{k}={config.get(k)}".encode())
    return h.hexdigest()[:16]


def _save_dataset_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save the processed DataFrame to parquet (news_emb as bytes)."""
    df_out = df.copy()
    # Convert numpy arrays in news_emb to bytes for parquet compatibility
    for col in ("news_emb", NEWS_HYBRID_COLUMN):
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
            )
    # Drop list/object columns that parquet can't handle natively
    drop_cols = [
        c
        for c in ("news_titles", "news_content", "news_title_sentiment_scores")
        if c in df_out.columns
    ]
    # Also drop any remaining columns with dtype 'object' that aren't
    # already handled (except news_emb which is now bytes, and symbol)
    for c in df_out.columns:
        if df_out[c].dtype == object and c not in ("news_emb", NEWS_HYBRID_COLUMN, "symbol"):
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
        for col in ("news_emb", NEWS_HYBRID_COLUMN):
            if col in df.columns:
                df[col] = df[col].apply(
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


def _export_sentiment_outputs(
    df_featured: pd.DataFrame,
    article_trace: pd.DataFrame | None,
    cache_hash: str,
) -> None:
    _SENTIMENT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    bar_cols = ["symbol", "news_count", "has_news", *SENTIMENT_TRACE_COLUMNS]
    present_bar_cols = [col for col in bar_cols if col in df_featured.columns]
    if present_bar_cols:
        bar_frame = df_featured.reset_index().rename(columns={df_featured.index.name or "index": "time"})
        bar_frame[present_bar_cols + ["time"]].to_csv(
            _SENTIMENT_EXPORT_DIR / f"sentiment_bars_{cache_hash}.csv",
            index=False,
            encoding="utf-8",
        )

    if article_trace is not None and not article_trace.empty:
        article_trace.to_csv(
            _SENTIMENT_EXPORT_DIR / f"sentiment_articles_{cache_hash}.csv",
            index=False,
            encoding="utf-8",
        )


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
    news_sentiment_enabled: bool = bool(config.get("news_sentiment_enabled", False))
    phase2_output_dir: str | Path = config.get("phase2_output_dir", "outputs/phase2/latest")
    news_sentiment_device: str = str(config.get("news_sentiment_device", "cpu"))
    news_sentiment_export_trace: bool = bool(config.get("news_sentiment_export_trace", True))
    news_sentiment_batch_size: int = int(config.get("news_sentiment_batch_size", 32))

    # When rebuild_data is True, bypass the dataset cache only.
    # News scraper cache (news_use_cache) is intentionally preserved so that
    # repeated runs do not re-scrape the web from scratch.
    if rebuild_data:
        logger.info("rebuild_data=True → dataset cache bypassed (news cache preserved)")

    # --- Try loading from dataset cache ---
    cfg_hash = _config_hash(config)
    cache_path = _DATASET_CACHE_DIR / f"dataset_{cfg_hash}.parquet"
    if not rebuild_data:
        cached_df = _load_dataset_cache(cache_path)
        if cached_df is not None:
            cached_df = cached_df.dropna(subset=[target_col])
            if news_sentiment_export_trace and any(col in cached_df.columns for col in SENTIMENT_TRACE_COLUMNS):
                _export_sentiment_outputs(cached_df, article_trace=None, cache_hash=cfg_hash)
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
    sentiment_inferencer = None
    if news_sentiment_enabled:
        sentiment_bundle = load_phase2_phobert_inference_bundle(
            phase2_output_dir,
            device=news_sentiment_device,
        )
        sentiment_inferencer = Phase2PhoBERTInferencer(sentiment_bundle)
        logger.info("Hybrid news sentiment enabled via Phase 2 PhoBERT handoff")
    encoder = NewsEncoder(
        sentiment_inferencer=sentiment_inferencer,
        sentiment_batch_size=news_sentiment_batch_size,
    )

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
    if news_sentiment_export_trace and any(col in df_all.columns for col in SENTIMENT_TRACE_COLUMNS):
        _export_sentiment_outputs(df_all, encoder.last_sentiment_trace, cache_hash=cfg_hash)
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
            "news_emb", NEWS_HYBRID_COLUMN, "has_news", "news_count",
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
        df_all.loc[sym_mask, market_feature_cols] = sym_df[market_feature_cols].astype("float32")

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


# ---------------------------------------------------------------------------
# Single-cutoff preparation for multi-agent inference
# ---------------------------------------------------------------------------

import threading

_prepare_cache: dict[str, dict[str, Any]] = {}
_prepare_lock = threading.Lock()


def clear_prepare_cache() -> None:
    """Clear the prepare_single_cutoff cache (call between graph runs)."""
    with _prepare_lock:
        _prepare_cache.clear()


def prepare_single_cutoff(
    symbol: str,
    cutoff: str,
    sequence_len: int = 30,
    *,
    news_cache_dir: str | Path = "cache/news",
    ohlcv_source: str = "KBS",
    phase2_output_dir: str | Path = "outputs/phase2/latest",
    news_sentiment_device: str = "cpu",
) -> dict[str, Any]:
    """Prepare all market + news tensors for one (symbol, cutoff) request.

    This is the single-symbol, single-cutoff helper used by the multi-agent
    graph's Market and News agents. It reuses the same components as
    ``run_pipeline()`` but operates on exactly one window ending at ``cutoff``.

    Leakage safety: all data is filtered to ``<= cutoff`` before processing.

    Args:
        symbol: Stock ticker (e.g. "VCB", "BID").
        cutoff: ISO date string for the prediction time (e.g. "2025-03-31").
            No data after this date is used.
        sequence_len: Number of bars in the context window (default 30).
        news_cache_dir: Path to the news JSON cache directory.
        ohlcv_source: Data source for OHLCV fetching.
        phase2_output_dir: Path to PhoBERT handoff artifacts.
        news_sentiment_device: Device for PhoBERT inference.

    Returns:
        Dict with keys:
            close_window: np.ndarray (seq_len,)
            market_window: np.ndarray (seq_len, n_features)
            market_tabular: np.ndarray (n_features,)
            token_ids: np.ndarray — Chronos token IDs for the close window
            attention_mask: np.ndarray — Chronos attention mask
            news_emb: np.ndarray (seq_len, news_dim) — hybrid 773-dim embeddings
            news_mask: np.ndarray (seq_len,) — True where bar has no news
            articles: list[dict] — article metadata per bar
            sentiment_features: dict[str, float] — 6 scalar sentiment features
            vnindex_ret: float — latest VN-Index log return
            vnindex_vol_ratio: float — latest VN-Index volume ratio
    """
    # Thread-safe request-scoped cache: same (symbol, cutoff, seq_len) → same result
    cache_key = f"{symbol}_{cutoff}_{sequence_len}"

    # Fast path: check without lock
    if cache_key in _prepare_cache:
        logger.debug("prepare_single_cutoff cache HIT: {}", cache_key)
        return _prepare_cache[cache_key]

    # Serialize: only one thread computes; others block then get cache hit
    with _prepare_lock:
        # Double-check after acquiring lock
        if cache_key in _prepare_cache:
            logger.debug("prepare_single_cutoff cache HIT (after lock): {}", cache_key)
            return _prepare_cache[cache_key]

        result_dict = _compute_single_cutoff(
            symbol, cutoff, sequence_len,
            news_cache_dir=news_cache_dir,
            ohlcv_source=ohlcv_source,
            phase2_output_dir=phase2_output_dir,
            news_sentiment_device=news_sentiment_device,
        )
        _prepare_cache[cache_key] = result_dict
        return result_dict


def _compute_single_cutoff(
    symbol: str,
    cutoff: str,
    sequence_len: int,
    *,
    news_cache_dir: str | Path,
    ohlcv_source: str,
    phase2_output_dir: str | Path,
    news_sentiment_device: str,
) -> dict[str, Any]:
    """Internal computation for prepare_single_cutoff (no caching)."""
    import json as _json
    from datetime import timedelta

    cutoff_ts = pd.Timestamp(cutoff)

    # --- Fetch buffer to have enough bars after TA warm-up ---
    # We need seq_len bars of fully-computed features, but TA indicators need
    # ~30 warm-up bars themselves. Fetch 3× seq_len to be safe.
    fetch_start = (cutoff_ts - timedelta(days=sequence_len * 5)).strftime("%Y-%m-%d")
    fetch_end = cutoff_ts.strftime("%Y-%m-%d")

    fetcher = VnstockDataFetcher()
    engineer = FeatureEngineer()
    aligner = TemporalAligner()

    # 1. Fetch OHLCV (strictly <= cutoff)
    df_ohlcv = fetcher.fetch_ohlcv(symbol, fetch_start, fetch_end, "1D", ohlcv_source)
    df_ohlcv = df_ohlcv[df_ohlcv.index <= cutoff_ts]

    if len(df_ohlcv) < sequence_len:
        raise ValueError(
            f"Not enough OHLCV bars for {symbol} up to {cutoff}: "
            f"got {len(df_ohlcv)}, need {sequence_len}"
        )

    # 2. Load news from JSON cache (strictly <= cutoff)
    news_cache_path = Path(news_cache_dir)
    news_files = list(news_cache_path.glob(f"{symbol}_*.json"))
    articles_raw: list[dict] = []
    for nf in news_files:
        with open(nf, "r", encoding="utf-8") as f:
            articles_raw.extend(_json.load(f))

    # Filter by published_date <= cutoff
    df_news = pd.DataFrame(articles_raw) if articles_raw else pd.DataFrame(
        columns=["published_date", "title", "content"]
    )
    if not df_news.empty and "published_date" in df_news.columns:
        df_news["published_date"] = pd.to_datetime(df_news["published_date"])
        df_news = df_news[df_news["published_date"] <= cutoff_ts]

    # 3. Temporal alignment
    df_aligned = aligner.assign_news_to_bars(df_ohlcv, df_news)
    df_aligned = aligner.add_null_mask(df_aligned)

    # 4. Technical indicators
    df_featured = engineer.compute_technical(df_aligned)
    df_featured["symbol"] = symbol

    # 5. VN-Index macro features
    try:
        df_vnindex = fetcher.fetch_ohlcv("VNINDEX", fetch_start, fetch_end, "1D", ohlcv_source)
        df_vnindex = df_vnindex[df_vnindex.index <= cutoff_ts]
        vnidx_close = df_vnindex["close"]
        vnidx_features = pd.DataFrame(index=df_vnindex.index)
        vnidx_features["vnindex_ret"] = np.log(vnidx_close / vnidx_close.shift(1))
        vnidx_vol_ma = df_vnindex["volume"].rolling(window=20, min_periods=1).mean()
        vnidx_features["vnindex_vol_ratio"] = df_vnindex["volume"] / vnidx_vol_ma
        df_featured = df_featured.merge(
            vnidx_features, left_index=True, right_index=True, how="left",
        )
        for col in ("vnindex_ret", "vnindex_vol_ratio"):
            df_featured[col] = df_featured[col].ffill().bfill().fillna(0.0)
    except Exception:
        logger.warning("VN-Index fetch failed in prepare_single_cutoff — using zeros")
        df_featured["vnindex_ret"] = 0.0
        df_featured["vnindex_vol_ratio"] = 1.0

    # 6. News encoding (PhoBERT sentiment + vietnamese-embedding)
    try:
        sentiment_bundle = load_phase2_phobert_inference_bundle(
            phase2_output_dir, device=news_sentiment_device,
        )
        sentiment_inferencer = Phase2PhoBERTInferencer(sentiment_bundle)
    except Exception:
        logger.warning("PhoBERT bundle load failed — news encoding without sentiment")
        sentiment_inferencer = None

    encoder = NewsEncoder(sentiment_inferencer=sentiment_inferencer)
    df_featured = encoder.encode_dataframe(df_featured, text_col="news_content", use_cache=True)

    # 7. Take the last seq_len bars (the window ending at cutoff)
    # Drop rows without valid features (TA warm-up NaN rows)
    # Use explicit canonical column order matching training dataset
    _CANONICAL_MARKET_COLS = [
        "open", "high", "low", "close", "volume",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_lower", "bb_mid", "bb_upper", "atr_14",
        "vol_ratio", "log_ret",
        "vnindex_ret", "vnindex_vol_ratio",
        "sentiment_mean", "sentiment_max_abs",
        "sentiment_positive_ratio", "sentiment_negative_ratio",
        "sentiment_score_count", "sentiment_missing_flag",
    ]
    # Only include columns actually present in df_featured
    market_cols = [c for c in _CANONICAL_MARKET_COLS if c in df_featured.columns]

    # Forward-fill any remaining NaNs in market features, then drop rows still NaN
    for col in market_cols:
        df_featured[col] = df_featured[col].ffill()
    df_valid = df_featured.dropna(subset=market_cols)

    if len(df_valid) < sequence_len:
        raise ValueError(
            f"Not enough valid bars for {symbol} after TA warm-up: "
            f"got {len(df_valid)}, need {sequence_len}"
        )

    window_df = df_valid.iloc[-sequence_len:]

    # 8. Extract arrays
    close_window = window_df["close"].values.astype(np.float32)
    market_window = window_df[market_cols].values.astype(np.float32)
    market_tabular = market_window[-1]  # last bar as tabular summary

    # News embeddings
    news_col = NEWS_HYBRID_COLUMN if NEWS_HYBRID_COLUMN in window_df.columns else "news_emb"
    if news_col in window_df.columns:
        news_embs = window_df[news_col].tolist()
        # Stack into (seq_len, news_dim)
        news_dim = len(news_embs[-1]) if isinstance(news_embs[-1], np.ndarray) else 773
        news_emb = np.zeros((sequence_len, news_dim), dtype=np.float32)
        for i, emb in enumerate(news_embs):
            if isinstance(emb, np.ndarray) and emb.size > 0:
                news_emb[i] = emb
    else:
        news_emb = np.zeros((sequence_len, 773), dtype=np.float32)

    # News mask: True where the bar has no news (zero embedding)
    news_mask = news_emb.sum(axis=-1) == 0  # (seq_len,)

    # Collect article metadata per bar, with sentiment scores
    # Build a title→score lookup from the encoder's sentiment trace or re-score
    title_score_map: dict[tuple[int, str], float] = {}
    if encoder.last_sentiment_trace is not None and not encoder.last_sentiment_trace.empty:
        # Trace available from fresh encoding (not from cache)
        # row_position is relative to df_featured (the full dataframe passed to encode_dataframe)
        trace_df = encoder.last_sentiment_trace
        for row in trace_df.itertuples(index=False):
            key = (int(row.row_position), str(getattr(row, "title_raw", "")))
            title_score_map[key] = float(row.sentiment_score)
    elif sentiment_inferencer is not None and "news_titles" in window_df.columns:
        # Cache hit path: re-score titles for just the window (0-based positions)
        _scores_by_row, _trace = encoder._score_titles(window_df.reset_index())
        if not _trace.empty:
            for row in _trace.itertuples(index=False):
                key = (int(row.row_position), str(getattr(row, "title_raw", "")))
                title_score_map[key] = float(row.sentiment_score)

    # Map window bar indices to df_featured iloc positions (for trace lookup)
    # window_df comes from df_valid.iloc[-sequence_len:], df_valid is a subset of df_featured
    window_df_featured_ilocs = [
        df_featured.index.get_loc(ts) for ts in window_df.index
    ] if encoder.last_sentiment_trace is not None else []

    articles_out: list[dict] = []
    if "news_titles" in window_df.columns:
        for idx, (_, row) in enumerate(window_df.iterrows()):
            titles = row.get("news_titles", [])
            if isinstance(titles, list):
                for title in titles:
                    # Look up score: try window-local index (cache path)
                    score = title_score_map.get((idx, title))
                    if score is None and idx < len(window_df_featured_ilocs):
                        # Try global df_featured position (fresh encode path)
                        score = title_score_map.get((window_df_featured_ilocs[idx], title))
                    articles_out.append({
                        "title": title,
                        "published_at": str(row.name),
                        "bar_index": idx,
                        "sentiment_score": score,
                    })

    # Sentiment scalar features (window-level aggregates, not just last bar)
    sentiment_features = {}
    for col in SENTIMENT_TRACE_COLUMNS:
        if col in window_df.columns:
            values = window_df[col].dropna()
            sentiment_features[col] = float(values.mean()) if len(values) > 0 else 0.0
        else:
            sentiment_features[col] = 0.0

    # 9. Tokenize close window for Chronos
    # Tokenization requires the backbone — defer to caller if not available here
    # Return raw close_window; the fusion_agent will tokenize via the backbone
    token_ids = np.array([], dtype=np.int64)
    attention_mask_arr = np.array([], dtype=np.int64)

    result_dict = {
        "close_window": close_window,
        "market_window": market_window,
        "market_tabular": market_tabular,
        "token_ids": token_ids,
        "attention_mask": attention_mask_arr,
        "news_emb": news_emb,
        "news_mask": news_mask,
        "articles": articles_out,
        "sentiment_features": sentiment_features,
        "market_feature_cols": market_cols,
    }
    return result_dict

