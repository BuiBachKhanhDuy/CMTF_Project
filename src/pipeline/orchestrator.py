"""Build datasets from market data, aligned news, and engineered features."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.sentiment import PhoBERTInferencer, load_phobert_inference_bundle

from .data_fetcher import VnstockDataFetcher
from .temporal_aligner import TemporalAligner
from .feature_engineer import FeatureEngineer
from .news_encoder import NEWS_HYBRID_COLUMN, SENTIMENT_TRACE_COLUMNS, NewsEncoder
from .dataset_builder import CMTFDataset

_DATASET_CACHE_DIR = Path("./cache/dataset")
_SENTIMENT_EXPORT_DIR = Path("./artifacts/hybrid_sentiment")

# Increment these versions when the dataset schema or merge behavior changes.
_DATASET_SCHEMA_VERSION = "dataset_schema_v2"
_VNINDEX_MERGE_VERSION = "vnindex_merge_on_time_v2_mom"

# All VNINDEX-derived feature columns (single source of truth for merge fill/validation).
_VNINDEX_FEATURE_COLS = (
    "vnindex_ret",
    "vnindex_vol_ratio",
    "vnindex_mom_5d",
    "vnindex_mom_20d",
)

# Canonical market-only schema used by both pipeline and single-cutoff path.
# IMPORTANT:
# - Keep this pure market-only: OHLCV, technical indicators, macro index features.
# - Do NOT include sentiment-derived columns here.
_CANONICAL_MARKET_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_lower",
    "bb_mid",
    "bb_upper",
    "atr_14",
    "vol_ratio",
    "log_ret",
    "vnindex_ret",
    "vnindex_vol_ratio",
    "vnindex_mom_5d",
    "vnindex_mom_20d",
]


# -----------------------------------------------------------------------------
# Hashing / cache utilities
# -----------------------------------------------------------------------------

def _config_hash(config: dict[str, Any]) -> str:
    """Compute a short hash of config keys and schema choices that affect the dataset."""
    keys = [
        "symbols",
        "start",
        "end",
        "interval",
        "ohlcv_source",
        "news_source",
        "news_sources",
        "news_similarity_threshold",
        "sequence_len",
        "horizon",
        "target_horizon_days",
        "train_end",
        "val_end",
        "normalize_method",
        "news_sentiment_enabled",
        "sentiment_output_dir",
        "news_sentiment_device",
        "news_sentiment_batch_size",
        "news_use_cache",
    ]

    h = hashlib.sha256()
    h.update(_DATASET_SCHEMA_VERSION.encode())
    h.update(_VNINDEX_MERGE_VERSION.encode())
    h.update(("market_cols=" + ",".join(_CANONICAL_MARKET_COLS)).encode())

    for k in keys:
        h.update(f"{k}={config.get(k)}".encode())

    return h.hexdigest()[:16]


def _save_dataset_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save the processed DataFrame to parquet (news embeddings stored as bytes)."""
    df_out = df.copy()

    for col in ("news_emb", NEWS_HYBRID_COLUMN):
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
            )

    # Drop large / object-heavy raw text fields from cache payload
    drop_cols = [c for c in ("news_titles", "news_content", "news_title_sentiment_scores") if c in df_out.columns]
    for c in df_out.columns:
        if df_out[c].dtype == object and c not in ("news_emb", NEWS_HYBRID_COLUMN, "symbol"):
            drop_cols.append(c)

    if drop_cols:
        df_out = df_out.drop(columns=list(set(drop_cols)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(cache_path, index=True)
    logger.info("Dataset cached → {} ({} rows)", cache_path.name, len(df_out))


def _load_dataset_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached DataFrame and restore news embeddings from bytes."""
    if not cache_path.exists():
        return None

    try:
        df = pd.read_parquet(cache_path)

        for col in ("news_emb", NEWS_HYBRID_COLUMN):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda b: np.frombuffer(b, dtype=np.float32).copy()
                    if isinstance(b, (bytes, bytearray)) else b
                )

        if "news_titles" not in df.columns:
            df["news_titles"] = [[] for _ in range(len(df))]
        if "news_content" not in df.columns:
            df["news_content"] = [[] for _ in range(len(df))]

        date_range = ""
        try:
            idx = pd.to_datetime(df.index)
            date_range = f", {idx.min()} → {idx.max()}"
        except Exception:
            pass

        logger.info(
            "Dataset loaded from cache: {} ({} rows{})",
            cache_path.name,
            len(df),
            date_range,
        )
        return df

    except Exception:
        logger.warning("Corrupt dataset cache {} — rebuilding", cache_path.name)
        return None


# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------

def _resolve_market_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return canonical market feature columns present and numeric in df.

    This helper keeps training and single-cutoff paths aligned.
    """
    market_cols = [
        c for c in _CANONICAL_MARKET_COLS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    missing_canonical = [c for c in _CANONICAL_MARKET_COLS if c not in df.columns]
    if missing_canonical:
        logger.warning("Missing canonical market columns: {}", missing_canonical)

    if not market_cols:
        raise ValueError("No canonical market feature columns available")

    non_numeric = [
        c for c in market_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Canonical market columns must be numeric: {non_numeric}")

    return market_cols


# -----------------------------------------------------------------------------
# Sentiment exports
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Macro feature helpers
# -----------------------------------------------------------------------------

def _build_vnindex_features(df_vnindex: pd.DataFrame) -> pd.DataFrame:
    """Build VNINDEX macro features indexed by time."""
    vnidx_close = df_vnindex["close"]
    vnidx_features = pd.DataFrame(index=df_vnindex.index)
    vnidx_ret = np.log(vnidx_close / vnidx_close.shift(1))
    vnidx_features["vnindex_ret"] = vnidx_ret
    vnidx_vol_ma = df_vnindex["volume"].rolling(window=20, min_periods=1).mean()
    vnidx_features["vnindex_vol_ratio"] = df_vnindex["volume"] / vnidx_vol_ma
    # Stationary macro-momentum features: rolling sums of log returns.
    # Orthogonal to absolute price-level features; safe under per-window instance norm.
    vnidx_ret_filled = vnidx_ret.fillna(0.0)
    vnidx_features["vnindex_mom_5d"] = vnidx_ret_filled.rolling(5, min_periods=1).sum()
    vnidx_features["vnindex_mom_20d"] = vnidx_ret_filled.rolling(20, min_periods=1).sum()
    return vnidx_features

def _merge_vnindex_features_multi_symbol(df_all: pd.DataFrame, vnidx_features: pd.DataFrame) -> pd.DataFrame:
    """Safely merge VNINDEX features into multi-symbol training data by time column."""
    df_left = df_all.reset_index().rename(columns={df_all.index.name or "index": "time"}).copy()
    df_right = vnidx_features.reset_index().rename(columns={vnidx_features.index.name or "index": "time"}).copy()

    # Normalize timezone handling before merge:
    # convert tz-aware timestamps to naive local timestamps
    if pd.api.types.is_datetime64tz_dtype(df_left["time"]):
        df_left["time"] = df_left["time"].dt.tz_localize(None)
    else:
        df_left["time"] = pd.to_datetime(df_left["time"])

    if pd.api.types.is_datetime64tz_dtype(df_right["time"]):
        df_right["time"] = df_right["time"].dt.tz_localize(None)
    else:
        df_right["time"] = pd.to_datetime(df_right["time"])

    merged = df_left.merge(df_right, on="time", how="left")
    merged = merged.set_index("time")

    for col in _VNINDEX_FEATURE_COLS:
        if col not in merged.columns:
            raise ValueError(f"Missing merged macro column: {col}")
        merged[col] = merged[col].ffill().bfill().fillna(0.0)

    return merged

def _merge_vnindex_features_single_symbol(df_featured: pd.DataFrame, vnidx_features: pd.DataFrame) -> pd.DataFrame:
    """Merge VNINDEX features into a single-symbol frame by aligned index."""
    left = df_featured.copy()
    right = vnidx_features.copy()

    if isinstance(left.index, pd.DatetimeIndex) and left.index.tz is not None:
        left.index = left.index.tz_localize(None)
    else:
        left.index = pd.to_datetime(left.index)

    if isinstance(right.index, pd.DatetimeIndex) and right.index.tz is not None:
        right.index = right.index.tz_localize(None)
    else:
        right.index = pd.to_datetime(right.index)

    merged = left.merge(right, left_index=True, right_index=True, how="left")

    for col in _VNINDEX_FEATURE_COLS:
        if col not in merged.columns:
            raise ValueError(f"Missing merged macro column: {col}")
        merged[col] = merged[col].ffill().bfill().fillna(0.0)

    return merged


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(config: dict[str, Any], allow_missing_target: bool = False) -> CMTFDataset:
    """Execute the full CMTF data-ingestion pipeline.

    ``allow_missing_target``: default False preserves exact existing behaviour — every
    research/ablation result in this project is built on the dropna below removing any
    row whose forward-return target is NaN (correct: training/backtesting need a real
    label). True is used ONLY by the live-inference path (``live_inference.py``): the
    most recent ~horizon trading days before ``end`` always have a NaN target (the
    future hasn't happened yet), so dropping them silently makes it impossible to ever
    serve a prediction for a genuinely current/live date. Cached under a distinct path
    (see ``cache_path`` below) so it can never collide with — or be served by — the
    standard research cache.
    """
    symbols: list[str] = config["symbols"]
    start: str = config["start"]
    end: str = config["end"]
    interval: str = config.get("interval", "1D")
    ohlcv_source: str = config.get("ohlcv_source", "KBS")
    news_source: str = config.get("news_source", "VCI")
    news_sources: tuple[str, ...] = tuple(config.get("news_sources", ("cafef_banking", "vietstock")))
    news_use_cache: bool = bool(config.get("news_use_cache", True))
    news_export_trace: bool = bool(config.get("news_export_trace", True))
    news_similarity_threshold: float = float(config.get("news_similarity_threshold", 85.0))
    log_news_coverage: bool = bool(config.get("log_news_coverage", True))
    seq_len: int = int(config.get("sequence_len", 30))
    horizon: int = int(config.get("horizon", 1))
    target_horizon_days: int = int(config.get("target_horizon_days", 1))
    target_col = f"fwd_ret_{target_horizon_days}d"
    train_end: str = config["train_end"]
    val_end: str = config["val_end"]
    norm_method: str = config.get("normalize_method", "zscore")
    rebuild_data: bool = bool(config.get("rebuild_data", False))
    news_sentiment_enabled: bool = bool(config.get("news_sentiment_enabled", False))
    sentiment_output_dir: str | Path = config.get("sentiment_output_dir", "outputs/sentiment/latest")
    news_sentiment_device: str = str(config.get("news_sentiment_device", "cpu"))
    news_sentiment_export_trace: bool = bool(config.get("news_sentiment_export_trace", True))
    news_sentiment_batch_size: int = int(config.get("news_sentiment_batch_size", 32))

    if rebuild_data:
        logger.info("rebuild_data=True → dataset cache bypassed (news cache preserved)")

    cfg_hash = _config_hash(config)
    cache_suffix = "_livewide" if allow_missing_target else ""
    cache_path = _DATASET_CACHE_DIR / f"dataset_{cfg_hash}{cache_suffix}.parquet"

    if not rebuild_data:
        cached_df = _load_dataset_cache(cache_path)
        if cached_df is not None:
            if not allow_missing_target:
                cached_df = cached_df.dropna(subset=[target_col])

            if news_sentiment_export_trace and any(col in cached_df.columns for col in SENTIMENT_TRACE_COLUMNS):
                _export_sentiment_outputs(cached_df, article_trace=None, cache_hash=cfg_hash)

            dataset = CMTFDataset(
                df_featured=cached_df,
                sequence_len=seq_len,
                horizon=horizon,
                target_horizon_days=target_horizon_days,
                allow_missing_target=allow_missing_target,
            )
            logger.info("Pipeline complete (from cache) | dataset length = {}", len(dataset))
            return dataset

    fetcher = VnstockDataFetcher()
    aligner = TemporalAligner()
    engineer = FeatureEngineer()

    sentiment_inferencer = None
    if news_sentiment_enabled:
        sentiment_bundle = load_phobert_inference_bundle(
            sentiment_output_dir,
            device=news_sentiment_device,
        )
        sentiment_inferencer = PhoBERTInferencer(sentiment_bundle)
        logger.info("Hybrid news sentiment enabled via sentiment-encoder PhoBERT handoff")

    encoder = NewsEncoder(
        sentiment_inferencer=sentiment_inferencer,
        sentiment_batch_size=news_sentiment_batch_size,
    )

    all_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        logger.info("━━━ Processing {} ━━━", symbol)

        try:
            df_ohlcv = fetcher.fetch_ohlcv(symbol, start, end, interval, ohlcv_source)
        except Exception:
            logger.exception("Skipping {} — OHLCV fetch failed", symbol)
            continue

        try:
            mode = str(news_source).strip().lower()
            if mode in {"vci", "api"}:
                df_news = fetcher.fetch_news(symbol, source="VCI")
                if not df_news.empty:
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    df_news = df_news[
                        (df_news["published_date"] >= start_ts) &
                        (df_news["published_date"] <= end_ts)
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

        df_featured = engineer.compute_technical(df_aligned)
        df_featured["symbol"] = symbol
        all_frames.append(df_featured)

    if not all_frames:
        raise ValueError("No data fetched for any symbol — aborting pipeline")

    df_all = pd.concat(all_frames, axis=0)
    df_all = df_all.sort_index()
    logger.info("Combined DataFrame: {} rows × {} cols", *df_all.shape)

    # Merge VNINDEX macro features
    try:
        df_vnindex = fetcher.fetch_ohlcv("VNINDEX", start, end, interval, ohlcv_source)
        vnidx_features = _build_vnindex_features(df_vnindex)
        df_all = _merge_vnindex_features_multi_symbol(df_all, vnidx_features)
        logger.info("VN-Index macro features added (vnindex_ret, vnindex_vol_ratio)")
    except Exception:
        logger.exception("VN-Index macro feature pipeline failed — proceeding without macro features")
        df_all["vnindex_ret"] = 0.0
        df_all["vnindex_vol_ratio"] = 1.0

    # Encode news
    df_all = encoder.encode_dataframe(df_all, text_col="news_content", use_cache=(not rebuild_data))

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

    # Resolve canonical market features
    market_feature_cols = _resolve_market_feature_cols(df_all)

    for col in market_feature_cols:
        if df_all[col].dtype in ("int64", "int32"):
            df_all[col] = df_all[col].astype("float64")

    # Normalize per symbol using train-only fit
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

    # Keep target-valid rows only (skipped when allow_missing_target=True — the live-
    # inference path needs the most recent rows precisely BECAUSE their target is NaN).
    if not allow_missing_target:
        df_all = df_all.dropna(subset=[target_col])

    dataset = CMTFDataset(
        df_featured=df_all,
        sequence_len=seq_len,
        horizon=horizon,
        target_horizon_days=target_horizon_days,
        allow_missing_target=allow_missing_target,
    )

    _save_dataset_cache(df_all, cache_path)
    logger.info("Pipeline complete | dataset length = {}", len(dataset))
    return dataset


# -----------------------------------------------------------------------------
# Single-cutoff preparation for multi-agent inference
# -----------------------------------------------------------------------------

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
    sentiment_output_dir: str | Path = "outputs/sentiment/latest",
    news_sentiment_device: str = "cpu",
) -> dict[str, Any]:
    """Prepare all market + news tensors for one (symbol, cutoff) request."""
    cache_key = f"{symbol}_{cutoff}_{sequence_len}"

    if cache_key in _prepare_cache:
        logger.debug("prepare_single_cutoff cache HIT: {}", cache_key)
        return _prepare_cache[cache_key]

    with _prepare_lock:
        if cache_key in _prepare_cache:
            logger.debug("prepare_single_cutoff cache HIT (after lock): {}", cache_key)
            return _prepare_cache[cache_key]

        result_dict = _compute_single_cutoff(
            symbol,
            cutoff,
            sequence_len,
            news_cache_dir=news_cache_dir,
            ohlcv_source=ohlcv_source,
            sentiment_output_dir=sentiment_output_dir,
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
    sentiment_output_dir: str | Path,
    news_sentiment_device: str,
) -> dict[str, Any]:
    """Internal computation for prepare_single_cutoff (no caching)."""
    import json as _json
    from datetime import timedelta

    cutoff_ts = pd.Timestamp(cutoff)
    fetch_start = (cutoff_ts - timedelta(days=sequence_len * 5)).strftime("%Y-%m-%d")
    fetch_end = cutoff_ts.strftime("%Y-%m-%d")

    fetcher = VnstockDataFetcher()
    engineer = FeatureEngineer()
    aligner = TemporalAligner()

    df_ohlcv = fetcher.fetch_ohlcv(symbol, fetch_start, fetch_end, "1D", ohlcv_source)
    df_ohlcv = df_ohlcv[df_ohlcv.index <= cutoff_ts]

    if len(df_ohlcv) < sequence_len:
        raise ValueError(
            f"Not enough OHLCV bars for {symbol} up to {cutoff}: got {len(df_ohlcv)}, need {sequence_len}"
        )

    news_cache_path = Path(news_cache_dir)
    news_files = list(news_cache_path.glob(f"{symbol}_*.json"))
    articles_raw: list[dict] = []
    for nf in news_files:
        with open(nf, "r", encoding="utf-8") as f:
            articles_raw.extend(_json.load(f))

    df_news = (
        pd.DataFrame(articles_raw)
        if articles_raw
        else pd.DataFrame(columns=["published_date", "title", "content"])
    )
    if not df_news.empty and "published_date" in df_news.columns:
        df_news["published_date"] = pd.to_datetime(df_news["published_date"])
        df_news = df_news[df_news["published_date"] <= cutoff_ts]

    df_aligned = aligner.assign_news_to_bars(df_ohlcv, df_news)
    df_aligned = aligner.add_null_mask(df_aligned)

    df_featured = engineer.compute_technical(df_aligned)
    df_featured["symbol"] = symbol

    try:
        df_vnindex = fetcher.fetch_ohlcv("VNINDEX", fetch_start, fetch_end, "1D", ohlcv_source)
        df_vnindex = df_vnindex[df_vnindex.index <= cutoff_ts]
        vnidx_features = _build_vnindex_features(df_vnindex)
        df_featured = _merge_vnindex_features_single_symbol(df_featured, vnidx_features)
    except Exception:
        logger.exception("VN-Index macro feature pipeline failed in prepare_single_cutoff — using zeros")
        df_featured["vnindex_ret"] = 0.0
        df_featured["vnindex_vol_ratio"] = 1.0

    try:
        sentiment_bundle = load_phobert_inference_bundle(
            sentiment_output_dir,
            device=news_sentiment_device,
        )
        sentiment_inferencer = PhoBERTInferencer(sentiment_bundle)
    except Exception:
        logger.warning("PhoBERT bundle load failed — news encoding without sentiment")
        sentiment_inferencer = None

    encoder = NewsEncoder(sentiment_inferencer=sentiment_inferencer)
    df_featured = encoder.encode_dataframe(df_featured, text_col="news_content", use_cache=True)

    market_cols = _resolve_market_feature_cols(df_featured)

    for col in market_cols:
        df_featured[col] = df_featured[col].ffill()

    df_valid = df_featured.dropna(subset=market_cols)

    if len(df_valid) < sequence_len:
        raise ValueError(
            f"Not enough valid bars for {symbol} after TA warm-up: got {len(df_valid)}, need {sequence_len}"
        )

    window_df = df_valid.iloc[-sequence_len:]

    close_window = window_df["close"].values.astype(np.float32)
    market_window = window_df[market_cols].values.astype(np.float32)
    market_tabular = market_window[-1]

    news_col = NEWS_HYBRID_COLUMN if NEWS_HYBRID_COLUMN in window_df.columns else "news_emb"
    if news_col in window_df.columns:
        news_embs = window_df[news_col].tolist()
        news_dim = len(news_embs[-1]) if isinstance(news_embs[-1], np.ndarray) else 773
        news_emb = np.zeros((sequence_len, news_dim), dtype=np.float32)
        for i, emb in enumerate(news_embs):
            if isinstance(emb, np.ndarray) and emb.size > 0:
                news_emb[i] = emb
    else:
        news_emb = np.zeros((sequence_len, 773), dtype=np.float32)

    news_mask = news_emb.sum(axis=-1) == 0

    title_score_map: dict[tuple[int, str], float] = {}
    if encoder.last_sentiment_trace is not None and not encoder.last_sentiment_trace.empty:
        trace_df = encoder.last_sentiment_trace
        for row in trace_df.itertuples(index=False):
            key = (int(row.row_position), str(getattr(row, "title_raw", "")))
            title_score_map[key] = float(row.sentiment_score)
    elif sentiment_inferencer is not None and "news_titles" in window_df.columns:
        _scores_by_row, _trace = encoder._score_titles(window_df.reset_index())
        if not _trace.empty:
            for row in _trace.itertuples(index=False):
                key = (int(row.row_position), str(getattr(row, "title_raw", "")))
                title_score_map[key] = float(row.sentiment_score)

    window_df_featured_ilocs = [
        df_featured.index.get_loc(ts) for ts in window_df.index
    ] if encoder.last_sentiment_trace is not None else []

    articles_out: list[dict] = []
    if "news_titles" in window_df.columns:
        for idx, (_, row) in enumerate(window_df.iterrows()):
            titles = row.get("news_titles", [])
            if isinstance(titles, list):
                for title in titles:
                    score = title_score_map.get((idx, title))
                    if score is None and idx < len(window_df_featured_ilocs):
                        score = title_score_map.get((window_df_featured_ilocs[idx], title))
                    articles_out.append({
                        "title": title,
                        "published_at": str(row.name),
                        "bar_index": idx,
                        "sentiment_score": score,
                    })

    sentiment_features = {}
    for col in SENTIMENT_TRACE_COLUMNS:
        if col in window_df.columns:
            values = window_df[col].dropna()
            sentiment_features[col] = float(values.mean()) if len(values) > 0 else 0.0
        else:
            sentiment_features[col] = 0.0

    token_ids = np.array([], dtype=np.int64)
    attention_mask_arr = np.array([], dtype=np.int64)

    return {
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
