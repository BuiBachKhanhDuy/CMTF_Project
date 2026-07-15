"""CMTF Data Pipeline — News Encoder module.

Encodes Vietnamese news text into dense 768-dim embeddings using a
PhoBERT-based sentence transformer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.sentiment import (
    PhoBERTInferencer,
    aggregate_title_sentiment_scores,
    flatten_sentiment_scores,
)

_NEWS_DIM = 768
_DEFAULT_MODEL_NAME = "dangvantuan/vietnamese-embedding"
_EMBEDDINGS_CACHE_DIR = Path("./cache/embeddings")

# Per-ROW embedding cache (keyed on that row's own symbol/date/text content, not the
# whole dataframe). The whole-dataframe cache above (`encode_dataframe`'s cache_key)
# invalidates completely whenever ANY row changes — extending the pipeline's `end`
# date by even one day changes len(df)/date range/content hash, forcing a full
# re-encode of the entire historical corpus (confirmed: ~35-45 min for ~11k rows on
# CPU) every time a live-inference query needs one new day. This layer sits between
# that whole-df cache and the model: rows whose own content is unchanged are served
# from here regardless of what else changed in the surrounding dataframe, so a
# live-inference rebuild only ever pays for genuinely NEW rows.
_ROW_CACHE_PATH = _EMBEDDINGS_CACHE_DIR / "row_cache_v1.joblib"


def _row_cache_key(symbol: Any, date: Any, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(b"news_row_cache_v1")
    h.update(str(symbol).encode())
    h.update(str(date).encode())
    for t in texts:
        h.update(str(t).encode())
    return h.hexdigest()


def _load_row_cache() -> dict[str, dict[str, Any]]:
    if _ROW_CACHE_PATH.exists():
        try:
            return joblib.load(_ROW_CACHE_PATH)
        except Exception:
            logger.warning("Corrupt row embedding cache — starting fresh")
    return {}


def _save_row_cache(cache: dict[str, dict[str, Any]]) -> None:
    _EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, _ROW_CACHE_PATH, compress=3)
SENTIMENT_FEATURE_COLUMNS = (
    "sentiment_mean",
    "sentiment_max_abs",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
    "sentiment_score_count",
)
SENTIMENT_TRACE_COLUMNS = SENTIMENT_FEATURE_COLUMNS + ("sentiment_missing_flag",)
NEWS_HYBRID_COLUMN = "news_hybrid_emb"


class NewsEncoder:
    """Encodes Vietnamese news text into sentence embeddings.

    Uses the ``dangvantuan/vietnamese-embedding`` model (768-dim,
    PhoBERT-based) via the ``sentence-transformers`` library.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: Optional[str] = None,
        sentiment_inferencer: PhoBERTInferencer | None = None,
        sentiment_batch_size: int = 32,
        use_weighted_pooling: bool = True,
        export_hybrid_embedding: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device
        self.sentiment_inferencer = sentiment_inferencer
        self.sentiment_batch_size = int(sentiment_batch_size)
        self.use_weighted_pooling = bool(use_weighted_pooling)
        self.export_hybrid_embedding = bool(export_hybrid_embedding)
        self.last_sentiment_trace: pd.DataFrame | None = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
            # The model's max_position_embeddings (258) is less than the
            # default max_seq_length (512).  Clamp to avoid IndexError in
            # the RoBERTa position embedding layer.
            max_pos = self._model[0].auto_model.config.max_position_embeddings
            if self._model.max_seq_length > max_pos:
                logger.warning(
                    "Clamping max_seq_length {} → {} (max_position_embeddings)",
                    self._model.max_seq_length, max_pos,
                )
                self._model.max_seq_length = max_pos
        return self._model

    # ------------------------------------------------------------------
    # Single-window encoding
    # ------------------------------------------------------------------
    def encode_window(
        self,
        texts: list[str],
        null_mask: bool = False,
        weights: list[float] | None = None,
    ) -> dict[str, Any]:
        """Encode a list of news texts for one time window.

        Args:
            texts: List of article texts. May be empty.
            null_mask: If ``True``, force a zero-vector output
                (used for explicit ``[NO_NEWS]`` injection).
            weights: Optional per-article sentiment scores used for
                magnitude-weighted pooling when ``use_weighted_pooling``
                is True. Must be the same length as ``texts`` after
                filtering. Falls back to mean-pooling on mismatch.

        Returns:
            Dict with keys:
                - ``'embedding'``: ``np.ndarray`` of shape ``(768,)``
                - ``'has_news'``: ``bool``
        """
        if null_mask or not texts:
            return {
                "embedding": np.zeros(_NEWS_DIM, dtype=np.float32),
                "has_news": False,
            }

        # Filter out empty / whitespace-only strings
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return {
                "embedding": np.zeros(_NEWS_DIM, dtype=np.float32),
                "has_news": False,
            }

        # Truncate long texts to stay within the model's max token limit.
        # PhoBERT's tokenizer does not auto-truncate, so we cut at the
        # character level (rough proxy ≈ 1.5 chars/token for Vietnamese).
        max_chars = int(self.model.max_seq_length * 1.5)
        valid_texts = [t[:max_chars] for t in valid_texts]

        # Enable tokenizer-level truncation as a safety net
        self.model.tokenizer.model_max_length = self.model.max_seq_length
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Weighted pool: weight by sentiment magnitude so polarised articles
        # dominate over neutral ones. Falls back to mean-pool on any mismatch.
        if (
            self.use_weighted_pooling
            and weights is not None
            and len(weights) == len(valid_texts)
        ):
            w = np.abs(np.asarray(weights, dtype=np.float32)) + 1e-6
            pooled = np.average(embeddings, axis=0, weights=w).astype(np.float32)
        else:
            pooled = np.mean(embeddings, axis=0).astype(np.float32)
        return {"embedding": pooled, "has_news": True}

    # ------------------------------------------------------------------
    # DataFrame-level encoding
    # ------------------------------------------------------------------
    def encode_dataframe(
        self,
        df_aligned: pd.DataFrame,
        text_col: str = "news_content",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Encode news for every row in an aligned OHLCV+news DataFrame.

        Args:
            df_aligned: Output of
                :meth:`TemporalAligner.assign_news_to_bars`.
                Must contain ``text_col`` (list[str] per row) and
                ``news_missing_flag`` or ``has_news``.
            text_col: Column containing lists of news texts.
            use_cache: If True, load/save embeddings from disk cache.

        Returns:
            DataFrame with added columns ``news_emb`` (np.ndarray 768-dim)
            and ``has_news`` (bool).
        """
        df = df_aligned.copy()

        # --- Cache logic ---
        cache_key = self._compute_cache_key(
            df,
            text_col,
            self.model_name,
            sentiment_enabled=self.sentiment_inferencer is not None,
            sentiment_signature=self._sentiment_signature(),
            weighted_pooling=self.use_weighted_pooling,
        )
        cache_path = _EMBEDDINGS_CACHE_DIR / f"{cache_key}.npz"

        if use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path)
                emb_array = cached["embeddings"]
                has_news_array = cached["has_news"]
                hybrid_array = cached[NEWS_HYBRID_COLUMN] if NEWS_HYBRID_COLUMN in cached.files else None
                expected_hybrid = self.sentiment_inferencer is not None and self.export_hybrid_embedding
                has_expected_hybrid = (hybrid_array is not None) == expected_hybrid
                if len(emb_array) == len(df) and has_expected_hybrid:
                    df["news_emb"] = list(emb_array)
                    df["has_news"] = has_news_array.astype(bool)
                    for col in SENTIMENT_TRACE_COLUMNS:
                        if col in cached.files:
                            df[col] = cached[col].astype(np.float32)
                    if hybrid_array is not None:
                        df[NEWS_HYBRID_COLUMN] = list(hybrid_array)
                    self.last_sentiment_trace = None
                    logger.info(
                        "Embeddings loaded from cache: {} ({} rows)",
                        cache_path.name, len(df),
                    )
                    return df
                else:
                    logger.warning(
                        "Embedding cache schema mismatch (rows={} expected_rows={} hybrid_ok={}) — re-encoding",
                        len(emb_array), len(df), has_expected_hybrid,
                    )
            except Exception:
                logger.warning("Corrupt embedding cache {} — re-encoding", cache_path.name)

        sentiment_scores_by_row, sentiment_trace = self._score_titles(df)
        self.last_sentiment_trace = sentiment_trace

        embeddings: list[np.ndarray] = []
        has_news_flags: list[bool] = []
        hybrid_embeddings: list[np.ndarray] = []
        sentiment_records: list[dict[str, float]] = []

        # Per-row cache is only safe to use in the simple (no sentiment-weighted
        # pooling) path: sentiment scores feed into `encode_window` as pooling
        # weights, and caching would need to key on those too. Sentiment fusion is
        # off by default in production (`news_sentiment_enabled=False`), so this
        # covers the common case; the sentiment-enabled path re-encodes as before.
        use_row_cache = use_cache and self.sentiment_inferencer is None
        row_cache = _load_row_cache() if use_row_cache else {}
        row_cache_hits = 0
        has_symbol_col = "symbol" in df.columns
        has_time_col = "time" in df.columns

        logger.info("Encoding news for {} rows …", len(df))
        for pos in tqdm(range(len(df)), desc="Encoding news"):
            texts = df.iloc[pos][text_col]
            is_null = df.iloc[pos]["news_missing_flag"] if "news_missing_flag" in df.columns else (not texts)

            # Pass sentiment scores as pooling weights when available.
            # len(sentiment_scores_by_row[pos]) matches the number of titles
            # (not content articles), so only use when lengths agree.
            row_texts = texts if isinstance(texts, list) else []
            row_weights = sentiment_scores_by_row[pos]

            row_key = None
            if use_row_cache:
                sym = df.iloc[pos]["symbol"] if has_symbol_col else ""
                tm = df.iloc[pos]["time"] if has_time_col else pos
                row_key = _row_cache_key(sym, tm, row_texts if not is_null else [])
                cached_row = row_cache.get(row_key)
                if cached_row is not None:
                    result = {"embedding": cached_row["embedding"], "has_news": cached_row["has_news"]}
                    row_cache_hits += 1
                else:
                    result = self.encode_window(
                        texts=row_texts,
                        null_mask=bool(is_null),
                        weights=row_weights if len(row_weights) == len(row_texts) else None,
                    )
                    row_cache[row_key] = {"embedding": result["embedding"], "has_news": result["has_news"]}
            else:
                result = self.encode_window(
                    texts=row_texts,
                    null_mask=bool(is_null),
                    weights=row_weights if len(row_weights) == len(row_texts) else None,
                )
            embeddings.append(result["embedding"])
            has_news_flags.append(result["has_news"])

            sentiment_features = aggregate_title_sentiment_scores(sentiment_scores_by_row[pos])
            sentiment_records.append(sentiment_features)
            if self.sentiment_inferencer is not None and self.export_hybrid_embedding:
                hybrid_vec = np.concatenate(
                    [
                        result["embedding"],
                        np.asarray(
                            [sentiment_features[col] for col in SENTIMENT_FEATURE_COLUMNS],
                            dtype=np.float32,
                        ),
                    ]
                ).astype(np.float32)
                hybrid_embeddings.append(hybrid_vec)
        if use_row_cache:
            new_rows = len(df) - row_cache_hits
            logger.info(
                "Row embedding cache: {}/{} rows reused, {} newly encoded",
                row_cache_hits, len(df), new_rows,
            )
            if new_rows > 0:
                _save_row_cache(row_cache)

        df["news_emb"] = embeddings
        df["has_news"] = has_news_flags
        sentiment_frame = pd.DataFrame(sentiment_records, index=df.index)
        for col in SENTIMENT_TRACE_COLUMNS:
            df[col] = sentiment_frame[col].to_numpy(dtype=np.float32, copy=True)
        if self.sentiment_inferencer is not None and self.export_hybrid_embedding:
            df[NEWS_HYBRID_COLUMN] = hybrid_embeddings
        logger.info("News encoding complete")

        # --- Save to cache ---
        if use_cache:
            _EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "embeddings": np.stack(embeddings).astype(np.float32),
                "has_news": np.array(has_news_flags, dtype=bool),
            }

            for col in SENTIMENT_TRACE_COLUMNS:
                payload[col] = df[col].to_numpy(dtype=np.float32, copy=True)

            if self.sentiment_inferencer is not None and self.export_hybrid_embedding:
                hybrid_array = np.stack(hybrid_embeddings).astype(np.float32)
                df[NEWS_HYBRID_COLUMN] = hybrid_embeddings
                payload[NEWS_HYBRID_COLUMN] = hybrid_array

            np.savez_compressed(cache_path, **payload)
            logger.info("Embeddings cached → {} ({} rows)", cache_path.name, len(df))
            
        return df    

    def _score_titles(self, df: pd.DataFrame) -> tuple[list[list[float]], pd.DataFrame]:
        if self.sentiment_inferencer is None:
            return ([[] for _ in range(len(df))], pd.DataFrame())

        flat_titles: list[str] = []
        row_positions: list[int] = []
        row_symbols: list[str] = []
        row_times: list[Any] = []
        for pos in range(len(df)):
            titles = df.iloc[pos].get("news_titles", [])
            valid_titles = [str(title) for title in titles if str(title).strip()]
            symbol = str(df.iloc[pos].get("symbol", ""))
            time_value = df.iloc[pos].get("time", df.index[pos])
            for title in valid_titles:
                flat_titles.append(title)
                row_positions.append(pos)
                row_symbols.append(symbol)
                row_times.append(time_value)

        if not flat_titles:
            return ([[] for _ in range(len(df))], pd.DataFrame())

        scored = self.sentiment_inferencer.predict_titles(flat_titles, batch_size=self.sentiment_batch_size)
        scored.insert(0, "row_position", row_positions)
        scored.insert(1, "symbol", row_symbols)
        scored.insert(2, "time", row_times)

        scores_by_row: list[list[float]] = [[] for _ in range(len(df))]
        for row in scored.itertuples(index=False):
            scores_by_row[int(row.row_position)].append(float(row.sentiment_score))
        return scores_by_row, scored

    def _sentiment_signature(self) -> str:
        if self.sentiment_inferencer is None:
            return ""
        bundle = getattr(self.sentiment_inferencer, "bundle", None)
        handoff = getattr(bundle, "handoff", None)
        if not isinstance(handoff, dict):
            return type(self.sentiment_inferencer).__name__
        return "|".join(
            [
                str(handoff.get("checkpoint_path", "")),
                str(handoff.get("best_epoch", "")),
                str(handoff.get("best_selection_score", "")),
            ]
        )

    @staticmethod
    def _compute_cache_key(
        df: pd.DataFrame,
        text_col: str,
        model_name: str = "",
        *,
        sentiment_enabled: bool = False,
        sentiment_signature: str = "",
        weighted_pooling: bool = False,
    ) -> str:
        """Compute a deterministic cache key from the DataFrame contents."""
        h = hashlib.sha256()
        h.update(b"news_encoder_cache_v4")
        h.update(str(bool(weighted_pooling)).encode())
        h.update(str(len(df)).encode())
        if model_name:
            h.update(model_name.encode())
        h.update(str(bool(sentiment_enabled)).encode())
        if sentiment_signature:
            h.update(sentiment_signature.encode())
        # Include symbol info if available
        if "symbol" in df.columns:
            symbols = sorted(df["symbol"].unique())
            h.update(",".join(symbols).encode())
        # Include date range
        if "time" in df.columns:
            h.update(str(df["time"].min()).encode())
            h.update(str(df["time"].max()).encode())
        # Hash all article titles to detect content changes
        for pos in range(len(df)):
            texts = df.iloc[pos][text_col]
            if isinstance(texts, list):
                for t in texts:
                    h.update(str(t).encode())
            if sentiment_enabled and "news_titles" in df.columns:
                titles = df.iloc[pos]["news_titles"]
                if isinstance(titles, list):
                    for title in titles:
                        h.update(str(title).encode())
        return h.hexdigest()[:16]
