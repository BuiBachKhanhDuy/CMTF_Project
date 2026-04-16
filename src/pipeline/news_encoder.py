"""CMTF Data Pipeline — News Encoder module.

Encodes Vietnamese news text into dense 768-dim embeddings using a
PhoBERT-based sentence transformer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

_NEWS_DIM = 768
_DEFAULT_MODEL_NAME = "dangvantuan/vietnamese-embedding"
_EMBEDDINGS_CACHE_DIR = Path("./cache/embeddings")


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
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

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
    ) -> dict[str, Any]:
        """Encode a list of news texts for one time window.

        Args:
            texts: List of article texts. May be empty.
            null_mask: If ``True``, force a zero-vector output
                (used for explicit ``[NO_NEWS]`` injection).

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
        # Mean-pool across articles in the window
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
        cache_key = self._compute_cache_key(df, text_col, self.model_name)
        cache_path = _EMBEDDINGS_CACHE_DIR / f"{cache_key}.npz"

        if use_cache and cache_path.exists():
            try:
                cached = np.load(cache_path)
                emb_array = cached["embeddings"]       # (N, 768)
                has_news_array = cached["has_news"]     # (N,)
                if len(emb_array) == len(df):
                    df["news_emb"] = list(emb_array)
                    df["has_news"] = has_news_array.astype(bool)
                    logger.info(
                        "Embeddings loaded from cache: {} ({} rows)",
                        cache_path.name, len(df),
                    )
                    return df
                else:
                    logger.warning(
                        "Embedding cache row count mismatch ({} vs {}) — re-encoding",
                        len(emb_array), len(df),
                    )
            except Exception:
                logger.warning("Corrupt embedding cache {} — re-encoding", cache_path.name)

        embeddings: list[np.ndarray] = []
        has_news_flags: list[bool] = []

        logger.info("Encoding news for {} rows …", len(df))
        for pos in tqdm(range(len(df)), desc="Encoding news"):
            texts = df.iloc[pos][text_col]
            is_null = df.iloc[pos]["news_missing_flag"] if "news_missing_flag" in df.columns else (not texts)

            result = self.encode_window(
                texts=texts if isinstance(texts, list) else [],
                null_mask=bool(is_null),
            )
            embeddings.append(result["embedding"])
            has_news_flags.append(result["has_news"])

        df["news_emb"] = embeddings
        df["has_news"] = has_news_flags
        logger.info("News encoding complete")

        # --- Save to cache ---
        if use_cache:
            _EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                embeddings=np.stack(embeddings),
                has_news=np.array(has_news_flags),
            )
            logger.info("Embeddings cached → {} ({} rows)", cache_path.name, len(df))

        return df

    @staticmethod
    def _compute_cache_key(df: pd.DataFrame, text_col: str, model_name: str = "") -> str:
        """Compute a deterministic cache key from the DataFrame contents."""
        h = hashlib.sha256()
        h.update(str(len(df)).encode())
        if model_name:
            h.update(model_name.encode())
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
        return h.hexdigest()[:16]
