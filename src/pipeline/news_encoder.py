"""CMTF Data Pipeline — News Encoder module.

Encodes Vietnamese news text into dense 768-dim embeddings using a
PhoBERT-based sentence transformer.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

_NEWS_DIM = 768
_DEFAULT_MODEL_NAME = "dangvantuan/vietnamese-embedding"


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
    ) -> pd.DataFrame:
        """Encode news for every row in an aligned OHLCV+news DataFrame.

        Args:
            df_aligned: Output of
                :meth:`TemporalAligner.assign_news_to_bars`.
                Must contain ``text_col`` (list[str] per row) and
                ``news_missing_flag`` or ``has_news``.
            text_col: Column containing lists of news texts.

        Returns:
            DataFrame with added columns ``news_emb`` (np.ndarray 768-dim)
            and ``has_news`` (bool).
        """
        df = df_aligned.copy()

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
        return df
