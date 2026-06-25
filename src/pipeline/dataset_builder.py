"""CMTF Data Pipeline — Dataset Builder module.

Builds a PyTorch Dataset that yields aligned market + news tensors
for the Cross-Modal Temporal Fusion model.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

from .news_encoder import NEWS_HYBRID_COLUMN, SENTIMENT_TRACE_COLUMNS


class CMTFDataset(Dataset):
    """PyTorch Dataset for the Cross-Modal Temporal Fusion model.

    Each sample is a dict with:
        - ``'market'``:  ``Tensor[seq_len, n_market_features]``  (float32)
        - ``'news'``:    ``Tensor[seq_len, news_dim]``           (float32)
        - ``'mask'``:    ``Tensor[seq_len]``                     (bool, True = no news)
        - ``'target'``:  ``Tensor[horizon]``                     (float32)

    Notes:
        - We preserve the ``time`` column in ``self.df`` so benchmark code can
          later reconstruct chronological splits and align raw OHLCV.
        - Market features are true market-only features: no sentiment trace
          columns, no news-derived columns, no forward-return target columns.
    """

    _EXCLUDE_COLS = {
        "news_emb",
        NEWS_HYBRID_COLUMN,
        "has_news",
        "news_count",
        "news_titles",
        "news_content",
        "news_missing_flag",
        "symbol",
        "index",
        "level_0",
        "Unnamed: 0",
        *set(SENTIMENT_TRACE_COLUMNS),
    }

    def __init__(
        self,
        df_featured: pd.DataFrame,
        sequence_len: int = 30,
        horizon: int = 1,
        target_horizon_days: int = 1,
        news_representation: str = "text",  # "text" or "hybrid"
    ) -> None:
        self.sequence_len = int(sequence_len)
        self.horizon = int(horizon)
        self.target_horizon_days = int(target_horizon_days)
        self.news_representation = str(news_representation)
        self.target_col = f"fwd_ret_{self.target_horizon_days}d"

        df = df_featured.copy()

        # Preserve time as a column.
        # If it exists only in the index, restore it.
        if "time" not in df.columns:
            if df.index.name == "time":
                df = df.reset_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "time"})
            else:
                # Keep existing row order, but make the schema failure explicit
                raise ValueError("df_featured must contain a 'time' column or DatetimeIndex")

        # Clean stray index-like columns except preserved "time"
        stray_cols = [c for c in ("index", "level_0", "Unnamed: 0") if c in df.columns]
        if stray_cols:
            df = df.drop(columns=stray_cols)

        if "symbol" not in df.columns:
            raise ValueError("df_featured must contain a 'symbol' column")
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        if "has_news" not in df.columns:
            raise ValueError("Missing required column: has_news")

        # Keep canonical order and clean index only after preserving time
        self.df = df.reset_index(drop=True).copy()

        fwd_cols = [c for c in self.df.columns if re.match(r"^fwd_ret_\d+d$", str(c))]

        # Pure market-only features
        self.market_cols = [
            c
            for c in self.df.columns
            if c not in self._EXCLUDE_COLS
            and c not in fwd_cols
            and c != "time"
            and pd.api.types.is_numeric_dtype(self.df[c])
        ]

        if not self.market_cols:
            raise ValueError("No valid market feature columns found after exclusion rules")

        logger.info(
            "CMTFDataset | {} pure market features | seq_len={} | horizon={} | target={}",
            len(self.market_cols),
            self.sequence_len,
            self.horizon,
            self.target_col,
        )

        self._market = self.df[self.market_cols].astype(np.float32).to_numpy(copy=True)

        if news_representation == "text":
            news_column = "news_emb"
        elif news_representation == "hybrid":
            news_column = NEWS_HYBRID_COLUMN
        else:
            raise ValueError(
                f"Unsupported news_representation={news_representation!r}. "
                "Expected 'text' or 'hybrid'."
            )

        if news_column not in self.df.columns:
            raise ValueError(
                f"Missing required news embedding column: {news_column}. "
                f"Available columns: {list(self.df.columns)}"
            )

        news_emb_list = self.df[news_column].tolist()

        validated_news: list[np.ndarray] = []
        expected_dim: int | None = None

        for i, emb in enumerate(news_emb_list):
            arr = np.asarray(emb, dtype=np.float32)

            if arr.ndim != 1:
                raise ValueError(
                    f"Row {i} in column {news_column} must be a 1D embedding vector, got shape {arr.shape}"
                )

            if expected_dim is None:
                expected_dim = int(arr.shape[0])
            elif int(arr.shape[0]) != expected_dim:
                raise ValueError(
                    f"Inconsistent news embedding dim in column {news_column}: "
                    f"row 0 has dim {expected_dim}, row {i} has dim {arr.shape[0]}"
                )

            validated_news.append(arr)

        self._news = np.stack(validated_news).astype(np.float32)

        if self._news.ndim != 2:
            raise ValueError(f"Expected stacked news embeddings to be 2D, got {self._news.shape}")

        self.news_dim = int(self._news.shape[-1])
        if self.news_representation == "text" and self.news_dim != 768:
            raise ValueError(
                f"Expected text news embeddings to have dim 768, got {self.news_dim}"
            )

        if self.news_representation == "hybrid" and self.news_dim != 773:
            raise ValueError(
                f"Expected hybrid news embeddings to have dim 773, got {self.news_dim}"
            )
        logger.info(
            "CMTFDataset | news_representation={} | news_dim={}",
            self.news_representation,
            self.news_dim,
        )

        self._mask = (~self.df["has_news"].astype(bool)).to_numpy(copy=True)
        self._target = self.df[self.target_col].astype(np.float32).to_numpy(copy=True)

        self._valid_start = self.sequence_len - 1
        self._valid_end = len(self.df)

    def __len__(self) -> int:
        length = self._valid_end - self._valid_start
        return max(length, 0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        actual_idx = self._valid_start + idx
        start = actual_idx - self.sequence_len + 1
        end = actual_idx + 1

        market = torch.from_numpy(self._market[start:end].copy())
        news = torch.from_numpy(self._news[start:end].copy())
        mask = torch.from_numpy(self._mask[start:end].copy())

        target = torch.tensor(self._target[actual_idx], dtype=torch.float32)

        return {
            "market": market,
            "news": news,
            "mask": mask,
            "target": target,
        }