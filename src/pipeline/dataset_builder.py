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
        news_representation: str = "text",
        allow_missing_target: bool = False,
    ) -> None:
        self.sequence_len = int(sequence_len)
        self.horizon = int(horizon)
        self.target_horizon_days = int(target_horizon_days)
        self.news_representation = str(news_representation)
        self.target_col = f"fwd_ret_{self.target_horizon_days}d"

        df = df_featured.copy()

        # --------------------------------------------------
        # Ensure 'time' column
        # --------------------------------------------------
        if "time" not in df.columns:
            if df.index.name == "time":
                df = df.reset_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "time"})
            else:
                raise ValueError("df_featured must contain a 'time' column or DatetimeIndex")

        # Drop stray cols
        df = df.drop(columns=[c for c in ("index", "level_0", "Unnamed: 0") if c in df.columns])

        if "symbol" not in df.columns:
            raise ValueError("Missing 'symbol'")
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        if "has_news" not in df.columns:
            raise ValueError("Missing 'has_news' column")

        # Training requires a finite target. Live inference retains recent rows
        # whose forward returns are not yet observable.
        if not allow_missing_target:
            valid_mask = np.isfinite(df[self.target_col])
            df = df.loc[valid_mask].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("All rows removed after filtering invalid targets")

        self.df = df

        # --------------------------------------------------
        # Market features
        # --------------------------------------------------
        fwd_cols = [c for c in df.columns if re.match(r"^fwd_ret_\d+d$", str(c))]

        self.market_cols = [
            c for c in df.columns
            if c not in self._EXCLUDE_COLS
            and c not in fwd_cols
            and c != "time"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not self.market_cols:
            raise ValueError("No valid market feature columns")

        self._market = df[self.market_cols].astype(np.float32).to_numpy(copy=True)

        # --------------------------------------------------
        # News embeddings
        # --------------------------------------------------
        if news_representation == "text":
            news_column = "news_emb"
        elif news_representation == "hybrid":
            news_column = NEWS_HYBRID_COLUMN
        else:
            raise ValueError("news_representation must be 'text' or 'hybrid'")

        if news_column not in df.columns:
            raise ValueError(f"Missing news column: {news_column}")

        news_list = df[news_column].tolist()

        validated = []
        expected_dim = None

        for i, emb in enumerate(news_list):
            arr = np.asarray(emb, dtype=np.float32)

            if arr.ndim != 1:
                raise ValueError(f"Row {i}: expected 1D embedding, got {arr.shape}")

            if expected_dim is None:
                expected_dim = arr.shape[0]
            elif arr.shape[0] != expected_dim:
                raise ValueError("Inconsistent embedding dimension")

            validated.append(arr)

        self._news = np.stack(validated).astype(np.float32)

        if self._news.ndim != 2:
            raise ValueError("News embedding must be 2D")

        self.news_dim = self._news.shape[1]

        # Optional dimension checks
        if news_representation == "text" and self.news_dim != 768:
            raise ValueError(f"Expected 768, got {self.news_dim}")

        if news_representation == "hybrid" and self.news_dim != 773:
            raise ValueError(f"Expected 773, got {self.news_dim}")

        # --------------------------------------------------
        # Mask & target (AFTER filtering!)
        # --------------------------------------------------
        self._mask = (~df["has_news"].astype(bool)).to_numpy(copy=True)
        self._target = df[self.target_col].astype(np.float32).to_numpy(copy=True)

        # --------------------------------------------------
        # Sequence bounds
        # --------------------------------------------------
        self._valid_start = self.sequence_len - 1
        self._valid_end = len(df)

        if self._valid_end <= self._valid_start:
            raise ValueError(
                f"Not enough data: len={len(df)} < seq_len={self.sequence_len}"
            )

        logger.info(
            "Dataset ready | n_samples={} | seq_len={} | features={} | news_dim={}",
            len(self),
            self.sequence_len,
            len(self.market_cols),
            self.news_dim,
        )

    # --------------------------------------------------
    # Length
    # --------------------------------------------------
    def __len__(self) -> int:
        return self._valid_end - self._valid_start

    # --------------------------------------------------
    # Get item
    # --------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        actual_idx = self._valid_start + idx

        start = actual_idx - self.sequence_len + 1
        end = actual_idx + 1

        market = torch.from_numpy(self._market[start:end])
        news = torch.from_numpy(self._news[start:end])
        mask = torch.from_numpy(self._mask[start:end])

        target = torch.tensor(self._target[actual_idx], dtype=torch.float32)

        return {
            "market": market,        # [seq_len, features]
            "news": news,            # [seq_len, news_dim]
            "mask": mask,            # [seq_len]
            "target": target,        # scalar
        }
