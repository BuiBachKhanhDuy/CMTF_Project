"""CMTF Data Pipeline — Dataset Builder module.

Builds a PyTorch Dataset that yields aligned market + news tensors
for the Cross-Modal Temporal Fusion model.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from loguru import logger


class CMTFDataset(Dataset):
    """PyTorch Dataset for the Cross-Modal Temporal Fusion model.

    Each sample is a dict with:
        - ``'market'``:  ``Tensor[seq_len, n_market_features]``  (float32)
        - ``'news'``:    ``Tensor[seq_len, 768]``                (float32)
        - ``'mask'``:    ``Tensor[seq_len]``                     (bool, True = no news)
        - ``'target'``:  ``Tensor[horizon]``                     (float32)

    Args:
        df_featured: DataFrame containing OHLCV features, technical
            indicators, ``news_emb`` (np.ndarray 768-dim per row),
            ``has_news`` (bool), and forward-return target columns.
        sequence_len: Number of look-back bars per sample.
        horizon: Forward prediction horizon (default 1).
        target_horizon_days: Which forward-return label to learn,
            e.g. 1 -> ``fwd_ret_1d``, 5 -> ``fwd_ret_5d``.
    """

    # Columns that are NOT market input features
    _EXCLUDE_COLS = {
        "news_emb",
        "has_news",
        "news_count",
        "news_titles",
        "news_content",
        "news_missing_flag",
        "symbol",
    }

    def __init__(
        self,
        df_featured: pd.DataFrame,
        sequence_len: int = 30,
        horizon: int = 1,
        target_horizon_days: int = 1,
    ) -> None:
        self.sequence_len = sequence_len
        self.horizon = horizon
        self.target_horizon_days = int(target_horizon_days)
        self.target_col = f"fwd_ret_{self.target_horizon_days}d"
        self.df = df_featured.reset_index(drop=False)

        if self.target_col not in self.df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")

        fwd_cols = [c for c in self.df.columns if re.match(r"^fwd_ret_\d+d$", str(c))]

        # Identify column groups
        self.market_cols = [
            c
            for c in self.df.columns
            if c not in self._EXCLUDE_COLS
            and c not in fwd_cols
            and c != "time"
            and self.df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]
        logger.info(
            "CMTFDataset | {} market features | seq_len={} | horizon={} | target={}",
            len(self.market_cols),
            sequence_len,
            horizon,
            self.target_col,
        )

        # Pre-compute numpy arrays for speed
        self._market = self.df[self.market_cols].values.astype(np.float32)

        # News embeddings → (N, 768) array
        news_emb_list = self.df["news_emb"].tolist()
        self._news = np.stack(news_emb_list).astype(np.float32)  # (N, 768)

        # Mask: True where there is NO news (inverted has_news)
        self._mask = (~self.df["has_news"].astype(bool)).values  # True = no news

        # Target
        self._target = self.df[self.target_col].values.astype(np.float32)

        # Valid indices (need seq_len history and horizon forward target)
        self._valid_start = self.sequence_len - 1
        # Last valid index: need target at i + horizon - 1, so i + horizon - 1 < len
        self._valid_end = len(self.df) - self.horizon

    def __len__(self) -> int:
        length = self._valid_end - self._valid_start
        return max(length, 0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        actual_idx = self._valid_start + idx
        start = actual_idx - self.sequence_len + 1
        end = actual_idx + 1  # exclusive

        market = torch.from_numpy(self._market[start:end].copy())        # (seq_len, F)
        news = torch.from_numpy(self._news[start:end].copy())            # (seq_len, 768)
        mask = torch.from_numpy(self._mask[start:end].copy())            # (seq_len,)

        # Target: forward returns for `horizon` steps starting at actual_idx
        target_vals = self._target[actual_idx : actual_idx + self.horizon]
        target = torch.from_numpy(target_vals.copy())                     # (horizon,)

        return {
            "market": market,
            "news": news,
            "mask": mask,
            "target": target,
        }

    # ------------------------------------------------------------------
    # Walk-forward splits
    # ------------------------------------------------------------------
    def create_splits(
        self,
        train_end: str,
        val_end: str,
    ) -> tuple[Subset, Subset, Subset]:
        """Create train / val / test subsets by date (walk-forward).

        Args:
            train_end: Last date (inclusive) for training data.
            val_end: Last date (inclusive) for validation data.

        Returns:
            ``(train_subset, val_subset, test_subset)`` — each a
            :class:`torch.utils.data.Subset`.
        """
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)

        # The dataset index maps to actual_idx = _valid_start + dataset_idx
        # We need to check the *last* timestamp in each sequence window
        times = pd.to_datetime(self.df["time"])

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for dataset_idx in range(len(self)):
            actual_idx = self._valid_start + dataset_idx
            bar_time = times.iloc[actual_idx]

            if bar_time <= train_end_ts:
                train_indices.append(dataset_idx)
            elif bar_time <= val_end_ts:
                val_indices.append(dataset_idx)
            else:
                test_indices.append(dataset_idx)

        logger.info(
            "Splits | train={} | val={} | test={}",
            len(train_indices),
            len(val_indices),
            len(test_indices),
        )

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )
