"""CMTF Data Pipeline — Feature Engineering module.

Computes technical indicators and normalises market features for the
Cross-Modal Temporal Fusion model.
"""

from __future__ import annotations

import pickle
import importlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ta = None
for _module_name in (("pandas" + "_ta"), "pandas_ta_classic"):
    try:
        ta = importlib.import_module(_module_name)
        break
    except ImportError:
        continue
if ta is None:  # pragma: no cover - dependency error surfaced at runtime
    raise ImportError("Neither pandas_ta nor pandas_ta_classic is installed.")

ARTIFACTS_DIR = Path("./artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """Computes technical indicators and normalises features.

    Attributes:
        artifacts_dir: Directory for persisting fitted scalers.
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------
    def compute_technical(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Append standard technical indicator columns to an OHLCV DataFrame.

        Columns added:
            ``rsi_14``, ``macd``, ``macd_signal``, ``macd_hist``,
            ``bb_upper``, ``bb_mid``, ``bb_lower``, ``atr_14``,
            ``vol_ratio``, ``log_ret``, ``fwd_ret_1d``.

        Args:
            df_ohlcv: DataFrame with columns ``[open, high, low, close, volume]``
                indexed by datetime.

        Returns:
            Copy of the input DataFrame with indicator columns appended.
            ``fwd_ret_1d`` is the **prediction target** — never include it as
            an input feature.
        """
        df = df_ohlcv.copy()

        # RSI
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 1]
            df["macd_hist"] = macd.iloc[:, 2]

        # Bollinger Bands (20, 2)
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df["bb_lower"] = bbands.iloc[:, 0]
            df["bb_mid"] = bbands.iloc[:, 1]
            df["bb_upper"] = bbands.iloc[:, 2]

        # ATR
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Volume ratio
        vol_ma = df["volume"].rolling(window=20, min_periods=1).mean()
        df["vol_ratio"] = df["volume"] / vol_ma

        # Log return
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        # Forward return label (TARGET — NEVER use as input)
        df["fwd_ret_1d"] = np.log(df["close"].shift(-1) / df["close"])

        logger.info("Technical indicators computed | {} rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        method: Literal["zscore", "minmax"] = "zscore",
        split_date: Optional[str] = None,
        symbol: str = "default",
    ) -> pd.DataFrame:
        """Normalise selected feature columns.

        The scaler is **fit only on training data** (rows where
        ``index < split_date``) and then applied to the full DataFrame.

        Args:
            df: Input DataFrame indexed by datetime.
            feature_cols: Column names to normalise.
            method: ``'zscore'`` (StandardScaler) or ``'minmax'``.
            split_date: End of the training period (exclusive).
                If ``None``, the scaler is fit on the entire DataFrame
                (**only safe for debugging**).
            symbol: Used for naming the persisted scaler file.

        Returns:
            DataFrame with normalised feature columns.
        """
        df = df.copy()

        scaler = StandardScaler() if method == "zscore" else MinMaxScaler()

        if split_date is not None:
            train_mask = df.index < pd.Timestamp(split_date)
            train_data = df.loc[train_mask, feature_cols]
        else:
            logger.warning("No split_date provided — fitting scaler on full data (debug mode)")
            train_data = df[feature_cols]

        scaler.fit(train_data.values)
        df[feature_cols] = scaler.transform(df[feature_cols].values)

        # Persist scaler
        scaler_path = self.artifacts_dir / f"scaler_{symbol}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler saved → {}", scaler_path)

        return df
