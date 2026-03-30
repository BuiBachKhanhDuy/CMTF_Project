"""Chronos market-only predictor: zero-shot and linear-probe modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import Ridge


class ChronosMarketPredictor:
    """Amazon Chronos on raw close-price series (no news).

    Modes:
        * **zero-shot** — no training; Chronos predicts next close,
          converted to log-return.
        * **linear-probe** — Chronos encoder embeddings → Ridge
          regression → predicted return.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from chronos import ChronosPipeline

        logger.info("Loading Chronos model: {} …", model_name)
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.batch_size = batch_size
        self.device = device

        # Detect embedding dimension
        probe = torch.tensor([[1.0, 2.0, 3.0]])
        emb, _ = self.pipeline.embed(probe)
        self.d_model = emb.shape[-1]
        logger.info("Chronos d_model = {}", self.d_model)

    # ------------------------------------------------------------------
    # Zero-shot prediction
    # ------------------------------------------------------------------
    def zero_shot_predict(
        self,
        close_windows: np.ndarray,
        last_close: np.ndarray,
    ) -> np.ndarray:
        """Predict next-bar log-return using Chronos zero-shot.

        Args:
            close_windows: (N, seq_len) raw close prices per sample.
            last_close: (N,) the last close price in each window
                (used to convert predicted close → return).

        Returns:
            (N,) predicted log-returns.
        """
        all_preds: list[np.ndarray] = []
        n = len(close_windows)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.tensor(
                close_windows[start:end], dtype=torch.float32
            )
            forecast = self.pipeline.predict(
                batch, prediction_length=1, num_samples=20,
            )
            # forecast shape: (batch, num_samples, 1)
            median_close = forecast.median(dim=1).values.squeeze(-1).numpy()
            all_preds.append(median_close)

        pred_close = np.concatenate(all_preds)  # (N,)
        # Convert to log-return
        pred_returns = np.log(pred_close / last_close)
        return pred_returns

    # ------------------------------------------------------------------
    # Embeddings extraction
    # ------------------------------------------------------------------
    def get_embeddings(
        self, close_windows: np.ndarray
    ) -> np.ndarray:
        """Extract mean-pooled Chronos encoder embeddings.

        Args:
            close_windows: (N, seq_len) raw close prices.

        Returns:
            (N, d_model) embeddings.
        """
        all_embs: list[np.ndarray] = []
        n = len(close_windows)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.tensor(
                close_windows[start:end], dtype=torch.float32
            )
            with torch.no_grad():
                emb, _ = self.pipeline.embed(batch)
                # emb: (batch, num_tokens, d_model)
                pooled = emb.mean(dim=1).cpu().numpy()
            all_embs.append(pooled)

        return np.concatenate(all_embs, axis=0)  # (N, d_model)

    # ------------------------------------------------------------------
    # Linear-probe (embed → Ridge → return)
    # ------------------------------------------------------------------
    def linear_probe_predict(
        self,
        close_train: np.ndarray,
        y_train: np.ndarray,
        close_val: np.ndarray,
        y_val: np.ndarray,
        close_test: np.ndarray,
    ) -> np.ndarray:
        """Chronos embeddings + Ridge regression.

        Args:
            close_train: (N_train, seq_len) raw close windows.
            y_train: (N_train,) target returns.
            close_val: (N_val, seq_len) raw close windows.
            y_val: (N_val,) target returns (for alpha selection).
            close_test: (N_test, seq_len) raw close windows.

        Returns:
            (N_test,) predicted returns.
        """
        logger.info("Computing Chronos embeddings for linear probe …")
        emb_train = self.get_embeddings(close_train)
        emb_val = self.get_embeddings(close_val)
        emb_test = self.get_embeddings(close_test)

        # Select best Ridge alpha on validation set
        best_alpha, best_score = 1.0, float("inf")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha)
            model.fit(emb_train, y_train)
            val_pred = model.predict(emb_val)
            score = float(np.mean((val_pred - y_val) ** 2))
            if score < best_score:
                best_score = score
                best_alpha = alpha

        logger.info("Ridge best alpha = {} (val MSE = {:.6f})", best_alpha, best_score)

        # Refit on train + val
        emb_trainval = np.concatenate([emb_train, emb_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        model = Ridge(alpha=best_alpha)
        model.fit(emb_trainval, y_trainval)

        return model.predict(emb_test)
