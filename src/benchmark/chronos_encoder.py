"""Chronos market-only predictor: zero-shot and embedding extraction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from loguru import logger


class ChronosMarketPredictor:
    """Amazon Chronos on raw close-price series (no news).

    Modes:
        * **zero-shot** — no training; Chronos predicts next close,
          converted to log-return.
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
        seed: int = 42,
        horizon: int = 1,
        num_samples: int = 64,
        aggregation: str = "median",
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Predict H-step-ahead log-return using Chronos zero-shot on log-returns.

        Feeds log-returns (not absolute prices) to Chronos for better token
        diversity. The model forecasts future log-returns; cumulative sum gives
        the H-day return.

        Args:
            close_windows: (N, seq_len) raw close prices per sample.
            last_close: (N,) the last close price in each window (kept for API
                compatibility but unused in log-return mode).
            seed: RNG seed for reproducible Chronos sampling.
            horizon: Number of steps ahead to forecast (prediction_length).
            num_samples: Number of probabilistic samples drawn from Chronos.
            aggregation: Point estimate aggregation ("median" or "mean").
            return_diagnostics: If True, also return sample-dispersion metrics.

        Returns:
            (N,) predicted cumulative log-returns over *horizon* steps.
        """
        if aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be one of {'median', 'mean'}")

        all_preds: list[np.ndarray] = []
        all_std: list[np.ndarray] = []
        all_q10: list[np.ndarray] = []
        all_q90: list[np.ndarray] = []
        n = len(close_windows)

        # Convert absolute prices → log-returns for token diversity
        prices = np.clip(close_windows, 1e-12, None)
        log_returns_input = np.diff(np.log(prices), axis=1)  # (N, seq_len-1)
        log_returns_input = np.concatenate(
            [np.zeros((log_returns_input.shape[0], 1), dtype=log_returns_input.dtype),
             log_returns_input],
            axis=1,
        )  # (N, seq_len)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.tensor(
                log_returns_input[start:end], dtype=torch.float32
            )
            # Seed before each batch for deterministic sampling
            torch.manual_seed(seed + start)
            forecast = self.pipeline.predict(
                batch, prediction_length=horizon, num_samples=num_samples,
            )
            # forecast shape: (batch, num_samples, horizon)
            # Output is in log-return space; sum across horizon for cumulative return
            sampled_returns = forecast.sum(dim=-1).cpu().numpy()  # (batch, num_samples)

            if aggregation == "median":
                point_estimate = np.median(sampled_returns, axis=1)
            else:
                point_estimate = np.mean(sampled_returns, axis=1)

            all_preds.append(point_estimate)
            all_std.append(np.std(sampled_returns, axis=1))
            all_q10.append(np.quantile(sampled_returns, 0.10, axis=1))
            all_q90.append(np.quantile(sampled_returns, 0.90, axis=1))

        pred_returns = np.concatenate(all_preds)  # (N,)

        if not return_diagnostics:
            return pred_returns

        pred_std = np.concatenate(all_std)
        pred_q10 = np.concatenate(all_q10)
        pred_q90 = np.concatenate(all_q90)
        diagnostics = {
            "pred_std": pred_std,
            "pred_q10": pred_q10,
            "pred_q90": pred_q90,
            "pred_iqr80": pred_q90 - pred_q10,
        }
        logger.info(
            "Zero-shot uncertainty | mean std={:.6f} | mean iqr80={:.6f}",
            float(np.mean(pred_std)) if len(pred_std) else 0.0,
            float(np.mean(diagnostics["pred_iqr80"])) if len(pred_std) else 0.0,
        )
        return pred_returns, diagnostics

    # ------------------------------------------------------------------
    # Embeddings extraction
    # ------------------------------------------------------------------
    def get_embeddings(
        self,
        close_windows: np.ndarray,
        pooling: str = "mean",
        recency_bias: float = 2.0,
    ) -> np.ndarray:
        """Extract pooled Chronos encoder embeddings.

        Args:
            close_windows: (N, seq_len) raw close prices.
            pooling: Pooling strategy ("mean" or "recency_weighted").
            recency_bias: Last-token weight multiplier for recency pooling.

        Returns:
            (N, d_model) embeddings.
        """
        if pooling not in {"mean", "recency_weighted"}:
            raise ValueError("pooling must be one of {'mean', 'recency_weighted'}")
        if recency_bias < 1.0:
            raise ValueError("recency_bias must be >= 1.0")

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
                if pooling == "recency_weighted":
                    n_tokens = emb.shape[1]
                    weights = torch.linspace(
                        1.0,
                        recency_bias,
                        steps=n_tokens,
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    weights = weights / weights.sum()
                    pooled = (emb * weights[None, :, None]).sum(dim=1).cpu().numpy()
                else:
                    pooled = emb.mean(dim=1).cpu().numpy()
            all_embs.append(pooled)

        return np.concatenate(all_embs, axis=0)  # (N, d_model)

