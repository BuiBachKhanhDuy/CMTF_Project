"""Chronos market-only predictor: zero-shot and embedding extraction."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark.baseline_models import (
    BaseTorchMarketPredictor,
    BaseTorchHybridPredictor,
)


# =====================================================================
# Frozen-projection pooled-embedding memo cache
# =====================================================================
#
# Only valid when the ENTIRE market->pooled path is frozen (frozen T5 encoder
# AND frozen input_projection). Under that regime the pooled embedding of a
# window is a fixed function of (input_projection weights, window content), so
# it can be memoized and reused across:
#   * epochs within a single fit (removes the ``epochs`` multiplier),
#   * the 5 deepcopied OOF fold models (removes the ``folds`` multiplier),
#   * the main fit and predict passes.
#
# The cache key mixes a per-instance projection signature (hash of the frozen
# input_projection weights) with a per-row content hash, so distinct adapters
# (different projection inits) never collide, while deepcopies of the same
# adapter (identical weights) share entries. Bounded FIFO/LRU to cap memory.
_FROZEN_POOLED_CACHE: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
_FROZEN_POOLED_CACHE_MAXSIZE = 20000


def clear_frozen_pooled_cache() -> None:
    """Clear the module-level frozen-projection pooled-embedding cache."""
    _FROZEN_POOLED_CACHE.clear()


def _store_frozen_pooled(key: bytes, value: torch.Tensor) -> None:
    cache = _FROZEN_POOLED_CACHE
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _FROZEN_POOLED_CACHE_MAXSIZE:
        cache.popitem(last=False)


class ChronosMarketPredictor:
    """Amazon Chronos on raw close-price series (no news).

    Modes:
        * **zero-shot** — no training; Chronos predicts next close,
          converted to log-return.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 8,
        model_name: str = "amazon/chronos-t5-small",
    ):
        from chronos import ChronosPipeline

        self.device = device
        self.batch_size = batch_size
        logger.info("Loading Chronos model: {} …", model_name)
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.d_model = self.pipeline.model.model.encoder.config.d_model

    def zero_shot_predict(
            self,
            close_windows: np.ndarray,
            last_close: np.ndarray,
            seed: int = 42,
            horizon: int = 1,
            num_samples: int = 5,
            aggregation: str = "median",
            return_diagnostics: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
            """Predict H-step-ahead log-return using Chronos zero-shot on log-returns."""
            if aggregation not in {"median", "mean"}:
                raise ValueError("aggregation must be one of {'median', 'mean'}")

            all_preds: list[np.ndarray] = []
            all_std: list[np.ndarray] = []
            all_q10: list[np.ndarray] = []
            all_q90: list[np.ndarray] = []
            n = len(close_windows)

            if n == 0:
                out = np.empty((0,), dtype=np.float32)
                if not return_diagnostics:
                    return out
                return out, {
                    "pred_std": out.copy(),
                    "pred_q10": out.copy(),
                    "pred_q90": out.copy(),
                    "pred_iqr80": out.copy(),
                }

            prices = np.clip(np.asarray(close_windows, dtype=np.float32), 1e-12, None)
            log_returns_input = np.diff(np.log(prices), axis=1)
            log_returns_input = np.concatenate(
                [
                    np.zeros((log_returns_input.shape[0], 1), dtype=log_returns_input.dtype),
                    log_returns_input,
                ],
                axis=1,
            )

            logger.info(
                "Chronos zero-shot start | N={} | seq_len={} | horizon={} | batch_size={} | num_samples={}",
                n,
                log_returns_input.shape[1],
                horizon,
                self.batch_size,
                num_samples,
            )

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)

                logger.info(
                    "Chronos zero-shot batch {}/{} | rows {}:{}",
                    (start // self.batch_size) + 1,
                    (n + self.batch_size - 1) // self.batch_size,
                    start,
                    end,
                )

                # Chronos 1.x tokenizes contexts on CPU: its tokenizer owns
                # ``boundaries``/``centers`` as ordinary CPU tensors (not
                # model buffers).  ``ChronosPipeline.predict`` subsequently
                # moves token ids and masks to ``self.model.device`` before
                # running the model.  Moving this context batch to CUDA here
                # therefore makes ``torch.bucketize`` mix CUDA inputs with CPU
                # boundaries and fail.  Keep pipeline inputs on CPU regardless
                # of where the Chronos model itself is loaded.
                batch = torch.tensor(
                    log_returns_input[start:end],
                    dtype=torch.float32,
                    device="cpu",
                )

                torch.manual_seed(seed + start)

                with torch.no_grad():
                    forecast = self.pipeline.predict(
                        batch,
                        prediction_length=horizon,
                        num_samples=num_samples,
                    )

                # Expected shape: (batch, num_samples, horizon)
                if not isinstance(forecast, torch.Tensor):
                    forecast = torch.as_tensor(forecast)

                sampled_returns = forecast.sum(dim=-1).cpu().numpy()

                if aggregation == "median":
                    point_estimate = np.median(sampled_returns, axis=1)
                else:
                    point_estimate = np.mean(sampled_returns, axis=1)

                all_preds.append(point_estimate.astype(np.float32))
                all_std.append(np.std(sampled_returns, axis=1).astype(np.float32))
                all_q10.append(np.quantile(sampled_returns, 0.10, axis=1).astype(np.float32))
                all_q90.append(np.quantile(sampled_returns, 0.90, axis=1).astype(np.float32))

            pred_returns = np.concatenate(all_preds, axis=0).astype(np.float32)

            if not return_diagnostics:
                return pred_returns

            pred_std = np.concatenate(all_std, axis=0).astype(np.float32)
            pred_q10 = np.concatenate(all_q10, axis=0).astype(np.float32)
            pred_q90 = np.concatenate(all_q90, axis=0).astype(np.float32)

            diagnostics = {
                "pred_std": pred_std,
                "pred_q10": pred_q10,
                "pred_q90": pred_q90,
                "pred_iqr80": (pred_q90 - pred_q10).astype(np.float32),
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


class ChronosAdapter(BaseTorchMarketPredictor):
    """
    Apple-to-Apple Chronos Feature Extractor.
    Takes multivariate market_windows, maps them to T5 embedding space,
    passes them through the frozen T5 encoder via inputs_embeds,
    and applies a regression head.
    """
    def __init__(
        self,
        input_dim: int,
        model_name: str = "amazon/chronos-t5-small",
        dropout: float = 0.3,
        device: str = "cpu",
        pipeline=None,
        target_scale: float = 1.0,
        freeze_input_projection: bool = True,
        cache_frozen_embeddings: bool = True,
    ):
        super().__init__(target_scale=target_scale, device=device)

        if pipeline is not None:
            logger.info("ChronosAdapter: reusing shared ChronosPipeline")
            self.pipeline = pipeline
        else:
            from chronos import ChronosPipeline
            logger.info("Loading Chronos model for feature extraction: {} …", model_name)
            self.pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32,
            )

        self.encoder = self.pipeline.model.model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.chronos_d_model = self.encoder.config.d_model
        self.d_model = self.chronos_d_model
        self.hidden_dim = self.chronos_d_model      # backward compatibility
        self.seq_output_dim = self.chronos_d_model  # explicit sequence width

        self.input_projection = nn.Linear(input_dim, self.chronos_d_model)
        self.dropout = nn.Dropout(dropout)

        # Freezing the projection permits pooled-embedding memoization. It is an
        # opt-in modeling choice because it removes projection training.
        self.freeze_input_projection = bool(freeze_input_projection)
        if self.freeze_input_projection:
            for param in self.input_projection.parameters():
                param.requires_grad = False
            logger.info(
                "ChronosAdapter: input_projection FROZEN (linear-probe mode); "
                "pooled-embedding memoization {}",
                "ENABLED" if cache_frozen_embeddings else "disabled",
            )
        # Memoization only makes sense (and is only numerically valid) when the
        # full market->pooled path is frozen.
        self._cache_frozen_embeddings = (
            bool(cache_frozen_embeddings) and self.freeze_input_projection
        )
        self._proj_sig: bytes | None = None

        self.regressor = nn.Sequential(
            nn.Linear(self.chronos_d_model, self.chronos_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.chronos_d_model // 2, 1)
        )
        self.to(self.device)

    def _projection_signature(self) -> bytes:
        """Stable hash of the (frozen) input_projection weights.

        Used as a cache-key prefix so pooled embeddings from adapters with
        different projection weights never collide, while deepcopies of the
        same (frozen) adapter share cache entries.
        """
        if self._proj_sig is None:
            hasher = hashlib.blake2b(digest_size=16)
            for param in self.input_projection.parameters():
                arr = param.detach().to("cpu", torch.float32).contiguous().numpy()
                hasher.update(arr.tobytes())
            self._proj_sig = hasher.digest()
        return self._proj_sig

    def _cached_pooled_embeddings(self, market_tensors: torch.Tensor) -> torch.Tensor:
        """Frozen-path pooled embeddings with per-row memoization.

        Numerically identical to the uncached frozen path (mean-pool of the
        frozen T5 last hidden state); it just avoids recomputing T5 for windows
        already seen this run.
        """
        device = market_tensors.device
        arr = market_tensors.detach().to("cpu", torch.float32).contiguous().numpy()
        n = arr.shape[0]
        sig = self._projection_signature()

        keys: list[bytes] = []
        for i in range(n):
            hasher = hashlib.blake2b(digest_size=16)
            hasher.update(sig)
            hasher.update(arr[i].tobytes())
            keys.append(hasher.digest())

        # Collect the FIRST occurrence of each unique key that is not already
        # cached. Deduping within the batch (not just against the cache) means
        # repeated windows in one batch cost a single heavy T5 pass.
        unique_missing: "OrderedDict[bytes, int]" = OrderedDict()
        for i, key in enumerate(keys):
            if key not in _FROZEN_POOLED_CACHE and key not in unique_missing:
                unique_missing[key] = i

        if unique_missing:
            miss_rows = list(unique_missing.values())
            miss_batch = market_tensors.index_select(
                0, torch.as_tensor(miss_rows, device=device)
            )
            with torch.no_grad():
                embeds = self.input_projection(miss_batch)
                outputs = self.encoder(inputs_embeds=embeds)
                pooled = outputs.last_hidden_state.mean(dim=1).detach().to("cpu")
            for j, key in enumerate(unique_missing):
                _store_frozen_pooled(key, pooled[j].clone())

        stacked = torch.stack([_FROZEN_POOLED_CACHE[key] for key in keys], dim=0)
        return stacked.to(device)

    def encode_sequence_torch(self, market_tensors: torch.Tensor) -> torch.Tensor:
        """Return (batch, seq_len, d_model)"""
        # market_tensors: (batch, seq_len, input_dim)
        embeds = self.input_projection(market_tensors)
        
        with torch.no_grad():
            outputs = self.encoder(inputs_embeds=embeds)
            
        return outputs.last_hidden_state

    def encode_pooled_torch(self, market_tensors: torch.Tensor) -> torch.Tensor:
        """Return (batch, d_model)"""
        if self._cache_frozen_embeddings:
            return self._cached_pooled_embeddings(market_tensors)
        seq_embs = self.encode_sequence_torch(market_tensors)
        # Average pooling
        return seq_embs.mean(dim=1)
        
    def _encode_market_tensors(self, market_tensors: torch.Tensor) -> torch.Tensor:
        return self.encode_pooled_torch(market_tensors)

    def forward(self, market_tensors: torch.Tensor) -> torch.Tensor:
        embs = self._encode_market_tensors(market_tensors)
        embs = self.dropout(embs)
        return self.regressor(embs).squeeze(-1)


class ChronosHybridAdapter(BaseTorchHybridPredictor):
    """
    Apple-to-Apple Chronos Hybrid Extractor.
    Extracts market embeddings using the frozen T5 encoder,
    concatenates with tabular features, and applies a regression head.
    """

    def __init__(
        self,
        input_dim: int,
        tabular_dim: int,
        model_name: str = "amazon/chronos-t5-small",
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        from chronos import ChronosPipeline

        logger.info("Loading Chronos model for hybrid extraction: {} …", model_name)
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.encoder = self.pipeline.model.model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.chronos_d_model = self.encoder.config.d_model

        # Map multivariate input to chronos embed dim
        self.input_projection = nn.Linear(input_dim, self.chronos_d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Regression head for concatenated features
        combined_dim = self.chronos_d_model + tabular_dim
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, 1)
        )
        self.to(self.device)
        
    def encode_sequence_torch(self, market_tensors: torch.Tensor) -> torch.Tensor:
        embeds = self.input_projection(market_tensors)
        with torch.no_grad():
            outputs = self.encoder(inputs_embeds=embeds)
        return outputs.last_hidden_state

    def encode_pooled_torch(self, market_tensors: torch.Tensor) -> torch.Tensor:
        seq_embs = self.encode_sequence_torch(market_tensors)
        return seq_embs.mean(dim=1)

    def _encode_market_tensors(self, market_tensors: torch.Tensor) -> torch.Tensor:
        return self.encode_pooled_torch(market_tensors)

    def forward(self, market_tensors: torch.Tensor, tabular_tensors: torch.Tensor) -> torch.Tensor:
        market_embs = self._encode_market_tensors(market_tensors)
        combined = torch.cat([market_embs, tabular_tensors], dim=-1)
        combined = self.dropout(combined)
        return self.regressor(combined).squeeze(-1)

