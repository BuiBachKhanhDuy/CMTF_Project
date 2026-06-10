"""Encoder protocol for the ablation benchmark framework.

Every model participating in the ablation must implement this protocol.
Fusion wrappers depend only on this interface, enabling model-agnostic composition.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class BaseEncoder(Protocol):
    """Protocol that every benchmark model must satisfy."""

    @property
    def d_model(self) -> int:
        """Embedding dimension exposed by encode(). 0 means no latent space."""
        ...

    @property
    def supports_sequence(self) -> bool:
        """Whether the model accepts (N, seq_len, input_dim) market windows."""
        ...

    def encode(self, market_windows: np.ndarray) -> np.ndarray:
        """Encode market windows into latent representations.

        Args:
            market_windows: (N, seq_len, input_dim) or model-specific input.

        Returns:
            (N, d_model) latent embeddings.

        Raises:
            NotImplementedError if model has no latent space (d_model == 0).
        """
        ...

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        """Predict forward returns from market data alone.

        Args:
            market_windows: (N, seq_len, input_dim) or model-specific input.

        Returns:
            (N,) predicted log returns.
        """
        ...


@runtime_checkable
class TemporalEncoder(BaseEncoder, Protocol):
    """Optional protocol for encoders that support true sequence-level fusion."""

    @property
    def supports_temporal_fusion(self) -> bool:
        """Whether the encoder exposes differentiable temporal states."""
        ...

    @property
    def sequence_d_model(self) -> int:
        """Hidden size of the temporal sequence returned by encode_sequence_torch()."""
        ...

    def encode_sequence_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Encode market windows into per-timestep hidden states."""
        ...

    def encode_pooled_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Encode market windows into pooled (B, d_model) embeddings (differentiable)."""
        ...

    def predict_market_only_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Differentiable market-only prediction path used for staged fine-tuning."""
        ...

    def encoder_parameters(self) -> list[torch.nn.Parameter]:
        """Parameter list for staged freezing/unfreezing in temporal fusion."""
        ...
