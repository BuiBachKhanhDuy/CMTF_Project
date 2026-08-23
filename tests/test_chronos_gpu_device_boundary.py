"""Regression tests for Chronos CPU tokenization with a CUDA model."""

from __future__ import annotations

import numpy as np
import torch

from src.benchmark.chronos_encoder import ChronosMarketPredictor


class _CpuTokenizingPipeline:
    """Minimal ChronosPipeline stand-in that rejects non-CPU contexts.

    Chronos 1.x tokenizers retain their bin boundaries on CPU; the real
    pipeline moves token IDs to the model device only after tokenization.
    """

    def __init__(self) -> None:
        self.context_devices: list[torch.device] = []

    def predict(self, context: torch.Tensor, **_: object) -> torch.Tensor:
        self.context_devices.append(context.device)
        assert context.device.type == "cpu"
        return torch.zeros((context.shape[0], 2, 1), dtype=torch.float32)


def test_zero_shot_keeps_context_cpu_when_chronos_model_uses_cuda():
    """The CUDA setting is for Chronos weights, not tokenizer input tensors."""
    predictor = object.__new__(ChronosMarketPredictor)
    predictor.device = "cuda"
    predictor.batch_size = 2
    predictor.pipeline = pipeline = _CpuTokenizingPipeline()

    predictions = predictor.zero_shot_predict(
        close_windows=np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
        last_close=np.array([12.0, 22.0], dtype=np.float32),
        num_samples=2,
    )

    assert predictions.shape == (2,)
    assert pipeline.context_devices == [torch.device("cpu")]
