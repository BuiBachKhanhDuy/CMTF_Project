"""Tests for the opt-in frozen-projection pooled-embedding cache in ChronosAdapter.

These verify three properties without downloading the real Chronos weights (a
lightweight stub encoder is injected via the ``pipeline`` hook):

1. Default (trainable) mode is unchanged: input_projection trains, no memoization.
2. Frozen-probe mode's cached pooled path is numerically identical to the
   uncached frozen path (pure memoization).
3. Memoization collapses redundant heavy T5 passes to one per unique window,
   and the cache is shared across deepcopied fold models.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.benchmark import chronos_encoder
from src.benchmark.chronos_encoder import ChronosAdapter, clear_frozen_pooled_cache


class _FakeT5Encoder(nn.Module):
    """Deterministic stand-in for the frozen Chronos T5 encoder.

    Counts how many rows it processes so tests can assert memoization removes
    redundant forward passes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.config = types.SimpleNamespace(d_model=d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Make it a non-trivial but fixed transform.
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.forward_calls = 0
        self.rows_seen = 0

    def forward(self, inputs_embeds=None):  # noqa: D401 - mimics HF signature
        self.forward_calls += 1
        self.rows_seen += int(inputs_embeds.shape[0])
        hidden = torch.tanh(self.proj(inputs_embeds))
        return types.SimpleNamespace(last_hidden_state=hidden)


def _make_pipeline(d_model: int) -> types.SimpleNamespace:
    encoder = _FakeT5Encoder(d_model)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model=types.SimpleNamespace(encoder=encoder))
    )


def _make_adapter(freeze: bool, cache: bool = True, d_model: int = 8, input_dim: int = 3):
    pipeline = _make_pipeline(d_model)
    torch.manual_seed(0)
    return ChronosAdapter(
        input_dim=input_dim,
        dropout=0.0,
        device="cpu",
        pipeline=pipeline,
        freeze_input_projection=freeze,
        cache_frozen_embeddings=cache,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_frozen_pooled_cache()
    yield
    clear_frozen_pooled_cache()


def test_default_mode_is_trainable_and_uncached():
    adapter = _make_adapter(freeze=False)
    assert adapter.freeze_input_projection is False
    assert adapter._cache_frozen_embeddings is False
    # Projection remains trainable in the default apple-to-apple adapter.
    assert all(p.requires_grad for p in adapter.input_projection.parameters())
    # No cache entries created by the default path.
    x = torch.randn(4, 5, 3)
    _ = adapter.encode_pooled_torch(x)
    assert len(chronos_encoder._FROZEN_POOLED_CACHE) == 0


def test_frozen_mode_freezes_projection():
    adapter = _make_adapter(freeze=True)
    assert adapter.freeze_input_projection is True
    assert adapter._cache_frozen_embeddings is True
    assert all(not p.requires_grad for p in adapter.input_projection.parameters())


def test_cached_pooled_matches_uncached_frozen_path():
    adapter = _make_adapter(freeze=True)
    x = torch.randn(6, 5, 3)

    # Reference: the uncached frozen computation (mean-pool of frozen T5).
    reference = adapter.encode_sequence_torch(x).mean(dim=1)

    clear_frozen_pooled_cache()
    cached = adapter.encode_pooled_torch(x)  # cache-enabled path

    assert cached.shape == reference.shape
    torch.testing.assert_close(cached, reference, rtol=1e-6, atol=1e-6)


def test_memoization_dedups_rows_and_skips_recompute():
    adapter = _make_adapter(freeze=True)
    encoder = adapter.encoder

    # 3 unique rows, duplicated -> 6 rows total but only 3 unique windows.
    base = torch.randn(3, 5, 3)
    x = torch.cat([base, base], dim=0)

    clear_frozen_pooled_cache()
    encoder.rows_seen = 0
    encoder.forward_calls = 0

    first = adapter.encode_pooled_torch(x)
    # Only the 3 unique windows should ever hit the heavy encoder.
    assert encoder.rows_seen == 3
    assert len(chronos_encoder._FROZEN_POOLED_CACHE) == 3

    # A second pass over the same data must not touch the encoder at all.
    rows_before = encoder.rows_seen
    second = adapter.encode_pooled_torch(x)
    assert encoder.rows_seen == rows_before
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-6)


def test_cache_shared_across_deepcopied_fold_models():
    """OOF deepcopies must share cache entries (identical frozen weights)."""
    adapter = _make_adapter(freeze=True)
    x = torch.randn(4, 5, 3)

    clear_frozen_pooled_cache()
    _ = adapter.encode_pooled_torch(x)
    assert len(chronos_encoder._FROZEN_POOLED_CACHE) == 4

    fold_model = copy.deepcopy(adapter)
    fold_encoder = fold_model.encoder
    fold_encoder.rows_seen = 0

    out = fold_model.encode_pooled_torch(x)
    # Deepcopy preserves the (frozen) projection weights, so the signature and
    # row hashes match -> the fold model reuses the shared cache, no recompute.
    assert fold_encoder.rows_seen == 0
    assert out.shape == (4, fold_model.d_model)


def test_forward_grad_flows_only_to_head_in_frozen_mode():
    adapter = _make_adapter(freeze=True)
    x = torch.randn(5, 5, 3)
    target = torch.randn(5)

    pred = adapter.forward(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    # Frozen projection receives no gradient; the regressor head does.
    for p in adapter.input_projection.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    head_grads = [p.grad for p in adapter.regressor.parameters() if p.grad is not None]
    assert head_grads, "regressor head should receive gradients"
    assert any(torch.count_nonzero(g) > 0 for g in head_grads)
