from __future__ import annotations

import torch

from src.phase2 import (
    CustomTransformerSentimentModel,
    SentimentQueryAttentionHead,
    expected_value_from_probabilities,
)


def test_expected_value_from_probabilities_uses_negative_neutral_positive_order():
    probabilities = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )

    expected = expected_value_from_probabilities(probabilities)

    assert torch.allclose(expected, torch.tensor([-0.7, 0.0, 0.7]))


def test_sentiment_query_attention_head_shapes_and_padding_guard():
    head = SentimentQueryAttentionHead(hidden_dim=16, num_heads=4, dropout=0.0)
    hidden_states = torch.randn(3, 5, 16)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    outputs = head(hidden_states, attention_mask=attention_mask)

    assert outputs["logits"].shape == (3, 3)
    assert outputs["probabilities"].shape == (3, 3)
    assert outputs["expected_value"].shape == (3,)
    assert outputs["attention_weights"].shape == (3, 5)
    assert torch.isfinite(outputs["expected_value"]).all()


def test_custom_transformer_sentiment_model_forward_shapes():
    model = CustomTransformerSentimentModel(
        vocab_size=32,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        max_length=12,
    )
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["hidden_states"].shape == (2, 6, 16)
    assert outputs["pooled_state"].shape == (2, 16)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["probabilities"].shape == (2, 3)
    assert outputs["expected_value"].shape == (2,)