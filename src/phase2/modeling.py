"""Shared title-encoder models for Phase 2 sentiment comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def expected_value_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Decode a scalar sentiment score from class probabilities."""

    label_values = probabilities.new_tensor([-1.0, 0.0, 1.0])
    return (probabilities * label_values).sum(dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic positional encoding for the custom transformer branch."""

    def __init__(self, hidden_dim: int, max_length: int = 512) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe = torch.zeros(max_length, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        return hidden_states + self.pe[:, :seq_len, :]


class SentimentQueryAttentionHead(nn.Module):
    """Learned sentiment query that cross-attends over token-level states."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_classes != 3:
            raise ValueError("Phase 2 sentiment head expects exactly 3 classes")

        self.sentiment_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.register_buffer(
            "label_values",
            torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, seq_len, hidden_dim)")

        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = hidden_states.new_ones((batch_size, seq_len), dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)

        key_padding_mask = ~attention_mask
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad] = False

        query = self.sentiment_query.expand(batch_size, -1, -1)
        attended, attn_weights = self.cross_attention(
            query,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        pooled = self.output_norm(attended.squeeze(1))
        pooled = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        logits = self.classifier(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        expected_value = (probabilities * self.label_values.to(probabilities.device)).sum(dim=-1)
        return {
            "pooled_state": pooled,
            "logits": logits,
            "probabilities": probabilities,
            "expected_value": expected_value,
            "attention_weights": attn_weights.squeeze(1),
        }


class CustomTransformerSentimentModel(nn.Module):
    """Lightweight TransformerEncoder branch for text-only Phase 2 comparison."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_length: int = 256,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        self.pad_token_id = int(pad_token_id)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_token_id)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim=hidden_dim, max_length=max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = SentimentQueryAttentionHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)

        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        encoded = self.encoder(hidden_states, src_key_padding_mask=~attention_mask.bool())
        outputs = self.head(encoded, attention_mask=attention_mask)
        outputs["hidden_states"] = encoded
        return outputs


class PhoBERTSentimentModel(nn.Module):
    """PhoBERT backbone plus the shared learned-query cross-attention head."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        projection_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "PhoBERTSentimentModel requires the transformers package. "
                "Install it from requirements.txt before running Phase 2 training."
            ) from exc

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = int(self.backbone.config.hidden_size)
        self.projection = (
            nn.Identity()
            if backbone_dim == projection_dim
            else nn.Linear(backbone_dim, projection_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = SentimentQueryAttentionHead(
            hidden_dim=projection_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(self.projection(backbone_outputs.last_hidden_state))
        outputs = self.head(hidden_states, attention_mask=attention_mask)
        outputs["hidden_states"] = hidden_states
        return outputs