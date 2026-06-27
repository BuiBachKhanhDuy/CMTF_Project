"""GPT4TS predictor for time series forecasting.

Uses a pretrained GPT-2 model to process time series data by projecting inputs
to the transformer's dimension and reading out predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger
import numpy as np

from transformers import GPT2Model, GPT2Config

from .baseline_models import (
    BaseTorchMarketPredictor,
    BaseTorchHybridPredictor,
    _ensure_market_sequence_tensor,
)

class GPT4TSPredictor(BaseTorchMarketPredictor):
    """GPT4TS baseline predictor using a pretrained GPT-2 backbone."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64, # for the head
        num_layers: int = 3, # how many GPT layers to use if subset
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        target_scale: float = 100.0,
        device: str = "cpu",
        pretrained: bool = True,
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        
        # Load GPT-2
        if pretrained:
            logger.info("Loading pretrained GPT-2 for GPT4TS...")
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            config = GPT2Config()
            self.gpt2 = GPT2Model(config)
            
        self.gpt2_hidden_dim = self.gpt2.config.n_embd
        self.d_model = self.gpt2_hidden_dim

        # Keep only the first `num_layers` layers for efficiency
        if num_layers > 0 and num_layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:num_layers]
            
        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(input_dim, self.gpt2_hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.gpt2_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self.to(self.device)

    def _encode_market_tensors(self, market_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        inputs_embeds = self.input_proj(x)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        
        # Pooling: take the last token's representation
        pooled = hidden_states[:, -1, :]
        return hidden_states, pooled

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self._encode_market_tensors(market_windows)
        return seq_output

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        _, pooled = self._encode_market_tensors(market_windows)
        return pooled

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        pooled = self._encode_tensor(market_windows)
        pred = self.fc(pooled)
        return pred.squeeze(-1)
        
    def fit(self, *args, **kwargs) -> dict:
        kwargs["model_name"] = kwargs.get("model_name", "GPT4TS")
        learning_rate = kwargs.get("learning_rate", 1e-3)

        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=learning_rate,
                weight_decay=1e-5,
            )
            kwargs["optimizer"] = optimizer

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            )
            kwargs["scheduler"] = scheduler

        return super().fit(*args, **kwargs)


class GPT4TSHybridPredictor(BaseTorchHybridPredictor):
    """GPT4TS hybrid predictor combining sequence and tabular features."""

    def __init__(
        self,
        input_dim: int = 1,
        tabular_dim: int | None = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        target_scale: float = 100.0,
        device: str = "cpu",
        pretrained: bool = True,
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.tabular_dim = tabular_dim
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        
        if self.tabular_dim is None:
            raise ValueError("tabular_dim must be provided for GPT4TSHybridPredictor")

        # Load GPT-2
        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            config = GPT2Config()
            self.gpt2 = GPT2Model(config)
            
        self.gpt2_hidden_dim = self.gpt2.config.n_embd
        self.seq_dim = self.gpt2_hidden_dim
        self.d_model = self.gpt2_hidden_dim

        # Keep only the first `num_layers` layers for efficiency
        if num_layers > 0 and num_layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:num_layers]
            
        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(input_dim, self.gpt2_hidden_dim)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(self.seq_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)

    def _encode_sequence_branch(self, market_windows: torch.Tensor) -> torch.Tensor:
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        inputs_embeds = self.input_proj(x)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states[:, -1, :]
        return pooled

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        seq_emb = self._encode_sequence_branch(market_windows)
        tab_emb = self.tabular_mlp(market_tabular)
        pred = self.head(torch.cat([seq_emb, tab_emb], dim=-1))
        return pred.squeeze(-1)
        
    def fit(self, *args, **kwargs) -> dict:
        kwargs["model_name"] = kwargs.get("model_name", "GPT4TS-Hybrid")
        learning_rate = kwargs.get("learning_rate", 1e-3)

        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=learning_rate,
                weight_decay=1e-5,
            )
            kwargs["optimizer"] = optimizer

        return super().fit(*args, **kwargs)
