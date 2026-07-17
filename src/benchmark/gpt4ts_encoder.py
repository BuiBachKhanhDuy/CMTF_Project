"""GPT4TS predictor for time series forecasting.

Refactor goals:
1. Keep target_scale unified at 1.0 default across baseline and hybrid.
2. Make backbone adaptation explicit and stronger by default.
3. Improve short-horizon performance with overlapping patches and better pooling.
4. Add optional recent-feature residual branch to reduce collapse risk.
5. Use optimizer parameter groups for safer transformer adaptation.
6. Preserve compatibility with BaseTorchMarketPredictor / BaseTorchHybridPredictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from transformers import GPT2Config, GPT2Model

from .baseline_models import (
    BaseTorchHybridPredictor,
    BaseTorchMarketPredictor,
    GLOBAL_LOSS_CONFIG,
    _ensure_market_sequence_tensor,
)


# =====================================================================
# CONFIG
# =====================================================================

@dataclass(frozen=True)
class GPT4TSDefaultConfig:
    num_layers: int = 6
    # Conservative defaults (L3S1 = 28 tokens on 30-step window).
    # The benchmark runner applies horizon-adaptive patch sizes at construction time:
    # L3S1 for 1D/5D (fine-grained), L6S3 for 20D+ (macro patterns, ~3x faster).
    patch_length: int = 3
    patch_stride: int = 1
    dropout: float = 0.1
    pooling: str = "attn"
    # Unfreeze ONLY the top 1 GPT-2 block (lightweight partial adaptation).
    # Full fine-tuning of the surviving stack was removed (too heavy for the
    # target machine); this is now the single supported adaptation depth.
    unfreeze_top_k_blocks: int = 1
    unfreeze_final_layer_norm: bool = True
    head_hidden_dim: int = 128
    backbone_lr: float = 5e-5
    head_lr: float = 3e-4


GPT4TS_DEFAULTS = GPT4TSDefaultConfig()


# =====================================================================
# PATCH EMBEDDING
# =====================================================================

class GPT4TSPatchEmbedder(nn.Module):
    """Convert market windows into GPT-2 input embeddings via temporal patching.

    Input:
        (B, L, C)
    Output:
        (B, T_patch, D)
    """

    def __init__(
        self,
        input_dim: int,
        patch_length: int,
        embed_dim: int,
        patch_stride: int | None = None,
    ):
        super().__init__()

        if patch_length < 1:
            raise ValueError(f"patch_length must be >= 1, got {patch_length}")

        self.input_dim = int(input_dim)
        self.patch_length = int(patch_length)
        self.embed_dim = int(embed_dim)
        self.patch_stride = self.patch_length if patch_stride is None else int(patch_stride)

        if self.patch_stride < 1:
            raise ValueError(f"patch_stride must be >= 1, got {self.patch_stride}")

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.patch_embed = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.embed_dim,
            kernel_size=self.patch_length,
            stride=self.patch_stride,
        )
        self.patch_norm = nn.LayerNorm(self.embed_dim)

    def _pad_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Pad sequence by repeating last timestep so Conv1d patching lands cleanly."""
        seq_len = x.shape[1]

        if seq_len < self.patch_length:
            pad_len = self.patch_length - seq_len
            pad = x[:, -1:, :].expand(-1, pad_len, -1)
            x = torch.cat([x, pad], dim=1)
            seq_len = x.shape[1]

        remainder = (seq_len - self.patch_length) % self.patch_stride
        if remainder != 0:
            pad_len = self.patch_stride - remainder
            pad = x[:, -1:, :].expand(-1, pad_len, -1)
            x = torch.cat([x, pad], dim=1)

        return x

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        x = self._pad_sequence(x)
        x = self.input_norm(x)
        x = x.transpose(1, 2)                 # (B, C, L)
        patches = self.patch_embed(x)         # (B, D, T_patch)
        patches = patches.transpose(1, 2)     # (B, T_patch, D)
        return self.patch_norm(patches)


# =====================================================================
# POOLING
# =====================================================================

class AttentionPooling(nn.Module):
    """Simple learned attention pooling over token sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, D)
        attn_logits = self.score(hidden_states)          # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (attn_weights * hidden_states).sum(dim=1)
        return pooled


# =====================================================================
# SHARED MIXIN
# =====================================================================

class _GPT4TSMixin:
    """Shared GPT4TS helpers for pooling, trainability, features, and optimizer groups."""

    @staticmethod
    def _validate_pooling(pooling: str) -> str:
        pooling = str(pooling).strip().lower()
        valid = {"last", "mean", "last_mean", "attn"}
        if pooling not in valid:
            raise ValueError(f"pooling must be one of {valid}, got {pooling!r}")
        return pooling

    @staticmethod
    def _configure_backbone_trainability(
        gpt2: GPT2Model,
        unfreeze_top_k_blocks: int = GPT4TS_DEFAULTS.unfreeze_top_k_blocks,
        unfreeze_final_layer_norm: bool = GPT4TS_DEFAULTS.unfreeze_final_layer_norm,
    ) -> None:
        """Freeze all GPT-2 params, then optionally unfreeze top-k blocks + ln_f."""
        for param in gpt2.parameters():
            param.requires_grad = False

        if unfreeze_top_k_blocks > 0:
            n_blocks = len(gpt2.h)
            k = min(int(unfreeze_top_k_blocks), n_blocks)
            for block in gpt2.h[n_blocks - k :]:
                for param in block.parameters():
                    param.requires_grad = True

        if unfreeze_final_layer_norm and hasattr(gpt2, "ln_f") and gpt2.ln_f is not None:
            for param in gpt2.ln_f.parameters():
                param.requires_grad = True

    @staticmethod
    def _pool_hidden_states(
        hidden_states: torch.Tensor,
        pooling: str,
        attn_pool: AttentionPooling | None = None,
    ) -> torch.Tensor:
        if pooling == "last":
            return hidden_states[:, -1, :]
        if pooling == "mean":
            return hidden_states.mean(dim=1)
        if pooling == "last_mean":
            return 0.5 * hidden_states[:, -1, :] + 0.5 * hidden_states.mean(dim=1)
        if pooling == "attn":
            if attn_pool is None:
                raise ValueError("attn_pool is required when pooling='attn'")
            return attn_pool(hidden_states)
        raise RuntimeError(f"Unexpected pooling={pooling!r}")

    @staticmethod
    def _recent_window_features(
        market_windows: torch.Tensor,
        input_dim: int,
    ) -> torch.Tensor:
        """Simple direct local features to reduce collapse risk.

        Features per channel:
          - last value
          - mean of recent 3 steps
          - recent trend (last - value 3 steps ago)
        Output dim = 3 * input_dim
        """
        x = _ensure_market_sequence_tensor(market_windows, input_dim)
        seq_len = x.shape[1]

        last = x[:, -1, :]
        recent_len = min(seq_len, 3)
        recent_mean = x[:, -recent_len:, :].mean(dim=1)

        if seq_len >= 3:
            trend = x[:, -1, :] - x[:, -3, :]
        else:
            trend = x[:, -1, :] - x[:, 0, :]

        return torch.cat([last, recent_mean, trend], dim=-1)

    def _build_optimizer_with_param_groups(
        self,
        *,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-5,
    ) -> torch.optim.Optimizer:
        backbone_params = []
        non_backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("gpt2."):
                backbone_params.append(param)
            else:
                non_backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
            )
        if non_backbone_params:
            param_groups.append(
                {"params": non_backbone_params, "lr": head_lr, "weight_decay": weight_decay}
            )

        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer")

        return torch.optim.AdamW(param_groups)

    @staticmethod
    def prediction_diagnostics(pred: torch.Tensor) -> dict[str, float]:
        pred = pred.detach()
        return {
            "pred_mean": float(pred.mean().item()),
            "pred_std": float(pred.std(unbiased=False).item()),
            "pct_pred_pos": float((pred > 0).float().mean().item()),
            "pct_pred_neg": float((pred < 0).float().mean().item()),
            "pct_pred_near_zero_1e4": float((pred.abs() < 1e-4).float().mean().item()),
            "pct_pred_near_zero_1e3": float((pred.abs() < 1e-3).float().mean().item()),
        }


# =====================================================================
# GPT4TS BASELINE
# =====================================================================

class GPT4TSPredictor(BaseTorchMarketPredictor, _GPT4TSMixin):
    """Stronger GPT4TS baseline predictor on multivariate market windows."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = GPT4TS_DEFAULTS.head_hidden_dim,
        num_layers: int = GPT4TS_DEFAULTS.num_layers,
        patch_length: int = GPT4TS_DEFAULTS.patch_length,
        patch_stride: int | None = GPT4TS_DEFAULTS.patch_stride,
        dropout: float = GPT4TS_DEFAULTS.dropout,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
        device: str = "cpu",
        pretrained: bool = True,
        pooling: str = GPT4TS_DEFAULTS.pooling,
        unfreeze_top_k_blocks: int = GPT4TS_DEFAULTS.unfreeze_top_k_blocks,
        unfreeze_final_layer_norm: bool = GPT4TS_DEFAULTS.unfreeze_final_layer_norm,
        use_recent_residual: bool = True,
    ):
        super().__init__(target_scale=target_scale, device=device)

        self.input_dim = int(input_dim)
        self.patch_length = int(patch_length)
        self.patch_stride = patch_stride
        self.huber_delta = float(huber_delta)
        self.sign_penalty_weight = float(sign_penalty_weight)
        self.head_hidden_dim = int(hidden_dim)
        self.pooling = self._validate_pooling(pooling)
        self.unfreeze_top_k_blocks = int(unfreeze_top_k_blocks)
        self.unfreeze_final_layer_norm = bool(unfreeze_final_layer_norm)
        self.use_recent_residual = bool(use_recent_residual)

        if pretrained:
            logger.info("Loading pretrained GPT-2 for GPT4TS...")
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            config = GPT2Config()
            self.gpt2 = GPT2Model(config)

        if num_layers > 0 and num_layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:num_layers]

        self.gpt2_hidden_dim = int(self.gpt2.config.n_embd)
        self.d_model = self.gpt2_hidden_dim
        self.hidden_dim = self.gpt2_hidden_dim
        self.seq_output_dim = self.gpt2_hidden_dim

        self._configure_backbone_trainability(
            self.gpt2,
            unfreeze_top_k_blocks=self.unfreeze_top_k_blocks,
            unfreeze_final_layer_norm=self.unfreeze_final_layer_norm,
        )

        self.patch_embedder = GPT4TSPatchEmbedder(
            input_dim=self.input_dim,
            patch_length=self.patch_length,
            embed_dim=self.gpt2_hidden_dim,
            patch_stride=self.patch_stride,
        )

        self.attn_pool = AttentionPooling(self.gpt2_hidden_dim)

        recent_out_dim = 0
        if self.use_recent_residual:
            recent_in_dim = 3 * self.input_dim
            recent_out_dim = max(self.head_hidden_dim // 2, 1)
            self.recent_proj = nn.Sequential(
                nn.Linear(recent_in_dim, recent_out_dim),
                nn.LayerNorm(recent_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.fc = nn.Sequential(
            nn.Linear(self.gpt2_hidden_dim + recent_out_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden_dim, max(self.head_hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.head_hidden_dim // 2, 1), 1),
        )

        self.to(self.device)

        logger.info(
            "GPT4TS initialized | pooling={} | patch_length={} | patch_stride={} | "
            "unfreeze_top_k_blocks={} | unfreeze_ln_f={} | use_recent_residual={} | target_scale={}",
            self.pooling,
            self.patch_length,
            self.patch_stride,
            self.unfreeze_top_k_blocks,
            self.unfreeze_final_layer_norm,
            self.use_recent_residual,
            self.target_scale,
        )

    def _encode_market_tensors(self, market_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.patch_embedder(market_windows)
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        pooled = self._pool_hidden_states(
            hidden_states,
            self.pooling,
            attn_pool=self.attn_pool,
        )
        return hidden_states, pooled

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self._encode_market_tensors(market_windows)
        return seq_output

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        _, pooled = self._encode_market_tensors(market_windows)
        return pooled

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        pooled = self._encode_tensor(market_windows)

        if self.use_recent_residual:
            recent_feat = self._recent_window_features(market_windows, self.input_dim)
            recent_feat = self.recent_proj(recent_feat)
            pooled = torch.cat([pooled, recent_feat], dim=-1)

        pred = self.fc(pooled)
        return pred.squeeze(-1)

    def fit(self, *args, **kwargs) -> dict:
        kwargs["model_name"] = kwargs.get("model_name", "GPT4TS")
        backbone_lr = kwargs.pop("backbone_lr", GPT4TS_DEFAULTS.backbone_lr)
        head_lr = kwargs.pop("head_lr", GPT4TS_DEFAULTS.head_lr)

        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = self._build_optimizer_with_param_groups(
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                weight_decay=1e-5,
            )
            kwargs["optimizer"] = optimizer

        if "scheduler" not in kwargs or kwargs["scheduler"] is None:
            kwargs["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                kwargs["optimizer"],
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            )

        logger.info(
            "GPT4TS optimizer | backbone_lr={} | head_lr={}",
            backbone_lr,
            head_lr,
        )
        return super().fit(*args, **kwargs)


# =====================================================================
# GPT4TS HYBRID
# =====================================================================

class GPT4TSHybridPredictor(BaseTorchHybridPredictor, _GPT4TSMixin):
    """Stronger GPT4TS hybrid predictor combining sequence and tabular features."""

    def __init__(
        self,
        input_dim: int = 1,
        tabular_dim: int | None = None,
        hidden_dim: int = GPT4TS_DEFAULTS.head_hidden_dim,
        num_layers: int = GPT4TS_DEFAULTS.num_layers,
        patch_length: int = GPT4TS_DEFAULTS.patch_length,
        patch_stride: int | None = GPT4TS_DEFAULTS.patch_stride,
        dropout: float = GPT4TS_DEFAULTS.dropout,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
        device: str = "cpu",
        pretrained: bool = True,
        pooling: str = GPT4TS_DEFAULTS.pooling,
        unfreeze_top_k_blocks: int = GPT4TS_DEFAULTS.unfreeze_top_k_blocks,
        unfreeze_final_layer_norm: bool = GPT4TS_DEFAULTS.unfreeze_final_layer_norm,
        use_recent_residual: bool = True,
    ):
        super().__init__(target_scale=target_scale, device=device)

        self.input_dim = int(input_dim)
        self.patch_length = int(patch_length)
        self.patch_stride = patch_stride
        self.tabular_dim = tabular_dim
        self.huber_delta = float(huber_delta)
        self.sign_penalty_weight = float(sign_penalty_weight)
        self.pooling = self._validate_pooling(pooling)
        self.unfreeze_top_k_blocks = int(unfreeze_top_k_blocks)
        self.unfreeze_final_layer_norm = bool(unfreeze_final_layer_norm)
        self.use_recent_residual = bool(use_recent_residual)
        self.head_hidden_dim = int(hidden_dim)

        if self.tabular_dim is None:
            raise ValueError("tabular_dim must be provided for GPT4TSHybridPredictor")

        if pretrained:
            logger.info("Loading pretrained GPT-2 for GPT4TS Hybrid...")
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            config = GPT2Config()
            self.gpt2 = GPT2Model(config)

        if num_layers > 0 and num_layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:num_layers]

        self.gpt2_hidden_dim = int(self.gpt2.config.n_embd)
        self.seq_dim = self.gpt2_hidden_dim
        self.d_model = self.gpt2_hidden_dim

        self._configure_backbone_trainability(
            self.gpt2,
            unfreeze_top_k_blocks=self.unfreeze_top_k_blocks,
            unfreeze_final_layer_norm=self.unfreeze_final_layer_norm,
        )

        self.patch_embedder = GPT4TSPatchEmbedder(
            input_dim=self.input_dim,
            patch_length=self.patch_length,
            embed_dim=self.gpt2_hidden_dim,
            patch_stride=self.patch_stride,
        )

        self.attn_pool = AttentionPooling(self.gpt2_hidden_dim)

        recent_out_dim = 0
        if self.use_recent_residual:
            recent_in_dim = 3 * self.input_dim
            recent_out_dim = max(hidden_dim // 2, 1)
            self.recent_proj = nn.Sequential(
                nn.Linear(recent_in_dim, recent_out_dim),
                nn.LayerNorm(recent_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.seq_proj = nn.LayerNorm(self.seq_dim)
        self.tab_proj = nn.LayerNorm(hidden_dim)

        fusion_in_dim = self.seq_dim + hidden_dim + recent_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

        self.to(self.device)

        logger.info(
            "GPT4TS Hybrid initialized | pooling={} | patch_length={} | patch_stride={} | "
            "unfreeze_top_k_blocks={} | unfreeze_ln_f={} | use_recent_residual={} | target_scale={}",
            self.pooling,
            self.patch_length,
            self.patch_stride,
            self.unfreeze_top_k_blocks,
            self.unfreeze_final_layer_norm,
            self.use_recent_residual,
            self.target_scale,
        )

    def _encode_sequence_branch(self, market_windows: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.patch_embedder(market_windows)
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        pooled = self._pool_hidden_states(
            hidden_states,
            self.pooling,
            attn_pool=self.attn_pool,
        )
        return pooled

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        seq_emb = self._encode_sequence_branch(market_windows)
        tab_emb = self.tabular_mlp(market_tabular)

        seq_emb = self.seq_proj(seq_emb)
        tab_emb = self.tab_proj(tab_emb)

        fused = [seq_emb, tab_emb]
        if self.use_recent_residual:
            recent_feat = self._recent_window_features(market_windows, self.input_dim)
            recent_feat = self.recent_proj(recent_feat)
            fused.append(recent_feat)

        pred = self.head(torch.cat(fused, dim=-1))
        return pred.squeeze(-1)

    def fit(self, *args, **kwargs) -> dict:
        kwargs["model_name"] = kwargs.get("model_name", "GPT4TS-Hybrid")
        backbone_lr = kwargs.pop("backbone_lr", GPT4TS_DEFAULTS.backbone_lr)
        head_lr = kwargs.pop("head_lr", GPT4TS_DEFAULTS.head_lr)

        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = self._build_optimizer_with_param_groups(
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                weight_decay=1e-5,
            )
            kwargs["optimizer"] = optimizer

        if "scheduler" not in kwargs or kwargs["scheduler"] is None:
            kwargs["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                kwargs["optimizer"],
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            )

        logger.info(
            "GPT4TS Hybrid optimizer | backbone_lr={} | head_lr={}",
            backbone_lr,
            head_lr,
        )
        return super().fit(*args, **kwargs)