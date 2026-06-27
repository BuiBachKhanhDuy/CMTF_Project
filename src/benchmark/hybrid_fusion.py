"""hybrid_fusion.py

Standalone multimodal hybrid fusion model.

This class implements the Cross-Modal Temporal Fusion (CMTF) architecture
used in the ablation benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .baseline_models import LSTMPredictor, CNNLSTMPredictor, sign_aware_huber_loss
from .news_module import (
    STANDARD_NEWS_DIM,
    NewsProjector,
    _as_bool_mask,
)
from .training_utils import compute_huber_delta
from .gpt4ts_encoder import GPT4TSPredictor
from .chronos_encoder import ChronosAdapter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@dataclass
class _BestState:
    fusion_state: dict[str, torch.Tensor]
    encoder_state: dict[str, torch.Tensor] | None = None


def build_market_encoder(
    model_name: str,
    input_dim: int,
    seq_len: int,
    horizon: int = 1,
    device: str = "cpu",
    **kwargs,
):
    """
    Factory for market-only encoder models used by HybridFusionPredictor.
    """
    name = str(model_name).strip().lower()
    market_dim = input_dim

    if name in {"lstm", "lstm_predictor"}:
        return LSTMPredictor(
            input_dim=input_dim,
            device=device,
            **kwargs,
        )

    if name in {"cnn_lstm", "cnnlstm", "cnn-lstm"}:
        return CNNLSTMPredictor(
            input_dim=input_dim,
            device=device,
            **kwargs,
        )

    if name == "gpt4ts":
        return GPT4TSPredictor(
            input_dim=market_dim,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.3),
            device=device,
        )
    elif name == "chronos":
        return ChronosAdapter(
            input_dim=market_dim,
            model_name="amazon/chronos-t5-small",
            dropout=kwargs.get("dropout", 0.3),
            device=device,
        )

    raise ValueError(
        f"Unsupported market encoder model_name={model_name!r}. "
        "Expected one of: 'lstm', 'cnn_lstm', 'gpt4ts', 'chronos'."
    )


# ---------------------------------------------------------------------------
# HybridFusionPredictor (CMTF implementation)
# ---------------------------------------------------------------------------

class HybridFusionPredictor(nn.Module):
    """Standalone multimodal hybrid fusion model implementing CMTF."""

    def __init__(
        self,
        market_encoder,
        raw_news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        fusion_market_dim: int = 128,
        fusion_hidden_dim: int = 64,
        n_heads: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        use_cross_attention: bool = True,
        use_positional_encoding: bool = True,
        use_news_gate: bool = True,
        use_variance_reg: bool = True,
        use_two_stage: bool = True,
        use_aux_loss: bool = True,
        freeze_market_encoder: bool = False,
        recency_gate_k: int = 5,
        target_scale: float = 100.0,
        aux_loss_weight: float = 0.3,
        encoder_lr_scale: float = 0.1,
        stage1_ratio: float = 0.33,
        market_epochs: int = 100,
        fusion_epochs: int = 60,
        market_patience: int = 15,
        fusion_patience: int = 12,
        news_gate_alpha: float = 1.0,
        variance_reg_coeff: float = 0.001,
        output_mode: str = "market_plus_fusion",
        device: str = "cpu",
        news_projector: NewsProjector | None = None,
        use_interaction_prod: bool = True,
        use_interaction_diff: bool = True,
        use_news_context_prod: bool = True,
        use_cosine_sim: bool = True,
        use_pooled_news: bool = True,
        fusion_style: str = "handcrafted",
        market_query_mode: str = "multi",
    ) -> None:
        super().__init__()

        if not hasattr(market_encoder, "encode_pooled_torch"):
            raise ValueError(
                "market_encoder must expose encode_pooled_torch; "
                "use LSTMPredictor or CNNLSTMPredictor."
            )
        if not hasattr(market_encoder, "encode_sequence_torch"):
            raise ValueError(
                "market_encoder must expose encode_sequence_torch; "
                "use LSTMPredictor or CNNLSTMPredictor."
            )
        if getattr(market_encoder, "d_model", 0) <= 0:
            raise ValueError("market_encoder must expose d_model > 0")

        if output_mode not in {"market_plus_fusion", "fusion_plus_news"}:
            raise ValueError(
                f"Unsupported output_mode={output_mode!r}. "
                "Expected 'market_plus_fusion' or 'fusion_plus_news'."
            )

        self.fusion_style = str(fusion_style).strip().lower()
        if self.fusion_style not in {"handcrafted", "learned"}:
            raise ValueError(
                f"Unsupported fusion_style={fusion_style!r}. "
                "Expected 'handcrafted' or 'learned'."
            )

        self.market_query_mode = str(market_query_mode).strip().lower()
        if self.market_query_mode not in {"multi", "last", "recent", "global"}:
            raise ValueError(
                f"Unsupported market_query_mode={market_query_mode!r}. "
                "Expected one of: 'multi', 'last', 'recent', 'global'."
            )

        self.market_encoder = market_encoder
        self.device = device
        self.raw_news_dim = int(raw_news_dim)
        self.projected_news_dim = int(projected_news_dim)
        self.market_d_model = int(market_encoder.d_model)
        self.market_seq_dim = int(getattr(market_encoder, "hidden_dim", fusion_market_dim))
        self.fusion_market_dim = int(fusion_market_dim)
        self.seq_len = int(seq_len)
        self.huber_delta = float(huber_delta)
        self.sign_penalty_weight = float(sign_penalty_weight)

        self.use_cross_attention = bool(use_cross_attention)
        self.use_variance_reg = bool(use_variance_reg)
        self.use_news_gate = bool(use_news_gate)
        self.use_two_stage = bool(use_two_stage)
        self.use_aux_loss = bool(use_aux_loss)
        self.freeze_market_encoder = bool(freeze_market_encoder)

        self.use_interaction_prod = bool(use_interaction_prod)
        self.use_interaction_diff = bool(use_interaction_diff)
        self.use_news_context_prod = bool(use_news_context_prod)
        self.use_cosine_sim = bool(use_cosine_sim)
        self.use_pooled_news = bool(use_pooled_news)

        self.recency_gate_k = int(recency_gate_k)
        self.target_scale = float(target_scale)
        self.aux_loss_weight = float(aux_loss_weight)
        self.encoder_lr_scale = float(encoder_lr_scale)

        self.stage1_ratio = float(stage1_ratio)
        self.market_epochs = int(market_epochs)
        self.fusion_epochs = int(fusion_epochs)
        self.market_patience = int(market_patience)
        self.fusion_patience = int(fusion_patience)

        self.news_gate_alpha = float(max(0.0, min(1.0, news_gate_alpha)))
        self.variance_reg_coeff = float(max(0.0, variance_reg_coeff))
        self.output_mode = output_mode

        self.direction_epsilon = 0.5
        self.direction_warmup_epochs = 5

        # --- Trainable news projector ---
        self.news_projector: NewsProjector = news_projector or NewsProjector(
            input_dim=self.raw_news_dim,
            output_dim=self.projected_news_dim,
            dropout=dropout,
        )

        # --- News token encoder ---
        self.news_token_mlp = nn.Sequential(
            nn.Linear(self.projected_news_dim, self.fusion_market_dim),
            nn.LayerNorm(self.fusion_market_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_market_dim, self.fusion_market_dim),
            nn.LayerNorm(self.fusion_market_dim),
            nn.ReLU(),
        )

        # Positional encoding for news sequence
        self.news_pos_enc: nn.Embedding | None = None
        if use_positional_encoding:
            self.news_pos_enc = nn.Embedding(self.seq_len, self.fusion_market_dim)
            nn.init.normal_(self.news_pos_enc.weight, std=0.02)

        # Null news token
        self.null_news_token = nn.Parameter(torch.zeros(1, 1, self.fusion_market_dim))
        nn.init.normal_(self.null_news_token, std=0.01)

        # --- Market pooled projection ---
        self.market_proj = nn.Sequential(
            nn.Linear(self.market_d_model, self.fusion_market_dim),
            nn.LayerNorm(self.fusion_market_dim),
            nn.ReLU(),
        )

        # --- Market sequence projection ---
        self.market_seq_proj = nn.Sequential(
            nn.Linear(self.market_seq_dim, self.fusion_market_dim),
            nn.LayerNorm(self.fusion_market_dim),
            nn.ReLU(),
        )

        # --- Recency gate ---
        self.recency_pos_enc = nn.Embedding(self.seq_len, self.fusion_market_dim)
        nn.init.normal_(self.recency_pos_enc.weight, std=0.02)

        self.recency_gate_net = nn.Sequential(
            nn.Linear(self.fusion_market_dim * 3, self.fusion_market_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_market_dim, 1),
            nn.Sigmoid(),
        )

        # --- News gate ---
        self.news_gate: nn.Module | None = None
        if self.use_news_gate:
            self.news_gate = nn.Sequential(
                nn.Linear(self.market_d_model, self.fusion_market_dim),
                nn.Sigmoid(),
            )

        # --- Cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_market_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(self.fusion_market_dim)

        # --- Learned attention pooling over news ---
        self.news_summary_query = nn.Parameter(
            torch.randn(1, 1, self.fusion_market_dim) * 0.02
        )
        self.news_summary_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_market_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.news_summary_norm = nn.LayerNorm(self.fusion_market_dim)

        # --- Fallback pooled-news summarizer for cross-attention OFF ---
        self.news_pool_proj = nn.Sequential(
            nn.Linear(self.fusion_market_dim, self.fusion_market_dim),
            nn.LayerNorm(self.fusion_market_dim),
            nn.ReLU(),
        )

        # --- Dynamic fused feature size ---
        prediction_input_dim = self._prediction_input_dim()

        # --- Fusion prediction head ---
        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, max(fusion_hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fusion_hidden_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.prediction_head[-1].weight)
        nn.init.zeros_(self.prediction_head[-1].bias)

        # --- Fusion delta head ---
        self.fusion_delta_head = nn.Sequential(
            nn.Linear(prediction_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, max(fusion_hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fusion_hidden_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.fusion_delta_head[-1].weight)
        nn.init.zeros_(self.fusion_delta_head[-1].bias)

        # --- News residual head ---
        news_residual_input_dim = self._news_residual_input_dim()
        self.news_residual_head = nn.Sequential(
            nn.Linear(news_residual_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, max(fusion_hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fusion_hidden_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.news_residual_head[-1].weight)
        nn.init.zeros_(self.news_residual_head[-1].bias)

        # --- Market baseline head ---
        self.market_aux_head = nn.Linear(self.fusion_market_dim, 1)
        nn.init.xavier_uniform_(self.market_aux_head.weight)
        nn.init.zeros_(self.market_aux_head.bias)

        self.is_fitted = False
        self.last_attn_weights: torch.Tensor | None = None
        self.last_attended_news: torch.Tensor | None = None
        self.last_recency_gate: torch.Tensor | None = None

        self.to(self.device)

    # ------------------------------------------------------------------
    # Dim helpers
    # ------------------------------------------------------------------

    def _prediction_input_dim(self) -> int:
        # Learned style: let the head learn interactions from [market_latent, attn_out]
        if self.fusion_style == "learned":
            return self.fusion_market_dim * 2

        # Handcrafted style: current feature-rich construction
        dim = self.fusion_market_dim * 2  # market_latent + attn_out

        if self.use_pooled_news:
            dim += self.fusion_market_dim
        if self.use_interaction_prod:
            dim += self.fusion_market_dim
        if self.use_interaction_diff:
            dim += self.fusion_market_dim
        if self.use_news_context_prod:
            dim += self.fusion_market_dim
        if self.use_cosine_sim:
            dim += 1

        return dim

    def _news_residual_input_dim(self) -> int:
        # Learned style: residual learned directly from attended news
        if self.fusion_style == "learned":
            return self.fusion_market_dim

        # Handcrafted style
        return self.fusion_market_dim * 2 if self.use_pooled_news else self.fusion_market_dim

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def fusion_parameters(self) -> list[nn.Parameter]:
        """All parameters except the market encoder."""
        params = []
        for mod in [
            self.news_projector,
            self.news_token_mlp,
            self.recency_pos_enc,
            self.recency_gate_net,
            self.cross_attn,
            self.post_attn_norm,
            self.news_summary_attn,
            self.news_summary_norm,
            self.news_pool_proj,
            self.prediction_head,
            self.fusion_delta_head,
            self.news_residual_head,
            self.market_proj,
            self.market_seq_proj,
            self.market_aux_head,
        ]:
            params += list(mod.parameters())

        params.append(self.null_news_token)
        params.append(self.news_summary_query)

        if self.news_gate is not None:
            params += list(self.news_gate.parameters())
        if self.news_pos_enc is not None:
            params += list(self.news_pos_enc.parameters())

        return params

    def set_market_encoder_requires_grad(self, enabled: bool) -> None:
        for p in self.market_encoder.encoder_parameters():
            p.requires_grad_(enabled)

    # ------------------------------------------------------------------
    # News encoding
    # ------------------------------------------------------------------

    def _encode_news_tokens(
        self,
        news_proj: torch.Tensor,
        news_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = news_proj.shape
        device = news_proj.device

        if news_mask is None:
            pad_mask = news_proj.abs().sum(-1) == 0
        else:
            pad_mask = news_mask.to(dtype=torch.bool, device=device)

        flat = news_proj.reshape(B * S, -1)
        tokens = self.news_token_mlp(flat).reshape(B, S, self.fusion_market_dim)

        if self.news_pos_enc is not None:
            pos = torch.arange(S, device=device).clamp(max=self.news_pos_enc.num_embeddings - 1)
            tokens = tokens + self.news_pos_enc(pos).unsqueeze(0)

        null = self.null_news_token.expand(B, -1, -1)
        tokens_with_null = torch.cat([null, tokens], dim=1)

        null_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        pad_mask_with_null = torch.cat([null_mask, pad_mask], dim=1)

        return tokens_with_null, pad_mask_with_null

    def _apply_recency_gating(
        self,
        market_latent: torch.Tensor,
        news_tokens: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = news_tokens.shape
        device = news_tokens.device

        positions = torch.arange(S, device=device).clamp(
            max=self.recency_pos_enc.num_embeddings - 1
        )
        pos_emb = self.recency_pos_enc(positions).unsqueeze(0).expand(B, -1, -1)
        market_ctx = market_latent.unsqueeze(1).expand(-1, S, -1)

        gate_input = torch.cat([news_tokens, market_ctx, pos_emb], dim=-1)
        gate = self.recency_gate_net(gate_input)

        if self.recency_gate_k > 0:
            distances = (S - 1) - torch.arange(S, device=device, dtype=torch.float32)
            decay = torch.exp(-distances / float(max(self.recency_gate_k, 1))).view(1, S, 1)
            gate = gate * decay

        gate = gate.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        self.last_recency_gate = gate.detach()

        return news_tokens * gate

    def _build_market_queries(
        self,
        market_seq_proj: torch.Tensor,
    ) -> torch.Tensor:
        last_q = market_seq_proj[:, -1, :]

        recent_len = min(5, market_seq_proj.size(1))
        recent_q = market_seq_proj[:, -recent_len:, :].mean(dim=1)

        global_q = market_seq_proj.mean(dim=1)

        if self.market_query_mode == "last":
            return last_q.unsqueeze(1)

        if self.market_query_mode == "recent":
            return recent_q.unsqueeze(1)

        if self.market_query_mode == "global":
            return global_q.unsqueeze(1)

        if self.market_query_mode == "multi":
            return torch.stack([last_q, recent_q, global_q], dim=1)

        raise RuntimeError(f"Unexpected market_query_mode={self.market_query_mode!r}")

    def _pool_news_summary(
        self,
        news_tokens_gated: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = news_tokens_gated.size(0)
        query = self.news_summary_query.expand(B, -1, -1)
        pooled, _ = self.news_summary_attn(
            query=query,
            key=news_tokens_gated,
            value=news_tokens_gated,
            key_padding_mask=pad_mask,
        )
        return self.news_summary_norm(pooled.squeeze(1))

    def _apply_news_gate(
        self,
        attn_out: torch.Tensor,
        market_emb: torch.Tensor,
    ) -> torch.Tensor:
        sigmoid_gate = self.news_gate(market_emb)
        effective_gate = (
            self.news_gate_alpha * sigmoid_gate
            + (1.0 - self.news_gate_alpha)
        )
        gated = attn_out * effective_gate
        return 0.5 * attn_out + 0.5 * gated

    # ------------------------------------------------------------------
    # Forward fusion
    # ------------------------------------------------------------------

    def _build_fused_features(
        self,
        market_latent: torch.Tensor,
        attn_out: torch.Tensor,
        pooled_news: torch.Tensor,
    ) -> torch.Tensor:
        # Learned style: let model infer interaction from core modalities only
        if self.fusion_style == "learned":
            return torch.cat([market_latent, attn_out], dim=-1)

        # Handcrafted style: current feature-rich path
        interaction_prod = market_latent * attn_out
        interaction_diff = torch.abs(market_latent - attn_out)
        news_context_prod = market_latent * pooled_news
        cosine_sim = F.cosine_similarity(market_latent, attn_out, dim=-1).unsqueeze(-1)

        features: list[torch.Tensor] = [market_latent, attn_out]

        if self.use_pooled_news:
            features.append(pooled_news)
        if self.use_interaction_prod:
            features.append(interaction_prod)
        if self.use_interaction_diff:
            features.append(interaction_diff)
        if self.use_news_context_prod:
            features.append(news_context_prod)
        if self.use_cosine_sim:
            features.append(cosine_sim)

        return torch.cat(features, dim=-1)

    def _build_news_residual_input(
        self,
        attn_out: torch.Tensor,
        pooled_news: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_style == "learned":
            return attn_out

        if self.use_pooled_news:
            return torch.cat([attn_out, pooled_news], dim=-1)

        return attn_out

    def _forward_fusion(
        self,
        market_emb: torch.Tensor,
        market_seq: torch.Tensor,
        news_proj: torch.Tensor,
        news_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        market_latent = self.market_proj(market_emb)                  # (B, D)
        market_seq_proj = self.market_seq_proj(market_seq)            # (B, T, D)
        market_queries = self._build_market_queries(market_seq_proj)  # (B, Q, D)

        news_tokens, pad_mask = self._encode_news_tokens(news_proj, news_mask)

        null_token = news_tokens[:, :1, :]
        seq_tokens = news_tokens[:, 1:, :]
        seq_pad = pad_mask[:, 1:]
        seq_gated = self._apply_recency_gating(market_latent, seq_tokens, seq_pad)
        news_tokens_gated = torch.cat([null_token, seq_gated], dim=1)

        pooled_news = self._pool_news_summary(news_tokens_gated, pad_mask)  # (B, D)

        if self.use_cross_attention:
            attn_out_multi, attn_weights = self.cross_attn(
                query=market_queries,
                key=news_tokens_gated,
                value=news_tokens_gated,
                key_padding_mask=pad_mask,
            )
            attn_out_multi = self.post_attn_norm(attn_out_multi)
            attn_out = attn_out_multi.mean(dim=1)

            if self.news_gate is not None:
                attn_out = self._apply_news_gate(attn_out, market_emb)

            self.last_attn_weights = attn_weights.detach() if attn_weights is not None else None
            self.last_attended_news = attn_out.detach()

        else:
            valid_mask = (~pad_mask).unsqueeze(-1).float()
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            pooled_news_fallback = (news_tokens_gated * valid_mask).sum(dim=1) / denom
            attn_out = self.news_pool_proj(pooled_news_fallback)

            if self.news_gate is not None:
                attn_out = self._apply_news_gate(attn_out, market_emb)

            self.last_attn_weights = None
            self.last_attended_news = attn_out.detach()

        fused = self._build_fused_features(market_latent, attn_out, pooled_news)

        fusion_pred = self.prediction_head(fused).squeeze(-1)
        market_pred = self.market_aux_head(market_latent).squeeze(-1)
        fusion_delta = self.fusion_delta_head(fused).squeeze(-1)

        news_residual_input = self._build_news_residual_input(attn_out, pooled_news)
        news_residual = self.news_residual_head(news_residual_input).squeeze(-1)

        if self.output_mode == "market_plus_fusion":
            final_pred = market_pred + fusion_delta
        elif self.output_mode == "fusion_plus_news":
            final_pred = fusion_pred + news_residual
        else:
            raise RuntimeError(f"Unexpected output_mode={self.output_mode!r}")

        aux_pred = market_pred

        return final_pred, aux_pred

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        fused_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        enable_direction = epoch >= self.direction_warmup_epochs

        main_loss = sign_aware_huber_loss(
            fused_pred,
            target,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
            direction_epsilon=self.direction_epsilon,
            enable_direction_loss=enable_direction,
        )

        total = main_loss

        if self.use_aux_loss:
            aux_loss = nn.functional.huber_loss(aux_pred, target, delta=self.huber_delta)
            total = total + self.aux_loss_weight * aux_loss

        if self.use_variance_reg and self.last_attended_news is not None:
            attn_var = self.last_attended_news.var(dim=0).mean()
            floor = torch.tensor(0.01, device=total.device)
            total = total + self.variance_reg_coeff * torch.relu(floor - attn_var)

        return total

    # ------------------------------------------------------------------
    # Pre-bake encoder outputs (stage 1 only)
    # ------------------------------------------------------------------

    def _prebake_encoder_outputs(
        self,
        market_windows: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.market_encoder.eval()
        X = torch.as_tensor(np.asarray(market_windows, dtype=np.float32))
        seqs, embs, preds = [], [], []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i : i + batch_size].to(self.device)
                seqs.append(self.market_encoder.encode_sequence_torch(xb).cpu().numpy())
                embs.append(self.market_encoder.encode_pooled_torch(xb).cpu().numpy())
                preds.append(self.market_encoder.predict_market_only_torch(xb).cpu().numpy())

        return (
            np.concatenate(seqs, axis=0),
            np.concatenate(embs, axis=0),
            np.concatenate(preds, axis=0),
        )

    # ------------------------------------------------------------------
    # Training stages
    # ------------------------------------------------------------------

    def _train_stage1(
        self,
        market_train: np.ndarray,
        news_train_raw: np.ndarray,
        targets_train: np.ndarray,
        market_val: np.ndarray,
        news_val_raw: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None,
        news_mask_val: np.ndarray | None,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        patience: int,
        batch_size: int,
    ) -> dict:
        logger.info("HybridFusion stage 1: pre-baking encoder outputs")
        seq_tr, emb_tr, mpred_tr = self._prebake_encoder_outputs(market_train, batch_size)
        seq_v, emb_v, mpred_v = self._prebake_encoder_outputs(market_val, batch_size)

        news_tr_proj = self.news_projector.ensure_projected(
            np.asarray(news_train_raw, dtype=np.float32)
        )
        news_v_proj = self.news_projector.ensure_projected(
            np.asarray(news_val_raw, dtype=np.float32)
        )
        mask_tr = _as_bool_mask(news_mask_train, news_tr_proj).astype(bool)
        mask_v = _as_bool_mask(news_mask_val, news_v_proj).astype(bool)

        y_tr = np.asarray(targets_train, dtype=np.float32) * self.target_scale
        y_v = np.asarray(targets_val, dtype=np.float32) * self.target_scale

        store_raw_tr = news_train_raw.shape[-1] == self.raw_news_dim
        news_loader_tr = (
            np.asarray(news_train_raw, dtype=np.float32) if store_raw_tr else news_tr_proj
        )
        store_raw_v = news_val_raw.shape[-1] == self.raw_news_dim
        news_loader_v = (
            np.asarray(news_val_raw, dtype=np.float32) if store_raw_v else news_v_proj
        )

        return self._run_epoch_loop(
            (seq_tr, emb_tr), news_loader_tr, mask_tr, y_tr, mpred_tr,
            (seq_v, emb_v), news_loader_v, mask_v, y_v, mpred_v,
            optimizer=optimizer,
            scheduler=None,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            stage_name="Stage1",
            use_live_encoder=False,
            store_raw_news=store_raw_tr,
            save_encoder_state=False,
        )

    def _train_stage2(
        self,
        market_train: np.ndarray,
        news_train_raw: np.ndarray,
        targets_train: np.ndarray,
        market_val: np.ndarray,
        news_val_raw: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None,
        news_mask_val: np.ndarray | None,
        base_lr: float,
        epochs: int,
        patience: int,
        batch_size: int,
    ) -> dict:
        self.set_market_encoder_requires_grad(True)

        market_tr_t = torch.as_tensor(np.asarray(market_train, dtype=np.float32))
        market_v_t = torch.as_tensor(np.asarray(market_val, dtype=np.float32))

        news_tr_raw = np.asarray(news_train_raw, dtype=np.float32)
        news_v_raw = np.asarray(news_val_raw, dtype=np.float32)
        store_raw = news_tr_raw.shape[-1] == self.raw_news_dim

        with torch.no_grad():
            news_tr_proj = self.news_projector.ensure_projected(news_tr_raw)
            news_v_proj = self.news_projector.ensure_projected(news_v_raw)
        mask_tr = _as_bool_mask(news_mask_train, news_tr_proj).astype(bool)
        mask_v = _as_bool_mask(news_mask_val, news_v_proj).astype(bool)

        y_tr = np.asarray(targets_train, dtype=np.float32) * self.target_scale
        y_v = np.asarray(targets_val, dtype=np.float32) * self.target_scale

        news_loader_tr = news_tr_raw if store_raw else news_tr_proj
        news_loader_v = news_v_raw if store_raw else news_v_proj

        encoder_lr = base_lr * self.encoder_lr_scale
        optimizer = torch.optim.AdamW(
            [
                {"params": self.fusion_parameters(), "lr": base_lr},
                {"params": self.market_encoder.encoder_parameters(), "lr": encoder_lr},
            ],
            weight_decay=1e-5,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )

        return self._run_epoch_loop(
            market_tr_t, news_loader_tr, mask_tr, y_tr, None,
            market_v_t, news_loader_v, mask_v, y_v, None,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            stage_name="Stage2",
            use_live_encoder=True,
            store_raw_news=store_raw,
            save_encoder_state=True,
        )

    def _run_epoch_loop(
        self,
        X_tr,
        news_tr,
        mask_tr: np.ndarray,
        y_tr: np.ndarray,
        mpred_tr,
        X_v,
        news_v,
        mask_v: np.ndarray,
        y_v: np.ndarray,
        mpred_v,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        epochs: int = 0,
        patience: int = 0,
        batch_size: int = 32,
        stage_name: str = "",
        use_live_encoder: bool = False,
        store_raw_news: bool = False,
        save_encoder_state: bool = False,
    ) -> dict:
        n_tr = len(y_tr)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state: _BestState | None = None
        patience_counter = 0

        news_tr_t = torch.as_tensor(news_tr, dtype=torch.float32)
        mask_tr_t = torch.as_tensor(mask_tr, dtype=torch.bool)
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
        news_v_t = torch.as_tensor(news_v, dtype=torch.float32)
        mask_v_t = torch.as_tensor(mask_v, dtype=torch.bool)
        y_v_t = torch.as_tensor(y_v, dtype=torch.float32)

        if not use_live_encoder:
            seq_tr_np, emb_tr_np = X_tr
            seq_v_np, emb_v_np = X_v
            seq_tr_t = torch.as_tensor(seq_tr_np, dtype=torch.float32)
            emb_tr_t = torch.as_tensor(emb_tr_np, dtype=torch.float32)
            seq_v_t = torch.as_tensor(seq_v_np, dtype=torch.float32)
            emb_v_t = torch.as_tensor(emb_v_np, dtype=torch.float32)
        else:
            X_tr_t = X_tr
            X_v_t = X_v

        for epoch in range(int(epochs)):
            self.train()
            if not use_live_encoder:
                self.market_encoder.eval()

            perm = torch.randperm(n_tr)
            epoch_loss, n_batches = 0.0, 0

            for i in range(0, n_tr, batch_size):
                idx = perm[i : i + batch_size]

                mb_news = news_tr_t[idx].to(self.device)
                mb_mask = mask_tr_t[idx].to(self.device)
                mb_y = y_tr_t[idx].to(self.device)

                if use_live_encoder:
                    mb_market = X_tr_t[idx].to(self.device)
                    mb_emb = self.market_encoder.encode_pooled_torch(mb_market)
                    mb_seq = self.market_encoder.encode_sequence_torch(mb_market)
                else:
                    mb_seq = seq_tr_t[idx].to(self.device)
                    mb_emb = emb_tr_t[idx].to(self.device)

                if store_raw_news:
                    mb_news_proj = self.news_projector(mb_news)
                else:
                    mb_news_proj = mb_news

                optimizer.zero_grad()
                fused_pred, aux_pred = self._forward_fusion(mb_emb, mb_seq, mb_news_proj, mb_mask)
                loss = self._compute_loss(fused_pred, aux_pred, mb_y, epoch=epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            self.eval()
            val_loss_accum, v_batches = 0.0, 0

            with torch.no_grad():
                for i in range(0, len(y_v), batch_size):
                    sl = slice(i, i + batch_size)
                    mb_news_v = news_v_t[sl].to(self.device)
                    mb_mask_v = mask_v_t[sl].to(self.device)
                    mb_y_v = y_v_t[sl].to(self.device)

                    if use_live_encoder:
                        mb_market_v = X_v_t[sl].to(self.device)
                        mb_emb_v = self.market_encoder.encode_pooled_torch(mb_market_v)
                        mb_seq_v = self.market_encoder.encode_sequence_torch(mb_market_v)
                    else:
                        mb_seq_v = seq_v_t[sl].to(self.device)
                        mb_emb_v = emb_v_t[sl].to(self.device)

                    if store_raw_news:
                        mb_news_proj_v = self.news_projector(mb_news_v)
                    else:
                        mb_news_proj_v = mb_news_v

                    fp_v, ap_v = self._forward_fusion(mb_emb_v, mb_seq_v, mb_news_proj_v, mb_mask_v)
                    v_loss = self._compute_loss(fp_v, ap_v, mb_y_v, epoch=epoch)
                    val_loss_accum += float(v_loss.item())
                    v_batches += 1

            val_loss = val_loss_accum / max(v_batches, 1)
            history["val_loss"].append(val_loss)
            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _BestState(
                    fusion_state=_clone_state(self),
                    encoder_state=(
                        _clone_state(self.market_encoder)
                        if save_encoder_state
                        else None
                    ),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("{} early stop at epoch {}", stage_name, epoch + 1)
                    break

        if best_state is not None:
            self.load_state_dict(best_state.fusion_state)
            if best_state.encoder_state is not None:
                self.market_encoder.load_state_dict(best_state.encoder_state)

        return history

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        market_windows_train: np.ndarray,
        news_embs_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray,
        news_embs_val: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        market_fit_kwargs: dict | None = None,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12,
    ) -> dict:
        market_fit_kwargs = market_fit_kwargs or {}

        y_scaled = np.asarray(targets_train, dtype=np.float32) * self.target_scale
        self.huber_delta = compute_huber_delta(y_scaled)

        logger.info("HybridFusion/CMTF: fitting market encoder")
        self.market_encoder.fit(
            market_windows_train,
            targets_train,
            market_windows_val,
            targets_val,
            **market_fit_kwargs,
        )

        logger.info("HybridFusion/CMTF: stage 1 (frozen encoder, fusion warm-up)")
        self.set_market_encoder_requires_grad(False)

        stage1_epochs = max(3, int(epochs * self.stage1_ratio))
        stage1_epochs = min(stage1_epochs, max(3, epochs - 1))
        optimizer_s1 = torch.optim.AdamW(
            self.fusion_parameters(), lr=lr, weight_decay=1e-5
        )

        hist1 = self._train_stage1(
            market_windows_train, news_embs_train, targets_train,
            market_windows_val, news_embs_val, targets_val,
            news_mask_train, news_mask_val,
            optimizer=optimizer_s1,
            epochs=stage1_epochs,
            patience=min(patience, 8),
            batch_size=batch_size,
        )

        if not self.use_two_stage or self.freeze_market_encoder:
            logger.info(
                "HybridFusion/CMTF: skipping stage 2 (use_two_stage={}, freeze={})",
                self.use_two_stage,
                self.freeze_market_encoder,
            )
            self.is_fitted = True
            return hist1

        logger.info(
            "HybridFusion/CMTF: stage 2 (end-to-end fine-tuning, encoder_lr_scale={})",
            self.encoder_lr_scale,
        )

        hist2 = self._train_stage2(
            market_windows_train, news_embs_train, targets_train,
            market_windows_val, news_embs_val, targets_val,
            news_mask_train, news_mask_val,
            base_lr=lr,
            epochs=max(10, epochs - stage1_epochs),
            patience=patience,
            batch_size=batch_size,
        )

        self.is_fitted = True
        return {
            "train_loss": hist1["train_loss"] + hist2["train_loss"],
            "val_loss": hist1["val_loss"] + hist2["val_loss"],
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
        news_mask: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HybridFusionPredictor.predict called before fit.")

        self.eval()
        market_windows = np.asarray(market_windows, dtype=np.float32)
        news_proj = self.news_projector.ensure_projected(
            np.asarray(news_embs, dtype=np.float32)
        )
        mask = _as_bool_mask(news_mask, news_proj).astype(bool)

        X_t = torch.as_tensor(market_windows, dtype=torch.float32)
        N_t = torch.as_tensor(news_proj, dtype=torch.float32)
        NM_t = torch.as_tensor(mask, dtype=torch.bool)

        preds = np.zeros(len(market_windows), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(market_windows), batch_size):
                sl = slice(i, i + batch_size)
                mb_x = X_t[sl].to(self.device)
                mb_n = N_t[sl].to(self.device)
                mb_nm = NM_t[sl].to(self.device)

                mb_emb = self.market_encoder.encode_pooled_torch(mb_x)
                mb_seq = self.market_encoder.encode_sequence_torch(mb_x)
                fused_pred, _ = self._forward_fusion(mb_emb, mb_seq, mb_n, mb_nm)
                preds[sl] = fused_pred.cpu().numpy()

        return (preds / self.target_scale).astype(np.float32)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.market_encoder.predict_market_only(market_windows)


# Backward/forward naming convenience
CMTFPredictor = HybridFusionPredictor