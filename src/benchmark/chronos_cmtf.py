"""Fixed Chronos + Cross-Modal Temporal Fusion predictor.

FIXES from original implementation:
1. ✓ Reuses ChronosLoRAPredictor backbone (no double wrapping)
2. ✓ Simplified fusion architecture (single attention mechanism)
3. ✓ Residual design: baseline_pred + news_residual (direct path IS the baseline)
4. ✓ Removed unused classification head
5. ✓ Removed news density weighting bias
6. ✓ Matched hyperparameters with LoRA baseline
7. ✓ Same loss function as baseline

ADDITIONAL FIXES (v7):
8. ✓ Xavier init on residual head final layer (was zeros — blocked gradients)
9. ✓ LayerNorm after cross-attention (stabilizes gradient flow)
10. ✓ Learnable news_weight parameter (init=0.1, grows if news is useful)
11. ✓ Increased default hidden_dim 128 → 256
12. ✓ Zero-news parity gate: fusion output is zeroed when all news is masked,
       so CMTF == baseline LoRA exactly when no news is present
13. ✓ Checkpoint loading uses strict=False + version tag v7 to avoid
       silent corruption from old v6 checkpoints

Architecture:
    pred = baseline_pred + news_gate * fusion(baseline_features, news_emb)

    The baseline_pred comes from the frozen/trainable LoRA path unchanged.
    The fusion head adds a news-conditioned residual ON TOP.
    When all news slots are masked, news_gate=0 and CMTF == baseline exactly.

    This enables a FAIR test: does adding news improve over market-only?
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from .baseline_models import ChronosLoRAEncoderBackbone, sign_aware_huber_loss

CMTF_VERSION = "v8"


class _LegacyChronosBackboneAdapter:
    """Minimal adapter for predictor objects that do not expose `.backbone`."""

    def __init__(self, chronos_predictor, device: str = "cpu") -> None:
        self.chronos = chronos_predictor
        self.device = device
        self.d_model = int(chronos_predictor.d_model)
        self.tokenizer = chronos_predictor.pipeline.tokenizer

        pipeline_model = chronos_predictor.pipeline.model
        self.transformer = getattr(pipeline_model, "model", pipeline_model).to(device)

    def tokenize_windows(self, close_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        close_tensor = torch.as_tensor(close_windows, dtype=torch.float32)
        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(close_tensor)
        return token_ids.cpu().numpy(), attention_mask.cpu().numpy()

    def encode_tokenized(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.transformer.encoder(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        hidden = encoder_out.last_hidden_state
        mask = attention_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def trainable_parameter_names(self) -> list[str]:
        return []

    def checkpoint_state(self) -> dict:
        return {}

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        return None


class CrossModalFusionHead(nn.Module):
    """Lightweight cross-modal fusion head.

    Architecture:
        1. Compress news embeddings to match market dimension
        2. Cross-attention: market (query) attends to news (key/value)
        3. Concatenate [market_emb, attended_news, tabular]
        4. Simple regression head

    No gates, no direct paths, no architectural bias — just learn if news helps.

    Attributes:
        market_dim: Dimension of market embeddings from Chronos
        news_dim: Dimension of news embeddings (e.g., 768 for BERT)
        tabular_dim: Dimension of optional tabular features
        hidden_dim: Hidden layer size for fusion
        n_heads: Number of attention heads
    """

    def __init__(
        self,
        market_dim: int = 512,
        news_dim: int = 768,
        tabular_dim: int = 0,
        hidden_dim: int | None = None,
        fusion_dim: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.market_dim = market_dim
        self.news_dim = news_dim
        self.tabular_dim = tabular_dim
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else (fusion_dim if fusion_dim is not None else 256))
        self.seq_len = seq_len

        self.news_proj = nn.Sequential(
            nn.Linear(news_dim, market_dim),
            nn.LayerNorm(market_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=market_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.tabular_proj = (
            nn.Linear(tabular_dim, self.hidden_dim)
            if tabular_dim > 0
            else None
        )

        fusion_input_dim = market_dim * 2
        if tabular_dim > 0:
            fusion_input_dim += self.hidden_dim

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        market_emb: torch.Tensor,
        news_emb: torch.Tensor,
        tabular_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        news_proj = self.news_proj(news_emb)

        news_mask = (news_emb.abs().sum(-1) == 0)
        all_masked = news_mask.all(dim=1)
        if all_masked.any():
            news_mask = news_mask.clone()
            news_mask[all_masked] = False

        query = market_emb.unsqueeze(1)
        attended_news, attn_weights = self.cross_attn(
            query,
            news_proj,
            news_proj,
            key_padding_mask=news_mask,
        )

        attended_news = attended_news.squeeze(1)
        self.last_attn_weights = attn_weights.detach()

        fusion_input = torch.cat([market_emb, attended_news], dim=-1)

        if self.tabular_proj is not None and tabular_emb is not None:
            tabular_h = self.tabular_proj(tabular_emb)
            fusion_input = torch.cat([fusion_input, tabular_h], dim=-1)

        return self.regression_head(fusion_input).squeeze(-1)


SimpleFusionHead = CrossModalFusionHead


class ResidualNewsFusionHead(nn.Module):
    """News-conditioned residual branch on top of the baseline market prediction.

    Contract: when all news slots are masked (no news available), the forward()
    output is exactly zero — so the caller's `baseline_pred + fusion(...)` equals
    `baseline_pred` unchanged. This preserves zero-news parity with the LoRA baseline.

    The gate is a per-sample binary float (1.0 if any news present, 0.0 otherwise),
    applied AFTER the residual head so gradients still flow through the head on
    mixed batches where some samples have news and some do not.
    """

    def __init__(
        self,
        baseline_dim: int,
        market_dim: int = 512,
        news_dim: int = 768,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.baseline_dim = int(baseline_dim)
        self.market_dim = int(market_dim)
        self.news_dim = int(news_dim)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = seq_len

        self.market_query_proj = nn.Sequential(
            nn.Linear(self.baseline_dim, self.market_dim),
            nn.LayerNorm(self.market_dim),
            nn.ReLU(),
        )
        self.news_proj = nn.Sequential(
            nn.Linear(self.news_dim, self.market_dim),
            nn.LayerNorm(self.market_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.market_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # FIX 3: LayerNorm after cross-attention stabilizes gradient flow
        self.post_attn_norm = nn.LayerNorm(self.market_dim)

        # Learnable positional encoding over the news window so the model
        # can learn recency preference (recent bar T > bar T-29).
        max_seq = int(seq_len) if seq_len is not None else 30
        self.news_pos_enc = nn.Embedding(max_seq, self.market_dim)
        nn.init.normal_(self.news_pos_enc.weight, std=0.02)

        self.null_news_token = nn.Parameter(torch.zeros(1, 1, self.market_dim))

        self.residual_head = nn.Sequential(
            nn.Linear(self.baseline_dim + self.market_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        # FIX 1: Xavier init — zero init blocked all gradient flow for first epochs
        nn.init.xavier_uniform_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # FIX 4: Learnable scale for news contribution — starts conservative at 0.1
        self.news_weight = nn.Parameter(torch.tensor(0.1))

        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        baseline_features: torch.Tensor,
        news_emb: torch.Tensor,
        news_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns news-conditioned residual. Exactly zero when no news is present.

        Args:
            baseline_features: (B, baseline_dim) market features from LoRA path
            news_emb: (B, S, news_dim) news embeddings; all-zero rows = padding
            news_mask: (B, S) bool mask, True = ignore. Inferred from news_emb if None.

        Returns:
            (B,) residual to add to baseline_pred. Zero where no news exists.
        """
        if news_mask is None:
            # True where entire news slot is zero-padded
            computed_mask = news_emb.abs().sum(-1) == 0  # (B, S)
        else:
            computed_mask = news_mask.to(device=news_emb.device, dtype=torch.bool)

        # FIX 1 (High): Compute per-sample gate BEFORE modifying the mask.
        # has_news=1.0 if at least one news token is real, 0.0 if all are padding.
        # Multiplying the final output by this gate guarantees zero output when
        # no news is present, preserving parity with the baseline LoRA model.
        has_news = (~computed_mask.all(dim=1)).float()  # (B,)

        news_proj = self.news_proj(news_emb)

        # Inject positional encoding so the model can weight recent news (bar T)
        # differently from older context (bar T-29). Applied before null-token
        # substitution so masked positions still carry their position identity.
        S = news_emb.shape[1]
        positions = torch.arange(S, device=news_emb.device)
        # Clamp positions to the embedding table size (handles variable seq_len at inference)
        positions = positions.clamp(max=self.news_pos_enc.num_embeddings - 1)
        news_proj = news_proj + self.news_pos_enc(positions).unsqueeze(0)

        # For fully-masked samples, substitute null token so attention doesn't NaN
        all_masked = computed_mask.all(dim=1)
        if all_masked.any():
            news_proj = news_proj.clone()
            computed_mask = computed_mask.clone()
            news_proj[all_masked, 0:1, :] = self.null_news_token.expand(
                int(all_masked.sum().item()), -1, -1
            )
            computed_mask[all_masked, 0] = False

        query = self.market_query_proj(baseline_features).unsqueeze(1)
        attended_news, attn_weights = self.cross_attn(
            query,
            news_proj,
            news_proj,
            key_padding_mask=computed_mask,
        )
        self.last_attn_weights = attn_weights.detach()

        # FIX 3: Normalize after attention before concatenation
        attended_news = self.post_attn_norm(attended_news.squeeze(1))

        fusion_input = torch.cat([baseline_features, attended_news], dim=-1)
        residual = self.news_weight * self.residual_head(fusion_input).squeeze(-1)

        # FIX 1 (High): Zero out residual for samples with no news.
        # Gradients still flow for samples that DO have news in the same batch.
        return residual * has_news

    def debug_news_contribution(self) -> None:
        """Print diagnostics to confirm news is being used after training."""
        import math
        w = self.news_weight.item()
        print(f"news_weight (learned scale): {w:.4f}  [was 0.1 at init; growing = news is useful]")
        if self.last_attn_weights is not None:
            attn = self.last_attn_weights
            entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
            S = attn.shape[-1]
            max_entropy = math.log(S) if S > 1 else 1.0
            print(f"Attention entropy: {entropy:.3f} / max {max_entropy:.3f}  [lower = more selective]")


class ChronosCMTFPredictor:
    """Chronos LoRA + News Fusion predictor.

    DESIGN PRINCIPLES:
    1. Reuse trained ChronosLoRAPredictor backbone (no redundancy)
    2. Optionally freeze backbone to test pure fusion benefit
    3. Match training setup with baseline for fair comparison
    4. Simple architecture with no built-in bias

    Prediction formula:
        pred = baseline_pred + fusion(baseline_features, news_emb)

    When no news is present (all slots masked), fusion() returns 0 and
    CMTF is numerically identical to the LoRA baseline.

    Args:
        chronos_lora_predictor: Pre-initialized (optionally pre-trained) LoRA model
        news_dim: News embedding dimension
        tabular_dim: Optional tabular feature dimension
        hidden_dim: Fusion head hidden dimension (default 256)
        n_heads: Cross-attention heads
        dropout: Dropout rate
        freeze_backbone: If True, only train fusion head (test pure news benefit)
        huber_delta: Same as baseline
        sign_penalty_weight: Same as baseline
        device: Device for training
    """

    def __init__(
        self,
        chronos_lora_predictor,
        news_dim: int = 768,
        tabular_dim: int = 0,
        hidden_dim: int | None = None,
        fusion_dim: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        huber_delta: float = 0.02,
        sign_penalty_weight: float | None = None,
        dir_penalty_weight: float | None = None,
        seq_len: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.lora_predictor = chronos_lora_predictor
        self.backbone = getattr(chronos_lora_predictor, "backbone", None)
        if self.backbone is None:
            try:
                self.backbone = ChronosLoRAEncoderBackbone(
                    chronos_lora_predictor,
                    device=device,
                )
            except Exception:
                self.backbone = _LegacyChronosBackboneAdapter(chronos_lora_predictor, device=device)
        self.transformer = self.backbone.transformer
        self.tokenizer = self.backbone.tokenizer

        self.device = device
        self.news_dim = news_dim
        self.tabular_dim = tabular_dim
        self.market_input_dim = int(getattr(chronos_lora_predictor, "market_input_dim", 0))
        # FIX 2: Default hidden_dim increased from 128 to 256
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else (fusion_dim if fusion_dim is not None else 256))
        self.n_heads = n_heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.freeze_backbone = freeze_backbone
        self.huber_delta = huber_delta
        if sign_penalty_weight is None:
            sign_penalty_weight = dir_penalty_weight if dir_penalty_weight is not None else 0.05
        self.sign_penalty_weight = float(sign_penalty_weight)
        self.baseline_feature_dim = int(
            getattr(chronos_lora_predictor, "combined_feature_dim", self.backbone.d_model + self.tabular_dim)
        )

        self.fusion = ResidualNewsFusionHead(
            baseline_dim=self.baseline_feature_dim,
            market_dim=self.backbone.d_model,
            news_dim=news_dim,
            hidden_dim=self.hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
        ).to(device)

        self.is_fitted = False

        logger.info(
            "CMTF {} initialized | backbone_dim={} | news_dim={} | tabular_dim={} | freeze_backbone={} | hidden_dim={}",
            CMTF_VERSION, self.backbone.d_model, news_dim, tabular_dim, freeze_backbone, self.hidden_dim,
        )

    def _baseline_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.backbone.trainable_parameters())
        market_encoder = getattr(self.lora_predictor, "market_encoder", None)
        if isinstance(market_encoder, nn.Module):
            params.extend(list(market_encoder.parameters()))
        return params

    def _set_market_path_mode(self, training: bool) -> None:
        market_encoder = getattr(self.lora_predictor, "market_encoder", None)
        regression_head = getattr(self.lora_predictor, "regression_head", None)

        if self.freeze_backbone:
            self.transformer.eval()
            if isinstance(market_encoder, nn.Module):
                market_encoder.eval()
            if isinstance(regression_head, nn.Module):
                regression_head.eval()
            return

        if training:
            self.transformer.train()
            if isinstance(market_encoder, nn.Module):
                market_encoder.train()
        else:
            self.transformer.eval()
            if isinstance(market_encoder, nn.Module):
                market_encoder.eval()
        if isinstance(regression_head, nn.Module):
            regression_head.eval()

    def _extract_market_state(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        market_windows: torch.Tensor | None = None,
        market_tabular: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.lora_predictor, "extract_tokenized_features") and hasattr(self.lora_predictor, "regress_features"):
            features = self.lora_predictor.extract_tokenized_features(
                token_ids,
                attention_mask,
                market_windows=market_windows,
                market_tabular=market_tabular,
            )
            baseline_pred = self.lora_predictor.regress_features(features)
            return features, baseline_pred

        market_emb = self.backbone.encode_tokenized(token_ids, attention_mask)
        feature_parts = [market_emb]
        if market_tabular is not None:
            feature_parts.append(market_tabular)
        features = torch.cat(feature_parts, dim=1) if len(feature_parts) > 1 else market_emb
        baseline_pred = torch.zeros(features.shape[0], dtype=features.dtype, device=features.device)
        logger.warning(
            "CMTF _extract_market_state: lora_predictor missing extract_tokenized_features/"
            "regress_features — baseline_pred is all-zeros. Fusion trains in pure-prediction "
            "mode instead of residual mode. Check that a ChronosLoRAPredictor was passed."
        )
        return features, baseline_pred

    def tokenize_windows(self, close_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize close price windows using shared backbone."""
        return self.backbone.tokenize_windows(close_windows)

    def _regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Same loss as ChronosLoRAPredictor for fair comparison."""
        return sign_aware_huber_loss(
            pred,
            target,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
        )

    def fit(
        self,
        close_train: np.ndarray,
        news_train: np.ndarray,
        y_train: np.ndarray,
        close_val: np.ndarray,
        news_val: np.ndarray,
        y_val: np.ndarray,
        tabular_train: np.ndarray | None = None,
        tabular_val: np.ndarray | None = None,
        market_windows_train: np.ndarray | None = None,
        market_windows_val: np.ndarray | None = None,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        epochs: int = 20,
        lr: float = 1e-4,
        patience: int = 5,
        batch_size: int = 32,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        """Train the fusion head (and optionally fine-tune backbone)."""
        train_token_ids, train_attention_mask = self.tokenize_windows(close_train)
        val_token_ids, val_attention_mask = self.tokenize_windows(close_val)

        return self.fit_tokenized(
            train_token_ids,
            train_attention_mask,
            news_train,
            y_train,
            val_token_ids,
            val_attention_mask,
            news_val,
            y_val,
            tabular_train=tabular_train,
            tabular_val=tabular_val,
            market_windows_train=market_windows_train,
            market_windows_val=market_windows_val,
            news_mask_train=news_mask_train,
            news_mask_val=news_mask_val,
            epochs=epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
        )

    def fit_tokenized(
        self,
        token_ids_train: np.ndarray,
        attention_mask_train: np.ndarray,
        news_train: np.ndarray,
        y_train: np.ndarray,
        token_ids_val: np.ndarray,
        attention_mask_val: np.ndarray,
        news_val: np.ndarray,
        y_val: np.ndarray,
        tabular_train: np.ndarray | None = None,
        tabular_val: np.ndarray | None = None,
        market_windows_train: np.ndarray | None = None,
        market_windows_val: np.ndarray | None = None,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        epochs: int = 20,
        lr: float = 1e-4,
        patience: int = 5,
        batch_size: int = 32,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        """Train from tokenized data with exact same setup as baseline."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Rebuild fusion head for fresh training
        self.fusion = ResidualNewsFusionHead(
            baseline_dim=self.baseline_feature_dim,
            market_dim=self.backbone.d_model,
            news_dim=self.news_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
            seq_len=self.seq_len,
        ).to(self.device)

        # Convert to tensors
        train_token = torch.as_tensor(token_ids_train, dtype=torch.long)
        train_mask = torch.as_tensor(attention_mask_train, dtype=torch.long)
        train_news = torch.as_tensor(news_train, dtype=torch.float32)
        train_y = torch.as_tensor(y_train, dtype=torch.float32)
        train_tab = (
            torch.as_tensor(tabular_train, dtype=torch.float32)
            if tabular_train is not None
            else None
        )
        train_market = (
            torch.as_tensor(market_windows_train, dtype=torch.float32)
            if market_windows_train is not None
            else None
        )
        train_news_mask = (
            torch.as_tensor(news_mask_train, dtype=torch.bool)
            if news_mask_train is not None
            else None
        )

        val_token = torch.as_tensor(token_ids_val, dtype=torch.long)
        val_mask = torch.as_tensor(attention_mask_val, dtype=torch.long)
        val_news = torch.as_tensor(news_val, dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        val_tab = (
            torch.as_tensor(tabular_val, dtype=torch.float32, device=self.device)
            if tabular_val is not None
            else None
        )
        val_market = (
            torch.as_tensor(market_windows_val, dtype=torch.float32, device=self.device)
            if market_windows_val is not None
            else None
        )
        val_news_mask = (
            torch.as_tensor(news_mask_val, dtype=torch.bool, device=self.device)
            if news_mask_val is not None
            else None
        )

        # Build DataLoader
        train_tensors = [train_token, train_mask, train_news]
        if train_market is not None:
            train_tensors.append(train_market)
        if train_tab is not None:
            train_tensors.append(train_tab)
        if train_news_mask is not None:
            train_tensors.append(train_news_mask)
        train_tensors.append(train_y)
        train_ds = TensorDataset(*train_tensors)

        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_gen,
        )

        # Setup optimizer — differential learning rates:
        # fusion head uses full lr; backbone LoRA adapters use 0.1× lr so
        # pre-trained representations are updated conservatively.
        fusion_params = list(self.fusion.parameters())
        backbone_params = self._baseline_trainable_parameters()
        # params for gradient clipping (all trainable params)
        params = fusion_params + backbone_params

        if self.freeze_backbone:
            logger.info("Backbone FROZEN — only training fusion head")
            optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)
        else:
            logger.info(
                "Backbone TRAINABLE — joint training with fusion head "
                "(fusion lr={:.2e}, backbone lr={:.2e})",
                lr, lr * 0.1,
            )
            optimizer = torch.optim.AdamW(
                [
                    {"params": fusion_params, "lr": lr},
                    {"params": backbone_params, "lr": lr * 0.1},
                ],
                weight_decay=1e-5,
            )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self._set_market_path_mode(training=True)
            self.fusion.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch_idx = 0
                mb_token = batch[batch_idx]; batch_idx += 1
                mb_mask = batch[batch_idx]; batch_idx += 1
                mb_news = batch[batch_idx].to(self.device); batch_idx += 1
                mb_market = None
                if train_market is not None:
                    mb_market = batch[batch_idx].to(self.device); batch_idx += 1
                mb_tab = None
                if train_tab is not None:
                    mb_tab = batch[batch_idx].to(self.device); batch_idx += 1
                mb_news_mask = None
                if train_news_mask is not None:
                    mb_news_mask = batch[batch_idx].to(self.device); batch_idx += 1
                mb_y = batch[batch_idx].to(self.device)

                optimizer.zero_grad()

                if self.freeze_backbone:
                    with torch.no_grad():
                        baseline_features, baseline_pred = self._extract_market_state(
                            mb_token, mb_mask,
                            market_windows=mb_market, market_tabular=mb_tab,
                        )
                else:
                    baseline_features, baseline_pred = self._extract_market_state(
                        mb_token, mb_mask,
                        market_windows=mb_market, market_tabular=mb_tab,
                    )

                pred = baseline_pred + self.fusion(baseline_features, mb_news, mb_news_mask)
                loss = self._regression_loss(pred, mb_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            self._set_market_path_mode(training=False)
            self.fusion.eval()

            with torch.no_grad():
                val_features, val_baseline_pred = self._extract_market_state(
                    val_token, val_mask,
                    market_windows=val_market, market_tabular=val_tab,
                )
                val_pred = val_baseline_pred + self.fusion(val_features, val_news, val_news_mask)
                val_loss = self._regression_loss(val_pred, val_y).item()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.get_checkpoint()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                logger.debug(
                    "CMTF epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, "
                    "pred_mean={:.5f}, pred_std={:.4f}, news_weight={:.4f}",
                    epoch + 1, epochs, avg_train, val_loss,
                    float(val_pred.mean().item()), float(val_pred.std().item()),
                    float(self.fusion.news_weight.item()),
                )

            if patience_counter >= patience:
                logger.info("CMTF early stopping at epoch {}", epoch + 1)
                break

        if best_state is not None:
            self.load_checkpoint(best_state)

        self.is_fitted = True
        logger.info(
            "CMTF training done | best val loss = {:.6f} | final news_weight = {:.4f}",
            best_val_loss,
            float(self.fusion.news_weight.item()),
        )
        return history

    def predict(
        self,
        close_test: np.ndarray,
        news_test: np.ndarray,
        tabular_test: np.ndarray | None = None,
        market_windows_test: np.ndarray | None = None,
        news_mask_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict returns using market + news."""
        token_ids, attention_mask = self.tokenize_windows(close_test)
        return self.predict_tokenized(
            token_ids,
            attention_mask,
            news_test,
            tabular_test=tabular_test,
            market_windows_test=market_windows_test,
            news_mask_test=news_mask_test,
        )

    def predict_tokenized(
        self,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        news_test: np.ndarray,
        tabular_test: np.ndarray | None = None,
        market_windows_test: np.ndarray | None = None,
        news_mask_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict from tokenized data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        token_ids_t = torch.as_tensor(token_ids, dtype=torch.long)
        attention_mask_t = torch.as_tensor(attention_mask, dtype=torch.long)
        test_news = torch.as_tensor(news_test, dtype=torch.float32, device=self.device)
        test_tab = (
            torch.as_tensor(tabular_test, dtype=torch.float32, device=self.device)
            if tabular_test is not None
            else None
        )
        test_market = (
            torch.as_tensor(market_windows_test, dtype=torch.float32, device=self.device)
            if market_windows_test is not None
            else None
        )
        test_news_mask = (
            torch.as_tensor(news_mask_test, dtype=torch.bool, device=self.device)
            if news_mask_test is not None
            else None
        )

        self._set_market_path_mode(training=False)
        self.fusion.eval()

        with torch.no_grad():
            test_features, test_baseline_pred = self._extract_market_state(
                token_ids_t,
                attention_mask_t,
                market_windows=test_market,
                market_tabular=test_tab,
            )
            pred = test_baseline_pred + self.fusion(test_features, test_news, test_news_mask)
            return pred.cpu().numpy().astype(np.float32)

    def get_checkpoint(self) -> dict:
        """Save model state including version tag for compatibility checks."""
        checkpoint = {
            "version": CMTF_VERSION,
            "fusion_state": {k: v.detach().cpu().clone() for k, v in self.fusion.state_dict().items()},
            "is_fitted": self.is_fitted,
        }

        if not self.freeze_backbone:
            if hasattr(self.lora_predictor, "checkpoint_state"):
                checkpoint["backbone"] = self.lora_predictor.checkpoint_state()
            else:
                checkpoint["backbone"] = self.backbone.checkpoint_state()

        return checkpoint

    def load_checkpoint(self, ckpt: dict | None) -> None:
        """Load model state.

        FIX 2 (Medium): Uses strict=False so that old checkpoints missing
        post_attn_norm or news_weight do not crash — missing keys are logged
        as warnings and kept at their randomly initialized values rather than
        silently producing wrong results. Version mismatch is also logged.
        """
        if ckpt is None:
            return

        ckpt_version = ckpt.get("version", "unknown")
        if ckpt_version != CMTF_VERSION:
            logger.warning(
                "CMTF checkpoint version mismatch: got '{}', expected '{}'. "
                "Missing keys will be kept at init values.",
                ckpt_version, CMTF_VERSION,
            )

        missing, unexpected = self.fusion.load_state_dict(
            ckpt["fusion_state"], strict=False
        )
        if missing:
            logger.warning("CMTF load_checkpoint: missing keys in fusion state: {}", missing)
        if unexpected:
            logger.warning("CMTF load_checkpoint: unexpected keys in fusion state: {}", unexpected)

        if not self.freeze_backbone and "backbone" in ckpt:
            if hasattr(self.lora_predictor, "load_checkpoint_state"):
                self.lora_predictor.load_checkpoint_state(ckpt["backbone"])
            else:
                self.backbone.load_checkpoint_state(ckpt["backbone"])

        self.is_fitted = bool(ckpt.get("is_fitted", True))

    def get_attention_weights(self) -> np.ndarray | None:
        """Get last batch attention weights for interpretability.

        Returns:
            (B, 1, S) attention weights or None if not available.
        """
        if self.fusion.last_attn_weights is not None:
            return self.fusion.last_attn_weights.cpu().numpy()
        return None

    def debug_news_contribution(self) -> None:
        """Print diagnostics to confirm news is being used after training."""
        self.fusion.debug_news_contribution()