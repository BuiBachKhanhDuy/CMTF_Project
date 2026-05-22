"""Cross-Modal Temporal Fusion with CNN+LSTM backbone.

Architecture:
    market_windows (B, seq_len, input_dim)
      → CausalDilatedTCN blocks (Bai et al. 2018)
      → LSTM → temporal attention (Shi et al. 2022)
      → market_emb (B, hidden_dim)

    pred = baseline_pred + ResidualNewsFusionHead(market_emb, news_emb)

When all news slots are masked, the fusion head returns 0 and the model
is numerically identical to a standalone CNN-LSTM predictor.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from .baseline_models import sign_aware_huber_loss, _CausalDilatedBlock, _TARGET_SCALE

CNNLSTM_CMTF_VERSION = "v4"


class ResidualNewsFusionHead(nn.Module):
    """News-conditioned residual branch on top of the baseline market prediction.

    Contract: when all news slots are masked (no news available), the forward()
    output is exactly zero — so the caller's `baseline_pred + fusion(...)` equals
    `baseline_pred` unchanged. This preserves zero-news parity with the LoRA baseline.

    Gating mechanism (MSGCA, Zong & Zhou 2024, arXiv:2406.06594):
        Instead of a binary has_news gate (which is always 1.0 with seq_len=30),
        uses a learned per-dimension sigmoid gate conditioned on the primary
        modality (market features) to suppress noisy/constant cross-attention
        output: H_stable = H_unstable ⊙ σ(H_primary · W_b + b').

    Recency-aware density gate (Fix 4):
        Only counts news in the last K=5 bars. With ~30% bar coverage,
        P(no news in last 5 bars) ≈ 0.17 → meaningful suppression vs binary gate.

    Residual bounding (Fix 3):
        The residual output is passed through tanh and scaled by 0.5*huber_delta
        to prevent sign-flipping of small baseline predictions.
    """
    _RECENT_BARS_K: int = 5

    def __init__(
        self,
        baseline_dim: int,
        market_dim: int = 512,
        news_dim: int = 768,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_len: int | None = None,
        huber_delta: float = 0.02,
    ) -> None:
        super().__init__()
        self.baseline_dim = int(baseline_dim)
        self.market_dim = int(market_dim)
        self.news_dim = int(news_dim)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = seq_len
        self.huber_delta = huber_delta

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
        self.post_attn_norm = nn.LayerNorm(self.market_dim)

        # Fix 1 (MSGCA Eq.14): Learned per-dimension sigmoid gate conditioned
        # on the primary modality (market features). Suppresses constant/noisy
        # cross-attention dimensions. Reference: Zong & Zhou 2024, arXiv:2406.06594.
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.baseline_dim, self.market_dim),
            nn.Sigmoid(),
        )

        # Learnable positional encoding over the news window
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
        nn.init.xavier_uniform_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # Learnable scale for news contribution — starts small so model
        # defaults to near-zero residual and must learn to use news.
        self.news_weight = nn.Parameter(torch.tensor(0.1))

        self.last_attn_weights: torch.Tensor | None = None
        # Stored for variance regularization (Fix 5) — set in forward()
        self.last_attended_news: torch.Tensor | None = None

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
            computed_mask = news_emb.abs().sum(-1) == 0  # (B, S)
        else:
            computed_mask = news_mask.to(device=news_emb.device, dtype=torch.bool)

        # --- Fix 4: Recency-aware soft density gate ---
        # Only count news in the last K bars. With ~30% bar coverage,
        # P(no news in last 5) ≈ 0.17 → meaningful suppression.
        S = news_emb.shape[1]
        K = min(self._RECENT_BARS_K, S)
        recent_mask = computed_mask[:, -K:]  # (B, K)
        news_density = (~recent_mask).float().mean(dim=1)  # (B,) in [0,1]
        # Hard zero when entire window has no news (preserves baseline parity)
        has_any_news = (~computed_mask.all(dim=1)).float()  # (B,)

        news_proj = self.news_proj(news_emb)

        # Positional encoding for recency preference
        positions = torch.arange(S, device=news_emb.device)
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

        attended_news = self.post_attn_norm(attended_news.squeeze(1))  # (B, market_dim)

        # --- Fix 1 (MSGCA Eq.14): Per-dimension sigmoid gate from market ---
        # H_stable = H_unstable ⊙ σ(H_primary · W + b)
        gate = self.fusion_gate(baseline_features)  # (B, market_dim)
        attended_news_gated = attended_news * gate

        # Store for variance regularization (Fix 5)
        self.last_attended_news = attended_news_gated

        fusion_input = torch.cat([baseline_features, attended_news_gated], dim=-1)
        raw_residual = self.residual_head(fusion_input).squeeze(-1)  # (B,)

        # --- Fix 3+7: Bound residual with tanh to prevent sign-flipping ---
        # max = 0.15 * huber_delta; for delta=0.02 → ±0.003 max residual.
        # With news_weight=0.1 and density≈0.3, effective max ≈ 0.00009.
        # This ensures news can only micro-nudge, never flip baseline signs.
        max_contribution = self.huber_delta * 0.15
        bounded_residual = torch.tanh(raw_residual) * max_contribution

        residual = self.news_weight * bounded_residual

        # Apply density gate (Fix 4) and hard zero gate (parity guarantee)
        return residual * news_density * has_any_news

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


class CNNLSTMCMTFPredictor(nn.Module):
    """Causal dilated CNN + LSTM market encoder with cross-modal news fusion.

    Replaces the heavyweight Chronos LoRA backbone with a lightweight
    causal-dilated TCN → LSTM → temporal-attention encoder trained directly
    on OHLCV + indicator windows (market_windows), fused with news embeddings
    via the same ResidualNewsFusionHead used by ChronosCMTFPredictor.

    Architecture:
        market_windows (B, seq_len, input_dim)
          → Linear input projection
          → CausalDilatedBlock × len(dilations)  [Bai et al. 2018, causal TCN]
          → LSTM(num_filters, hidden_dim, num_layers)
          → temporal attention (Shi et al. 2022)  → market_emb (B, hidden_dim)

        baseline_pred = regression_head(market_emb)
        news_residual  = ResidualNewsFusionHead(market_emb, news_emb, news_mask)
        pred           = baseline_pred + news_residual

    When all news slots are masked, news_residual == 0 and this model is
    numerically identical to a standalone CNNLSTMPredictor baseline.
    """

    def __init__(
        self,
        input_dim: int,
        news_dim: int = 768,
        hidden_dim: int = 64,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
        dilations: tuple[int, ...] = (1, 2, 4),
        fusion_dim: int = 128,
        fusion_market_dim: int = 256,
        n_heads: int = 4,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.01,
        seq_len: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fusion_market_dim = fusion_market_dim
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.device = device

        # --- Market encoder (fully trainable) ---
        self.input_proj = nn.Linear(input_dim, num_filters)
        self.tcn_blocks = nn.ModuleList([
            _CausalDilatedBlock(num_filters, kernel_size, d, dropout)
            for d in dilations
        ])
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        # Baseline prediction head (market-only path)
        self.regression_head = nn.Linear(hidden_dim, 1)

        # --- News fusion head ---
        # Use fusion_market_dim (default 256) as cross-attention dimension
        # to avoid over-compressing 768-d news embeddings into 64-d hidden space.
        self.fusion = ResidualNewsFusionHead(
            baseline_dim=hidden_dim,
            market_dim=fusion_market_dim,
            news_dim=news_dim,
            hidden_dim=fusion_dim,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
            huber_delta=huber_delta,
        )

        self.is_fitted = False
        self.to(device)

        receptive_field = sum(2 * (kernel_size - 1) * d for d in dilations)
        logger.info(
            "CNN-LSTM CMTF {} initialized | input_dim={}, filters={}, dilations={}, "
            "hidden={}, layers={}, receptive_field={}, news_dim={}, fusion_dim={}",
            CNNLSTM_CMTF_VERSION, input_dim, num_filters, list(dilations),
            hidden_dim, num_layers, receptive_field, news_dim, fusion_dim,
        )

    def _encode_market(self, x: torch.Tensor) -> torch.Tensor:
        """Encode market windows → (batch, hidden_dim) context vector."""
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)            # (batch, seq_len, num_filters)
        x = x.permute(0, 2, 1)           # (batch, num_filters, seq_len)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)           # (batch, seq_len, num_filters)
        lstm_out, _ = self.lstm(x)        # (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out)      # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        return (weights * lstm_out).sum(dim=1)  # (batch, hidden_dim)

    def forward(
        self,
        market_windows: torch.Tensor,
        news_emb: torch.Tensor,
        news_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred, market_emb, baseline_pred)."""
        market_emb = self._encode_market(market_windows)
        baseline_pred = self.regression_head(self.drop(market_emb)).squeeze(-1)
        # Fix 2: Detach encoder output before fusion — prevents fusion gradients
        # from corrupting the market encoder's learned representations.
        news_residual = self.fusion(market_emb.detach(), news_emb, news_mask)
        return baseline_pred + news_residual, market_emb, baseline_pred

    def _loss(
        self, pred: torch.Tensor, target: torch.Tensor,
        baseline_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_loss = sign_aware_huber_loss(
            pred, target,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
        )
        # Fix 6: Auxiliary baseline loss — anchors encoder so its standalone
        # predictions stay calibrated even when training jointly with fusion.
        # Without this, the encoder co-adapts with the fusion head's bias.
        if baseline_pred is not None:
            aux_loss = torch.nn.functional.huber_loss(
                baseline_pred, target, delta=self.huber_delta
            )
            base_loss = base_loss + aux_loss
        # Fix 5: Variance regularization — penalise near-constant attended_news
        # to prevent centroid collapse. If var < threshold, add penalty.
        attended = self.fusion.last_attended_news
        if attended is not None and attended.requires_grad:
            var_penalty = torch.relu(0.01 - attended.var(dim=0).mean())
            return base_loss + 0.1 * var_penalty
        return base_loss

    def fit(
        self,
        market_windows_train: np.ndarray,
        news_train: np.ndarray,
        y_train: np.ndarray,
        market_windows_val: np.ndarray,
        news_val: np.ndarray,
        y_val: np.ndarray,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        epochs: int = 80,
        lr: float = 1e-3,
        patience: int = 12,
        batch_size: int = 32,
        seed: int = 42,
        backbone_state_dict: dict | None = None,
        freeze_encoder_epochs: int = 20,
    ) -> dict[str, list[float]]:
        """Two-stage training with optional backbone warm-start.

        Stage 1 (freeze_encoder_epochs): If backbone_state_dict is provided,
            load pre-trained CNN-LSTM encoder weights and freeze them. Train
            only the fusion head so it learns to extract news signal without
            corrupting the market representation.
        Stage 2 (remaining epochs): Unfreeze all parameters and fine-tune
            jointly with a reduced learning rate for the encoder.

        This eliminates gradient interference between the market prediction
        and cross-attention query objectives during early training.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Rebuild fusion head for fresh training
        self.fusion.__init__(
            baseline_dim=self.hidden_dim,
            market_dim=self.fusion_market_dim,
            news_dim=self.fusion.news_dim,
            hidden_dim=self.fusion.hidden_dim,
            n_heads=self.fusion.cross_attn.num_heads,
            dropout=self.drop.p,
            seq_len=self.fusion.seq_len,
            huber_delta=self.huber_delta,
        )
        self.fusion.to(self.device)

        # Warm-start: load pre-trained CNN-LSTM backbone into encoder
        if backbone_state_dict is not None:
            encoder_keys = {
                k: v for k, v in backbone_state_dict.items()
                if any(k.startswith(p) for p in (
                    "input_proj.", "tcn_blocks.", "lstm.", "attn.", "regression_head.",
                    # CNNLSTMPredictor uses "fc" and "dropout" names
                    "fc.", "dropout.",
                ))
            }
            # Map CNNLSTMPredictor's naming to our naming
            mapped_keys: dict[str, torch.Tensor] = {}
            for k, v in encoder_keys.items():
                if k.startswith("fc."):
                    mapped_keys[k.replace("fc.", "regression_head.")] = v
                elif k.startswith("dropout."):
                    mapped_keys[k.replace("dropout.", "drop.")] = v
                else:
                    mapped_keys[k] = v

            # Check for shape compatibility before loading
            current_sd = self.state_dict()
            shape_ok = all(
                current_sd[k].shape == v.shape
                for k, v in mapped_keys.items()
                if k in current_sd
            )
            if shape_ok:
                self.load_state_dict(mapped_keys, strict=False)
                n_loaded = sum(1 for k in mapped_keys if k in current_sd)
                logger.info(
                    "CNN-LSTM CMTF warm-start: loaded {} encoder params from backbone",
                    n_loaded,
                )
            else:
                logger.warning(
                    "CNN-LSTM CMTF warm-start SKIPPED: backbone shape mismatch "
                    "(backbone num_filters≠model num_filters). Training from scratch."
                )
                backbone_state_dict = None  # Disable two-stage since warm-start failed

        X_tr = torch.as_tensor(market_windows_train, dtype=torch.float32)
        N_tr = torch.as_tensor(news_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32) * _TARGET_SCALE
        NM_tr = (
            torch.as_tensor(news_mask_train, dtype=torch.bool)
            if news_mask_train is not None else None
        )

        X_v = torch.as_tensor(market_windows_val, dtype=torch.float32, device=self.device)
        N_v = torch.as_tensor(news_val, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE
        NM_v = (
            torch.as_tensor(news_mask_val, dtype=torch.bool, device=self.device)
            if news_mask_val is not None else None
        )

        train_tensors: list[torch.Tensor] = [X_tr, N_tr]
        if NM_tr is not None:
            train_tensors.append(NM_tr)
        train_tensors.append(y_tr)
        has_mask = NM_tr is not None

        train_ds = TensorDataset(*train_tensors)
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

        # Identify encoder vs fusion parameters for two-stage training
        encoder_params = list(self.input_proj.parameters()) + \
            list(self.tcn_blocks.parameters()) + \
            list(self.lstm.parameters()) + \
            list(self.attn.parameters()) + \
            list(self.regression_head.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        fusion_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]

        # Stage 1: freeze encoder, train fusion head only
        use_two_stage = backbone_state_dict is not None and freeze_encoder_epochs > 0
        if use_two_stage:
            for p in encoder_params:
                p.requires_grad_(False)
            optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)
            logger.info(
                "CNN-LSTM CMTF Stage 1: training fusion head only ({} epochs, encoder frozen)",
                freeze_encoder_epochs,
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        stage_switched = False

        for epoch in range(epochs):
            # Stage transition: unfreeze encoder with lower LR
            if use_two_stage and not stage_switched and epoch >= freeze_encoder_epochs:
                for p in encoder_params:
                    p.requires_grad_(True)
                # Encoder gets 10× lower LR to preserve learned representations
                optimizer = torch.optim.AdamW([
                    {"params": encoder_params, "lr": lr * 0.1},
                    {"params": fusion_params, "lr": lr},
                ], weight_decay=1e-5)
                stage_switched = True
                patience_counter = 0  # Reset patience for stage 2
                logger.info(
                    "CNN-LSTM CMTF Stage 2: joint fine-tuning (encoder LR={:.1e})",
                    lr * 0.1,
                )

            self.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                idx = 0
                mb_x = batch[idx].to(self.device); idx += 1
                mb_n = batch[idx].to(self.device); idx += 1
                mb_nm = None
                if has_mask:
                    mb_nm = batch[idx].to(self.device); idx += 1
                mb_y = batch[idx].to(self.device)

                optimizer.zero_grad()
                pred, _, baseline_pred = self.forward(mb_x, mb_n, mb_nm)
                loss = self._loss(pred, mb_y, baseline_pred=baseline_pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            self.eval()
            with torch.no_grad():
                val_pred, _, _ = self.forward(X_v, N_v, NM_v)
                val_loss = self._loss(val_pred, y_v).item()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stage_name = "Stage 1" if (use_two_stage and not stage_switched) else "Stage 2" if use_two_stage else ""
                    logger.info("CNN-LSTM CMTF {} early stopping at epoch {}", stage_name, epoch + 1)
                    # If still in stage 1, transition to stage 2 instead of stopping
                    if use_two_stage and not stage_switched:
                        for p in encoder_params:
                            p.requires_grad_(True)
                        optimizer = torch.optim.AdamW([
                            {"params": encoder_params, "lr": lr * 0.1},
                            {"params": fusion_params, "lr": lr},
                        ], weight_decay=1e-5)
                        stage_switched = True
                        patience_counter = 0
                        logger.info(
                            "CNN-LSTM CMTF Stage 2: joint fine-tuning (encoder LR={:.1e})",
                            lr * 0.1,
                        )
                        continue
                    break

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "CNN-LSTM CMTF epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, "
                    "news_weight={:.4f}",
                    epoch + 1, epochs, avg_train, val_loss,
                    float(self.fusion.news_weight.item()),
                )

        if best_state is not None:
            self.load_state_dict(best_state)
        self.is_fitted = True
        logger.info(
            "CNN-LSTM CMTF training done | best val loss = {:.6f} | news_weight = {:.4f}",
            best_val_loss,
            float(self.fusion.news_weight.item()),
        )
        return history

    def predict(
        self,
        market_windows_test: np.ndarray,
        news_test: np.ndarray,
        news_mask_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict fused market+news return for each test window."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self.eval()
        X = torch.as_tensor(market_windows_test, dtype=torch.float32, device=self.device)
        N = torch.as_tensor(news_test, dtype=torch.float32, device=self.device)
        NM = (
            torch.as_tensor(news_mask_test, dtype=torch.bool, device=self.device)
            if news_mask_test is not None else None
        )
        with torch.no_grad():
            pred, _, _ = self.forward(X, N, NM)
        return (pred.cpu().numpy() / _TARGET_SCALE).astype(np.float32)

    def predict_with_explanation(
        self,
        market_windows_test: np.ndarray,
        news_test: np.ndarray,
        news_mask_test: np.ndarray | None = None,
    ) -> dict:
        """Single-sample inference with interpretability data.

        Returns the same keys as ChronosCMTFPredictor.predict_with_explanation
        for multiagent compatibility:
            baseline_pred, final_pred, news_residual,
            attn_weights, quality_gate, news_weight
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self.eval()
        X = torch.as_tensor(market_windows_test, dtype=torch.float32, device=self.device)
        N = torch.as_tensor(news_test, dtype=torch.float32, device=self.device)
        NM = (
            torch.as_tensor(news_mask_test, dtype=torch.bool, device=self.device)
            if news_mask_test is not None else None
        )
        with torch.no_grad():
            final_pred_t, _, baseline_pred_t = self.forward(X, N, NM)

        attn = getattr(self.fusion, "last_attn_weights", None)
        S = N.shape[1]
        if attn is not None:
            a = attn[0]
            if a.dim() == 3:
                a = a.mean(0)
            attn_weights = a[0].cpu().numpy().astype(np.float32)
        else:
            attn_weights = np.full(S, 1.0 / S, dtype=np.float32)

        news_weight = float(self.fusion.news_weight.detach().cpu().item())
        bp = float(baseline_pred_t[0].cpu().item()) / _TARGET_SCALE
        fp = float(final_pred_t[0].cpu().item()) / _TARGET_SCALE
        return {
            "baseline_pred": bp,
            "final_pred": fp,
            "news_residual": fp - bp,
            "attn_weights": attn_weights,
            "quality_gate": news_weight,
            "news_weight": news_weight,
        }

    def get_checkpoint(self) -> dict:
        """Save full model state including version tag."""
        return {
            "version": CNNLSTM_CMTF_VERSION,
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            "is_fitted": self.is_fitted,
        }

    def load_checkpoint(self, ckpt: dict | None) -> None:
        """Restore model state; logs a warning on version mismatch."""
        if ckpt is None:
            return
        ckpt_version = ckpt.get("version", "unknown")
        if ckpt_version != CNNLSTM_CMTF_VERSION:
            logger.warning(
                "CNN-LSTM CMTF checkpoint version mismatch: got '{}', expected '{}'.",
                ckpt_version, CNNLSTM_CMTF_VERSION,
            )
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            logger.warning("CNN-LSTM CMTF load_checkpoint: missing keys: {}", missing)
        if unexpected:
            logger.warning("CNN-LSTM CMTF load_checkpoint: unexpected keys: {}", unexpected)
        self.is_fitted = bool(ckpt.get("is_fitted", True))

    def get_attention_weights(self) -> np.ndarray | None:
        """Return last batch news attention weights for interpretability."""
        if self.fusion.last_attn_weights is not None:
            return self.fusion.last_attn_weights.cpu().numpy()
        return None

    def debug_news_contribution(self) -> None:
        """Print diagnostics to confirm news is contributing after training."""
        self.fusion.debug_news_contribution()