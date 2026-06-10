"""Fusion wrappers: Early, Late, Hybrid (CMTF) fusion strategies.

Hybrid implements Crossmodal Temporal Fusion via cross-attention over
temporally windowed news with optional two-stage training, auxiliary
baseline loss, and variance regularization.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger

from .baseline_models import sign_aware_huber_loss, _TARGET_SCALE, LSTMPredictor
from .encoder_protocol import TemporalEncoder


# ======================================================================
# Residual News Fusion Head
# ======================================================================


class ResidualNewsFusionHead(nn.Module):
    """News-conditioned residual branch on top of the baseline market prediction.

    Contract: when all news slots are masked (no news available), forward()
    returns exactly zero — so `baseline_pred + fusion(...)` equals `baseline_pred`.

    Phase 3 fixes applied:
      - Reduced capacity: market_dim=128, hidden_dim=64, n_heads=2 (was 512/256/4)
      - news_window: only the last `news_window` bars enter cross-attention
        (horizon-aware windowing: 1d→1 bar, 5d→5 bars, 20d→20 bars)
      - recency_gate_k gates the density score using actual recent bars
      - use_news_gate toggles the MSGCA per-dim gate
      - use_positional_encoding toggles positional encoding
    """

    def __init__(
        self,
        baseline_dim: int,
        market_dim: int = 128,
        news_dim: int = 768,
        hidden_dim: int = 64,
        n_heads: int = 2,
        dropout: float = 0.2,
        seq_len: int | None = None,
        huber_delta: float = 0.02,
        use_positional_encoding: bool = False,
        recency_gate_k: int = 5,
        use_news_gate: bool = True,
        news_window: int | None = None,
    ) -> None:
        super().__init__()
        self.baseline_dim = int(baseline_dim)
        self.market_dim = int(market_dim)
        self.news_dim = int(news_dim)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = seq_len
        self.huber_delta = huber_delta
        self.use_positional_encoding = bool(use_positional_encoding)
        self.recency_gate_k = int(recency_gate_k)
        self.use_news_gate = bool(use_news_gate)
        self.news_window = int(news_window) if news_window is not None else None

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

        if self.use_news_gate:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.baseline_dim, self.market_dim),
                nn.Sigmoid(),
            )

        max_seq = int(seq_len) if seq_len is not None else 30
        if self.use_positional_encoding:
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

        self.news_weight = nn.Parameter(torch.tensor(0.1))

        self.last_attn_weights: torch.Tensor | None = None
        self.last_attended_news: torch.Tensor | None = None

    def forward(
        self,
        baseline_features: torch.Tensor,
        news_emb: torch.Tensor,
        news_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            baseline_features: (B, baseline_dim) pooled market encoder embedding.
            news_emb: (B, S, news_dim) — all-zero rows = no-news padding.
            news_mask: (B, S) bool, True = ignore. Inferred from news_emb if None.

        Returns:
            (B,) residual. Exactly zero when no news in window.
        """
        # --- Horizon-aware windowing: slice to last news_window bars ---
        if self.news_window is not None and news_emb.shape[1] > self.news_window:
            news_emb = news_emb[:, -self.news_window:, :]
            if news_mask is not None:
                news_mask = news_mask[:, -self.news_window:]

        if news_mask is None:
            computed_mask = news_emb.abs().sum(-1) == 0
        else:
            computed_mask = news_mask.to(device=news_emb.device, dtype=torch.bool)

        # Recency-aware soft density gate
        S = news_emb.shape[1]
        K = min(self.recency_gate_k, S)
        recent_mask = computed_mask[:, -K:]
        news_density = (~recent_mask).float().mean(dim=1)
        has_any_news = (~computed_mask.all(dim=1)).float()

        news_proj = self.news_proj(news_emb)

        if self.use_positional_encoding:
            positions = torch.arange(S, device=news_emb.device)
            positions = positions.clamp(max=self.news_pos_enc.num_embeddings - 1)
            news_proj = news_proj + self.news_pos_enc(positions).unsqueeze(0)

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
        attended_news = self.post_attn_norm(attended_news.squeeze(1))

        if self.use_news_gate:
            gate = self.fusion_gate(baseline_features)
            attended_news = attended_news * gate

        self.last_attended_news = attended_news

        fusion_input = torch.cat([baseline_features, attended_news], dim=-1)
        raw_residual = self.residual_head(fusion_input).squeeze(-1)

        max_contribution = self.huber_delta * 0.5
        bounded_residual = torch.tanh(raw_residual) * max_contribution

        residual = self.news_weight * bounded_residual
        return residual * news_density * has_any_news


# ======================================================================
# News Branch (used by LateFusionWrapper)
# ======================================================================

class NewsBranchPredictor(nn.Module):
    """Lightweight MLP that predicts return from horizon-windowed news embeddings.

    Architecture: mean(news_seq[-news_window:], dim=1) → Linear → ReLU → Linear(1)
    """

    def __init__(self, news_dim: int = 768, hidden_dim: int = 128, device: str = "cpu",
                 news_window: int | None = None):
        super().__init__()
        self.news_dim = news_dim
        self.device = device
        self.news_window = int(news_window) if news_window is not None else None
        self.mlp = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(device)

    def forward(self, news_embs: torch.Tensor, news_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            news_embs: (B, seq_len, news_dim)
            news_mask: (B, seq_len) bool, True = no news.

        Returns: (B,) predicted log return contribution.
        """
        # Horizon-aware windowing: only use last news_window bars
        if self.news_window is not None and news_embs.shape[1] > self.news_window:
            news_embs = news_embs[:, -self.news_window:, :]
            if news_mask is not None:
                news_mask = news_mask[:, -self.news_window:]

        if news_mask is not None:
            mask_expanded = (~news_mask).float().unsqueeze(-1)  # (B, S, 1)
            masked_embs = news_embs * mask_expanded
            denom = mask_expanded.sum(dim=1).clamp_min(1.0)
            pooled = masked_embs.sum(dim=1) / denom
            has_any_news = (~news_mask).any(dim=1).float()
        else:
            pooled = news_embs.mean(dim=1)
            has_any_news = (news_embs.abs().sum(dim=(1, 2)) > 0).float()
        return self.mlp(pooled).squeeze(-1) * has_any_news


# ======================================================================
# Early Fusion Wrapper
# ======================================================================

class EarlyFusionWrapper:
    """Concatenates news embeddings to market windows before feeding to encoder.

    For sequence models (LSTM, CNN-LSTM): input becomes (N, seq_len, F + news_dim).
    The encoder is rebuilt with expanded input_dim.
    """

    def __init__(self, encoder_cls, encoder_kwargs: dict, news_dim: int = 768, device: str = "cpu"):
        self.news_dim = news_dim
        self.device = device
        expanded_kwargs = {**encoder_kwargs, "input_dim": encoder_kwargs["input_dim"] + news_dim}
        self.encoder = encoder_cls(**expanded_kwargs)
        self._original_input_dim = encoder_kwargs["input_dim"]

    @property
    def d_model(self) -> int:
        return self.encoder.d_model

    @property
    def supports_sequence(self) -> bool:
        return True

    def _concat_inputs(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        if market_windows.ndim == 2:
            market_windows = market_windows[:, :, np.newaxis]
        return np.concatenate([market_windows, news_embs], axis=-1).astype(np.float32)

    def fit(
        self,
        market_windows_train: np.ndarray,
        news_embs_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray,
        news_embs_val: np.ndarray,
        targets_val: np.ndarray,
        **kwargs,
    ) -> dict:
        X_train = self._concat_inputs(market_windows_train, news_embs_train)
        X_val = self._concat_inputs(market_windows_val, news_embs_val)
        return self.encoder.fit(X_train, targets_train, X_val, targets_val, **kwargs)

    def predict(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        X = self._concat_inputs(market_windows, news_embs)
        return self.encoder.predict_market_only(X)

    def encode(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        X = self._concat_inputs(market_windows, news_embs)
        return self.encoder.encode(X)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        seq_len = market_windows.shape[1] if market_windows.ndim == 3 else market_windows.shape[1]
        batch = market_windows.shape[0]
        zero_news = np.zeros((batch, seq_len, self.news_dim), dtype=np.float32)
        return self.predict(market_windows, zero_news)


# ======================================================================
# Late Fusion Wrapper
# ======================================================================

class LateFusionWrapper:
    """Averages market encoder prediction with a separate news branch prediction.

    pred_final = pred_market + alpha * pred_news

    Phase 3 fix: alpha is now learned (was fixed 0.3) and NewsBranchPredictor
    uses only the last `horizon` news bars (horizon-aware pooling).
    """

    def __init__(
        self,
        encoder,
        news_dim: int = 768,
        alpha: float = 0.3,
        device: str = "cpu",
        horizon: int = 1,
    ):
        self.encoder = encoder
        self.alpha = alpha
        self.device = device
        self.horizon = horizon
        # Use last `horizon` bars for news pooling (horizon-aware)
        self.news_branch = NewsBranchPredictor(
            news_dim=news_dim, device=device, news_window=horizon
        )
        self._is_news_fitted = False

    @property
    def d_model(self) -> int:
        return self.encoder.d_model

    @property
    def supports_sequence(self) -> bool:
        return self.encoder.supports_sequence

    def fit_encoder(self, market_windows_train, targets_train, market_windows_val, targets_val, **kwargs):
        return self.encoder.fit(market_windows_train, targets_train, market_windows_val, targets_val, **kwargs)

    def fit_news_branch(
        self,
        news_embs_train: np.ndarray,
        targets_train: np.ndarray,
        news_embs_val: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        market_preds_train: np.ndarray | None = None,
        market_preds_val: np.ndarray | None = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 8,
    ) -> dict:
        if market_preds_train is not None:
            residual_train = (targets_train - market_preds_train).astype(np.float32)
            residual_val = (targets_val - market_preds_val).astype(np.float32)
        else:
            residual_train = targets_train.astype(np.float32)
            residual_val = targets_val.astype(np.float32)

        N_tr = torch.as_tensor(news_embs_train, dtype=torch.float32)
        y_tr = torch.as_tensor(residual_train, dtype=torch.float32) * _TARGET_SCALE
        N_v = torch.as_tensor(news_embs_val, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(residual_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE

        tensors = [N_tr, y_tr]
        NM_tr = None
        if news_mask_train is not None:
            NM_tr = torch.as_tensor(news_mask_train, dtype=torch.bool)
            tensors.insert(1, NM_tr)
        NM_v = torch.as_tensor(news_mask_val, dtype=torch.bool, device=self.device) if news_mask_val is not None else None

        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.news_branch.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.news_branch.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if NM_tr is not None:
                    mb_n, mb_nm, mb_y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                else:
                    mb_n, mb_y = batch[0].to(self.device), batch[1].to(self.device)
                    mb_nm = None
                optimizer.zero_grad()
                pred = self.news_branch(mb_n, mb_nm)
                loss = nn.functional.huber_loss(pred, mb_y, delta=1.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            self.news_branch.eval()
            with torch.no_grad():
                val_pred = self.news_branch(N_v, NM_v)
                val_loss = nn.functional.huber_loss(val_pred, y_v, delta=1.0).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.news_branch.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.news_branch.load_state_dict(best_state)
        self._is_news_fitted = True
        return history

    def predict(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
        news_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        pred_market = self.encoder.predict_market_only(market_windows)

        self.news_branch.eval()
        N = torch.as_tensor(news_embs, dtype=torch.float32, device=self.device)
        NM = torch.as_tensor(news_mask, dtype=torch.bool, device=self.device) if news_mask is not None else None
        with torch.no_grad():
            pred_news = self.news_branch(N, NM).cpu().numpy() / _TARGET_SCALE

        return pred_market + self.alpha * pred_news

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.encoder.predict_market_only(market_windows)


# ======================================================================
# Hybrid Fusion Wrapper (CMTF)
# ======================================================================

class HybridFusionWrapper(nn.Module):
    """Crossmodal Temporal Fusion via cross-attention over news embeddings.

    Architecture: pred = market_pred + fusion_residual
      where fusion_residual is produced by ResidualNewsFusionHead via
      cross-attention between the market latent embedding and windowed news.

    Training modes:
      - two_stage (default, requires TemporalEncoder):
          Stage 1: freeze encoder, train fusion only
          Stage 2: unfreeze encoder at 0.1× LR, joint fine-tuning
      - single_stage: frozen encoder embeddings, train fusion only

    Regularization:
      - aux_loss: auxiliary huber(baseline_pred, target) to keep encoder calibrated
      - variance_reg: relu(0.01 - attn_output.var()) prevents attention collapse
    """

    def __init__(
        self,
        encoder,
        news_dim: int = 768,
        fusion_dim: int = 64,
        fusion_market_dim: int = 128,
        n_heads: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.05,
        use_positional_encoding: bool = False,
        recency_gate_k: int = 5,
        use_news_gate: bool = True,
        horizon: int = 1,
        device: str = "cpu",
        # --- CMTF flags ---
        use_two_stage: bool = True,
        use_aux_loss: bool = True,
        use_variance_reg: bool = True,
    ):
        super().__init__()
        if encoder.d_model == 0:
            raise ValueError("HybridFusionWrapper requires encoder with d_model > 0")

        self.encoder = encoder
        self.device = device
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.horizon = horizon
        self._d_model = encoder.d_model
        self.use_two_stage = use_two_stage
        self.use_aux_loss = use_aux_loss
        self.use_variance_reg = use_variance_reg
        # LSTM needs VR to prevent attention collapse (0.01); CNN-LSTM's dilated TCN
        # already creates feature diversity across bars, so 0.01 forces unneeded
        # dispersion. Use 0.001 for CNN-LSTM: enough to prevent Stage 1 entropy
        # collapse without restricting attention focus in Stage 2.
        self._vr_coeff = 0.01 if isinstance(encoder, LSTMPredictor) else 0.001

        # Check if encoder supports differentiable temporal fusion
        self._is_temporal = isinstance(encoder, TemporalEncoder) and getattr(
            encoder, "supports_temporal_fusion", False
        )

        # Cross-modal fusion head
        self.fusion = ResidualNewsFusionHead(
            baseline_dim=self._d_model,
            market_dim=fusion_market_dim,
            news_dim=news_dim,
            hidden_dim=fusion_dim,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
            huber_delta=huber_delta,
            use_positional_encoding=use_positional_encoding,
            recency_gate_k=recency_gate_k,
            use_news_gate=use_news_gate,
            news_window=horizon,
        )
        self.fusion.to(device)
        self.is_fitted = False

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def supports_sequence(self) -> bool:
        return self.encoder.supports_sequence

    def _fusion_parameters(self) -> list[nn.Parameter]:
        """Return all trainable fusion parameters (excludes encoder)."""
        return list(self.fusion.parameters())

    def _forward_fusion(
        self,
        emb: torch.Tensor,
        news: torch.Tensor,
        market_pred: torch.Tensor,
        news_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute fused prediction: market_pred + cross-attention news residual."""
        residual = self.fusion(emb, news, news_mask)
        return market_pred + residual

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        market_pred: torch.Tensor,
        news: torch.Tensor,
        news_mask: torch.Tensor | None,
        emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute total loss with optional aux loss and variance reg."""
        loss = sign_aware_huber_loss(
            pred, target,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
        )

        # Auxiliary baseline loss: keep encoder calibrated during fine-tuning
        if self.use_aux_loss:
            aux = nn.functional.huber_loss(market_pred, target, delta=self.huber_delta)
            loss = loss + aux

        # Variance regularization: prevent cross-attention centroid collapse.
        # Coefficient reduced 0.1 → 0.01: CNN-LSTM's TCN already creates per-bar
        # feature diversity; 0.1 over-penalised attention focus and caused −21% DA%
        # at 20D by preventing the model from attending the single informative event.
        # _vr_coeff=0.01 for LSTM, 0.0 for CNN-LSTM (set in __init__ by encoder type).
        if self.use_variance_reg and self._vr_coeff > 0 and self.fusion.last_attended_news is not None:
            attn_var = self.fusion.last_attended_news.var(dim=0).mean()
            loss = loss + self._vr_coeff * torch.relu(torch.tensor(0.01, device=loss.device) - attn_var)

        return loss

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
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12,
        freeze_encoder: bool = True,
    ) -> dict:
        del freeze_encoder  # controlled by use_two_stage now

        # --- Route to two-stage or single-stage training ---
        if self.use_two_stage and self._is_temporal:
            return self._fit_two_stage(
                market_windows_train, news_embs_train, targets_train,
                market_windows_val, news_embs_val, targets_val,
                news_mask_train, news_mask_val,
                batch_size=batch_size, lr=lr, patience=patience,
            )
        else:
            return self._fit_single_stage(
                market_windows_train, news_embs_train, targets_train,
                market_windows_val, news_embs_val, targets_val,
                news_mask_train, news_mask_val,
                epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
            )

    # ------------------------------------------------------------------
    # Single-stage training (frozen encoder, pre-computed embeddings)
    # ------------------------------------------------------------------

    def _fit_single_stage(
        self,
        mw_train, ne_train, y_train,
        mw_val, ne_val, y_val,
        nm_train, nm_val,
        *,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12,
    ) -> dict:
        """Original single-stage training with frozen encoder embeddings."""
        emb_train = self.encoder.encode(mw_train).astype(np.float32)
        emb_val = self.encoder.encode(mw_val).astype(np.float32)
        mpred_train = self.encoder.predict_market_only(mw_train).astype(np.float32) * _TARGET_SCALE
        mpred_val = self.encoder.predict_market_only(mw_val).astype(np.float32) * _TARGET_SCALE

        E_tr = torch.as_tensor(emb_train, dtype=torch.float32)
        N_tr = torch.as_tensor(ne_train, dtype=torch.float32)
        M_tr = torch.as_tensor(mpred_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32) * _TARGET_SCALE
        NM_tr = torch.as_tensor(nm_train, dtype=torch.bool) if nm_train is not None else None

        E_v = torch.as_tensor(emb_val, dtype=torch.float32, device=self.device)
        N_v = torch.as_tensor(ne_val, dtype=torch.float32, device=self.device)
        M_v = torch.as_tensor(mpred_val, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE
        NM_v = torch.as_tensor(nm_val, dtype=torch.bool, device=self.device) if nm_val is not None else None

        tensors = [E_tr, N_tr, M_tr]
        if NM_tr is not None:
            tensors.append(NM_tr)
        tensors.append(y_tr)
        has_mask = NM_tr is not None

        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        fusion_params = self._fusion_parameters()
        optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)

        return self._train_loop(
            loader, optimizer, fusion_params, has_mask,
            E_v, N_v, M_v, y_v, NM_v,
            epochs=epochs, patience=patience, stage_name="SingleStage",
        )

    # ------------------------------------------------------------------
    # Two-stage training (differentiable encoder path)
    # ------------------------------------------------------------------

    def _fit_two_stage(
        self,
        mw_train, ne_train, y_train,
        mw_val, ne_val, y_val,
        nm_train, nm_val,
        *,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12,
    ) -> dict:
        """Two-stage CMTF training with encoder fine-tuning."""
        # Convert market windows to tensors for differentiable path
        MW_tr = torch.as_tensor(mw_train, dtype=torch.float32)
        N_tr = torch.as_tensor(ne_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32) * _TARGET_SCALE
        NM_tr = torch.as_tensor(nm_train, dtype=torch.bool) if nm_train is not None else None

        MW_v = torch.as_tensor(mw_val, dtype=torch.float32, device=self.device)
        N_v = torch.as_tensor(ne_val, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE
        NM_v = torch.as_tensor(nm_val, dtype=torch.bool, device=self.device) if nm_val is not None else None

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Save encoder checkpoint before fine-tuning
        encoder_checkpoint = copy.deepcopy(self.encoder.state_dict())

        # ---- Stage 1: Freeze encoder, train fusion only ----
        # Scale epochs by horizon: too many frozen-encoder epochs on a weak signal
        # (e.g. 20 epochs at 5D) create a degenerate basin that Stage 2 cannot escape.
        stage1_epochs = max(5, min(20, self.horizon * 2))  # 5/10/20 for 1D/5D/20D
        logger.info("CMTF Stage 1: training fusion (encoder frozen, {} epochs)", stage1_epochs)
        for p in self.encoder.encoder_parameters():
            p.requires_grad_(False)
        # For LSTM only: freeze news_weight during Stage 1 to prevent it decaying
        # toward zero (the fusion head minimises loss by zeroing news when the
        # encoder already explains the target, leaving Stage 2 unable to recover).
        # CNN-LSTM does NOT get this freeze — its convolutional encoder is more
        # robust and the freeze+VR=0 interaction caused attention entropy collapse.
        if isinstance(self.encoder, LSTMPredictor):
            self.fusion.news_weight.requires_grad_(False)

        fusion_params = self._fusion_parameters()
        optimizer_s1 = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)

        # Build dataset with raw market windows (for differentiable encode)
        tensors = [MW_tr, N_tr]
        if NM_tr is not None:
            tensors.append(NM_tr)
        tensors.append(y_tr)
        has_mask = NM_tr is not None

        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        h1 = self._train_loop_temporal(
            loader, optimizer_s1, fusion_params, has_mask,
            MW_v, N_v, y_v, NM_v,
            epochs=stage1_epochs, patience=8, stage_name="Stage1",
        )
        history["train_loss"].extend(h1["train_loss"])
        history["val_loss"].extend(h1["val_loss"])

        # ---- Stage 2: Unfreeze encoder with differential LR ----
        logger.info("CMTF Stage 2: joint fine-tuning (encoder 0.1× LR)")
        for p in self.encoder.encoder_parameters():
            p.requires_grad_(True)
        # Unfreeze news_weight (LSTM only) and reset to init value so Stage 2
        # starts with a live fusion signal instead of a collapsed one.
        if isinstance(self.encoder, LSTMPredictor):
            self.fusion.news_weight.requires_grad_(True)
            with torch.no_grad():
                self.fusion.news_weight.fill_(0.1)

        encoder_params = list(self.encoder.encoder_parameters())
        all_params = fusion_params + encoder_params
        optimizer_s2 = torch.optim.AdamW([
            {"params": fusion_params, "lr": lr},
            {"params": encoder_params, "lr": lr * 0.1},
        ], weight_decay=1e-5)

        h2 = self._train_loop_temporal(
            loader, optimizer_s2, all_params, has_mask,
            MW_v, N_v, y_v, NM_v,
            epochs=40, patience=patience, stage_name="Stage2",
            save_encoder=True,
        )
        history["train_loss"].extend(h2["train_loss"])
        history["val_loss"].extend(h2["val_loss"])

        self.is_fitted = True
        return history

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        clip_params: list,
        has_mask: bool,
        E_v, N_v, M_v, y_v, NM_v,
        *,
        epochs: int,
        patience: int,
        stage_name: str = "",
    ) -> dict:
        """Single-stage loop over pre-computed embeddings."""
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self._set_fusion_train(True)
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                idx = 0
                mb_e = batch[idx].to(self.device); idx += 1
                mb_n = batch[idx].to(self.device); idx += 1
                mb_market = batch[idx].to(self.device); idx += 1
                mb_nm = batch[idx].to(self.device) if has_mask else None
                if has_mask:
                    idx += 1
                mb_y = batch[idx].to(self.device)

                optimizer.zero_grad()
                pred = self._forward_fusion(mb_e, mb_n, mb_market, mb_nm)
                loss = self._compute_loss(pred, mb_y, mb_market, mb_n, mb_nm, emb=mb_e)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            # Validation
            self._set_fusion_train(False)
            with torch.no_grad():
                val_pred = self._forward_fusion(E_v, N_v, M_v, NM_v)
                val_loss = sign_aware_huber_loss(
                    val_pred, y_v,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                ).item()
            history["val_loss"].append(val_loss)

            best_state, patience_counter, should_stop = self._check_early_stop(
                val_loss, best_val_loss, best_state, patience_counter, patience,
                epoch, stage_name,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if should_stop:
                break

        self._restore_best_state(best_state)
        self.is_fitted = True
        return history

    def _train_loop_temporal(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        clip_params: list,
        has_mask: bool,
        MW_v, N_v, y_v, NM_v,
        *,
        epochs: int,
        patience: int,
        stage_name: str = "",
        save_encoder: bool = False,
    ) -> dict:
        """Training loop using differentiable encoder path (TemporalEncoder)."""
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self._set_fusion_train(True)
            self.encoder.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                idx = 0
                mb_mw = batch[idx].to(self.device); idx += 1
                mb_n = batch[idx].to(self.device); idx += 1
                mb_nm = batch[idx].to(self.device) if has_mask else None
                if has_mask:
                    idx += 1
                mb_y = batch[idx].to(self.device)

                optimizer.zero_grad()

                # Differentiable encode + predict
                mb_e = self.encoder.encode_pooled_torch(mb_mw)
                mb_market = self.encoder.predict_market_only_torch(mb_mw)

                pred = self._forward_fusion(mb_e, mb_n, mb_market, mb_nm)
                loss = self._compute_loss(pred, mb_y, mb_market, mb_n, mb_nm, emb=mb_e)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            # Validation
            self._set_fusion_train(False)
            self.encoder.eval()
            with torch.no_grad():
                v_e = self.encoder.encode_pooled_torch(MW_v)
                v_m = self.encoder.predict_market_only_torch(MW_v)
                val_pred = self._forward_fusion(v_e, N_v, v_m, NM_v)
                val_loss = sign_aware_huber_loss(
                    val_pred, y_v,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                ).item()
            history["val_loss"].append(val_loss)

            best_state, patience_counter, should_stop = self._check_early_stop(
                val_loss, best_val_loss, best_state, patience_counter, patience,
                epoch, stage_name, save_encoder=save_encoder,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if should_stop:
                break

        self._restore_best_state(best_state, restore_encoder=save_encoder)
        return history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_fusion_train(self, mode: bool) -> None:
        self.fusion.train(mode)

    def _check_early_stop(
        self, val_loss, best_val_loss, best_state, patience_counter, patience,
        epoch, stage_name, save_encoder=False,
    ):
        if val_loss < best_val_loss:
            state = {
                "fusion": {k: v.detach().cpu().clone() for k, v in self.fusion.state_dict().items()},
            }
            if save_encoder:
                state["encoder"] = {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}
            return state, 0, False
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("HybridFusion {} early stopping at epoch {}", stage_name, epoch + 1)
                return best_state, patience_counter, True
            return best_state, patience_counter, False

    def _restore_best_state(self, best_state, restore_encoder=False):
        if best_state is None:
            return
        self.fusion.load_state_dict(best_state["fusion"])
        if restore_encoder and "encoder" in best_state:
            self.encoder.load_state_dict(best_state["encoder"])

    def predict(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
        news_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        emb = self.encoder.encode(market_windows).astype(np.float32)
        market_pred = self.encoder.predict_market_only(market_windows).astype(np.float32) * _TARGET_SCALE
        E = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        M = torch.as_tensor(market_pred, dtype=torch.float32, device=self.device)
        N = torch.as_tensor(news_embs, dtype=torch.float32, device=self.device)
        NM = torch.as_tensor(news_mask, dtype=torch.bool, device=self.device) if news_mask is not None else None

        self._set_fusion_train(False)
        with torch.no_grad():
            pred = self._forward_fusion(E, N, M, NM)
        return (pred.cpu().numpy() / _TARGET_SCALE).astype(np.float32)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.encoder.predict_market_only(market_windows)
