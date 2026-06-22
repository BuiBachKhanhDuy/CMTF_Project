"""Fusion wrappers: Early, Late, Hybrid (CMTF) fusion strategies.

Refactored for:
1. Shared standardized news projection dimension across wrappers.
2. Safer early fusion via projected news instead of raw 768-dim concat.
3. Direction-aware loss only on strong moves and only after warmup (Hybrid).
4. Validation objective aligned with training objective (Hybrid).
5. Easier maintenance through shared helpers and consistent interfaces.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from sklearn.model_selection import KFold, TimeSeriesSplit

from .baseline_models import sign_aware_huber_loss


# ======================================================================
# Global Standard
# ======================================================================

STANDARD_NEWS_DIM = 32


# ======================================================================
# Utility & Helper Functions
# ======================================================================

def generate_oof_market_predictions(
    base_encoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    use_time_series_split: bool = True,
    **fit_kwargs
) -> np.ndarray:
    """Generate OOF market predictions to reduce residual-branch leakage."""
    print(f"[Utility] Generating OOF predictions using {n_splits} splits...")
    oof_preds = np.zeros_like(y_train, dtype=np.float32)

    cv = (
        TimeSeriesSplit(n_splits=n_splits)
        if use_time_series_split
        else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    initial_state = copy.deepcopy(base_encoder)

    for fold, (tr_idx, v_idx) in enumerate(cv.split(X_train)):
        print(f"  -> Training Fold {fold + 1}/{n_splits}...")

        fold_model = copy.deepcopy(initial_state)

        X_fold_tr, y_fold_tr = X_train[tr_idx], y_train[tr_idx]
        X_fold_val, y_fold_val = X_train[v_idx], y_train[v_idx]

        fold_model.fit(X_fold_tr, y_fold_tr, X_fold_val, y_fold_val, **fit_kwargs)
        oof_preds[v_idx] = fold_model.predict_market_only(X_fold_val)

    print("[Utility] OOF predictions generation completed.")
    return oof_preds


def _as_bool_mask(news_mask: np.ndarray | torch.Tensor | None, news_embs) -> np.ndarray | torch.Tensor:
    if news_mask is not None:
        return news_mask
    if isinstance(news_embs, np.ndarray):
        return np.isclose(news_embs, 0.0).all(axis=-1)
    return torch.isclose(news_embs, torch.zeros(1, device=news_embs.device, dtype=news_embs.dtype)).all(dim=-1)


# ======================================================================
# Shared News Projection
# ======================================================================

class SharedNewsProjector:
    """Deterministic shared projection from raw news embeddings to a standard dimension.

    Uses a fixed random projection matrix for consistency across wrappers and runs.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = STANDARD_NEWS_DIM,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((self.input_dim, self.output_dim), dtype=np.float32)
        proj /= np.sqrt(max(self.input_dim, 1))
        self.proj = proj.astype(np.float32)

    def transform(self, news_embs: np.ndarray) -> np.ndarray:
        arr = np.asarray(news_embs, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected news_embs shape (N, S, {self.input_dim}), got {arr.shape}"
            )
        out = np.einsum("nsd,dk->nsk", arr, self.proj, optimize=True)
        return out.astype(np.float32)


# ======================================================================
# Core Modules
# ======================================================================

class ResidualNewsFusionHead(nn.Module):
    """News-conditioned residual branch on top of the baseline market prediction."""

    def __init__(
        self,
        baseline_dim: int,
        market_dim: int = 128,
        news_dim: int = STANDARD_NEWS_DIM,
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
        self.baseline_dim = baseline_dim
        self.market_dim = market_dim
        self.news_dim = news_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.huber_delta = huber_delta
        self.use_positional_encoding = use_positional_encoding
        self.recency_gate_k = recency_gate_k
        self.use_news_gate = use_news_gate
        self.news_window = news_window

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

        max_seq = seq_len if seq_len is not None else 30
        if self.use_positional_encoding:
            self.news_pos_enc = nn.Embedding(max_seq, self.market_dim)
            nn.init.normal_(self.news_pos_enc.weight, std=0.02)

        self.null_news_token = nn.Parameter(torch.zeros(1, 1, self.market_dim))

        self.residual_head = nn.Sequential(
            nn.Linear(self.baseline_dim + self.market_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.hidden_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.news_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.last_attn_weights: torch.Tensor | None = None
        self.last_attended_news: torch.Tensor | None = None

    def forward(
        self,
        baseline_features: torch.Tensor,
        news_emb: torch.Tensor,
        news_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.news_window is not None and news_emb.shape[1] > self.news_window:
            news_emb = news_emb[:, -self.news_window:, :]
            if news_mask is not None:
                news_mask = news_mask[:, -self.news_window:]

        if news_mask is None:
            computed_mask = news_emb.abs().sum(-1) == 0
        else:
            computed_mask = news_mask.to(device=news_emb.device, dtype=torch.bool)

        S = news_emb.shape[1]
        K = min(self.recency_gate_k, S)
        recent_mask = computed_mask[:, -K:]
        news_density = (~recent_mask).float().mean(dim=1)

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

        self.last_attended_news = attended_news.detach()

        fusion_input = torch.cat([baseline_features, attended_news], dim=-1)
        raw_residual = self.residual_head(fusion_input).squeeze(-1)

        bounded_residual = self.huber_delta * torch.nn.functional.softsign(raw_residual)
        return self.news_weight * bounded_residual * news_density


class NewsBranchPredictor(nn.Module):
    """Lightweight MLP that predicts return from horizon-windowed news embeddings."""

    def __init__(self, news_dim: int = STANDARD_NEWS_DIM, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.news_dim = news_dim
        self.mlp = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        self.learned_alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, news_embs: torch.Tensor, news_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = ~news_mask
        has_any_news = valid_mask.any(dim=1).float()
        mask_expanded = valid_mask.float().unsqueeze(-1)
        masked_embs = news_embs * mask_expanded
        denom = mask_expanded.sum(dim=1).clamp_min(1.0)
        pooled = masked_embs.sum(dim=1) / denom
        out = self.mlp(pooled).squeeze(-1)
        return out * has_any_news * self.learned_alpha


# ======================================================================
# Base Wrapper for Shared News Projection
# ======================================================================

class _BaseFusionWrapper:
    def __init__(
        self,
        raw_news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        news_projector: SharedNewsProjector | None = None,
    ) -> None:
        self.raw_news_dim = int(raw_news_dim)
        self.projected_news_dim = int(projected_news_dim)
        self.news_projector = news_projector or SharedNewsProjector(
            input_dim=self.raw_news_dim,
            output_dim=self.projected_news_dim,
            seed=42,
        )

    def _project_news(self, news_embs: np.ndarray) -> np.ndarray:
        return self.news_projector.transform(news_embs)


# ======================================================================
# Fusion Wrappers
# ======================================================================

class EarlyFusionWrapper(_BaseFusionWrapper):
    """Concatenates projected news embeddings to market windows before feeding to the encoder."""

    def __init__(
        self,
        encoder_cls,
        encoder_kwargs: dict,
        news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        news_projector: SharedNewsProjector | None = None,
    ):
        super().__init__(
            raw_news_dim=news_dim,
            projected_news_dim=projected_news_dim,
            news_projector=news_projector,
        )
        self._original_input_dim = encoder_kwargs.get("input_dim", 0)
        expanded_kwargs = {
            **encoder_kwargs,
            "input_dim": self._original_input_dim + self.projected_news_dim,
        }
        self.encoder = encoder_cls(**expanded_kwargs)

    @property
    def d_model(self) -> int:
        return getattr(self.encoder, "d_model", None)

    @property
    def supports_sequence(self) -> bool:
        return getattr(self.encoder, "supports_sequence", True)

    def _concat_inputs(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        if market_windows.ndim == 2:
            market_windows = np.expand_dims(market_windows, axis=-1)
        news_proj = self._project_news(news_embs)
        return np.concatenate([market_windows, news_proj], axis=-1).astype(np.float32)

    def fit(
        self,
        market_train: np.ndarray,
        news_train: np.ndarray,
        targets_train: np.ndarray,
        market_val: np.ndarray,
        news_val: np.ndarray,
        targets_val: np.ndarray,
        **kwargs
    ) -> dict:
        X_train = self._concat_inputs(market_train, news_train)
        X_val = self._concat_inputs(market_val, news_val)
        return self.encoder.fit(X_train, targets_train, X_val, targets_val, **kwargs)

    def predict(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        X = self._concat_inputs(market_windows, news_embs)
        return self.encoder.predict(X)

    def encode(self, market_windows: np.ndarray, news_embs: np.ndarray) -> np.ndarray:
        X = self._concat_inputs(market_windows, news_embs)
        return self.encoder.encode(X)

    def predict_without_news(self, market_windows: np.ndarray) -> np.ndarray:
        batch_size, seq_len = market_windows.shape[0], market_windows.shape[1]
        zero_news = np.zeros((batch_size, seq_len, self.raw_news_dim), dtype=np.float32)
        return self.predict(market_windows, zero_news)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict_without_news(market_windows)


class LateFusionWrapper(_BaseFusionWrapper):
    """Adds a separate projected-news branch on top of market encoder prediction."""

    def __init__(
        self,
        encoder,
        news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        alpha: float = 0.3,
        device: str = "cpu",
        horizon: int = 1,
        target_scale: float = 100.0,
        freeze_encoder: bool = False,
        news_projector: SharedNewsProjector | None = None,
        **kwargs
    ):
        super().__init__(
            raw_news_dim=news_dim,
            projected_news_dim=projected_news_dim,
            news_projector=news_projector,
        )
        self.encoder = encoder
        self.alpha = alpha
        self.device = device
        self.horizon = horizon
        self.target_scale = target_scale
        self.freeze_encoder = freeze_encoder
        self.news_branch = NewsBranchPredictor(
            news_dim=self.projected_news_dim,
            hidden_dim=128,
            dropout_rate=0.2,
        )
        self.news_branch.to(self.device)
        self._is_news_fitted = False

    @property
    def d_model(self) -> int:
        return getattr(self.encoder, "d_model", None)

    @property
    def supports_sequence(self) -> bool:
        return getattr(self.encoder, "supports_sequence", True)

    def fit(
        self,
        market_train: np.ndarray,
        news_train: np.ndarray,
        targets_train: np.ndarray,
        market_val: np.ndarray,
        news_val: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None = None,
        news_mask_val: np.ndarray | None = None,
        n_splits: int = 5,
        use_time_series_split: bool = True,
        epochs_news: int = 30,
        batch_size_news: int = 32,
        lr_news: float = 1e-3,
        patience_news: int = 8,
        **encoder_fit_kwargs
    ) -> dict:
        news_train_proj = self._project_news(news_train)
        news_val_proj = self._project_news(news_val)

        print("--- PHASE 1: Generating OOF Market Predictions ---")
        oof_preds_train = generate_oof_market_predictions(
            base_encoder=self.encoder,
            X_train=market_train,
            y_train=targets_train,
            n_splits=n_splits,
            use_time_series_split=use_time_series_split,
            **encoder_fit_kwargs
        )

        print("\n--- PHASE 2: Fitting Main Market Encoder ---")
        self.encoder.fit(market_train, targets_train, market_val, targets_val, **encoder_fit_kwargs)
        market_preds_val = self.encoder.predict_market_only(market_val)

        print("\n--- PHASE 3: Fitting News Branch on Residuals ---")
        history = self.fit_news_branch(
            news_train_proj,
            targets_train,
            news_val_proj,
            targets_val,
            news_mask_train,
            news_mask_val,
            oof_preds_train,
            market_preds_val,
            epochs_news,
            batch_size_news,
            lr_news,
            patience_news,
        )
        print("--- Late Fusion Training Complete ---")
        return history

    def fit_news_branch(
        self,
        news_embs_train,
        targets_train,
        news_embs_val,
        targets_val,
        news_mask_train,
        news_mask_val,
        market_preds_train,
        market_preds_val,
        epochs,
        batch_size,
        lr,
        patience
    ) -> dict:
        residual_train = (targets_train - market_preds_train).astype(np.float32)
        residual_val = (targets_val - market_preds_val).astype(np.float32)

        train_mask = _as_bool_mask(news_mask_train, news_embs_train)
        val_mask = _as_bool_mask(news_mask_val, news_embs_val)

        def _build_loader(N_np, y_np, M_np):
            tensors = [
                torch.as_tensor(N_np, dtype=torch.float32),
                torch.as_tensor(M_np, dtype=torch.bool),
                torch.as_tensor(y_np, dtype=torch.float32) * self.target_scale,
            ]
            return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)

        loader_tr = _build_loader(news_embs_train, residual_train, train_mask)
        loader_v = _build_loader(news_embs_val, residual_val, val_mask)

        optimizer = torch.optim.Adam(self.news_branch.parameters(), lr=lr)
        best_val_loss, best_state, patience_counter = float("inf"), None, 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.news_branch.train()
            epoch_loss, n_batches = 0.0, 0

            for batch in loader_tr:
                mb_n, mb_nm, mb_y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

                optimizer.zero_grad()
                pred = self.news_branch(mb_n, mb_nm)
                loss = nn.functional.huber_loss(pred, mb_y, delta=1.0)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            self.news_branch.eval()
            val_epoch_loss, v_batches = 0.0, 0

            with torch.no_grad():
                for batch in loader_v:
                    mb_n_v, mb_nm_v, mb_y_v = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                    val_pred = self.news_branch(mb_n_v, mb_nm_v)
                    v_loss = nn.functional.huber_loss(val_pred, mb_y_v, delta=1.0)
                    val_epoch_loss += v_loss.item()
                    v_batches += 1

            val_loss = val_epoch_loss / max(v_batches, 1)
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
        news_mask: np.ndarray | None = None
    ) -> np.ndarray:
        pred_market = self.encoder.predict_market_only(market_windows)

        news_proj = self._project_news(news_embs)
        mask = _as_bool_mask(news_mask, news_proj)

        self.news_branch.eval()
        N = torch.as_tensor(news_proj, dtype=torch.float32, device=self.device)
        NM = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            pred_news = self.news_branch(N, NM).cpu().numpy() / self.target_scale

        return pred_market + pred_news

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.encoder.predict_market_only(market_windows)

class HybridFusionWrapper(_BaseFusionWrapper, nn.Module):
    """Joint multimodal crossmodal temporal fusion over projected news embeddings.

    Predicts the final target directly from fused market/news representations.
    """

    def __init__(
        self,
        encoder,
        news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        fusion_dim: int = 64,
        fusion_market_dim: int = 128,
        n_heads: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        use_positional_encoding: bool = False,
        recency_gate_k: int = 5,
        use_news_gate: bool = True,
        horizon: int = 1,
        device: str = "cpu",
        use_two_stage: bool = True,
        use_aux_loss: bool = True,
        use_variance_reg: bool = True,
        target_scale: float = 100.0,
        freeze_encoder: bool = False,
        news_projector: SharedNewsProjector | None = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        _BaseFusionWrapper.__init__(
            self,
            raw_news_dim=news_dim,
            projected_news_dim=projected_news_dim,
            news_projector=news_projector,
        )

        if getattr(encoder, "d_model", 0) == 0:
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
        self.target_scale = target_scale
        self.freeze_encoder = freeze_encoder

        self.direction_epsilon = 0.5
        self.direction_warmup_epochs = 5

        self._has_custom_news_weight = getattr(self.encoder, "has_custom_news_weight", False)
        self._vr_coeff = 0.01 if self._has_custom_news_weight else 0.001
        self._is_temporal = getattr(self.encoder, "supports_temporal_fusion", False)

        self.news_encoder = nn.Sequential(
            nn.Linear(self.projected_news_dim, fusion_market_dim),
            nn.LayerNorm(fusion_market_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_market_dim, fusion_market_dim),
            nn.LayerNorm(fusion_market_dim),
            nn.ReLU(),
        )

        self.market_proj = nn.Sequential(
            nn.Linear(self._d_model, fusion_market_dim),
            nn.LayerNorm(fusion_market_dim),
            nn.ReLU(),
        )

        if use_news_gate:
            self.news_gate = nn.Sequential(
                nn.Linear(self._d_model, fusion_market_dim),
                nn.Sigmoid(),
            )
        else:
            self.news_gate = None

        if use_positional_encoding:
            max_seq = seq_len if seq_len is not None else 30
            self.news_pos_enc = nn.Embedding(max_seq, fusion_market_dim)
            nn.init.normal_(self.news_pos_enc.weight, std=0.02)
        else:
            self.news_pos_enc = None

        self.null_news_token = nn.Parameter(torch.zeros(1, 1, fusion_market_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_market_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(fusion_market_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_market_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, max(fusion_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fusion_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.prediction_head[-1].weight)
        nn.init.zeros_(self.prediction_head[-1].bias)

        self.is_fitted = False
        self.last_attn_weights: torch.Tensor | None = None
        self.last_attended_news: torch.Tensor | None = None

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def supports_sequence(self) -> bool:
        return getattr(self.encoder, "supports_sequence", True)

    def _fusion_parameters(self) -> list[nn.Parameter]:
        params = list(self.news_encoder.parameters())
        params += list(self.market_proj.parameters())
        params += list(self.cross_attn.parameters())
        params += list(self.post_attn_norm.parameters())
        params += list(self.prediction_head.parameters())
        if self.news_gate is not None:
            params += list(self.news_gate.parameters())
        if self.news_pos_enc is not None:
            params += list(self.news_pos_enc.parameters())
        params += [self.null_news_token]
        return params

    def _encode_news(
        self,
        news: torch.Tensor,
        news_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if news.ndim != 3:
            raise ValueError(f"Expected news shape (B, S, D), got {news.shape}")

        if news_mask is None:
            valid_mask = news.abs().sum(-1) > 0
        else:
            valid_mask = ~news_mask.to(dtype=torch.bool, device=news.device)

        B, S, D = news.shape
        news_flat = news.reshape(B * S, D)
        news_proj = self.news_encoder(news_flat).reshape(B, S, -1)

        if self.news_pos_enc is not None:
            pos = torch.arange(S, device=news.device).clamp(max=self.news_pos_enc.num_embeddings - 1)
            news_proj = news_proj + self.news_pos_enc(pos).unsqueeze(0)

        mask_3d = valid_mask.unsqueeze(-1)
        news_proj = news_proj * mask_3d.float()

        denom = mask_3d.float().sum(dim=1).clamp_min(1.0)
        pooled = news_proj.sum(dim=1) / denom

        all_masked = ~valid_mask.any(dim=1)
        if all_masked.any():
            pooled = pooled.clone()
            pooled[all_masked] = self.null_news_token.squeeze(0).squeeze(0)

        return pooled, valid_mask

    def _forward_fusion(
        self,
        emb: torch.Tensor,
        news: torch.Tensor,
        market_pred: torch.Tensor | None,
        news_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        market_latent = self.market_proj(emb)
        news_latent, _ = self._encode_news(news, news_mask)

        q = market_latent.unsqueeze(1)
        k = news_latent.unsqueeze(1)
        v = news_latent.unsqueeze(1)

        attn_out, attn_weights = self.cross_attn(q, k, v)
        attn_out = self.post_attn_norm(attn_out.squeeze(1))

        if self.news_gate is not None:
            gate = self.news_gate(emb)
            attn_out = attn_out * gate

        self.last_attn_weights = attn_weights.detach() if attn_weights is not None else None
        self.last_attended_news = attn_out.detach()

        fused = torch.cat([market_latent, attn_out], dim=-1)
        pred = self.prediction_head(fused).squeeze(-1)
        return pred

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        market_pred: torch.Tensor,
        news: torch.Tensor,
        news_mask: torch.Tensor | None,
        emb: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        enable_direction_loss = epoch >= self.direction_warmup_epochs

        loss = sign_aware_huber_loss(
            pred,
            target,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
            direction_epsilon=self.direction_epsilon,
            enable_direction_loss=enable_direction_loss,
        )

        if self.use_aux_loss and market_pred is not None:
            aux = nn.functional.huber_loss(market_pred, target, delta=self.huber_delta)
            loss = loss + aux

        if self.use_variance_reg and self._vr_coeff > 0 and self.last_attended_news is not None:
            attn_var = self.last_attended_news.var(dim=0).mean()
            loss = loss + self._vr_coeff * torch.relu(
                torch.tensor(0.01, device=loss.device) - attn_var
            )

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
        **kwargs
    ) -> dict:
        if self.use_two_stage and self._is_temporal:
            return self._fit_two_stage(
                market_windows_train,
                news_embs_train,
                targets_train,
                market_windows_val,
                news_embs_val,
                targets_val,
                news_mask_train,
                news_mask_val,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
            )
        return self._fit_single_stage(
            market_windows_train,
            news_embs_train,
            targets_train,
            market_windows_val,
            news_embs_val,
            targets_val,
            news_mask_train,
            news_mask_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )

    def _fit_single_stage(
        self,
        mw_train,
        ne_train,
        y_train,
        mw_val,
        ne_val,
        y_val,
        nm_train,
        nm_val,
        *,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12
    ) -> dict:
        mpred_train_oof = generate_oof_market_predictions(self.encoder, mw_train, y_train, n_splits=5)
        mpred_train = mpred_train_oof.astype(np.float32) * self.target_scale

        mpred_val = self.encoder.predict_market_only(mw_val).astype(np.float32) * self.target_scale
        emb_train = self.encoder.encode(mw_train).astype(np.float32)
        emb_val = self.encoder.encode(mw_val).astype(np.float32)

        train_mask = _as_bool_mask(nm_train, ne_train)
        val_mask = _as_bool_mask(nm_val, ne_val)

        def _to_tensor(x, dtype, device="cpu"):
            return torch.as_tensor(x, dtype=dtype, device=device) if x is not None else None

        tensors = [
            _to_tensor(emb_train, torch.float32),
            _to_tensor(ne_train, torch.float32),
            _to_tensor(mpred_train, torch.float32),
            _to_tensor(train_mask, torch.bool),
            _to_tensor(y_train, torch.float32) * self.target_scale,
        ]
        loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)

        E_v = _to_tensor(emb_val, torch.float32, self.device)
        N_v = _to_tensor(ne_val, torch.float32, self.device)
        M_v = _to_tensor(mpred_val, torch.float32, self.device)
        y_v = _to_tensor(y_val, torch.float32, self.device) * self.target_scale
        NM_v = _to_tensor(val_mask, torch.bool, self.device)

        fusion_params = self._fusion_parameters()
        optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)

        return self._train_loop(
            loader,
            optimizer,
            fusion_params,
            E_v,
            N_v,
            M_v,
            y_v,
            NM_v,
            epochs=epochs,
            patience=patience,
            stage_name="SingleStage",
        )

    def _fit_two_stage(
        self,
        mw_train,
        ne_train,
        y_train,
        mw_val,
        ne_val,
        y_val,
        nm_train,
        nm_val,
        *,
        batch_size: int = 32,
        lr: float = 5e-4,
        patience: int = 12
    ) -> dict:
        train_mask = _as_bool_mask(nm_train, ne_train)
        val_mask = _as_bool_mask(nm_val, ne_val)

        def _to_tensor(x, dtype, device="cpu"):
            return torch.as_tensor(x, dtype=dtype, device=device) if x is not None else None

        tensors = [
            _to_tensor(mw_train, torch.float32),
            _to_tensor(ne_train, torch.float32),
            _to_tensor(train_mask, torch.bool),
            _to_tensor(y_train, torch.float32) * self.target_scale,
        ]
        loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)

        MW_v = _to_tensor(mw_val, torch.float32, self.device)
        N_v = _to_tensor(ne_val, torch.float32, self.device)
        y_v = _to_tensor(y_val, torch.float32, self.device) * self.target_scale
        NM_v = _to_tensor(val_mask, torch.bool, self.device)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        stage1_epochs = max(5, min(20, self.horizon * 2))
        logger.info(f"CMTF Stage 1: training fusion (encoder frozen, {stage1_epochs} epochs)")

        for p in self.encoder.encoder_parameters():
            p.requires_grad_(False)

        fusion_params = self._fusion_parameters()
        optimizer_s1 = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=1e-5)

        h1 = self._train_loop_temporal(
            loader,
            optimizer_s1,
            fusion_params,
            MW_v,
            N_v,
            y_v,
            NM_v,
            epochs=stage1_epochs,
            patience=8,
            stage_name="Stage1",
        )
        history["train_loss"].extend(h1["train_loss"])
        history["val_loss"].extend(h1["val_loss"])

        logger.info("CMTF Stage 2: joint fine-tuning (encoder 0.1× LR)")
        for p in self.encoder.encoder_parameters():
            p.requires_grad_(True)

        encoder_params = list(self.encoder.encoder_parameters())
        all_params = fusion_params + encoder_params
        optimizer_s2 = torch.optim.AdamW(
            [
                {"params": fusion_params, "lr": lr},
                {"params": encoder_params, "lr": lr * 0.1},
            ],
            weight_decay=1e-5,
        )

        h2 = self._train_loop_temporal(
            loader,
            optimizer_s2,
            all_params,
            MW_v,
            N_v,
            y_v,
            NM_v,
            epochs=40,
            patience=patience,
            stage_name="Stage2",
            save_encoder=True,
        )
        history["train_loss"].extend(h2["train_loss"])
        history["val_loss"].extend(h2["val_loss"])

        self.is_fitted = True
        return history

    def _train_loop(
        self,
        loader,
        optimizer,
        clip_params,
        E_v,
        N_v,
        M_v,
        y_v,
        NM_v,
        *,
        epochs,
        patience,
        stage_name=""
    ):
        best_val_loss, best_state, patience_counter = float("inf"), None, 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self._set_fusion_train(True)
            epoch_loss, n_batches = 0.0, 0

            for batch in loader:
                mb_e = batch[0].to(self.device)
                mb_n = batch[1].to(self.device)
                mb_market = batch[2].to(self.device)
                mb_nm = batch[3].to(self.device)
                mb_y = batch[4].to(self.device)

                optimizer.zero_grad()
                pred = self._forward_fusion(mb_e, mb_n, mb_market, mb_nm)
                loss = self._compute_loss(
                    pred, mb_y, mb_market, mb_n, mb_nm, emb=mb_e, epoch=epoch
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            self._set_fusion_train(False)
            with torch.no_grad():
                val_pred = self._forward_fusion(E_v, N_v, M_v, NM_v)
                val_loss = self._compute_loss(
                    val_pred, y_v, M_v, N_v, NM_v, emb=E_v, epoch=epoch
                ).item()
            history["val_loss"].append(val_loss)

            best_state, patience_counter, should_stop = self._check_early_stop(
                val_loss, best_val_loss, best_state, patience_counter, patience, epoch, stage_name
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
        loader,
        optimizer,
        clip_params,
        MW_v,
        N_v,
        y_v,
        NM_v,
        *,
        epochs,
        patience,
        stage_name="",
        save_encoder=False
    ):
        best_val_loss, best_state, patience_counter = float("inf"), None, 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self._set_fusion_train(True)
            self.encoder.train()
            epoch_loss, n_batches = 0.0, 0

            for batch in loader:
                mb_mw = batch[0].to(self.device)
                mb_n = batch[1].to(self.device)
                mb_nm = batch[2].to(self.device)
                mb_y = batch[3].to(self.device)

                optimizer.zero_grad()
                mb_e = self.encoder.encode_pooled_torch(mb_mw)
                mb_market = self.encoder.predict_market_only_torch(mb_mw)

                pred = self._forward_fusion(mb_e, mb_n, mb_market, mb_nm)
                loss = self._compute_loss(
                    pred, mb_y, mb_market, mb_n, mb_nm, emb=mb_e, epoch=epoch
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            self._set_fusion_train(False)
            self.encoder.eval()
            with torch.no_grad():
                v_e = self.encoder.encode_pooled_torch(MW_v)
                v_m = self.encoder.predict_market_only_torch(MW_v)
                val_pred = self._forward_fusion(v_e, N_v, v_m, NM_v)
                val_loss = self._compute_loss(
                    val_pred, y_v, v_m, N_v, NM_v, emb=v_e, epoch=epoch
                ).item()
            history["val_loss"].append(val_loss)

            best_state, patience_counter, should_stop = self._check_early_stop(
                val_loss, best_val_loss, best_state, patience_counter, patience, epoch, stage_name, save_encoder=save_encoder
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if should_stop:
                break

        self._restore_best_state(best_state, restore_encoder=save_encoder)
        return history

    def _set_fusion_train(self, mode: bool) -> None:
        self.fusion.train(mode)

    def _check_early_stop(self, val_loss, best_val_loss, best_state, patience_counter, patience, epoch, stage_name, save_encoder=False):
        if val_loss < best_val_loss:
            state = {"fusion": {k: v.detach().cpu().clone() for k, v in self.fusion.state_dict().items()}}
            if save_encoder:
                state["encoder"] = {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}
            return state, 0, False
        patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"HybridFusion {stage_name} early stopping at epoch {epoch + 1}")
            return best_state, patience_counter, True
        return best_state, patience_counter, False

    def _restore_best_state(self, best_state, restore_encoder=False):
        if best_state is None:
            return
        self.fusion.load_state_dict(best_state["fusion"])
        if restore_encoder and "encoder" in best_state:
            self.encoder.load_state_dict(best_state["encoder"])

    def predict(self, market_windows: np.ndarray, news_embs: np.ndarray, news_mask: np.ndarray | None = None) -> np.ndarray:
        news_proj = self._project_news(news_embs)
        mask = _as_bool_mask(news_mask, news_proj)

        emb = self.encoder.encode(market_windows).astype(np.float32)
        market_pred = self.encoder.predict_market_only(market_windows).astype(np.float32) * self.target_scale

        E = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        M = torch.as_tensor(market_pred, dtype=torch.float32, device=self.device)
        N = torch.as_tensor(news_proj, dtype=torch.float32, device=self.device)
        NM = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        self._set_fusion_train(False)
        with torch.no_grad():
            pred = self._forward_fusion(E, N, M, NM)
        return (pred.cpu().numpy() / self.target_scale).astype(np.float32)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.encoder.predict_market_only(market_windows)