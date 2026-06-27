from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .training_utils import compute_huber_delta, train_with_early_stopping


# =====================================================================
# LOSSES
# =====================================================================

def sign_aware_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    huber_delta: float = 0.02,
    sign_penalty_weight: float = 0.005,
    direction_epsilon: float = 0.5,
    weights: torch.Tensor | None = None,
    margin: float = 0.0,
    enable_direction_loss: bool = True,
) -> torch.Tensor:
    loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none")
    loss_huber = loss_fn(pred, target)

    if weights is not None:
        norm_weights = weights / weights.mean().clamp_min(1e-8)
        loss_huber = loss_huber * norm_weights

    loss = loss_huber.mean()

    if not enable_direction_loss:
        return loss

    active_direction = (torch.abs(target) > direction_epsilon).float()
    direction_term = torch.relu(margin - pred * torch.sign(target))
    loss_wrong_dir = direction_term * active_direction

    if weights is not None:
        loss_wrong_dir = loss_wrong_dir * norm_weights

    if active_direction.sum() > 0:
        loss = loss + sign_penalty_weight * (
            loss_wrong_dir.sum() / active_direction.sum().clamp_min(1.0)
        )

    return loss


# =====================================================================
# HELPERS
# =====================================================================

def _ensure_market_sequence_tensor(
    market_windows: torch.Tensor,
    expected_input_dim: int,
) -> torch.Tensor:
    if market_windows.ndim == 2:
        market_windows = market_windows.unsqueeze(-1)
    if market_windows.ndim != 3:
        raise ValueError(
            "market_windows must have shape (batch, seq_len) or (batch, seq_len, input_dim)"
        )
    if market_windows.shape[-1] != expected_input_dim:
        raise ValueError(
            f"Expected input_dim={expected_input_dim}, got last_dim={market_windows.shape[-1]}"
        )
    return market_windows


def _as_float32_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("Expected at least a 2D array")
    return array


def extract_market_summary_features(market_windows: np.ndarray) -> np.ndarray:
    """
    Shared engineered summary features for fair tabular comparisons.
    Shape in:
        (N, seq_len) or (N, seq_len, n_features)
    Shape out:
        (N, 8 * n_features)
    """
    X = _as_float32_array(market_windows)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3:
        raise ValueError("market_windows must have shape (N, seq_len) or (N, seq_len, n_features)")

    last_step = X[:, -1, :]
    window_mean = X.mean(axis=1)
    window_std = X.std(axis=1)
    window_min = X.min(axis=1)
    window_max = X.max(axis=1)
    trend = X[:, -1, :] - X[:, 0, :]
    recent_len = min(X.shape[1], 5)
    recent_mean = X[:, -recent_len:, :].mean(axis=1)
    recent_std = X[:, -recent_len:, :].std(axis=1)

    features = np.concatenate(
        [
            last_step,
            window_mean,
            window_std,
            window_min,
            window_max,
            trend,
            recent_mean,
            recent_std,
        ],
        axis=1,
    )
    return features.astype(np.float32)


# =====================================================================
# BASE CLASS FOR TORCH MARKET MODELS
# =====================================================================

class BaseTorchMarketPredictor(nn.Module):
    """Common training/inference logic for single-input torch market models."""

    def __init__(self, target_scale: float = 100.0, device: str = "cpu"):
        super().__init__()
        self.target_scale = float(target_scale)
        self.device = device
        self.huber_delta = 1.0
        self.sign_penalty_weight = 0.05

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray,
        targets_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
        warmup_epochs: int = 0,
        model_name: str = "TorchModel",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        X_train = torch.tensor(market_windows_train, dtype=torch.float32)
        y_train = torch.tensor(targets_train, dtype=torch.float32) * self.target_scale
        X_val = torch.tensor(market_windows_val, dtype=torch.float32)
        y_val = torch.tensor(targets_val, dtype=torch.float32) * self.target_scale

        self.huber_delta = compute_huber_delta(y_train.numpy())
        logger.debug("{} huber_delta={:.4f}", model_name, self.huber_delta)

        def _loss_fn(pred, target, epoch=0):
            enable_direction_loss = epoch >= warmup_epochs
            return sign_aware_huber_loss(
                pred,
                target,
                huber_delta=self.huber_delta,
                sign_penalty_weight=self.sign_penalty_weight,
                direction_epsilon=0.5,
                enable_direction_loss=enable_direction_loss,
            )

        return train_with_early_stopping(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            _loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            warmup_epochs=warmup_epochs,
            model_name=model_name,
        )

    def predict(self, market_windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32)
        n_samples = len(X)
        preds = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size].to(self.device)
                pred_batch = self.forward(X_batch)
                preds[i : i + batch_size] = pred_batch.cpu().numpy()

        return preds / self.target_scale

    def get_embeddings(self, market_windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32)
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size].to(self.device)
                emb_batch = self._encode_tensor(X_batch)
                embeddings_list.append(emb_batch.cpu().numpy())

        return np.concatenate(embeddings_list, axis=0)

    @property
    def supports_sequence(self) -> bool:
        return True

    @property
    def supports_temporal_fusion(self) -> bool:
        return True

    def encode_sequence_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self._encode_sequence_tensor(market_windows)

    def encode_pooled_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self._encode_tensor(market_windows)

    def predict_market_only_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self.forward(market_windows)

    def encoder_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.parameters())

    def encode(self, market_windows: np.ndarray) -> np.ndarray:
        return self.get_embeddings(market_windows)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)


# =====================================================================
# BASE CLASS FOR HYBRID TORCH MODELS (SEQUENCE + TABULAR)
# =====================================================================

class BaseTorchHybridPredictor(nn.Module):
    """Common training/inference logic for dual-input torch models."""

    def __init__(self, target_scale: float = 100.0, device: str = "cpu"):
        super().__init__()
        self.target_scale = float(target_scale)
        self.device = device
        self.huber_delta = 1.0
        self.sign_penalty_weight = 0.05

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _tabular_from_windows(self, market_windows: np.ndarray) -> np.ndarray:
        return extract_market_summary_features(market_windows)

    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray,
        targets_val: np.ndarray,
        market_tabular_train: np.ndarray | None = None,
        market_tabular_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
        warmup_epochs: int = 0,
        model_name: str = "HybridTorchModel",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict:
        if market_tabular_train is None:
            market_tabular_train = self._tabular_from_windows(market_windows_train)
        if market_tabular_val is None:
            market_tabular_val = self._tabular_from_windows(market_windows_val)

        X_train_seq = torch.tensor(market_windows_train, dtype=torch.float32)
        X_val_seq = torch.tensor(market_windows_val, dtype=torch.float32)

        X_train_tab = torch.tensor(market_tabular_train, dtype=torch.float32)
        X_val_tab = torch.tensor(market_tabular_val, dtype=torch.float32)

        y_train = torch.tensor(targets_train, dtype=torch.float32) * self.target_scale
        y_val = torch.tensor(targets_val, dtype=torch.float32) * self.target_scale

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.huber_delta = compute_huber_delta(y_train.numpy())
        logger.debug("{} huber_delta={:.4f}", model_name, self.huber_delta)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_train = len(X_train_seq)

        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                xb_seq = X_train_seq[idx].to(self.device)
                xb_tab = X_train_tab[idx].to(self.device)
                yb = y_train[idx].to(self.device)

                optimizer.zero_grad()
                pred = self.forward(xb_seq, xb_tab)
                loss = sign_aware_huber_loss(
                    pred,
                    yb,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                    direction_epsilon=0.5,
                    enable_direction_loss=(epoch >= warmup_epochs),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            self.eval()
            with torch.no_grad():
                pred_val = self.forward(
                    X_val_seq.to(self.device),
                    X_val_tab.to(self.device),
                )
                val_loss = sign_aware_huber_loss(
                    pred_val,
                    y_val.to(self.device),
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                    direction_epsilon=0.5,
                    enable_direction_loss=(epoch >= warmup_epochs),
                ).item()

            val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def predict(
        self,
        market_windows: np.ndarray,
        market_tabular: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        self.eval()

        if market_tabular is None:
            market_tabular = self._tabular_from_windows(market_windows)

        X_seq = torch.tensor(market_windows, dtype=torch.float32)
        X_tab = torch.tensor(market_tabular, dtype=torch.float32)

        preds = np.zeros(len(X_seq), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                xb_seq = X_seq[i : i + batch_size].to(self.device)
                xb_tab = X_tab[i : i + batch_size].to(self.device)
                pred_batch = self.forward(xb_seq, xb_tab)
                preds[i : i + batch_size] = pred_batch.cpu().numpy()

        return preds / self.target_scale

    @property
    def supports_sequence(self) -> bool:
        return True

    @property
    def supports_temporal_fusion(self) -> bool:
        return True

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)

# =====================================================================
# LSTM PREDICTOR
# =====================================================================

class LSTMPredictor(BaseTorchMarketPredictor):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        target_scale: float = 100.0,
        device: str = "cpu",
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.d_model = num_layers * hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.to(self.device)

    def _encode_market_tensors(self, market_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        seq_output, (hidden_state, _) = self.lstm(x)
        hidden_all = hidden_state.permute(1, 0, 2)
        pooled = hidden_all.reshape(hidden_all.size(0), -1)
        return seq_output, pooled

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
        kwargs["model_name"] = kwargs.get("model_name", "LSTM")
        learning_rate = kwargs.get("learning_rate", 1e-3)

        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
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


# =====================================================================
# CNN-LSTM PREDICTOR
# =====================================================================

class _CausalDilatedBlock(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self._causal_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            num_filters, num_filters, kernel_size=kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(
            num_filters, num_filters, kernel_size=kernel_size, dilation=dilation, padding=0
        )
        self.norm2 = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.pad(x, (self._causal_pad, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = F.pad(out, (self._causal_pad, 0))
        out = self.conv2(out)
        out = self.norm2(out)
        return self.dropout(self.relu(out + residual))


class CNNLSTMPredictor(BaseTorchMarketPredictor):
    def __init__(
        self,
        input_dim: int = 1,
        num_filters: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        dilations: tuple[int, ...] = (1, 2, 4),
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.1,
        target_scale: float = 100.0,
        device: str = "cpu",
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.d_model = num_layers * hidden_dim

        dropout = max(dropout, 0.15)
        self.input_proj = nn.Linear(input_dim, num_filters)

        self.tcn_blocks = nn.ModuleList(
            [_CausalDilatedBlock(num_filters, kernel_size, d, dropout) for d in dilations]
        )

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.to(self.device)

    def _encode_market_tensors(self, market_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)

        for block in self.tcn_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        seq_output, (hidden_state, _) = self.lstm(x)
        hidden_all = hidden_state.permute(1, 0, 2)
        encoding = hidden_all.reshape(hidden_all.size(0), -1)
        return seq_output, encoding

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self._encode_market_tensors(market_windows)
        return seq_output

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        _, encoding = self._encode_market_tensors(market_windows)
        return encoding

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        _, encoding = self._encode_market_tensors(market_windows)
        out = self.dropout(encoding)
        out = self.fc(out)
        return out.squeeze(-1)

    @property
    def sequence_d_model(self) -> int:
        return self.hidden_dim

    def fit(self, *args, **kwargs) -> dict:
        kwargs["model_name"] = kwargs.get("model_name", "CNN-LSTM")
        learning_rate = kwargs.get("learning_rate", 1e-3)
    
        if "optimizer" not in kwargs or kwargs["optimizer"] is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
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


# =====================================================================
# HYBRID LSTM (RAW SEQUENCE + ENGINEERED TABULAR)
# =====================================================================

class LSTMHybridPredictor(BaseTorchHybridPredictor):
    def __init__(
        self,
        input_dim: int = 1,
        tabular_dim: int | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.005,
        target_scale: float = 100.0,
        device: str = "cpu",
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.seq_dim = num_layers * hidden_dim
        self.tabular_dim = tabular_dim

        if self.tabular_dim is None:
            raise ValueError("tabular_dim must be provided for LSTMHybridPredictor")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

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
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state.permute(1, 0, 2).reshape(x.size(0), -1)

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        seq_emb = self._encode_sequence_branch(market_windows)
        tab_emb = self.tabular_mlp(market_tabular)
        pred = self.head(torch.cat([seq_emb, tab_emb], dim=-1))
        return pred.squeeze(-1)


# =====================================================================
# HYBRID CNN-LSTM (RAW SEQUENCE + ENGINEERED TABULAR)
# =====================================================================

class CNNLSTMHybridPredictor(BaseTorchHybridPredictor):
    def __init__(
        self,
        input_dim: int = 1,
        tabular_dim: int | None = None,
        num_filters: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        dilations: tuple[int, ...] = (1, 2, 4),
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.1,
        target_scale: float = 100.0,
        device: str = "cpu",
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.tabular_dim = tabular_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.seq_dim = num_layers * hidden_dim

        if self.tabular_dim is None:
            raise ValueError("tabular_dim must be provided for CNNLSTMHybridPredictor")

        dropout = max(dropout, 0.15)

        self.input_proj = nn.Linear(input_dim, num_filters)
        self.tcn_blocks = nn.ModuleList(
            [_CausalDilatedBlock(num_filters, kernel_size, d, dropout) for d in dilations]
        )

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

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
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state.permute(1, 0, 2).reshape(x.size(0), -1)

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        seq_emb = self._encode_sequence_branch(market_windows)
        tab_emb = self.tabular_mlp(market_tabular)
        pred = self.head(torch.cat([seq_emb, tab_emb], dim=-1))
        return pred.squeeze(-1)


# =====================================================================
# RANDOM FOREST WRAPPER
# =====================================================================

class RandomForestRegressor_Wrapper:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        max_features: str | int | float | None = "sqrt",
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _extract_features(market_windows: np.ndarray) -> np.ndarray:
        return extract_market_summary_features(market_windows)

    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray | None = None,
        targets_val: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        X_train = self._extract_features(market_windows_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, targets_train)
        self.is_fitted = True

    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = self._extract_features(market_windows)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).astype(np.float32)

    @property
    def d_model(self) -> int:
        return 0

    @property
    def supports_sequence(self) -> bool:
        return False

    def encode(self, market_windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RandomForest has no latent space")

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)


# =====================================================================
# LINEAR SUMMARY BASELINE
# =====================================================================

class LinearSummaryRegressor_Wrapper:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray | None = None,
        targets_val: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        X_train = extract_market_summary_features(market_windows_train)
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, targets_train)
        self.is_fitted = True

    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = extract_market_summary_features(market_windows)
        X = self.scaler.transform(X)
        return self.model.predict(X).astype(np.float32)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)
    
    @property
    def d_model(self) -> int:
        return 0
    
    @property
    def supports_sequence(self) -> bool:
        return False
    
    def encode(self, market_windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LinearSummaryRegressor has no latent space")


# =====================================================================
# MLP SUMMARY BASELINE
# =====================================================================

class MLPSummaryPredictor:
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        target_scale: float = 100.0,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = 0.01,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.target_scale = target_scale
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.device = device
        self.scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.is_fitted = False

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        ).to(self.device)

    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
        market_windows_val: np.ndarray,
        targets_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
        **kwargs,
    ) -> dict:
        X_train = extract_market_summary_features(market_windows_train)
        X_val = extract_market_summary_features(market_windows_val)

        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)

        y_train = (np.asarray(targets_train, dtype=np.float32) * self.target_scale)
        y_val = (np.asarray(targets_val, dtype=np.float32) * self.target_scale)

        self.model = self._build_model(X_train.shape[1])

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_train = len(X_train_t)

        huber_delta = compute_huber_delta(y_train_t.detach().cpu().numpy())

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                pred = self.model(X_train_t[idx]).squeeze(-1)
                loss = sign_aware_huber_loss(
                    pred,
                    y_train_t[idx],
                    huber_delta=huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                    direction_epsilon=0.5,
                    enable_direction_loss=True,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(X_val_t).squeeze(-1)
                val_loss = sign_aware_huber_loss(
                    pred_val,
                    y_val_t,
                    huber_delta=huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                    direction_epsilon=0.5,
                    enable_direction_loss=True,
                ).item()

            val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def predict(self, market_windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = extract_market_summary_features(market_windows)
        X = self.scaler.transform(X).astype(np.float32)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = np.zeros(len(X_t), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                pred = self.model(X_t[i : i + batch_size]).squeeze(-1)
                preds[i : i + batch_size] = pred.cpu().numpy()

        return preds / self.target_scale

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)


# =====================================================================
# FINE-TUNED CHRONOS PREDICTOR
# =====================================================================

class FineTunedChronosPredictor:
    def __init__(
        self,
        chronos_predictor,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        tabular_dim: int = 0,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = 0.05,
        target_scale: float = 100.0,
        device: str = "cpu",
    ):
        self.chronos = chronos_predictor
        self.device = device
        self.d_model = chronos_predictor.d_model
        self.tabular_dim = int(tabular_dim)
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.target_scale = float(target_scale)

        self._embedding_cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self._embedding_cache_max_entries = 32

        self.adapter = nn.Sequential(
            nn.Linear(self.d_model + self.tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)

        self.is_fitted = False

    def _prune_embedding_cache(self) -> None:
        while len(self._embedding_cache) > self._embedding_cache_max_entries:
            self._embedding_cache.popitem(last=False)

    def _embedding_cache_key(self, close_windows: np.ndarray) -> tuple[int, str]:
        close_array = np.ascontiguousarray(close_windows, dtype=np.float32)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(close_array.shape).encode())
        digest.update(close_array.dtype.str.encode())
        digest.update(close_array.view(np.uint8))
        return id(self.chronos), digest.hexdigest()

    def _get_cached_embeddings(self, close_windows: np.ndarray, *, cache_label: str) -> np.ndarray:
        cache_key = self._embedding_cache_key(close_windows)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            self._embedding_cache.move_to_end(cache_key)
            return cached.copy()

        logger.info("Computing Chronos embeddings [{}]", cache_label)
        embeddings = np.asarray(self.chronos.get_embeddings(close_windows), dtype=np.float32)
        self._embedding_cache[cache_key] = embeddings.copy()
        self._prune_embedding_cache()
        return embeddings

    def _combine_features(
        self,
        embeddings: np.ndarray,
        market_tabular: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.tabular_dim == 0:
            return embeddings.astype(np.float32)
        if market_tabular is None:
            raise ValueError("market_tabular is required when tabular_dim > 0")
        tabular = np.asarray(market_tabular, dtype=np.float32)
        if tabular.ndim != 2 or tabular.shape[1] != self.tabular_dim:
            raise ValueError(
                f"Expected market_tabular shape (N, {self.tabular_dim}), got {tabular.shape}"
            )
        if embeddings.shape[0] != tabular.shape[0]:
            raise ValueError(
                f"Batch size mismatch: embeddings={embeddings.shape[0]}, tabular={tabular.shape[0]}"
            )
        return np.concatenate([embeddings, tabular], axis=1).astype(np.float32)

    def fit(
        self,
        close_windows_train: np.ndarray,
        targets_train: np.ndarray,
        close_windows_val: np.ndarray,
        targets_val: np.ndarray,
        market_tabular_train: np.ndarray | None = None,
        market_tabular_val: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        patience: int = 5,
        **kwargs,
    ) -> dict:
        import torch.optim as optim

        emb_train = self._combine_features(
            self._get_cached_embeddings(close_windows_train, cache_label="train"),
            market_tabular_train,
        )
        emb_val = self._combine_features(
            self._get_cached_embeddings(close_windows_val, cache_label="val"),
            market_tabular_val,
        )

        X_train = torch.tensor(emb_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device) * self.target_scale
        X_val = torch.tensor(emb_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val, dtype=torch.float32, device=self.device) * self.target_scale


        optimizer = optim.AdamW(self.adapter.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_train = len(X_train)

        for epoch in range(epochs):
            self.adapter.train()
            indices = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                batch_indices = indices[i : i + batch_size]
                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                optimizer.zero_grad()
                pred = self.adapter(X_batch).squeeze(-1)
                loss = sign_aware_huber_loss(
                    pred,
                    y_batch,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            self.adapter.eval()
            with torch.no_grad():
                pred_val = self.adapter(X_val).squeeze(-1)
                val_loss = sign_aware_huber_loss(
                    pred_val,
                    y_val,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                ).item()
                val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.adapter.load_state_dict(best_state)

        self.is_fitted = True
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def predict(
        self,
        close_windows: np.ndarray,
        market_tabular: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        embeddings = self._combine_features(
            self._get_cached_embeddings(close_windows, cache_label="predict"),
            market_tabular,
        )

        X = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        self.adapter.eval()
        with torch.no_grad():
            preds = self.adapter(X).squeeze(-1).cpu().numpy()

        return (preds / self.target_scale).astype(np.float32)