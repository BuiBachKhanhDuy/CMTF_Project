"""Baseline models for comparison: LSTM, Random Forest, Fine-tuned Chronos.

CORRECTED VERSION - Fixed major logic issues:
1. Consistent target scaling across all models
2. Fixed sign-aware Huber loss to avoid double penalty
3. Removed problematic magnitude-sign decomposition
4. Added proper validation for feature dimensions
5. Fixed scaler state management

Active trainable baselines now share a market-information contract:
they predict forward log return while consuming either close-only windows
or multivariate market tensors derived from OHLCV + technical indicators.

Models:
    - LSTMPredictor: sequence encoder over market windows
    - RandomForestRegressor_Wrapper: tree baseline over window summaries
    - FineTunedChronosPredictor: Chronos embeddings plus optional market tabular branch
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .training_utils import compute_huber_delta, train_with_early_stopping


def sign_aware_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    huber_delta: float = 0.02,
    sign_penalty_weight: float = 0.05,
    direction_epsilon: float = 1e-4,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber regression loss with an additional wrong-direction penalty.
    
    FIXED: Removed double penalty issue. Now the sign penalty is ONLY applied
    when predictions have the wrong sign, and it's scaled to avoid overwhelming
    the base regression loss.
    
    The sign penalty encourages directional accuracy without creating quadratic
    penalties that push predictions toward zero.
    """
    loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none")
    loss_huber = loss_fn(pred, target)

    # Treat tiny target returns as neutral to avoid unstable sign penalties
    # around sideways-market noise.
    active_direction = torch.abs(target) > direction_epsilon
    wrong_sign_mask = (pred * target < 0) & active_direction
    
    # Penalty based on magnitude of wrong-direction predictions
    # This is separate from Huber loss to avoid double-counting
    loss_wrong_dir = torch.abs(pred) * wrong_sign_mask.float()

    if weights is not None:
        norm_weights = weights / weights.mean().clamp_min(1e-8)
        loss_huber = loss_huber * norm_weights
        loss_wrong_dir = loss_wrong_dir * norm_weights

    return loss_huber.mean() + sign_penalty_weight * loss_wrong_dir.mean()


def _ensure_market_sequence_tensor(
    market_windows: torch.Tensor,
    expected_input_dim: int,
) -> torch.Tensor:
    """Normalize close-only and multivariate windows into LSTM-ready tensors."""
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


# Scale factor for targets during training to avoid mean-collapse.
# Log returns (~0.008) are scaled to (~0.8) so the loss has meaningful gradients.
_TARGET_SCALE = 100.0


class LSTMPredictor(nn.Module):
    """LSTM sequence encoder for market return prediction.

    Encodes multivariate market windows via a multi-layer LSTM and predicts
    forward log-returns through a projection head.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        sign_penalty_weight: float = 0.1,      # FIX 1: raised from 0.01 â†’ 0.1
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.device = device
        self.d_model = num_layers * hidden_dim  # embedding dim matches _encode_tensor output

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # FIX 2: projection head to prevent collapse to near-zero constant
        # Input matches num_layers * hidden_dim from _encode_tensor
        self.fc = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.to(device)

    def _encode_market_tensors(
        self,
        market_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal states and pooled latent encoding for market windows."""
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)

        # FIX 3: use ALL layers' hidden states, not just the last layer
        seq_output, (hidden_state, _) = self.lstm(x)
        # hidden_state: (num_layers, batch, hidden_dim)
        # concatenate across layers â†’ (batch, num_layers * hidden_dim)
        hidden_all = hidden_state.permute(1, 0, 2)          # (batch, num_layers, hidden_dim)
        pooled = hidden_all.reshape(hidden_all.size(0), -1)   # (batch, num_layers * hidden_dim)
        return seq_output, pooled

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Return per-timestep hidden states from the last LSTM layer."""
        seq_output, _ = self._encode_market_tensors(market_windows)
        return seq_output

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Return concatenation of all layer hidden states for richer encoding."""
        _, pooled = self._encode_market_tensors(market_windows)
        return pooled

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_windows: (batch, seq_len) or (batch, seq_len, input_dim)

        Returns:
            (batch,) - predicted log-return
        """
        last_hidden = self._encode_tensor(market_windows)
        pred = self.fc(last_hidden)
        return pred.squeeze(-1)  # (batch,)

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
    ) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        X_train = torch.tensor(market_windows_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device) * _TARGET_SCALE
        X_val = torch.tensor(market_windows_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE

        self.huber_delta = compute_huber_delta(y_train.cpu().numpy())
        logger.debug("LSTM huber_delta={:.4f} (midpoint p25/p75 of |targets|)", self.huber_delta)

        def _loss_fn(pred, target):
            return sign_aware_huber_loss(
                pred, target,
                huber_delta=self.huber_delta,
                sign_penalty_weight=self.sign_penalty_weight,
            )

        return train_with_early_stopping(
            self, X_train, y_train, X_val, y_val, _loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            warmup_epochs=warmup_epochs,
            model_name="LSTM",
        )

    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.forward(X)

        return preds.cpu().numpy() / _TARGET_SCALE

    def get_embeddings(self, market_windows: np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self._encode_tensor(X)
        return embeddings.cpu().numpy()

    # --- BaseEncoder protocol ---

    @property
    def supports_sequence(self) -> bool:
        return True

    @property
    def supports_temporal_fusion(self) -> bool:
        return True

    @property
    def sequence_d_model(self) -> int:
        return self.hidden_dim

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


class RandomForestRegressor_Wrapper:
    """Random Forest baseline over multivariate market windows.
    
    UNCHANGED: This model was already correctly implemented.
    Random Forest doesn't need target scaling, only feature scaling.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        max_features: str | int | float | None = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        
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
        """Extract deterministic summary features from market windows.

        Accepts either close-only windows ``(N, seq_len)`` or multivariate market
        windows ``(N, seq_len, n_features)``. In the multivariate case, the model
        sees the full OHLCV + technical-indicator tensor through sequence summaries.
        """
        X = _as_float32_array(market_windows)
        if X.ndim == 2:
            X = X[:, :, None]
        if X.ndim != 3:
            raise ValueError(
                "market_windows must have shape (N, seq_len) or (N, seq_len, n_features)"
            )

        last_step = X[:, -1, :]
        window_mean = X.mean(axis=1)
        window_std = X.std(axis=1)
        window_min = X.min(axis=1)
        window_max = X.max(axis=1)
        trend = X[:, -1, :] - X[:, 0, :]
        recent_mean = X[:, -min(X.shape[1], 5) :, :].mean(axis=1)
        recent_std = X[:, -min(X.shape[1], 5) :, :].std(axis=1)

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
    
    def fit(
        self,
        market_windows_train: np.ndarray,
        targets_train: np.ndarray,
    ) -> None:
        """Train Random Forest on extracted features.
        
        Args:
            market_windows_train: market windows with close-only or multivariate shape
            targets_train: (N_train,) log-return targets
        """
        X_train = self._extract_features(market_windows_train)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, targets_train)
        self.is_fitted = True
        logger.info(
            "Random Forest trained: {} samples, {} features",
            len(X_train), X_train.shape[1]
        )
    
    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        """Generate predictions on test set.
        
        Args:
            market_windows: market windows with close-only or multivariate shape
            
        Returns:
            (N,) predicted log-returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._extract_features(market_windows)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        return preds.astype(np.float32)

    # --- BaseEncoder protocol ---

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


class FineTunedChronosPredictor:
    """Fine-tune Chronos embeddings for return prediction.
    
    FIXED: Major changes to target scaling and prediction consistency:
    1. Removed target scaling entirely - predictions and targets in same space
    2. Model now directly learns to predict log-returns without scaling artifacts
    3. More stable training and inference
    
    This allows the model to adapt from price forecasting to log-return prediction
    while optionally conditioning on market tabular features.
    """

    _embedding_cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
    _embedding_cache_max_entries = 32
    
    def __init__(
        self,
        chronos_predictor,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        tabular_dim: int = 0,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = 0.05,
        device: str = "cpu",
    ):
        """
        Args:
            chronos_predictor: ChronosMarketPredictor instance (with unfrozen encoder)
            hidden_dim: Hidden dimension of adapter MLP
            dropout: Dropout rate in adapter
            tabular_dim: Dimension of optional market tabular features
            device: Device to use for training
        """
        self.chronos = chronos_predictor
        self.device = device
        self.d_model = chronos_predictor.d_model  # Usually 512
        self.tabular_dim = int(tabular_dim)
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        
        # Adapter head (learns task-specific transformation)
        self.adapter = nn.Sequential(
            nn.Linear(self.d_model + self.tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)
        
        # FIXED: No scaler needed - predictions are directly in log-return space
        self.is_fitted = False

    @classmethod
    def _prune_embedding_cache(cls) -> None:
        while len(cls._embedding_cache) > cls._embedding_cache_max_entries:
            cls._embedding_cache.popitem(last=False)

    def _embedding_cache_key(self, close_windows: np.ndarray) -> tuple[int, str]:
        close_array = np.ascontiguousarray(close_windows, dtype=np.float32)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(close_array.shape).encode())
        digest.update(close_array.dtype.str.encode())
        digest.update(close_array.view(np.uint8))
        return id(self.chronos), digest.hexdigest()

    def _get_cached_embeddings(
        self,
        close_windows: np.ndarray,
        *,
        cache_label: str,
    ) -> np.ndarray:
        cache_key = self._embedding_cache_key(close_windows)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            self._embedding_cache.move_to_end(cache_key)
            logger.debug("Chronos embedding cache hit [{}]", cache_label)
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
        """Combine Chronos embeddings with optional market tabular features.
        
        FIXED: Added validation to ensure dimensions match expectations.
        """
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
    ) -> dict:
        """Fine-tune Chronos encoder with adapter head.
        
        FIXED: Removed target scaling - model learns directly in log-return space.
        This prevents scaling artifacts and makes predictions more interpretable.
        
        Uses low learning rate to preserve pre-trained knowledge while adapting to task.
        
        Args:
            close_windows_train: (N_train, seq_len) raw close prices
            targets_train: (N_train,) log-return targets
            close_windows_val: (N_val, seq_len) raw close prices
            targets_val: (N_val,) log-return targets
            market_tabular_train: optional latest-step OHLCV + indicator features
            market_tabular_val: optional latest-step OHLCV + indicator features
            epochs: Max number of training epochs
            batch_size: Training batch size
            learning_rate: Low learning rate for fine-tuning
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        import torch.optim as optim
        
        logger.info("Fine-tuning Chronos with adapter head (lr={})", learning_rate)

        # Resolve Chronos embeddings through the shared cache so repeated fits
        # and HPO trials do not recompute the same windows unnecessarily.
        emb_train = self._combine_features(
            self._get_cached_embeddings(close_windows_train, cache_label="train"),
            market_tabular_train,
        )
        emb_val = self._combine_features(
            self._get_cached_embeddings(close_windows_val, cache_label="val"),
            market_tabular_val,
        )
        
        # FIXED: No target scaling - use raw log-returns
        # Convert to tensors
        X_train = torch.tensor(emb_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(emb_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val, dtype=torch.float32, device=self.device)
        
        # Setup training
        optimizer = optim.AdamW(self.adapter.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        n_train = len(X_train)
        
        for epoch in range(epochs):
            # Training
            self.adapter.train()
            indices = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i : i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
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
            
            train_loss = epoch_loss / n_batches
            train_losses.append(train_loss)
            
            # Validation
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
            
            if (epoch + 1) % 5 == 0:
                logger.debug(
                    "Fine-tuned Chronos epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}",
                    epoch + 1, epochs, train_loss, val_loss
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Fine-tuned Chronos early stopping at epoch {}", epoch + 1)
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
        """Generate predictions on test set.
        
        FIXED: No inverse transform needed - predictions are already in log-return space.
        
        Args:
            close_windows: (N, seq_len) raw close prices
            market_tabular: optional latest-step OHLCV + indicator features
            
        Returns:
            (N,) predicted log-returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get embeddings
        embeddings = self._combine_features(
            self._get_cached_embeddings(close_windows, cache_label="predict"),
            market_tabular,
        )
        X = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        
        # Predict
        self.adapter.eval()
        with torch.no_grad():
            preds = self.adapter(X).squeeze(-1).cpu().numpy()
        
        # FIXED: No inverse transform - predictions are already in log-return space
        return preds.astype(np.float32)


class _CausalDilatedBlock(nn.Module):
    """Causal dilated convolutional residual block (from TCN, Bai et al. 2018).

    Uses left-only padding to enforce strict causality â€” no future information
    leaks into the current timestep. Dilated convolutions expand the receptive
    field exponentially without adding parameters.

    Receptive field per block: 2 * (kernel_size - 1) * dilation steps.
    """

    def __init__(self, num_filters: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self._causal_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size,
                               dilation=dilation, padding=0)
        self.norm1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size,
                               dilation=dilation, padding=0)
        self.norm2 = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_filters, seq_len)
        residual = x
        # Left-pad only â†’ strict causality (no future leak)
        out = F.pad(x, (self._causal_pad, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = F.pad(out, (self._causal_pad, 0))
        out = self.conv2(out)
        out = self.norm2(out)
        # Residual connection: stabilises gradient flow in deep stacks
        return self.dropout(self.relu(out + residual))


class CNNLSTMPredictor(nn.Module):
    """Causal dilated CNN + LSTM with all-layer concat for market return prediction.

    Architecture based on three published findings:
    - Bai et al. (2018) arXiv:1803.01271 (TCN): causal dilated convolutions with
      residual connections outperform standard LSTM on sequence tasks; dilation
      rates [1, 2, 4] give a receptive field of 28 steps with only 3 blocks.
    - Chakraborty & Basu (2024) arXiv:2410.12807: hierarchical CNNâ†’LSTM where CNN
      identifies local price-volume patterns, LSTM captures macro temporal dynamics.

    Architecture:
        (batch, seq_len, input_dim)
        â†’ Linear input projection â†’ (batch, seq_len, num_filters)
        â†’ permute â†’ (batch, num_filters, seq_len)
        â†’ CausalDilatedBlock Ã— len(dilations)  [dilation=1,2,4 by default]
        â†’ permute â†’ (batch, seq_len, num_filters)
        â†’ LSTM(num_filters, hidden_dim, num_layers)
        â†’ All-layer hidden state concat â†’ (batch, num_layers * hidden_dim)
        â†’ Dropout â†’ Linear bottleneck(num_layers*hidden_dim, hidden_dim//2, 1) â†’ (batch,)

    Effective receptive field with kernel=3, dilations=[1,2,4]:
        2*(3-1)*1 + 2*(3-1)*2 + 2*(3-1)*4 = 4+8+16 = 28 timesteps
    """

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
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.device = device
        self.d_model = num_layers * hidden_dim  # all-layer concat output dim

        # FIX: enforce minimum dropout to prevent overfitting with high param count
        dropout = max(dropout, 0.15)

        # Project raw features into filter space before CNN
        self.input_proj = nn.Linear(input_dim, num_filters)

        # Causal dilated TCN blocks
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

        self.dropout = nn.Dropout(dropout)
        # All-layer concat bottleneck head (matches LSTM architecture)
        self.fc = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.to(device)

        receptive_field = sum(2 * (kernel_size - 1) * d for d in dilations)
        logger.info(
            "CNN-LSTM (causal+attn) initialized | input_dim={}, filters={}, "
            "dilations={}, hidden={}, layers={} | receptive_field={}",
            input_dim, num_filters, list(dilations), hidden_dim, num_layers, receptive_field,
        )

    def _encode_market_tensors(
        self,
        market_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal states and pooled latent encoding for market windows."""
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        # x: (batch, seq_len, input_dim)

        # Project to filter dimension
        x = self.input_proj(x)            # (batch, seq_len, num_filters)
        x = x.permute(0, 2, 1)           # (batch, num_filters, seq_len)

        # Causal dilated TCN blocks (each block has internal residual connection)
        for block in self.tcn_blocks:
            x = block(x)                  # (batch, num_filters, seq_len)

        x = x.permute(0, 2, 1)           # (batch, seq_len, num_filters)

        # LSTM â€” all-layer hidden state concat (matches LSTMPredictor encoding)
        seq_output, (hidden_state, _) = self.lstm(x)  # hidden: (num_layers, batch, hidden_dim)
        hidden_all = hidden_state.permute(1, 0, 2)  # (batch, num_layers, hidden_dim)
        encoding = hidden_all.reshape(hidden_all.size(0), -1)  # (batch, num_layers*hidden_dim)

        return seq_output, encoding

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Return per-timestep hidden states from the last LSTM layer."""
        seq_output, _ = self._encode_market_tensors(market_windows)
        return seq_output

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Return pooled encoder state matching the market-only regression head."""
        _, encoding = self._encode_market_tensors(market_windows)
        return encoding

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_windows: (batch, seq_len) or (batch, seq_len, input_dim)

        Returns:
            (batch,) predicted log-return
        """
        _, encoding = self._encode_market_tensors(market_windows)

        out = self.dropout(encoding)
        out = self.fc(out)                # (batch, 1)
        return out.squeeze(-1)            # (batch,)

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
    ) -> dict:
        """Train CNN-LSTM with early stopping on validation loss.

        Uses gradient clipping (max_norm=1.0) and ReduceLROnPlateau scheduler
        following Yang (2025) and Wang et al. (2026) best practices for
        hybrid CNN-LSTM financial forecasting.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        X_train = torch.tensor(market_windows_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device) * _TARGET_SCALE
        X_val = torch.tensor(market_windows_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val, dtype=torch.float32, device=self.device) * _TARGET_SCALE

        self.huber_delta = compute_huber_delta(y_train.cpu().numpy())
        logger.debug("CNN-LSTM huber_delta={:.4f} (IQR midpoint of |targets|)", self.huber_delta)

        def _loss_fn(pred, target):
            return sign_aware_huber_loss(
                pred, target,
                huber_delta=self.huber_delta,
                sign_penalty_weight=self.sign_penalty_weight,
            )

        return train_with_early_stopping(
            self, X_train, y_train, X_val, y_val, _loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            warmup_epochs=warmup_epochs,
            model_name="CNN-LSTM",
        )

    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        """Generate predictions from market windows (unscaled to original target space)."""
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.forward(X)
        return preds.cpu().numpy() / _TARGET_SCALE

    def checkpoint_state(self) -> dict:
        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    # --- BaseEncoder protocol ---

    @property
    def supports_sequence(self) -> bool:
        return True

    @property
    def supports_temporal_fusion(self) -> bool:
        return True

    @property
    def sequence_d_model(self) -> int:
        return self.hidden_dim

    def encode_sequence_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self._encode_sequence_tensor(market_windows)

    def encode_pooled_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self._encode_tensor(market_windows)

    def predict_market_only_torch(self, market_windows: torch.Tensor) -> torch.Tensor:
        return self.forward(market_windows)

    def encoder_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.parameters())

    def encode(self, market_windows: np.ndarray) -> np.ndarray:
        """Encode market windows via TCN + LSTM â†’ (N, num_layers * hidden_dim)."""
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            encoding = self._encode_tensor(X)
        return encoding.cpu().numpy()

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict(market_windows)


