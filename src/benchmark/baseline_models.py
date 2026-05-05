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
    - ChronosLoRAPredictor: true Chronos encoder fine-tuning via LoRA
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


class ChronosLoRAEncoderBackbone:
    """Shared Chronos encoder backbone adapted via LoRA."""

    def __init__(
        self,
        chronos_predictor,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        device: str = "cpu",
    ) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        self.chronos = chronos_predictor
        self.device = device
        self.d_model = chronos_predictor.d_model
        self.tokenizer = chronos_predictor.pipeline.tokenizer

        pipeline_model = chronos_predictor.pipeline.model
        base_transformer = getattr(pipeline_model, "model", pipeline_model)
        transformer = copy.deepcopy(base_transformer).to(device)
        transformer.requires_grad_(False)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],
            bias="none",
        )
        self.transformer = get_peft_model(transformer, lora_config)
        self.transformer.requires_grad_(False)
        self._enable_encoder_lora_parameters()

    def _enable_encoder_lora_parameters(self) -> None:
        from peft.tuners.lora import LoraLayer

        encoder = getattr(self.transformer, "encoder", None)
        if encoder is None:
            raise AttributeError("LoRA-wrapped Chronos model does not expose an encoder module")

        def _iter_adapter_leaves(adapter_container):
            if adapter_container is None:
                return
            if isinstance(adapter_container, torch.nn.ModuleDict):
                for module in adapter_container.values():
                    yield from module.parameters()
                return
            if isinstance(adapter_container, torch.nn.ParameterDict):
                yield from adapter_container.values()
                return
            if isinstance(adapter_container, torch.nn.Module):
                yield from adapter_container.parameters()
                return
            if isinstance(adapter_container, torch.nn.Parameter):
                yield adapter_container

        enabled_params = 0
        for module in encoder.modules():
            if not isinstance(module, LoraLayer):
                continue
            for attr_name in (
                "lora_A",
                "lora_B",
                "lora_embedding_A",
                "lora_embedding_B",
                "lora_magnitude_vector",
            ):
                for param in _iter_adapter_leaves(getattr(module, attr_name, None)):
                    param.requires_grad = True
                    enabled_params += int(param.numel())

        if enabled_params == 0:
            raise RuntimeError("No encoder LoRA parameters were enabled for training")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [param for param in self.transformer.parameters() if param.requires_grad]

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.transformer.named_parameters() if param.requires_grad]

    def tokenize_windows(self, close_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize close price windows for the Chronos encoder."""
        close_tensor = torch.as_tensor(close_windows, dtype=torch.float32)
        if close_tensor.ndim != 2:
            raise ValueError("close_windows must have shape (N, seq_len) for Chronos tokenization")
        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(close_tensor)
        return token_ids.cpu().numpy(), attention_mask.cpu().numpy()

    def encode_tokenized(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized windows through the LoRA-adapted Chronos encoder."""
        encoder_out = self.transformer.encoder(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        hidden = encoder_out.last_hidden_state
        mask = attention_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def get_embeddings(self, close_windows: np.ndarray) -> np.ndarray:
        """Extract pooled encoder embeddings for raw close windows."""
        token_ids, attention_mask = self.tokenize_windows(close_windows)
        self.transformer.eval()
        with torch.no_grad():
            pooled = self.encode_tokenized(
                torch.as_tensor(token_ids, dtype=torch.long),
                torch.as_tensor(attention_mask, dtype=torch.long),
            )
        return pooled.cpu().numpy().astype(np.float32)

    def checkpoint_state(self) -> dict:
        from peft import get_peft_model_state_dict

        return {"peft_state": get_peft_model_state_dict(self.transformer)}

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.transformer, checkpoint["peft_state"])


class LSTMPredictor(nn.Module):
    """Sequence baseline that predicts next-horizon log return.

    Supports both close-only windows with shape ``(batch, seq_len)`` and
    multivariate market windows with shape ``(batch, seq_len, input_dim)``.
    
    UNCHANGED: This model was already correctly implemented.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.device = device
        self.d_model = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output layer: predict single return value
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        """Return the last hidden state for each market window."""
        x = _ensure_market_sequence_tensor(market_windows, self.input_dim)
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]
        
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
        patience: int = 10,
    ) -> dict:
        """Train LSTM with early stopping on validation loss.
        
        Args:
            market_windows_train: training market windows
            targets_train: (N_train,) log-return targets
            market_windows_val: validation market windows
            targets_val: (N_val,) log-return targets
            epochs: Max number of training epochs
            batch_size: Training batch size
            learning_rate: Adam learning rate
            patience: Early stopping patience
            
        Returns:
            Training history dict with train/val losses
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        # Convert to tensors
        X_train = torch.tensor(market_windows_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(market_windows_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val, dtype=torch.float32, device=self.device)
        
        n_train = len(X_train)
        
        for epoch in range(epochs):
            # Training
            self.train()
            indices = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i : i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                optimizer.zero_grad()
                pred = self.forward(X_batch)
                loss = sign_aware_huber_loss(
                    pred,
                    y_batch,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                )
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            train_losses.append(train_loss)
            
            # Validation
            self.eval()
            with torch.no_grad():
                pred_val = self.forward(X_val)
                val_loss = sign_aware_huber_loss(
                    pred_val,
                    y_val,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                ).item()
                val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "LSTM epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}",
                    epoch + 1, epochs, train_loss, val_loss
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("LSTM early stopping at epoch {}", epoch + 1)
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }
    
    def predict(self, market_windows: np.ndarray) -> np.ndarray:
        """Generate predictions on test set.
        
        Args:
            market_windows: market windows with close-only or multivariate shape
            
        Returns:
            (N,) predicted log-returns
        """
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            preds = self.forward(X)
        
        return preds.cpu().numpy()

    def get_embeddings(self, market_windows: np.ndarray) -> np.ndarray:
        """Expose frozen LSTM hidden states for downstream fusion models."""
        self.eval()
        X = torch.tensor(market_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self._encode_tensor(X)
        return embeddings.cpu().numpy()


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


class ChronosLoRAPredictor:
    """True Chronos fine-tuning via LoRA adapters on the encoder stack.
    
    MAJOR FIXES:
    1. Removed problematic magnitude-sign decomposition architecture
    2. Removed auxiliary sign classification loss
    3. Removed target scaling to avoid scale mismatch issues
    4. Now uses simple regression head like other models
    5. Maintains LoRA fine-tuning of Chronos encoder
    
    This simplified architecture is more stable and easier to train while
    still benefiting from LoRA-based encoder adaptation.
    """

    def __init__(
        self,
        chronos_predictor,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        tabular_dim: int = 0,
        market_input_dim: int = 0,
        market_hidden_dim: int = 64,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = 0.05,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        device: str = "cpu",
    ):
        self.chronos = chronos_predictor
        self.device = device
        self.tabular_dim = int(tabular_dim)
        self.market_input_dim = int(market_input_dim)
        self.market_hidden_dim = int(market_hidden_dim) if self.market_input_dim > 0 else 0
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.backbone = ChronosLoRAEncoderBackbone(
            chronos_predictor,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
        )
        self.transformer = self.backbone.transformer
        self.tokenizer = self.backbone.tokenizer
        self.d_model = self.backbone.d_model

        # Optional market encoder for multivariate sequences
        self.market_encoder = None
        if self.market_input_dim > 0:
            self.market_encoder = nn.LSTM(
                input_size=self.market_input_dim,
                hidden_size=self.market_hidden_dim,
                num_layers=1,
                batch_first=True,
            ).to(device)

        # FIXED: Simple regression head instead of magnitude-sign decomposition
        combined_dim = self.d_model + self.tabular_dim + self.market_hidden_dim
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)

        trainable_names = self.backbone.trainable_parameter_names()
        logger.info(
            "Chronos LoRA initialized | trainable adapter params={} | encoder-only={} | market_input_dim={} | SIMPLIFIED regression head",
            len(trainable_names),
            all("encoder." in name for name in trainable_names),
            self.market_input_dim,
        )

        # FIXED: No scaler - predictions in log-return space
        self.is_fitted = False

    def _combine_tensor_features(
        self,
        embeddings: torch.Tensor,
        market_windows: torch.Tensor | None = None,
        market_tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine Chronos embeddings with optional market features.
        
        FIXED: Added better validation and error messages.
        """
        feature_parts = [embeddings]

        if self.market_input_dim > 0:
            if market_windows is None:
                raise ValueError("market_windows is required when market_input_dim > 0")
            market_windows = _ensure_market_sequence_tensor(market_windows, self.market_input_dim)
            _, (market_hidden, _) = self.market_encoder(market_windows.to(self.device))
            feature_parts.append(market_hidden[-1])

        if self.tabular_dim > 0:
            if market_tabular is None:
                raise ValueError("market_tabular is required when tabular_dim > 0")
            if market_tabular.ndim != 2 or market_tabular.shape[1] != self.tabular_dim:
                raise ValueError(
                    f"Expected market_tabular shape (N, {self.tabular_dim}), got {tuple(market_tabular.shape)}"
                )
            feature_parts.append(market_tabular)

        if len(feature_parts) == 1:
            return embeddings
        return torch.cat(feature_parts, dim=1)

    @property
    def combined_feature_dim(self) -> int:
        """Dimension of the pre-regression market representation."""
        return self.d_model + self.market_hidden_dim + self.tabular_dim

    def extract_tokenized_features(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        market_windows: torch.Tensor | None = None,
        market_tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the exact market feature tensor consumed by the baseline head."""
        pooled = self._encode_tokenized(token_ids, attention_mask)
        return self._combine_tensor_features(pooled, market_windows, market_tabular)

    def regress_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map pre-regression market features to the baseline scalar prediction."""
        return self.regression_head(features).squeeze(-1)

    def tokenize_windows(self, close_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize close price windows for Chronos encoder."""
        return self.backbone.tokenize_windows(close_windows)

    def _encode_tokenized(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized sequences through LoRA-adapted Chronos encoder."""
        return self.backbone.encode_tokenized(token_ids, attention_mask)

    def get_embeddings(self, close_windows: np.ndarray) -> np.ndarray:
        """Extract embeddings from Chronos encoder for given price windows."""
        return self.backbone.get_embeddings(close_windows)

    def fit(
        self,
        close_windows_train: np.ndarray,
        targets_train: np.ndarray,
        close_windows_val: np.ndarray,
        targets_val: np.ndarray,
        market_tabular_train: np.ndarray | None = None,
        market_tabular_val: np.ndarray | None = None,
        market_windows_train: np.ndarray | None = None,
        market_windows_val: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        patience: int = 5,
        pruning_callback=None,
    ) -> dict:
        """Train model with tokenization wrapper."""
        train_token_ids, train_attention_mask = self.tokenize_windows(close_windows_train)
        val_token_ids, val_attention_mask = self.tokenize_windows(close_windows_val)
        return self.fit_tokenized(
            train_token_ids,
            train_attention_mask,
            targets_train,
            val_token_ids,
            val_attention_mask,
            targets_val,
            market_tabular_train=market_tabular_train,
            market_tabular_val=market_tabular_val,
            market_windows_train=market_windows_train,
            market_windows_val=market_windows_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            pruning_callback=pruning_callback,
        )

    def fit_tokenized(
        self,
        token_ids_train: np.ndarray,
        attention_mask_train: np.ndarray,
        targets_train: np.ndarray,
        token_ids_val: np.ndarray,
        attention_mask_val: np.ndarray,
        targets_val: np.ndarray,
        market_tabular_train: np.ndarray | None = None,
        market_tabular_val: np.ndarray | None = None,
        market_windows_train: np.ndarray | None = None,
        market_windows_val: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        patience: int = 5,
        pruning_callback=None,
    ) -> dict:
        """Fine-tune Chronos LoRA with regression head.
        
        FIXED: Major simplification:
        1. Removed target scaling
        2. Removed magnitude-sign decomposition
        3. Removed auxiliary sign classification
        4. Simple regression loss only
        
        This makes training more stable and predictions more interpretable.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Convert to tensors
        token_train = torch.as_tensor(token_ids_train, dtype=torch.long)
        mask_train = torch.as_tensor(attention_mask_train, dtype=torch.long)
        token_val = torch.as_tensor(token_ids_val, dtype=torch.long)
        mask_val = torch.as_tensor(attention_mask_val, dtype=torch.long)

        # FIXED: No target scaling
        y_train = torch.as_tensor(targets_train, dtype=torch.float32)
        y_val = torch.as_tensor(targets_val, dtype=torch.float32, device=self.device)

        # Optional market features
        train_tab = None
        val_tab = None
        if market_tabular_train is not None:
            train_tab = torch.as_tensor(market_tabular_train, dtype=torch.float32)
        if market_tabular_val is not None:
            val_tab = torch.as_tensor(market_tabular_val, dtype=torch.float32, device=self.device)

        train_market = None
        val_market = None
        if market_windows_train is not None:
            train_market = torch.as_tensor(market_windows_train, dtype=torch.float32)
        if market_windows_val is not None:
            val_market = torch.as_tensor(market_windows_val, dtype=torch.float32, device=self.device)

        # Build training dataset
        dataset_tensors = [token_train, mask_train]
        if train_market is not None:
            dataset_tensors.append(train_market)
        if train_tab is not None:
            dataset_tensors.append(train_tab)
        dataset_tensors.append(y_train)
        train_ds = TensorDataset(*dataset_tensors)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # Setup optimizer for all trainable parameters
        params = list(self.regression_head.parameters())
        if self.market_encoder is not None:
            params.extend(self.market_encoder.parameters())
        params.extend(self.backbone.trainable_parameters())
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=1e-5)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.transformer.train()
            self.regression_head.train()
            if self.market_encoder is not None:
                self.market_encoder.train()
            
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # Unpack batch
                batch_idx = 0
                mb_token_ids = batch[batch_idx]
                batch_idx += 1
                mb_attention_mask = batch[batch_idx]
                batch_idx += 1
                mb_market = None
                if train_market is not None:
                    mb_market = batch[batch_idx]
                    batch_idx += 1
                mb_tabular = None
                if train_tab is not None:
                    mb_tabular = batch[batch_idx]
                    batch_idx += 1
                mb_targets = batch[batch_idx]

                # Forward pass
                optimizer.zero_grad()
                features = self.extract_tokenized_features(
                    mb_token_ids,
                    mb_attention_mask,
                    mb_market.to(self.device) if mb_market is not None else None,
                    mb_tabular.to(self.device) if mb_tabular is not None else None,
                )
                
                # FIXED: Simple regression prediction
                pred = self.regress_features(features)
                
                # FIXED: Single regression loss (no auxiliary losses)
                loss = sign_aware_huber_loss(
                    pred,
                    mb_targets.to(self.device),
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # Validation
            self.transformer.eval()
            self.regression_head.eval()
            if self.market_encoder is not None:
                self.market_encoder.eval()
            
            with torch.no_grad():
                val_features = self.extract_tokenized_features(token_val, mask_val, val_market, val_tab)
                pred_val = self.regress_features(val_features)
                
                val_loss = sign_aware_huber_loss(
                    pred_val,
                    y_val,
                    huber_delta=self.huber_delta,
                    sign_penalty_weight=self.sign_penalty_weight,
                ).item()
            val_losses.append(val_loss)

            if pruning_callback is not None:
                pruning_callback(epoch, val_loss)

            if (epoch + 1) % 5 == 0:
                logger.debug(
                    "Chronos LoRA epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, pred_mean={:.5f}, pred_std={:.4f}",
                    epoch + 1, epochs, train_loss, val_loss, 
                    float(pred_val.mean().item()), float(pred_val.std().item())
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.checkpoint_state()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Chronos LoRA early stopping at epoch {}", epoch + 1)
                    break

        if best_state is not None:
            self.load_checkpoint_state(best_state)

        self.is_fitted = True
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def checkpoint_state(self) -> dict:
        """Save model state for checkpointing.
        
        FIXED: Removed scaler state, updated for new architecture.
        """
        return {
            **self.backbone.checkpoint_state(),
            "regression_head_state": {k: v.detach().cpu().clone() for k, v in self.regression_head.state_dict().items()},
            "market_encoder_state": (
                {k: v.detach().cpu().clone() for k, v in self.market_encoder.state_dict().items()}
                if self.market_encoder is not None else None
            ),
            "is_fitted": self.is_fitted,
        }

    def load_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore model from checkpoint.
        
        FIXED: Removed scaler restoration, updated for new architecture.
        """
        self.backbone.load_checkpoint_state(checkpoint)
        self.regression_head.load_state_dict(checkpoint["regression_head_state"])
        if self.market_encoder is not None and checkpoint.get("market_encoder_state") is not None:
            self.market_encoder.load_state_dict(checkpoint["market_encoder_state"])
        self.is_fitted = bool(checkpoint.get("is_fitted", True))

    def predict(
        self,
        close_windows: np.ndarray,
        market_tabular: np.ndarray | None = None,
        market_windows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate predictions with tokenization wrapper."""
        token_ids, attention_mask = self.tokenize_windows(close_windows)
        return self.predict_tokenized(
            token_ids,
            attention_mask,
            market_tabular=market_tabular,
            market_windows=market_windows,
        )

    def predict_tokenized(
        self,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        market_tabular: np.ndarray | None = None,
        market_windows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate predictions on test set.
        
        FIXED: No inverse transform - predictions already in log-return space.
        
        Args:
            token_ids: Tokenized close price windows
            attention_mask: Attention mask for tokens
            market_tabular: Optional market tabular features
            market_windows: Optional market sequence windows
            
        Returns:
            (N,) predicted log-returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        token_ids_t = torch.as_tensor(token_ids, dtype=torch.long)
        attention_mask_t = torch.as_tensor(attention_mask, dtype=torch.long)
        tabular_t = None
        if market_tabular is not None:
            tabular_t = torch.as_tensor(market_tabular, dtype=torch.float32, device=self.device)
        market_t = None
        if market_windows is not None:
            market_t = torch.as_tensor(market_windows, dtype=torch.float32, device=self.device)

        self.transformer.eval()
        self.regression_head.eval()
        if self.market_encoder is not None:
            self.market_encoder.eval()
        
        with torch.no_grad():
            features = self.extract_tokenized_features(token_ids_t, attention_mask_t, market_t, tabular_t)
            
            # FIXED: Simple regression prediction
            preds = self.regress_features(features).cpu().numpy()

        # FIXED: No inverse transform needed
        return preds.astype(np.float32)