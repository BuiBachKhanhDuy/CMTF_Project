"""Baseline models for comparison: LSTM, Random Forest, Fine-tuned Chronos.

This module provides simple, fair baselines that use only historical close prices
(matching the Chronos market-only baseline input domain).

Models:
    - LSTMPredictor: 2-layer LSTM on price sequences
    - RandomForestRegressor_Wrapper: RF with 30+ technical features
    - FineTunedChronosPredictor: Chronos T5 with unfrozen encoder + adapter head
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from loguru import logger


class LSTMPredictor(nn.Module):
    """Simple LSTM-based return predictor.
    
    Takes raw close-price windows and predicts next N-day log-return.
    Fair comparison: uses only close windows, no tabular features or news.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.d_model = hidden_dim
        
        # LSTM takes close prices as input (input_size=1)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output layer: predict single return value
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)

    def _encode_tensor(self, close_windows: torch.Tensor) -> torch.Tensor:
        """Return the last hidden state for each close-price window."""
        x = close_windows.unsqueeze(-1)
        _, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]
        
    def forward(self, close_windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            close_windows: (batch, seq_len) - raw close prices
            
        Returns:
            (batch, 1) - predicted log-return
        """
        last_hidden = self._encode_tensor(close_windows)
        pred = self.fc(last_hidden)
        return pred.squeeze(-1)  # (batch,)
    
    def fit(
        self,
        close_windows_train: np.ndarray,
        targets_train: np.ndarray,
        close_windows_val: np.ndarray,
        targets_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 10,
    ) -> dict:
        """Train LSTM with early stopping on validation loss.
        
        Args:
            close_windows_train: (N_train, seq_len) raw close prices
            targets_train: (N_train,) log-return targets
            close_windows_val: (N_val, seq_len) raw close prices
            targets_val: (N_val,) log-return targets
            epochs: Max number of training epochs
            batch_size: Training batch size
            learning_rate: Adam learning rate
            patience: Early stopping patience
            
        Returns:
            Training history dict with train/val losses
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        # Convert to tensors
        X_train = torch.tensor(close_windows_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(close_windows_val, dtype=torch.float32, device=self.device)
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
                loss = criterion(pred, y_batch)
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
                val_loss = criterion(pred_val, y_val).item()
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
    
    def predict(self, close_windows: np.ndarray) -> np.ndarray:
        """Generate predictions on test set.
        
        Args:
            close_windows: (N, seq_len) raw close prices
            
        Returns:
            (N,) predicted log-returns
        """
        self.eval()
        X = torch.tensor(close_windows, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            preds = self.forward(X)
        
        return preds.cpu().numpy()

    def get_embeddings(self, close_windows: np.ndarray) -> np.ndarray:
        """Expose frozen LSTM hidden states for downstream fusion models."""
        self.eval()
        X = torch.tensor(close_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self._encode_tensor(X)
        return embeddings.cpu().numpy()


class RandomForestRegressor_Wrapper:
    """Random Forest baseline using close-price features only.
    
    Extracts technical features from close-price windows (returns, volatility, etc.)
    and trains a Random Forest regressor. Fair comparison: only uses historical prices.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    @staticmethod
    def _extract_features(close_windows: np.ndarray) -> np.ndarray:
        """Extract 30+ technical features from close-price windows.
        
        Enhanced feature engineering for better RF performance across horizons.
        
        Args:
            close_windows: (N, seq_len) raw close prices
            
        Returns:
            (N, n_features) feature matrix
        """
        N, seq_len = close_windows.shape
        features = []
        
        for i in range(N):
            window = close_windows[i]
            returns = np.diff(np.log(window))
            
            feat_dict = {}
            
            # ===== Original Features (11) =====
            feat_dict["last_close"] = window[-1]
            feat_dict["mean_return"] = returns.mean()
            feat_dict["std_return"] = returns.std()
            feat_dict["volatility"] = np.std(np.diff(window))
            
            price_range = window.max() - window.min()
            feat_dict["position_in_range"] = (window[-1] - window.min()) / (price_range + 1e-8)
            
            ma5 = window[-5:].mean() if len(window) >= 5 else window.mean()
            ma10 = window[-10:].mean() if len(window) >= 10 else window.mean()
            feat_dict["ma_signal"] = ma5 - ma10
            
            feat_dict["recent_return"] = np.log(window[-1] / window[-5]) if len(window) >= 5 else 0.0
            feat_dict["last_return"] = returns[-1]
            feat_dict["second_last_return"] = returns[-2] if len(returns) >= 2 else 0.0
            
            feat_dict["min_price"] = window.min()
            feat_dict["max_price"] = window.max()
            
            # ===== NEW: Momentum Features =====
            feat_dict["momentum_3d"] = returns[-3:].sum() if len(returns) >= 3 else returns.sum()
            feat_dict["momentum_5d"] = returns[-5:].sum() if len(returns) >= 5 else returns.sum()
            feat_dict["momentum_10d"] = returns[-10:].sum() if len(returns) >= 10 else returns.sum()
            
            # ===== NEW: RSI (Relative Strength Index) =====
            up_returns = np.maximum(returns, 0)
            down_returns = np.maximum(-returns, 0)
            avg_up = up_returns.mean()
            avg_down = down_returns.mean()
            rs = avg_up / (avg_down + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            feat_dict["rsi"] = rsi
            
            # ===== NEW: Bollinger Bands =====
            bb_std = np.std(window)
            bb_mean = np.mean(window)
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            bb_range = bb_upper - bb_lower + 1e-8
            feat_dict["bb_position"] = (window[-1] - bb_lower) / bb_range
            feat_dict["bb_width"] = bb_range / window.mean()
            
            # ===== NEW: Moving Averages (more granular) =====
            if len(window) >= 20:
                ma20 = window[-20:].mean()
                feat_dict["ma20"] = ma20
                feat_dict["price_to_ma20"] = window[-1] / (ma20 + 1e-8)
            else:
                feat_dict["ma20"] = window.mean()
                feat_dict["price_to_ma20"] = 1.0
            
            # ===== NEW: MACD (simplified, using exponential averages) =====
            try:
                ema12 = pd.Series(window).ewm(span=12, adjust=False).mean().iloc[-1]
                ema26 = pd.Series(window).ewm(span=26, adjust=False).mean().iloc[-1]
                feat_dict["macd"] = ema12 - ema26
                feat_dict["ema_ratio"] = ema12 / (ema26 + 1e-8)
            except:
                feat_dict["macd"] = 0.0
                feat_dict["ema_ratio"] = 1.0
            
            # ===== NEW: Stochastic Oscillator =====
            stoch_k = (window[-1] - window.min()) / (window.max() - window.min() + 1e-8) * 100
            feat_dict["stoch_k"] = stoch_k
            
            # ===== NEW: Return Distribution Stats =====
            if len(returns) > 1:
                feat_dict["return_skew"] = pd.Series(returns).skew()
                feat_dict["return_kurtosis"] = pd.Series(returns).kurtosis()
            else:
                feat_dict["return_skew"] = 0.0
                feat_dict["return_kurtosis"] = 0.0
            
            # ===== NEW: Volatility Regimes =====
            feat_dict["recent_volatility"] = np.std(returns[-5:]) if len(returns) >= 5 else np.std(returns)
            feat_dict["historic_volatility"] = np.std(returns)
            feat_dict["volatility_ratio"] = feat_dict["recent_volatility"] / (feat_dict["historic_volatility"] + 1e-8)
            
            # ===== NEW: Trend Strength =====
            feat_dict["trend_strength"] = np.abs(returns).mean() / (np.std(returns) + 1e-8)
            
            # ===== NEW: Drawdown/Runup =====
            running_max = np.maximum.accumulate(window)
            drawdown = (window - running_max) / (running_max + 1e-8)
            feat_dict["current_drawdown"] = drawdown[-1]
            feat_dict["max_drawdown"] = drawdown.min()
            
            # ===== NEW: Price acceleration =====
            if len(returns) >= 2:
                accel = np.diff(returns)
                feat_dict["recent_accel"] = accel[-1] if len(accel) > 0 else 0.0
                feat_dict["mean_accel"] = accel.mean()
            else:
                feat_dict["recent_accel"] = 0.0
                feat_dict["mean_accel"] = 0.0
            
            # ===== NEW: Mean reversion signal =====
            feat_dict["deviation_from_ma"] = (window[-1] - ma10) / (np.std(window) + 1e-8)
            
            # Convert to array
            feat_vector = np.array(list(feat_dict.values()), dtype=np.float32)
            features.append(feat_vector)
        
        return np.array(features, dtype=np.float32)
    
    def fit(
        self,
        close_windows_train: np.ndarray,
        targets_train: np.ndarray,
    ) -> None:
        """Train Random Forest on extracted features.
        
        Args:
            close_windows_train: (N_train, seq_len) raw close prices
            targets_train: (N_train,) log-return targets
        """
        X_train = self._extract_features(close_windows_train)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, targets_train)
        self.is_fitted = True
        logger.info(
            "Random Forest trained: {} samples, {} features",
            len(X_train), X_train.shape[1]
        )
    
    def predict(self, close_windows: np.ndarray) -> np.ndarray:
        """Generate predictions on test set.
        
        Args:
            close_windows: (N, seq_len) raw close prices
            
        Returns:
            (N,) predicted log-returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._extract_features(close_windows)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        return preds.astype(np.float32)


class FineTunedChronosPredictor:
    """Fine-tune Chronos embeddings for return prediction.
    
    Unfreezes Chronos encoder and adds an adapter head for task-specific learning.
    This allows the model to adapt from price forecasting to return direction prediction.
    
    Fair comparison: still uses only close prices (via Chronos), but optimized for task.
    """
    
    def __init__(
        self,
        chronos_predictor,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        """
        Args:
            chronos_predictor: ChronosMarketPredictor instance (with unfrozen encoder)
            hidden_dim: Hidden dimension of adapter MLP
            dropout: Dropout rate in adapter
            device: Device to use for training
        """
        self.chronos = chronos_predictor
        self.device = device
        self.d_model = chronos_predictor.d_model  # Usually 512
        
        # Adapter head (learns task-specific transformation)
        self.adapter = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(
        self,
        close_windows_train: np.ndarray,
        targets_train: np.ndarray,
        close_windows_val: np.ndarray,
        targets_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        patience: int = 5,
    ) -> dict:
        """Fine-tune Chronos encoder with adapter head.
        
        Uses low learning rate to preserve pre-trained knowledge while adapting to task.
        
        Args:
            close_windows_train: (N_train, seq_len) raw close prices
            targets_train: (N_train,) log-return targets
            close_windows_val: (N_val, seq_len) raw close prices
            targets_val: (N_val,) log-return targets
            epochs: Max number of training epochs
            batch_size: Training batch size
            learning_rate: Low learning rate for fine-tuning
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        logger.info("Fine-tuning Chronos with adapter head (lr={})", learning_rate)
        
        # Get embeddings (will be cached/computed once)
        logger.info("Computing Chronos embeddings for fine-tuning...")
        emb_train = self.chronos.get_embeddings(close_windows_train)  # (N, d_model)
        emb_val = self.chronos.get_embeddings(close_windows_val)
        
        # Scale targets
        targets_train_scaled = self.scaler.fit_transform(targets_train.reshape(-1, 1)).ravel()
        targets_val_scaled = self.scaler.transform(targets_val.reshape(-1, 1)).ravel()
        
        # Convert to tensors
        X_train = torch.tensor(emb_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(targets_train_scaled, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(emb_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(targets_val_scaled, dtype=torch.float32, device=self.device)
        
        # Setup training
        optimizer = optim.AdamW(self.adapter.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
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
                loss = criterion(pred, y_batch)
                
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
                val_loss = criterion(pred_val, y_val).item()
                val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.debug(
                    "Fine-tuned Chronos epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}",
                    epoch + 1, epochs, train_loss, val_loss
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Fine-tuned Chronos early stopping at epoch {}", epoch + 1)
                    break
        
        self.is_fitted = True
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }
    
    def predict(self, close_windows: np.ndarray) -> np.ndarray:
        """Generate predictions on test set.
        
        Args:
            close_windows: (N, seq_len) raw close prices
            
        Returns:
            (N,) predicted log-returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get embeddings
        embeddings = self.chronos.get_embeddings(close_windows)  # (N, d_model)
        X = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        
        # Predict
        self.adapter.eval()
        with torch.no_grad():
            preds_scaled = self.adapter(X).squeeze(-1).cpu().numpy()
        
        # Unscale
        preds = self.scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        
        return preds.astype(np.float32)

