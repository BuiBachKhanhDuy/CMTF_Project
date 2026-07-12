from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

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
# GLOBAL LOSS CONFIG — single source of truth for all torch models
# =====================================================================

@dataclass(frozen=True)
class GlobalLossConfig:
    """Shared loss hyperparameters inherited by every trainable torch model.

    Safer defaults for unit-scaled targets:
      sign_penalty_weight       — direction remains secondary to regression
      direction_epsilon         — ignore tiny/noisy targets
      direction_margin_fraction — require modest correct-direction confidence
      direction_min_margin      — minimum directional confidence floor
      direction_ramp_epochs     — smooth ramp after warmup to avoid loss discontinuity
      debug_logging             — whether to emit validation diagnostics during fit
    """
    sign_penalty_weight: float = 0.3
    direction_epsilon: float = 0.1
    direction_margin_fraction: float = 0.1
    direction_min_margin: float = 0.1
    direction_ramp_epochs: int = 3
    debug_logging: bool = True
    # Anti-collapse variance regulariser. DEFAULT OFF (0.0) after a controlled
    # A/B: a global 0.5 activated on ordinary low-variance cells (any model whose
    # pred_std < 10%% of target std, which is common for noisy daily returns), and
    # did NOT improve DA/IC on a real signal-bearing 5D cell (it nudged them
    # slightly worse) — anti-collapse is not the same as skill. The opt-in,
    # horizon-scaled values already tuned in run_model_benchmark.py
    # ({1D:0.0, 5D:0.02, 20D:0.05}) remain the supported way to enable it; those
    # are 10-25x smaller than the un-validated 0.5. Set >0 explicitly per model
    # (model.variance_reg_weight = ...) only for cells that actually collapse.
    variance_reg_weight: float = 0.0
    # Balance direction-loss gradient per class to prevent majority-class collapse.
    class_balance_dir: bool = True


GLOBAL_LOSS_CONFIG = GlobalLossConfig()


# =====================================================================
# LOSSES
# =====================================================================

def sign_aware_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    huber_delta: float = 0.02,
    sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
    direction_epsilon: float = GLOBAL_LOSS_CONFIG.direction_epsilon,
    weights: torch.Tensor | None = None,
    margin: float | None = None,
    direction_margin_fraction: float = GLOBAL_LOSS_CONFIG.direction_margin_fraction,
    direction_min_margin: float = GLOBAL_LOSS_CONFIG.direction_min_margin,
    enable_direction_loss: bool = True,
    variance_reg_weight: float = 0.0,
    class_balance_dir: bool = False,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none")
    loss_huber = loss_fn(pred, target)

    if weights is not None:
        norm_weights = weights / weights.mean().clamp_min(1e-8)
        loss_huber = loss_huber * norm_weights
    else:
        norm_weights = None

    huber_mean = loss_huber.mean()
    loss = huber_mean

    active_direction = (torch.abs(target) > direction_epsilon).float()

    if margin is not None:
        per_sample_margin = torch.full_like(target, float(margin))
    else:
        base_margin = torch.clamp(torch.abs(target), min=float(direction_epsilon))
        per_sample_margin = torch.clamp(
            direction_margin_fraction * base_margin,
            min=float(direction_min_margin),
        )

    signed_alignment = pred * torch.sign(target)
    direction_term = torch.relu(per_sample_margin - signed_alignment)
    loss_wrong_dir = direction_term * active_direction

    if norm_weights is not None:
        loss_wrong_dir = loss_wrong_dir * norm_weights

    direction_mean = torch.tensor(0.0, device=pred.device)
    if enable_direction_loss and active_direction.sum().item() > 0:
        if class_balance_dir:
            # Inverse-frequency class weighting: upscale minority-class direction
            # loss to prevent the majority direction dominating training.
            active_mask = active_direction.bool()
            n_active = float(active_mask.sum().clamp(min=1).item())
            n_pos_a = float((target[active_mask] > 0).sum().clamp(min=1).item())
            n_neg_a = float((target[active_mask] <= 0).sum().clamp(min=1).item())
            pos_w = min(n_active / (2.0 * n_pos_a), 5.0)
            neg_w = min(n_active / (2.0 * n_neg_a), 5.0)
            class_w = torch.where(
                target > 0,
                torch.full_like(target, pos_w),
                torch.full_like(target, neg_w),
            )
            direction_mean = (loss_wrong_dir * class_w).sum() / active_direction.sum().clamp_min(1.0)
        else:
            direction_mean = loss_wrong_dir.sum() / active_direction.sum().clamp_min(1.0)
        loss = loss + sign_penalty_weight * direction_mean

    # Variance regulariser: penalise near-constant predictions (anti-collapse).
    # Active only when variance_reg_weight > 0 (long-horizon curriculum).
    # Requires pred_std >= 10% of target_std; penalises shortfall linearly.
    variance_reg = torch.tensor(0.0, device=pred.device)
    if variance_reg_weight > 0.0 and pred.numel() > 1:
        pred_std = pred.std(unbiased=False)
        target_std_est = target.std(unbiased=False).detach()
        shortfall = torch.relu(0.1 * target_std_est - pred_std)
        variance_reg = shortfall
        loss = loss + variance_reg_weight * variance_reg

    if not return_debug:
        return loss

    debug = {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_huber": float(huber_mean.detach().cpu().item()),
        "loss_direction": float(direction_mean.detach().cpu().item()),
        "active_ratio": float(active_direction.mean().detach().cpu().item()),
        "margin_mean": float(per_sample_margin.mean().detach().cpu().item()),
        "signed_alignment_mean": float(signed_alignment.mean().detach().cpu().item()),
        "pred_mean": float(pred.mean().detach().cpu().item()),
        "pred_std": float(pred.std(unbiased=False).detach().cpu().item()),
        "target_mean": float(target.mean().detach().cpu().item()),
        "target_std": float(target.std(unbiased=False).detach().cpu().item()),
        "pct_pred_pos": float((pred > 0).float().mean().detach().cpu().item()),
        "pct_pred_neg": float((pred < 0).float().mean().detach().cpu().item()),
        "pct_pred_near_zero_1e4": float((torch.abs(pred) < 1e-4).float().mean().detach().cpu().item()),
        "pct_pred_near_zero_1e3": float((torch.abs(pred) < 1e-3).float().mean().detach().cpu().item()),
        "variance_reg": float(variance_reg.detach().cpu().item()),
    }
    return loss, debug


def _direction_ramp_factor(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Smoothly increase direction-loss influence after warmup."""
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
    return float(max(0.0, min(1.0, progress)))


def _scheduled_sign_aware_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    epoch: int,
    warmup_epochs: int,
    huber_delta: float,
    sign_penalty_weight: float,
    direction_epsilon: float,
    direction_margin_fraction: float,
    direction_min_margin: float,
    direction_ramp_epochs: int = GLOBAL_LOSS_CONFIG.direction_ramp_epochs,
    weights: torch.Tensor | None = None,
    margin: float | None = None,
    variance_reg_weight: float = 0.0,
    class_balance_dir: bool = False,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Shared scheduled loss with smooth direction ramp for all torch models."""
    ramp = _direction_ramp_factor(epoch, warmup_epochs, direction_ramp_epochs)
    effective_sign_penalty_weight = sign_penalty_weight * ramp
    enable_direction_loss = effective_sign_penalty_weight > 0.0

    result = sign_aware_huber_loss(
        pred,
        target,
        huber_delta=huber_delta,
        sign_penalty_weight=effective_sign_penalty_weight,
        direction_epsilon=direction_epsilon,
        weights=weights,
        margin=margin,
        direction_margin_fraction=direction_margin_fraction,
        direction_min_margin=direction_min_margin,
        enable_direction_loss=enable_direction_loss,
        variance_reg_weight=variance_reg_weight,
        class_balance_dir=class_balance_dir,
        return_debug=return_debug,
    )

    if not return_debug:
        return result

    loss, debug = result
    debug["direction_ramp"] = float(ramp)
    debug["effective_sign_penalty_weight"] = float(effective_sign_penalty_weight)
    return loss, debug


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


def _log_epoch_debug(model_name: str, epoch: int, stage: str, debug: dict[str, float]) -> None:
    logger.debug(
        "{} epoch={} {} | total={:.6f} huber={:.6f} dir={:.6f} "
        "ramp={:.3f} sign_w={:.3f} active={:.3f} margin={:.3f} "
        "pred_mean={:.6f} pred_std={:.6f} pos={:.3f} neg={:.3f} near0_1e3={:.3f}",
        model_name,
        epoch,
        stage,
        debug.get("loss_total", float("nan")),
        debug.get("loss_huber", float("nan")),
        debug.get("loss_direction", float("nan")),
        debug.get("direction_ramp", float("nan")),
        debug.get("effective_sign_penalty_weight", float("nan")),
        debug.get("active_ratio", float("nan")),
        debug.get("margin_mean", float("nan")),
        debug.get("pred_mean", float("nan")),
        debug.get("pred_std", float("nan")),
        debug.get("pct_pred_pos", float("nan")),
        debug.get("pct_pred_neg", float("nan")),
        debug.get("pct_pred_near_zero_1e3", float("nan")),
    )


# =====================================================================
# BASE CLASS FOR TORCH MARKET MODELS
# =====================================================================

class BaseTorchMarketPredictor(nn.Module):
    """Common training/inference logic for single-input torch market models."""

    def __init__(self, target_scale: float = 1.0, device: str = "cpu"):
        super().__init__()
        self.target_scale = float(target_scale)
        self.device = device
        self.huber_delta = 1.0
        self.sign_penalty_weight = GLOBAL_LOSS_CONFIG.sign_penalty_weight

        self.direction_epsilon = GLOBAL_LOSS_CONFIG.direction_epsilon
        self.direction_margin_fraction = GLOBAL_LOSS_CONFIG.direction_margin_fraction
        self.direction_min_margin = GLOBAL_LOSS_CONFIG.direction_min_margin
        self.direction_ramp_epochs = GLOBAL_LOSS_CONFIG.direction_ramp_epochs
        self.debug_logging = GLOBAL_LOSS_CONFIG.debug_logging
        self.variance_reg_weight = GLOBAL_LOSS_CONFIG.variance_reg_weight
        self.class_balance_dir = GLOBAL_LOSS_CONFIG.class_balance_dir

    def _encode_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode_sequence_tensor(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, market_windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_scheduled_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: int,
        warmup_epochs: int,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        return _scheduled_sign_aware_loss(
            pred,
            target,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
            direction_epsilon=self.direction_epsilon,
            direction_margin_fraction=self.direction_margin_fraction,
            direction_min_margin=self.direction_min_margin,
            direction_ramp_epochs=self.direction_ramp_epochs,
            variance_reg_weight=self.variance_reg_weight,
            class_balance_dir=self.class_balance_dir,
            return_debug=return_debug,
        )

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

        direction_epsilon = self.direction_epsilon
        active_ratio_train = float((torch.abs(y_train) > direction_epsilon).float().mean().item())
        active_ratio_val = float((torch.abs(y_val) > direction_epsilon).float().mean().item())

        logger.debug(
            "{} direction_epsilon={:.6f} | active_ratio_train={:.4f} | active_ratio_val={:.4f}",
            model_name,
            direction_epsilon,
            active_ratio_train,
            active_ratio_val,
        )

        self.huber_delta = compute_huber_delta(y_train.numpy())
        logger.debug("{} huber_delta={:.4f}", model_name, self.huber_delta)

        def _loss_fn(pred, target, epoch=0):
            return self._compute_scheduled_loss(
                pred,
                target,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
            )

        logger.debug(
            "{} config | sign_penalty_weight={:.4f} | warmup_epochs={} | direction_ramp_epochs={} | "
            "target_scale={:.4f} | direction_margin_fraction={:.6f} | direction_min_margin={:.6f} | "
            "early_stopping_policy=scheduled_full_loss",
            model_name,
            self.sign_penalty_weight,
            warmup_epochs,
            self.direction_ramp_epochs,
            self.target_scale,
            self.direction_margin_fraction,
            self.direction_min_margin,
        )

        result = train_with_early_stopping(
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
            val_loss_fn=_loss_fn,
        )

        if self.debug_logging:
            self.eval()
            with torch.no_grad():
                pred_val = self.forward(X_val.to(self.device))
                _, val_debug = self._compute_scheduled_loss(
                    pred_val,
                    y_val.to(self.device),
                    epoch=max(0, len(result.get("val_losses", [])) - 1),
                    warmup_epochs=warmup_epochs,
                    return_debug=True,
                )
            _log_epoch_debug(model_name, len(result.get("val_losses", [])) - 1, "val_final", val_debug)

        return result

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

    def __init__(self, target_scale: float = 1.0, device: str = "cpu"):
        super().__init__()
        self.target_scale = float(target_scale)
        self.device = device
        self.huber_delta = 1.0
        self.sign_penalty_weight = GLOBAL_LOSS_CONFIG.sign_penalty_weight

        self.direction_epsilon = GLOBAL_LOSS_CONFIG.direction_epsilon
        self.direction_margin_fraction = GLOBAL_LOSS_CONFIG.direction_margin_fraction
        self.direction_min_margin = GLOBAL_LOSS_CONFIG.direction_min_margin
        self.direction_ramp_epochs = GLOBAL_LOSS_CONFIG.direction_ramp_epochs
        self.debug_logging = GLOBAL_LOSS_CONFIG.debug_logging
        self.variance_reg_weight = GLOBAL_LOSS_CONFIG.variance_reg_weight
        self.class_balance_dir = GLOBAL_LOSS_CONFIG.class_balance_dir

    def forward(self, market_windows: torch.Tensor, market_tabular: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _tabular_from_windows(self, market_windows: np.ndarray) -> np.ndarray:
        return extract_market_summary_features(market_windows)

    def _compute_scheduled_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: int,
        warmup_epochs: int,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        return _scheduled_sign_aware_loss(
            pred,
            target,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            huber_delta=self.huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
            direction_epsilon=self.direction_epsilon,
            direction_margin_fraction=self.direction_margin_fraction,
            direction_min_margin=self.direction_min_margin,
            direction_ramp_epochs=self.direction_ramp_epochs,
            variance_reg_weight=self.variance_reg_weight,
            class_balance_dir=self.class_balance_dir,
            return_debug=return_debug,
        )

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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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

        direction_epsilon = self.direction_epsilon
        active_ratio_train = float((torch.abs(y_train) > direction_epsilon).float().mean().item())
        active_ratio_val = float((torch.abs(y_val) > direction_epsilon).float().mean().item())

        logger.debug(
            "{} direction_epsilon={:.6f} | active_ratio_train={:.4f} | active_ratio_val={:.4f}",
            model_name,
            direction_epsilon,
            active_ratio_train,
            active_ratio_val,
        )

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
            )

        self.huber_delta = compute_huber_delta(y_train.numpy())
        logger.debug("{} huber_delta={:.4f}", model_name, self.huber_delta)
        logger.debug(
            "{} config | sign_penalty_weight={:.4f} | warmup_epochs={} | direction_ramp_epochs={} | "
            "target_scale={:.4f} | direction_margin_fraction={:.6f} | direction_min_margin={:.6f} | "
            "early_stopping_policy=scheduled_full_loss",
            model_name,
            self.sign_penalty_weight,
            warmup_epochs,
            self.direction_ramp_epochs,
            self.target_scale,
            self.direction_margin_fraction,
            self.direction_min_margin,
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        val_losses_clean: list[float] = []
        pred_means: list[float] = []
        pred_pct_pos: list[float] = []
        pred_pct_neg: list[float] = []
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
                loss = self._compute_scheduled_loss(
                    pred,
                    yb,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
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
                
                # Full validation loss (may be gated by warmup ramp for logging/scheduler)
                val_loss_obj, val_debug = self._compute_scheduled_loss(
                    pred_val,
                    y_val.to(self.device),
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    return_debug=True,
                )
                val_loss = float(val_loss_obj.item())
                
                # Clean validation loss: call with epoch=999 to bypass warmup ramp.
                # This ensures early stopping compares apples-to-apples across epochs.
                val_loss_clean_obj = self._compute_scheduled_loss(
                    pred_val,
                    y_val.to(self.device),
                    epoch=999,
                    warmup_epochs=warmup_epochs,
                )
                val_loss_clean = float(val_loss_clean_obj.item())

            if self.debug_logging:
                _log_epoch_debug(model_name, epoch, "val", val_debug)

            val_losses.append(val_loss)
            val_losses_clean.append(val_loss_clean)
            
            # Capture per-epoch prediction statistics
            pred_val_cpu = pred_val.cpu().numpy()
            pred_means.append(float(np.mean(pred_val_cpu)))
            pred_pct_pos.append(float(100.0 * np.mean(pred_val_cpu > 0)))
            pred_pct_neg.append(float(100.0 * np.mean(pred_val_cpu < 0)))

            if scheduler is not None:
                scheduler.step(val_loss)

            # Early stopping uses CLEAN validation loss (unbiased by warmup ramp)
            if val_loss_clean < best_val_loss:
                best_val_loss = float(val_loss_clean)
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
            "val_losses_clean": val_losses_clean,
            "best_val_loss": best_val_loss,
            "pred_means": pred_means,
            "pred_pct_pos": pred_pct_pos,
            "pred_pct_neg": pred_pct_neg,
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
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(target_scale=target_scale, device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.huber_delta = huber_delta
        self.sign_penalty_weight = sign_penalty_weight
        self.d_model = num_layers * hidden_dim
        self.seq_output_dim = hidden_dim

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
        self.norm1 = nn.GroupNorm(1, num_filters)
        self.conv2 = nn.Conv1d(
            num_filters, num_filters, kernel_size=kernel_size, dilation=dilation, padding=0
        )
        self.norm2 = nn.GroupNorm(1, num_filters)
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
        out = self.relu(out)

        return out + residual


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
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
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
        self.seq_output_dim = hidden_dim

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
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
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
            nn.Linear(self.tabular_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.seq_proj = nn.LayerNorm(self.seq_dim)
        self.tab_proj = nn.LayerNorm(hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(self.seq_dim + hidden_dim // 2, hidden_dim),
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
        seq_emb = self.seq_proj(seq_emb)
        tab_emb = self.tab_proj(tab_emb)
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
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
        target_scale: float = 1.0,
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
            nn.Linear(self.tabular_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.seq_proj = nn.LayerNorm(self.seq_dim)
        self.tab_proj = nn.LayerNorm(hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(self.seq_dim + hidden_dim // 2, hidden_dim),
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
        seq_emb = self.seq_proj(seq_emb)
        tab_emb = self.tab_proj(tab_emb)
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
    ) -> dict:
        X_train = self._extract_features(market_windows_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, targets_train)
        self.is_fitted = True
        
        # Return training logs for compatibility with benchmark logging
        train_preds = self.model.predict(X_train_scaled).astype(np.float32)
        train_loss = float(np.mean((train_preds - targets_train) ** 2) ** 0.5)
        
        val_loss = np.nan
        if market_windows_val is not None and targets_val is not None:
            X_val = self._extract_features(market_windows_val)
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val_scaled).astype(np.float32)
            val_loss = float(np.mean((val_preds - targets_val) ** 2) ** 0.5)
        
        return {
            "train_losses": [train_loss],
            "val_losses": [val_loss],
            "best_val_loss": val_loss,
        }

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
    ) -> dict:
        X_train = extract_market_summary_features(market_windows_train)
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, targets_train)
        self.is_fitted = True
        
        # Return training logs for compatibility with benchmark logging
        train_preds = self.model.predict(X_train).astype(np.float32)
        train_loss = float(np.mean((train_preds - targets_train) ** 2) ** 0.5)
        
        val_loss = np.nan
        if market_windows_val is not None and targets_val is not None:
            X_val = extract_market_summary_features(market_windows_val)
            X_val = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val).astype(np.float32)
            val_loss = float(np.mean((val_preds - targets_val) ** 2) ** 0.5)
        
        return {
            "train_losses": [train_loss],
            "val_losses": [val_loss],
            "best_val_loss": val_loss,
        }

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
        target_scale: float = 1.0,
        huber_delta: float = 0.02,
        sign_penalty_weight: float = GLOBAL_LOSS_CONFIG.sign_penalty_weight,
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

        self.direction_epsilon = GLOBAL_LOSS_CONFIG.direction_epsilon
        self.direction_margin_fraction = GLOBAL_LOSS_CONFIG.direction_margin_fraction
        self.direction_min_margin = GLOBAL_LOSS_CONFIG.direction_min_margin
        self.direction_ramp_epochs = GLOBAL_LOSS_CONFIG.direction_ramp_epochs
        self.debug_logging = GLOBAL_LOSS_CONFIG.debug_logging
        self.variance_reg_weight = GLOBAL_LOSS_CONFIG.variance_reg_weight
        self.class_balance_dir = GLOBAL_LOSS_CONFIG.class_balance_dir

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

    def _compute_scheduled_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: int,
        warmup_epochs: int,
        huber_delta: float,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        return _scheduled_sign_aware_loss(
            pred,
            target,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            huber_delta=huber_delta,
            sign_penalty_weight=self.sign_penalty_weight,
            direction_epsilon=self.direction_epsilon,
            direction_margin_fraction=self.direction_margin_fraction,
            direction_min_margin=self.direction_min_margin,
            direction_ramp_epochs=self.direction_ramp_epochs,
            variance_reg_weight=self.variance_reg_weight,
            class_balance_dir=self.class_balance_dir,
            return_debug=return_debug,
        )

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
        **kwargs,
    ) -> dict:
        X_train = extract_market_summary_features(market_windows_train)
        X_val = extract_market_summary_features(market_windows_val)

        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)

        y_train = np.asarray(targets_train, dtype=np.float32) * self.target_scale
        y_val = np.asarray(targets_val, dtype=np.float32) * self.target_scale

        self.model = self._build_model(X_train.shape[1])

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        direction_epsilon = self.direction_epsilon
        active_ratio_train = float((torch.abs(y_train_t) > direction_epsilon).float().mean().item())
        active_ratio_val = float((torch.abs(y_val_t) > direction_epsilon).float().mean().item())

        logger.debug(
            "MLP-Summary direction_epsilon={:.6f} | active_ratio_train={:.4f} | active_ratio_val={:.4f}",
            direction_epsilon,
            active_ratio_train,
            active_ratio_val,
        )
        logger.debug(
            "MLP-Summary config | sign_penalty_weight={:.4f} | warmup_epochs={} | direction_ramp_epochs={} | "
            "target_scale={:.4f} | direction_margin_fraction={:.6f} | direction_min_margin={:.6f} | "
            "early_stopping_policy=scheduled_full_loss",
            self.sign_penalty_weight,
            warmup_epochs,
            self.direction_ramp_epochs,
            self.target_scale,
            self.direction_margin_fraction,
            self.direction_min_margin,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
        )

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
                loss = self._compute_scheduled_loss(
                    pred,
                    y_train_t[idx],
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    huber_delta=huber_delta,
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
                val_loss_obj, val_debug = self._compute_scheduled_loss(
                    pred_val,
                    y_val_t,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    huber_delta=huber_delta,
                    return_debug=True,
                )
                val_loss = float(val_loss_obj.item())

            if self.debug_logging:
                _log_epoch_debug("MLP-Summary", epoch, "val", val_debug)

            val_losses.append(val_loss)
            scheduler.step(val_loss)

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
