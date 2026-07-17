"""Shared training utilities for neural baseline models.

Eliminates duplicated training loops across LSTMPredictor, CNNLSTMPredictor, etc.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


def compute_huber_delta(scaled_targets: np.ndarray, floor: float = 0.01) -> float:
    """IQR-midpoint huber delta from scaled target distribution."""
    abs_targets = np.abs(scaled_targets)
    p25 = float(np.percentile(abs_targets, 25))
    p75 = float(np.percentile(abs_targets, 75))
    return max((p25 + p75) / 2.0, floor)


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_with_early_stopping(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    max_grad_norm: float = 1.0,
    model_name: str = "Model",
    warmup_epochs: int = 0,
    val_loss_fn=None,
) -> dict:
    """Generic training loop with early stopping and gradient clipping.

    Args:
        model: nn.Module to train.
        X_train, y_train: Training tensors (CPU or device tensors, y already scaled).
        X_val, y_val: Validation tensors (CPU or device tensors, y already scaled).
        loss_fn: Callable(pred, target, epoch=...) -> scalar loss tensor.
        optimizer: Pre-configured optimizer.
        scheduler: Optional LR scheduler (must have .step(val_loss)).
        epochs: Max epochs.
        batch_size: Mini-batch size.
        patience: Early stopping patience.
        max_grad_norm: Gradient clipping norm.
        model_name: For logging.
        warmup_epochs: Linear LR warmup from 10%→100% of base LR over this many epochs.
        val_loss_fn: Optional separate validation loss function. If None, uses loss_fn.
                     IMPORTANT: For early stopping to be unbiased, this should NOT gate
                     the direction penalty on epoch (i.e., always use full loss, not warmup-ramp).

    Returns:
        {
            "train_losses": [...], 
            "val_losses": [...], 
            "best_val_loss": float,
            "val_losses_clean": [...],  # Clean loss (for comparison across epochs)
            "pred_means": [...],         # Mean of val predictions per epoch
            "pred_pct_pos": [...],       # % of positive predictions per epoch
            "pred_pct_neg": [...],       # % of negative predictions per epoch
        }
    """
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    _val_loss_fn = val_loss_fn if val_loss_fn is not None else loss_fn

    n_train = len(X_train)
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_losses_clean: list[float] = []
    pred_means: list[float] = []
    pred_pct_pos: list[float] = []
    pred_pct_neg: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0

    device = getattr(model, "device", "cpu")

    for epoch in range(epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            scale = 0.1 + 0.9 * (epoch + 1) / warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * scale
        elif warmup_epochs > 0 and epoch == warmup_epochs:
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr

        model.train()
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i : i + batch_size]

            X_batch = X_train[batch_idx].to(device)
            y_batch = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val.to(device))
            
            # Full validation loss (may be gated by warmup ramp for logging/scheduler)
            val_loss = _val_loss_fn(pred_val, y_val.to(device), epoch=epoch).item()
            
            # Clean validation loss: call with epoch=999 to bypass warmup ramp.
            # This ensures early stopping compares apples-to-apples across epochs.
            # (Pass epoch=999 ensures _direction_ramp_factor returns 1.0 for all models)
            val_loss_clean = _val_loss_fn(pred_val, y_val.to(device), epoch=999).item()
            
        val_losses.append(val_loss)
        val_losses_clean.append(val_loss_clean)
        
        # Capture per-epoch prediction statistics
        pred_val_cpu = pred_val.cpu().numpy()
        pred_means.append(float(np.mean(pred_val_cpu)))
        pred_pct_pos.append(float(100.0 * np.mean(pred_val_cpu > 0)))
        pred_pct_neg.append(float(100.0 * np.mean(pred_val_cpu < 0)))

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            lr_str = f", lr={optimizer.param_groups[0]['lr']:.2e}" if scheduler else ""
            logger.debug(
                "{} epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, val_clean={:.4f}, "
                "pred_mean={:.6f} pos%={:.1f} neg%={:.1f}",
                model_name, epoch + 1, epochs, train_losses[-1], val_loss, val_loss_clean,
                pred_means[-1], pred_pct_pos[-1], pred_pct_neg[-1],
            )

        # Early stopping uses CLEAN validation loss (unbiased by warmup ramp)
        if val_loss_clean < best_val_loss:
            best_val_loss = val_loss_clean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("{} early stopping at epoch {}", model_name, epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_losses_clean": val_losses_clean,
        "best_val_loss": best_val_loss,
        "pred_means": pred_means,
        "pred_pct_pos": pred_pct_pos,
        "pred_pct_neg": pred_pct_neg,
    }