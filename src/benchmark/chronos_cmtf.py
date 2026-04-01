"""Chronos + Cross-Modal Temporal Fusion predictor.

Freezes the Chronos encoder and trains a lightweight cross-attention
fusion head that merges market embeddings with a *sequence* of news
embeddings (one per bar in the look-back window).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from .chronos_market import ChronosMarketPredictor


class CrossModalFusionHead(nn.Module):
    """Cross-attention fusion: market query attends to a news *sequence*.

    Market embedding (B, market_dim) is projected once as the query.
    News embeddings (B, seq_len, news_dim) form the key/value sequence,
    giving the attention layer multiple positions to attend over.
    A learned ``news_default`` token replaces all-zero news rows so the
    model can distinguish "no news" from "neutral news".
    """

    def __init__(
        self,
        market_dim: int = 512,
        news_dim: int = 768,
        tabular_dim: int = 0,
        fusion_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tabular_dim = tabular_dim
        self.market_proj = nn.Linear(market_dim, fusion_dim)
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim) if tabular_dim > 0 else None
        self.news_proj = nn.Linear(news_dim, fusion_dim)

        # Learned replacement for missing-news positions
        self.news_default = nn.Parameter(torch.randn(1, 1, news_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=n_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        market_emb: torch.Tensor,
        news_emb: torch.Tensor,
        tabular_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            market_emb: (B, market_dim) — Chronos encoder output.
            news_emb:   (B, seq_len, news_dim) — per-bar news embeddings.
            tabular_emb: (B, tabular_dim) optional engineered market features.

        Returns:
            (B,) predicted return.
        """
        # Replace all-zero news positions with the learned default token
        zero_mask = (news_emb.abs().sum(dim=-1, keepdim=True) == 0)  # (B, S, 1)
        news_filled = torch.where(zero_mask, self.news_default, news_emb)

        market_q = self.market_proj(market_emb)
        if self.tabular_proj is not None and tabular_emb is not None:
            market_q = market_q + self.tabular_proj(tabular_emb)

        q = market_q.unsqueeze(1)                         # (B, 1, F)
        kv = self.news_proj(news_filled)                  # (B, S, F)

        attn_out, _ = self.cross_attn(q, kv, kv)         # (B, 1, F)
        fused = self.norm(attn_out.squeeze(1) + q.squeeze(1))  # residual

        return self.head(fused).squeeze(-1)                # (B,)


class ChronosCMTFPredictor:
    """Chronos backbone (frozen) + trainable cross-modal fusion head."""

    def __init__(
        self,
        chronos_predictor: ChronosMarketPredictor,
        news_dim: int = 768,
        tabular_dim: int = 0,
        fusion_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        self.chronos = chronos_predictor
        self.device = device
        self.fusion = CrossModalFusionHead(
            market_dim=chronos_predictor.d_model,
            news_dim=news_dim,
            tabular_dim=tabular_dim,
            fusion_dim=fusion_dim,
        ).to(device)

    # ------------------------------------------------------------------
    # Loss that combines MSE with a directional (sign) penalty
    # ------------------------------------------------------------------
    @staticmethod
    def _combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        """MSE + sign-agreement penalty.

        ``alpha`` controls the weight of the directional term.
        """
        mse = nn.functional.mse_loss(pred, target)
        # Soft sign agreement: penalise when pred and target disagree
        sign_agree = (pred * target).clamp(min=-1.0)  # positive when same sign
        dir_loss = -sign_agree.mean()  # lower is better
        return mse + alpha * dir_loss

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
        epochs: int = 80,
        lr: float = 1e-3,
        patience: int = 15,
        batch_size: int = 64,
    ) -> dict[str, list[float]]:
        """Train the fusion head with mini-batch SGD.

        Args:
            close_train: (N_train, seq_len) raw close windows.
            news_train:  (N_train, seq_len, 768) per-bar news embeddings.
            y_train:     (N_train,) target returns.
            close_val / news_val / y_val: validation data.
            tabular_train / tabular_val: optional engineered market features.
            epochs: Maximum training epochs.
            lr: Learning rate.
            patience: Early-stopping patience.
            batch_size: Mini-batch size.

        Returns:
            Training history dict with 'train_loss' and 'val_loss'.
        """
        logger.info("Computing Chronos embeddings (frozen) …")
        emb_train = self.chronos.get_embeddings(close_train)
        emb_val = self.chronos.get_embeddings(close_val)

        train_m = torch.tensor(emb_train, dtype=torch.float32, device=self.device)
        train_n = torch.tensor(news_train, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        train_t = (
            torch.tensor(tabular_train, dtype=torch.float32, device=self.device)
            if tabular_train is not None
            else None
        )

        val_m = torch.tensor(emb_val, dtype=torch.float32, device=self.device)
        val_n = torch.tensor(news_val, dtype=torch.float32, device=self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        val_t = (
            torch.tensor(tabular_val, dtype=torch.float32, device=self.device)
            if tabular_val is not None
            else None
        )

        # Mini-batch DataLoader
        if train_t is None:
            train_ds = TensorDataset(train_m, train_n, train_y)
        else:
            train_ds = TensorDataset(train_m, train_n, train_t, train_y)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        )

        optimizer = torch.optim.AdamW(self.fusion.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # --- Train (mini-batch) ---
            self.fusion.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                if train_t is None:
                    mb_m, mb_n, mb_y = batch
                    mb_t = None
                else:
                    mb_m, mb_n, mb_t, mb_y = batch
                optimizer.zero_grad()
                pred = self.fusion(mb_m, mb_n, mb_t)
                loss = self._combined_loss(pred, mb_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            # --- Validate ---
            self.fusion.eval()
            with torch.no_grad():
                val_pred = self.fusion(val_m, val_n, val_t)
                val_loss = self._combined_loss(val_pred, val_y).item()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.fusion.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch {}/{} | train={:.6f} | val={:.6f}",
                    epoch + 1, epochs, avg_train, val_loss,
                )

            if patience_counter >= patience:
                logger.info("Early stopping at epoch {}", epoch + 1)
                break

        if best_state is not None:
            self.fusion.load_state_dict(best_state)
        logger.info("CMTF fusion training done | best val loss = {:.6f}", best_val_loss)
        return history

    def predict(
        self,
        close_test: np.ndarray,
        news_test: np.ndarray,
        tabular_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict returns using frozen Chronos embeddings + trained fusion head.

        Args:
            close_test: (N_test, seq_len) raw close windows.
            news_test:  (N_test, seq_len, 768) per-bar news embeddings.
            tabular_test: (N_test, F_tab) optional engineered market features.

        Returns:
            (N_test,) predicted returns.
        """
        emb_test = self.chronos.get_embeddings(close_test)

        test_m = torch.tensor(emb_test, dtype=torch.float32, device=self.device)
        test_n = torch.tensor(news_test, dtype=torch.float32, device=self.device)
        test_t = (
            torch.tensor(tabular_test, dtype=torch.float32, device=self.device)
            if tabular_test is not None
            else None
        )

        self.fusion.eval()
        with torch.no_grad():
            return self.fusion(test_m, test_n, test_t).cpu().numpy()
