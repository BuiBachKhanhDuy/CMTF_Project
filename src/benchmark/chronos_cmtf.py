"""Chronos + Cross-Modal Temporal Fusion predictor.

Freezes the Chronos encoder and trains a lightweight fusion head that
merges market embeddings with aggregated news embeddings.

Architecture:
    1. **Gated cross-attention** (Chapter 4, Eq. 4.x):
       A  = CrossAttn(Q_num, K_text, V_text)
       g  = σ(W_g [Q_num ⊕ A])          ← learned gate
       H  = Q_num + g ⊙ A               ← numerical dominance (residual)
    2. **FiLM modulation** (Perez et al., AAAI 2018): news generates
       γ, β to scale/shift the gated representation.
    3. **GRN gating** (Lim et al., IJoF 2021): second gate that learns
       when to ignore news modulation entirely.

Numerical features are the residual backbone throughout — text only
modifies, never replaces.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from .chronos_market import ChronosMarketPredictor


class CrossModalFusionHead(nn.Module):
    """Gated Cross-Attention + FiLM + GRN fusion head.

    Three-stage fusion with **numerical dominance** — market features form
    the residual backbone and text only modifies via learned gates:

    1. **Gated Cross-Attention** (Chapter 4):
       - A  = MultiHeadAttn(Q=market, K=news, V=news)  with causal mask
       - g  = σ(W_g [market ⊕ A])                      learned gate
       - H  = market + g ⊙ A                            residual add
       Gate is initialised near zero so training starts at market-only.

    2. **FiLM modulation** (Perez et al., AAAI 2018): pooled news
       generates scale (γ) and shift (β) on the gated representation.

    3. **GRN gating** (Lim et al., IJoF 2021): final gate that blends
       FiLM-modulated and unmodulated representations.

    Dual-head output: regression + direction classification (BCE auxiliary).

    Attributes:
        last_gate_values: dict populated during forward() with gate
            statistics for interpretability logging.
    """

    def __init__(
        self,
        market_dim: int = 512,
        news_dim: int = 768,
        tabular_dim: int = 0,
        fusion_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.tabular_dim = tabular_dim
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.market_dim = market_dim
        self.market_proj = nn.Linear(market_dim, fusion_dim)
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim) if tabular_dim > 0 else None

        # Direct linear regression path: [market_emb, tabular] → scalar.
        # This is the LP-equivalent path — a single linear function of all
        # raw market features, with NO activation, NO dropout, NO bottleneck.
        # Ensures CMTF ≥ LP: even when the fusion path collapses, this path
        # carries the full market signal.  Final output = direct + fusion.
        direct_in = market_dim + (tabular_dim if tabular_dim > 0 else 0)
        self.direct_reg = nn.Linear(direct_in, 1, bias=True)

        # News compression: 768 → fusion_dim
        self.news_compress = nn.Sequential(
            nn.Linear(news_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # ============================================================
        # A1. Gated Cross-Attention
        # ============================================================
        # Q = market_proj(market_emb)  (B, 1, F)  ← single query
        # K, V = news_compress(news)   (B, S, F)  ← sequence of keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Learned gate:  g = σ(W_g [Q_num ⊕ A])
        self.cross_gate = nn.Linear(fusion_dim * 2, fusion_dim)

        # FiLM conditioning network: news_pool + density → γ, β
        # (Perez et al., AAAI 2018 "FiLM: Visual Reasoning with a
        #  General Conditioning Layer")
        self.film = nn.Sequential(
            nn.Linear(fusion_dim + 1, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.film_gamma = nn.Linear(fusion_dim, fusion_dim)
        self.film_beta = nn.Linear(fusion_dim, fusion_dim)

        # GRN gate: learns when news modulation helps vs market-only
        # (Lim et al., IJoF 2021 "Temporal Fusion Transformers")
        self.grn_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.grn_norm = nn.LayerNorm(fusion_dim)

        # FFN block
        ffn_hidden = fusion_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, fusion_dim),
        )
        self.ffn_norm = nn.LayerNorm(fusion_dim)

        # Nonlinear fusion regression head (adds on top of direct path)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        # Classification head (direction: up=1 / down=0)
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        # ---- Initialisation: start at market-only baseline ----
        # Cross-attention gate: init near zero so g≈0 → H ≈ Q_num
        nn.init.zeros_(self.cross_gate.weight)
        nn.init.constant_(self.cross_gate.bias, -2.0)  # σ(-2)≈0.12

        # FiLM: identity modulation γ=1, β=0 at start
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Fusion reg_head output layer → zero so model starts at direct path
        nn.init.zeros_(self.reg_head[-1].weight)
        nn.init.zeros_(self.reg_head[-1].bias)

        # Gate value logging for interpretability
        self.last_gate_values: dict[str, float] = {}

    def forward(
        self,
        market_emb: torch.Tensor,
        news_emb: torch.Tensor,
        tabular_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = market_emb.shape[0]

        # --- News aggregation: masked mean-pooling (for FiLM path) ---
        news_mask = (news_emb.abs().sum(-1) > 0)  # (B, S) bool
        news_count = news_mask.sum(-1, keepdim=True).clamp(min=1).float()
        density = news_count / news_emb.shape[1]  # (B, 1) ∈ [0, 1]

        news_compressed = self.news_compress(news_emb)  # (B, S, F)
        news_masked = news_compressed * news_mask.unsqueeze(-1).float()
        news_pool = news_masked.sum(1) / news_count  # (B, F)

        # --- Market projection ---
        market_h = self.market_proj(market_emb)  # (B, F)
        if self.tabular_proj is not None and tabular_emb is not None:
            market_h = market_h + self.tabular_proj(tabular_emb)

        # ============================================================
        # A1. Gated Cross-Attention
        # Q = market_h  (B, 1, F) — single numerical query
        # K, V = news_compressed (B, S, F) — text sequence
        # key_padding_mask: True = ignore (zero-padded news positions)
        # ============================================================
        query = market_h.unsqueeze(1)            # (B, 1, F)
        # Invert mask: True positions are IGNORED in nn.MultiheadAttention
        attn_key_mask = ~news_mask               # (B, S) True = pad
        # If all positions are padded, unmask everything to avoid NaN
        all_pad = attn_key_mask.all(dim=1)       # (B,)
        if all_pad.any():
            attn_key_mask = attn_key_mask.clone()
            attn_key_mask[all_pad] = False

        attn_out, _ = self.cross_attn(
            query, news_compressed, news_compressed,
            key_padding_mask=attn_key_mask,
        )                                         # (B, 1, F)
        attn_out = attn_out.squeeze(1)            # (B, F)

        # Learned gate: g = σ(W_g [Q_num ⊕ A])
        gate_input = torch.cat([market_h, attn_out], dim=-1)  # (B, 2F)
        cross_g = torch.sigmoid(self.cross_gate(gate_input))  # (B, F)

        # Numerical dominance: H = Q_num + g ⊙ A  (A2)
        market_h = market_h + cross_g * attn_out  # (B, F)

        # Log gate statistics for interpretability
        with torch.no_grad():
            self.last_gate_values["cross_gate_mean"] = float(cross_g.mean())
            self.last_gate_values["cross_gate_std"] = float(cross_g.std())

        # --- FiLM modulation: news scales/shifts the gated representation ---
        film_input = torch.cat([news_pool, density], dim=-1)  # (B, F+1)
        film_h = self.film(film_input)
        gamma = 1.0 + self.film_gamma(film_h)  # centred at 1
        beta = self.film_beta(film_h)
        modulated = gamma * market_h + beta  # (B, F)

        # --- GRN gating: residual from market-only ---
        grn_input = torch.cat([market_h, modulated], dim=-1)  # (B, 2F)
        grn_g = self.grn_gate(grn_input)  # (B, F) ∈ [0, 1]
        fused = self.grn_norm(grn_g * modulated + (1 - grn_g) * market_h)

        # Log GRN gate
        with torch.no_grad():
            self.last_gate_values["grn_gate_mean"] = float(grn_g.mean())

        # --- FFN + residual ---
        fused = self.ffn_norm(fused + self.ffn(fused))

        # --- Direct LP-like linear regression path ---
        # Single linear layer on raw [market_emb, tabular] — no activation,
        # no dropout, no bottleneck.  Equivalent to Ridge regression.
        if self.tabular_proj is not None and tabular_emb is not None:
            direct_in = torch.cat([market_emb, tabular_emb], dim=-1)
        else:
            direct_in = market_emb
        reg_direct = self.direct_reg(direct_in).squeeze(-1)  # (B,)

        # --- Additive combination: LP_linear + fusion_nonlinear ---
        reg_fused = self.reg_head(fused).squeeze(-1)  # (B,)
        reg_out = reg_direct + reg_fused  # fusion adds on top of LP

        cls_logit = self.cls_head(fused).squeeze(-1)  # (B,)
        return reg_out, cls_logit


class ChronosCMTFPredictor:
    """Chronos backbone (frozen) + trainable cross-modal fusion head."""

    def __init__(
        self,
        chronos_predictor: ChronosMarketPredictor,
        news_dim: int = 768,
        tabular_dim: int = 0,
        fusion_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_len: int = 30,
        bce_weight: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.chronos = chronos_predictor
        self.device = device
        self.bce_weight = bce_weight
        self.seq_len = seq_len
        # Target normalisation stats (set during fit)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._val_reg_median: float = 0.0
        self._pred_scale: float = 1.0  # variance calibration (EMOS)
        self.fusion = CrossModalFusionHead(
            market_dim=chronos_predictor.d_model,
            news_dim=news_dim,
            tabular_dim=tabular_dim,
            fusion_dim=fusion_dim,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
        ).to(device)

    # ------------------------------------------------------------------
    # Loss: CCC regression + BCE auxiliary for direction
    # ------------------------------------------------------------------
    def _combined_loss(
        self,
        pred_reg: torch.Tensor,
        pred_cls: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CCC regression + BCE direction loss.

        Uses Concordance Correlation Coefficient (Lin, 1989) instead of
        MSE to prevent prediction variance collapse.  CCC simultaneously
        optimises correlation *and* variance matching, so the model
        cannot minimise the loss by predicting near-constant values.

        CCC = 2·cov(ŷ, y) / (var(ŷ) + var(y) + (mean(ŷ) - mean(y))²)

        cls_head provides auxiliary directional gradient via BCE
        (Pei et al., ECAI 2025).
        """
        if weights is None:
            weights = torch.ones_like(target)

        # --- Regression: CCC loss (Lin, 1989) ---
        if len(pred_reg) >= 8:
            w = weights / weights.sum()  # normalise to sum=1
            mean_p = (w * pred_reg).sum()
            mean_t = (w * target).sum()
            var_p = (w * (pred_reg - mean_p) ** 2).sum()
            var_t = (w * (target - mean_t) ** 2).sum()
            cov_pt = (w * (pred_reg - mean_p) * (target - mean_t)).sum()
            ccc = 2.0 * cov_pt / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)
            loss_reg = 1.0 - ccc
        else:
            # Fallback to MSE for tiny last batches
            w = weights / weights.mean()
            loss_reg = (w * (pred_reg - target) ** 2).mean()

        # --- Direction: BCE auxiliary ---
        w_bce = weights / weights.mean()
        dir_label = (target > 0).float()
        loss_bce = nn.functional.binary_cross_entropy_with_logits(
            pred_cls, dir_label, weight=w_bce, reduction="mean",
        )

        return (1.0 - self.bce_weight) * loss_reg + self.bce_weight * loss_bce

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
        lr: float = 5e-4,
        patience: int = 25,
        batch_size: int = 32,
        seed: int = 42,
        precomputed_emb_train: np.ndarray | None = None,
        precomputed_emb_val: np.ndarray | None = None,
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
            precomputed_emb_train: (N_train, d_model) pre-computed Chronos
                embeddings. If provided, skip internal embedding computation.
            precomputed_emb_val: (N_val, d_model) pre-computed Chronos
                embeddings for validation set.

        Returns:
            Training history dict with 'train_loss' and 'val_loss'.
        """
        # Pin RNG before weight init + training for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Re-create fusion head from scratch for deterministic init
        self.fusion = CrossModalFusionHead(
            market_dim=self.chronos.d_model,
            news_dim=self.fusion.news_compress[0].in_features,
            tabular_dim=self.fusion.tabular_dim,
            fusion_dim=self.fusion.market_proj.out_features,
            n_heads=self.fusion.n_heads,
            dropout=self.fusion.reg_head[2].p,  # Dropout is at index 2 in reg_head
            seq_len=self.seq_len,
        ).to(self.device)

        if precomputed_emb_train is not None and precomputed_emb_val is not None:
            logger.info("Using pre-computed Chronos embeddings")
            emb_train = precomputed_emb_train
            emb_val = precomputed_emb_val
        else:
            logger.info("Computing Chronos embeddings (frozen) …")
            emb_train = self.chronos.get_embeddings(close_train)
            emb_val = self.chronos.get_embeddings(close_val)

        # ---- Target z-score normalisation ----
        self._y_mean = float(np.mean(y_train))
        self._y_std = float(np.std(y_train))
        if self._y_std < 1e-8:
            self._y_std = 1.0
        y_train_norm = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_val_norm = ((y_val - self._y_mean) / self._y_std).astype(np.float32)
        logger.info(
            "Target z-score: mean={:.6f}, std={:.6f}, up%={:.1f}%",
            self._y_mean, self._y_std, 100.0 * (y_train > 0).mean(),
        )

        train_m = torch.tensor(emb_train, dtype=torch.float32, device=self.device)
        train_n = torch.tensor(news_train, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y_train_norm, dtype=torch.float32, device=self.device)
        train_t = (
            torch.tensor(tabular_train, dtype=torch.float32, device=self.device)
            if tabular_train is not None
            else None
        )

        # Per-sample weight based on news *density* — the fraction of bars in
        # the lookback window that carry a real (non-zero) news embedding.
        # Weights range smoothly from 1× (no news bars) to 2× (all bars have news).
        news_bar_present = (np.abs(news_train).sum(axis=-1) > 0)  # (N, seq_len) bool
        news_density = news_bar_present.mean(axis=1).astype(np.float32)  # (N,)
        sample_weights = (1.0 + news_density).astype(np.float32)
        logger.info(
            "News-density weighting: density min={:.3f} mean={:.3f} max={:.3f}",
            float(news_density.min()), float(news_density.mean()), float(news_density.max()),
        )
        train_w = torch.tensor(sample_weights, dtype=torch.float32, device=self.device)

        val_m = torch.tensor(emb_val, dtype=torch.float32, device=self.device)
        val_n = torch.tensor(news_val, dtype=torch.float32, device=self.device)
        val_y = torch.tensor(y_val_norm, dtype=torch.float32, device=self.device)
        val_t = (
            torch.tensor(tabular_val, dtype=torch.float32, device=self.device)
            if tabular_val is not None
            else None
        )

        # Val-set news density weights (consistent with train weighting)
        val_news_present = (np.abs(news_val).sum(axis=-1) > 0)  # (N_val, seq_len)
        val_density = val_news_present.mean(axis=1).astype(np.float32)
        val_w = torch.tensor(
            (1.0 + val_density).astype(np.float32),
            dtype=torch.float32, device=self.device,
        )

        # Mini-batch DataLoader (weights always included as last element)
        if train_t is None:
            train_ds = TensorDataset(train_m, train_n, train_y, train_w)
        else:
            train_ds = TensorDataset(train_m, train_n, train_t, train_y, train_w)
        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
            generator=loader_gen,
        )

        optimizer = torch.optim.AdamW(self.fusion.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

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
                    mb_m, mb_n, mb_y, mb_w = batch
                    mb_t = None
                else:
                    mb_m, mb_n, mb_t, mb_y, mb_w = batch
                optimizer.zero_grad()
                pred_reg, pred_cls = self.fusion(mb_m, mb_n, mb_t)
                loss = self._combined_loss(pred_reg, pred_cls, mb_y, weights=mb_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            # --- Validate ---
            self.fusion.eval()
            with torch.no_grad():
                val_reg, val_cls = self.fusion(val_m, val_n, val_t)
                val_loss = self._combined_loss(val_reg, val_cls, val_y, weights=val_w).item()

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

            scheduler.step()

        if best_state is not None:
            self.fusion.load_state_dict(best_state)

        # ---- Compute val-based centering offset ----
        self.fusion.eval()
        with torch.no_grad():
            val_reg_out, _ = self.fusion(val_m, val_n, val_t)
            val_preds = val_reg_out.cpu().numpy() * self._y_std + self._y_mean
            self._val_reg_median = float(np.median(val_preds))
            val_centred = val_preds - self._val_reg_median
            val_y_np = y_val
            nonzero = val_y_np != 0
            val_da = 100.0 * np.mean(np.sign(val_y_np[nonzero]) == np.sign(val_centred[nonzero])) if nonzero.any() else 0.0

        # ---- Variance calibration (EMOS-style, Gneiting et al. 2005) ----
        # Scale predictions to match actual return variance on val set.
        pred_std = float(np.std(val_centred))
        actual_std = float(np.std(y_val))
        self._pred_scale = float(np.clip(
            actual_std / max(pred_std, 1e-8), 0.5, 15.0,
        ))
        logger.info(
            "CMTF val DA%: {:.1f}% (centering={:.6f}, pred_std={:.6f}, actual_std={:.6f}, scale={:.2f})",
            val_da, self._val_reg_median, pred_std, actual_std, self._pred_scale,
        )

        logger.info("CMTF fusion training done | best val loss = {:.6f}", best_val_loss)
        return history

    def predict(
        self,
        close_test: np.ndarray,
        news_test: np.ndarray,
        tabular_test: np.ndarray | None = None,
        precomputed_emb_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict returns using frozen Chronos embeddings + trained fusion head.

        Args:
            close_test: (N_test, seq_len) raw close windows.
            news_test:  (N_test, seq_len, 768) per-bar news embeddings.
            tabular_test: (N_test, F_tab) optional engineered market features.
            precomputed_emb_test: (N_test, d_model) pre-computed Chronos
                embeddings. If provided, skip internal embedding computation.

        Returns:
            (N_test,) predicted returns.
        """
        if precomputed_emb_test is not None:
            emb_test = precomputed_emb_test
        else:
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
            reg_out, _ = self.fusion(test_m, test_n, test_t)
            # Denormalise from z-score space
            preds = reg_out.cpu().numpy() * self._y_std + self._y_mean
            # Zero-centre using *validation* prediction median, which
            # preserves any directional bias learned during training.
            # (Previous approach used test median, which forced a 50/50
            # up/down split and destroyed DA% when the test period had
            # a bullish or bearish trend.)
            centred = preds - self._val_reg_median
            # Variance calibration (EMOS-style): scale to match actual
            # return distribution observed on validation set.
            scaled = centred * self._pred_scale
            n_pos = int(np.sum(scaled > 0))
            n_neg = int(np.sum(scaled < 0))
            logger.info(
                "CMTF predict: val_median={:.6f}, scale={:.2f} → {}+/{}− predictions",
                self._val_reg_median, self._pred_scale, n_pos, n_neg,
            )
            return scaled

    # ------------------------------------------------------------------
    # Checkpoint save / load (includes normalisation scalars)
    # ------------------------------------------------------------------
    def get_checkpoint(self) -> dict:
        """Return a dict containing model weights + normalisation params."""
        return {
            "state_dict": self.fusion.state_dict(),
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "val_reg_median": self._val_reg_median,
            "pred_scale": self._pred_scale,
        }

    def load_checkpoint(self, ckpt: dict | None) -> None:
        """Load model weights + normalisation params from a checkpoint dict.

        Also supports legacy checkpoints that are plain state_dicts.
        """
        if ckpt is None:
            return
        if "state_dict" in ckpt:
            self.fusion.load_state_dict(ckpt["state_dict"])
            self._y_mean = float(ckpt.get("y_mean", 0.0))
            self._y_std = float(ckpt.get("y_std", 1.0))
            self._val_reg_median = float(ckpt.get("val_reg_median", 0.0))
            self._pred_scale = float(ckpt.get("pred_scale", 1.0))
        else:
            # Legacy: plain state_dict without normalisation params
            self.fusion.load_state_dict(ckpt)
            self._y_mean = 0.0
            self._y_std = 1.0
            self._val_reg_median = 0.0
