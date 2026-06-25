"""fusion_wrappers.py

Early and Late fusion strategies.

Deep refactor summary
---------------------
EarlyFusionWrapper
  - NewsProjector (trainable nn.Module) replaces fixed random matrix.
  - Projector is registered as self.news_projector so its parameters
    are part of the encoder's optimizer graph automatically.
  - Null news embedding (nn.Parameter) replaces zero-vector padding:
    predict_without_news now broadcasts the null token across the
    sequence so "no news" inputs are in-distribution at inference time.

LateFusionWrapper
  - NewsProjector owned here; trained jointly with NewsBranchPredictor.
  - Mask is always computed on the *projected* embeddings, never on raw,
    eliminating the previous mask/representation mismatch.
  - NewsBranchPredictor rebuilt on AttentionPoolingNewsEncoder:
    mean pooling removed, learned_alpha removed.
  - OOF generation preserved and corrected: residual_train is computed
    from oof_preds; market context fed to the branch during training is
    also the OOF prediction, keeping them consistent.
  - Validation batched; no more full-dataset GPU materialisation.
  - NewsProjector gradients flow through fit_news_branch because
    projected tensors are re-computed inside the training loop via
    the nn.Module forward pass, not pre-baked into numpy.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from .training_utils import compute_huber_delta
from .baseline_models import sign_aware_huber_loss
from .news_module import (
    STANDARD_NEWS_DIM,
    NewsProjector,
    NewsBranchPredictor,
    _as_bool_mask,
    _to_numpy_float32,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _require_attr(obj, attr: str, ctx: str = "") -> None:
    if not hasattr(obj, attr):
        prefix = f"{ctx}: " if ctx else ""
        raise AttributeError(
            f"{prefix}required attribute '{attr}' not found on {type(obj).__name__}"
        )


# ---------------------------------------------------------------------------
# OOF market prediction generator
# ---------------------------------------------------------------------------

def generate_oof_market_predictions(
    base_encoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    gap: int = 0,
    **fit_kwargs,
) -> np.ndarray:
    """Generate out-of-fold market predictions to avoid leakage in residual training.

    Uses TimeSeriesSplit exclusively; KFold is not valid for time-series.
    """
    _require_attr(base_encoder, "fit", "generate_oof_market_predictions")
    _require_attr(base_encoder, "predict_market_only", "generate_oof_market_predictions")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    logger.info("Generating OOF market predictions ({} splits)", n_splits)
    oof_preds = np.zeros(len(y_train), dtype=np.float32)
    initial_state = copy.deepcopy(base_encoder)

    cv = TimeSeriesSplit(n_splits=n_splits, gap=max(gap, 0))
    for fold, (tr_idx, v_idx) in enumerate(cv.split(X_train), start=1):
        logger.info("  OOF fold {}/{}", fold, n_splits)
        fold_model = copy.deepcopy(initial_state)
        fold_model.fit(
            X_train[tr_idx], y_train[tr_idx],
            X_train[v_idx], y_train[v_idx],
            **fit_kwargs,
        )
        oof_preds[v_idx] = _to_numpy_float32(fold_model.predict_market_only(X_train[v_idx]))

    logger.info("OOF generation complete")
    return oof_preds


# ---------------------------------------------------------------------------
# Early Fusion Wrapper
# ---------------------------------------------------------------------------

class EarlyFusionWrapper(nn.Module):
    """Concatenate projected news embeddings to market windows before encoder input.

    The NewsProjector is an nn.Module registered as self.news_projector.
    Because the encoder is not always an nn.Module (e.g. sklearn wrappers
    are accepted for compatibility) we cannot always register it as a
    submodule.  Instead, any optimizer passed to fit() should be built after
    construction so it can see self.news_projector.parameters() explicitly.

    Zero-news at inference time uses a learned null_news_embedding
    (nn.Parameter of shape (1, 1, projected_news_dim)) broadcast across the
    sequence, keeping "no news" inputs in-distribution.

    Parameters
    ----------
    encoder_cls:
        Callable that constructs the market encoder.
    encoder_kwargs:
        Keyword arguments forwarded to encoder_cls.  input_dim will be
        expanded by projected_news_dim automatically.
    raw_news_dim:
        Dimensionality of raw news embeddings fed in by the caller.
    projected_news_dim:
        Target dim of the trainable projector.
    projector_dropout:
        Dropout inside the NewsProjector.
    """

    def __init__(
        self,
        encoder_cls,
        encoder_kwargs: dict,
        raw_news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        projector_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.raw_news_dim = int(raw_news_dim)
        self.projected_news_dim = int(projected_news_dim)

        self.news_projector = NewsProjector(
            input_dim=self.raw_news_dim,
            output_dim=self.projected_news_dim,
            dropout=projector_dropout,
        )

        # Learned null news embedding: in-distribution "no news" signal
        self.null_news_embedding = nn.Parameter(
            torch.zeros(1, 1, self.projected_news_dim)
        )
        nn.init.normal_(self.null_news_embedding, std=0.01)

        self._original_input_dim = int(encoder_kwargs.get("input_dim", 0))
        expanded_kwargs = {
            **encoder_kwargs,
            "input_dim": self._original_input_dim + self.projected_news_dim,
        }
        self.encoder = encoder_cls(**expanded_kwargs)

    # ------------------------------------------------------------------
    # Properties forwarded from encoder
    # ------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        return getattr(self.encoder, "d_model", 0)

    @property
    def supports_sequence(self) -> bool:
        return getattr(self.encoder, "supports_sequence", True)

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _project_news_tensor(self, news_embs: torch.Tensor) -> torch.Tensor:
        """Project news through the trainable projector (gradient-aware)."""
        if news_embs.shape[-1] == self.projected_news_dim:
            return news_embs
        if news_embs.shape[-1] == self.raw_news_dim:
            return self.news_projector(news_embs)
        raise ValueError(
            f"News last dim must be {self.raw_news_dim} or {self.projected_news_dim}, "
            f"got {news_embs.shape[-1]}"
        )

    def _concat_inputs_numpy(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
    ) -> np.ndarray:
        """Numpy path for fit/predict: project then concatenate."""
        market_windows = np.asarray(market_windows, dtype=np.float32)
        if market_windows.ndim == 2:
            market_windows = market_windows[:, :, None]
        if market_windows.ndim != 3:
            raise ValueError(f"Expected (N, S, F), got {market_windows.shape}")

        news_proj = self.news_projector.ensure_projected(news_embs)

        if market_windows.shape[:2] != news_proj.shape[:2]:
            raise ValueError(
                f"Market/news shape mismatch: {market_windows.shape} vs {news_proj.shape}"
            )
        return np.concatenate([market_windows, news_proj], axis=-1).astype(np.float32)

    def _forward_torch(
        self,
        market_windows: torch.Tensor,
        news_embs: torch.Tensor,
    ) -> torch.Tensor:
        if market_windows.ndim == 2:
            market_windows = market_windows.unsqueeze(-1)
        if market_windows.ndim != 3:
            raise ValueError(f"Expected market tensor (B, S, F), got {tuple(market_windows.shape)}")

        if news_embs.ndim != 3:
            raise ValueError(f"Expected news tensor (B, S, D), got {tuple(news_embs.shape)}")

        if news_embs.shape[-1] == self.raw_news_dim:
            news_proj = self.news_projector(news_embs)
        elif news_embs.shape[-1] == self.projected_news_dim:
            news_proj = news_embs
        else:
            raise ValueError(
                f"News last dim must be {self.raw_news_dim} or {self.projected_news_dim}, "
                f"got {news_embs.shape[-1]}"
            )

        if market_windows.shape[:2] != news_proj.shape[:2]:
            raise ValueError(
                f"Market/news shape mismatch: {tuple(market_windows.shape)} vs {tuple(news_proj.shape)}"
            )

        x = torch.cat([market_windows, news_proj], dim=-1)
        return self.encoder.forward(x)
    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        market_train: np.ndarray,
        news_train: np.ndarray,
        targets_train: np.ndarray,
        market_val: np.ndarray,
        news_val: np.ndarray,
        targets_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
        warmup_epochs: int = 0,
        **kwargs,
    ) -> dict:
        if not isinstance(self.encoder, nn.Module):
            raise TypeError("EarlyFusionWrapper currently supports only torch encoders.")

        device = getattr(self.encoder, "device", "cpu")

        X_m_tr = torch.as_tensor(market_train, dtype=torch.float32, device=device)
        X_n_tr = torch.as_tensor(news_train, dtype=torch.float32, device=device)
        y_tr = torch.as_tensor(targets_train, dtype=torch.float32, device=device) * self.encoder.target_scale

        X_m_v = torch.as_tensor(market_val, dtype=torch.float32, device=device)
        X_n_v = torch.as_tensor(news_val, dtype=torch.float32, device=device)
        y_v = torch.as_tensor(targets_val, dtype=torch.float32, device=device) * self.encoder.target_scale

        optimizer = torch.optim.AdamW(
            list(self.news_projector.parameters()) + list(self.encoder.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        huber_delta = compute_huber_delta(y_tr.detach().cpu().numpy())

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_train = len(X_m_tr)

        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                pred = self._forward_torch(X_m_tr[idx], X_n_tr[idx]).squeeze(-1)
                loss = sign_aware_huber_loss(
                    pred,
                    y_tr[idx],
                    huber_delta=huber_delta,
                    sign_penalty_weight=getattr(self.encoder, "sign_penalty_weight", 0.05),
                    direction_epsilon=0.5,
                    enable_direction_loss=(epoch >= warmup_epochs),
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.news_projector.parameters()) + list(self.encoder.parameters()),
                    1.0,
                )
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            self.eval()
            with torch.no_grad():
                pred_v = self._forward_torch(X_m_v, X_n_v).squeeze(-1)
                val_loss = sign_aware_huber_loss(
                    pred_v,
                    y_v,
                    huber_delta=huber_delta,
                    sign_penalty_weight=getattr(self.encoder, "sign_penalty_weight", 0.05),
                    direction_epsilon=0.5,
                    enable_direction_loss=(epoch >= warmup_epochs),
                ).item()

            val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = _clone_state(self)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("EarlyFusion early stop at epoch {}", epoch + 1)
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
        news_embs: np.ndarray,
    ) -> np.ndarray:
        X = self._concat_inputs_numpy(market_windows, news_embs)
        return self.encoder.predict(X)

    def encode(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
    ) -> np.ndarray:
        _require_attr(self.encoder, "encode", "EarlyFusionWrapper.encode")
        X = self._concat_inputs_numpy(market_windows, news_embs)
        return self.encoder.encode(X)

    def predict_without_news(self, market_windows: np.ndarray) -> np.ndarray:
        """Predict using the learned null news embedding instead of zero vectors.

        Broadcasts null_news_embedding across the sequence length so that
        "no news" is represented by a consistent in-distribution token
        rather than an out-of-distribution zero vector.
        """
        market_windows = np.asarray(market_windows, dtype=np.float32)
        if market_windows.ndim == 2:
            seq_len = market_windows.shape[1]
        elif market_windows.ndim == 3:
            seq_len = market_windows.shape[1]
        else:
            raise ValueError(f"Unexpected market_windows shape: {market_windows.shape}")

        B = market_windows.shape[0]
        # Detach: null embedding is an inference-time constant here
        null_proj = (
            self.null_news_embedding
            .detach()
            .expand(B, seq_len, -1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return self.predict(market_windows, null_proj)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.predict_without_news(market_windows)


# ---------------------------------------------------------------------------
# Late Fusion Wrapper
# ---------------------------------------------------------------------------

class LateFusionWrapper(nn.Module):
    """Train market encoder first, then train a news residual branch.

    Deep refactor changes
    ---------------------
    - NewsProjector (trainable nn.Module) replaces the random matrix.
      It is trained jointly with NewsBranchPredictor in fit_news_branch,
      because projected tensors are recomputed inside the training loop
      via a differentiable forward pass rather than pre-baked into numpy.
    - Mask computed on *projected* embeddings: _as_bool_mask is called
      after projection, eliminating the mask/representation mismatch.
    - Mean pooling removed (lives in NewsBranchPredictor →
      AttentionPoolingNewsEncoder).
    - learned_alpha removed; branch confidence is implicit.
    - Validation batched: no full-dataset GPU materialisation.
    - OOF predictions are consistent with the residual targets: both
      residual_train and the market context fed to the branch during
      training are derived from the same OOF predictions.

    Parameters
    ----------
    encoder:
        Pre-constructed market-only encoder.
    raw_news_dim:
        Dimensionality of raw news embeddings.
    projected_news_dim:
        Target dim of the trainable projector.
    seq_len:
        News sequence length; passed to NewsBranchPredictor.
    fusion_dim:
        Internal dim for the attention encoder and residual MLP.
    device:
        Torch device string.
    horizon:
        Forecast horizon; used as OOF gap.
    target_scale:
        Scale applied to targets before training the news branch.
    freeze_encoder:
        If True, the market encoder is not updated after initial fit.
    projector_dropout:
        Dropout inside NewsProjector.
    branch_dropout:
        Dropout inside NewsBranchPredictor.
    """

    def __init__(
        self,
        encoder,
        raw_news_dim: int = 768,
        projected_news_dim: int = STANDARD_NEWS_DIM,
        seq_len: int = 30,
        fusion_dim: int = 128,
        device: str = "cpu",
        horizon: int = 1,
        target_scale: float = 100.0,
        freeze_encoder: bool = False,
        projector_dropout: float = 0.1,
        branch_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.raw_news_dim = int(raw_news_dim)
        self.projected_news_dim = int(projected_news_dim)
        self.device = device
        self.horizon = int(horizon)
        self.target_scale = float(target_scale)
        self.freeze_encoder = freeze_encoder

        # Trainable projector — registered as submodule so checkpointing works
        self.news_projector = NewsProjector(
            input_dim=self.raw_news_dim,
            output_dim=self.projected_news_dim,
            dropout=projector_dropout,
        ).to(self.device)

        # News residual branch
        self.news_branch = NewsBranchPredictor(
            news_dim=self.projected_news_dim,
            fusion_dim=fusion_dim,
            seq_len=seq_len,
            dropout=branch_dropout,
        ).to(self.device)

        self._is_news_fitted = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        return getattr(self.encoder, "d_model", 0)

    @property
    def supports_sequence(self) -> bool:
        return getattr(self.encoder, "supports_sequence", True)

    def news_parameters(self):
        """All parameters that belong to the news branch (projector + branch)."""
        return list(self.news_projector.parameters()) + list(self.news_branch.parameters())

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

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
        epochs_news: int = 30,
        batch_size_news: int = 32,
        lr_news: float = 1e-3,
        patience_news: int = 8,
        **encoder_fit_kwargs,
    ) -> dict:
        _require_attr(self.encoder, "fit", "LateFusionWrapper.fit")
        _require_attr(self.encoder, "predict_market_only", "LateFusionWrapper.fit")

        # Phase 1: OOF market predictions to build unbiased residual targets
        logger.info("LateFusion phase 1: generating OOF market predictions")
        oof_preds_train = generate_oof_market_predictions(
            base_encoder=self.encoder,
            X_train=market_train,
            y_train=targets_train,
            n_splits=n_splits,
            gap=max(self.horizon, 1),
            **encoder_fit_kwargs,
        )

        # Phase 2: fit the main market encoder on full training data
        logger.info("LateFusion phase 2: fitting main market encoder")
        self.encoder.fit(
            market_train, targets_train,
            market_val, targets_val,
            **encoder_fit_kwargs,
        )
        market_preds_val = _to_numpy_float32(self.encoder.predict_market_only(market_val))

        # Phase 3: train news projector + branch jointly
        logger.info("LateFusion phase 3: fitting news branch + projector")
        history = self._fit_news_branch(
            news_train_raw=news_train,
            targets_train=targets_train,
            news_val_raw=news_val,
            targets_val=targets_val,
            news_mask_train=news_mask_train,
            news_mask_val=news_mask_val,
            oof_preds_train=oof_preds_train,
            market_preds_val=market_preds_val,
            epochs=epochs_news,
            batch_size=batch_size_news,
            lr=lr_news,
            patience=patience_news,
        )
        logger.info("LateFusion training complete")
        return history

    def _fit_news_branch(
        self,
        news_train_raw: np.ndarray,
        targets_train: np.ndarray,
        news_val_raw: np.ndarray,
        targets_val: np.ndarray,
        news_mask_train: np.ndarray | None,
        news_mask_val: np.ndarray | None,
        oof_preds_train: np.ndarray,
        market_preds_val: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
    ) -> dict:
        # Residual targets (what the news branch must learn to predict)
        residual_train = (
            _to_numpy_float32(targets_train) - _to_numpy_float32(oof_preds_train)
        ).astype(np.float32)
        residual_val = (
            _to_numpy_float32(targets_val) - _to_numpy_float32(market_preds_val)
        ).astype(np.float32)

        # Project news: mask is derived from projected embeddings to avoid mismatch
        # NOTE: we project to numpy here only to build the mask and the DataLoader.
        # Inside the training loop we re-project via the nn.Module forward pass
        # so that gradients flow through the projector.
        with torch.no_grad():
            news_train_proj = self.news_projector.ensure_projected(
                np.asarray(news_train_raw, dtype=np.float32)
            )
            news_val_proj = self.news_projector.ensure_projected(
                np.asarray(news_val_raw, dtype=np.float32)
            )

        # Masks derived from projected representations
        mask_train = _as_bool_mask(news_mask_train, news_train_proj).astype(bool)
        mask_val = _as_bool_mask(news_mask_val, news_val_proj).astype(bool)

        # Determine whether raw or projected arrays should go into the DataLoader.
        # We store raw if input was raw (so the projector can be trained), else projected.
        raw_dim = news_train_raw.shape[-1]
        store_raw = (raw_dim == self.raw_news_dim)
        news_for_loader_train = np.asarray(news_train_raw, dtype=np.float32) if store_raw else news_train_proj
        news_for_loader_val = np.asarray(news_val_raw, dtype=np.float32) if store_raw else news_val_proj

        def _make_loader(news_np, mask_np, residual_np, market_np, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.as_tensor(news_np, dtype=torch.float32),
                torch.as_tensor(mask_np, dtype=torch.bool),
                torch.as_tensor(market_np * self.target_scale, dtype=torch.float32),
                torch.as_tensor(residual_np * self.target_scale, dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        loader_tr = _make_loader(
            news_for_loader_train, mask_train, residual_train, oof_preds_train, shuffle=True
        )
        loader_v = _make_loader(
            news_for_loader_val, mask_val, residual_val, market_preds_val, shuffle=False
        )

        # Single optimizer covering projector + branch jointly
        optimizer = torch.optim.AdamW(self.news_parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(int(epochs)):
            # --- training ---
            self.news_projector.train()
            self.news_branch.train()
            epoch_loss, n_batches = 0.0, 0

            for mb_news_raw, mb_mask, mb_market, mb_y in loader_tr:
                mb_news_raw = mb_news_raw.to(self.device)
                mb_mask = mb_mask.to(self.device)
                mb_market = mb_market.to(self.device)
                mb_y = mb_y.to(self.device)

                # Project inside the loop so projector gradients are live
                if store_raw:
                    mb_news_proj = self.news_projector(mb_news_raw)
                else:
                    mb_news_proj = mb_news_raw  # already projected, projector not updated

                optimizer.zero_grad()
                pred = self.news_branch(mb_news_proj, mb_mask, market_pred=mb_market)
                loss = nn.functional.huber_loss(pred, mb_y, delta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.news_parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))

            # --- validation (batched) ---
            self.news_projector.eval()
            self.news_branch.eval()
            val_loss_accum, v_batches = 0.0, 0

            with torch.no_grad():
                for mb_news_raw_v, mb_mask_v, mb_market_v, mb_y_v in loader_v:
                    mb_news_raw_v = mb_news_raw_v.to(self.device)
                    mb_mask_v = mb_mask_v.to(self.device)
                    mb_market_v = mb_market_v.to(self.device)
                    mb_y_v = mb_y_v.to(self.device)

                    if store_raw:
                        mb_news_proj_v = self.news_projector(mb_news_raw_v)
                    else:
                        mb_news_proj_v = mb_news_raw_v

                    val_pred = self.news_branch(mb_news_proj_v, mb_mask_v, market_pred=mb_market_v)
                    v_loss = nn.functional.huber_loss(val_pred, mb_y_v, delta=1.0)
                    val_loss_accum += float(v_loss.item())
                    v_batches += 1

            val_loss = val_loss_accum / max(v_batches, 1)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "projector": _clone_state(self.news_projector),
                    "branch": _clone_state(self.news_branch),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("LateFusion news branch early stop at epoch {}", epoch + 1)
                    break

        if best_state is not None:
            self.news_projector.load_state_dict(best_state["projector"])
            self.news_branch.load_state_dict(best_state["branch"])

        self._is_news_fitted = True
        return history

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        market_windows: np.ndarray,
        news_embs: np.ndarray,
        news_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        _require_attr(self.encoder, "predict_market_only", "LateFusionWrapper.predict")
        if not self._is_news_fitted:
            raise RuntimeError("LateFusionWrapper.predict called before fit.")

        pred_market = _to_numpy_float32(self.encoder.predict_market_only(market_windows))

        # Project news and derive mask from projected representation
        news_proj = self.news_projector.ensure_projected(
            np.asarray(news_embs, dtype=np.float32)
        )
        mask = _as_bool_mask(news_mask, news_proj)

        self.news_projector.eval()
        self.news_branch.eval()

        N = torch.as_tensor(news_proj, dtype=torch.float32, device=self.device)
        NM = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        M = torch.as_tensor(pred_market * self.target_scale, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_news = (
                self.news_branch(N, NM, market_pred=M).cpu().numpy() / self.target_scale
            )

        return (pred_market + pred_news).astype(np.float32)

    def predict_market_only(self, market_windows: np.ndarray) -> np.ndarray:
        return self.encoder.predict_market_only(market_windows)