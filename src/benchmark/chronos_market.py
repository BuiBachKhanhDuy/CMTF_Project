"""Chronos market-only predictor: zero-shot and linear-probe modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class ChronosMarketPredictor:
    """Amazon Chronos on raw close-price series (no news).

    Modes:
        * **zero-shot** — no training; Chronos predicts next close,
          converted to log-return.
        * **linear-probe** — Chronos encoder embeddings → Ridge
          regression → predicted return.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from chronos import ChronosPipeline

        logger.info("Loading Chronos model: {} …", model_name)
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.batch_size = batch_size
        self.device = device

        # Detect embedding dimension
        probe = torch.tensor([[1.0, 2.0, 3.0]])
        emb, _ = self.pipeline.embed(probe)
        self.d_model = emb.shape[-1]
        logger.info("Chronos d_model = {}", self.d_model)

    # ------------------------------------------------------------------
    # Zero-shot prediction
    # ------------------------------------------------------------------
    def zero_shot_predict(
        self,
        close_windows: np.ndarray,
        last_close: np.ndarray,
        seed: int = 42,
        horizon: int = 1,
        num_samples: int = 64,
        aggregation: str = "median",
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Predict H-step-ahead log-return using Chronos zero-shot.

        Args:
            close_windows: (N, seq_len) raw close prices per sample.
            last_close: (N,) the last close price in each window
                (used to convert predicted close → return).
            seed: RNG seed for reproducible Chronos sampling.
            horizon: Number of steps ahead to forecast (prediction_length).
            num_samples: Number of probabilistic samples drawn from Chronos.
            aggregation: Point estimate aggregation in return space
                ("median" or "mean").
            return_diagnostics: If True, also return sample-dispersion metrics.

        Returns:
            (N,) predicted log-returns over *horizon* steps.
            If ``return_diagnostics=True``, returns
            ``(pred_returns, diagnostics_dict)``.
        """
        if aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be one of {'median', 'mean'}")

        all_preds: list[np.ndarray] = []
        all_std: list[np.ndarray] = []
        all_q10: list[np.ndarray] = []
        all_q90: list[np.ndarray] = []
        n = len(close_windows)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.tensor(
                close_windows[start:end], dtype=torch.float32
            )
            # Seed before each batch for deterministic sampling
            torch.manual_seed(seed + start)
            forecast = self.pipeline.predict(
                batch, prediction_length=horizon, num_samples=num_samples,
            )
            # forecast shape: (batch, num_samples, horizon)
            sampled_close = forecast[:, :, horizon - 1].cpu().numpy()

            # Aggregate in return-space to align with linear-probe semantics.
            base_close = np.clip(last_close[start:end], 1e-12, None)[:, None]
            sampled_returns = np.log(np.clip(sampled_close, 1e-12, None) / base_close)

            if aggregation == "median":
                point_estimate = np.median(sampled_returns, axis=1)
            else:
                point_estimate = np.mean(sampled_returns, axis=1)

            all_preds.append(point_estimate)
            all_std.append(np.std(sampled_returns, axis=1))
            all_q10.append(np.quantile(sampled_returns, 0.10, axis=1))
            all_q90.append(np.quantile(sampled_returns, 0.90, axis=1))

        pred_returns = np.concatenate(all_preds)  # (N,)

        if not return_diagnostics:
            return pred_returns

        pred_std = np.concatenate(all_std)
        pred_q10 = np.concatenate(all_q10)
        pred_q90 = np.concatenate(all_q90)
        diagnostics = {
            "pred_std": pred_std,
            "pred_q10": pred_q10,
            "pred_q90": pred_q90,
            "pred_iqr80": pred_q90 - pred_q10,
        }
        logger.info(
            "Zero-shot uncertainty | mean std={:.6f} | mean iqr80={:.6f}",
            float(np.mean(pred_std)) if len(pred_std) else 0.0,
            float(np.mean(diagnostics["pred_iqr80"])) if len(pred_std) else 0.0,
        )
        return pred_returns, diagnostics

    # ------------------------------------------------------------------
    # Embeddings extraction
    # ------------------------------------------------------------------
    def get_embeddings(
        self,
        close_windows: np.ndarray,
        pooling: str = "mean",
        recency_bias: float = 2.0,
    ) -> np.ndarray:
        """Extract pooled Chronos encoder embeddings.

        Args:
            close_windows: (N, seq_len) raw close prices.
            pooling: Pooling strategy ("mean" or "recency_weighted").
            recency_bias: Last-token weight multiplier for recency pooling.

        Returns:
            (N, d_model) embeddings.
        """
        if pooling not in {"mean", "recency_weighted"}:
            raise ValueError("pooling must be one of {'mean', 'recency_weighted'}")
        if recency_bias < 1.0:
            raise ValueError("recency_bias must be >= 1.0")

        all_embs: list[np.ndarray] = []
        n = len(close_windows)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.tensor(
                close_windows[start:end], dtype=torch.float32
            )
            with torch.no_grad():
                emb, _ = self.pipeline.embed(batch)
                # emb: (batch, num_tokens, d_model)
                if pooling == "recency_weighted":
                    n_tokens = emb.shape[1]
                    weights = torch.linspace(
                        1.0,
                        recency_bias,
                        steps=n_tokens,
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    weights = weights / weights.sum()
                    pooled = (emb * weights[None, :, None]).sum(dim=1).cpu().numpy()
                else:
                    pooled = emb.mean(dim=1).cpu().numpy()
            all_embs.append(pooled)

        return np.concatenate(all_embs, axis=0)  # (N, d_model)

    # ------------------------------------------------------------------
    # Linear-probe (embed → Ridge → return)
    # ------------------------------------------------------------------
    def linear_probe_predict(
        self,
        close_train: np.ndarray,
        y_train: np.ndarray,
        close_val: np.ndarray,
        y_val: np.ndarray,
        close_test: np.ndarray,
        tabular_train: Optional[np.ndarray] = None,
        tabular_val: Optional[np.ndarray] = None,
        tabular_test: Optional[np.ndarray] = None,
        allow_tabular_features: bool = False,
        embedding_pooling: str = "mean",
        recency_bias: float = 2.0,
        alpha_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        objective_weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
        pca_components: int = 50,
    ) -> np.ndarray:
        """Chronos embeddings + Ridge regression.

        Args:
            close_train: (N_train, seq_len) raw close windows.
            y_train: (N_train,) target returns.
            close_val: (N_val, seq_len) raw close windows.
            y_val: (N_val,) target returns (for alpha selection).
            close_test: (N_test, seq_len) raw close windows.
            tabular_train: (N_train, F_tab) optional engineered OHLCV/TA features.
            tabular_val: (N_val, F_tab) optional engineered OHLCV/TA features.
            tabular_test: (N_test, F_tab) optional engineered OHLCV/TA features.
            allow_tabular_features: Must be True to use tabular arrays.
            embedding_pooling: Embedding pooling strategy.
            recency_bias: Last-token weight multiplier for recency pooling.
            alpha_values: Candidate Ridge regularization strengths.
            objective_weights: Weights for (MSE, directional error, IC error).

        Returns:
            (N_test,) predicted returns.
        """
        logger.info("Computing Chronos embeddings for linear probe …")
        emb_train = self.get_embeddings(
            close_train, pooling=embedding_pooling, recency_bias=recency_bias
        )
        emb_val = self.get_embeddings(
            close_val, pooling=embedding_pooling, recency_bias=recency_bias
        )
        emb_test = self.get_embeddings(
            close_test, pooling=embedding_pooling, recency_bias=recency_bias
        )

        # PCA on Chronos embeddings to reduce dimensionality.
        # Raw embeddings have 512 dims with std ~0.005 while tabular features
        # have ~17 dims with std ~1.0.  The 30:1 dimension imbalance causes
        # Ridge to either overfit through noisy embedding dims (low alpha) or
        # ignore embeddings entirely (high alpha).  PCA reduces embeddings to
        # a comparable count of principal components.
        if pca_components > 0 and emb_train.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            emb_train = pca.fit_transform(emb_train)
            emb_val = pca.transform(emb_val)
            emb_test = pca.transform(emb_test)
            logger.info(
                "PCA reduced embeddings {} → {} dims (explained var: {:.1%})",
                self.d_model, pca_components,
                float(np.sum(pca.explained_variance_ratio_)),
            )

        tabular_arrays = (tabular_train, tabular_val, tabular_test)
        has_any_tabular = any(x is not None for x in tabular_arrays)
        has_all_tabular = all(x is not None for x in tabular_arrays)
        if has_any_tabular and not has_all_tabular:
            raise ValueError(
                "tabular_train/tabular_val/tabular_test must be all provided or all None"
            )
        if has_any_tabular and not allow_tabular_features:
            raise ValueError(
                "Tabular features are disabled for fairness. Set allow_tabular_features=True"
            )

        if has_all_tabular:
            logger.info(
                "Linear probe using Chronos embeddings + {} tabular market features",
                tabular_train.shape[1],
            )
            emb_train = np.concatenate([emb_train, tabular_train], axis=1)
            emb_val = np.concatenate([emb_val, tabular_val], axis=1)
            emb_test = np.concatenate([emb_test, tabular_test], axis=1)

        # Safety net: if any NaN remains, impute from train-only column means.
        if np.isnan(emb_train).any() or np.isnan(emb_val).any() or np.isnan(emb_test).any():
            logger.warning("NaN detected in linear-probe features; applying train-mean imputation")
            col_means = np.nanmean(emb_train, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            emb_train = np.where(np.isnan(emb_train), col_means, emb_train)
            emb_val = np.where(np.isnan(emb_val), col_means, emb_val)
            emb_test = np.where(np.isnan(emb_test), col_means, emb_test)

        # Select best Ridge alpha using a financial-aligned objective.
        # Lower is better: combines MSE, directional error, and IC error.
        candidates: list[dict[str, float]] = []
        for alpha in alpha_values:
            model = Ridge(alpha=alpha)
            model.fit(emb_train, y_train)
            val_pred = model.predict(emb_val)

            mse = float(np.mean((val_pred - y_val) ** 2))
            nonzero = y_val != 0
            if nonzero.any():
                da = float(np.mean(np.sign(val_pred[nonzero]) == np.sign(y_val[nonzero])))
            else:
                da = 0.0

            # Spearman IC via rank-correlation without extra dependencies.
            if len(y_val) >= 3:
                yr = np.argsort(np.argsort(y_val))
                pr = np.argsort(np.argsort(val_pred))
                if np.std(yr) > 0 and np.std(pr) > 0:
                    ic = float(np.corrcoef(yr, pr)[0, 1])
                else:
                    ic = 0.0
            else:
                ic = 0.0

            # Sign-balance penalty: reject alphas where >90% of val
            # predictions share the same sign (degenerate one-sided output).
            n_pos = int(np.sum(val_pred > 0))
            n_neg = int(np.sum(val_pred < 0))
            n_total = len(val_pred)
            sign_ratio = max(n_pos, n_neg) / max(n_total, 1)
            if sign_ratio > 0.90:
                # Heavily penalise — model is predicting nearly all one sign
                da = max(da * 0.5, 0.0)
                logger.debug(
                    "Ridge alpha={}: sign_ratio={:.2f} ({}+/{}−) — penalised",
                    alpha, sign_ratio, n_pos, n_neg,
                )

            candidates.append({"alpha": float(alpha), "mse": mse, "da": da, "ic": ic})

        mse_vals = np.array([c["mse"] for c in candidates], dtype=np.float64)
        da_vals = np.array([c["da"] for c in candidates], dtype=np.float64)
        ic_vals = np.array([c["ic"] for c in candidates], dtype=np.float64)

        def _norm(x: np.ndarray) -> np.ndarray:
            if len(x) == 0:
                return x
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if x_max <= x_min:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        mse_n = _norm(mse_vals)
        da_n = _norm(da_vals)
        ic_n = _norm(ic_vals)

        w_mse, w_da, w_ic = objective_weights
        objective = w_mse * mse_n + w_da * (1.0 - da_n) + w_ic * (1.0 - ic_n)
        best_idx = int(np.argmin(objective))
        best = candidates[best_idx]
        best_alpha = best["alpha"]

        logger.info(
            "Ridge best alpha = {} | val MSE={:.6f} DA={:.3f} IC={:.3f}",
            best_alpha,
            best["mse"],
            best["da"],
            best["ic"],
        )

        # Refit on train + val
        emb_trainval = np.concatenate([emb_train, emb_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        model = Ridge(alpha=best_alpha)
        model.fit(emb_trainval, y_trainval)

        raw_preds = model.predict(emb_test)

        # Zero-centre predictions to remove training-period level bias.
        # In quantitative finance, alpha signals should be zero-centred so
        # that the model's ranking ability (measured by IC) determines the
        # sign, not the training-set mean return.  Without this, a strongly
        # negative training mean forces *all* predictions negative regardless
        # of the embeddings — destroying precision/recall/F1 at long horizons.
        pred_median = float(np.median(raw_preds))
        centred_preds = raw_preds - pred_median
        n_pos = int(np.sum(centred_preds > 0))
        n_neg = int(np.sum(centred_preds < 0))
        logger.info(
            "LP zero-centring: median_shift={:.6f} → {}+/{}− predictions",
            pred_median, n_pos, n_neg,
        )
        return centred_preds
