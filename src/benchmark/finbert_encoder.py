"""FinBERT market-only predictor: zero-shot sentiment-to-return baseline.

Uses ProsusAI/finbert (financial BERT fine-tuned for 3-way sentiment) in a
zero-shot regime: OHLCV close windows are verbalised into English financial
prose, classified without any task-specific training, and mapped to log-return
predictions via the official FinBERT score (P(positive) - P(negative)).

No news text is consumed — only derived market statistics from close prices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_MAX_LENGTH = 512


def close_window_to_financial_text(
    close_window: np.ndarray,
    horizon: int = 1,
) -> str:
    """Verbalise a close-price window into FinBERT-compatible financial prose."""
    prices = np.clip(np.asarray(close_window, dtype=np.float64), 1e-12, None)
    if prices.size < 2:
        last = float(prices[-1]) if prices.size else 1.0
        return (
            f"Market report: the stock is trading at {last:.2f}. "
            f"Analyst outlook for the next {horizon}-day price direction."
        )

    log_rets = np.diff(np.log(prices))
    total_ret_pct = (prices[-1] / prices[0] - 1.0) * 100.0
    vol_pct = float(np.std(log_rets) * 100.0) if log_rets.size > 1 else 0.0

    n_recent = min(5, log_rets.size)
    recent_parts: list[str] = []
    for idx, r in enumerate(log_rets[-n_recent:], start=1):
        if r > 0:
            move = f"rose by {r * 100:.2f}%"
        elif r < 0:
            move = f"fell by {abs(r) * 100:.2f}%"
        else:
            move = "was unchanged"
        recent_parts.append(f"session {idx}: price {move}")

    if total_ret_pct > 1.0:
        trend = "bullish"
    elif total_ret_pct < -1.0:
        trend = "bearish"
    else:
        trend = "sideways"

    horizon_phrase = f"{horizon}-day" if horizon > 1 else "next-day"

    return (
        f"Market report: the stock shows a {trend} trend with a cumulative return "
        f"of {total_ret_pct:+.2f}% over the observation window. "
        f"Recent price action: {'; '.join(recent_parts)}. "
        f"Realized daily volatility is {vol_pct:.2f}%. "
        f"Analyst outlook for {horizon_phrase} price direction."
    )


class FinBERTMarketPredictor:
    """ProsusAI/finbert on verbalised close-price windows (no news).

    Mode:
        * **zero-shot** — no training; FinBERT sentiment score
          (P(positive) - P(negative)) scaled by recent window volatility
          yields a log-return point estimate.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_length = max_length

        logger.info("Loading FinBERT model: {} …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        self.id2label = dict(self.model.config.id2label)
        self._label_ids = {
            "negative": self._resolve_label_id("negative"),
            "neutral": self._resolve_label_id("neutral"),
            "positive": self._resolve_label_id("positive"),
        }
        self.d_model = int(getattr(self.model.config, "hidden_size", 768))

    def _resolve_label_id(self, name: str) -> int:
        target = name.lower()
        for label_id, label_name in self.id2label.items():
            if str(label_name).lower() == target:
                return int(label_id)
        raise KeyError(f"FinBERT label '{name}' not found in {self.id2label}")

    def _sentiment_scores(self, texts: list[str]) -> np.ndarray:
        """Return FinBERT sentiment score = P(positive) - P(negative) per row."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        pos_idx = self._label_ids["positive"]
        neg_idx = self._label_ids["negative"]
        return (probs[:, pos_idx] - probs[:, neg_idx]).astype(np.float32)

    def _window_volatility_scale(self, close_windows: np.ndarray, horizon: int) -> np.ndarray:
        """Per-window recent log-return std, horizon-adjusted."""
        prices = np.clip(np.asarray(close_windows, dtype=np.float64), 1e-12, None)
        log_rets = np.diff(np.log(prices), axis=1)
        vol = np.std(log_rets, axis=1, ddof=1)
        vol = np.where(np.isfinite(vol) & (vol > 1e-8), vol, 1e-4)
        horizon_scale = float(max(horizon, 1)) ** 0.5
        return (vol * horizon_scale).astype(np.float32)

    def zero_shot_predict(
        self,
        close_windows: np.ndarray,
        last_close: np.ndarray | None = None,
        seed: int = 42,
        horizon: int = 1,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Predict H-step-ahead log-return via FinBERT zero-shot sentiment mapping."""
        del last_close  # API parity with ChronosMarketPredictor; unused for FinBERT.

        close_windows = np.asarray(close_windows, dtype=np.float64)
        n = close_windows.shape[0]
        if n == 0:
            out = np.empty((0,), dtype=np.float32)
            if not return_diagnostics:
                return out
            return out, {
                "sentiment_score": out.copy(),
                "vol_scale": out.copy(),
            }

        torch.manual_seed(seed)

        logger.info(
            "FinBERT zero-shot start | N={} | seq_len={} | horizon={} | batch_size={}",
            n,
            close_windows.shape[1],
            horizon,
            self.batch_size,
        )

        all_preds: list[np.ndarray] = []
        all_sentiment: list[np.ndarray] = []
        all_vol: list[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_windows = close_windows[start:end]

            logger.info(
                "FinBERT zero-shot batch {}/{} | rows {}:{}",
                (start // self.batch_size) + 1,
                (n + self.batch_size - 1) // self.batch_size,
                start,
                end,
            )

            texts = [
                close_window_to_financial_text(batch_windows[i], horizon=horizon)
                for i in range(batch_windows.shape[0])
            ]
            sentiment = self._sentiment_scores(texts)
            vol_scale = self._window_volatility_scale(batch_windows, horizon=horizon)
            pred_returns = (sentiment * vol_scale).astype(np.float32)

            all_preds.append(pred_returns)
            all_sentiment.append(sentiment)
            all_vol.append(vol_scale)

        pred_returns = np.concatenate(all_preds, axis=0).astype(np.float32)

        if not return_diagnostics:
            return pred_returns

        sentiment_all = np.concatenate(all_sentiment, axis=0).astype(np.float32)
        vol_all = np.concatenate(all_vol, axis=0).astype(np.float32)
        diagnostics = {
            "sentiment_score": sentiment_all,
            "vol_scale": vol_all,
        }
        logger.info(
            "FinBERT zero-shot diagnostics | mean |sentiment|={:.4f} | mean vol_scale={:.6f}",
            float(np.mean(np.abs(sentiment_all))) if sentiment_all.size else 0.0,
            float(np.mean(vol_all)) if vol_all.size else 0.0,
        )
        return pred_returns, diagnostics

    def get_embeddings(
        self,
        close_windows: np.ndarray,
        pooling: str = "cls",
    ) -> np.ndarray:
        """Extract pooled FinBERT token embeddings from verbalised close windows.

        Args:
            close_windows: (N, seq_len) raw close prices.
            pooling: ``"cls"`` or ``"mean"`` over token hidden states.

        Returns:
            (N, d_model) embeddings.
        """
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be one of {'cls', 'mean'}")

        close_windows = np.asarray(close_windows, dtype=np.float64)
        n = close_windows.shape[0]
        if n == 0:
            return np.empty((0, self.d_model), dtype=np.float32)

        all_embs: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_windows = close_windows[start:end]
            texts = [close_window_to_financial_text(batch_windows[i]) for i in range(end - start)]

            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model.bert(**encoded)
                hidden = outputs.last_hidden_state
                if pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

            all_embs.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embs, axis=0)
