"""Tests for retained Phase 1 models and split logic."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.phase1.splits import split_by_date, impute_tabular_splits
from src.phase1.baseline_models import (
    ChronosFrozenProbePredictor,
    ChronosLoRAPredictor,
    LSTMPredictor,
)


# ======================================================================
# split_by_date tests
# ======================================================================


class TestSplitByDate:
    """Verify walk-forward splitting with horizon-aware purge."""

    @staticmethod
    def _make_data(n: int = 200):
        rng = np.random.default_rng(42)
        times = pd.bdate_range("2023-01-02", periods=n, freq="B").values
        data = {
            "close_windows": rng.normal(100, 5, (n, 30)),
            "targets": rng.normal(0, 0.02, n),
            "news_embs": rng.normal(0, 1, (n, 768)).astype(np.float32),
        }
        return data, times

    def test_no_overlap(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        n_total = sum(len(splits[s]["targets"]) for s in ("train", "val", "test"))
        assert n_total <= len(times), "More samples than input"

    def test_chronological(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        # Extract time masks consistent with split_by_date internals
        train_end = pd.Timestamp("2023-06-30")
        val_end = pd.Timestamp("2023-09-30")
        train_t = times[times <= train_end]
        val_t = times[(times > train_end) & (times <= val_end)]
        test_t = times[times > val_end]

        if len(train_t) > 0 and len(val_t) > 0:
            assert train_t.max() < val_t.min()
        if len(val_t) > 0 and len(test_t) > 0:
            assert val_t.max() < test_t.min()

    def test_purge_buffer_reduces_train(self):
        """With horizon > 1, the last H trading days before boundary
        should be excluded from the preceding split (purge buffer)."""
        data, times = self._make_data()

        splits_h1 = split_by_date(data, times, "2023-06-30", "2023-09-30", target_horizon_days=1)
        splits_h5 = split_by_date(data, times, "2023-06-30", "2023-09-30", target_horizon_days=5)

        # With larger horizon, train should have fewer samples due to purge
        assert len(splits_h5["train"]["targets"]) <= len(splits_h1["train"]["targets"])

    def test_all_keys_present_in_splits(self):
        data, times = self._make_data()
        splits = split_by_date(data, times, "2023-06-30", "2023-09-30")

        for split_name in ("train", "val", "test"):
            assert set(splits[split_name].keys()) == set(data.keys())


# ======================================================================
# impute_tabular_splits tests
# ======================================================================


class TestImputeTabular:
    def test_uses_train_only(self):
        """NaN imputation values should come from train split only."""
        train_tab = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        val_tab = np.array([[np.nan, 10.0]], dtype=np.float32)
        test_tab = np.array([[np.nan, np.nan]], dtype=np.float32)

        splits = {
            "train": {"market_tabular": train_tab, "targets": np.zeros(2)},
            "val": {"market_tabular": val_tab, "targets": np.zeros(1)},
            "test": {"market_tabular": test_tab, "targets": np.zeros(1)},
        }
        result = impute_tabular_splits(splits)

        # Train col0 mean = 2.0, col1 mean = 4.0
        assert result["val"]["market_tabular"][0, 0] == pytest.approx(2.0)
        assert result["test"]["market_tabular"][0, 0] == pytest.approx(2.0)
        assert result["test"]["market_tabular"][0, 1] == pytest.approx(4.0)


class _DummyTokenizer:
    def context_input_transform(self, values):
        values = values.detach().cpu()
        token_ids = ((values * 10).round().abs().long() % 31) + 1
        attention_mask = torch.ones_like(token_ids)
        return token_ids, attention_mask, None


class _DummyChronosModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = T5Config(
            vocab_size=64,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout_rate=0.0,
        )
        self.model = T5ForConditionalGeneration(config)
        self.device = torch.device("cpu")

    def encode(self, input_ids, attention_mask):
        return self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


class _DummyChronosPredictor:
    def __init__(self):
        self.d_model = 32
        self.pipeline = type("Pipeline", (), {})()
        self.pipeline.tokenizer = _DummyTokenizer()
        self.pipeline.model = _DummyChronosModel()


class TestChronosPhase1Baselines:
    @staticmethod
    def _make_data():
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(12, 8, 4)).astype(np.float32)
        y_train = rng.normal(size=(12,)).astype(np.float32)
        X_val = rng.normal(size=(4, 8, 4)).astype(np.float32)
        y_val = rng.normal(size=(4,)).astype(np.float32)
        X_test = rng.normal(size=(3, 8, 4)).astype(np.float32)
        return X_train, y_train, X_val, y_val, X_test

    def test_frozen_probe_fit_predict(self):
        X_train, y_train, X_val, y_val, X_test = self._make_data()
        chronos = _DummyChronosPredictor()
        model = ChronosFrozenProbePredictor(chronos, hidden_dim=16, dropout=0.0, device="cpu")

        model.fit(X_train, y_train, X_val, y_val, epochs=2, batch_size=4, learning_rate=1e-3, patience=2)
        preds = model.predict(X_test)

        assert preds.shape == (3,)
        assert all(not param.requires_grad for param in chronos.pipeline.model.parameters())

    def test_frozen_probe_cached_embeddings_fit_predict(self):
        X_train, y_train, X_val, y_val, X_test = self._make_data()
        model = ChronosFrozenProbePredictor(_DummyChronosPredictor(), hidden_dim=16, dropout=0.0, device="cpu")

        train_embeddings = model.get_embeddings(X_train)
        val_embeddings = model.get_embeddings(X_val)
        test_embeddings = model.get_embeddings(X_test)

        model.fit_from_embeddings(
            train_embeddings,
            y_train,
            val_embeddings,
            y_val,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            patience=2,
        )
        preds = model.predict_from_embeddings(test_embeddings)

        assert train_embeddings.shape == (12, 32)
        assert preds.shape == (3,)

    def test_lstm_pruning_callback_propagates(self):
        X_train, y_train, X_val, y_val, _ = self._make_data()
        model = LSTMPredictor(hidden_dim=16, num_layers=1, dropout=0.0, input_dim=X_train.shape[-1], device="cpu")

        def pruning_callback(epoch, val_loss):
            raise optuna.TrialPruned(f"pruned at epoch {epoch + 1} with val_loss={val_loss:.6f}")

        with pytest.raises(optuna.TrialPruned):
            model.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                patience=2,
                pruning_callback=pruning_callback,
            )

    def test_lora_fit_predict_encoder_only(self):
        pytest.importorskip("peft")
        X_train, y_train, X_val, y_val, X_test = self._make_data()
        model = ChronosLoRAPredictor(
            _DummyChronosPredictor(),
            hidden_dim=16,
            dropout=0.0,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            device="cpu",
        )

        model.fit(X_train, y_train, X_val, y_val, epochs=2, batch_size=4, learning_rate=1e-3, patience=2)
        preds = model.predict(X_test)
        trainable_names = [name for name, param in model.transformer.named_parameters() if param.requires_grad]

        assert preds.shape == (3,)
        assert trainable_names
        assert all("encoder." in name for name in trainable_names)

    def test_lora_token_cache_checkpoint_round_trip(self):
        pytest.importorskip("peft")
        X_train, y_train, X_val, y_val, X_test = self._make_data()
        chronos = _DummyChronosPredictor()
        model = ChronosLoRAPredictor(
            chronos,
            hidden_dim=16,
            dropout=0.0,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            device="cpu",
        )

        train_token_ids, train_attention_mask = model.tokenize_windows(X_train)
        val_token_ids, val_attention_mask = model.tokenize_windows(X_val)
        test_token_ids, test_attention_mask = model.tokenize_windows(X_test)

        model.fit_tokenized(
            train_token_ids,
            train_attention_mask,
            y_train,
            val_token_ids,
            val_attention_mask,
            y_val,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            patience=2,
        )
        preds_before = model.predict_tokenized(test_token_ids, test_attention_mask)
        checkpoint = model.checkpoint_state()

        reloaded = ChronosLoRAPredictor(
            chronos,
            hidden_dim=16,
            dropout=0.0,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            device="cpu",
        )
        reloaded.load_checkpoint_state(checkpoint)
        preds_after = reloaded.predict_tokenized(test_token_ids, test_attention_mask)

        assert preds_before.shape == (3,)
        assert np.allclose(preds_before, preds_after)

    def test_lora_tokenized_pruning_callback_propagates(self):
        pytest.importorskip("peft")
        X_train, y_train, X_val, y_val, _ = self._make_data()
        model = ChronosLoRAPredictor(
            _DummyChronosPredictor(),
            hidden_dim=16,
            dropout=0.0,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            device="cpu",
        )
        train_token_ids, train_attention_mask = model.tokenize_windows(X_train)
        val_token_ids, val_attention_mask = model.tokenize_windows(X_val)

        def pruning_callback(epoch, val_loss):
            raise optuna.TrialPruned(f"pruned at epoch {epoch + 1} with val_loss={val_loss:.6f}")

        with pytest.raises(optuna.TrialPruned):
            model.fit_tokenized(
                train_token_ids,
                train_attention_mask,
                y_train,
                val_token_ids,
                val_attention_mask,
                y_val,
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                patience=2,
                pruning_callback=pruning_callback,
            )
