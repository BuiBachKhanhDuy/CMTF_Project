"""Focused contract tests for active benchmark baseline models."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.benchmark.baseline_models import (
    CNNLSTMPredictor,
    ChronosLoRAPredictor,
    FineTunedChronosPredictor,
    LSTMPredictor,
    RandomForestRegressor_Wrapper,
    sign_aware_huber_loss,
)
from src.benchmark.cnn_lstm_cmtf import CNNLSTMCMTFPredictor


class _DummyChronosPredictor:
    def __init__(self, d_model: int = 8):
        self.d_model = d_model
        self.calls = 0

    def get_embeddings(self, close_windows: np.ndarray) -> np.ndarray:
        self.calls += 1
        close_windows = np.asarray(close_windows, dtype=np.float32)
        mean_feature = close_windows.mean(axis=1, keepdims=True)
        std_feature = close_windows.std(axis=1, keepdims=True)
        max_feature = close_windows.max(axis=1, keepdims=True)
        min_feature = close_windows.min(axis=1, keepdims=True)
        stacked = np.concatenate([mean_feature, std_feature, max_feature, min_feature], axis=1)
        repeats = int(np.ceil(self.d_model / stacked.shape[1]))
        tiled = np.tile(stacked, (1, repeats))
        return tiled[:, : self.d_model].astype(np.float32)


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

    @property
    def device(self):
        return next(self.model.parameters()).device


class _DummyChronosLoRAPredictor:
    def __init__(self):
        self.d_model = 32
        self.pipeline = type("Pipeline", (), {})()
        self.pipeline.tokenizer = _DummyTokenizer()
        self.pipeline.model = _DummyChronosModel()


def _make_multivariate_data():
    rng = np.random.default_rng(7)
    market_windows_train = rng.normal(size=(24, 10, 6)).astype(np.float32)
    market_windows_val = rng.normal(size=(8, 10, 6)).astype(np.float32)
    market_windows_test = rng.normal(size=(5, 10, 6)).astype(np.float32)

    close_train = market_windows_train[:, :, 3]
    close_val = market_windows_val[:, :, 3]
    close_test = market_windows_test[:, :, 3]

    tab_train = market_windows_train[:, -1, :]
    tab_val = market_windows_val[:, -1, :]
    tab_test = market_windows_test[:, -1, :]

    y_train = (0.2 * market_windows_train[:, -1, 0] - 0.1 * market_windows_train[:, -1, 1]).astype(np.float32)
    y_val = (0.2 * market_windows_val[:, -1, 0] - 0.1 * market_windows_val[:, -1, 1]).astype(np.float32)

    return {
        "market_windows_train": market_windows_train,
        "market_windows_val": market_windows_val,
        "market_windows_test": market_windows_test,
        "close_train": close_train,
        "close_val": close_val,
        "close_test": close_test,
        "tab_train": tab_train,
        "tab_val": tab_val,
        "tab_test": tab_test,
        "y_train": y_train,
        "y_val": y_val,
    }


def test_sign_aware_huber_loss_penalizes_wrong_direction():
    import torch

    target = torch.tensor([0.05, -0.04], dtype=torch.float32)
    correct = torch.tensor([0.04, -0.03], dtype=torch.float32)
    wrong = torch.tensor([-0.04, 0.03], dtype=torch.float32)

    correct_loss = sign_aware_huber_loss(correct, target, huber_delta=0.02, sign_penalty_weight=0.2)
    wrong_loss = sign_aware_huber_loss(wrong, target, huber_delta=0.02, sign_penalty_weight=0.2)

    assert float(wrong_loss) > float(correct_loss)


def test_sign_aware_huber_loss_ignores_near_zero_targets_for_direction_penalty():
    import torch

    target = torch.tensor([5e-5, -5e-5], dtype=torch.float32)
    flipped = torch.tensor([-0.02, 0.02], dtype=torch.float32)

    base_loss = sign_aware_huber_loss(
        flipped,
        target,
        huber_delta=0.02,
        sign_penalty_weight=0.0,
        direction_epsilon=1e-4,
    )
    penalized_loss = sign_aware_huber_loss(
        flipped,
        target,
        huber_delta=0.02,
        sign_penalty_weight=0.2,
        direction_epsilon=1e-4,
    )

    assert torch.isclose(base_loss, penalized_loss, atol=1e-8, rtol=0.0)


def test_lstm_predictor_accepts_multivariate_market_windows():
    data = _make_multivariate_data()
    model = LSTMPredictor(input_dim=6, hidden_dim=12, num_layers=1, dropout=0.0, device="cpu")

    history = model.fit(
        data["market_windows_train"],
        data["y_train"],
        data["market_windows_val"],
        data["y_val"],
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
    )
    preds = model.predict(data["market_windows_test"])
    embeddings = model.get_embeddings(data["market_windows_test"])

    assert preds.shape == (5,)
    assert preds.dtype == np.float32 or preds.dtype == np.float64
    assert embeddings.shape == (5, 12)
    assert np.isfinite(history["best_val_loss"])


def test_random_forest_accepts_multivariate_market_windows():
    data = _make_multivariate_data()
    model = RandomForestRegressor_Wrapper(
        n_estimators=20,
        max_depth=4,
        min_samples_split=2,
        random_state=42,
    )

    model.fit(data["market_windows_train"], data["y_train"])
    preds = model.predict(data["market_windows_test"])

    assert preds.shape == (5,)
    assert preds.dtype == np.float32


def test_finetuned_chronos_accepts_market_tabular_branch():
    data = _make_multivariate_data()
    model = FineTunedChronosPredictor(
        _DummyChronosPredictor(),
        hidden_dim=16,
        dropout=0.0,
        tabular_dim=data["tab_train"].shape[1],
        device="cpu",
    )

    history = model.fit(
        data["close_train"],
        data["y_train"],
        data["close_val"],
        data["y_val"],
        market_tabular_train=data["tab_train"],
        market_tabular_val=data["tab_val"],
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
    )
    preds = model.predict(data["close_test"], market_tabular=data["tab_test"])

    assert preds.shape == (5,)
    assert preds.dtype == np.float32
    assert np.isfinite(history["best_val_loss"])


def test_finetuned_chronos_reuses_cached_embeddings_across_repeated_calls():
    data = _make_multivariate_data()
    chronos = _DummyChronosPredictor()
    FineTunedChronosPredictor._embedding_cache.clear()

    model_a = FineTunedChronosPredictor(
        chronos,
        hidden_dim=16,
        dropout=0.0,
        tabular_dim=data["tab_train"].shape[1],
        device="cpu",
    )
    model_a.fit(
        data["close_train"],
        data["y_train"],
        data["close_val"],
        data["y_val"],
        market_tabular_train=data["tab_train"],
        market_tabular_val=data["tab_val"],
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        patience=1,
    )
    _ = model_a.predict(data["close_test"], market_tabular=data["tab_test"])

    model_b = FineTunedChronosPredictor(
        chronos,
        hidden_dim=16,
        dropout=0.0,
        tabular_dim=data["tab_train"].shape[1],
        device="cpu",
    )
    model_b.fit(
        data["close_train"],
        data["y_train"],
        data["close_val"],
        data["y_val"],
        market_tabular_train=data["tab_train"],
        market_tabular_val=data["tab_val"],
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        patience=1,
    )
    _ = model_b.predict(data["close_test"], market_tabular=data["tab_test"])

    assert chronos.calls == 3


def test_chronos_lora_fit_predict_checkpoint_round_trip():
    pytest.importorskip("peft")
    data = _make_multivariate_data()
    chronos = _DummyChronosLoRAPredictor()
    model = ChronosLoRAPredictor(
        chronos,
        hidden_dim=16,
        dropout=0.0,
        market_input_dim=data["market_windows_train"].shape[-1],
        market_hidden_dim=12,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        device="cpu",
    )

    history = model.fit(
        data["close_train"],
        data["y_train"],
        data["close_val"],
        data["y_val"],
        market_windows_train=data["market_windows_train"],
        market_windows_val=data["market_windows_val"],
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
    )
    preds_before = model.predict(data["close_test"], market_windows=data["market_windows_test"])
    checkpoint = model.checkpoint_state()

    reloaded = ChronosLoRAPredictor(
        chronos,
        hidden_dim=16,
        dropout=0.0,
        market_input_dim=data["market_windows_train"].shape[-1],
        market_hidden_dim=12,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        device="cpu",
    )
    reloaded.load_checkpoint_state(checkpoint)
    preds_after = reloaded.predict(data["close_test"], market_windows=data["market_windows_test"])
    trainable_names = [name for name, param in model.transformer.named_parameters() if param.requires_grad]

    assert preds_before.shape == (5,)
    assert np.allclose(preds_before, preds_after)
    assert np.isfinite(history["best_val_loss"])
    assert trainable_names
    assert all("encoder." in name for name in trainable_names)


def test_cnn_lstm_cmtf_fit_predict_checkpoint_round_trip():
    data = _make_multivariate_data()
    rng = np.random.default_rng(11)
    news_train = rng.normal(size=(24, 10, 16)).astype(np.float32)
    news_val = rng.normal(size=(8, 10, 16)).astype(np.float32)
    news_test = rng.normal(size=(5, 10, 16)).astype(np.float32)
    news_mask_train = np.zeros((24, 10), dtype=bool)
    news_mask_val = np.zeros((8, 10), dtype=bool)
    news_mask_test = np.zeros((5, 10), dtype=bool)
    news_mask_test[:, -2:] = True
    news_test[:, -2:, :] = 0.0

    model = CNNLSTMCMTFPredictor(
        input_dim=data["market_windows_train"].shape[-1],
        news_dim=16,
        hidden_dim=16,
        num_filters=16,
        num_layers=2,
        fusion_dim=16,
        fusion_market_dim=16,
        n_heads=2,
        dropout=0.0,
        sign_penalty_weight=0.05,
        seq_len=data["market_windows_train"].shape[1],
        device="cpu",
    )

    history = model.fit(
        data["market_windows_train"],
        news_train,
        data["y_train"],
        data["market_windows_val"],
        news_val,
        data["y_val"],
        news_mask_train=news_mask_train,
        news_mask_val=news_mask_val,
        epochs=2,
        batch_size=8,
        patience=2,
        seed=11,
        freeze_encoder_epochs=0,
    )
    preds_before = model.predict(
        data["market_windows_test"],
        news_test,
        news_mask_test=news_mask_test,
    )
    checkpoint = model.get_checkpoint()

    reloaded = CNNLSTMCMTFPredictor(
        input_dim=data["market_windows_train"].shape[-1],
        news_dim=16,
        hidden_dim=16,
        num_filters=16,
        num_layers=2,
        fusion_dim=16,
        fusion_market_dim=16,
        n_heads=2,
        dropout=0.0,
        sign_penalty_weight=0.05,
        seq_len=data["market_windows_train"].shape[1],
        device="cpu",
    )
    reloaded.load_checkpoint(checkpoint)
    preds_after = reloaded.predict(
        data["market_windows_test"],
        news_test,
        news_mask_test=news_mask_test,
    )

    assert preds_before.shape == (5,)
    assert np.allclose(preds_before, preds_after)
    assert np.isfinite(history["val_loss"][-1])
    assert checkpoint["is_fitted"] is True


def test_cnn_lstm_cmtf_zero_news_matches_cnn_lstm_baseline():
    data = _make_multivariate_data()
    baseline = CNNLSTMPredictor(
        input_dim=data["market_windows_train"].shape[-1],
        num_filters=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        device="cpu",
    )
    baseline.fit(
        data["market_windows_train"],
        data["y_train"],
        data["market_windows_val"],
        data["y_val"],
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
    )

    seq_len = data["market_windows_test"].shape[1]
    zero_news = np.zeros((len(data["market_windows_test"]), seq_len, 16), dtype=np.float32)
    zero_news_mask = np.ones((len(data["market_windows_test"]), seq_len), dtype=bool)

    baseline_preds = baseline.predict(
        data["market_windows_test"],
    )

    cmtf = CNNLSTMCMTFPredictor(
        input_dim=data["market_windows_train"].shape[-1],
        news_dim=16,
        hidden_dim=16,
        num_filters=16,
        num_layers=2,
        fusion_dim=16,
        fusion_market_dim=16,
        n_heads=2,
        dropout=0.0,
        sign_penalty_weight=0.05,
        seq_len=seq_len,
        device="cpu",
    )
    baseline_state = baseline.checkpoint_state()["state_dict"]
    cmtf.load_state_dict(
        {
            (
                key.replace("fc.", "regression_head.")
                if key.startswith("fc.") else key
            ): value
            for key, value in baseline_state.items()
        },
        strict=False,
    )
    cmtf.is_fitted = True

    preds = cmtf.predict(
        data["market_windows_test"],
        zero_news,
        news_mask_test=zero_news_mask,
    )

    assert np.allclose(preds, baseline_preds, atol=1e-6, rtol=0.0)