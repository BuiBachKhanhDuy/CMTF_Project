"""Focused contract tests for active benchmark baseline models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.benchmark.baseline_models import (
    CNNLSTMPredictor,
    LSTMPredictor,
    RandomForestRegressor_Wrapper,
    sign_aware_huber_loss,
)


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


def test_sign_aware_huber_loss_penalizes_zero_prediction_on_directional_target():
    import torch

    target = torch.tensor([0.05], dtype=torch.float32)
    zero_pred = torch.tensor([0.0], dtype=torch.float32)
    signed_pred = torch.tensor([0.05], dtype=torch.float32)

    zero_loss = sign_aware_huber_loss(
        zero_pred,
        target,
        huber_delta=0.02,
        sign_penalty_weight=0.2,
    )
    signed_loss = sign_aware_huber_loss(
        signed_pred,
        target,
        huber_delta=0.02,
        sign_penalty_weight=0.2,
    )

    assert float(zero_loss) > float(signed_loss)


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
