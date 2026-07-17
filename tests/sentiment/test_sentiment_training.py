from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.sentiment import (
    CustomTransformerSentimentModel,
    PreprocessingConfig,
    TrainingConfig,
    apply_preprocessing,
    build_custom_text_datasets,
    build_dataloaders,
    build_phobert_text_datasets,
    build_prediction_frame,
    evaluate_sentiment_model,
    make_class_weights,
    train_sentiment_model,
)


def _build_toy_sentiment_frame() -> pd.DataFrame:
    rows = []
    templates = {
        "negative": [
            "cổ phiếu giảm mạnh",
            "thị trường lao dốc",
            "áp lực bán tăng cao",
            "kết quả kinh doanh suy yếu",
        ],
        "neutral": [
            "lịch sự kiện chứng khoán",
            "doanh nghiệp công bố thông tin",
            "thị trường đi ngang",
            "cổ phiếu giao dịch ổn định",
        ],
        "positive": [
            "lợi nhuận tăng trưởng mạnh",
            "cổ phiếu bứt phá tích cực",
            "dòng tiền quay trở lại",
            "doanh thu tăng tốt",
        ],
    }
    split_plan = ["train", "train", "train", "train", "val", "test"]

    for label_name, texts in templates.items():
        label_id = {"negative": 0, "neutral": 1, "positive": 2}[label_name]
        target_value = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}[label_name]
        for idx, split_name in enumerate(split_plan):
            rows.append(
                {
                    "article_id": f"{label_name}-{idx}",
                    "source_file": "toy.csv",
                    "title_raw": texts[idx % len(texts)],
                    "label_original": label_name,
                    "label_name": label_name,
                    "label_id": label_id,
                    "target_value": target_value,
                    "split": split_name,
                }
            )
    frame = pd.DataFrame(rows)
    return apply_preprocessing(frame, PreprocessingConfig())


def test_train_sentiment_model_smoke(tmp_path):
    frame = _build_toy_sentiment_frame()
    datasets, vocab = build_custom_text_datasets(frame, max_length=8)
    dataloaders = build_dataloaders(datasets, batch_size=4)

    model = CustomTransformerSentimentModel(
        vocab_size=len(vocab),
        hidden_dim=24,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        max_length=8,
    )
    class_weights = make_class_weights(datasets["train"].frame["label_id"])
    artifacts = train_sentiment_model(
        model,
        dataloaders,
        TrainingConfig(
            batch_size=4,
            epochs=2,
            learning_rate=5e-3,
            patience=2,
            device="cpu",
            checkpoint_name="toy_sentiment.pt",
        ),
        class_weights=class_weights,
        output_dir=tmp_path,
    )

    test_metrics = evaluate_sentiment_model(model, dataloaders["test"], device="cpu")
    prediction_df = build_prediction_frame(
        datasets["test"].frame,
        probabilities=test_metrics["probabilities"],
        expected_values=test_metrics["expected_values"],
        attention_weights=test_metrics["attention_weights"],
    )

    assert artifacts.best_epoch >= 1
    assert len(artifacts.history["train_loss"]) >= 1
    assert artifacts.checkpoint_path is not None
    assert artifacts.best_selection_metric == "macro_f1"
    assert np.isfinite(artifacts.best_selection_score)
    assert {"accuracy", "macro_f1", "mae", "rmse"}.issubset(test_metrics.keys())
    assert len(prediction_df) == len(datasets["test"])
    assert {"prob_negative", "prob_neutral", "prob_positive", "predicted_expected_value"}.issubset(prediction_df.columns)

    checkpoint = torch.load(artifacts.checkpoint_path, map_location="cpu")
    assert checkpoint["best_selection_metric"] == "macro_f1"
    assert np.isfinite(checkpoint["best_selection_score"])


def test_build_phobert_text_datasets_with_injected_tokenizer():
    class FakeTokenizer:
        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            assert truncation is True
            assert padding == "max_length"
            assert return_tensors == "np"
            input_ids = []
            attention_mask = []
            for index, _ in enumerate(texts):
                length = min(max_length, 3 + index)
                ids = list(range(1, length + 1)) + [0] * (max_length - length)
                mask = [1] * length + [0] * (max_length - length)
                input_ids.append(ids)
                attention_mask.append(mask)
            return {
                "input_ids": np.asarray(input_ids, dtype="int64"),
                "attention_mask": np.asarray(attention_mask, dtype="int64"),
            }

    frame = _build_toy_sentiment_frame()
    datasets, tokenizer = build_phobert_text_datasets(
        frame,
        max_length=8,
        tokenizer=FakeTokenizer(),
    )

    assert tokenizer is not None
    assert set(datasets.keys()) == {"train", "val", "test"}
    assert datasets["train"].input_ids.shape[1] == 8
    assert len(datasets["val"]) == 3
    assert len(datasets["test"]) == 3