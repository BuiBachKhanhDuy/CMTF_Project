"""Tokenization, dataloading, and training utilities for the sentiment encoder comparison."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .evaluation import compute_title_level_metrics


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 2e-4
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    patience: int = 3
    device: str = "cpu"
    grad_clip_norm: float = 1.0
    checkpoint_name: str = "sentiment_model.pt"
    selection_metric: str = "macro_f1"


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    history: dict[str, list[float]]
    best_epoch: int
    best_val_rmse: float
    best_selection_metric: str
    best_selection_score: float
    checkpoint_path: str | None


class SimpleVocabulary:
    """Minimal vocabulary for the custom transformer branch."""

    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad_token_id = token_to_id[self.PAD]
        self.unk_token_id = token_to_id[self.UNK]
        self.cls_token_id = token_to_id[self.CLS]
        self.sep_token_id = token_to_id[self.SEP]

    @classmethod
    def build(cls, texts: pd.Series, min_freq: int = 1) -> "SimpleVocabulary":
        counter: Counter[str] = Counter()
        for text in texts.fillna("").astype(str):
            counter.update(text.split())

        tokens = [cls.PAD, cls.UNK, cls.CLS, cls.SEP]
        for token, freq in sorted(counter.items()):
            if freq >= min_freq:
                tokens.append(token)
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def save_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def encode(self, text: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = [self.CLS, *str(text or "").split(), self.SEP]
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens[:max_length]]
        attention_mask = [1] * len(token_ids)
        pad_length = max_length - len(token_ids)
        if pad_length > 0:
            token_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        return np.asarray(token_ids, dtype=np.int64), np.asarray(attention_mask, dtype=np.int64)

    def batch_encode(self, texts: pd.Series, max_length: int) -> tuple[np.ndarray, np.ndarray]:
        encoded = [self.encode(text, max_length=max_length) for text in texts.fillna("").astype(str)]
        input_ids = np.stack([item[0] for item in encoded], axis=0)
        attention_mask = np.stack([item[1] for item in encoded], axis=0)
        return input_ids, attention_mask


class EncodedTextDataset(Dataset):
    """Pre-tokenized dataset used by both the custom and PhoBERT branches."""

    def __init__(self, frame: pd.DataFrame, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.input_ids = torch.as_tensor(np.asarray(input_ids).copy(), dtype=torch.long)
        self.attention_mask = torch.as_tensor(np.asarray(attention_mask).copy(), dtype=torch.bool)
        self.labels = torch.as_tensor(
            self.frame["label_id"].to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
        )
        self.target_values = torch.as_tensor(
            self.frame["target_value"].to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "target_values": self.target_values[idx],
            "row_index": idx,
        }


def _split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Split {split_name!r} is empty")
        splits[split_name] = split_df
    return splits


def build_custom_text_datasets(
    df: pd.DataFrame,
    text_col: str = "title_clean",
    max_length: int = 64,
    min_freq: int = 1,
) -> tuple[dict[str, EncodedTextDataset], SimpleVocabulary]:
    """Build split datasets for the custom transformer branch."""

    split_frames = _split_frame(df)
    vocab = SimpleVocabulary.build(split_frames["train"][text_col], min_freq=min_freq)
    datasets: dict[str, EncodedTextDataset] = {}
    for split_name, split_df in split_frames.items():
        input_ids, attention_mask = vocab.batch_encode(split_df[text_col], max_length=max_length)
        datasets[split_name] = EncodedTextDataset(split_df, input_ids, attention_mask)
    return datasets, vocab


def build_phobert_text_datasets(
    df: pd.DataFrame,
    tokenizer_name: str = "vinai/phobert-base-v2",
    text_col: str = "title_clean",
    max_length: int = 64,
    tokenizer: Any | None = None,
) -> tuple[dict[str, EncodedTextDataset], Any]:
    """Build split datasets for the PhoBERT branch."""

    split_frames = _split_frame(df)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "build_phobert_text_datasets requires the transformers package."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    datasets: dict[str, EncodedTextDataset] = {}
    for split_name, split_df in split_frames.items():
        encoded = tokenizer(
            split_df[text_col].fillna("").astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        datasets[split_name] = EncodedTextDataset(
            split_df,
            encoded["input_ids"],
            encoded["attention_mask"],
        )
    return datasets, tokenizer


def build_dataloaders(
    datasets: dict[str, EncodedTextDataset],
    batch_size: int = 16,
) -> dict[str, DataLoader]:
    """Create deterministic dataloaders for train, val, and test splits."""

    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def make_class_weights(labels: pd.Series) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced three-class training."""

    counts = labels.value_counts().sort_index().reindex([0, 1, 2], fill_value=0)
    counts = counts.replace(0, 1)
    weights = counts.sum() / (len(counts) * counts)
    return torch.as_tensor(weights.to_numpy(dtype=np.float32), dtype=torch.float32)


def evaluate_sentiment_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    loss_fn: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Run evaluation and return metrics plus raw prediction arrays."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    expected_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    attention_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            target_values = batch["target_values"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if loss_fn is not None:
                loss = loss_fn(outputs["logits"], labels)
                batch_size = int(labels.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

            logits_list.append(outputs["logits"].detach().cpu().numpy())
            probs_list.append(outputs["probabilities"].detach().cpu().numpy())
            expected_list.append(outputs["expected_value"].detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            targets_list.append(target_values.detach().cpu().numpy())
            attention_list.append(outputs["attention_weights"].detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    probabilities = np.concatenate(probs_list, axis=0)
    expected_values = np.concatenate(expected_list, axis=0)
    label_ids = np.concatenate(labels_list, axis=0)
    target_values = np.concatenate(targets_list, axis=0)
    attention_weights = np.concatenate(attention_list, axis=0)
    metrics = compute_title_level_metrics(
        label_ids=label_ids,
        probabilities=probabilities,
        expected_values=expected_values,
        target_values=target_values,
    )
    if loss_fn is not None and total_examples > 0:
        metrics["loss"] = total_loss / total_examples
    else:
        metrics["loss"] = float("nan")
    metrics["logits"] = logits
    metrics["probabilities"] = probabilities
    metrics["expected_values"] = expected_values
    metrics["label_ids"] = label_ids
    metrics["target_values"] = target_values
    metrics["attention_weights"] = attention_weights
    return metrics


def train_sentiment_model(
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
    output_dir: str | Path | None = None,
) -> TrainingArtifacts:
    """Train one model variant and early-stop on the configured validation metric."""

    device = torch.device(config.device)
    model.to(device)
    optimizer = _build_optimizer(model, config)
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_macro_f1": [],
    }
    best_epoch = 0
    best_val_rmse = float("inf")
    best_selection_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    checkpoint_path: Path | None = None
    if output_dir is not None:
        checkpoint_path = Path(output_dir) / config.checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch_idx in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in dataloaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(total_examples, 1)
        val_metrics = evaluate_sentiment_model(model, dataloaders["val"], device=config.device, loss_fn=loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_rmse"].append(float(val_metrics["rmse"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))

        current_rmse = float(val_metrics["rmse"])
        current_macro_f1 = float(val_metrics["macro_f1"])
        improved = False
        if config.selection_metric == "macro_f1":
            improved = current_macro_f1 > best_selection_score
            if improved:
                best_selection_score = current_macro_f1
        elif config.selection_metric == "rmse":
            improved = current_rmse < best_val_rmse
        else:
            raise ValueError(f"Unsupported selection_metric: {config.selection_metric}")

        if improved:
            best_val_rmse = current_rmse
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "training_config": asdict(config),
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val_rmse,
                        "best_selection_metric": config.selection_metric,
                        "best_selection_score": best_selection_score,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainingArtifacts(
        history=history,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        best_selection_metric=config.selection_metric,
        best_selection_score=best_selection_score,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


def _build_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    no_decay_terms = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    named_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_params:
        raise ValueError("Model has no trainable parameters")

    backbone_prefix = "backbone."
    grouped_params: list[dict[str, Any]] = []
    for is_backbone in (False, True):
        lr = config.learning_rate
        if is_backbone:
            if config.backbone_learning_rate is None:
                continue
            lr = float(config.backbone_learning_rate)
        for uses_decay in (True, False):
            params = [
                parameter
                for name, parameter in named_params
                if name.startswith(backbone_prefix) == is_backbone
                and (not any(term in name for term in no_decay_terms)) == uses_decay
            ]
            if not params:
                continue
            grouped_params.append(
                {
                    "params": params,
                    "lr": lr,
                    "weight_decay": config.weight_decay if uses_decay else 0.0,
                }
            )

    if config.backbone_learning_rate is None:
        return torch.optim.AdamW(
            (parameter for _, parameter in named_params),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    return torch.optim.AdamW(grouped_params)