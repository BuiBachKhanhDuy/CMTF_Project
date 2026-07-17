"""Evaluation metrics and visualization helpers for the sentiment encoder comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    y_true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
    y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_pearson(y_true_rank, y_pred_rank)


def compute_title_level_metrics(
    label_ids: np.ndarray,
    probabilities: np.ndarray,
    expected_values: np.ndarray,
    target_values: np.ndarray,
) -> dict[str, Any]:
    """Compute both classification and expected-value regression metrics."""

    predicted_ids = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(label_ids, predicted_ids))
    precision, recall, f1, support = precision_recall_fscore_support(
        label_ids,
        predicted_ids,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    conf = confusion_matrix(label_ids, predicted_ids, labels=[0, 1, 2])
    residuals = expected_values - target_values

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "pearson": _safe_pearson(target_values, expected_values),
        "spearman": _safe_spearman(target_values, expected_values),
        "confusion_matrix": conf,
        "support": int(support.sum()) if hasattr(support, "sum") else int(len(label_ids)),
    }


def build_prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    expected_values: np.ndarray,
    attention_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach prediction outputs to the canonical sentiment dataframe rows."""

    out = frame.reset_index(drop=True).copy()
    if len(out) != len(probabilities) or len(out) != len(expected_values):
        raise ValueError("Prediction outputs must align 1:1 with the input frame")

    out["prob_negative"] = probabilities[:, 0]
    out["prob_neutral"] = probabilities[:, 1]
    out["prob_positive"] = probabilities[:, 2]
    out["predicted_label_id"] = probabilities.argmax(axis=1)
    label_lookup = {0: "negative", 1: "neutral", 2: "positive"}
    out["predicted_label_name"] = out["predicted_label_id"].map(label_lookup)
    out["predicted_expected_value"] = expected_values
    out["prediction_confidence"] = probabilities.max(axis=1)
    if "target_value" in out.columns:
        out["residual"] = out["predicted_expected_value"] - out["target_value"]
    if attention_weights is not None:
        out["attention_max"] = attention_weights.max(axis=1)
        out["attention_entropy"] = -np.sum(
            np.clip(attention_weights, 1e-8, 1.0) * np.log(np.clip(attention_weights, 1e-8, 1.0)),
            axis=1,
        )
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history: dict[str, list[float]], save_path: str | Path) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="Train loss", color="#0f766e")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val loss", color="#dc2626")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("val_rmse", []), label="Val RMSE", color="#2563eb")
    axes[1].plot(epochs, history.get("val_macro_f1", []), label="Val Macro F1", color="#ea580c")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: str | Path,
    labels: tuple[str, str, str] = ("negative", "neutral", "positive"),
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="YlGnBu")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_expected_value_scatter(
    target_values: np.ndarray,
    expected_values: np.ndarray,
    save_path: str | Path,
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_values, expected_values, alpha=0.7, color="#0f766e")
    limits = [min(target_values.min(), expected_values.min()), max(target_values.max(), expected_values.max())]
    ax.plot(limits, limits, linestyle="--", color="#1f2937")
    ax.set_xlabel("True Expected Value")
    ax.set_ylabel("Predicted Expected Value")
    ax.set_title("Expected Value Scatter")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_histogram(
    target_values: np.ndarray,
    expected_values: np.ndarray,
    save_path: str | Path,
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    residuals = expected_values - target_values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=min(20, max(5, len(residuals) // 2)), color="#7c3aed", alpha=0.85)
    ax.axvline(0.0, color="#1f2937", linestyle="--")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Prediction Residual")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_distribution_by_class(
    prediction_df: pd.DataFrame,
    save_path: str | Path,
    label_col: str = "label_name",
    score_col: str = "predicted_expected_value",
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    labels = ["negative", "neutral", "positive"]
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [
        prediction_df.loc[prediction_df[label_col] == label, score_col].to_numpy(dtype=np.float64)
        for label in labels
    ]
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.axhline(0.0, linestyle="--", color="#1f2937")
    ax.set_title("Predicted Score by True Class")
    ax.set_ylabel("Predicted Expected Value")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    save_path: str | Path,
    metric_cols: tuple[str, ...] = ("rmse", "mae", "accuracy", "macro_f1"),
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        ax.bar(metrics_df["model_name"], metrics_df[metric], color=["#0f766e", "#2563eb", "#dc2626", "#f59e0b"][: len(metrics_df)])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_preprocessing_label_distribution(
    frame: pd.DataFrame,
    save_path: str | Path,
    label_col: str = "label_name",
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    counts = frame[label_col].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=["#b91c1c", "#6b7280", "#15803d"])
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_length_comparison(
    frame: pd.DataFrame,
    save_path: str | Path,
    raw_len_col: str = "raw_token_len",
    clean_len_col: str = "clean_token_len",
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame[raw_len_col], alpha=0.55, bins=20, label="Raw tokens", color="#64748b")
    ax.hist(frame[clean_len_col], alpha=0.55, bins=20, label="Clean tokens", color="#0f766e")
    ax.set_title("Token Length Before vs After Preprocessing")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_preprocessing_examples_table(
    examples_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    path = Path(save_path)
    _ensure_parent(path)

    fig_height = max(2.5, 0.6 * len(examples_df) + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=examples_df.values.tolist(),
        colLabels=examples_df.columns.tolist(),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path