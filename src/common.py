"""Shared constants and utilities across all phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
PHASE2_DATASET_ROOT = Path("data/phase2")
PHASE2_OUTPUT_ROOT = Path("outputs/phase2/latest")
JAVA_RUNTIME_ROOT = Path("cache/java_runtime")
VNCORENLP_ROOT = Path("cache/vncorenlp")
VNCORENLP_WORDSEGMENTER_ROOT = VNCORENLP_ROOT / "models" / "wordsegmenter"
VNCORENLP_JAR_PATH = VNCORENLP_ROOT / "VnCoreNLP-1.1.1.jar"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write a JSON-serializable object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(data, indent=indent, default=_default, ensure_ascii=False), encoding="utf-8")


def select_best_row(
    df: pd.DataFrame,
    score_col: str = "macro_f1",
    higher_is_better: bool = True,
) -> pd.Series:
    """Select the row with the best score from a DataFrame."""
    if df.empty:
        raise ValueError("Cannot select best row from empty DataFrame")
    if higher_is_better:
        idx = df[score_col].idxmax()
    else:
        idx = df[score_col].idxmin()
    return df.loc[idx]


def plot_metric_panels(
    df: pd.DataFrame,
    label_col: str,
    metric_cols: tuple[str, ...] | list[str],
    save_path: Path,
    title: str = "",
) -> None:
    """Plot a multi-panel bar chart for comparing metrics across labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    labels = df[label_col].unique()
    x = np.arange(len(labels))

    for ax, col in zip(axes, metric_cols):
        values = [df[df[label_col] == lbl][col].values[0] if col in df.columns else 0 for lbl in labels]
        ax.bar(x, values, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
