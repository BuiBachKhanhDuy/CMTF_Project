"""Dataset loading and canonical schema utilities for the sentiment encoder pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

_CANONICAL_LABELS = ("negative", "neutral", "positive")
_LABEL_TO_ID = {label: idx for idx, label in enumerate(_CANONICAL_LABELS)}
_LABEL_TO_TARGET = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}
_TITLE_COLUMN_CANDIDATES = (
    "title",
    "headline",
    "title_vi",
    "titlevn",
    "article_title",
    "text",
    "content",
)
_LABEL_COLUMN_CANDIDATES = (
    "label",
    "sentiment",
    "class",
    "impact",
    "category",
    "target",
    "y",
)
_SPLIT_COLUMN_CANDIDATES = ("split", "subset", "fold", "partition")
_ARTICLE_ID_COLUMN_CANDIDATES = ("article_id", "id", "news_id", "uid")
_SUPPORTED_FILE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}


@dataclass(frozen=True, slots=True)
class DatasetLoadConfig:
    """Configuration for loading the external supervised sentiment dataset."""

    dataset_path: str | Path
    random_seed: int = 42
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    prefer_existing_split: bool = True
    label_scheme: str = "auto"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Serializable metadata describing a canonicalized sentiment dataset."""

    dataset_path: str
    source_files: tuple[str, ...]
    row_count: int
    title_column: str
    label_column: str
    split_column: str
    label_scheme: str
    generated_split: bool
    split_counts: dict[str, int]
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SentimentDatasetBundle:
    """Canonical dataset plus manifest for sentiment text experiments."""

    dataframe: pd.DataFrame
    manifest: DatasetManifest


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _safe_float_ratio(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or out >= 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return out


def _validate_split_sizes(config: DatasetLoadConfig) -> None:
    train_size = _safe_float_ratio(config.train_size, "train_size")
    val_size = _safe_float_ratio(config.val_size, "val_size")
    test_size = _safe_float_ratio(config.test_size, "test_size")
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_size + val_size + test_size must sum to 1.0, "
            f"got {total:.4f}"
        )


def _discover_tabular_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    files = sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_FILE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No supported tabular files found under dataset path: {dataset_path}"
        )
    return files


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def _detect_split_from_filename(path: Path) -> str | None:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "valid" in name or re.search(r"\bval\b", name):
        return "val"
    if "test" in name:
        return "test"
    return None


def _find_first_matching_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        matched = normalized.get(_normalize_column_name(candidate))
        if matched is not None:
            return matched
    return None


def _normalize_split_name(value: Any) -> str | None:
    token = str(value).strip().lower()
    if not token or token == "nan":
        return None
    if token.startswith("train"):
        return "train"
    if token.startswith("val") or token.startswith("valid") or token.startswith("dev"):
        return "val"
    if token.startswith("test"):
        return "test"
    return None


def _choose_label_scheme(values: pd.Series, configured_scheme: str) -> str:
    if configured_scheme != "auto":
        return configured_scheme

    cleaned = values.dropna()
    if cleaned.empty:
        raise ValueError("Cannot infer label scheme from empty label column")

    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().all():
        unique = set(int(v) for v in numeric.astype(int).unique())
        if unique <= {-1, 0, 1}:
            return "signed"
        if unique <= {0, 1, 2}:
            return "zero_based"
        if unique <= {1, 2, 3}:
            return "source_repo"
        raise ValueError(
            f"Unsupported numeric label set for auto detection: {sorted(unique)}"
        )

    return "text"


def _map_numeric_label(label_value: int, label_scheme: str) -> str:
    if label_scheme == "signed":
        mapping = {-1: "negative", 0: "neutral", 1: "positive"}
    elif label_scheme == "zero_based":
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
    elif label_scheme == "source_repo":
        # The downloaded raw_data.xlsx stores 1/2/3 with the opposite polarity of
        # the README examples: 1=negative, 2=neutral, 3=positive. We align to the
        # file contents because training uses the local dataset, not the README.
        mapping = {1: "negative", 2: "neutral", 3: "positive"}
    else:
        raise ValueError(f"Unsupported numeric label scheme: {label_scheme}")

    if label_value not in mapping:
        raise ValueError(
            f"Label value {label_value} is not valid for scheme {label_scheme}"
        )
    return mapping[label_value]


def _map_text_label(raw_value: Any) -> str:
    token = str(raw_value).strip().lower()
    if not token or token == "nan":
        raise ValueError("Empty label value is not supported")

    if "positive" in token or "pos" == token or "tich" in token:
        return "positive"
    if "neutral" in token or "trung" in token or "neu" == token:
        return "neutral"
    if "negative" in token or "neg" == token or "tieu" in token:
        return "negative"

    digit_match = re.search(r"-?\d+", token)
    if digit_match:
        digit = int(digit_match.group())
        if digit in {-1, 0, 1}:
            return _map_numeric_label(digit, "signed")
        if digit in {0, 1, 2}:
            return _map_numeric_label(digit, "zero_based")
        if digit in {1, 2, 3}:
            return _map_numeric_label(digit, "source_repo")

    raise ValueError(f"Could not normalize label value: {raw_value!r}")


def _normalize_label_series(values: pd.Series, label_scheme: str) -> pd.Series:
    if label_scheme in {"signed", "zero_based", "source_repo"}:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Label scheme {label_scheme} expects numeric labels, found non-numeric values"
            )
        return numeric.astype(int).map(lambda value: _map_numeric_label(int(value), label_scheme))
    if label_scheme == "text":
        return values.map(_map_text_label)
    raise ValueError(f"Unsupported label scheme: {label_scheme}")


def deterministic_stratified_split(
    labels: pd.Series,
    random_seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> np.ndarray:
    """Generate deterministic stratified split labels for a three-way split."""

    sizes = [float(train_size), float(val_size), float(test_size)]
    if not np.isclose(sum(sizes), 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    n_rows = len(labels)
    indices = np.arange(n_rows)
    label_array = labels.to_numpy()

    label_counts = labels.value_counts()
    if n_rows < 6 or (not label_counts.empty and int(label_counts.min()) < 2):
        rng = np.random.default_rng(random_seed)
        shuffled = rng.permutation(indices)

        n_train = int(round(n_rows * train_size))
        n_val = int(round(n_rows * val_size))
        n_test = n_rows - n_train - n_val

        if n_rows >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            while (n_train + n_val + n_test) > n_rows:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            while (n_train + n_val + n_test) < n_rows:
                n_train += 1

        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]

        split = np.empty(n_rows, dtype=object)
        split[train_idx] = "train"
        split[val_idx] = "val"
        split[test_idx] = "test"
        return split

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        stratify=label_array,
        random_state=random_seed,
    )

    temp_labels = label_array[temp_idx]
    val_ratio_within_temp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio_within_temp),
        stratify=temp_labels,
        random_state=random_seed,
    )

    split = np.empty(n_rows, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def _build_article_ids(frame: pd.DataFrame, id_column: str | None) -> pd.Series:
    if id_column is not None:
        ids = frame[id_column].astype(str).str.strip()
        if ids.nunique(dropna=True) == len(frame):
            return ids

    generated = [f"{row['__source_file']}::{int(row['__source_row'])}" for _, row in frame.iterrows()]
    return pd.Series(generated, index=frame.index, dtype="string")


def load_sentiment_dataset(config: DatasetLoadConfig) -> SentimentDatasetBundle:
    """Load the external title-level sentiment dataset into a canonical schema."""

    _validate_split_sizes(config)
    dataset_path = Path(config.dataset_path)
    discovered_files = _discover_tabular_files(dataset_path)

    frames: list[pd.DataFrame] = []
    included_files: list[Path] = []
    for file_path in discovered_files:
        frame = _read_tabular_file(file_path)
        if frame.empty:
            continue

        columns = list(frame.columns)
        title_column = _find_first_matching_column(columns, _TITLE_COLUMN_CANDIDATES)
        label_column = _find_first_matching_column(columns, _LABEL_COLUMN_CANDIDATES)
        if title_column is None or label_column is None:
            logger.info(
                "Skipping unlabeled sentiment dataset file: {}",
                file_path.name,
            )
            continue

        frame = frame.copy()
        frame["__source_file"] = file_path.name
        frame["__source_row"] = np.arange(len(frame))
        inferred_split = _detect_split_from_filename(file_path)
        if inferred_split is not None:
            frame["__split_from_filename"] = inferred_split
        frames.append(frame)
        included_files.append(file_path)

    if not frames:
        raise ValueError(
            f"No labeled rows loaded from dataset path: {dataset_path}"
        )

    raw_df = pd.concat(frames, axis=0, ignore_index=True)
    columns = list(raw_df.columns)

    title_column = _find_first_matching_column(columns, _TITLE_COLUMN_CANDIDATES)
    if title_column is None:
        raise ValueError(
            "Could not detect a title column. Supported candidates: "
            f"{_TITLE_COLUMN_CANDIDATES}"
        )

    label_column = _find_first_matching_column(columns, _LABEL_COLUMN_CANDIDATES)
    if label_column is None:
        raise ValueError(
            "Could not detect a label column. Supported candidates: "
            f"{_LABEL_COLUMN_CANDIDATES}"
        )

    split_column = _find_first_matching_column(columns, _SPLIT_COLUMN_CANDIDATES)
    id_column = _find_first_matching_column(columns, _ARTICLE_ID_COLUMN_CANDIDATES)

    label_scheme = _choose_label_scheme(raw_df[label_column], config.label_scheme)
    label_name = _normalize_label_series(raw_df[label_column], label_scheme)

    canonical = pd.DataFrame(
        {
            "article_id": _build_article_ids(raw_df, id_column),
            "source_file": raw_df["__source_file"].astype(str),
            "title_raw": raw_df[title_column].fillna("").astype(str).str.strip(),
            "label_original": raw_df[label_column],
            "label_name": label_name,
        }
    )

    canonical = canonical[canonical["title_raw"].str.len() > 0].reset_index(drop=True)
    canonical["label_id"] = canonical["label_name"].map(_LABEL_TO_ID).astype(int)
    canonical["target_value"] = canonical["label_name"].map(_LABEL_TO_TARGET).astype(float)

    generated_split = True
    split_source = "generated"
    if config.prefer_existing_split:
        if split_column is not None:
            existing_split = raw_df[split_column].map(_normalize_split_name)
            existing_split = existing_split.loc[canonical.index]
            if existing_split.notna().all():
                canonical["split"] = existing_split.astype(str)
                generated_split = False
                split_source = split_column
        if "split" not in canonical.columns and "__split_from_filename" in raw_df.columns:
            inferred = raw_df["__split_from_filename"].map(_normalize_split_name)
            inferred = inferred.loc[canonical.index]
            if inferred.notna().all():
                canonical["split"] = inferred.astype(str)
                generated_split = False
                split_source = "filename"

    if "split" not in canonical.columns:
        canonical["split"] = deterministic_stratified_split(
            labels=canonical["label_name"],
            random_seed=config.random_seed,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
        )
        split_source = "generated"

    canonical = canonical[
        [
            "article_id",
            "source_file",
            "title_raw",
            "label_original",
            "label_name",
            "label_id",
            "target_value",
            "split",
        ]
    ].copy()

    manifest = DatasetManifest(
        dataset_path=str(dataset_path),
        source_files=tuple(sorted({str(path.name) for path in included_files})),
        row_count=int(len(canonical)),
        title_column=title_column,
        label_column=label_column,
        split_column=split_source,
        label_scheme=label_scheme,
        generated_split=bool(generated_split),
        split_counts={
            key: int(value)
            for key, value in canonical["split"].value_counts().sort_index().items()
        },
        label_counts={
            key: int(value)
            for key, value in canonical["label_name"].value_counts().sort_index().items()
        },
    )

    logger.info(
        "Sentiment dataset loaded | rows={} | files={} | title_col={} | label_col={} | split_source={}",
        manifest.row_count,
        len(manifest.source_files),
        manifest.title_column,
        manifest.label_column,
        manifest.split_column,
    )
    return SentimentDatasetBundle(dataframe=canonical, manifest=manifest)


def save_dataset_manifest(manifest: DatasetManifest, output_path: str | Path) -> Path:
    """Persist a dataset manifest as JSON for reproducible sentiment runs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path