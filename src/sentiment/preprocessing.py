"""Text preprocessing utilities and diagnostics for sentiment title models."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configurable text cleanup for sentiment experiments."""

    lowercase: bool = False
    strip_html: bool = True
    remove_punctuation: bool = True
    normalize_unicode_form: str = "NFC"
    collapse_whitespace: bool = True
    segmentation: str = "none"


class VnCoreNLPSegmenter:
    """Optional word segmenter backed by VnCoreNLP."""

    def __init__(self, jar_path: str | Path, max_heap_size: str = "-Xmx500m") -> None:
        from vncorenlp import VnCoreNLP

        self.jar_path = str(jar_path)
        self._segmenter = VnCoreNLP(self.jar_path, annotators="wseg", max_heap_size=max_heap_size)

    def __call__(self, text: str) -> str:
        sentences = self._segmenter.tokenize(text)
        return " ".join(" ".join(sentence) for sentence in sentences)


def clean_title(
    text: str,
    config: PreprocessingConfig,
    segmenter: Callable[[str], str] | None = None,
) -> str:
    """Normalize one Vietnamese title into a model-ready text string."""

    out = str(text or "")
    out = unicodedata.normalize(config.normalize_unicode_form, out)
    if config.strip_html:
        out = _HTML_TAG_RE.sub(" ", out)
    if config.remove_punctuation:
        out = _PUNCT_RE.sub(" ", out)
    if config.lowercase:
        out = out.lower()
    if config.collapse_whitespace:
        out = _MULTISPACE_RE.sub(" ", out).strip()

    if config.segmentation == "vncorenlp":
        if segmenter is None:
            raise ValueError(
                "segmentation='vncorenlp' requires a segmenter callable"
            )
        out = segmenter(out)
        if config.collapse_whitespace:
            out = _MULTISPACE_RE.sub(" ", out).strip()

    return out


def apply_preprocessing(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    text_col: str = "title_raw",
    output_col: str = "title_clean",
    segmenter: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Apply shared sentiment preprocessing and attach diagnostics columns."""

    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")

    out = df.copy()
    raw_series = out[text_col].fillna("").astype(str)
    out[output_col] = raw_series.map(lambda value: clean_title(value, config, segmenter=segmenter))
    out["raw_char_len"] = raw_series.str.len().astype(int)
    out["clean_char_len"] = out[output_col].str.len().astype(int)
    out["raw_token_len"] = raw_series.map(lambda value: len(value.split())).astype(int)
    out["clean_token_len"] = out[output_col].map(lambda value: len(value.split())).astype(int)
    out["clean_is_empty"] = out[output_col].eq("")
    out["duplicate_raw_title"] = raw_series.duplicated(keep=False)
    out["duplicate_clean_title"] = out[output_col].duplicated(keep=False)
    return out


def build_preprocessing_report(
    df: pd.DataFrame,
    raw_col: str = "title_raw",
    clean_col: str = "title_clean",
    label_col: str = "label_name",
) -> dict[str, object]:
    """Summarize what preprocessing changed for visualization and reporting."""

    if raw_col not in df.columns or clean_col not in df.columns:
        raise ValueError(f"Expected columns {raw_col!r} and {clean_col!r} in dataframe")

    label_distribution: dict[str, int] = {}
    if label_col in df.columns:
        label_distribution = {
            str(key): int(value)
            for key, value in df[label_col].value_counts().sort_index().items()
        }

    return {
        "row_count": int(len(df)),
        "null_raw_titles": int(df[raw_col].fillna("").eq("").sum()),
        "empty_clean_titles": int(df[clean_col].fillna("").eq("").sum()),
        "duplicate_raw_titles": int(df[raw_col].duplicated(keep=False).sum()),
        "duplicate_clean_titles": int(df[clean_col].duplicated(keep=False).sum()),
        "avg_raw_char_len": float(df[raw_col].fillna("").map(len).mean()),
        "avg_clean_char_len": float(df[clean_col].fillna("").map(len).mean()),
        "avg_raw_token_len": float(df[raw_col].fillna("").map(lambda value: len(str(value).split())).mean()),
        "avg_clean_token_len": float(df[clean_col].fillna("").map(lambda value: len(str(value).split())).mean()),
        "label_distribution": label_distribution,
    }


def sample_preprocessing_examples(
    df: pd.DataFrame,
    raw_col: str = "title_raw",
    clean_col: str = "title_clean",
    n_samples: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic sample of raw-to-clean examples for reporting."""

    if raw_col not in df.columns or clean_col not in df.columns:
        raise ValueError(f"Expected columns {raw_col!r} and {clean_col!r} in dataframe")

    if len(df) <= n_samples:
        return df[[raw_col, clean_col]].copy().reset_index(drop=True)

    sample = df.sample(n=n_samples, random_state=random_seed)
    return sample[[raw_col, clean_col]].reset_index(drop=True)


def build_length_histogram_payload(
    df: pd.DataFrame,
    raw_len_col: str = "raw_token_len",
    clean_len_col: str = "clean_token_len",
) -> dict[str, np.ndarray]:
    """Expose token-length arrays for later plotting without re-computing them."""

    if raw_len_col not in df.columns or clean_len_col not in df.columns:
        raise ValueError(
            f"Expected token-length columns {raw_len_col!r} and {clean_len_col!r}"
        )

    return {
        "raw_token_len": df[raw_len_col].to_numpy(dtype=np.int32),
        "clean_token_len": df[clean_len_col].to_numpy(dtype=np.int32),
    }