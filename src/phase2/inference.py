"""Reusable Phase 2 PhoBERT inference utilities for downstream phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from .handoff import DEFAULT_PHASE2_OUTPUT_ROOT, resolve_phase2_phobert_handoff
from .modeling import PhoBERTSentimentModel
from .preprocessing import PreprocessingConfig, VnCoreNLPSegmenter, apply_preprocessing

_DEFAULT_VNCORENLP_JAR_PATH = Path("data/external/vncorenlp/VnCoreNLP-1.1.1.jar")


def _resolve_handoff_path(output_dir: str | Path, handoff_path: str | Path) -> Path:
    path = Path(handoff_path)
    if path.is_absolute():
        return path
    return Path(output_dir).resolve() / path


@dataclass(frozen=True, slots=True)
class Phase2PhoBERTInferenceBundle:
    """Loaded Phase 2 PhoBERT inference artifacts."""

    handoff: dict[str, Any]
    preprocessing_config: PreprocessingConfig
    tokenizer: Any
    model: PhoBERTSentimentModel
    device: str
    segmenter: Any | None = None


def _build_preprocessing_config(handoff: dict[str, Any]) -> PreprocessingConfig:
    args = dict(handoff.get("preprocessing", {}))
    return PreprocessingConfig(
        lowercase=bool(args.get("lowercase", False)),
        strip_html=bool(args.get("strip_html", True)),
        remove_punctuation=bool(args.get("remove_punctuation", True)),
        normalize_unicode_form=str(args.get("normalize_unicode_form", "NFC")),
        collapse_whitespace=bool(args.get("collapse_whitespace", True)),
        segmentation=str(args.get("segmentation", "none")),
    )


def _load_tokenizer(tokenizer_dir: str | Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Phase 2 PhoBERT inference requires the transformers package."
        ) from exc
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def _load_segmenter(
    preprocessing_config: PreprocessingConfig,
    jar_path: str | Path | None = None,
):
    if preprocessing_config.segmentation != "vncorenlp":
        return None

    resolved = Path(jar_path) if jar_path is not None else _DEFAULT_VNCORENLP_JAR_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"VnCoreNLP jar path not found: {resolved}")
    return VnCoreNLPSegmenter(resolved)


def load_phase2_phobert_inference_bundle(
    output_dir: str | Path = DEFAULT_PHASE2_OUTPUT_ROOT,
    *,
    device: str = "cpu",
    vncorenlp_jar_path: str | Path | None = None,
) -> Phase2PhoBERTInferenceBundle:
    """Load the trained Phase 2 PhoBERT branch for downstream inference."""

    handoff = resolve_phase2_phobert_handoff(output_dir)
    handoff["checkpoint_path"] = str(_resolve_handoff_path(output_dir, handoff["checkpoint_path"]))
    handoff["tokenizer_dir"] = str(_resolve_handoff_path(output_dir, handoff["tokenizer_dir"]))
    preprocessing_config = _build_preprocessing_config(handoff)
    try:
        segmenter = _load_segmenter(preprocessing_config, jar_path=vncorenlp_jar_path)
    except (FileNotFoundError, ImportError, OSError) as exc:
        if preprocessing_config.segmentation != "vncorenlp":
            raise
        logger.warning(
            "Phase 2 inference could not initialize VnCoreNLP ({}); falling back to non-segmented preprocessing",
            exc,
        )
        preprocessing_config = PreprocessingConfig(
            lowercase=preprocessing_config.lowercase,
            strip_html=preprocessing_config.strip_html,
            remove_punctuation=preprocessing_config.remove_punctuation,
            normalize_unicode_form=preprocessing_config.normalize_unicode_form,
            collapse_whitespace=preprocessing_config.collapse_whitespace,
            segmentation="none",
        )
        segmenter = None
    tokenizer = _load_tokenizer(handoff["tokenizer_dir"])

    model = PhoBERTSentimentModel(
        model_name=str(handoff.get("model_name", "vinai/phobert-base-v2")),
        projection_dim=int(handoff.get("projection_dim", 256)),
        num_heads=int(handoff.get("num_heads", 4)),
        dropout=float(handoff.get("dropout", 0.1)),
        freeze_backbone=bool(handoff.get("freeze_backbone", False)),
    )

    checkpoint = torch.load(handoff["checkpoint_path"], map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return Phase2PhoBERTInferenceBundle(
        handoff=handoff,
        preprocessing_config=preprocessing_config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        segmenter=segmenter,
    )


class Phase2PhoBERTInferencer:
    """Thin downstream inference wrapper around the trained Phase 2 model."""

    def __init__(self, bundle: Phase2PhoBERTInferenceBundle) -> None:
        self.bundle = bundle

    @property
    def max_length(self) -> int:
        return int(self.bundle.handoff.get("max_length", 128))

    def preprocess_titles(self, titles: Sequence[str]) -> list[str]:
        frame = pd.DataFrame({"title_raw": list(titles)})
        processed = apply_preprocessing(
            frame,
            config=self.bundle.preprocessing_config,
            text_col="title_raw",
            output_col="title_clean",
            segmenter=self.bundle.segmenter,
        )
        return processed["title_clean"].fillna("").astype(str).tolist()

    def predict_titles(
        self,
        titles: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        raw_titles = [str(title or "") for title in titles]
        clean_titles = self.preprocess_titles(raw_titles)
        if not raw_titles:
            return pd.DataFrame(
                columns=[
                    "title_raw",
                    "title_clean",
                    "sentiment_score",
                    "prob_negative",
                    "prob_neutral",
                    "prob_positive",
                ]
            )

        encoded = self.bundle.tokenizer(
            clean_titles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        score_batches: list[np.ndarray] = []
        prob_batches: list[np.ndarray] = []
        model = self.bundle.model
        device = self.bundle.device

        with torch.no_grad():
            for start in range(0, len(raw_titles), batch_size):
                end = start + batch_size
                outputs = model(
                    input_ids=input_ids[start:end].to(device),
                    attention_mask=attention_mask[start:end].to(device),
                )
                score_batches.append(outputs["expected_value"].detach().cpu().numpy())
                prob_batches.append(outputs["probabilities"].detach().cpu().numpy())

        scores = np.concatenate(score_batches, axis=0).astype(np.float32)
        probabilities = np.concatenate(prob_batches, axis=0).astype(np.float32)
        return pd.DataFrame(
            {
                "title_raw": raw_titles,
                "title_clean": clean_titles,
                "sentiment_score": scores,
                "prob_negative": probabilities[:, 0],
                "prob_neutral": probabilities[:, 1],
                "prob_positive": probabilities[:, 2],
            }
        )


def aggregate_title_sentiment_scores(scores: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Aggregate title-level expected-value scores into bar-level features."""

    values = np.asarray(list(scores), dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "sentiment_mean": 0.0,
            "sentiment_max_abs": 0.0,
            "sentiment_positive_ratio": 0.0,
            "sentiment_negative_ratio": 0.0,
            "sentiment_score_count": 0.0,
            "sentiment_missing_flag": 1.0,
        }

    return {
        "sentiment_mean": float(values.mean()),
        "sentiment_max_abs": float(np.abs(values).max()),
        "sentiment_positive_ratio": float((values > 0).mean()),
        "sentiment_negative_ratio": float((values < 0).mean()),
        "sentiment_score_count": float(values.size),
        "sentiment_missing_flag": 0.0,
    }


def flatten_sentiment_scores(score_frame: pd.DataFrame) -> list[float]:
    """Extract scalar scores from a title-level score frame."""

    if "sentiment_score" not in score_frame.columns:
        return []
    return [float(value) for value in score_frame["sentiment_score"].tolist() if np.isfinite(value)]
