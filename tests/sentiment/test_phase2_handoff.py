from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from src.sentiment.handoff import build_phase2_phobert_handoff, resolve_phase2_phobert_handoff, save_phase2_phobert_handoff
from src.sentiment.inference import (
    Phase2PhoBERTInferencer,
    aggregate_title_sentiment_scores,
    load_phase2_phobert_inference_bundle,
)


def test_phase2_phobert_handoff_builds_and_saves(tmp_path):
    output_dir = tmp_path / "phase2_latest"
    (output_dir / "phobert" / "tokenizer").mkdir(parents=True)
    (output_dir / "phobert" / "phobert.pt").write_bytes(b"checkpoint")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phobert_model": "vinai/phobert-base-v2",
                "shared_hidden_dim": 256,
                "custom_heads": 4,
                "custom_dropout": 0.1,
                "freeze_phobert": False,
                "max_length": 128,
                "selection_metric": "macro_f1",
                "preprocessing": {
                    "lowercase": False,
                    "disable_punctuation_removal": False,
                    "use_vncorenlp": True,
                    "disable_vncorenlp": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "phobert" / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": 5,
                "best_selection_score": 0.77,
                "checkpoint_path": str(output_dir / "phobert" / "phobert.pt"),
            }
        ),
        encoding="utf-8",
    )

    handoff = build_phase2_phobert_handoff(output_dir)
    handoff_path = save_phase2_phobert_handoff(output_dir)
    resolved = resolve_phase2_phobert_handoff(output_dir)

    assert handoff["variant"] == "phobert"
    assert handoff["preprocessing"]["segmentation"] == "vncorenlp"
    assert handoff_path.exists()
    assert resolved["checkpoint_path"] == handoff["checkpoint_path"]


def test_phase2_phobert_inference_bundle_loads_and_scores_titles(tmp_path, monkeypatch):
    output_dir = tmp_path / "phase2_latest"
    tokenizer_dir = output_dir / "phobert" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "phobert" / "phobert.pt"
    torch.save({"model_state_dict": {"dummy": torch.tensor(1.0)}}, checkpoint_path)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phobert_model": "vinai/phobert-base-v2",
                "shared_hidden_dim": 256,
                "custom_heads": 4,
                "custom_dropout": 0.1,
                "freeze_phobert": False,
                "max_length": 16,
                "selection_metric": "macro_f1",
                "preprocessing": {
                    "lowercase": False,
                    "disable_punctuation_removal": False,
                    "use_vncorenlp": False,
                    "disable_vncorenlp": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "phobert" / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": 3,
                "best_selection_score": 0.81,
                "checkpoint_path": str(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            assert truncation is True
            assert padding == "max_length"
            assert max_length == 16
            assert return_tensors == "pt"
            batch = len(texts)
            return {
                "input_ids": torch.ones((batch, max_length), dtype=torch.long),
                "attention_mask": torch.ones((batch, max_length), dtype=torch.long),
            }

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.loaded_state = None
            self.device = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            batch = int(input_ids.shape[0])
            probs = torch.tensor(
                [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]],
                dtype=torch.float32,
            )[:batch]
            expected = torch.tensor([0.3, -0.5], dtype=torch.float32)[:batch]
            return {
                "probabilities": probs,
                "expected_value": expected,
            }

    monkeypatch.setattr("src.phase2.inference._load_tokenizer", lambda tokenizer_dir: FakeTokenizer())
    monkeypatch.setattr("src.phase2.inference.PhoBERTSentimentModel", FakeModel)

    bundle = load_phase2_phobert_inference_bundle(output_dir, device="cpu")
    inferencer = Phase2PhoBERTInferencer(bundle)
    scored = inferencer.predict_titles(["Lợi nhuận tăng mạnh", "Cổ phiếu giảm sâu"])
    aggregated = aggregate_title_sentiment_scores(scored["sentiment_score"].tolist())

    assert bundle.handoff["checkpoint_path"] == str(checkpoint_path.resolve())
    assert list(scored.columns) == [
        "title_raw",
        "title_clean",
        "sentiment_score",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
    ]
    assert np.allclose(scored["sentiment_score"].to_numpy(), np.array([0.3, -0.5], dtype=np.float32))
    assert aggregated["sentiment_score_count"] == 2.0
    assert aggregated["sentiment_missing_flag"] == 0.0
    assert aggregated["sentiment_positive_ratio"] == 0.5
    assert aggregated["sentiment_negative_ratio"] == 0.5


def test_phase2_phobert_inference_bundle_falls_back_without_java(tmp_path, monkeypatch):
    output_dir = tmp_path / "phase2_latest"
    tokenizer_dir = output_dir / "phobert" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "phobert" / "phobert.pt"
    jar_path = tmp_path / "VnCoreNLP-1.1.1.jar"
    jar_path.write_bytes(b"jar")
    torch.save({"model_state_dict": {"dummy": torch.tensor(1.0)}}, checkpoint_path)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phobert_model": "vinai/phobert-base-v2",
                "shared_hidden_dim": 256,
                "custom_heads": 4,
                "custom_dropout": 0.1,
                "freeze_phobert": False,
                "max_length": 16,
                "selection_metric": "macro_f1",
                "preprocessing": {
                    "lowercase": False,
                    "disable_punctuation_removal": False,
                    "use_vncorenlp": True,
                    "disable_vncorenlp": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "phobert" / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": 3,
                "best_selection_score": 0.81,
                "checkpoint_path": str(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            batch = len(texts)
            return {
                "input_ids": torch.ones((batch, max_length), dtype=torch.long),
                "attention_mask": torch.ones((batch, max_length), dtype=torch.long),
            }

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.loaded_state = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr("src.phase2.inference._load_tokenizer", lambda tokenizer_dir: FakeTokenizer())
    monkeypatch.setattr("src.phase2.inference.PhoBERTSentimentModel", FakeModel)
    monkeypatch.setattr(
        "src.phase2.inference.VnCoreNLPSegmenter",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("Java missing")),
    )

    bundle = load_phase2_phobert_inference_bundle(
        output_dir,
        device="cpu",
        vncorenlp_jar_path=jar_path,
    )

    assert bundle.segmenter is None
    assert bundle.preprocessing_config.segmentation == "none"