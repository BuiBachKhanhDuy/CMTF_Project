"""Reusable Phase 2 handoff artifacts for downstream phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PHASE2_OUTPUT_ROOT = Path("outputs/phase2/latest")

PHASE3_HANDOFF_FILENAME = "phase3_phobert_handoff.json"


def _portable_path(path: Path, *, relative_to: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(relative_to))
        except ValueError:
            return str(path)
    return str(path)


def build_phase2_phobert_handoff(output_dir: str | Path = DEFAULT_PHASE2_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_dir)
    root_abs = root.resolve()
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    training_summary = json.loads((root / "phobert" / "training_summary.json").read_text(encoding="utf-8"))

    checkpoint_path = Path(training_summary["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root_abs / checkpoint_path).resolve()

    tokenizer_dir = root_abs / "phobert" / "tokenizer"
    preprocessing_args = run_config.get("preprocessing", {})
    return {
        "variant": "phobert",
        "phase2_output_dir": str(root),
        "checkpoint_path": _portable_path(checkpoint_path, relative_to=root_abs),
        "tokenizer_dir": _portable_path(tokenizer_dir, relative_to=root_abs),
        "model_name": str(run_config.get("phobert_model", "vinai/phobert-base-v2")),
        "projection_dim": int(run_config.get("shared_hidden_dim", 256)),
        "num_heads": int(run_config.get("custom_heads", 4)),
        "dropout": float(run_config.get("custom_dropout", 0.1)),
        "freeze_backbone": bool(run_config.get("freeze_phobert", False)),
        "max_length": int(run_config.get("max_length", 128)),
        "selection_metric": str(run_config.get("selection_metric", "macro_f1")),
        "best_selection_score": float(training_summary.get("best_selection_score", 0.0)),
        "best_epoch": int(training_summary.get("best_epoch", 0)),
        "preprocessing": {
            "lowercase": bool(preprocessing_args.get("lowercase", False)),
            "remove_punctuation": not bool(preprocessing_args.get("disable_punctuation_removal", False)),
            "segmentation": "vncorenlp" if bool(preprocessing_args.get("use_vncorenlp", False)) and not bool(preprocessing_args.get("disable_vncorenlp", False)) else "none",
            "normalize_unicode_form": "NFC",
            "strip_html": True,
            "collapse_whitespace": True,
        },
    }


def save_phase2_phobert_handoff(output_dir: str | Path = DEFAULT_PHASE2_OUTPUT_ROOT) -> Path:
    root = Path(output_dir)
    handoff = build_phase2_phobert_handoff(root)
    handoff_path = root / PHASE3_HANDOFF_FILENAME
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    return handoff_path


def resolve_phase2_phobert_handoff(output_dir: str | Path = DEFAULT_PHASE2_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_dir)
    handoff_path = root / PHASE3_HANDOFF_FILENAME
    if handoff_path.exists():
        return json.loads(handoff_path.read_text(encoding="utf-8"))
    return build_phase2_phobert_handoff(root)