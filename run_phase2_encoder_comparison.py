"""Run Phase 2 title-level encoder comparison with shared cross-attention."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

from src.common import (
    JAVA_RUNTIME_ROOT,
    PHASE2_DATASET_ROOT,
    PHASE2_OUTPUT_ROOT,
    VNCORENLP_JAR_PATH,
    VNCORENLP_ROOT,
    VNCORENLP_WORDSEGMENTER_ROOT,
    plot_metric_panels,
    select_best_row,
    write_json,
)
from src.phase2 import (
    CustomTransformerSentimentModel,
    DatasetLoadConfig,
    PhoBERTSentimentModel,
    PreprocessingConfig,
    TrainingConfig,
    VnCoreNLPSegmenter,
    apply_preprocessing,
    build_custom_text_datasets,
    build_dataloaders,
    build_phobert_text_datasets,
    build_prediction_frame,
    build_preprocessing_report,
    compute_title_level_metrics,
    evaluate_sentiment_model,
    load_phase2_dataset,
    make_class_weights,
    plot_confusion_matrix,
    plot_expected_value_scatter,
    plot_length_comparison,
    plot_metric_comparison,
    plot_preprocessing_examples_table,
    plot_preprocessing_label_distribution,
    plot_residual_histogram,
    plot_score_distribution_by_class,
    plot_training_curves,
    save_phase2_phobert_handoff,
    sample_preprocessing_examples,
    save_dataset_manifest,
    train_sentiment_model,
)

DEFAULT_DATASET_ARCHIVE_URL = (
    "https://codeload.github.com/209sontung/"
    "Vietnamese-stock-article-classification/zip/refs/heads/main"
)
DEFAULT_VNCORENLP_JAR_URL = (
    "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
    "VnCoreNLP-1.1.1.jar"
)
DEFAULT_VNCORENLP_VOCAB_URL = (
    "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/"
    "wordsegmenter/vi-vocab"
)
DEFAULT_VNCORENLP_RDR_URL = (
    "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/"
    "wordsegmenter/wordsegmenter.rdr"
)
DEFAULT_WINDOWS_JRE_URL = (
    "https://github.com/adoptium/temurin21-binaries/releases/download/"
    "jdk-21.0.11%2B10/OpenJDK21U-jre_x64_windows_hotspot_21.0.11_10.zip"
)
SUPPORTED_DATASET_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 encoder comparison")
    parser.add_argument("--dataset-path", type=str, default=str(PHASE2_DATASET_ROOT))
    parser.add_argument("--dataset-download-url", type=str, default=DEFAULT_DATASET_ARCHIVE_URL)
    parser.add_argument("--skip-dataset-download", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(PHASE2_OUTPUT_ROOT))
    parser.add_argument("--variant", choices=["custom", "phobert", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--phobert-backbone-learning-rate", type=float, default=5e-7)
    parser.add_argument("--selection-metric", choices=["macro_f1", "rmse"], default="macro_f1")
    parser.add_argument("--shared-hidden-dim", type=int, default=256)
    parser.add_argument("--custom-layers", type=int, default=3)
    parser.add_argument("--custom-heads", type=int, default=4)
    parser.add_argument("--custom-ffn-dim", type=int, default=512)
    parser.add_argument("--custom-dropout", type=float, default=0.1)
    parser.add_argument("--phobert-model", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--freeze-phobert", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--disable-punctuation-removal", action="store_true")
    parser.add_argument("--use-vncorenlp", action="store_true")
    parser.add_argument("--disable-vncorenlp", action="store_true")
    parser.add_argument("--vncorenlp-jar", type=str, default="")
    return parser.parse_args(argv)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _has_supported_dataset_files(dataset_path: Path) -> bool:
    if dataset_path.is_file():
        return dataset_path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
    if not dataset_path.exists():
        return False
    return any(
        child.is_file() and child.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        for child in dataset_path.rglob("*")
    )


def _download_dataset_archive_bytes(download_url: str) -> bytes:
    with urlopen(download_url) as response:
        return response.read()


def _download_dataset_once(dataset_path: Path, download_url: str) -> Path:
    if dataset_path.suffix:
        raise ValueError(
            "Auto-downloaded Phase 2 dataset paths must point to a directory, not a file"
        )

    temp_dir = dataset_path.parent / f".{dataset_path.name}_download"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        archive_bytes = _download_dataset_archive_bytes(download_url)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and "/Dataset/" in member.filename
            ]
            if not members:
                raise ValueError(
                    "Downloaded archive does not contain a Dataset/ directory with supported files"
                )

            for member in members:
                relative_name = member.filename.split("/Dataset/", 1)[1]
                if not relative_name:
                    continue
                target_path = temp_dir / relative_name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists():
            if dataset_path.is_dir():
                shutil.rmtree(dataset_path)
            else:
                dataset_path.unlink()
        shutil.move(str(temp_dir), str(dataset_path))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return dataset_path


def _resolve_dataset_path(
    dataset_path: str | Path,
    download_url: str = DEFAULT_DATASET_ARCHIVE_URL,
    skip_download: bool = False,
) -> Path:
    resolved = Path(dataset_path)
    if _has_supported_dataset_files(resolved):
        print(f"Using cached Phase 2 dataset at {resolved}")
        return resolved

    if resolved.exists():
        is_empty_dir = resolved.is_dir() and not any(resolved.iterdir())
        if not is_empty_dir:
            raise FileNotFoundError(
                f"Dataset path exists but has no supported tabular files: {resolved}"
            )

    if skip_download:
        raise FileNotFoundError(f"Dataset path not found: {resolved}")

    print(f"Dataset not found at {resolved}; downloading once from {download_url}")
    return _download_dataset_once(resolved, download_url)


def _download_binary(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def _download_file_once(url: str, target_path: Path) -> Path:
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(_download_binary(url))
    return target_path


def _prepend_to_path(path_entry: Path) -> None:
    entry = str(path_entry)
    current = os.environ.get("PATH", "")
    if current.startswith(entry + os.pathsep) or current == entry:
        return
    os.environ["PATH"] = entry + os.pathsep + current if current else entry


def _activate_java_binary(java_binary: Path) -> Path:
    if not java_binary.exists():
        raise FileNotFoundError(f"Java binary not found: {java_binary}")
    _prepend_to_path(java_binary.parent)
    return java_binary


def _find_local_java_binary() -> Path | None:
    executable_name = "java.exe" if os.name == "nt" else "java"
    existing = shutil.which("java")
    if existing is not None:
        return Path(existing)
    if not JAVA_RUNTIME_ROOT.exists():
        return None
    matches = sorted(JAVA_RUNTIME_ROOT.rglob(executable_name))
    return matches[0] if matches else None


def _ensure_java_runtime() -> Path:
    existing = _find_local_java_binary()
    if existing is not None:
        return _activate_java_binary(existing)

    archive_name = Path(DEFAULT_WINDOWS_JRE_URL).name
    archive_path = JAVA_RUNTIME_ROOT / archive_name
    JAVA_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    _download_file_once(DEFAULT_WINDOWS_JRE_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(JAVA_RUNTIME_ROOT)

    java_binary = _find_local_java_binary()
    if java_binary is None:
        raise FileNotFoundError("Failed to provision a local Java runtime for VnCoreNLP")
    return _activate_java_binary(java_binary)


def _ensure_vncorenlp_assets(jar_path: str | Path | None = None) -> Path:
    if jar_path:
        resolved = Path(jar_path)
        if not resolved.exists():
            raise FileNotFoundError(f"VnCoreNLP jar path not found: {resolved}")
        return resolved

    _download_file_once(DEFAULT_VNCORENLP_JAR_URL, VNCORENLP_JAR_PATH)
    _download_file_once(DEFAULT_VNCORENLP_VOCAB_URL, VNCORENLP_WORDSEGMENTER_ROOT / "vi-vocab")
    _download_file_once(DEFAULT_VNCORENLP_RDR_URL, VNCORENLP_WORDSEGMENTER_ROOT / "wordsegmenter.rdr")
    return VNCORENLP_JAR_PATH


def _should_use_vncorenlp(args: argparse.Namespace) -> bool:
    if bool(args.disable_vncorenlp):
        return False
    return bool(args.use_vncorenlp or args.variant in {"phobert", "both"})


def _prepare_preprocessing(args: argparse.Namespace) -> tuple[PreprocessingConfig, Any | None]:
    config = PreprocessingConfig(
        lowercase=bool(args.lowercase),
        remove_punctuation=not bool(args.disable_punctuation_removal),
        segmentation="vncorenlp" if _should_use_vncorenlp(args) else "none",
    )
    if config.segmentation != "vncorenlp":
        return config, None
    _ensure_java_runtime()
    jar_path = _ensure_vncorenlp_assets(args.vncorenlp_jar or None)
    return config, VnCoreNLPSegmenter(jar_path)


def _save_preprocessing_artifacts(
    processed_df: pd.DataFrame,
    dataset_manifest: Any,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    processed_df.to_csv(output_dir / "phase2_preprocessed.csv", index=False)
    save_dataset_manifest(dataset_manifest, output_dir / "dataset_manifest.json")

    report = build_preprocessing_report(processed_df)
    examples = sample_preprocessing_examples(processed_df)
    write_json(output_dir / "preprocessing_report.json", report)
    examples.to_csv(output_dir / "preprocessing_examples.csv", index=False)
    plot_preprocessing_label_distribution(processed_df, figures_dir / "preprocess_label_distribution.png")
    plot_length_comparison(processed_df, figures_dir / "preprocess_length_comparison.png")
    plot_preprocessing_examples_table(examples, figures_dir / "preprocess_examples_table.png")


def _training_config_from_args(args: argparse.Namespace, checkpoint_name: str, device: str) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        backbone_learning_rate=float(args.phobert_backbone_learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        device=device,
        checkpoint_name=checkpoint_name,
        selection_metric=str(args.selection_metric),
    )


def _save_variant_metrics(
    model_name: str,
    split_name: str,
    metrics: dict[str, Any],
    checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "split": split_name,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "pearson": float(metrics["pearson"]),
        "spearman": float(metrics["spearman"]),
        "support": int(metrics["support"]),
        "checkpoint_path": checkpoint_path or "",
    }


def _build_phase2_data_overview(processed_df: pd.DataFrame) -> pd.DataFrame:
    label_names = ["negative", "neutral", "positive"]
    rows: list[dict[str, Any]] = []
    for split_name, split_df in processed_df.groupby("split", sort=True):
        row = {
            "split": str(split_name),
            "row_count": int(len(split_df)),
            "avg_raw_token_len": float(split_df["raw_token_len"].mean()) if "raw_token_len" in split_df.columns else 0.0,
            "avg_clean_token_len": float(split_df["clean_token_len"].mean()) if "clean_token_len" in split_df.columns else 0.0,
        }
        label_counts = split_df["label_name"].value_counts() if "label_name" in split_df.columns else pd.Series(dtype=int)
        for label_name in label_names:
            row[f"{label_name}_count"] = int(label_counts.get(label_name, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_phase2_benchmark_summary(
    metrics_df: pd.DataFrame,
    *,
    selection_metric: str,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    higher_is_better = selection_metric != "rmse"
    rows: list[dict[str, Any]] = []
    for split_name, split_df in metrics_df.groupby("split", sort=True):
        best_row = select_best_row(split_df, score_col=selection_metric, higher_is_better=higher_is_better)
        rows.append(
            {
                "split": str(split_name),
                "selection_metric": selection_metric,
                "best_model": str(best_row["model_name"]),
                "best_selection_score": float(best_row[selection_metric]),
                "macro_f1": float(best_row["macro_f1"]),
                "accuracy": float(best_row["accuracy"]),
                "rmse": float(best_row["rmse"]),
                "mae": float(best_row["mae"]),
            }
        )
    return pd.DataFrame(rows)


def _run_variant(
    *,
    model_name: str,
    model: torch.nn.Module,
    datasets: dict[str, Any],
    dataloaders: dict[str, Any],
    class_weights: torch.Tensor,
    output_dir: Path,
    training_config: TrainingConfig,
) -> list[dict[str, Any]]:
    variant_dir = output_dir / model_name
    figures_dir = variant_dir / "figures"
    variant_dir.mkdir(parents=True, exist_ok=True)

    training_artifacts = train_sentiment_model(
        model,
        dataloaders,
        training_config,
        class_weights=class_weights,
        output_dir=variant_dir,
    )
    plot_training_curves(training_artifacts.history, figures_dir / "training_curves.png")

    summary_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        metrics = evaluate_sentiment_model(model, dataloaders[split_name], device=training_config.device)
        prediction_df = build_prediction_frame(
            datasets[split_name].frame,
            probabilities=metrics["probabilities"],
            expected_values=metrics["expected_values"],
            attention_weights=metrics["attention_weights"],
        )
        prediction_df.to_csv(variant_dir / f"{split_name}_predictions.csv", index=False)

        plot_confusion_matrix(metrics["confusion_matrix"], figures_dir / f"{split_name}_confusion_matrix.png")
        plot_expected_value_scatter(
            metrics["target_values"],
            metrics["expected_values"],
            figures_dir / f"{split_name}_expected_value_scatter.png",
        )
        plot_residual_histogram(
            metrics["target_values"],
            metrics["expected_values"],
            figures_dir / f"{split_name}_residual_histogram.png",
        )
        plot_score_distribution_by_class(
            prediction_df,
            figures_dir / f"{split_name}_score_distribution.png",
        )
        summary_rows.append(
            _save_variant_metrics(
                model_name=model_name,
                split_name=split_name,
                metrics=metrics,
                checkpoint_path=training_artifacts.checkpoint_path,
            )
        )

        serializable_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (float, int, np.floating, np.integer))
        }
        write_json(variant_dir / f"{split_name}_metrics.json", serializable_metrics)

    write_json(
        variant_dir / "training_summary.json",
        {
            "best_epoch": training_artifacts.best_epoch,
            "best_val_rmse": training_artifacts.best_val_rmse,
            "best_selection_metric": training_artifacts.best_selection_metric,
            "best_selection_score": training_artifacts.best_selection_score,
            "checkpoint_path": training_artifacts.checkpoint_path,
            "history": training_artifacts.history,
        },
    )
    return summary_rows


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_global_seed(int(args.seed))
    dataset_path = _resolve_dataset_path(
        dataset_path=args.dataset_path,
        download_url=args.dataset_download_url,
        skip_download=bool(args.skip_dataset_download),
    )

    dataset_bundle = load_phase2_dataset(
        DatasetLoadConfig(dataset_path=dataset_path, random_seed=int(args.seed))
    )
    preprocessing_config, segmenter = _prepare_preprocessing(args)
    processed_df = apply_preprocessing(
        dataset_bundle.dataframe,
        preprocessing_config,
        segmenter=segmenter,
    )

    _save_preprocessing_artifacts(
        processed_df=processed_df,
        dataset_manifest=dataset_bundle.manifest,
        output_dir=output_dir,
    )
    write_json(
        output_dir / "run_config.json",
        {
            "dataset_path": str(dataset_path),
            "dataset_download_url": args.dataset_download_url,
            "variant": args.variant,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "max_length": int(args.max_length),
            "min_freq": int(args.min_freq),
            "device": device,
            "phobert_backbone_learning_rate": float(args.phobert_backbone_learning_rate),
            "selection_metric": args.selection_metric,
            "shared_hidden_dim": int(args.shared_hidden_dim),
            "custom_layers": int(args.custom_layers),
            "custom_heads": int(args.custom_heads),
            "custom_ffn_dim": int(args.custom_ffn_dim),
            "custom_dropout": float(args.custom_dropout),
            "phobert_model": args.phobert_model,
            "freeze_phobert": bool(args.freeze_phobert),
            "preprocessing": vars(args),
        },
    )

    data_overview_df = _build_phase2_data_overview(processed_df)
    if not data_overview_df.empty:
        data_overview_df.to_csv(output_dir / "data_overview.csv", index=False)
        plot_metric_panels(
            data_overview_df,
            label_col="split",
            metric_cols=("row_count", "avg_raw_token_len", "avg_clean_token_len"),
            save_path=output_dir / "figures" / "data_overview.png",
            title="Phase 2 Data Overview",
        )

    metrics_rows: list[dict[str, Any]] = []
    class_weights = make_class_weights(processed_df[processed_df["split"] == "train"]["label_id"])

    if args.variant in {"custom", "both"}:
        custom_datasets, vocab = build_custom_text_datasets(
            processed_df,
            max_length=int(args.max_length),
            min_freq=int(args.min_freq),
        )
        custom_dataloaders = build_dataloaders(custom_datasets, batch_size=int(args.batch_size))
        custom_model = CustomTransformerSentimentModel(
            vocab_size=len(vocab),
            hidden_dim=int(args.shared_hidden_dim),
            num_layers=int(args.custom_layers),
            num_heads=int(args.custom_heads),
            feedforward_dim=int(args.custom_ffn_dim),
            dropout=float(args.custom_dropout),
            max_length=int(args.max_length),
        )
        vocab.save_json(output_dir / "custom_transformer" / "vocab.json")
        metrics_rows.extend(
            _run_variant(
                model_name="custom_transformer",
                model=custom_model,
                datasets=custom_datasets,
                dataloaders=custom_dataloaders,
                class_weights=class_weights,
                output_dir=output_dir,
                training_config=TrainingConfig(
                    batch_size=int(args.batch_size),
                    epochs=int(args.epochs),
                    learning_rate=float(args.learning_rate),
                    backbone_learning_rate=None,
                    weight_decay=float(args.weight_decay),
                    patience=int(args.patience),
                    device=device,
                    checkpoint_name="custom_transformer.pt",
                    selection_metric=str(args.selection_metric),
                ),
            )
        )

    if args.variant in {"phobert", "both"}:
        phobert_datasets, tokenizer = build_phobert_text_datasets(
            processed_df,
            tokenizer_name=args.phobert_model,
            max_length=int(args.max_length),
        )
        phobert_dataloaders = build_dataloaders(phobert_datasets, batch_size=int(args.batch_size))
        phobert_model = PhoBERTSentimentModel(
            model_name=args.phobert_model,
            projection_dim=int(args.shared_hidden_dim),
            num_heads=int(args.custom_heads),
            dropout=float(args.custom_dropout),
            freeze_backbone=bool(args.freeze_phobert),
        )
        tokenizer.save_pretrained(output_dir / "phobert" / "tokenizer")
        metrics_rows.extend(
            _run_variant(
                model_name="phobert",
                model=phobert_model,
                datasets=phobert_datasets,
                dataloaders=phobert_dataloaders,
                class_weights=class_weights,
                output_dir=output_dir,
                training_config=_training_config_from_args(args, "phobert.pt", device),
            )
        )
        save_phase2_phobert_handoff(output_dir)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "comparison_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False)
    benchmark_summary_df = _build_phase2_benchmark_summary(
        metrics_df,
        selection_metric=str(args.selection_metric),
    )
    if not benchmark_summary_df.empty:
        benchmark_summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)
    if not metrics_df.empty:
        for split_name in metrics_df["split"].unique().tolist():
            plot_metric_comparison(
                metrics_df[metrics_df["split"] == split_name],
                output_dir / "figures" / f"{split_name}_metric_comparison.png",
            )
            plot_metric_panels(
                metrics_df[metrics_df["split"] == split_name],
                label_col="model_name",
                metric_cols=("macro_f1", "accuracy", "rmse", "mae"),
                save_path=output_dir / "figures" / f"model_comparison_{split_name}.png",
                title=f"Phase 2 Model Comparison ({split_name})",
            )


if __name__ == "__main__":
    main()