from __future__ import annotations

import io
import json
import zipfile

import pandas as pd

from run_sentiment_benchmark import _resolve_dataset_path, main


def _build_fake_dataset_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(
            "Vietnamese-stock-article-classification-main/Dataset/train.csv",
            "title,label\nA,1\nB,2\nC,3\n",
        )
        archive.writestr(
            "Vietnamese-stock-article-classification-main/Dataset/val.csv",
            "title,label\nD,1\nE,2\nF,3\n",
        )
        archive.writestr(
            "Vietnamese-stock-article-classification-main/Dataset/test.csv",
            "title,label\nG,1\nH,2\nI,3\n",
        )
    return buffer.getvalue()


def test_resolve_dataset_path_downloads_archive_once(tmp_path, monkeypatch):
    archive_bytes = _build_fake_dataset_archive()
    calls = {"count": 0}

    def fake_download(download_url: str) -> bytes:
        calls["count"] += 1
        assert download_url == "https://example.com/dataset.zip"
        return archive_bytes

    monkeypatch.setattr(
        "run_sentiment_benchmark._download_dataset_archive_bytes",
        fake_download,
    )

    dataset_dir = tmp_path / "cache_dataset" / "Dataset"
    resolved_first = _resolve_dataset_path(
        dataset_path=dataset_dir,
        download_url="https://example.com/dataset.zip",
    )
    resolved_second = _resolve_dataset_path(
        dataset_path=dataset_dir,
        download_url="https://example.com/dataset.zip",
    )

    assert resolved_first == dataset_dir
    assert resolved_second == dataset_dir
    assert calls["count"] == 1
    assert (dataset_dir / "train.csv").exists()
    assert (dataset_dir / "val.csv").exists()
    assert (dataset_dir / "test.csv").exists()


def test_phase2_runner_custom_variant_smoke(tmp_path):
    rows = []
    split_plan = ["train", "train", "train", "train", "val", "test"]
    label_map = [("negative", 3), ("neutral", 2), ("positive", 1)]
    title_templates = {
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
    for label_name, numeric_label in label_map:
        for idx, split_name in enumerate(split_plan):
            rows.append(
                {
                    "title": title_templates[label_name][idx % len(title_templates[label_name])],
                    "label": numeric_label,
                    "split": split_name,
                }
            )

    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    output_dir = tmp_path / "phase2_artifacts"

    main(
        [
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--variant",
            "custom",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--max-length",
            "8",
            "--shared-hidden-dim",
            "16",
            "--custom-layers",
            "1",
            "--custom-heads",
            "4",
            "--custom-ffn-dim",
            "32",
        ]
    )

    assert (output_dir / "dataset_manifest.json").exists()
    assert (output_dir / "phase2_preprocessed.csv").exists()
    assert (output_dir / "comparison_metrics.csv").exists()
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "benchmark_summary.csv").exists()
    assert (output_dir / "data_overview.csv").exists()
    assert (output_dir / "custom_transformer" / "test_predictions.csv").exists()
    assert (output_dir / "figures" / "test_metric_comparison.png").exists()
    assert (output_dir / "figures" / "model_comparison_test.png").exists()
    assert (output_dir / "figures" / "data_overview.png").exists()

    training_summary = json.loads(
        (output_dir / "custom_transformer" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert training_summary["best_selection_metric"] == "macro_f1"
    assert isinstance(training_summary["best_selection_score"], float)

    benchmark_summary = pd.read_csv(output_dir / "benchmark_summary.csv")
    assert "best_model" in benchmark_summary.columns