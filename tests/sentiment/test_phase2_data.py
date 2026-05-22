from __future__ import annotations

import pandas as pd

from src.sentiment import (
    DatasetLoadConfig,
    PreprocessingConfig,
    apply_preprocessing,
    build_preprocessing_report,
    deterministic_stratified_split,
    load_phase2_dataset,
    sample_preprocessing_examples,
)


def test_load_phase2_dataset_maps_source_repo_numeric_labels(tmp_path):
    df = pd.DataFrame(
        {
            "title": [
                "A good article",
                "A neutral article",
                "A bad article",
            ],
            "label": [1, 2, 3],
        }
    )
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    bundle = load_phase2_dataset(DatasetLoadConfig(dataset_path=csv_path))

    assert bundle.dataframe["label_name"].tolist() == ["negative", "neutral", "positive"]
    assert bundle.dataframe["target_value"].tolist() == [-1.0, 0.0, 1.0]
    assert bundle.manifest.label_scheme == "source_repo"


def test_load_phase2_dataset_uses_filename_splits_when_available(tmp_path):
    train_df = pd.DataFrame({"title": ["t1", "t2", "t3"], "label": [1, 1, 2]})
    val_df = pd.DataFrame({"title": ["v1", "v2", "v3"], "label": [2, 3, 3]})
    test_df = pd.DataFrame({"title": ["e1", "e2", "e3"], "label": [1, 2, 3]})
    train_df.to_csv(tmp_path / "train.csv", index=False)
    val_df.to_csv(tmp_path / "val.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)

    bundle = load_phase2_dataset(DatasetLoadConfig(dataset_path=tmp_path))

    assert bundle.manifest.generated_split is False
    assert bundle.manifest.split_column == "filename"
    assert set(bundle.dataframe["split"].unique()) == {"train", "val", "test"}


def test_load_phase2_dataset_skips_unlabeled_auxiliary_files(tmp_path):
    labeled_df = pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "label": [1, 2, 3],
        }
    )
    labeled_df.to_csv(tmp_path / "raw_data.csv", index=False)
    (tmp_path / "notes.txt").write_text("train_0\nauxiliary row\n", encoding="utf-8")

    bundle = load_phase2_dataset(DatasetLoadConfig(dataset_path=tmp_path))

    assert bundle.manifest.row_count == 3
    assert bundle.manifest.source_files == ("raw_data.csv",)
    assert bundle.dataframe["label_name"].tolist() == ["negative", "neutral", "positive"]


def test_deterministic_stratified_split_is_reproducible():
    labels = pd.Series(["negative"] * 10 + ["neutral"] * 10 + ["positive"] * 10)

    split_a = deterministic_stratified_split(labels, random_seed=7)
    split_b = deterministic_stratified_split(labels, random_seed=7)

    assert split_a.tolist() == split_b.tolist()
    assert set(split_a.tolist()) == {"train", "val", "test"}


def test_apply_preprocessing_adds_clean_columns_and_duplicates():
    df = pd.DataFrame(
        {
            "title_raw": [
                "<b>VCB</b> tăng mạnh!!!",
                "VCB tăng mạnh",
                "   ",
            ],
            "label_name": ["positive", "positive", "neutral"],
        }
    )

    processed = apply_preprocessing(df, PreprocessingConfig())

    assert processed["title_clean"].tolist() == ["VCB tăng mạnh", "VCB tăng mạnh", ""]
    assert processed["duplicate_clean_title"].tolist() == [True, True, False]
    assert processed["clean_is_empty"].tolist() == [False, False, True]


def test_preprocessing_report_and_examples_are_deterministic():
    df = pd.DataFrame(
        {
            "title_raw": [
                "AAA tăng mạnh!!!",
                "BBB đi ngang",
                "CCC giảm sâu???",
                "DDD bứt phá",
            ],
            "label_name": ["positive", "neutral", "negative", "positive"],
        }
    )
    processed = apply_preprocessing(df, PreprocessingConfig())

    report = build_preprocessing_report(processed)
    examples_a = sample_preprocessing_examples(processed, n_samples=2, random_seed=11)
    examples_b = sample_preprocessing_examples(processed, n_samples=2, random_seed=11)

    assert report["row_count"] == 4
    assert report["label_distribution"] == {"negative": 1, "neutral": 1, "positive": 2}
    assert examples_a.equals(examples_b)