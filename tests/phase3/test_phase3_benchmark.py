from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from run_phase3_fusion_benchmark import main
from src.phase3 import (
    ChronosFusionRegressor,
    FusionTrainingConfig,
    build_phase2_phobert_news_feature_matrix,
)
from src.phase3.data import Phase3DatasetBundle


class _FakePhoBERTScorer:
    def score_titles(self, titles):
        rows = []
        for idx, _title in enumerate(titles):
            rows.append(
                {
                    "predicted_expected_value": 0.2 + 0.1 * idx,
                    "prob_negative": 0.1,
                    "prob_neutral": 0.2,
                    "prob_positive": 0.7,
                    "prediction_confidence": 0.7,
                    "attention_max": 0.5 + 0.1 * idx,
                }
            )
        return pd.DataFrame(rows)


def test_build_phase2_phobert_news_feature_matrix_aggregates_sentiment():
    frame = pd.DataFrame(
        {
            "news_titles_raw": [["title 1", "title 2"]],
            "news_published_at": [["2024-06-03 08:00:00", "2024-06-03 12:00:00"]],
            "news_cutoff": [pd.Timestamp("2024-06-04 08:00:00")],
            "timestamp": [pd.Timestamp("2024-06-04 08:00:00")],
        }
    )

    matrix, feature_names = build_phase2_phobert_news_feature_matrix(frame, scorer=_FakePhoBERTScorer())

    assert matrix.shape == (1, len(feature_names))
    values = dict(zip(feature_names, matrix[0].tolist(), strict=True))
    assert values["news_phobert_has_any"] == 1.0
    assert values["news_phobert_title_count"] == 2.0
    assert values["news_phobert_expected_mean"] > 0.0
    assert values["news_phobert_positive_prob_mean"] == pytest.approx(0.7)


def test_chronos_fusion_regressor_fits_tiny_dataset():
    market_train = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.4, 0.6]], dtype=np.float32)
    news_train = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    targets_train = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    market_val = np.asarray([[0.15, 0.25, 0.35]], dtype=np.float32)
    news_val = np.asarray([[0.25, 0.75]], dtype=np.float32)
    targets_val = np.asarray([0.02], dtype=np.float32)

    model = ChronosFusionRegressor(market_dim=3, news_dim=2, hidden_dim=16, news_hidden_dim=8)
    history = model.fit(
        market_train,
        news_train,
        targets_train,
        market_val,
        news_val,
        targets_val,
        FusionTrainingConfig(epochs=5, batch_size=2, learning_rate=1e-2, patience=3),
    )
    preds, diagnostics = model.predict(market_val, news_val, return_diagnostics=True)

    assert history["best_val_loss"] >= 0.0
    assert preds.shape == (1,)
    assert diagnostics["gate"].shape == (1,)


class _FakeChronos:
    def __init__(self, *args, **kwargs):
        self.d_model = 4
        self.pipeline = type(
            "_Pipeline",
            (),
            {
                "model": type(
                    "_Model",
                    (),
                    {"parameters": lambda self: []},
                )(),
            },
        )()

    def zero_shot_predict(self, close_windows, last_close, seed=42, horizon=1, aggregation="median"):
        del seed, horizon, aggregation
        return np.mean(close_windows, axis=1).astype(np.float32) / np.maximum(last_close.astype(np.float32), 1.0)

    def get_embeddings(self, close_windows, pooling="mean", recency_bias=2.0):
        del pooling, recency_bias
        means = np.mean(close_windows, axis=1)
        stds = np.std(close_windows, axis=1)
        maxs = np.max(close_windows, axis=1)
        mins = np.min(close_windows, axis=1)
        return np.stack([means, stds, maxs, mins], axis=1).astype(np.float32)


def test_phase3_benchmark_runner_smoke(tmp_path, monkeypatch):
    timestamps = pd.to_datetime(["2024-06-03", "2024-07-03", "2024-10-03"])
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "symbol": ["VCB", "VCB", "VCB"],
            "timestamp": timestamps,
            "split": ["train", "val", "test"],
            "target": [0.02, 0.01, 0.03],
            "news_count": [2, 1, 3],
            "close_window": [
                np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
                np.asarray([11.0, 12.0, 13.0], dtype=np.float64),
                np.asarray([12.0, 13.0, 14.0], dtype=np.float64),
            ],
            "market_window": [
                np.asarray([[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]], dtype=np.float32),
                np.asarray([[11.0, 12.0], [12.0, 13.0], [13.0, 14.0]], dtype=np.float32),
                np.asarray([[12.0, 13.0], [13.0, 14.0], [14.0, 15.0]], dtype=np.float32),
            ],
            "last_close": [12.0, 13.0, 14.0],
            "news_titles_raw": [
                ["VN-Index tang manh", "Vietcombank tang truong"],
                ["Thi truong giang co"],
                ["Vietcombank but pha", "Thanh khoan tang", "Khoi ngoai mua rong"],
            ],
            "news_published_at": [
                ["2024-06-02 08:00:00", "2024-06-03 07:00:00"],
                ["2024-07-02 08:00:00"],
                ["2024-10-01 08:00:00", "2024-10-02 08:00:00", "2024-10-03 07:00:00"],
            ],
            "news_cutoff": timestamps,
            "news_window_start": timestamps - pd.Timedelta(days=5),
        }
    )
    bundle = Phase3DatasetBundle(
        dataframe=frame,
        manifest={
            "row_count": 3,
            "news_coverage_ratio": 1.0,
            "avg_titles_per_sample": 2.0,
            "samples_without_news": 0,
            "purged_samples": 0,
        },
    )

    monkeypatch.setattr(
        "run_phase3_fusion_benchmark.assemble_phase3_datasets",
        lambda args: ({1: bundle}, [{"horizon_days": 1, "row_count": 3, "news_coverage_ratio": 1.0}]),
    )
    monkeypatch.setattr("run_phase3_fusion_benchmark.ChronosMarketPredictor", _FakeChronos)
    monkeypatch.setattr(
        "run_phase3_fusion_benchmark.resolve_phase1_best_model",
        lambda **kwargs: {"best_model": "Chronos Frozen Probe", "source": "test"},
    )
    monkeypatch.setattr(
        "run_phase3_fusion_benchmark.build_phase2_phobert_news_feature_matrix",
        lambda frame, phase2_output_dir, device: (
            np.asarray([[0.1, 0.7], [0.0, 0.4], [0.2, 0.9]], dtype=np.float32),
            ["news_phobert_expected_mean", "news_phobert_confidence_mean"],
        ),
    )

    output_dir = tmp_path / "phase3_benchmark"
    main([
        "--symbols", "VCB",
        "--horizons", "1",
        "--output-dir", str(output_dir),
        "--epochs", "3",
        "--batch-size", "2",
        "--learning-rate", "1e-2",
    ])

    assert (output_dir / "benchmark_summary.csv").exists()
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "data_overview.csv").exists()
    assert (output_dir / "1d" / "benchmark_metrics.csv").exists()
    assert (output_dir / "1d" / "benchmark_predictions.csv").exists()
    assert (output_dir / "1d" / "model_comparison.csv").exists()
    assert (output_dir / "1d" / "benchmark_test_metrics.png").exists()
    assert (output_dir / "figures" / "data_overview.png").exists()

    summary = pd.read_csv(output_dir / "benchmark_summary.csv")
    assert int(summary.loc[0, "horizon_days"]) == 1
    assert summary.loc[0, "resolved_phase1_baseline"] == "Chronos Frozen Probe"

    model_comparison = pd.read_csv(output_dir / "model_comparison.csv")
    assert model_comparison["model_name"].tolist() == ["Chronos Frozen Probe", "Chronos Frozen Probe + CMTF"]

    explainability = json.loads((output_dir / "1d" / "explainability_summary.json").read_text(encoding="utf-8"))
    assert explainability["baseline_model_name"] == "Chronos Frozen Probe"
    assert explainability["news_feature_source"] == "phase2-phobert"
    assert "news_feature_names" in explainability


def test_phase3_benchmark_summary_uses_lowest_composite_score(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "sample_id": ["a"],
            "symbol": ["VCB"],
            "timestamp": [pd.Timestamp("2024-10-03")],
            "split": ["test"],
            "target": [0.03],
            "news_count": [2],
            "close_window": [np.asarray([12.0, 13.0, 14.0], dtype=np.float64)],
            "last_close": [14.0],
            "news_titles_raw": [["Vietcombank but pha"]],
            "news_published_at": [["2024-10-03 07:00:00"]],
            "news_cutoff": [pd.Timestamp("2024-10-03")],
            "news_window_start": [pd.Timestamp("2024-09-28")],
        }
    )
    bundle = Phase3DatasetBundle(
        dataframe=frame,
        manifest={
            "row_count": 1,
            "news_coverage_ratio": 1.0,
            "avg_titles_per_sample": 1.0,
            "samples_without_news": 0,
            "purged_samples": 0,
        },
    )

    monkeypatch.setattr(
        "run_phase3_fusion_benchmark.assemble_phase3_datasets",
        lambda args: ({1: bundle}, [{"horizon_days": 1, "row_count": 1, "news_coverage_ratio": 1.0}]),
    )
    monkeypatch.setattr("run_phase3_fusion_benchmark.ChronosMarketPredictor", _FakeChronos)
    monkeypatch.setattr(
        "run_phase3_fusion_benchmark.resolve_phase1_best_model",
        lambda **kwargs: {"best_model": "Chronos Frozen Probe", "source": "test"},
    )
    monkeypatch.setattr(
        "run_phase3_fusion_benchmark._fit_models_for_horizon",
        lambda frame, horizon, chronos, args, baseline_name: (
            pd.DataFrame(
                [
                    {"model": "Chronos Frozen Probe", "split": "test", "horizon_days": 1, "CompositeScore": 0.19, "RMSE": 0.01, "DA%": 60.0, "IC": 0.2},
                    {"model": "Chronos Frozen Probe + CMTF", "split": "test", "horizon_days": 1, "CompositeScore": 0.18, "RMSE": 0.011, "DA%": 55.0, "IC": 0.15},
                ]
            ),
            pd.DataFrame({"sample_id": ["a"]}),
            {"baseline_model_name": baseline_name, "news_feature_names": []},
        ),
    )

    output_dir = tmp_path / "phase3_benchmark_summary"
    main([
        "--symbols", "VCB",
        "--horizons", "1",
        "--output-dir", str(output_dir),
    ])

    summary = pd.read_csv(output_dir / "benchmark_summary.csv")
    assert summary.loc[0, "best_model"] == "Chronos Frozen Probe + CMTF"
    assert float(summary.loc[0, "best_composite_score"]) == 0.18