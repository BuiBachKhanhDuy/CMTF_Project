from __future__ import annotations

import pandas as pd

from run_phase1_market_benchmark import _build_phase1_comparison_frame, _build_phase1_summary_row


def test_phase1_reporting_builds_comparison_and_summary():
    results_df = pd.DataFrame(
        [
            {"Experiment": "Chronos Zero-Shot", "Symbol": "AVG", "TargetHorizonD": 1, "CompositeScore": 0.11, "RMSE": 0.02, "DA%": 40.0, "IC": 0.1},
            {"Experiment": "Chronos Frozen Probe", "Symbol": "AVG", "TargetHorizonD": 1, "CompositeScore": 0.19, "RMSE": 0.01, "DA%": 60.0, "IC": 0.2},
            {"Experiment": "VCB-only row", "Symbol": "VCB", "TargetHorizonD": 1, "CompositeScore": 0.05, "RMSE": 0.03, "DA%": 35.0, "IC": 0.0},
        ]
    )

    comparison_df = _build_phase1_comparison_frame(results_df)

    assert comparison_df["model_name"].tolist() == ["Chronos Zero-Shot", "Chronos Frozen Probe"]
    assert comparison_df["split"].tolist() == ["test", "test"]

    summary_row = _build_phase1_summary_row(comparison_df, horizon_days=1)
    assert summary_row["best_model"] == "Chronos Frozen Probe"
    assert summary_row["best_composite_score"] == 0.19
