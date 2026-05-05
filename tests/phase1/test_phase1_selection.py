from __future__ import annotations

import pandas as pd

from src.phase1.selection import resolve_phase1_best_model


def test_resolve_phase1_best_model_falls_back_to_legacy_csv(tmp_path):
    pd.DataFrame(
        [
            {"Experiment": "Chronos Zero-Shot", "Symbol": "AVG", "TargetHorizonD": 1, "CompositeScore": 0.16, "RMSE": 0.02, "DA%": 49.0, "IC": 0.1},
            {"Experiment": "Chronos LoRA", "Symbol": "AVG", "TargetHorizonD": 1, "CompositeScore": 0.24, "RMSE": 0.018, "DA%": 50.0, "IC": 0.2},
            {"Experiment": "Chronos Frozen Probe", "Symbol": "AVG", "TargetHorizonD": 1, "CompositeScore": 0.23, "RMSE": 0.019, "DA%": 48.0, "IC": 0.15},
        ]
    ).to_csv(tmp_path / "phase1_market_benchmark_1d.csv", index=False)

    resolved = resolve_phase1_best_model(horizon_days=1, results_root=tmp_path)

    assert resolved["best_model"] == "Chronos Frozen Probe"
    assert resolved["source"] == "legacy_benchmark_csv"
    assert resolved["best_composite_score"] == 0.23