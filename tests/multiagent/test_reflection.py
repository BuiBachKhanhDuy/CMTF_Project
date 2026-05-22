"""Tests for offline reflection policy updates."""

from __future__ import annotations

import pandas as pd

from src.multiagent.reflection import DEFAULT_POLICY, update_policy_from_history


def test_reflection_requires_min_samples():
    df = pd.DataFrame(
        {
            "action": ["long", "short"],
            "fused_score": [0.02, -0.02],
            "realized_return": [0.01, 0.01],
        }
    )
    out = update_policy_from_history(df, min_samples=10)
    assert out["version"] == DEFAULT_POLICY["version"]


def test_reflection_updates_policy_version_and_thresholds():
    # poor performance should tighten policy
    rows = []
    for _ in range(35):
        rows.append({"action": "long", "fused_score": 0.03, "realized_return": -0.01})
    df = pd.DataFrame(rows)

    out = update_policy_from_history(df, min_samples=30)
    assert out["version"] == DEFAULT_POLICY["version"] + 1
    assert out["reduced_min_confidence"] >= DEFAULT_POLICY["reduced_min_confidence"]
