"""Tests for the directional selection score used by CMTF and the gate."""

from __future__ import annotations

import numpy as np

from src.benchmark.fusion_selection import (
    selection_score,
    da_fraction,
    rank_ic,
)


class TestSelectionScoreComponents:
    def test_da_fraction_bounds(self):
        rng = np.random.default_rng(4)
        target = rng.normal(size=200)
        assert 0.0 <= da_fraction(target, target) <= 1.0
        # Perfect-sign predictor scores ~1.0 on the active subset.
        assert da_fraction(target, target) > 0.95

    def test_rank_ic_perfect_and_degenerate(self):
        x = np.arange(50, dtype=float)
        assert rank_ic(x, x) > 0.99
        assert rank_ic(np.ones(50), x) == 0.0

    def test_selection_score_prefers_directional_alignment(self):
        rng = np.random.default_rng(5)
        target = rng.normal(size=300)
        good = target + rng.normal(scale=0.3, size=300)
        bad = rng.normal(size=300)
        assert selection_score(good, target) > selection_score(bad, target)
