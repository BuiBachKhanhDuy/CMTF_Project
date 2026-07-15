"""Tests for the calibration metrics and the eval ladder scaffolding."""

import numpy as np

from src.benchmark.calibration import (
    aurc,
    risk_coverage_curve,
    selective_da_at_coverage,
    paired_bootstrap_aurc,
)


def _skillful_confidence(n=400, seed=0):
    """Build (pred, truth, conf) where high confidence ⇒ lower directional error."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    conf = np.abs(rng.normal(size=n)) + 0.1
    # High-confidence samples get the sign right more often.
    p_correct = np.clip(0.5 + 0.3 * (conf / conf.max()), 0.5, 0.95)
    correct = rng.random(n) < p_correct
    pred = np.where(correct, truth, -truth) * conf
    return pred, truth, conf


class TestRiskCoverage:
    def test_curve_shapes(self):
        pred, truth, conf = _skillful_confidence()
        covs, risks = risk_coverage_curve(pred, truth, conf)
        assert covs[-1] == 1.0
        assert len(covs) == len(risks)
        assert np.all((risks >= 0) & (risks <= 1))

    def test_skillful_confidence_lowers_early_risk(self):
        """A skillful confidence should have lower risk at low coverage than full book."""
        pred, truth, conf = _skillful_confidence()
        covs, risks = risk_coverage_curve(pred, truth, conf)
        assert risks[0] < risks[-1]  # most-confident subset beats full book

    def test_aurc_skill_beats_noskill(self):
        pred, truth, conf = _skillful_confidence()
        rng = np.random.default_rng(1)
        noskill = conf[rng.permutation(len(conf))]
        assert aurc(pred, truth, conf) < aurc(pred, truth, noskill)

    def test_selective_da(self):
        pred, truth, conf = _skillful_confidence()
        full = selective_da_at_coverage(pred, truth, conf, 1.0)["DA%"]
        top = selective_da_at_coverage(pred, truth, conf, 0.25)["DA%"]
        assert top > full


class TestPairedBootstrap:
    def test_significant_when_skillful(self):
        pred, truth, conf = _skillful_confidence(n=600)
        rng = np.random.default_rng(2)
        noskill = conf[rng.permutation(len(conf))]
        r = paired_bootstrap_aurc(pred, conf, pred, noskill, truth, n_boot=500)
        assert r["delta_aurc"] < 0  # skillful confidence lowers AURC
        assert "ci_low" in r and "ci_high" in r
