"""Tests for Phase 2 (market proposal) and Phase 3 (news proposal) logic.

Verifies that proposals produce logically correct signals from indicators.
"""

import numpy as np
import pytest

from src.multiagent.agents.market_agent import _build_market_proposal, _get_feature
from src.multiagent.agents.news_agent import _build_news_proposal

_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_lower", "bb_mid", "bb_upper", "atr_14",
    "vol_ratio", "log_ret",
    "vnindex_ret", "vnindex_vol_ratio",
    "sentiment_mean", "sentiment_max_abs",
    "sentiment_positive_ratio", "sentiment_negative_ratio",
    "sentiment_score_count", "sentiment_missing_flag",
]


def _make_tabular(**overrides) -> np.ndarray:
    """Build a market_tabular array with sensible defaults and overrides."""
    defaults = {
        "close": 100.0, "rsi_14": 50.0, "macd_hist": 0.0,
        "atr_14": 1.0, "bb_lower": 95.0, "bb_mid": 100.0, "bb_upper": 105.0,
    }
    defaults.update(overrides)
    tab = np.zeros(len(_COLS), dtype=np.float32)
    for name, val in defaults.items():
        if name in _COLS:
            tab[_COLS.index(name)] = val
    return tab


_LOW_VOL = {"vol_20d": 15.0, "max_drawdown_pct": 3.0, "trend_pct": 5.0}
_HIGH_VOL = {"vol_20d": 50.0, "max_drawdown_pct": 10.0, "trend_pct": -5.0}


# ── Phase 2: Market Proposal ──────────────────────────────────────────


class TestMarketProposal:
    def test_oversold_all_long(self):
        """RSI=30, MACD_hist positive, close below BB mid → all long signals → long."""
        tab = _make_tabular(rsi_14=30.0, macd_hist=0.5, close=95.0, bb_mid=100.0)
        p = _build_market_proposal(tab, _COLS, _LOW_VOL)
        assert p["direction"] == "long"
        assert p["score"] > 0
        assert p["confidence"] >= 0.7  # all agree + low vol (15% → 0.75)

    def test_overbought_all_short(self):
        """RSI=75, MACD_hist negative, close above BB mid → all short signals → short."""
        tab = _make_tabular(rsi_14=75.0, macd_hist=-0.5, close=107.0, bb_mid=100.0)
        p = _build_market_proposal(tab, _COLS, _LOW_VOL)
        assert p["direction"] == "short"
        assert p["score"] < 0
        assert p["confidence"] >= 0.7

    def test_mixed_signals_low_confidence(self):
        """RSI oversold (long) but MACD negative (short) → low confidence."""
        tab = _make_tabular(rsi_14=30.0, macd_hist=-0.5, close=100.0, bb_mid=100.0)
        p = _build_market_proposal(tab, _COLS, _LOW_VOL)
        assert p["confidence"] < 0.8  # signals disagree

    def test_neutral_rsi_flat(self):
        """RSI=50, MACD=0, close=mid → neutral → flat."""
        tab = _make_tabular(rsi_14=50.0, macd_hist=0.0, close=100.0, bb_mid=100.0)
        p = _build_market_proposal(tab, _COLS, _LOW_VOL)
        assert p["direction"] == "flat"
        assert abs(p["score"]) < 0.001

    def test_high_vol_reduces_confidence(self):
        """Same bullish signals but high vol → lower confidence."""
        tab = _make_tabular(rsi_14=30.0, macd_hist=0.5, close=95.0, bb_mid=100.0)
        p_low = _build_market_proposal(tab, _COLS, _LOW_VOL)
        p_high = _build_market_proposal(tab, _COLS, _HIGH_VOL)
        assert p_high["confidence"] < p_low["confidence"]

    def test_score_bounded(self):
        """Score should be in [-0.05, 0.05] regardless of extreme inputs."""
        tab = _make_tabular(rsi_14=0.0, macd_hist=100.0, close=50.0, bb_mid=200.0)
        p = _build_market_proposal(tab, _COLS, _LOW_VOL)
        assert -0.05 <= p["score"] <= 0.05

    def test_no_backward_momentum_dependency(self):
        """Score should NOT depend on trend_pct — that's backward looking."""
        tab = _make_tabular(rsi_14=40.0, macd_hist=0.2, close=98.0, bb_mid=100.0)
        p_up = _build_market_proposal(tab, _COLS, {"vol_20d": 15.0, "max_drawdown_pct": 3.0, "trend_pct": 20.0})
        p_down = _build_market_proposal(tab, _COLS, {"vol_20d": 15.0, "max_drawdown_pct": 3.0, "trend_pct": -20.0})
        assert p_up["score"] == p_down["score"]  # trend_pct must not affect score


# ── Phase 3: News Proposal ────────────────────────────────────────────


class TestNewsProposal:
    def test_high_coverage_fresh_articles_high_trust(self):
        """Good coverage + low staleness → high trust weight."""
        metrics = {"sentiment_mean": 0.5, "coverage": 15, "staleness_frac": 0.1}
        p = _build_news_proposal(metrics)
        assert p["confidence"] > 0.5  # high trust
        assert p["direction"] == "long"  # positive sentiment

    def test_no_coverage_zero_trust(self):
        """Zero coverage → zero trust weight regardless of sentiment."""
        metrics = {"sentiment_mean": 0.9, "coverage": 0, "staleness_frac": 0.0}
        p = _build_news_proposal(metrics)
        assert p["confidence"] == 0.0  # no news → no trust

    def test_all_stale_zero_trust(self):
        """All articles stale → zero trust weight."""
        metrics = {"sentiment_mean": 0.5, "coverage": 10, "staleness_frac": 1.0}
        p = _build_news_proposal(metrics)
        assert p["confidence"] == 0.0

    def test_score_is_raw_sentiment(self):
        """Score should be raw sentiment_mean, NOT scaled by quality."""
        metrics = {"sentiment_mean": 0.4, "coverage": 5, "staleness_frac": 0.3}
        p = _build_news_proposal(metrics)
        assert p["score"] == pytest.approx(0.4, abs=1e-5)

    def test_trust_weight_in_quality_dict(self):
        """quality dict must include trust_weight matching confidence."""
        metrics = {"sentiment_mean": 0.3, "coverage": 8, "staleness_frac": 0.2}
        p = _build_news_proposal(metrics)
        assert "trust_weight" in p["quality"]
        assert p["quality"]["trust_weight"] == p["confidence"]

    def test_negative_sentiment_short(self):
        """Negative sentiment → short direction."""
        metrics = {"sentiment_mean": -0.3, "coverage": 10, "staleness_frac": 0.1}
        p = _build_news_proposal(metrics)
        assert p["direction"] == "short"
