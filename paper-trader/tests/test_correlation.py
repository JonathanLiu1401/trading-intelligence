"""Tests for analytics/correlation.py — concentration-honesty diagnostic.

The Pearson math, the return chain, the weight-Herfindahl and the
correlation-adjusted effective-bets formula are all locked to exact
hand-computed values. The verdict-threshold branching is tested against the
module's own (independently locked) ρ output — a wrong ρ fails the math
tests; a wrong threshold fails the verdict tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.correlation import (
    DOMINANT_WEIGHT,
    HIGH_CORR,
    MIN_RETURNS,
    MOD_CORR,
    _pearson,
    _returns,
    build_correlation,
)


def _closes_from_returns(start, rets):
    """Build a close series whose simple daily returns are exactly `rets`."""
    out = [float(start)]
    for r in rets:
        out.append(out[-1] * (1.0 + r))
    return out


# ───────────────────────────── _returns ─────────────────────────────────

class TestReturns:
    def test_simple_returns_exact(self):
        assert _returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])

    def test_bad_bar_breaks_chain_but_continues(self):
        # 0 and NaN reset the chain; the post-gap segment still yields a
        # return (one bad yfinance bar must not zero the whole series).
        r = _returns([100.0, 0.0, 50.0, 55.0, float("nan"), 200.0, 220.0])
        assert r == pytest.approx([0.1, 0.1])

    def test_non_numeric_is_skipped(self):
        assert _returns([100.0, "x", 120.0, 132.0]) == pytest.approx([0.1])

    def test_too_few_points(self):
        assert _returns([100.0]) == []
        assert _returns([]) == []


# ───────────────────────────── _pearson ─────────────────────────────────

class TestPearson:
    def test_perfect_positive(self):
        x = [0.1, -0.2, 0.3, -0.4, 0.05]
        y = [v * 3.0 + 1.0 for v in x]   # positive affine ⇒ ρ = +1
        assert _pearson(x, y) == 1.0

    def test_perfect_negative(self):
        x = [0.1, -0.2, 0.3, -0.4, 0.05]
        y = [-2.0 * v for v in x]        # negative affine ⇒ ρ = -1
        assert _pearson(x, y) == -1.0

    def test_known_fractional_value(self):
        # x=[1,2,3,4], y=[2,1,4,3]: mx=2.5,my=2.5; sxy=Σ(x-2.5)(y-2.5)
        # = (-1.5)(-0.5)+(-0.5)(-1.5)+(0.5)(1.5)+(1.5)(0.5)=0.75+0.75+
        # 0.75+0.75=3.0; sxx=syy=1.5²·2+0.5²·2=5.0 ⇒ ρ=3/5=0.6.
        assert _pearson([1, 2, 3, 4], [2, 1, 4, 3]) == 0.6

    def test_flat_series_is_none(self):
        assert _pearson([0.0, 0.0, 0.0], [0.1, -0.2, 0.3]) is None

    def test_length_mismatch_or_too_short(self):
        assert _pearson([0.1, 0.2], [0.1]) is None
        assert _pearson([0.1], [0.2]) is None


# ─────────────────────── state / sample-size gate ───────────────────────

class TestStateGate:
    def test_no_data_when_no_positions(self):
        r = build_correlation([], {})
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None

    def test_options_flagged_and_skipped(self):
        poss = [{"ticker": "NVDA", "market_value": 100, "type": "call"},
                {"ticker": "AMD", "market_value": 100, "type": "put"}]
        r = build_correlation(poss, {})
        assert r["state"] == "NO_DATA"           # no stock positions
        assert sorted(r["skipped_options"]) == ["AMD", "NVDA"]
        assert r["n_stock_positions"] == 0

    def test_single_stock_is_insufficient(self):
        rets = [0.01, -0.02, 0.03] * 5
        poss = [{"ticker": "NVDA", "market_value": 100, "type": "stock"}]
        hist = {"NVDA": _closes_from_returns(100, rets)}
        r = build_correlation(poss, hist)
        assert r["state"] == "INSUFFICIENT"
        assert r["verdict"] is None
        assert r["n_correlatable"] == 1
        assert r["effective_independent_bets"] == 1.0  # n_corr==1 arm

    def test_short_series_is_insufficient(self):
        # Fewer than MIN_RETURNS aligned returns ⇒ not correlatable.
        short = _closes_from_returns(100, [0.01] * (MIN_RETURNS - 2))
        poss = [{"ticker": "A", "market_value": 50, "type": "stock"},
                {"ticker": "B", "market_value": 50, "type": "stock"}]
        r = build_correlation(poss, {"A": short, "B": short})
        assert r["state"] == "INSUFFICIENT"
        assert r["n_correlatable"] == 0
        assert sorted(r["short_series_tickers"]) == ["A", "B"]


# ─────────────────────── verdicts & exact metrics ───────────────────────

class TestVerdicts:
    def _two(self, ra, rb, mv_a=50, mv_b=50):
        poss = [{"ticker": "A", "market_value": mv_a, "type": "stock"},
                {"ticker": "B", "market_value": mv_b, "type": "stock"}]
        hist = {"A": _closes_from_returns(100, ra),
                "B": _closes_from_returns(100, rb)}
        return build_correlation(poss, hist)

    def test_concentrated_when_returns_identical(self):
        ra = [0.02, -0.01, 0.03, -0.02, 0.01,
              0.04, -0.03, 0.02, -0.01, 0.05, 0.01, -0.02]
        r = self._two(ra, list(ra))            # ρ exactly +1
        assert r["state"] == "OK"
        assert r["mean_pairwise_corr"] == 1.0
        assert r["verdict"] == "CONCENTRATED"
        # n=2, mean_corr=1 ⇒ n_eff = 2/(1+1·1) = 1.0 (a single real bet).
        assert r["effective_independent_bets"] == 1.0
        assert r["max_pair"] == {"tickers": ["A", "B"], "corr": 1.0}

    def test_diversified_when_returns_anticorrelated(self):
        ra = [0.02, -0.01, 0.03, -0.02, 0.01,
              0.04, -0.03, 0.02, -0.01, 0.05, 0.01, -0.02]
        rb = [-v for v in ra]                   # ρ exactly -1
        r = self._two(ra, rb)
        assert r["mean_pairwise_corr"] == -1.0
        assert r["verdict"] == "DIVERSIFIED"
        # mean_corr=-1, n=2 ⇒ denom = 1+(1)(-1) = 0 ⇒ eff_bets None
        # (cannot divide) — the honest "undefined", not a fabricated number.
        assert r["effective_independent_bets"] is None

    def test_single_name_risk_overrides_correlation(self):
        # Even with perfectly correlated returns, a 70%-weight name reads as
        # single-name risk first (weight ≥ DOMINANT_WEIGHT).
        ra = [0.02, -0.01, 0.03, -0.02, 0.01,
              0.04, -0.03, 0.02, -0.01, 0.05, 0.01, -0.02]
        r = self._two(ra, list(ra), mv_a=70, mv_b=30)
        assert r["top_weight_pct"] == 70.0
        assert r["top_weight_ticker"] == "A"
        assert r["verdict"] == "SINGLE_NAME_RISK"

    def test_moderate_band(self):
        # ρ = 0.6 (the locked _pearson fixture, tiled to clear MIN_RETURNS),
        # weights 50/50 so SINGLE_NAME_RISK does not pre-empt. 0.6 ≥
        # MOD_CORR (0.40) and < HIGH_CORR (0.70) ⇒ MODERATE.
        x = [0.01, 0.02, 0.03, 0.04] * 3        # 12 returns
        y = [0.02, 0.01, 0.04, 0.03] * 3
        r = self._two(x, y)
        # ρ≈0.6 (the exact value is locked in TestPearson; through the
        # close→return float round-trip assert only the band + verdict).
        assert MOD_CORR <= r["mean_pairwise_corr"] < HIGH_CORR
        assert abs(r["mean_pairwise_corr"] - 0.6) < 1e-3
        assert r["verdict"] == "MODERATE"

    def test_hhi_and_effective_positions_exact(self):
        ra = [0.02, -0.01, 0.03, -0.02, 0.01,
              0.04, -0.03, 0.02, -0.01, 0.05, 0.01, -0.02]
        # 60/40 weights ⇒ HHI = 0.36+0.16 = 0.52 ⇒ eff_naive = 1/0.52.
        r = self._two(ra, list(ra), mv_a=60, mv_b=40)
        assert r["weight_hhi"] == 0.52
        assert r["effective_positions_naive"] == round(1.0 / 0.52, 4)
        assert r["weights"] == {"A": 0.6, "B": 0.4}

    def test_effective_bets_formula_with_zero_corr(self):
        # Construct two series with ρ exactly 0 over the window, equal
        # weight: n_eff = 2/(1+1·0) = 2.0 (two genuinely independent bets).
        x = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1,
             0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        y = [0.1, 0.1, -0.1, -0.1, 0.1, 0.1,
             -0.1, -0.1, 0.1, 0.1, -0.1, -0.1]
        r = self._two(x, y)
        assert r["mean_pairwise_corr"] == 0.0   # by construction
        assert r["effective_independent_bets"] == 2.0
        assert r["verdict"] == "DIVERSIFIED"

    def test_alignment_uses_common_tail(self):
        # A has 20 returns, B has 12 → aligned to the common 12; ρ over the
        # last 12 of identical patterns is exactly 1.0.
        long_r = [0.01, -0.02, 0.03, -0.01] * 5     # 20 returns
        short_r = long_r[-12:]                       # 12 returns
        poss = [{"ticker": "A", "market_value": 50, "type": "stock"},
                {"ticker": "B", "market_value": 50, "type": "stock"}]
        hist = {"A": _closes_from_returns(100, long_r),
                "B": _closes_from_returns(100, short_r)}
        r = build_correlation(poss, hist)
        assert r["n_correlatable"] == 2
        assert r["mean_pairwise_corr"] == 1.0


class TestPurity:
    def test_never_raises_on_garbage(self):
        r = build_correlation([{"ticker": None}, {"foo": "bar"}],
                              {"X": ["not", "numbers"]})
        assert isinstance(r, dict)
        assert r["state"] in ("NO_DATA", "INSUFFICIENT")

    def test_dominant_weight_constant_is_a_fraction(self):
        assert 0.0 < DOMINANT_WEIGHT < 1.0
