"""Tests for analytics/tail_risk.py — left-tail / downside-shape diagnostic.

Discrete, index-sensitive metrics (the real regression risks: an
off-by-one in the nearest-rank VaR index, the wrong tail set in CVaR, a
streak boundary, the resample "last-write-wins") are pinned with
**exact hand-computed literals**. The continuous stats (annualised vol,
skew) are cross-checked against an *independent* implementation
(`statistics.pstdev`/`fmean`, an algebraically different path from the
module's sum-of-squares) so a formula regression in either fails loudly
without a brittle 3dp literal.

The honesty gate (NO_DATA / INSUFFICIENT / OK) mirrors
build_correlation / build_churn and is locked branch-by-branch.
"""
from __future__ import annotations

import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.tail_risk import MIN_RETURNS, build_tail_risk

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _curve(values: list[float], *, same_day_dupe: bool = False) -> list[dict]:
    """One equity point per UTC day (day i = _BASE + i days)."""
    rows: list[dict] = []
    for i, v in enumerate(values):
        ts = (_BASE + timedelta(days=i)).isoformat()
        rows.append({"timestamp": ts, "total_value": v,
                     "cash": 0.0, "sp500_price": None})
    if same_day_dupe and rows:
        # A second, EARLIER-written row on day 0 that must be overwritten
        # by the real day-0 value (last-write-wins resample contract).
        rows.insert(0, {"timestamp": (_BASE).isoformat(),
                        "total_value": values[0] * 99,
                        "cash": 0.0, "sp500_price": None})
    return rows


# The canonical hand-computed fixture. 25 daily points → 24 returns.
#   returns multiset = {+0.10, -0.10, -0.10, -0.20, +0.25} ∪ {0.0}×19
#   sorted: [-0.20, -0.10, -0.10, 0×19, +0.10, +0.25]
_OK_VALUES = [100.0, 110.0, 99.0, 89.1] + [89.1] * 17 + [71.28, 89.1, 89.1, 89.1]


class TestHonestyGate:
    def test_empty_is_no_data(self):
        r = build_tail_risk([])
        assert r["state"] == "NO_DATA"
        assert r["n_returns"] == 0
        assert r["var_95_pct"] is None
        assert r["return_skew"] is None
        assert "no equity history" in r["headline"].lower()

    def test_single_day_only_is_no_data(self):
        r = build_tail_risk(_curve([1000.0]))
        assert r["state"] == "NO_DATA"
        assert r["n_days"] == 1
        assert r["n_returns"] == 0

    def test_below_min_returns_is_insufficient_but_numeric(self):
        # 6 daily points → 5 returns < MIN_RETURNS.
        r = build_tail_risk(_curve([100, 101, 100, 102, 101, 103]))
        assert r["state"] == "INSUFFICIENT"
        assert r["n_returns"] == 5
        assert r["var_95_pct"] is not None          # numerics still emitted
        assert r["annualized_vol_pct"] is not None
        assert "withheld" in r["headline"].lower()
        assert f"5/{MIN_RETURNS}" in r["headline"]

    def test_exactly_min_returns_is_ok(self):
        # MIN_RETURNS+1 points → exactly MIN_RETURNS returns.
        r = build_tail_risk(_curve([100.0] * (MIN_RETURNS + 1)))
        assert r["n_returns"] == MIN_RETURNS
        assert r["state"] == "OK"


class TestHandComputedFixture:
    """Every index-sensitive number pinned exactly."""

    def setup_method(self):
        self.r = build_tail_risk(_curve(_OK_VALUES))

    def test_shape_and_state(self):
        assert self.r["state"] == "OK"
        assert self.r["n_days"] == 25
        assert self.r["n_returns"] == 24

    def test_var_nearest_rank_indices(self):
        # 95%: idx = ceil(0.05*24)-1 = 1 → sorted[1] = -0.10 → +10.00
        assert self.r["var_95_pct"] == 10.00
        # 99%: idx = ceil(0.01*24)-1 = 0 → sorted[0] = -0.20 → +20.00
        assert self.r["var_99_pct"] == 20.00

    def test_cvar_tail_set(self):
        # Positional ES: k = ceil(0.05*24) = 2 worst returns =
        # {-0.20, -0.10}; mean = -0.15 → +15.00. (Float-robust: the two
        # theoretically-equal -0.10s differ in the last bit, so a
        # value-threshold filter would wrongly drop one — this slice
        # does not.)
        assert self.r["cvar_95_pct"] == 15.00

    def test_worst_and_best_day(self):
        assert self.r["worst_day_pct"] == -20.00
        assert self.r["best_day_pct"] == 25.00

    def test_max_consecutive_down_streak(self):
        # r2=-0.10 and r3=-0.10 are consecutive (run=2); the lone -0.20
        # is isolated (0 before, +0.25 after). A naive "count all
        # negatives" or an off-by-one resets-wrong would not give 2.
        assert self.r["max_consecutive_down_days"] == 2

    def test_downside_deviation_exact(self):
        # Σ min(r,0)² = 0.10²+0.10²+0.20² = 0.06 ; /24 = 0.0025
        # sqrt = 0.05 ; ×√252 ×100 = 79.37253... → 79.37
        assert self.r["downside_deviation_pct"] == 79.37

    def test_ulcer_index_exact(self):
        # dd% from running peak 110: [0,0,10,19,19×17,35.2,19,19,19]
        # Σdd² = 100 + 361·21 + 1239.04 = 8920.04 ; /25 = 356.8016
        # sqrt = 18.8892 → 18.89
        assert self.r["ulcer_index_pct"] == 18.89

    def test_annualized_vol_matches_independent_impl(self):
        _, returns = _series(_OK_VALUES)
        ref = round(statistics.pstdev(returns) * (252 ** 0.5) * 100.0, 2)
        assert self.r["annualized_vol_pct"] == ref
        # Loose sanity band — catches a dropped ×√252 (~7) or ×100 (~1.2)
        # gross-unit regression even if module+ref shared a bug.
        assert 50.0 < self.r["annualized_vol_pct"] < 200.0

    def test_skew_sign_and_independent_impl(self):
        _, returns = _series(_OK_VALUES)
        mu = statistics.fmean(returns)
        sd = statistics.pstdev(returns)
        m3 = statistics.fmean([(x - mu) ** 3 for x in returns])
        ref = round(m3 / sd ** 3, 3)
        assert self.r["return_skew"] == ref
        # The lone +0.25 is a fat right outlier → right-skewed.
        assert self.r["return_skew"] > 0
        assert 0.70 < self.r["return_skew"] < 0.82

    def test_headline_is_dense_and_consistent(self):
        h = self.r["headline"]
        assert "95% 1-day VaR 10.00%" in h
        assert "CVaR 15.00%" in h
        assert "worst day -20.00%" in h


def _series(values: list[float]) -> tuple[list[float], list[float]]:
    """Independent re-derivation of the daily return series for cross-checks."""
    returns = [values[i] / values[i - 1] - 1.0 for i in range(1, len(values))]
    return values, returns


class TestFlatBookIsTheLiveShape:
    """The real 2026-05-14 live data: hundreds of points all at $1000."""

    def setup_method(self):
        self.r = build_tail_risk(_curve([1000.0] * 25))

    def test_state_ok_but_all_risk_is_zero(self):
        assert self.r["state"] == "OK"
        assert self.r["annualized_vol_pct"] == 0.0
        assert self.r["downside_deviation_pct"] == 0.0
        assert self.r["ulcer_index_pct"] == 0.0
        assert self.r["max_consecutive_down_days"] == 0
        assert self.r["worst_day_pct"] == 0.0

    def test_var_is_zero_not_negative_zero(self):
        # -(0.0)*100 must serialise as a clean 0.0, never -0.0.
        assert self.r["var_95_pct"] == 0.0
        assert str(self.r["var_95_pct"]) == "0.0"

    def test_skew_is_none_when_no_dispersion(self):
        # std == 0 → skew undefined, must be None (never a fabricated 0).
        assert self.r["return_skew"] is None


class TestEdgeCases:
    def test_non_positive_prior_value_never_divides_by_zero(self):
        # equity touches 0 then recovers — the 0-prior transition is
        # skipped, no ZeroDivisionError, the rest still computed.
        r = build_tail_risk(_curve([100.0, 0.0, 50.0, 55.0, 60.0]))
        assert "error" not in r
        # 100→0 (prev>0, valid), 0→50 SKIPPED (prev==0), 50→55, 55→60.
        assert r["n_returns"] == 3

    def test_same_day_rows_last_write_wins(self):
        # Day 0 has an earlier bogus row (value*99) then the real value;
        # the resample must keep the LAST, so returns match the clean set.
        clean = build_tail_risk(_curve(_OK_VALUES))
        duped = build_tail_risk(_curve(_OK_VALUES, same_day_dupe=True))
        assert duped["n_days"] == clean["n_days"]
        assert duped["var_95_pct"] == clean["var_95_pct"]
        assert duped["worst_day_pct"] == clean["worst_day_pct"]

    def test_left_skew_is_negative(self):
        # 24 mild gains + one big loss → left-skewed (negative).
        vals = [100.0]
        for _ in range(23):
            vals.append(vals[-1] * 1.001)
        vals.append(vals[-1] * 0.70)  # one large negative outlier
        r = build_tail_risk(_curve(vals))
        assert r["state"] == "OK"
        assert r["return_skew"] is not None
        assert r["return_skew"] < 0

    def test_right_skew_is_positive(self):
        vals = [100.0]
        for _ in range(23):
            vals.append(vals[-1] * 0.999)
        vals.append(vals[-1] * 1.30)  # one large positive outlier
        r = build_tail_risk(_curve(vals))
        assert r["state"] == "OK"
        assert r["return_skew"] > 0
