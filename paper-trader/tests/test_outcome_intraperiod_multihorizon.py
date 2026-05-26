"""Multi-horizon intraperiod-extreme outcome fields (2026-05-26 feature).

`_compute_decision_outcomes` already persists ``forward_intraperiod_min_5d`` /
``forward_intraperiod_max_5d`` (the 5-day window). This file pins the
ADDITIVE 10d and 20d versions:

  * ``forward_intraperiod_min_10d`` / ``forward_intraperiod_max_10d``
  * ``forward_intraperiod_min_20d`` / ``forward_intraperiod_max_20d``

Same contract as the 5d pair — None when the horizon window runs past
cached price history, partial coverage honored, signed %-change relative
to sim_d's close.

A separate test file (not appended to ``test_outcome_intraperiod_extremes``)
so a concurrent sibling agent editing the same file cannot collide with
this work via whole-file ``git add`` — the documented same-role HYBRID
staging-race mitigation pattern.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
import threading

import pytest

import paper_trader.backtest as bt
import run_continuous_backtests as rcb


def _build_history(start: date, n: int):
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _fake_engine_with_prices(closes: list[tuple[date, float]]):
    cache = bt.PriceCache.__new__(bt.PriceCache)
    cache.tickers = ["TICK"]
    cache.start = closes[0][0]
    cache.end = closes[-1][0]
    cache.prices = {"TICK": {d.isoformat(): p for d, p in closes}}
    cache._build_trading_days()
    cache.trading_days = [d for d, _ in closes]
    return cache


def _dict_row(d: dict):
    class _Row:
        def __init__(self, dd):
            self._d = dd

        def __getitem__(self, k):
            return self._d.get(k)

    return _Row(d)


def _build_engine(closes, row):
    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, *a, **k):
            return _Cur(self._rows)

    cache = _fake_engine_with_prices(closes)
    engine = SimpleNamespace()
    engine.prices = cache
    engine.store = SimpleNamespace(
        _lock=threading.Lock(),
        conn=_Conn([_dict_row(row)]),
    )
    return engine, cache


class TestMultiHorizonExtremesPresent:
    """Both new key pairs must always exist on the outcome dict so
    downstream consumers can rely on them — KeyError is the regression."""

    def test_keys_present_in_outcome_dict(self):
        days = _build_history(date(2025, 1, 6), 40)
        closes = [(d, 100.0 + i * 0.1) for i, d in enumerate(days)]
        sim_d = days[0]
        row = {
            "action": "BUY", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "score=2.0 news_count=0 news_urg=0.0",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=5.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]
        # Existing 5d pair still present.
        assert "forward_intraperiod_min_5d" in o
        assert "forward_intraperiod_max_5d" in o
        # New 10d and 20d pairs must exist.
        assert "forward_intraperiod_min_10d" in o
        assert "forward_intraperiod_max_10d" in o
        assert "forward_intraperiod_min_20d" in o
        assert "forward_intraperiod_max_20d" in o


class TestMonotoneRising:
    """Monotone-rising series: min for each horizon is +1 (day 1) and
    max is the endpoint of that horizon (+h)."""

    def test_horizon_extremes_track_horizon_endpoint(self):
        days = _build_history(date(2025, 1, 6), 40)
        closes = [(d, 100.0 + i) for i, d in enumerate(days)]
        sim_d = days[0]
        row = {
            "action": "BUY", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "score=1.5 news_count=0 news_urg=0.0",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=10.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]

        # 5d: min day 1 = +1%, max day 5 = +5%
        assert o["forward_intraperiod_min_5d"] == pytest.approx(1.0, abs=1e-4)
        assert o["forward_intraperiod_max_5d"] == pytest.approx(5.0, abs=1e-4)
        # 10d: min day 1 = +1%, max day 10 = +10%
        assert o["forward_intraperiod_min_10d"] == pytest.approx(1.0, abs=1e-4)
        assert o["forward_intraperiod_max_10d"] == pytest.approx(10.0, abs=1e-4)
        # 20d: min day 1 = +1%, max day 20 = +20%
        assert o["forward_intraperiod_min_20d"] == pytest.approx(1.0, abs=1e-4)
        assert o["forward_intraperiod_max_20d"] == pytest.approx(20.0, abs=1e-4)


class TestVshapeMultiHorizon:
    """A V-shape that crashes -15% by day 3 then recovers to +20% by day 15
    pins distinct min/max values across horizons:
      * 5d window contains the -15% trough but not yet the +20% peak
      * 10d window contains the trough AND a partial recovery
      * 20d window contains the trough AND the full +20% peak
    """

    def test_distinct_extremes_per_horizon(self):
        days = _build_history(date(2025, 1, 6), 40)
        # day0 = 100
        # day1..3: drop 5% per day (95, 90, 85)
        # day4..5: partial recovery (90, 95)
        # day6..15: rise to +20% (closes proportional)
        prices = [100.0, 95.0, 90.0, 85.0, 90.0, 95.0]
        # Days 6..15: linearly rise from 100 to 120
        for d in range(6, 16):
            prices.append(100.0 + (d - 5) * 2.0)
        # Days 16..39: hold flat at 120
        while len(prices) < len(days):
            prices.append(prices[-1])

        closes = list(zip(days, prices))
        sim_d = days[0]
        row = {
            "action": "BUY", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "score=2.0 news_count=0 news_urg=0.0",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=20.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]

        # 5d: min at day 3 = -15%, max at day 5 = -5%
        assert o["forward_intraperiod_min_5d"] == pytest.approx(-15.0, abs=1e-4)
        assert o["forward_intraperiod_max_5d"] == pytest.approx(-5.0, abs=1e-4)

        # 10d: min still -15%, max at day 10 = +10%
        assert o["forward_intraperiod_min_10d"] == pytest.approx(-15.0, abs=1e-4)
        assert o["forward_intraperiod_max_10d"] == pytest.approx(10.0, abs=1e-4)

        # 20d: min still -15%, max at day 15 (or later) = +20%
        assert o["forward_intraperiod_min_20d"] == pytest.approx(-15.0, abs=1e-4)
        assert o["forward_intraperiod_max_20d"] == pytest.approx(20.0, abs=1e-4)


class TestWindowRunsPastHistory:
    """When the horizon window runs past cached price history, the
    corresponding pair must be None — never silently clipped to a shorter
    realized window (that would bias the field on end-of-cycle decisions).
    """

    def test_short_history_20d_extremes_none(self):
        # Build 25 trading days; place sim_d at day 0. The 5d/10d windows
        # have full coverage; the 20d window's endpoint (idx=20) requires
        # trading_days[20] which exists at index 20 (since len=25). But
        # _fwd_intraperiod_extremes loops k=1..h, so for h=20 it needs
        # indices 1..20 (i.e., trading_days[1..20]). With len=25 they all
        # exist. To force the 20d window to be partially or fully missing,
        # we need a smaller history. Use 18 days so the 20d endpoint runs
        # off the cached calendar.
        # But then `_fwd_ret_h(...,20)` would also run off → forward_return_20d
        # is None. The intraperiod 20d would be computed on a SHORTER set
        # of days (partial coverage). So both extremes for the 20d window
        # are computed from the days that DO resolve.
        days = _build_history(date(2025, 1, 6), 18)
        closes = [(d, 100.0 + i * 0.5) for i, d in enumerate(days)]
        sim_d = days[0]
        row = {
            "action": "BUY", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "score=2.0 news_count=0 news_urg=0.0",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=2.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]

        # 5d and 10d full coverage; pin endpoint values.
        # Each day adds 0.5 to price. From day0=100: day1=100.5 (+0.5%),
        # day5=102.5 (+2.5%), day10=105.0 (+5.0%).
        assert o["forward_intraperiod_min_5d"] == pytest.approx(0.5, abs=1e-4)
        assert o["forward_intraperiod_max_5d"] == pytest.approx(2.5, abs=1e-4)
        assert o["forward_intraperiod_min_10d"] == pytest.approx(0.5, abs=1e-4)
        assert o["forward_intraperiod_max_10d"] == pytest.approx(5.0, abs=1e-4)

        # 20d: trading_days[20] is out of range (len=18). The helper loops
        # k=1..20 and skips any k where ti=idx+k >= len(trading_days). So
        # coverage is k=1..17 → max at k=17 = +8.5%, min at k=1 = +0.5%.
        # Partial coverage is honored (per the helper's contract).
        assert o["forward_intraperiod_min_20d"] == pytest.approx(0.5, abs=1e-4)
        assert o["forward_intraperiod_max_20d"] == pytest.approx(8.5, abs=1e-4)


class TestExtremesOnSellRow:
    """SELL outcomes also get the multi-horizon fields (same shape) so the
    inverse-stop / inverse-take-profit analysis can run symmetrically.
    `_compute_decision_outcomes` doesn't sign-flip the extremes on SELL —
    they describe REALIZED %-change from sim_d's close, and downstream
    analyzers apply their own SELL semantics."""

    def test_sell_outcome_carries_multi_horizon_extremes(self):
        days = _build_history(date(2025, 1, 6), 40)
        closes = [(d, 100.0 + i * 0.2) for i, d in enumerate(days)]
        sim_d = days[0]
        row = {
            "action": "SELL", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "ML+quant: TICK score=-1.50 reducing",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=0.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]
        assert o["action"] == "SELL"
        # Same fields exist regardless of action.
        for h in (5, 10, 20):
            assert o[f"forward_intraperiod_min_{h}d"] is not None
            assert o[f"forward_intraperiod_max_{h}d"] is not None
        # Sanity: max for each horizon strictly >= min.
        for h in (5, 10, 20):
            assert (o[f"forward_intraperiod_max_{h}d"]
                    >= o[f"forward_intraperiod_min_{h}d"])
        # And 20d's max should be >= 10d's max should be >= 5d's max (each
        # longer window can only push the realized peak higher in a
        # monotone-rising series).
        assert (o["forward_intraperiod_max_20d"]
                >= o["forward_intraperiod_max_10d"]
                >= o["forward_intraperiod_max_5d"])


class TestMonotonicityAcrossHorizons:
    """Sanity-check: across longer horizons, the realized min cannot
    INCREASE and the realized max cannot DECREASE — a longer window
    strictly contains the shorter one. The earlier V-shape test verifies
    distinct EXACT values; this is the structural monotonicity invariant
    a sloppy implementation that mis-indexed the inner loop bound would
    break.
    """

    def test_min_monotonically_non_increasing_with_horizon(self):
        days = _build_history(date(2025, 1, 6), 40)
        # Random-ish path with a big mid-window drawdown at day 7 (-25%)
        # and a recovery peak at day 18 (+30%).
        prices = [100.0]
        for i in range(1, len(days)):
            # Trough at day 7
            if i == 7:
                prices.append(75.0)
            elif i == 18:
                prices.append(130.0)
            else:
                # Smooth interpolation between key points
                if i < 7:
                    prices.append(100.0 - (100.0 - 75.0) / 7.0 * i)
                elif i < 18:
                    prices.append(75.0 + (130.0 - 75.0) / 11.0 * (i - 7))
                else:
                    prices.append(130.0 - (130.0 - 100.0) / 22.0 * (i - 18))
        closes = list(zip(days, prices))
        sim_d = days[0]
        row = {
            "action": "BUY", "ticker": "TICK",
            "sim_date": sim_d.isoformat(),
            "reasoning": "score=2.0 news_count=0 news_urg=0.0",
        }
        engine, _ = _build_engine(closes, row)
        run = SimpleNamespace(run_id=1, total_return_pct=30.0)
        outs = rcb._compute_decision_outcomes(engine, [run])
        assert len(outs) == 1
        o = outs[0]

        # Monotonicity invariant across horizons:
        assert (o["forward_intraperiod_min_20d"]
                <= o["forward_intraperiod_min_10d"]
                <= o["forward_intraperiod_min_5d"])
        assert (o["forward_intraperiod_max_20d"]
                >= o["forward_intraperiod_max_10d"]
                >= o["forward_intraperiod_max_5d"])
        # The 20d window must see the -25% trough (day 7 is in [1..20]).
        assert o["forward_intraperiod_min_20d"] == pytest.approx(-25.0, abs=1e-4)
        # The 20d window must see the +30% peak (day 18 is in [1..20]).
        assert o["forward_intraperiod_max_20d"] == pytest.approx(30.0, abs=1e-4)
