"""Tests for analytics/churn.py — overtrading & same-name re-entry churn.

Hand-computed arithmetic layered on the single-source-of-truth
``build_round_trips`` (AGENTS.md #10). A wrong re-entry count, a re-entry
counted outside ``REENTRY_WINDOW_DAYS``, a divide-by-zero on a zero-span
book, a verdict emitted before the STABLE sample-size gate, or a sub-day
loss-concentration that re-derives P&L instead of consuming round-trips
all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.churn import (
    REENTRY_WINDOW_DAYS,
    STABLE_MIN_RTS,
    build_churn,
)
from paper_trader.analytics.round_trips import build_round_trips

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(day_offset: float, sec: int = 0) -> str:
    return (_BASE + timedelta(days=day_offset, seconds=sec)).isoformat()


def _pair(tid, ticker, buy_day, sell_day, qty=1, bpx=100.0, spx=100.0):
    """One BUY+SELL that build_round_trips folds into a single round-trip."""
    return [
        {"id": tid, "timestamp": _ts(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": bpx, "value": qty * bpx,
         "strike": None, "expiry": None, "option_type": None},
        {"id": tid + 1, "timestamp": _ts(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": spx, "value": qty * spx,
         "strike": None, "expiry": None, "option_type": None},
    ]


def _distinct(n, hold_days, gap_days=1.0, qty=1, bpx=100.0, spx=100.0):
    """n round-trips on distinct tickers, each held `hold_days`, laid out
    sequentially with `gap_days` between one's close and the next's open
    (no same-name re-entry possible)."""
    trades = []
    tid = 1
    day = 0.0
    for i in range(n):
        trades += _pair(tid, f"T{i}", day, day + hold_days, qty, bpx, spx)
        tid += 2
        day += hold_days + gap_days
    return trades


class TestSampleSizeGate:
    def test_no_trades_is_no_data(self):
        r = build_churn([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 0
        assert r["reentry_rate_pct"] is None
        assert r["round_trips_per_day"] is None

    def test_few_trips_emerging_metrics_no_verdict(self):
        # NVDA closed then re-bought 1d later (the live NVDA→LITE→NVDA shape)
        # + one unrelated trip. Metrics present, verdict withheld.
        trades = (_pair(1, "NVDA", 0, 1, qty=1, bpx=100.0, spx=110.0)
                  + _pair(3, "NVDA", 2, 3, qty=1, bpx=108.0, spx=109.0)
                  + _pair(5, "LITE", 4, 8))
        r = build_churn(trades)
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 3
        assert r["n_reentries"] == 1            # exactly the NVDA re-buy
        assert r["reentry_rate_pct"] == round(1 / 3 * 100, 2)
        ev = r["reentry_events"][0]
        assert ev["ticker"] == "NVDA"
        assert ev["gap_days"] == 1.0            # exit day1 → entry day2
        # prior_pnl_usd consumed from build_round_trips, not recomputed:
        assert ev["prior_pnl_usd"] == 10.0      # bought 100 sold 110, qty 1
        assert "emerging" in r["headline"].lower()

    def test_twenty_trips_is_stable_with_verdict(self):
        r = build_churn(_distinct(STABLE_MIN_RTS, hold_days=3.0))
        assert r["state"] == "STABLE"
        assert r["verdict"] is not None


class TestReentryDetection:
    def test_distinct_names_zero_reentries(self):
        r = build_churn(_distinct(6, hold_days=2.0))
        assert r["n_round_trips"] == 6
        assert r["n_reentries"] == 0
        assert r["reentry_events"] == []

    def test_window_boundary_inclusive_and_exclusive(self):
        # exit at day1; re-open exactly REENTRY_WINDOW_DAYS later → counted.
        on = (_pair(1, "AMD", 0, 1)
              + _pair(3, "AMD", 1 + REENTRY_WINDOW_DAYS,
                      2 + REENTRY_WINDOW_DAYS))
        r_on = build_churn(on)
        assert r_on["n_reentries"] == 1
        assert r_on["reentry_events"][0]["gap_days"] == REENTRY_WINDOW_DAYS

        # one second past the window → NOT a re-entry.
        off = (_pair(1, "AMD", 0, 1)
               + _pair(3, "AMD", 1 + REENTRY_WINDOW_DAYS,
                       2 + REENTRY_WINDOW_DAYS, qty=1))
        # nudge the re-open 1s beyond the window
        off[2]["timestamp"] = _ts(1 + REENTRY_WINDOW_DAYS, sec=1)
        r_off = build_churn(off)
        assert r_off["n_reentries"] == 0

    def test_reentry_events_sorted_fastest_first(self):
        # MU re-bought 2d after close; AMD re-bought 0.5d after close.
        trades = (_pair(1, "MU", 0, 1) + _pair(3, "MU", 3, 4)
                  + _pair(5, "AMD", 0, 1) + _pair(7, "AMD", 1.5, 2.5))
        r = build_churn(trades)
        assert r["n_reentries"] == 2
        gaps = [e["gap_days"] for e in r["reentry_events"]]
        assert gaps == sorted(gaps)
        assert r["reentry_events"][0]["ticker"] == "AMD"   # 0.5d, fastest
        assert r["reentry_events"][0]["gap_days"] == 0.5


class TestVerdicts:
    def test_churning_via_reentry_rate(self):
        # 10 names, each traded twice back-to-back (1d gap) → 20 round-trips,
        # 10 fast same-name re-entries → 50% re-entry rate ⇒ CHURNING.
        trades = []
        tid = 1
        block = 0.0
        for i in range(10):
            tk = f"N{i}"
            trades += _pair(tid, tk, block, block + 1)        # rt 1
            trades += _pair(tid + 2, tk, block + 2, block + 3)  # re-entry
            tid += 4
            block += 10  # keep each name's window clear of the others
        r = build_churn(trades)
        assert r["state"] == "STABLE"
        assert r["n_round_trips"] == 20
        assert r["n_reentries"] == 10
        assert r["reentry_rate_pct"] == 50.0
        assert r["verdict"] == "CHURNING"
        assert "re-buy" in (r["verdict_reason"] or "")

    def test_churning_via_fast_cadence_zero_reentries(self):
        # 20 distinct names, each held 0.5d, packed tight → high
        # round-trips/active-day, sub-day median, zero re-entries.
        trades = []
        tid = 1
        t = 0.0
        for i in range(20):
            trades += _pair(tid, f"F{i}", t, t + 0.5)
            tid += 2
            t += 0.05
        r = build_churn(trades)
        assert r["state"] == "STABLE"
        assert r["n_reentries"] == 0
        assert r["median_hold_days"] == 0.5
        assert r["round_trips_per_day"] is not None
        assert r["round_trips_per_day"] >= 1.0
        assert r["verdict"] == "CHURNING"
        assert "per active day" in (r["verdict_reason"] or "")

    def test_buy_and_hold(self):
        # 20 distinct names, each held 30d, sequential → tiny cadence.
        r = build_churn(_distinct(20, hold_days=30.0, gap_days=1.0))
        assert r["state"] == "STABLE"
        assert r["median_hold_days"] == 30.0
        assert r["round_trips_per_day"] < 0.2
        assert r["n_reentries"] == 0
        assert r["verdict"] == "BUY_AND_HOLD"

    def test_active_turnover_between_the_lines(self):
        # 20 distinct names held 3d each, 1d gap → cadence ~0.25/day:
        # above the quiet line, below the churn line, no re-entries.
        r = build_churn(_distinct(20, hold_days=3.0, gap_days=1.0))
        assert r["state"] == "STABLE"
        assert r["n_reentries"] == 0
        assert r["median_hold_days"] == 3.0
        assert 0.2 <= r["round_trips_per_day"] < 1.0
        assert r["verdict"] == "ACTIVE_TURNOVER"


class TestLossConcentrationAndGuards:
    def test_sub_day_loss_concentration_exact(self):
        # 2 winners, 2 losers: one sub-day loss -$10, one 5d loss -$30.
        trades = (
            _pair(1, "W1", 0, 1, qty=1, bpx=100.0, spx=105.0)        # +5
            + _pair(3, "L1", 2, 2.5, qty=1, bpx=100.0, spx=90.0)     # -10, 0.5d
            + _pair(5, "W2", 3, 4, qty=1, bpx=100.0, spx=102.0)      # +2
            + _pair(7, "L2", 5, 10, qty=1, bpx=100.0, spx=70.0)      # -30, 5d
        )
        r = build_churn(trades)
        assert r["realized_loss_usd"] == -40.0
        assert r["sub_day_loss_usd"] == -10.0
        # share of realised loss booked in <1d trips = 10/40 = 25%
        assert r["churn_loss_concentration_pct"] == 25.0
        # consumed from build_round_trips, not recomputed:
        rts = build_round_trips(trades)
        assert r["realized_loss_usd"] == round(
            sum(x["pnl_usd"] for x in rts if x["pnl_usd"] < 0), 4)

    def test_zero_span_book_no_divide_by_zero(self):
        # single same-instant round-trip → span 0 ⇒ cadence None, no crash.
        r = build_churn(_pair(1, "X", 0, 0))
        assert r["n_round_trips"] == 1
        assert r["span_days"] == 0.0
        assert r["round_trips_per_day"] is None
        assert r["state"] == "EMERGING"

    def test_no_losses_concentration_is_none(self):
        r = build_churn(_distinct(4, hold_days=1.0, spx=110.0))  # all winners
        assert r["realized_loss_usd"] == 0.0
        assert r["churn_loss_concentration_pct"] is None
