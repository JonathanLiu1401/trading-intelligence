"""Tests for paper_trader.analytics.round_trip_postmortem.

Post-exit drift verdict ladder per closed round-trip. Asserts EXACT
verdicts, exact arithmetic for post_exit_drift_pct, summary tallies and
exit_quality_score, sample-size honesty (NO_DATA/INSUFFICIENT/OK), and
degrade-never-raise on garbage rows.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.round_trip_postmortem import (
    CORRECT_MAX_DRIFT_PCT,
    MIN_HOURS_SINCE_EXIT,
    MISSED_RUNNER_MIN_DRIFT_PCT,
    PREMATURE_MIN_DRIFT_PCT,
    WHIPSAW_MAX_HOLD_HOURS,
    WHIPSAW_MAX_LOSS_PCT,
    build_round_trip_postmortem,
)

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _rt(ticker, *, entry_h_ago, exit_h_ago, cost, proceeds, qty=1.0, type_="stock"):
    """Build a round-trip dict matching build_round_trips' shape."""
    entry_ts = (NOW - timedelta(hours=entry_h_ago)).isoformat()
    exit_ts = (NOW - timedelta(hours=exit_h_ago)).isoformat()
    pnl_usd = round(proceeds - cost, 4)
    pnl_pct = round(pnl_usd / cost * 100, 4) if cost > 1e-9 else None
    hold_days = round((entry_h_ago - exit_h_ago) / 24.0, 4)
    return {
        "ticker": ticker,
        "type": type_,
        "strike": None,
        "expiry": None,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "qty": qty,
        "cost": cost,
        "proceeds": proceeds,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "hold_days": hold_days,
        "n_buys": 1,
        "n_sells": 1,
        "entry_trade_ids": [1],
        "exit_trade_ids": [2],
    }


class TestStateLadder:
    def test_no_data_on_empty_input(self):
        out = build_round_trip_postmortem([], {}, now=NOW)
        assert out["state"] == "NO_DATA"
        assert out["n_input"] == 0
        assert out["n_scored"] == 0
        assert out["trips"] == []
        assert out["exit_quality_score"] is None

    def test_insufficient_when_exit_too_recent(self):
        # Exit 30 min ago — below the MIN_HOURS_SINCE_EXIT gate.
        rts = [_rt("ABC", entry_h_ago=2.0, exit_h_ago=0.5,
                   cost=100.0, proceeds=101.0)]
        out = build_round_trip_postmortem(rts, {"ABC": 102.0}, now=NOW)
        assert out["n_input"] == 1
        assert out["n_scored"] == 0
        # Single trip surfaces, but verdict is INSUFFICIENT.
        assert len(out["trips"]) == 1
        assert out["trips"][0]["verdict"] == "INSUFFICIENT"
        assert out["state"] == "INSUFFICIENT"

    def test_insufficient_when_current_price_missing(self):
        rts = [_rt("ABC", entry_h_ago=20.0, exit_h_ago=10.0,
                   cost=100.0, proceeds=101.0)]
        # Price not provided at all.
        out = build_round_trip_postmortem(rts, {}, now=NOW)
        assert out["trips"][0]["verdict"] == "INSUFFICIENT"
        assert out["trips"][0]["current_price"] is None
        assert out["n_scored"] == 0

    def test_insufficient_when_current_price_none(self):
        rts = [_rt("ABC", entry_h_ago=20.0, exit_h_ago=10.0,
                   cost=100.0, proceeds=101.0)]
        out = build_round_trip_postmortem(rts, {"ABC": None}, now=NOW)
        assert out["trips"][0]["verdict"] == "INSUFFICIENT"
        assert out["n_scored"] == 0

    def test_min_hours_since_exit_constant_is_tight(self):
        # If someone widens MIN_HOURS_SINCE_EXIT without thought, this fails —
        # a defense against a quietly-broader gate that silently dims fresh
        # post-exit signal (the DRAM 1h close case).
        assert 1.0 <= MIN_HOURS_SINCE_EXIT <= 4.0


class TestVerdicts:
    def test_correct_exit_when_price_falls(self):
        # Sold at 100, now 95 → drift -5%, hold long enough.
        rts = [_rt("DEF", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"DEF": 95.0}, now=NOW)
        trip = out["trips"][0]
        assert trip["verdict"] == "CORRECT"
        assert trip["post_exit_drift_pct"] == -5.0
        assert trip["current_price"] == 95.0

    def test_premature_when_price_rises_moderately(self):
        # Sold at 100, now 102 → drift +2% (above PREMATURE_MIN, below
        # MISSED_RUNNER_MIN), hold long enough.
        rts = [_rt("GHI", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"GHI": 102.0}, now=NOW)
        trip = out["trips"][0]
        assert trip["verdict"] == "PREMATURE"
        assert trip["post_exit_drift_pct"] == 2.0

    def test_missed_runner_when_price_rips(self):
        # Sold at 100, now 110 → drift +10% (≥ MISSED_RUNNER_MIN).
        rts = [_rt("JKL", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"JKL": 110.0}, now=NOW)
        trip = out["trips"][0]
        assert trip["verdict"] == "MISSED_RUNNER"
        assert trip["post_exit_drift_pct"] == 10.0

    def test_neutral_when_drift_inside_band(self):
        # Sold at 100, now 100.5 → +0.5% (inside neutral band).
        rts = [_rt("MNO", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"MNO": 100.5}, now=NOW)
        trip = out["trips"][0]
        assert trip["verdict"] == "NEUTRAL"
        assert trip["post_exit_drift_pct"] == 0.5

    def test_whipsaw_short_hold_small_loss_post_recovery(self):
        # Short hold (1h), small loss (-1%), price recovered +1.5% post-exit.
        rts = [_rt("DRAM", entry_h_ago=2.0, exit_h_ago=1.0,
                   cost=253.50, proceeds=250.95)]
        # Exit price avg ≈ 250.95; current ≈ 254.71 → drift +1.5%.
        # But exit is 1h ago < MIN_HOURS_SINCE_EXIT — adjust to ≥ 2h.
        rts = [_rt("DRAM", entry_h_ago=5.0, exit_h_ago=3.0,
                   cost=253.50, proceeds=250.95)]
        out = build_round_trip_postmortem(rts, {"DRAM": 254.71}, now=NOW)
        trip = out["trips"][0]
        # Hold = 2h (within WHIPSAW window), loss ≈ -1.0% (within tolerance),
        # post-exit drift > +PREMATURE_MIN/2, so verdict is WHIPSAW (the
        # short-hold variant of PREMATURE).
        assert trip["verdict"] == "WHIPSAW"
        # Hold reported in hours.
        assert trip["hold_hours"] == 2.0

    def test_whipsaw_disambiguated_from_correct_when_post_exit_falls(self):
        # Same short hold + small loss but price FELL post-exit → the exit
        # was right, not a whipsaw.
        rts = [_rt("DRAM", entry_h_ago=5.0, exit_h_ago=3.0,
                   cost=253.50, proceeds=250.95)]
        out = build_round_trip_postmortem(rts, {"DRAM": 245.0}, now=NOW)
        trip = out["trips"][0]
        assert trip["verdict"] == "CORRECT"

    def test_winner_with_continued_rise_is_premature_not_whipsaw(self):
        # Winner (pnl > 0), longer hold (>WHIPSAW_MAX_HOLD_HOURS), post-exit
        # drift +3% — verdict PREMATURE not WHIPSAW.
        rts = [_rt("WIN", entry_h_ago=72.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=110.0)]
        out = build_round_trip_postmortem(rts, {"WIN": 113.30}, now=NOW)
        trip = out["trips"][0]
        assert trip["pnl_usd"] == 10.0
        # 113.30 vs exit_price 110.0 → drift +3.0%
        assert trip["post_exit_drift_pct"] == 3.0
        assert trip["verdict"] == "PREMATURE"


class TestArithmetic:
    def test_entry_and_exit_prices_use_per_share_average(self):
        # qty=4, cost=400 → entry $100/sh; proceeds=420 → exit $105/sh.
        rts = [_rt("AVG", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=400.0, proceeds=420.0, qty=4.0)]
        out = build_round_trip_postmortem(rts, {"AVG": 110.25}, now=NOW)
        trip = out["trips"][0]
        assert trip["entry_price_avg"] == 100.0
        assert trip["exit_price_avg"] == 105.0
        # drift = (110.25 - 105.0) / 105.0 * 100 = 5.0
        assert trip["post_exit_drift_pct"] == 5.0
        assert trip["verdict"] == "MISSED_RUNNER"

    def test_hours_since_exit_arithmetic(self):
        rts = [_rt("HRS", entry_h_ago=10.0, exit_h_ago=5.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"HRS": 100.0}, now=NOW)
        trip = out["trips"][0]
        assert trip["hours_since_exit"] == 5.0
        assert trip["hold_hours"] == 5.0  # 10 - 5

    def test_zero_proceeds_skipped_not_crash(self):
        # Garbage row: qty 0 → division impossible. Builder must degrade.
        rts = [{
            "ticker": "BAD",
            "type": "stock",
            "strike": None,
            "expiry": None,
            "entry_ts": (NOW - timedelta(hours=48)).isoformat(),
            "exit_ts": (NOW - timedelta(hours=24)).isoformat(),
            "qty": 0.0,
            "cost": 0.0,
            "proceeds": 0.0,
            "pnl_usd": 0.0,
            "pnl_pct": None,
            "hold_days": 1.0,
            "n_buys": 1,
            "n_sells": 1,
            "entry_trade_ids": [1],
            "exit_trade_ids": [2],
        }]
        out = build_round_trip_postmortem(rts, {"BAD": 100.0}, now=NOW)
        # Either trip is dropped or surfaced as INSUFFICIENT, but no raise.
        assert out["state"] in ("INSUFFICIENT", "NO_DATA")


class TestAggregate:
    def test_exit_quality_score_arithmetic(self):
        # Build a 4-trip mix: 2 CORRECT, 1 PREMATURE, 1 MISSED_RUNNER.
        # Score = +1 +1 -1 -2 = -1, divided by n_scored=4 → -0.25.
        rts = [
            _rt("A", entry_h_ago=48.0, exit_h_ago=24.0,
                cost=100.0, proceeds=100.0),  # exit 100 → 95 = CORRECT
            _rt("B", entry_h_ago=48.0, exit_h_ago=20.0,
                cost=100.0, proceeds=100.0),  # exit 100 → 94 = CORRECT
            _rt("C", entry_h_ago=48.0, exit_h_ago=16.0,
                cost=100.0, proceeds=100.0),  # exit 100 → 102 = PREMATURE
            _rt("D", entry_h_ago=48.0, exit_h_ago=12.0,
                cost=100.0, proceeds=100.0),  # exit 100 → 110 = MISSED_RUNNER
        ]
        prices = {"A": 95.0, "B": 94.0, "C": 102.0, "D": 110.0}
        out = build_round_trip_postmortem(rts, prices, now=NOW)
        assert out["n_scored"] == 4
        assert out["verdict_counts"]["CORRECT"] == 2
        assert out["verdict_counts"]["PREMATURE"] == 1
        assert out["verdict_counts"]["MISSED_RUNNER"] == 1
        # +1 + 1 - 1 - 2 = -1 over 4 → -0.25
        assert out["exit_quality_score"] == -0.25

    def test_headline_flags_premature_skew(self):
        rts = [
            _rt("A", entry_h_ago=48.0, exit_h_ago=24.0,
                cost=100.0, proceeds=100.0),
            _rt("B", entry_h_ago=48.0, exit_h_ago=20.0,
                cost=100.0, proceeds=100.0),
            _rt("C", entry_h_ago=48.0, exit_h_ago=16.0,
                cost=100.0, proceeds=100.0),
        ]
        # All 3 in PREMATURE band (>=1%, <5%). B is 4% — keep it under
        # MISSED_RUNNER_MIN_DRIFT_PCT=5 so we exercise the pure-PREMATURE path.
        prices = {"A": 103.0, "B": 104.0, "C": 102.0}
        out = build_round_trip_postmortem(rts, prices, now=NOW)
        assert out["verdict_counts"]["PREMATURE"] == 3
        # Headline should call out the premature pattern.
        h = (out["headline"] or "").lower()
        assert "premature" in h or "too early" in h

    def test_max_n_clips_oldest(self):
        # 5 trips, max_n=2 — only the 2 newest by exit_ts are surfaced.
        rts = [
            _rt(f"T{i}", entry_h_ago=72.0,
                exit_h_ago=24.0 - i * 2.0,  # newer as i grows
                cost=100.0, proceeds=100.0)
            for i in range(5)
        ]
        prices = {f"T{i}": 95.0 for i in range(5)}
        out = build_round_trip_postmortem(rts, prices, now=NOW, max_n=2)
        assert len(out["trips"]) == 2
        # Newest-first ordering.
        ts0 = out["trips"][0]["exit_ts"]
        ts1 = out["trips"][1]["exit_ts"]
        assert ts0 > ts1

    def test_state_ok_when_at_least_one_scored(self):
        rts = [
            _rt("ABC", entry_h_ago=48.0, exit_h_ago=24.0,
                cost=100.0, proceeds=100.0),
        ]
        out = build_round_trip_postmortem(rts, {"ABC": 95.0}, now=NOW)
        assert out["state"] == "OK"


class TestNeverRaises:
    def test_none_inputs_degrade(self):
        out = build_round_trip_postmortem(None, None, now=NOW)
        assert out["state"] == "NO_DATA"

    def test_malformed_round_trip_row(self):
        # Missing required keys → row dropped, builder still returns.
        out = build_round_trip_postmortem(
            [{"ticker": "BAD"}], {"BAD": 100.0}, now=NOW
        )
        assert out["state"] in ("NO_DATA", "INSUFFICIENT")

    def test_negative_current_price_treated_as_missing(self):
        rts = [_rt("XYZ", entry_h_ago=48.0, exit_h_ago=24.0,
                   cost=100.0, proceeds=100.0)]
        out = build_round_trip_postmortem(rts, {"XYZ": -5.0}, now=NOW)
        assert out["trips"][0]["verdict"] == "INSUFFICIENT"


class TestConstants:
    """A regression so constants are not silently widened."""

    def test_thresholds_have_reasonable_values(self):
        assert PREMATURE_MIN_DRIFT_PCT > 0
        assert MISSED_RUNNER_MIN_DRIFT_PCT > PREMATURE_MIN_DRIFT_PCT
        assert CORRECT_MAX_DRIFT_PCT < 0
        assert WHIPSAW_MAX_HOLD_HOURS > 0
        assert WHIPSAW_MAX_LOSS_PCT > 0
