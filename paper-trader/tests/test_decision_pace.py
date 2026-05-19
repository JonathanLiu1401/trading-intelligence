"""Tests for analytics/decision_pace.py — rolling inter-decision latency.

Exact-value asserts on the percentile builder (linear-interpolation, NIST
type-7) and the open/closed split. A drift in the percentile formula, an
off-by-one on the trailing-cycle classification, a divide-by-zero on a
1-row trader, or a verdict emitted before STABLE all fail an assertion.

Tests read live module constants (cadence, factors, gates) so a retune of
``decision_pace.py`` cannot false-fail the suite — the digital-intern
"tests read live constants" discipline.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.decision_pace import (
    CADENCE_STALL_FACTOR,
    CADENCE_TOLERANCE_FACTOR,
    CLOSED_INTERVAL_S,
    OPEN_INTERVAL_S,
    STABLE_MIN_GAPS,
    WINDOWS,
    _percentile,
    build_decision_pace,
)

_NOW = datetime(2026, 5, 19, 21, 0, 0, tzinfo=timezone.utc)


def _row(offset_s: float, market_open: int = 1) -> dict:
    """Decision row at ``_NOW - offset_s`` (offset_s>=0 → in the past)."""
    return {
        "timestamp": (_NOW - timedelta(seconds=offset_s)).isoformat(),
        "market_open": market_open,
        "action_taken": "HOLD",
        "reasoning": "",
    }


def _evenly_spaced(n: int, gap_s: float, market_open: int = 1,
                   start_offset_s: float = 0.0) -> list[dict]:
    """n decisions spaced ``gap_s`` apart, newest-first."""
    return [_row(start_offset_s + i * gap_s, market_open) for i in range(n)]


class TestEmptyAndDegenerate:
    def test_empty_decisions_is_no_data(self):
        r = build_decision_pace([], now=_NOW)
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["n_decisions"] == 0
        assert r["n_gaps"] == 0
        assert r["last_decision_ts"] is None
        assert r["age_since_last_decision_s"] is None
        # Every window emits zero-sample distributions, never raises.
        for h in WINDOWS:
            w = r["windows"][f"{h}h"]
            assert w["all"]["n"] == 0
            assert w["all"]["p95"] is None

    def test_single_decision_is_no_data_one_row_zero_gaps(self):
        r = build_decision_pace([_row(0)], now=_NOW)
        assert r["n_decisions"] == 1
        assert r["n_gaps"] == 0
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        # The last_decision_ts still reports — operators need it even with 1 row.
        assert r["last_decision_ts"] is not None
        assert r["age_since_last_decision_s"] == 0.0

    def test_unparseable_timestamp_degrades_silently(self):
        rows = [
            _row(0), _row(60),
            {"timestamp": "not-a-date", "market_open": 1, "action_taken": "HOLD"},
            {"timestamp": None, "market_open": 1, "action_taken": "HOLD"},
        ]
        r = build_decision_pace(rows, now=_NOW)
        # Two parseable rows → exactly one gap, builder did not raise.
        assert r["n_decisions"] == 2
        assert r["n_gaps"] == 1


class TestPercentileBuilder:
    def test_percentile_empty_is_none(self):
        assert _percentile([], 50.0) is None

    def test_percentile_single_value(self):
        assert _percentile([42.0], 50.0) == 42.0
        assert _percentile([42.0], 95.0) == 42.0

    def test_percentile_linear_interpolation_matches_numpy_type_7(self):
        # values 1..10: p50 = 5.5, p95 = 9.55 (NIST/numpy default).
        vs = [float(i) for i in range(1, 11)]
        assert _percentile(vs, 50.0) == 5.5
        assert _percentile(vs, 95.0) == 9.55
        assert _percentile(vs, 0.0) == 1.0
        assert _percentile(vs, 100.0) == 10.0


class TestDistributionAndSplit:
    def test_evenly_spaced_open_cycles_exact_percentiles(self):
        # 11 decisions, 1800s apart, all market_open → 10 gaps of 1800s each.
        # All inside the 6h window (10*1800s = 18000s = 5h).
        rows = _evenly_spaced(11, OPEN_INTERVAL_S, market_open=1)
        r = build_decision_pace(rows, now=_NOW)
        assert r["n_decisions"] == 11
        assert r["n_gaps"] == 10
        assert r["state"] == "STABLE"
        w6 = r["windows"]["6h"]
        assert w6["open"]["n"] == 10
        assert w6["open"]["p50"] == OPEN_INTERVAL_S
        assert w6["open"]["p95"] == OPEN_INTERVAL_S
        assert w6["open"]["max"] == OPEN_INTERVAL_S
        # No closed-cycle gaps in this fixture.
        assert w6["closed"]["n"] == 0
        # Verdict: every gap equals cadence → HEALTHY.
        assert r["verdict"] == "HEALTHY"

    def test_open_and_closed_split_independently(self):
        # 6 open-market decisions (1800s cadence) BEFORE the close transition,
        # then 6 market-closed decisions (3600s cadence) AFTER. Both sit fully
        # inside the 24h window. Gap classification key: the *trailing*
        # cycle's market_open decides which bucket the gap lands in.
        gaps_open = 5      # 6 rows in open regime → 5 open gaps
        gaps_closed = 5    # 6 rows in closed regime → 5 closed gaps
        # … + 1 transition gap (closed-trailing) between the two regimes.
        open_rows = [
            {"timestamp": (_NOW - timedelta(seconds=(20 * 3600
                                                     - i * OPEN_INTERVAL_S))
                           ).isoformat(),
             "market_open": 1, "action_taken": "HOLD", "reasoning": ""}
            for i in range(6)
        ]
        closed_start = 12 * 3600  # 12h before _NOW
        closed_rows = [
            {"timestamp": (_NOW - timedelta(seconds=(closed_start
                                                     - i * CLOSED_INTERVAL_S))
                           ).isoformat(),
             "market_open": 0, "action_taken": "HOLD", "reasoning": ""}
            for i in range(6)
        ]
        r = build_decision_pace(open_rows + closed_rows, now=_NOW)
        w24 = r["windows"]["24h"]
        assert w24["open"]["n"] == gaps_open
        assert w24["open"]["p50"] == OPEN_INTERVAL_S
        assert w24["closed"]["n"] == gaps_closed + 1  # incl. transition gap
        # The 5 pure-closed gaps are exact cadence; the transition is larger.
        assert w24["closed"]["p50"] == CLOSED_INTERVAL_S
        assert w24["closed"]["max"] > CLOSED_INTERVAL_S

    def test_zero_span_dup_timestamps_no_crash(self):
        # Two rows at the same instant → one zero-second gap, no divide-by-zero.
        rows = [_row(0), _row(0)]
        r = build_decision_pace(rows, now=_NOW)
        assert r["n_gaps"] == 1
        w = r["windows"]["1h"]
        assert w["all"]["n"] == 1
        assert w["all"]["p50"] == 0.0
        assert w["all"]["max"] == 0.0


class TestVerdictLadder:
    def test_emerging_when_below_stable_gate(self):
        # Just below STABLE_MIN_GAPS in EVERY window — verdict withheld.
        rows = _evenly_spaced(STABLE_MIN_GAPS, OPEN_INTERVAL_S)
        # STABLE_MIN_GAPS rows → STABLE_MIN_GAPS-1 gaps, all in the 6h window.
        r = build_decision_pace(rows, now=_NOW)
        assert r["n_gaps"] == STABLE_MIN_GAPS - 1
        # No window has reached STABLE_MIN_GAPS samples → EMERGING.
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None
        assert "emerging" in r["headline"].lower()

    def test_lagging_when_fresh_window_p95_exceeds_tolerance(self):
        # Construct a fixture that meets STABLE on the 6h/24h windows AND
        # puts a single LAGGING-grade gap inside the freshest (1h) window.
        # Freshest row at offset 0; one gap of `lag_gap` to the next row;
        # then a long tail of evenly-spaced rows on cadence (these supply
        # the STABLE-gate samples in the wider windows).
        lag_gap = OPEN_INTERVAL_S * (
            (CADENCE_TOLERANCE_FACTOR + CADENCE_STALL_FACTOR) / 2.0)
        tail = _evenly_spaced(STABLE_MIN_GAPS + 5, OPEN_INTERVAL_S,
                              market_open=1, start_offset_s=lag_gap)
        rows = [_row(0.0, market_open=1)] + tail
        r = build_decision_pace(rows, now=_NOW)
        assert r["state"] == "STABLE"  # 6h/24h windows clear the gate
        w1 = r["windows"]["1h"]
        # Only the one trailing gap (offset 0, attributed to market_open=1)
        # falls inside the 1h window — but verdict only needs n>0 there.
        assert w1["open"]["n"] >= 1
        assert w1["open"]["max"] == lag_gap
        assert w1["open"]["p95"] >= CADENCE_TOLERANCE_FACTOR * OPEN_INTERVAL_S
        assert w1["open"]["p95"] < CADENCE_STALL_FACTOR * OPEN_INTERVAL_S
        assert r["verdict"] == "LAGGING"

    def test_stalled_when_fresh_window_p95_exceeds_stall(self):
        # Same shape, gap large enough to trip the STALL threshold.
        stall_gap = OPEN_INTERVAL_S * (CADENCE_STALL_FACTOR + 1.0)
        tail = _evenly_spaced(STABLE_MIN_GAPS + 5, OPEN_INTERVAL_S,
                              market_open=1, start_offset_s=stall_gap)
        rows = [_row(0.0, market_open=1)] + tail
        r = build_decision_pace(rows, now=_NOW)
        assert r["state"] == "STABLE"
        w1 = r["windows"]["1h"]
        assert w1["open"]["max"] == stall_gap
        assert w1["open"]["p95"] >= CADENCE_STALL_FACTOR * OPEN_INTERVAL_S
        assert r["verdict"] == "STALLED"
        assert r["verdict_reason"] and "p95" in r["verdict_reason"]
