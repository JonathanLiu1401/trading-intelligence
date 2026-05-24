"""Tests for paper_trader.analytics.passive_signal_density.

The analyzer measures the news-signal load during the CURRENT passive run
(HOLDs + NO_DECISIONs since the last FILLED/BLOCKED) and emits a verdict
that surfaces the trader-grade "rich news flow + zero action" pathology.

Behaviour locked here so a future refactor cannot silently:
  * shift the verdict boundary at median == LOW_SIGNAL_MEDIAN (must stay
    INFORMED_PASSIVE) or median == HIGH_SIGNAL_MEDIAN (must stay
    SIGNAL_RICH_PASSIVE — only ``>`` trips DEAFENING_SILENCE);
  * shift the sample floor at n_passive == MIN_PASSIVE_RUN (must commit
    to a verdict, n_passive == MIN_PASSIVE_RUN - 1 must stay INSUFFICIENT);
  * mis-classify a "BUY NVDA → FILLED" / "SELL AMD → BLOCKED" /
    "HOLD CASH → HOLD" / bare "NO_DECISION" action_taken;
  * fail on a malformed / empty / unknown-action ledger;
  * include UNKNOWN rows in the passive count (they should be silently
    skipped — same discipline as exit_only_streak._direction);
  * raise on a non-int / NULL signal_count column value.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.passive_signal_density import (
    HIGH_SIGNAL_MEDIAN,
    LOW_SIGNAL_MEDIAN,
    MIN_PASSIVE_RUN,
    RECENT_TAIL_LEN,
    _classify_decision,
    _coerce_signal_count,
    _median,
    build_passive_signal_density,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _row(action_taken: str | None, signal_count: int | None = 0,
         ts: datetime | None = None) -> dict:
    if ts is None:
        ts = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "action_taken": action_taken,
        "signal_count": signal_count,
        "timestamp": ts.isoformat(),
    }


def _passive_run(n: int, signal_counts: list[int] | int = 0) -> list[dict]:
    """Build `n` PASSIVE rows newest→oldest. `signal_counts` can be either
    a single int (same value for all rows) or a list of length n (newest
    first)."""
    if isinstance(signal_counts, int):
        counts = [signal_counts] * n
    else:
        counts = signal_counts
    assert len(counts) == n
    base = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i, sc in enumerate(counts):
        # Newest first: index 0 is the newest, so it's the freshest ts.
        rows.append(_row("HOLD CASH → HOLD", signal_count=sc,
                         ts=base - timedelta(minutes=i * 5)))
    return rows


# ── _classify_decision ──────────────────────────────────────────────────


class TestClassifyDecision:
    @pytest.mark.parametrize("a", [
        "BUY NVDA → FILLED", "SELL AMD → FILLED", "BUY_CALL TQQQ → FILLED",
        "REBALANCE → FILLED", "FILLED",
    ])
    def test_filled_is_active(self, a):
        assert _classify_decision(a) == "ACTIVE"

    @pytest.mark.parametrize("a", [
        "SELL AMD → BLOCKED", "BUY NVDA → BLOCKED",
        "BLOCKED", "BUY → BLOCKED",
    ])
    def test_blocked_is_active(self, a):
        assert _classify_decision(a) == "ACTIVE"

    @pytest.mark.parametrize("a", [
        "HOLD CASH → HOLD", "HOLD NVDA → HOLD", "HOLD → HOLD", "HOLD",
    ])
    def test_hold_is_passive(self, a):
        assert _classify_decision(a) == "PASSIVE"

    @pytest.mark.parametrize("a", ["NO_DECISION", "NO_DECISION (timeout)"])
    def test_no_decision_is_passive(self, a):
        assert _classify_decision(a) == "PASSIVE"

    @pytest.mark.parametrize("a", [None, "", "   ", 42, ["BUY"], {}])
    def test_non_string_or_empty_is_unknown(self, a):
        assert _classify_decision(a) == "UNKNOWN"

    @pytest.mark.parametrize("a", ["MAYBE", "DECISION", "OPEN", "XYZ → FOO"])
    def test_unknown_status_is_unknown(self, a):
        assert _classify_decision(a) == "UNKNOWN"

    def test_case_insensitive_status_token(self):
        # The status after the → is upper-cased before matching, so a mixed
        # case 'filled' must still classify as ACTIVE.
        assert _classify_decision("BUY NVDA → filled") == "ACTIVE"
        assert _classify_decision("hold cash → hold") == "PASSIVE"

    def test_arrow_whitespace_tolerant(self):
        # The status token after → is .strip()'d so extra whitespace is fine.
        assert _classify_decision("BUY NVDA→  FILLED  ") == "ACTIVE"


# ── _coerce_signal_count ────────────────────────────────────────────────


class TestCoerceSignalCount:
    def test_int_passthrough(self):
        assert _coerce_signal_count(7) == 7

    def test_zero(self):
        assert _coerce_signal_count(0) == 0

    def test_none_is_zero(self):
        # Schema is NOT NULL but a historical / fixture row that comes back
        # None must degrade rather than crash.
        assert _coerce_signal_count(None) == 0

    def test_float_truncates(self):
        assert _coerce_signal_count(3.7) == 3

    def test_string_int(self):
        assert _coerce_signal_count("4") == 4

    def test_string_float(self):
        assert _coerce_signal_count("4.9") == 4

    @pytest.mark.parametrize("v", ["abc", "", "  ", [], {}, object()])
    def test_unparseable_is_zero(self, v):
        assert _coerce_signal_count(v) == 0


# ── _median ─────────────────────────────────────────────────────────────


class TestMedian:
    def test_odd_count(self):
        assert _median([1, 5, 3]) == 3.0

    def test_even_count(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_single(self):
        assert _median([7]) == 7.0

    def test_zero_only(self):
        assert _median([0, 0, 0]) == 0.0

    def test_unsorted_input_sorted_internally(self):
        # _median sorts internally; an out-of-order input must still
        # yield the correct sample median.
        assert _median([10, 1, 5, 7, 3]) == 5.0


# ── build_passive_signal_density — empty / unknown rows ────────────────


class TestEmptyAndUnknown:
    def test_empty_returns_no_data(self):
        rep = build_passive_signal_density([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["n_passive"] == 0
        assert rep["median_signal_count"] is None

    def test_only_unknown_rows_returns_no_data(self):
        rep = build_passive_signal_density([
            _row("MAYBE", 10), _row("XYZ", 5), _row(None, 7),
        ])
        # All rows skipped → behaves as empty.
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None

    def test_unknown_rows_skipped_not_counted(self):
        # 5 PASSIVE rows + 2 UNKNOWN rows must yield n_passive=5 (UNKNOWN
        # silently dropped). Below MIN_PASSIVE_RUN+1 by deliberate design
        # — confirms UNKNOWN is not counted toward the floor.
        rows = _passive_run(MIN_PASSIVE_RUN, signal_counts=5)
        rows.insert(0, _row("MAYBE", 99))
        rows.append(_row("XYZ", 99))
        rep = build_passive_signal_density(rows)
        assert rep["n_passive"] == MIN_PASSIVE_RUN
        assert rep["verdict"] != "INSUFFICIENT"


# ── NO_PASSIVE_RUN — most recent decision is ACTIVE ────────────────────


class TestNoPassiveRun:
    def test_filled_is_active_terminator(self):
        rep = build_passive_signal_density([
            _row("BUY NVDA → FILLED", 5),
            _row("HOLD CASH → HOLD", 10),
        ])
        assert rep["verdict"] == "NO_PASSIVE_RUN"
        assert rep["n_passive"] == 0
        assert "not in a passive run" in rep["headline"].lower()

    def test_blocked_is_active_terminator(self):
        rep = build_passive_signal_density([
            _row("SELL AMD → BLOCKED", 5),
            _row("HOLD CASH → HOLD", 10),
        ])
        assert rep["verdict"] == "NO_PASSIVE_RUN"
        assert rep["n_passive"] == 0


# ── INSUFFICIENT — passive run below MIN_PASSIVE_RUN ───────────────────


class TestInsufficient:
    def test_n_minus_one_is_insufficient(self):
        # MIN_PASSIVE_RUN - 1 passive rows after an ACTIVE row → INSUFFICIENT
        rows = _passive_run(MIN_PASSIVE_RUN - 1, signal_counts=99)
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "INSUFFICIENT"
        assert rep["n_passive"] == MIN_PASSIVE_RUN - 1
        # Headline never claims to know the trader-grade verdict at this
        # sample size.
        assert "INSUFFICIENT" in rep["headline"]

    def test_exactly_at_floor_is_sufficient(self):
        # n_passive == MIN_PASSIVE_RUN must commit to a verdict (≥, not >).
        # Use signal_count=0 → INFORMED_PASSIVE arm; the boundary semantics
        # are tested separately in TestVerdictBoundaries.
        rows = _passive_run(MIN_PASSIVE_RUN, signal_counts=0)
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] != "INSUFFICIENT"
        assert rep["n_passive"] == MIN_PASSIVE_RUN

    def test_single_passive_row_after_fill(self):
        rep = build_passive_signal_density([
            _row("HOLD CASH → HOLD", 99),
            _row("BUY NVDA → FILLED", 0),
        ])
        # Newest is PASSIVE, but only 1 passive row → INSUFFICIENT.
        assert rep["verdict"] == "INSUFFICIENT"
        assert rep["n_passive"] == 1


# ── Verdict boundary discipline ────────────────────────────────────────


class TestVerdictBoundaries:
    def _run_with_median(self, median: int, n: int | None = None) -> dict:
        """Build a passive run whose signal_count median is exactly `median`.

        For odd n the middle element gets `median`; for even n both middles
        do. Surrounding values straddle median so the sort puts it dead
        center.
        """
        n = n or MIN_PASSIVE_RUN
        if n % 2 == 0:
            # even: two middle values both = median, rest = 0 and 2*median
            half = n // 2
            counts = [0] * (half - 1) + [median] * 2 + [median * 2] * (half - 1)
            counts[0] = 0
        else:
            half = n // 2
            counts = [0] * half + [median] + [median * 2] * half
        rows = _passive_run(n, signal_counts=counts)
        rows.append(_row("BUY NVDA → FILLED", 0))
        return build_passive_signal_density(rows)

    def test_median_below_low_threshold_is_informed(self):
        rep = self._run_with_median(LOW_SIGNAL_MEDIAN - 1)
        assert rep["verdict"] == "INFORMED_PASSIVE"

    def test_median_exactly_at_low_threshold_is_informed(self):
        # median == LOW_SIGNAL_MEDIAN must fall in INFORMED (≤).
        rep = self._run_with_median(LOW_SIGNAL_MEDIAN)
        assert rep["verdict"] == "INFORMED_PASSIVE"
        assert rep["median_signal_count"] == float(LOW_SIGNAL_MEDIAN)

    def test_median_just_above_low_is_signal_rich(self):
        rep = self._run_with_median(LOW_SIGNAL_MEDIAN + 1)
        assert rep["verdict"] == "SIGNAL_RICH_PASSIVE"

    def test_median_exactly_at_high_threshold_is_signal_rich(self):
        # median == HIGH_SIGNAL_MEDIAN must fall in SIGNAL_RICH (≤).
        # Only > HIGH_SIGNAL_MEDIAN trips DEAFENING_SILENCE.
        rep = self._run_with_median(HIGH_SIGNAL_MEDIAN)
        assert rep["verdict"] == "SIGNAL_RICH_PASSIVE"
        assert rep["median_signal_count"] == float(HIGH_SIGNAL_MEDIAN)

    def test_median_above_high_threshold_is_deafening(self):
        rep = self._run_with_median(HIGH_SIGNAL_MEDIAN + 1)
        assert rep["verdict"] == "DEAFENING_SILENCE"


# ── Verdict headline + counters ─────────────────────────────────────────


class TestVerdictDetails:
    def test_deafening_silence_counters(self):
        # 5 passive rows all with signal_count 15 → median 15 → DEAFENING.
        rows = _passive_run(5, signal_counts=15)
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "DEAFENING_SILENCE"
        assert rep["n_passive"] == 5
        assert rep["median_signal_count"] == 15.0
        assert rep["max_signal_count"] == 15
        assert rep["min_signal_count"] == 15
        # All 5 cycles are signal-rich (≥ HIGH_SIGNAL_MEDIAN).
        assert rep["n_signal_rich_cycles"] == 5
        assert "DEAFENING_SILENCE" in rep["headline"]

    def test_informed_passive_counters(self):
        # 5 passive rows all with signal_count 0 → median 0 → INFORMED.
        rows = _passive_run(5, signal_counts=0)
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "INFORMED_PASSIVE"
        assert rep["median_signal_count"] == 0.0
        assert rep["n_signal_rich_cycles"] == 0

    def test_fresh_book_no_active_treats_all_as_passive(self):
        # No FILLED/BLOCKED ever — passive run = entire history.
        rows = _passive_run(MIN_PASSIVE_RUN + 3, signal_counts=20)
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "DEAFENING_SILENCE"
        assert rep["n_passive"] == MIN_PASSIVE_RUN + 3
        # No active row → most_recent_active_ts stays None.
        assert rep["most_recent_active_ts"] is None
        assert rep["most_recent_active_action"] is None

    def test_no_decision_counts_as_passive(self):
        # Mixed HOLD + NO_DECISION rows must all count as passive.
        base = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        rows = [
            _row("HOLD CASH → HOLD", 12, base),
            _row("NO_DECISION", 15, base - timedelta(minutes=5)),
            _row("HOLD NVDA → HOLD", 14, base - timedelta(minutes=10)),
            _row("NO_DECISION", 13, base - timedelta(minutes=15)),
            _row("HOLD CASH → HOLD", 11, base - timedelta(minutes=20)),
            _row("BUY NVDA → FILLED", 0, base - timedelta(minutes=25)),
        ]
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "DEAFENING_SILENCE"
        assert rep["n_passive"] == 5

    def test_passive_run_boundaries_ts(self):
        # The newest passive row's ts is "ended"; the oldest is "started".
        base = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        ts_newest = base
        ts_oldest = base - timedelta(minutes=20)
        rows = [
            _row("HOLD CASH → HOLD", 15, ts_newest),
            _row("HOLD CASH → HOLD", 15, base - timedelta(minutes=5)),
            _row("HOLD CASH → HOLD", 15, base - timedelta(minutes=10)),
            _row("HOLD CASH → HOLD", 15, base - timedelta(minutes=15)),
            _row("HOLD CASH → HOLD", 15, ts_oldest),
            _row("BUY NVDA → FILLED", 0, base - timedelta(minutes=25)),
        ]
        rep = build_passive_signal_density(rows)
        assert rep["passive_run_ended_ts"] == ts_newest.isoformat()
        assert rep["passive_run_started_ts"] == ts_oldest.isoformat()
        assert rep["most_recent_active_ts"] == (
            base - timedelta(minutes=25)
        ).isoformat()


# ── Recent tail ────────────────────────────────────────────────────────


class TestRecentTail:
    def test_recent_tail_capped(self):
        # 2 * RECENT_TAIL_LEN passive rows; tail must cap at RECENT_TAIL_LEN.
        rows = _passive_run(RECENT_TAIL_LEN * 2, signal_counts=5)
        rep = build_passive_signal_density(rows)
        assert len(rep["recent_signal_counts"]) == RECENT_TAIL_LEN

    def test_recent_tail_newest_first(self):
        # The tail is newest-first (mirrors the rows orientation).
        # Make signal_counts strictly decreasing newest→oldest so we can
        # eyeball the order.
        counts = list(range(20, 20 - 8, -1))  # 20, 19, ..., 13
        rows = _passive_run(len(counts), signal_counts=counts)
        rep = build_passive_signal_density(rows)
        # Tail tip is newest (20); tail end is older (13).
        assert rep["recent_signal_counts"][0] == 20
        assert rep["recent_signal_counts"][-1] == 13


# ── Degrade safety ─────────────────────────────────────────────────────


class TestDegrade:
    def test_signal_count_null_does_not_crash(self):
        # NULL signal_count rows degrade-safe coerced to 0.
        rows = _passive_run(MIN_PASSIVE_RUN, signal_counts=0)
        # Inject NULL signal_count in one row.
        rows[2]["signal_count"] = None
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        assert rep["verdict"] == "INFORMED_PASSIVE"
        assert rep["n_passive"] == MIN_PASSIVE_RUN

    def test_unparseable_signal_count_coerced(self):
        rows = _passive_run(MIN_PASSIVE_RUN, signal_counts=0)
        rows[1]["signal_count"] = "weird"
        rows.append(_row("BUY NVDA → FILLED", 0))
        rep = build_passive_signal_density(rows)
        # No crash; unparseable coerced to 0 → still INFORMED.
        assert rep["verdict"] == "INFORMED_PASSIVE"

    def test_missing_action_taken_skipped(self):
        # A row with a None action_taken is UNKNOWN → silently dropped.
        rows = _passive_run(MIN_PASSIVE_RUN, signal_counts=0)
        rows.insert(0, _row(None, 99))
        rep = build_passive_signal_density(rows)
        # Despite an UNKNOWN row at the head, the engine treats the next
        # PASSIVE row as the newest in the passive run.
        assert rep["n_passive"] == MIN_PASSIVE_RUN
        assert rep["verdict"] != "INSUFFICIENT"


# ── Threshold constants are pinned ─────────────────────────────────────


class TestPinnedThresholds:
    def test_low_signal_median_constant(self):
        # If this changes, the verdict ladder shifts — pin it so a refactor
        # surfaces the regression.
        assert LOW_SIGNAL_MEDIAN == 3

    def test_high_signal_median_constant(self):
        assert HIGH_SIGNAL_MEDIAN == 10

    def test_min_passive_run_constant(self):
        assert MIN_PASSIVE_RUN == 5

    def test_low_below_high(self):
        # The verdict ladder relies on LOW < HIGH; pin the ordering so a
        # config-only edit can't invert the bands.
        assert LOW_SIGNAL_MEDIAN < HIGH_SIGNAL_MEDIAN
