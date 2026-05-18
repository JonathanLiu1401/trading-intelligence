"""Tests for the decision-loss clock in analytics/decision_forensics.py.

The clock folds the *current-regime* NO_DECISION history onto a 24h UTC clock so
a recurring host-load window (the dominant TIMEOUT_EMPTY cause) is actionable.
These assert the *logic*, not "no crash": a wrong regime window, a missing
min-sample guard, or a perturbed legacy-inclusive field will fail here.

The load-bearing property: the clock window and decision_reliability's
current-regime partition are derived from the *same* ``classify_failure`` legacy
contract, so they can never tell different stories on the same data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.decision_forensics import (
    CLOCK_HINT_MARGIN_PP,
    HOUR_MIN_SAMPLE,
    _hour_of_day_clock,
    _regime_boundary,
    build_decision_forensics,
)
from paper_trader.analytics.decision_reliability import (
    build_decision_reliability,
)

NOW = datetime(2026, 5, 18, 23, 59, 0, tzinfo=timezone.utc)
LEGACY = "claude returned no parseable JSON"          # tag == "legacy"
TIMEOUT = "claude returned no response (timeout/empty)"  # current parse failure
OK_JSON = '{"decision":{"action":"HOLD"}}'


def _at(dt: datetime, action="NO_DECISION", reasoning=TIMEOUT,
        market_open=True) -> dict:
    return {
        "timestamp": dt.isoformat(),
        "action_taken": action,
        "reasoning": reasoning,
        "market_open": market_open,
        "signal_count": 0,
    }


def _hour(day: int, hour: int) -> datetime:
    return datetime(2026, 5, day, hour, 30, 0, tzinfo=timezone.utc)


class TestRegimeBoundary:
    def test_none_when_no_legacy_rows(self):
        rows = [_at(_hour(17, 9)), _at(_hour(17, 10), "HOLD X → HOLD", OK_JSON)]
        assert _regime_boundary(rows) is None

    def test_is_newest_legacy_failure(self):
        rows = [
            _at(_hour(10, 8), reasoning=LEGACY),
            _at(_hour(12, 14), reasoning=LEGACY),   # newest legacy
            _at(_hour(11, 9), reasoning=LEGACY),
            _at(_hour(15, 10)),                     # current, later but not legacy
        ]
        assert _regime_boundary(rows) == _hour(12, 14)

    def test_legacy_reasoning_on_non_no_decision_row_is_ignored(self):
        # A FILLED row that happens to carry the legacy string is not a legacy
        # *failure* — decision_reliability gates on _is_no_decision first, so we
        # must too, or the two modules' regime boundary would diverge.
        rows = [
            _at(_hour(12, 14), "BUY X → FILLED", LEGACY),
            _at(_hour(11, 9), reasoning=LEGACY),
        ]
        assert _regime_boundary(rows) == _hour(11, 9)

    def test_unparseable_timestamp_legacy_row_does_not_crash(self):
        rows = [
            {"timestamp": "not-a-date", "action_taken": "NO_DECISION",
             "reasoning": LEGACY, "market_open": True},
            _at(_hour(11, 9), reasoning=LEGACY),
        ]
        assert _regime_boundary(rows) == _hour(11, 9)


class TestClockWindowProperty:
    def test_boundary_none_spans_full_history(self):
        rows = [_at(_hour(17, 9)), _at(_hour(17, 10)),
                _at(_hour(18, 9), "HOLD X → HOLD", OK_JSON)]
        hod, win_n, win_fail, worst, hint = _hour_of_day_clock(rows, None)
        assert win_n == 3            # every parsed row counted
        assert win_fail == 2         # two NO_DECISION rows
        assert sum(b["total"] for b in hod) == 3

    def test_no_bucket_counts_a_row_at_or_before_boundary(self):
        boundary = _hour(12, 14)
        rows = [
            _at(_hour(11, 9), reasoning=LEGACY),         # pre-boundary fail
            _at(boundary, "HOLD X → HOLD", OK_JSON),     # exactly at boundary
            _at(_hour(12, 13)),                          # pre-boundary timeout
            _at(_hour(13, 16)),                          # post-boundary fail
            _at(_hour(14, 16), "HOLD X → HOLD", OK_JSON),  # post-boundary ok
        ]
        hod, win_n, win_fail, _, _ = _hour_of_day_clock(rows, boundary)
        # Only the two strictly-after-boundary rows survive the window: the
        # 13:16 fail and the 14:16 OK — both land on clock-hour 16.
        assert win_n == 2
        assert win_fail == 1
        counted_hours = {b["hour"] for b in hod}
        # Pre/at-boundary rows (clock-hours 9 and 13, and the at-boundary row)
        # must not leak in.
        assert counted_hours == {16}
        h16 = next(b for b in hod if b["hour"] == 16)
        assert h16["total"] == 2 and h16["failures"] == 1

    def test_build_forensics_window_matches_helper(self):
        rows = [
            _at(_hour(10, 8), reasoning=LEGACY),
            _at(_hour(12, 14), reasoning=LEGACY),    # boundary
            _at(_hour(11, 9)),                       # pre-boundary, excluded
            _at(_hour(13, 16)),                      # post-boundary
            _at(_hour(13, 17), "HOLD X → HOLD", OK_JSON),
        ]
        r = build_decision_forensics(rows, now=NOW)
        assert r["regime_boundary"] == _hour(12, 14).isoformat()
        assert r["hour_of_day_window_n"] == 2
        assert r["hour_of_day_window_failures"] == 1


class TestWorstHoursAndHint:
    def _spam(self, day, hour, n, n_fail):
        # Spread by seconds from the top of the hour so n up to 3600 stays a
        # valid in-hour timestamp (and on the same UTC clock hour).
        base = datetime(2026, 5, day, hour, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n_fail):
            rows.append(_at(base + timedelta(seconds=i)))
        for i in range(n - n_fail):
            rows.append(_at(base + timedelta(seconds=1800 + i),
                            "HOLD X → HOLD", OK_JSON))
        return rows

    def test_min_sample_excludes_small_buckets(self):
        # 15:00 has 5 cycles all failing (100%) but below HOUR_MIN_SAMPLE;
        # 10:00 has 8 cycles, 7 failing (87.5%) — only 10:00 may be "worst".
        rows = self._spam(17, 15, HOUR_MIN_SAMPLE - 1, HOUR_MIN_SAMPLE - 1)
        rows += self._spam(17, 10, 8, 7)
        rows += self._spam(18, 3, 10, 1)  # quiet hour, dilutes overall
        hod, win_n, win_fail, worst, hint = _hour_of_day_clock(rows, None)
        worst_hours = {b["hour"] for b in worst}
        assert 10 in worst_hours
        assert 15 not in worst_hours          # suppressed by min sample
        assert "10:00" in hint and "15:00" not in hint
        assert "UTC" in hint

    def test_exactly_min_sample_is_eligible(self):
        rows = self._spam(17, 14, HOUR_MIN_SAMPLE, HOUR_MIN_SAMPLE)  # 6/6 fail
        rows += self._spam(18, 3, 20, 0)  # large clean bucket → low overall
        _, _, _, worst, hint = _hour_of_day_clock(rows, None)
        assert any(b["hour"] == 14 for b in worst)
        assert hint and "14:00" in hint

    def test_no_hint_without_margin(self):
        # Failures spread evenly: worst hour ≈ overall, below the margin.
        rows = []
        for h in (9, 10, 11, 12):
            rows += self._spam(17, h, 10, 5)  # every hour 50%
        _, _, _, worst, hint = _hour_of_day_clock(rows, None)
        assert hint == ""                     # no actionable concentration

    def test_hint_threshold_uses_margin_constant(self):
        # One hour well above overall by > CLOCK_HINT_MARGIN_PP fires the hint.
        rows = self._spam(17, 16, 12, 12)     # 100% fail, n=12
        rows += self._spam(18, 4, 40, 0)      # huge clean bucket
        _, _, win_fail, worst, hint = _hour_of_day_clock(rows, None)
        overall = round(win_fail / (12 + 40) * 100, 1)
        assert worst[0]["fail_pct"] - overall >= CLOCK_HINT_MARGIN_PP
        assert hint.startswith("Parse-failures concentrate at 16:00–17:00 UTC")


class TestAdditiveAndSchema:
    def test_existing_keys_unchanged(self):
        # Same fixture style as test_decision_forensics; the legacy-inclusive
        # fields must be byte-identical to before the clock was added.
        rows = [
            _at(_hour(18, 9), reasoning=TIMEOUT, market_open=True),
            _at(_hour(18, 9), "HOLD X → HOLD", OK_JSON, market_open=True),
            _at(_hour(18, 9), reasoning="parse_failed: x", market_open=False),
            _at(_hour(18, 9), reasoning="parse_failed: y", market_open=False),
        ]
        r = build_decision_forensics(rows, now=NOW)
        assert r["n_decisions"] == 4
        assert r["n_failures"] == 3
        assert r["failure_rate_pct"] == 75.0
        assert r["by_market"]["open"]["total"] == 2
        assert r["by_market"]["open"]["failures"] == 1
        assert r["by_market"]["closed"]["fail_pct"] == 100.0
        assert r["dominant_mode"] in {"TIMEOUT_EMPTY", "NO_JSON"}
        # New keys present alongside, not replacing.
        for k in ("regime_boundary", "hour_of_day", "hour_of_day_window_n",
                  "hour_of_day_window_failures", "hour_of_day_min_sample",
                  "worst_hours", "clock_hint"):
            assert k in r

    def test_empty_list_stable_schema(self):
        r = build_decision_forensics([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["regime_boundary"] is None
        assert r["hour_of_day"] == []
        assert r["worst_hours"] == []
        assert r["clock_hint"] == ""
        assert r["hour_of_day_window_n"] == 0
        assert r["hour_of_day_min_sample"] == HOUR_MIN_SAMPLE

    def test_deterministic_with_injected_now(self):
        rows = [_at(_hour(17, h)) for h in range(8)]
        a = build_decision_forensics(rows, now=NOW)
        b = build_decision_forensics(rows, now=NOW)
        assert a["hour_of_day"] == b["hour_of_day"]
        assert a["clock_hint"] == b["clock_hint"]


class TestSameStoryAsReliability:
    """The discriminator: a regime-drift bug here would hand the operator a
    confidently-wrong "reschedule your cron" recommendation. Both modules MUST
    partition on the identical legacy boundary."""

    def test_regime_boundary_matches_decision_reliability(self):
        rows = [
            _at(_hour(10, 8), reasoning=LEGACY),
            _at(_hour(12, 14), reasoning=LEGACY),
            _at(_hour(11, 9)),
            _at(_hour(13, 16)),
            _at(_hour(14, 17), "HOLD X → HOLD", OK_JSON),
            _at(_hour(15, 18), "BUY X → FILLED", LEGACY),  # not a legacy *fail*
        ]
        fz = build_decision_forensics(rows, now=NOW)
        rel = build_decision_reliability(rows, [], now=NOW)
        assert fz["regime_boundary"] == rel["regime_boundary"]

    def test_both_none_when_no_legacy(self):
        rows = [_at(_hour(17, 9)), _at(_hour(17, 10), "HOLD X → HOLD", OK_JSON)]
        fz = build_decision_forensics(rows, now=NOW)
        rel = build_decision_reliability(rows, [], now=NOW)
        assert fz["regime_boundary"] is None
        assert rel["regime_boundary"] is None
