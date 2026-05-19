"""Tests for paper_trader/analytics/quota_burnrate.py — the rolling-window
quota-exhaustion dominance verdict builder.

Locks the contract for the operator-facing question "right now, is quota
exhaustion my dominant NO_DECISION cause?" Each window is independent and
must apply the MIN_NO_DECISION_SAMPLES floor + QUOTA_DOMINANT_PCT verdict
gate. Bucketing precedence is delegated to ``decision_clock._classify_no_decision``;
these tests pin the rollup, gating, and degrade-never-raise behavior.
"""
from datetime import datetime, timedelta, timezone

from paper_trader.analytics.quota_burnrate import (
    MIN_NO_DECISION_SAMPLES,
    QUOTA_DOMINANT_PCT,
    build_quota_burnrate,
)


NOW = datetime(2026, 5, 19, 3, 0, 0, tzinfo=timezone.utc)


def _row(action, hours_ago, reasoning=""):
    ts = NOW - timedelta(hours=hours_ago)
    return {
        "action_taken": action,
        "reasoning": reasoning,
        "timestamp": ts.isoformat(),
    }


def test_quota_dominant_window():
    rows = [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(8)]
    rows += [_row("BUY_FILLED", 1.5)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    assert w["no_decision"] == 8
    assert w["quota_exhausted"] == 8
    assert w["filled"] == 1
    assert w["verdict"] == "QUOTA_DOMINANT"
    assert w["quota_pct"] == 100.0
    assert "QUOTA_DOMINANT" in w["headline"]


def test_mixed_when_quota_share_below_threshold():
    # 4 quota + 6 host = 40% quota share → MIXED (below 70%)
    rows = [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(4)]
    rows += [_row("NO_DECISION", 1.0, "host saturated, retry later") for _ in range(6)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    assert w["no_decision"] == 10
    assert w["quota_exhausted"] == 4
    assert w["host_saturated"] == 6
    assert w["verdict"] == "MIXED"
    assert w["quota_pct"] == 40.0


def test_low_samples_gates_verdict():
    # Two NO_DECISIONs is below the floor — even 100% quota share must
    # not flip to QUOTA_DOMINANT.
    assert MIN_NO_DECISION_SAMPLES == 3
    rows = [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(2)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    assert w["verdict"] == "LOW_SAMPLES"
    assert w["quota_pct"] is None


def test_exact_threshold_is_quota_dominant():
    # >=70% must flip the verdict (boundary inclusive). 7/10 = 70.0%.
    rows = [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(7)]
    rows += [_row("NO_DECISION", 1.0, "host saturated") for _ in range(3)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    assert w["quota_pct"] == QUOTA_DOMINANT_PCT
    assert w["verdict"] == "QUOTA_DOMINANT"


def test_window_cutoff_strict_excludes_older():
    # 5h-ago row is inside the 6h window; 9h-ago row is outside.
    rows = [
        _row("NO_DECISION", 5.0, "quota exhausted"),
        _row("NO_DECISION", 9.0, "quota exhausted"),
    ]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6, 24))
    w6 = out["windows"][0]
    w24 = out["windows"][1]
    assert w6["no_decision"] == 1
    assert w24["no_decision"] == 2


def test_filled_decisions_count_in_total_only():
    rows = [_row("BUY_FILLED", 1.0), _row("SELL_FILLED", 2.0)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    assert w["total"] == 2
    assert w["filled"] == 2
    assert w["no_decision"] == 0
    assert w["verdict"] == "LOW_SAMPLES"


def test_empty_input_degrades_quietly():
    out = build_quota_burnrate([], now=NOW)
    assert [w["verdict"] for w in out["windows"]] == [
        "LOW_SAMPLES",
        "LOW_SAMPLES",
        "LOW_SAMPLES",
    ]
    assert all(w["total"] == 0 for w in out["windows"])


def test_none_input_does_not_raise():
    # API contract: never raises — caller may pass None on a DB error path.
    out = build_quota_burnrate(None, now=NOW)
    assert out["windows"]


def test_unparseable_and_missing_timestamps_dropped():
    rows = [
        {"action_taken": "NO_DECISION", "reasoning": "quota exhausted",
         "timestamp": "not-an-iso"},
        {"action_taken": "NO_DECISION", "reasoning": "quota exhausted",
         "timestamp": None},
        {"action_taken": "NO_DECISION", "reasoning": "quota exhausted"},  # missing
    ]
    rows += [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(3)]
    out = build_quota_burnrate(rows, now=NOW, windows_hours=(6,))
    w = out["windows"][0]
    # Only the 3 valid rows survive — bad rows must not crash the build.
    assert w["total"] == 3
    assert w["quota_exhausted"] == 3


def test_default_three_windows_present():
    out = build_quota_burnrate([], now=NOW)
    assert [w["hours"] for w in out["windows"]] == [6, 24, 72]


def test_invalid_window_hours_skipped():
    out = build_quota_burnrate([], now=NOW, windows_hours=(6, 0, -3, "x", 24))
    assert [w["hours"] for w in out["windows"]] == [6, 24]


def test_as_of_uses_injected_now():
    out = build_quota_burnrate([], now=NOW)
    assert out["as_of"].startswith("2026-05-19T03:00:00")


def test_naive_now_is_assumed_utc():
    naive = NOW.replace(tzinfo=None)
    out = build_quota_burnrate(
        [_row("NO_DECISION", 1.0, "quota exhausted") for _ in range(3)],
        now=naive,
        windows_hours=(6,),
    )
    assert out["windows"][0]["no_decision"] == 3
