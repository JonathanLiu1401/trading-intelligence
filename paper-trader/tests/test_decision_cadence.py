"""Tests for ``paper_trader.analytics.decision_cadence``.

The cadence advisor reports the tier the dynamic-interval runner would pick
right now, the seconds since the last decision row, the ETA to the next
expected cycle, and an OVERDUE verdict when the loop is wedged past the
``OVERDUE_MULT × sleep_s`` boundary. Behaviour locked here so a refactor of
the runner's sleep ladder can never silently re-shape what an operator
watching the dashboard would see.

Coverage:
  * Tier mirroring — MARKET_OPEN / MARKET_CLOSED / QUIET_CLOSED /
    SESSION_OPEN selection lines up with the runner's actual sleep.
  * Verdict ladder — NO_DATA / ON_SCHEDULE / ELAPSED_NORMAL / OVERDUE
    fire on the documented boundaries.
  * Boundary semantics — secs_since == sleep_s stays ON_SCHEDULE;
    secs_since just over sleep_s steps to ELAPSED_NORMAL; secs_since just
    over ``OVERDUE_MULT × sleep_s`` steps to OVERDUE.
  * Clock-skew hardening — a future last_decision_ts clamps to 0s ago
    (the documented ``signals._age_hours`` discipline).
  * Failure contract — a malformed positions list / corrupt calendar / bad
    timestamp degrades, never raises.
  * ``is_cadence_overdue`` single-bool helper fires ONLY on the OVERDUE
    verdict (matches the ``is_intents_stale`` / ``is_failed_runs_hidden``
    pattern this surface is modelled on).
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from paper_trader.analytics import decision_cadence
from paper_trader.analytics.decision_cadence import (
    OVERDUE_MULT,
    build_decision_cadence,
    is_cadence_overdue,
)
from paper_trader.analytics import dynamic_interval as _di


# A path that cannot resolve to any earnings snapshot — _load_calendar_events
# returns [] for any unreadable / missing path, so this neutralises the
# real on-disk calendar and makes _held_has_earnings_today always False.
_EMPTY_CAL = Path("/tmp/__nonexistent_paper_trader_calendar__.json")


def _utc(year, month, day, hour=12, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── Tier mirroring ─────────────────────────────────────────────────────


class TestTierMirroring:
    """The cadence advisor's tier must match what compute_interval would
    pick for the SAME input. Re-computes both sides against the runner's
    own predicates so a future tier change cannot silently drift the
    two surfaces apart."""

    def test_market_open_weekday_midday_with_position(self):
        # Mon 2026-05-18 14:00 ET (18:00 UTC) → MARKET_OPEN tier.
        now = _utc(2026, 5, 18, 18, 0)
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["tier"] == "MARKET_OPEN"
        assert r["sleep_s"] == _di._MARKET_OPEN_S  # 300
        assert r["market_open"] is True

    def test_market_closed_weekend_with_position(self):
        # Sat 2026-05-16 14:00 UTC → market closed, position held.
        now = _utc(2026, 5, 16, 14, 0)
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["tier"] == "MARKET_CLOSED"
        assert r["sleep_s"] == _di._MARKET_CLOSED_S  # 3600
        assert r["market_open"] is False

    def test_quiet_closed_weekend_no_positions(self):
        # Sat with empty book → QUIET_CLOSED.
        now = _utc(2026, 5, 16, 14, 0)
        r = build_decision_cadence(
            [], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["tier"] == "QUIET_CLOSED"
        assert r["sleep_s"] == _di._QUIET_CLOSED_S  # 5400

    def test_session_open_first_thirty_minutes(self):
        # Mon 2026-05-18 09:40 ET (13:40 UTC) — first 30 min after the bell.
        now = _utc(2026, 5, 18, 13, 40)
        r = build_decision_cadence(
            [], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["tier"] == "SESSION_OPEN"
        assert r["sleep_s"] == _di._SESSION_OPEN_S  # 180


# ── Verdict ladder ─────────────────────────────────────────────────────


class TestVerdictLadder:
    """NO_DATA → ON_SCHEDULE → ELAPSED_NORMAL → OVERDUE, on the documented
    boundaries against the current tier's sleep_s."""

    def test_no_last_decision_returns_no_data(self):
        now = _utc(2026, 5, 18, 18, 0)
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "NO_DATA"
        assert r["since_last_decision_s"] is None
        assert r["next_decision_eta_s"] is None
        assert r["next_decision_expected_at"] is None
        assert r["is_overdue"] is False

    def test_recent_decision_is_on_schedule(self):
        # MARKET_OPEN tier (300s). 60s since last decision → ON_SCHEDULE.
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(seconds=60)).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "ON_SCHEDULE"
        assert r["since_last_decision_s"] == 60
        # ETA = (last + 300) - now = 240s.
        assert r["next_decision_eta_s"] == 240
        assert r["is_overdue"] is False

    def test_at_sleep_boundary_stays_on_schedule(self):
        # secs_since == sleep_s exactly → still ON_SCHEDULE (strict > only
        # steps to ELAPSED_NORMAL).
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(seconds=_di._MARKET_OPEN_S)).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "ON_SCHEDULE"
        assert r["since_last_decision_s"] == _di._MARKET_OPEN_S
        assert r["next_decision_eta_s"] == 0

    def test_just_past_sleep_steps_to_elapsed_normal(self):
        # secs_since strictly > sleep_s but ≤ OVERDUE_MULT * sleep_s.
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(seconds=_di._MARKET_OPEN_S + 30)).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "ELAPSED_NORMAL"
        assert r["is_overdue"] is False
        assert r["next_decision_eta_s"] == 0  # cycle imminent

    def test_at_overdue_boundary_stays_elapsed_normal(self):
        # secs_since == OVERDUE_MULT * sleep_s → boundary discipline says
        # ELAPSED_NORMAL (strict > only trips OVERDUE).
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(
            seconds=int(_di._MARKET_OPEN_S * OVERDUE_MULT)
        )).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "ELAPSED_NORMAL"
        assert r["is_overdue"] is False

    def test_well_past_overdue_steps_to_overdue(self):
        # secs_since clearly > OVERDUE_MULT * sleep_s → OVERDUE.
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(
            seconds=int(_di._MARKET_OPEN_S * OVERDUE_MULT) + 60
        )).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "OVERDUE"
        assert r["is_overdue"] is True
        assert "OVERDUE" in r["headline"]
        assert r["next_decision_eta_s"] == 0


# ── Clock-skew hardening ───────────────────────────────────────────────


class TestClockSkewHardening:
    """A wall-clock step-back can write a last_decision_ts strictly
    after ``now``. signals._age_hours / runner.alarm_latch_state already
    clamp to 0; decision_cadence must too — otherwise an operator sees
    "-42s ago" on the dashboard."""

    def test_future_last_decision_ts_clamps_to_zero(self):
        now = _utc(2026, 5, 18, 18, 0)
        # last is 5 minutes in the FUTURE (clock stepped backward after
        # the store write).
        last = (now + timedelta(seconds=300)).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["since_last_decision_s"] == 0
        assert r["verdict"] == "ON_SCHEDULE"
        assert r["is_overdue"] is False


# ── Failure contract ───────────────────────────────────────────────────


class TestFailureContract:
    """Internal failures must NEVER raise — the dashboard would 500.
    Degrade-safe to NO_DATA / sensible defaults."""

    def test_bad_iso_timestamp_treated_as_no_data(self):
        now = _utc(2026, 5, 18, 18, 0)
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], "not-an-iso-timestamp",
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "NO_DATA"
        assert r["since_last_decision_s"] is None

    def test_none_positions_treated_as_empty(self):
        now = _utc(2026, 5, 18, 18, 0)
        r = build_decision_cadence(
            None, None, now=now, calendar_path=_EMPTY_CAL,
        )
        # Mid-session weekday with no positions → MARKET_OPEN (positions
        # only matter for the QUIET_CLOSED tie-break when market is closed).
        assert r["tier"] == "MARKET_OPEN"
        assert r["n_positions"] == 0

    def test_naive_now_treated_as_utc(self):
        # A caller passing a naive datetime must not crash. The runner
        # itself always uses tz-aware UTC; this is defensive.
        now = datetime(2026, 5, 18, 18, 0)  # naive
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["tier"] == "MARKET_OPEN"


# ── is_cadence_overdue helper ──────────────────────────────────────────


class TestIsCadenceOverdue:
    """Single-bool surface; fires ONLY on the OVERDUE verdict — mirrors
    decision_conditionals.is_intents_stale's discipline. False on a
    None/non-dict/missing-verdict input."""

    @pytest.mark.parametrize("v,expected", [
        ("OVERDUE", True),
        ("ON_SCHEDULE", False),
        ("ELAPSED_NORMAL", False),
        ("NO_DATA", False),
        ("ERROR", False),
        (None, False),
    ])
    def test_overdue_only(self, v, expected):
        assert is_cadence_overdue({"verdict": v}) is expected

    def test_non_dict_input_returns_false(self):
        assert is_cadence_overdue(None) is False
        assert is_cadence_overdue("OVERDUE") is False
        assert is_cadence_overdue([1, 2, 3]) is False


# ── Output shape (regression-lock fields a dashboard will render) ──────


class TestOutputShape:
    """A dashboard endpoint will key off specific fields; pin the shape
    so a refactor cannot silently remove one."""

    def test_returns_documented_keys(self):
        now = _utc(2026, 5, 18, 18, 0)
        last = (now - timedelta(seconds=120)).isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        for key in (
            "as_of", "verdict", "headline",
            "tier", "sleep_s", "overdue_mult", "overdue_threshold_s",
            "last_decision_ts", "since_last_decision_s",
            "next_decision_expected_at", "next_decision_eta_s",
            "is_overdue", "n_positions", "market_open",
        ):
            assert key in r, f"missing key {key!r} in output"

    def test_overdue_threshold_matches_sleep_times_mult(self):
        now = _utc(2026, 5, 18, 18, 0)
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=now, calendar_path=_EMPTY_CAL,
        )
        assert r["overdue_threshold_s"] == int(
            r["sleep_s"] * r["overdue_mult"]
        )

    def test_next_expected_iso_matches_last_plus_sleep(self):
        now = _utc(2026, 5, 18, 18, 0)
        last_dt = now - timedelta(seconds=60)
        last = last_dt.isoformat()
        r = build_decision_cadence(
            [{"ticker": "NVDA"}], last,
            now=now, calendar_path=_EMPTY_CAL,
        )
        # Round-trip the emitted ISO; it must equal last + sleep_s seconds.
        parsed = datetime.fromisoformat(
            r["next_decision_expected_at"].replace("Z", "+00:00")
        )
        expected = last_dt + timedelta(seconds=r["sleep_s"])
        assert parsed == expected


# ── Internal error envelope ────────────────────────────────────────────


class TestErrorEnvelope:
    """If something inside the builder genuinely raises (we monkeypatch
    a predicate to make it happen), the outer wrapper returns an ERROR
    dict with the documented shape — never raises."""

    def test_predicate_raise_degrades_to_error_envelope(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("inject")
        # _compute_tier is called inside the try-block before the timestamp
        # arithmetic; force it to bubble all the way up by sabotaging the
        # cleanest seam: the parser of the wall clock.
        monkeypatch.setattr(
            decision_cadence, "_compute_tier", boom, raising=True,
        )

        r = build_decision_cadence(
            [{"ticker": "NVDA"}], None,
            now=_utc(2026, 5, 18, 18, 0),
            calendar_path=_EMPTY_CAL,
        )
        assert r["verdict"] == "ERROR"
        assert "inject" in r["headline"]
        # ERROR envelope still carries the documented keys (a dashboard
        # template must not break on this path).
        for key in (
            "as_of", "verdict", "headline", "tier", "sleep_s",
            "since_last_decision_s", "next_decision_expected_at",
            "next_decision_eta_s", "is_overdue", "n_positions",
            "market_open",
        ):
            assert key in r
