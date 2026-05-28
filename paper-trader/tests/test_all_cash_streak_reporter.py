"""Reporter wiring for the all-cash-streak verdict.

The ``_all_cash_streak_line`` helper closes a real dashboard→Discord
gap: when the book is contiguously 100% cash for hours/days, the
operator-facing Discord summary historically said nothing about
duration or alpha cost. This suite locks the suppression contract
(silence-when-nothing-actionable) AND the SSOT contract (the helper
ships the builder's headline verbatim — invariant #10) AND the wiring
contract (it actually appears in both ``send_hourly_summary`` and
``send_daily_close``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from paper_trader import reporter


class _FakeStore:
    """Tiny store stub — only ``equity_curve`` is consulted by the
    helper, but the hourly/daily summary call many other reads. We
    return safe defaults (empty lists / sensible dicts) so the
    surrounding render path doesn't blow up while we exercise just
    the streak wiring."""

    def __init__(self, equity_curve_rows=None):
        self._curve = equity_curve_rows or []

    def equity_curve(self, limit=5000):
        return list(self._curve)


# ── Helper-level behaviour ───────────────────────────────────────


def test_helper_returns_empty_on_no_data():
    """Empty curve → NO_DATA → silent. The helper must never render a
    bare "no data" line into the operator's hourly summary."""
    store = _FakeStore([])
    out = reporter._all_cash_streak_line(store)
    assert out == ""


def test_helper_returns_empty_on_insufficient_history():
    """Two-point curve → INSUFFICIENT_HISTORY → silent. The summary
    must not nag the operator on a fresh boot."""
    curve = [
        {"timestamp": "2026-05-26T10:00:00+00:00",
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5000.0},
        {"timestamp": "2026-05-26T11:00:00+00:00",
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5010.0},
    ]
    out = reporter._all_cash_streak_line(_FakeStore(curve))
    assert out == ""


def test_helper_silent_on_brief_holdout():
    """A < 6h flat streak is a normal cash window between exits;
    the helper must not fire (the silence-when-nothing-actionable
    precedent — never become a lying alarm).

    Timestamps must be NOW-relative — the builder's verdict ladder
    keys off ``hours_elapsed_to_now`` (now - start_ts), so a fixed
    historical timestamp set turns "brief" into "extended" as real
    wall-clock advances. NOW-relative anchoring tests the BRIEF
    semantic intent deterministically."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    # 4 points spanning 3h, ending ~30m before now — well under the 6h
    # BRIEF threshold (start_ts is 3.5h ago).
    curve = [
        {"timestamp": (now - timedelta(hours=3.5 - 1.0 * i)).isoformat(),
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5000.0 + i}
        for i in range(4)
    ]
    out = reporter._all_cash_streak_line(_FakeStore(curve))
    assert out == ""


def test_helper_silent_on_not_all_cash():
    """The newest point holds positions → NOT_ALL_CASH → silent. The
    historical-streak summary isn't actionable when the book IS
    currently deployed; the operator already sees positions in
    _portfolio_lines."""
    curve = [
        {"timestamp": "2026-05-26T10:00:00+00:00",
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5000.0},
        {"timestamp": "2026-05-26T11:00:00+00:00",
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5010.0},
        {"timestamp": "2026-05-26T12:00:00+00:00",
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5020.0},
        # Newest point: total_value > cash by $200 (open position)
        {"timestamp": "2026-05-26T13:00:00+00:00",
         "total_value": 1200.0, "cash": 1000.0, "sp500_price": 5030.0},
    ]
    out = reporter._all_cash_streak_line(_FakeStore(curve))
    assert out == ""


def test_helper_fires_on_extended_holdout():
    """A 6h-48h contiguous flat streak fires WITHOUT the ⚠️ prefix —
    the milder action tier. The block header must carry the
    EXTENDED_HOLDOUT verdict label and the builder's own headline
    verbatim (SSOT — invariant #10).

    Timestamps anchored to NOW so the streak is deterministically
    inside the EXTENDED window (~12h elapsed) regardless of when
    the test runs."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    # 10 hourly points starting ~12h ago — squarely inside EXTENDED.
    curve = [
        {"timestamp": (now - timedelta(hours=12.0 - i)).isoformat(),
         "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5000.0 + i}
        for i in range(10)
    ]
    out = reporter._all_cash_streak_line(_FakeStore(curve))
    assert out, "EXTENDED_HOLDOUT must fire the helper"
    assert "**CASH STREAK**" in out
    assert "EXTENDED_HOLDOUT" in out
    # No ⚠️ on the milder tier — the prefix is reserved for the
    # action-required PROLONGED tier (mirrors banked-vs-paper / engine-
    # latch precedent).
    assert not out.startswith("⚠️")


def test_helper_fires_on_prolonged_holdout_with_alarm_prefix():
    """A ≥ 48h contiguous flat streak fires WITH the ⚠️ prefix — the
    action-required tier (the operator needs to know the desk has been
    silently idle for two days+). Without the prefix this would be
    visually identical to the milder EXTENDED tier and the operator
    would lose the urgency signal."""
    # ~52h streak: hourly points across 53 cumulative hours.
    pts = []
    for h in range(0, 54):
        pts.append({
            "timestamp": f"2026-05-25T00:00:00+00:00".replace(
                "00:00:00", f"{h % 24:02d}:00:00"),
            "total_value": 1000.0, "cash": 1000.0, "sp500_price": 5000.0,
        })
    # Build a clean ~53h linear time series for predictability.
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 5, 24, 0, 0, 0, tzinfo=timezone.utc)
    curve = [
        {
            "timestamp": (start + timedelta(hours=h)).isoformat(),
            "total_value": 1000.0, "cash": 1000.0,
            "sp500_price": 5000.0 + h * 0.1,
        }
        for h in range(0, 54)
    ]
    out = reporter._all_cash_streak_line(_FakeStore(curve))
    assert out, "PROLONGED_HOLDOUT must fire the helper"
    assert out.startswith("⚠️"), \
        "PROLONGED_HOLDOUT must carry the action-required ⚠️ prefix"
    assert "**CASH STREAK**" in out
    assert "PROLONGED_HOLDOUT" in out


def test_helper_uses_builders_own_headline_verbatim():
    """Invariant #10 — the headline shipped to Discord is the
    builder's own, never re-derived. This test patches the builder to
    return a sentinel headline and verifies it appears unchanged in
    the helper's output."""
    fake_result = {
        "state": "OK",
        "verdict": "EXTENDED_HOLDOUT",
        "headline": "SENTINEL_HEADLINE_DO_NOT_CHANGE_ME",
        "verdict_detail": "irrelevant",
        "current_streak": {"hours_elapsed_to_now": 8.0, "cash_usd": 1000.0},
    }
    store = _FakeStore([{"timestamp": "x", "total_value": 1, "cash": 1}])
    with patch("paper_trader.analytics.all_cash_streak.build_all_cash_streak",
               return_value=fake_result):
        out = reporter._all_cash_streak_line(store)
    assert "SENTINEL_HEADLINE_DO_NOT_CHANGE_ME" in out, \
        "builder headline must ship verbatim (invariant #10)"


def test_helper_never_raises_on_builder_exception():
    """Failure contract mirrors the rest of reporter: a builder fault
    degrades to ``""`` ("no streak line this report"), **never** an
    exception ("no Discord summary this report"). This protects the
    primary surface."""
    store = _FakeStore([{"timestamp": "x", "total_value": 1, "cash": 1}])
    with patch("paper_trader.analytics.all_cash_streak.build_all_cash_streak",
               side_effect=RuntimeError("simulated builder explosion")):
        out = reporter._all_cash_streak_line(store)
    assert out == "", \
        "helper must degrade silently — never propagate a builder error"


def test_helper_never_raises_on_non_dict_result():
    """A defensive guard: a builder that returned something other than
    a dict (e.g. a future regression in the analytics module) must not
    crash the summary. The helper degrades to ``""``."""
    store = _FakeStore([{"timestamp": "x", "total_value": 1, "cash": 1}])
    with patch("paper_trader.analytics.all_cash_streak.build_all_cash_streak",
               return_value="not a dict"):
        out = reporter._all_cash_streak_line(store)
    assert out == ""


def test_helper_silent_on_empty_headline():
    """If the builder somehow returns a fireable verdict with an empty
    headline, the helper must NOT emit a bare ``**CASH STREAK** ◈
    EXTENDED_HOLDOUT`` with no body — that's the kind of half-formed
    block a trader can't action."""
    fake_result = {
        "state": "OK", "verdict": "EXTENDED_HOLDOUT", "headline": "",
    }
    store = _FakeStore([{"timestamp": "x", "total_value": 1, "cash": 1}])
    with patch("paper_trader.analytics.all_cash_streak.build_all_cash_streak",
               return_value=fake_result):
        out = reporter._all_cash_streak_line(store)
    assert out == ""


# ── Wiring into hourly + daily close ─────────────────────────────


def test_hourly_summary_invokes_all_cash_streak_helper():
    """Regression guard: the helper must actually be called by
    ``send_hourly_summary`` (the wiring, not just the helper's
    existence, is the real value). If a future refactor accidentally
    drops the call, this test fails."""
    called = {"n": 0}

    def fake_streak(_store):
        called["n"] += 1
        return ""

    # Patch _send to a no-op so the hourly summary can complete
    # without hitting openclaw; patch the streak helper to count
    # invocations.
    with patch.object(reporter, "_all_cash_streak_line", side_effect=fake_streak), \
         patch.object(reporter, "_send", return_value=True):
        # send_hourly_summary's other helpers all read from get_store();
        # we let them run against the live store — they're all
        # individually fault-tolerant ("" on exception) and we only
        # care that our helper got invoked at least once.
        try:
            reporter.send_hourly_summary()
        except Exception:
            # If the wider hourly path crashes for an unrelated reason
            # (e.g. yfinance offline in CI), we still want to know
            # whether our helper got reached BEFORE the crash. A
            # zero-call count is a real wiring regression.
            pass
    assert called["n"] >= 1, "send_hourly_summary must call _all_cash_streak_line"


def test_daily_close_invokes_all_cash_streak_helper():
    """Same regression guard for the daily close (mirrors the hourly
    wiring; both surfaces ship the operator the streak verdict)."""
    called = {"n": 0}

    def fake_streak(_store):
        called["n"] += 1
        return ""

    with patch.object(reporter, "_all_cash_streak_line", side_effect=fake_streak), \
         patch.object(reporter, "_send", return_value=True):
        try:
            reporter.send_daily_close()
        except Exception:
            pass
    assert called["n"] >= 1, "send_daily_close must call _all_cash_streak_line"


# ── Suppression discipline locked at the wiring level ────────────


def test_helper_silent_output_does_not_appear_in_summary_body():
    """When the helper returns "", the hourly body must NOT contain
    a stray "**CASH STREAK**" header. This locks the suppression
    contract end-to-end (helper silence → no operator-visible block)."""
    captured = {"body": None}

    def capture_send(message):
        captured["body"] = message
        return True

    # Force the helper to be silent.
    with patch.object(reporter, "_all_cash_streak_line", return_value=""), \
         patch.object(reporter, "_send", side_effect=capture_send):
        try:
            reporter.send_hourly_summary()
        except Exception:
            pass
    body = captured["body"] or ""
    assert "**CASH STREAK**" not in body, \
        "silent helper must not leave a CASH STREAK header in the summary"
