"""Tests for the ``last_real_decision`` analytics builder + reporter line.

The builder is the SSOT for "when did the engine last actually decide
something?" — the IDLE_STORM smoking gun. Discord operators see only the
``NEVER`` and ``STALE`` arms (silence on FRESH / DELAYED); the dashboard
endpoint surfaces all four. This file pins:

  * verdict-ladder correctness across the threshold boundaries
  * action_taken parsing for the 4 documented row shapes
  * never-raises envelope (bad timestamp / None row / non-dict)
  * monotone clamp on a wall-clock step-back
  * constant lockstep with ``analytics.runner_heartbeat`` (so a future
    retune of the cadence multipliers in heartbeat is detected here)
  * reporter ``_last_real_decision_line`` suppression contract
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from paper_trader.analytics.last_real_decision import (
    CLOSED_INTERVAL_S, LAGGING_MULT, OPEN_INTERVAL_S, STALLED_MULT,
    build_last_real_decision,
)


# ── Builder: verdict ladder ──────────────────────────────────────────────


def _row(ts_iso: str, action_taken: str = "BUY NVDA → FILLED") -> dict:
    return {
        "timestamp": ts_iso,
        "action_taken": action_taken,
        "reasoning": "",
        "portfolio_value": 1000.0,
        "cash": 500.0,
    }


def test_none_row_yields_never_state():
    out = build_last_real_decision(None, now=datetime(2026, 5, 29, tzinfo=timezone.utc))
    assert out["state"] == "NEVER"
    assert "never produced" in out["headline"]
    assert out["secs_since"] is None
    assert out["ticker"] is None
    assert out["last_real_ts"] is None
    # Cadence baseline is still reported so consumers can render it.
    assert out["expected_interval_s"] == CLOSED_INTERVAL_S  # market_open default False


def test_non_dict_row_yields_never_state():
    out = build_last_real_decision("not a dict")  # type: ignore[arg-type]
    assert out["state"] == "NEVER"


def test_fresh_when_within_one_lagging_window():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    # 100s ago — well within OPEN_INTERVAL_S * LAGGING_MULT
    ts = (now - timedelta(seconds=100)).isoformat()
    out = build_last_real_decision(_row(ts), now=now, market_open=True)
    assert out["state"] == "FRESH"
    assert out["secs_since"] == pytest.approx(100.0, abs=1.0)


def test_delayed_between_lagging_and_stalled():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    # OPEN_INTERVAL=300, LAGGING=1.25 → 375s; STALLED=2.0 → 600s.
    # 450s = between LAGGING (375) and STALLED (600).
    ts = (now - timedelta(seconds=450)).isoformat()
    out = build_last_real_decision(_row(ts), now=now, market_open=True)
    assert out["state"] == "DELAYED"


def test_stale_past_stalled_window():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    # Past STALLED (3600s open) — say 10000s ago.
    ts = (now - timedelta(seconds=10000)).isoformat()
    out = build_last_real_decision(_row(ts), now=now, market_open=True)
    assert out["state"] == "STALE"
    # The actionable headline calls it out by name.
    assert "STALE" in out["headline"]


def test_closed_market_uses_longer_baseline():
    """When market is closed, expected interval is 3600s — so a 450s gap
    under closed market is FRESH but DELAYED under open."""
    now = datetime(2026, 5, 29, 22, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=450)).isoformat()
    closed = build_last_real_decision(_row(ts), now=now, market_open=False)
    opened = build_last_real_decision(_row(ts), now=now, market_open=True)
    assert closed["state"] == "FRESH"
    assert opened["state"] == "DELAYED"


def test_boundary_just_past_lagging_is_delayed():
    """Exactly at LAGGING_MULT × expected is still FRESH; one second past it
    becomes DELAYED. Pins the ``>`` vs ``>=`` choice in the verdict ladder."""
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    boundary = OPEN_INTERVAL_S * LAGGING_MULT
    # secs equal to boundary → still FRESH (>= would tip to DELAYED).
    ts_at = (now - timedelta(seconds=boundary)).isoformat()
    out_at = build_last_real_decision(_row(ts_at), now=now, market_open=True)
    assert out_at["state"] == "FRESH"
    # secs just past boundary → DELAYED.
    ts_past = (now - timedelta(seconds=boundary + 5)).isoformat()
    out_past = build_last_real_decision(_row(ts_past), now=now, market_open=True)
    assert out_past["state"] == "DELAYED"


# ── Builder: parsing ─────────────────────────────────────────────────────


def test_parse_action_taken_buy_filled():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    out = build_last_real_decision(_row(ts, "BUY NVDA → FILLED"), now=now,
                                    market_open=True)
    assert out["verb"] == "BUY"
    assert out["ticker"] == "NVDA"
    assert out["status"] == "FILLED"


def test_parse_action_taken_hold():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    out = build_last_real_decision(_row(ts, "HOLD MU → HOLD"), now=now,
                                    market_open=True)
    assert out["verb"] == "HOLD"
    assert out["ticker"] == "MU"
    assert out["status"] == "HOLD"


def test_parse_action_taken_blocked():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    out = build_last_real_decision(_row(ts, "BUY LITE → BLOCKED"), now=now,
                                    market_open=True)
    assert out["verb"] == "BUY"
    assert out["ticker"] == "LITE"
    assert out["status"] == "BLOCKED"


def test_parse_action_taken_options():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    out = build_last_real_decision(_row(ts, "BUY_CALL NVDA → FILLED"),
                                    now=now, market_open=True)
    assert out["verb"] == "BUY_CALL"
    assert out["ticker"] == "NVDA"


def test_parse_missing_arrow_no_status():
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    # A free-text row missing the arrow should parse the verb/ticker but
    # status stays None — not raise.
    out = build_last_real_decision(_row(ts, "HOLD NVDA"), now=now,
                                    market_open=True)
    assert out["verb"] == "HOLD"
    assert out["ticker"] == "NVDA"
    assert out["status"] is None


# ── Builder: degrade-safety ──────────────────────────────────────────────


def test_unparseable_timestamp_degrades_to_stale():
    """Operator-safe degrade: a real row whose timestamp can't be parsed
    is treated as STALE (false-negative-protective). The dashboard
    endpoint uses the same arm."""
    out = build_last_real_decision(
        _row("not an ISO timestamp", "BUY NVDA → FILLED"),
        market_open=True,
    )
    assert out["state"] == "STALE"
    assert "unparseable" in out["headline"]
    # Verb/ticker still extracted from action_taken.
    assert out["verb"] == "BUY"
    assert out["ticker"] == "NVDA"


def test_wallclock_step_back_clamps_to_zero():
    """If the row timestamp is in the FUTURE (clock skew / NTP correction),
    secs_since clamps to 0 — never negative — so the verdict ladder stays
    monotone (NEVER < FRESH, never invented negative-age FRESH)."""
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    # Row timestamp 1h after now (impossible without clock skew).
    ts = (now + timedelta(hours=1)).isoformat()
    out = build_last_real_decision(_row(ts), now=now, market_open=True)
    assert out["secs_since"] == 0.0
    assert out["state"] == "FRESH"


def test_naive_now_is_treated_as_utc():
    """A caller that passes a naive datetime should not crash the builder.
    Mirrors the ``last_fill`` precedent."""
    now_naive = datetime(2026, 5, 29, 14, 0, 0)
    ts_aware = (datetime(2026, 5, 29, 13, 59, 0, tzinfo=timezone.utc)).isoformat()
    out = build_last_real_decision(_row(ts_aware), now=now_naive,
                                    market_open=True)
    assert out["state"] == "FRESH"
    assert out["secs_since"] == pytest.approx(60.0, abs=1.0)


def test_missing_action_taken_field():
    """A row missing action_taken entirely (legacy / corrupt) should not
    raise — verb/ticker/status all degrade to None."""
    now = datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=60)).isoformat()
    row = {"timestamp": ts}  # no action_taken
    out = build_last_real_decision(row, now=now, market_open=True)
    assert out["state"] == "FRESH"
    assert out["verb"] is None
    assert out["ticker"] is None
    assert out["status"] is None


# ── Constant lockstep with runner_heartbeat ──────────────────────────────


def test_constants_match_runner_heartbeat():
    """The cadence baseline and multipliers MUST stay in lockstep with
    ``analytics.runner_heartbeat``. A divergence would silently desync the
    Discord verdict from the dashboard's heartbeat verdict — exactly the
    kind of drift this builder exists to prevent for the same row."""
    from paper_trader.analytics import runner_heartbeat as rh
    assert OPEN_INTERVAL_S == rh.OPEN_INTERVAL_S
    assert CLOSED_INTERVAL_S == rh.CLOSED_INTERVAL_S
    assert LAGGING_MULT == rh.LAGGING_MULT
    assert STALLED_MULT == rh.STALLED_MULT


# ── Reporter: _last_real_decision_line suppression contract ─────────────


class _FakeStore:
    def __init__(self, row):
        self._row = row

    def last_real_decision(self):
        return self._row


def test_reporter_line_suppresses_fresh():
    """A FRESH verdict must return ''. The hourly summary's silence-when-
    nothing-actionable contract; a deciding-and-acting desk produces no
    extra noise."""
    from paper_trader import reporter
    now = datetime.now(timezone.utc)
    fresh_ts = (now - timedelta(seconds=60)).isoformat()
    store = _FakeStore({"timestamp": fresh_ts, "action_taken": "HOLD NVDA → HOLD"})
    with patch("paper_trader.reporter.market.is_market_open", return_value=True):
        line = reporter._last_real_decision_line(store)
    assert line == ""


def test_reporter_line_suppresses_delayed():
    """DELAYED is suppressed too — a quiet hour is not actionable yet."""
    from paper_trader import reporter
    now = datetime.now(timezone.utc)
    delayed_ts = (now - timedelta(seconds=450)).isoformat()  # past 1.25*300
    store = _FakeStore(
        {"timestamp": delayed_ts, "action_taken": "HOLD NVDA → HOLD"}
    )
    with patch("paper_trader.reporter.market.is_market_open", return_value=True):
        line = reporter._last_real_decision_line(store)
    assert line == ""


def test_reporter_line_surfaces_stale():
    """STALE fires — this is the actionable wedge signal."""
    from paper_trader import reporter
    now = datetime.now(timezone.utc)
    stale_ts = (now - timedelta(hours=5)).isoformat()  # past 2*300 = 600s
    store = _FakeStore(
        {"timestamp": stale_ts, "action_taken": "HOLD NVDA → HOLD"}
    )
    with patch("paper_trader.reporter.market.is_market_open", return_value=True):
        line = reporter._last_real_decision_line(store)
    assert line != ""
    assert "ENGINE DECIDING?" in line
    assert "STALE" in line
    assert "⚠️" in line


def test_reporter_line_surfaces_never():
    """NEVER fires too — a fresh-boot book that has only NO_DECISION rows."""
    from paper_trader import reporter
    store = _FakeStore(None)
    with patch("paper_trader.reporter.market.is_market_open", return_value=False):
        line = reporter._last_real_decision_line(store)
    assert line != ""
    assert "ENGINE DECIDING?" in line
    assert "NEVER" in line


def test_reporter_line_never_raises_on_store_fault():
    """A store fault inside last_real_decision must degrade to '', never
    propagate — the reporter additive contract: one bad block never kills
    the whole Discord summary."""
    from paper_trader import reporter

    class BadStore:
        def last_real_decision(self):
            raise RuntimeError("simulated store fault")

    line = reporter._last_real_decision_line(BadStore())
    assert line == ""


def test_reporter_line_never_raises_on_builder_fault():
    """A builder fault must also degrade silently."""
    from paper_trader import reporter
    store = _FakeStore({"timestamp": "ok"})
    with patch(
        "paper_trader.analytics.last_real_decision.build_last_real_decision",
        side_effect=RuntimeError("builder broke"),
    ):
        line = reporter._last_real_decision_line(store)
    assert line == ""
