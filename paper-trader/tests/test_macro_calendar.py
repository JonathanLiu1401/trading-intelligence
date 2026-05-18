"""Tests for analytics/macro_calendar.py — the forward FOMC rate-decision
awareness block fed into the live Opus decision prompt.

`event_calendar` gave the desk forward *single-name earnings* awareness.
This is the macro sibling, one dimension over: scheduled **FOMC rate
decisions** — market-wide events that move the whole book (leveraged ETFs
most violently). The discriminating locks here:

  * the **honesty bound** — `now` past `SCHEDULE_VALID_THROUGH` degrades to
    one honest line, never a fabricated event (written first; this is the
    test that stops the static table becoming a silent 2027 landmine);
  * **table↔bound no-drift** — `SCHEDULE_VALID_THROUGH` is exactly the last
    encoded instant, so extending one without the other fails RED;
  * the 8 encoded instants equal the federalreserve.gov-verified 2026 FOMC
    schedule, ET→UTC resolved across the 2026 DST boundary;
  * time-precision tiers (IMMINENT_HOURS < 24h, the material differentiator
    vs event_calendar's date-only granularity);
  * observational/no-directive (invariants #2/#12 — the event_calendar
    contract); never raises; `_build_payload` render position;
    `/api/macro-calendar` Flask parity.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import macro_calendar as mc
from paper_trader.analytics.macro_calendar import build_macro_calendar


def _dt(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# The federalreserve.gov-verified 2026 FOMC schedule. The market-moving
# instant is the 2nd day's 14:00 ET policy statement. ET→UTC: EST=UTC-5
# (Jan & Dec — outside 2026 DST), EDT=UTC-4 (Mar–Oct — 2026 DST is
# Mar 8 → Nov 1). A regression that edits any of these fails loudly.
_EXPECTED_UTC = [
    "2026-01-28T19:00:00+00:00",   # Jan 27-28  EST
    "2026-03-18T18:00:00+00:00",   # Mar 17-18  EDT
    "2026-04-29T18:00:00+00:00",   # Apr 28-29  EDT
    "2026-06-17T18:00:00+00:00",   # Jun 16-17  EDT
    "2026-07-29T18:00:00+00:00",   # Jul 28-29  EDT
    "2026-09-16T18:00:00+00:00",   # Sep 15-16  EDT
    "2026-10-28T18:00:00+00:00",   # Oct 27-28  EDT
    "2026-12-09T19:00:00+00:00",   # Dec 8-9    EST
]


# ───────────────────────── honesty bound (written FIRST) ────────────────────

def test_now_past_schedule_valid_through_degrades_no_event_no_raise():
    """The landmine guard: once the static table is exhausted the block must
    say so honestly — never fabricate an event, never raise."""
    past_end = _dt(_EXPECTED_UTC[-1]) + timedelta(days=1)
    out = build_macro_calendar(now=past_end)
    assert out["source_ok"] is False
    assert out["events"] == []
    block = out["prompt_block"].lower()
    assert "unavailable" in block or "not loaded" in block or "no " in block
    # still carries the autonomy-preserving preamble, no traceback text
    assert "autonomy" in block
    assert "Traceback" not in out["prompt_block"]


def test_schedule_valid_through_equals_last_encoded_instant_no_drift():
    """Extending `_FOMC_2026` but not the bound (or vice-versa) silently
    re-arms the landmine — this fails RED if they ever diverge."""
    last = max(_dt(s) for s in mc._FOMC_2026)
    assert _dt2(mc.SCHEDULE_VALID_THROUGH) == last


def _dt2(v):
    return v if isinstance(v, datetime) else _dt(v)


# ───────────────────────── verified-data lock ───────────────────────────────

def test_encoded_table_is_exactly_the_verified_2026_fomc_schedule():
    got = sorted(_dt(s) for s in mc._FOMC_2026)
    want = sorted(_dt(s) for s in _EXPECTED_UTC)
    assert got == want, f"FOMC table drifted from federalreserve.gov: {got}"


def test_et_to_utc_dst_boundary_is_correct():
    """Jan & Dec statements are 19:00 UTC (EST); Mar–Oct are 18:00 UTC
    (EDT). A flat +5 or +4 offset would break one of the two groups."""
    for s in mc._FOMC_2026:
        d = _dt(s)
        if d.month in (1, 12):
            assert d.hour == 19, f"{s} should be 19:00 UTC (EST)"
        else:
            assert d.hour == 18, f"{s} should be 18:00 UTC (EDT)"


# ───────────────────────── time-precision tiers ─────────────────────────────

def test_imminent_hours_tier_uses_hours_not_days():
    """Opus deciding 5h before an FOMC print ≠ 5d before — the material
    differentiator vs event_calendar's date-only granularity."""
    fomc = _dt(_EXPECTED_UTC[3])  # 2026-06-17 18:00Z
    out = build_macro_calendar(now=fomc - timedelta(hours=5))
    ev = out["events"]
    assert len(ev) == 1
    assert ev[0]["tier"] == "IMMINENT_HOURS"
    assert "5.0h" in out["prompt_block"]
    assert ev[0]["hours_away"] == pytest.approx(5.0, abs=0.05)


def test_imminent_tier_between_one_and_three_days():
    fomc = _dt(_EXPECTED_UTC[3])
    out = build_macro_calendar(now=fomc - timedelta(days=2))
    assert out["events"][0]["tier"] == "IMMINENT"
    assert "2.0d" in out["prompt_block"]


def test_upcoming_tier_within_horizon():
    fomc = _dt(_EXPECTED_UTC[3])
    out = build_macro_calendar(now=fomc - timedelta(days=9))
    assert out["events"][0]["tier"] == "UPCOMING"


def test_24h_boundary_exact():
    """< 24h ⇒ IMMINENT_HOURS; exactly/just over 24h ⇒ IMMINENT."""
    fomc = _dt(_EXPECTED_UTC[3])
    just_inside = build_macro_calendar(now=fomc - timedelta(hours=23.99))
    just_over = build_macro_calendar(now=fomc - timedelta(hours=24.01))
    assert just_inside["events"][0]["tier"] == "IMMINENT_HOURS"
    assert just_over["events"][0]["tier"] == "IMMINENT"


def test_three_day_boundary_exact():
    fomc = _dt(_EXPECTED_UTC[3])
    at_3d = build_macro_calendar(now=fomc - timedelta(days=3))
    over_3d = build_macro_calendar(now=fomc - timedelta(days=3, hours=1))
    assert at_3d["events"][0]["tier"] == "IMMINENT"        # <= 3.0d inclusive
    assert over_3d["events"][0]["tier"] == "UPCOMING"


def test_horizon_boundary_drops_distant_event():
    fomc = _dt(_EXPECTED_UTC[3])
    inside = build_macro_calendar(now=fomc - timedelta(days=14),
                                  horizon_days=14.0)
    outside = build_macro_calendar(now=fomc - timedelta(days=14, hours=1),
                                   horizon_days=14.0)
    assert any(e["event"] == "FOMC" for e in inside["events"])
    assert outside["events"] == []
    assert "no fomc" in outside["prompt_block"].lower() \
        or "no scheduled" in outside["prompt_block"].lower()


def test_past_event_dropped_after_grace():
    """The reaction window stays visible briefly, then the printed decision
    is no longer 'upcoming' and must not leak as a future event."""
    fomc = _dt(_EXPECTED_UTC[3])
    soon_after = build_macro_calendar(now=fomc + timedelta(minutes=30))
    long_after = build_macro_calendar(now=fomc + timedelta(hours=6))
    # within grace the just-printed decision still shows (it is THE event
    # the desk is reacting to); the *next* event (July) is far so the only
    # candidate is June.
    assert any(e["event"] == "FOMC" for e in soon_after["events"])
    # 6h later June is gone; July is >14d away → no events in horizon
    assert all(_dt(e["when_utc"]) > fomc + timedelta(hours=6)
               for e in long_after["events"])


def test_events_sorted_soonest_first():
    """Construct a `now` where two FOMCs fall in a wide horizon."""
    out = build_macro_calendar(now=_dt("2026-06-10T12:00:00+00:00"),
                               horizon_days=120.0)
    whens = [_dt(e["when_utc"]) for e in out["events"]]
    assert whens == sorted(whens)
    assert len(whens) >= 2


# ───────────────────────── observational contract ──────────────────────────

def test_block_is_observational_not_directive():
    """invariants #2/#12 — the event_calendar contract. States facts +
    reaffirms autonomy; issues no directive verb / cap."""
    fomc = _dt(_EXPECTED_UTC[3])
    out = build_macro_calendar(now=fomc - timedelta(days=2))
    b = out["prompt_block"].lower()
    assert "autonomy" in b
    for banned in ("you must", "you should", "do not buy", "reduce ",
                   "cap ", "limit your", "avoid trading"):
        assert banned not in b, f"directive leaked: {banned!r}"


def test_never_raises_on_garbage_now():
    for bad in (None, _dt("2026-06-10T00:00:00+00:00"),
                datetime(2026, 6, 10)):  # naive datetime
        out = build_macro_calendar(now=bad)
        assert isinstance(out, dict)
        assert "prompt_block" in out and "events" in out


def test_no_event_within_horizon_emits_honest_line_not_crash():
    # mid-window between meetings, tight horizon → no events but valid table
    out = build_macro_calendar(now=_dt("2026-05-18T12:00:00+00:00"),
                               horizon_days=7.0)
    assert out["source_ok"] is True
    assert out["events"] == []
    assert "no fomc" in out["prompt_block"].lower() \
        or "no scheduled" in out["prompt_block"].lower()


# ───────────────────────── _build_payload wiring ───────────────────────────

def test_build_payload_renders_macro_between_event_and_buying_power():
    from paper_trader import strategy

    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={},
        risk_mirror_block="RISK-MIRROR-MARKER",
        event_calendar_block="EVENT-CAL-MARKER",
        macro_calendar_block="MACRO-CAL-MARKER fomc",
        buying_power_block="BUYING-POWER-MARKER",
    )
    assert "MACRO-CAL-MARKER fomc" in payload
    # forward blocks adjacent: event → macro → buying_power → WATCHLIST
    assert payload.index("EVENT-CAL-MARKER") < payload.index("MACRO-CAL-MARKER")
    assert payload.index("MACRO-CAL-MARKER") < payload.index("BUYING-POWER-MARKER")
    assert payload.index("BUYING-POWER-MARKER") < payload.index("WATCHLIST PRICES")


def test_build_payload_none_macro_renders_no_stray_text():
    from paper_trader import strategy

    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={}, macro_calendar_block=None)
    assert "MACRO CALENDAR" not in payload
    assert "None" not in payload.split("PORTFOLIO")[1].split("WATCHLIST")[0]


# ───────────────────────── /api/macro-calendar parity ──────────────────────

class TestMacroCalendarEndpoint:
    """Prompt↔endpoint parity (the event_calendar / risk_mirror discipline):
    the route serves the SAME builder so the dashboard can see exactly the
    macro context Opus saw. Drives the real Flask view."""

    def test_endpoint_returns_builder_payload(self):
        from paper_trader import dashboard

        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            resp = client.get("/api/macro-calendar")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data
        assert "prompt_block" in data
        assert "events" in data
        assert "schedule_valid_through" in data
        assert "autonomy" in data["prompt_block"].lower()
