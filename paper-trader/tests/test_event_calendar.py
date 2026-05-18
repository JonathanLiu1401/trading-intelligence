"""Tests for analytics/event_calendar.py — the upcoming-earnings awareness
block fed into the live Opus decision prompt.

The live engine was completely earnings-blind: ``/api/earnings-risk`` existed
but was dashboard-only, and ``strategy.py`` had zero scheduled-catalyst
awareness, so Opus could BUY the day before a binary print with no idea it was
coming. This block closes that gap following the ``risk_mirror`` precedent:
observational only (invariants #2/#12), single source of truth with
``/api/earnings-risk`` tiering, prompt-block + ``/api/*`` parity, and — the
load-bearing constraint — it reads digital-intern's
``data/earnings_calendar.json`` **directly from disk** (never a :8080 network
hop on the live cycle) and is ``_safe``-wrapped so a missing / stale / corrupt
file degrades to an honest line, never an exception that sinks a trading cycle.

Discriminating regressions locked here:
* trusting the file's stale ``days_away`` instead of recomputing it from
  ``earnings_date`` vs ``now`` (THE bug-prone line — api_earnings recomputes;
  this must too),
* the HELD_IMMINENT ``<= 3`` day boundary drifting,
* a past event leaking into the prompt,
* a missing/corrupt file raising instead of degrading,
* the freshest snapshot not winning when two candidates exist,
* a directive verb leaking into an "observational only" block,
* the ``_build_payload`` wiring (renders in the advisory stack; ``None``
  renders no stray text).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import event_calendar
from paper_trader.analytics.event_calendar import build_event_calendar

_NOW = datetime(2026, 5, 18, 2, 0, 0, tzinfo=timezone.utc)


def _earnings_date(days_from_now: float) -> str:
    return (_NOW + timedelta(days=days_from_now)).isoformat()


def _write_calendar(path: Path, events: list[dict], as_of: datetime | None = None) -> None:
    snap = {
        "as_of": (as_of or _NOW).isoformat(),
        "horizon_days": 14,
        "n_events": len(events),
        "events": events,
    }
    path.write_text(json.dumps(snap))


def _pos(ticker: str) -> dict:
    return {"ticker": ticker, "type": "stock", "qty": 1.0,
            "avg_cost": 100.0, "current_price": 100.0}


# ───────────────────────── days_away recompute ─────────────────────────

def test_days_away_recomputed_from_earnings_date_not_stale_file_field(tmp_path):
    """THE discriminating test: the file's ``days_away`` is deliberately a
    garbage stale value; the builder must recompute it from ``earnings_date``
    vs the injected ``now`` (exactly as digital-intern's api_earnings does).
    A regression that trusts the file field would tier NVDA wrong."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": _earnings_date(1.91),
         "days_away": 999.0},  # stale/garbage — must be ignored
    ])
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=cal, now=_NOW)
    ev = out["events"][0]
    assert ev["ticker"] == "NVDA"
    assert ev["days_away"] == pytest.approx(1.91, abs=0.01)
    assert ev["tier"] == "HELD_IMMINENT"
    assert "1.9" in out["prompt_block"]
    assert "999" not in out["prompt_block"]


# ───────────────────────── tier boundaries ─────────────────────────

def test_held_exactly_three_days_is_imminent_just_over_is_soon(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "AAA", "earnings_date": _earnings_date(3.0), "days_away": 0},
        {"ticker": "BBB", "earnings_date": _earnings_date(3.01), "days_away": 0},
    ])
    out = build_event_calendar([_pos("AAA"), _pos("BBB")], {"AAA", "BBB"},
                               calendar_path=cal, now=_NOW)
    tiers = {e["ticker"]: e["tier"] for e in out["events"]}
    assert tiers["AAA"] == "HELD_IMMINENT"   # <= 3 inclusive (api_earnings rule)
    assert tiers["BBB"] == "HELD_SOON"


def test_in_play_not_held_is_watch(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "MRVL", "earnings_date": _earnings_date(8.9), "days_away": 0},
    ])
    out = build_event_calendar([], {"MRVL"}, calendar_path=cal, now=_NOW)
    assert out["events"][0]["tier"] == "WATCH"
    assert out["events"][0]["held"] is False


def test_neither_held_nor_in_play_is_dropped(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "ZZZ", "earnings_date": _earnings_date(2.0), "days_away": 0},
    ])
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=cal, now=_NOW)
    assert out["events"] == []
    assert "no" in out["summary"].lower()


def test_past_event_is_dropped(tmp_path):
    """An earnings date already in the past must never reach the prompt
    (mirrors api_earnings' ``days_away >= -0.5`` filter)."""
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": _earnings_date(-1.0), "days_away": 0},
        {"ticker": "AMD", "earnings_date": _earnings_date(2.0), "days_away": 0},
    ])
    out = build_event_calendar([_pos("NVDA"), _pos("AMD")], {"NVDA", "AMD"},
                               calendar_path=cal, now=_NOW)
    tickers = [e["ticker"] for e in out["events"]]
    assert "NVDA" not in tickers   # already reported
    assert "AMD" in tickers


def test_watch_beyond_horizon_dropped_but_held_always_kept(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "FAR", "earnings_date": _earnings_date(40.0), "days_away": 0},
        {"ticker": "HELDFAR", "earnings_date": _earnings_date(40.0), "days_away": 0},
    ])
    out = build_event_calendar([_pos("HELDFAR")], {"FAR", "HELDFAR"},
                               calendar_path=cal, now=_NOW, horizon_days=14.0)
    tickers = [e["ticker"] for e in out["events"]]
    assert "FAR" not in tickers          # watch beyond horizon → noise, dropped
    assert "HELDFAR" in tickers          # a held name's print is never hidden


# ───────────────────────── sort order ─────────────────────────

def test_sort_imminent_then_soon_then_watch_then_soonest(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "WATCHSOON", "earnings_date": _earnings_date(1.0), "days_away": 0},
        {"ticker": "HELDSOON", "earnings_date": _earnings_date(7.0), "days_away": 0},
        {"ticker": "HELDIMM", "earnings_date": _earnings_date(2.0), "days_away": 0},
        {"ticker": "HELDIMM2", "earnings_date": _earnings_date(0.5), "days_away": 0},
    ])
    out = build_event_calendar(
        [_pos("HELDSOON"), _pos("HELDIMM"), _pos("HELDIMM2")],
        {"WATCHSOON", "HELDSOON", "HELDIMM", "HELDIMM2"},
        calendar_path=cal, now=_NOW)
    order = [e["ticker"] for e in out["events"]]
    assert order == ["HELDIMM2", "HELDIMM", "HELDSOON", "WATCHSOON"]


# ───────────────────────── _safe contract ─────────────────────────

def test_missing_file_degrades_no_raise(tmp_path):
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=tmp_path / "nope.json", now=_NOW)
    assert out["source_ok"] is False
    assert out["events"] == []
    assert isinstance(out["prompt_block"], str) and out["prompt_block"]
    # Honest fallback, never an exception that sinks a trading cycle.
    assert "earnings" in out["prompt_block"].lower()


def test_corrupt_json_degrades_no_raise(tmp_path):
    cal = tmp_path / "c.json"
    cal.write_text("{not valid json at all ::::")
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=cal, now=_NOW)
    assert out["source_ok"] is False
    assert out["events"] == []
    assert isinstance(out["prompt_block"], str) and out["prompt_block"]


# ───────────────────────── freshness pick ─────────────────────────

def test_freshest_snapshot_wins_when_two_candidates(tmp_path):
    """Mirrors the signals._db_path freshness discipline (invariant #15):
    given two readable candidate snapshots, the one with the newer ``as_of``
    is chosen, so a stale USB copy can't shadow a fresh local one."""
    stale = tmp_path / "stale.json"
    fresh = tmp_path / "fresh.json"
    _write_calendar(stale, [
        {"ticker": "OLD", "earnings_date": _earnings_date(2.0), "days_away": 0}],
        as_of=_NOW - timedelta(hours=30))
    _write_calendar(fresh, [
        {"ticker": "NEW", "earnings_date": _earnings_date(2.0), "days_away": 0}],
        as_of=_NOW - timedelta(minutes=5))
    chosen = event_calendar._pick_freshest([stale, fresh])
    assert chosen == fresh
    # order-independent
    assert event_calendar._pick_freshest([fresh, stale]) == fresh


def test_pick_freshest_skips_unreadable(tmp_path):
    good = tmp_path / "good.json"
    _write_calendar(good, [], as_of=_NOW)
    chosen = event_calendar._pick_freshest([tmp_path / "missing.json", good])
    assert chosen == good
    assert event_calendar._pick_freshest([tmp_path / "a.json",
                                          tmp_path / "b.json"]) is None


# ───────────────────────── observational voice ─────────────────────────

def test_block_is_observational_not_directive(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": _earnings_date(1.0), "days_away": 0},
    ])
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=cal, now=_NOW)
    pb = out["prompt_block"]
    assert "NVDA" in pb
    low = pb.lower()
    # Same contract as risk_mirror: states facts + reaffirms autonomy, issues
    # no directive. No imperative trade verb telling Opus what to do.
    assert "autonomy" in low
    for directive in ("you should sell", "you must ", "do not buy",
                      "avoid adding", "reduce ", "exit the"):
        assert directive not in low


def test_no_relevant_events_emits_honest_line_not_crash(tmp_path):
    cal = tmp_path / "c.json"
    _write_calendar(cal, [])  # valid file, zero events
    out = build_event_calendar([_pos("NVDA")], {"NVDA"},
                               calendar_path=cal, now=_NOW)
    assert out["source_ok"] is True
    assert out["events"] == []
    assert "no" in out["prompt_block"].lower()
    assert "earnings" in out["prompt_block"].lower()


# ───────────────────────── _build_payload wiring ─────────────────────────

def test_build_payload_renders_event_calendar_in_advisory_stack():
    from paper_trader import strategy

    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    marker = "EVENT-CALENDAR-BLOCK-MARKER earnings NVDA"
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={},
        self_review_block=None,
        track_record_block=None,
        risk_mirror_block="RISK-MIRROR-MARKER",
        event_calendar_block=marker,
    )
    assert marker in payload
    # Advisory awareness sits with the other mirrors, before market data.
    assert payload.index("RISK-MIRROR-MARKER") < payload.index(marker)
    assert payload.index(marker) < payload.index("WATCHLIST PRICES")


def test_build_payload_none_event_calendar_renders_no_stray_text():
    from paper_trader import strategy

    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={}, event_calendar_block=None)
    assert "EARNINGS CALENDAR" not in payload
    assert "None" not in payload.split("PORTFOLIO")[1].split("WATCHLIST")[0]


# ───────────────────────── /api/event-calendar parity ─────────────────────────

class TestEventCalendarEndpoint:
    """Prompt↔endpoint parity (the tail_risk / risk_mirror discipline): the
    route must serve the SAME builder against the live store's held names so
    the dashboard/chat can see exactly the earnings context Opus saw. Drives
    the real Flask view through ``app.test_client()`` on a fresh temp Store,
    with the on-disk snapshot redirected to a known fixture."""

    def test_endpoint_reflects_held_name_imminent_earnings(
            self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        # Hold NVDA so the endpoint's held set is non-empty and the event
        # tiers HELD_IMMINENT (not merely WATCH). open_positions() reads the
        # positions table, so seed it via upsert_position (record_trade alone
        # does not open a position row).
        s.upsert_position("NVDA", "stock", 1.0, 100.0)

        cal = tmp_path / "earnings_calendar.json"
        _write_calendar(cal, [
            {"ticker": "NVDA",
             "earnings_date": (datetime.now(timezone.utc)
                               + timedelta(days=2)).isoformat(),
             "days_away": 0}],
            as_of=datetime.now(timezone.utc))
        monkeypatch.setattr(event_calendar, "_CANDIDATE_PATHS", (cal,))

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/event-calendar")
        finally:
            s.close()

        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data
        assert data["source_ok"] is True
        nvda = [e for e in data["events"] if e["ticker"] == "NVDA"]
        assert nvda and nvda[0]["tier"] == "HELD_IMMINENT"
        assert "autonomy" in data["prompt_block"].lower()
        assert "NVDA" in data["prompt_block"]
