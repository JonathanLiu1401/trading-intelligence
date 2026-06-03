"""Tests for the ``/api/decision-cadence`` dashboard endpoint.

The endpoint wires ``build_decision_cadence`` to the dashboard so a trader
can see — without tailing the runner log — what tier the dynamic sleep
ladder is in, when the engine last decided anything (NO_DECISION counts —
this is the LOOP cadence, distinct from ``/api/last-real-decision``), and
the ETA to the next expected cycle. The OVERDUE verdict is the actionable
state: ``since_last_decision_s`` past ``OVERDUE_MULT × sleep_s`` means the
loop is wedged.

These tests verify:
  * The endpoint exists and returns 200.
  * The documented payload keys are present.
  * The endpoint reads ``store.open_positions`` and
    ``store.recent_decisions(limit=1)[0].timestamp`` (the canonical SSOTs —
    not ``last_real_decision``, which would silently mask a wedged loop
    cycling NO_DECISION at the expected cadence).
  * CORS is stamped (digital-intern cross-reads from another port).
  * Store faults degrade to a body, never a 500 stack.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pathlib import Path

import paper_trader.dashboard as d
from paper_trader.analytics import decision_cadence as dc

# Capture the REAL builder before any test can monkeypatch the module
# attribute — _delegate uses this so the patched stub can call through
# without recursing into itself.
_REAL_BUILDER = dc.build_decision_cadence
_EMPTY_CAL = Path("/tmp/__nonexistent_for_test_decision_cadence__")


def _delegate(positions, last_ts, now):
    """Call the REAL builder with a frozen wall-clock + an unreadable
    calendar path so the test is deterministic regardless of the host's
    on-disk earnings snapshot."""
    return _REAL_BUILDER(
        positions, last_ts,
        now=now,
        calendar_path=_EMPTY_CAL,
    )


@pytest.fixture
def client(monkeypatch):
    """A test client backed by an in-memory store stub."""

    now = datetime(2026, 5, 18, 18, 0, tzinfo=timezone.utc)
    # Sit 60s after the last decision in MARKET_OPEN tier → ON_SCHEDULE.
    last_ts = (now - timedelta(seconds=60)).isoformat()

    class _Store:
        def open_positions(self):
            return [{"ticker": "NVDA"}]

        def recent_decisions(self, limit=20):
            return [{
                "id": 1,
                "timestamp": last_ts,
                "market_open": 1,
                "signal_count": 7,
                "action_taken": "HOLD CASH → HOLD",
                "reasoning": "",
                "portfolio_value": 1000.0,
                "cash": 1000.0,
            }]

    monkeypatch.setattr(d, "get_store", lambda: _Store())

    # Pin the wall-clock + neutralise the on-disk earnings calendar so the
    # test is deterministic. Patches the module-level symbol; the
    # dashboard's ``from … import build_decision_cadence`` inside the
    # endpoint resolves to the current attribute value at call time.
    monkeypatch.setattr(
        dc, "build_decision_cadence",
        lambda positions, last_ts_arg, **kw: _delegate(
            positions, last_ts_arg, now,
        ),
    )

    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        yield c, now, last_ts


_REQUIRED_KEYS = (
    "as_of", "verdict", "headline", "tier", "sleep_s",
    "overdue_mult", "overdue_threshold_s",
    "last_decision_ts", "since_last_decision_s",
    "next_decision_expected_at", "next_decision_eta_s",
    "is_overdue", "n_positions", "market_open",
)


def test_route_exists_and_returns_required_keys(client):
    c, _now, _last = client
    r = c.get("/api/decision-cadence")
    assert r.status_code == 200
    j = r.get_json()
    assert "error" not in j, j
    for k in _REQUIRED_KEYS:
        assert k in j, f"missing {k!r} in {sorted(j)}"


def test_recent_decision_reports_on_schedule(client):
    """60s after the last decision in MARKET_OPEN (300s tier) → ON_SCHEDULE,
    ETA ~240s, is_overdue False."""
    c, _now, _last = client
    j = c.get("/api/decision-cadence").get_json()
    assert j["verdict"] == "ON_SCHEDULE"
    assert j["tier"] == "MARKET_OPEN"
    assert j["sleep_s"] == 300
    assert j["since_last_decision_s"] == 60
    assert j["next_decision_eta_s"] == 240
    assert j["is_overdue"] is False
    assert j["n_positions"] == 1


def test_endpoint_pulls_timestamp_from_recent_decisions_not_last_real(
    monkeypatch,
):
    """A wedged loop that has been cycling NO_DECISION at the expected
    cadence must NOT read as ON_SCHEDULE because last_real_decision is 6h
    old — the endpoint MUST key off the newest decisions row regardless
    of verb. This is the design contract that distinguishes this surface
    from /api/last-real-decision (which is purpose-built for the opposite
    question)."""

    now = datetime(2026, 5, 18, 18, 0, tzinfo=timezone.utc)
    # Newest row is a NO_DECISION 60s ago; the last real (HOLD) is 6h ago.
    no_dec_ts = (now - timedelta(seconds=60)).isoformat()
    real_ts = (now - timedelta(hours=6)).isoformat()

    seen_limits: list[int] = []

    class _Store:
        def open_positions(self):
            return [{"ticker": "NVDA"}]

        def recent_decisions(self, limit=20):
            seen_limits.append(limit)
            return [
                {"timestamp": no_dec_ts, "action_taken": "NO_DECISION"},
                {"timestamp": real_ts, "action_taken": "HOLD CASH → HOLD"},
            ][:limit]

        def last_real_decision(self):
            # This MUST NOT be called by the endpoint — explicit poison.
            raise AssertionError(
                "endpoint must read recent_decisions, not last_real_decision"
            )

    monkeypatch.setattr(d, "get_store", lambda: _Store())

    captured: dict = {}

    def _capture(positions, last_ts_arg, **kw):
        captured["last_ts"] = last_ts_arg
        return _delegate(positions, last_ts_arg, now)

    monkeypatch.setattr(dc, "build_decision_cadence", _capture)

    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        r = c.get("/api/decision-cadence")
        j = r.get_json()

    assert captured["last_ts"] == no_dec_ts, (
        "endpoint must pass the NEWEST decisions row's ts, "
        "regardless of NO_DECISION verb"
    )
    # And the endpoint asked for the leanest read possible (limit=1) —
    # not the default 20 + Python filtering. Cheap loop-cadence read.
    assert 1 in seen_limits, seen_limits
    assert j["verdict"] == "ON_SCHEDULE"


def test_cors_header_present_for_cross_fetch(client):
    """Digital-intern's chat / the unified dashboard cross-read this; the
    global _cors after_request must stamp it like every sibling endpoint."""
    c, _now, _last = client
    r = c.get("/api/decision-cadence")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_store_fault_returns_500_payload_not_stack(monkeypatch):
    """A store raise must degrade to a JSON ``error`` body, never the
    Flask default HTML 500 stack — the unified dashboard parses JSON."""

    class _BoomStore:
        def open_positions(self):
            raise RuntimeError("inject store fault")

        def recent_decisions(self, limit=20):
            raise RuntimeError("inject store fault")

    monkeypatch.setattr(d, "get_store", lambda: _BoomStore())
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        r = c.get("/api/decision-cadence")
        assert r.status_code == 500
        j = r.get_json()
        assert "error" in j
        assert "inject store fault" in j["error"]


def test_no_decisions_returns_no_data_verdict(monkeypatch):
    """A fresh-boot store with no decisions yet → NO_DATA verdict, no
    crash, sensible defaults."""
    now = datetime(2026, 5, 18, 18, 0, tzinfo=timezone.utc)

    class _Store:
        def open_positions(self):
            return []

        def recent_decisions(self, limit=20):
            return []

    monkeypatch.setattr(d, "get_store", lambda: _Store())
    monkeypatch.setattr(
        dc, "build_decision_cadence",
        lambda positions, last_ts_arg, **kw: _delegate(
            positions, last_ts_arg, now,
        ),
    )

    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        j = c.get("/api/decision-cadence").get_json()
        assert j["verdict"] == "NO_DATA"
        assert j["since_last_decision_s"] is None
        # No positions + weekday-mid-session UTC → MARKET_OPEN tier.
        assert j["tier"] == "MARKET_OPEN"
