"""Tests for the standalone ``/api/alarm-latches`` endpoint + the additive
``latches`` block on ``/api/runner-heartbeat``.

The runner dedupes the two Discord silent-failure alarm classes
(consecutive-NO_DECISION circuit breaker, Claude quota exhaustion) via two
module-global latches. A trader sees the Discord FIRED / CLEARED bracket but
otherwise has no way to tell whether the latch is held *right now* — the gap
this endpoint closes.

Mirrors the ``test_core_dashboard_notify_health.py`` contract: a dedicated
*un-cached* endpoint plus the same data surfaced nested under
``/api/runner-heartbeat`` so panels read either source without drift.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, runner


@pytest.fixture(autouse=True)
def _reset_latch_globals(monkeypatch):
    """Each test starts with a clean latch state — no leftover wedge ts."""
    monkeypatch.setattr(runner, "_breaker_alert_active", False)
    monkeypatch.setattr(runner, "_quota_alert_active", False)
    monkeypatch.setattr(runner, "_consecutive_no_decisions", 0)
    monkeypatch.setattr(runner, "_no_decision_first_ts", None)
    monkeypatch.setattr(runner, "_quota_first_ts", None)


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


def _get_json(client, path):
    r = client.get(path)
    return r.status_code, r.get_json()


class TestAlarmLatchesEndpointContract:
    """Each latch combination produces the documented envelope shape."""

    def test_clean_state_returns_clear_verdict(self, client):
        status, body = _get_json(client, "/api/alarm-latches")
        assert status == 200
        assert body["verdict"] == "CLEAR"
        assert body["breaker_active"] is False
        assert body["quota_active"] is False
        assert body["any_active"] is False
        assert body["consecutive_no_decisions"] == 0
        assert body["breaker_outage_s"] is None
        assert body["quota_outage_s"] is None
        # The envelope adds service / as_of / a human headline.
        assert body["service"] == "paper_trader"
        assert "as_of" in body and body["as_of"]
        assert "no silent-failure alarm latches held" in body["headline"]

    def test_breaker_active_renders_latched_verdict(self, client, monkeypatch):
        ts = datetime.now(timezone.utc) - timedelta(seconds=600)
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        monkeypatch.setattr(runner, "_no_decision_first_ts", ts)
        status, body = _get_json(client, "/api/alarm-latches")
        assert status == 200
        assert body["verdict"] == "LATCHED"
        assert body["breaker_active"] is True
        assert body["quota_active"] is False
        assert body["any_active"] is True
        # Outage seconds round-trip — ±2 to absorb test-execution delay.
        assert abs(body["breaker_outage_s"] - 600) <= 2
        assert "CLAUDE BREAKER held" in body["headline"]

    def test_quota_active_renders_latched_verdict(self, client, monkeypatch):
        ts = datetime.now(timezone.utc) - timedelta(seconds=1800)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(runner, "_quota_first_ts", ts)
        status, body = _get_json(client, "/api/alarm-latches")
        assert status == 200
        assert body["verdict"] == "LATCHED"
        assert body["quota_active"] is True
        assert body["any_active"] is True
        assert abs(body["quota_outage_s"] - 1800) <= 2
        assert "QUOTA latch held" in body["headline"]

    def test_both_latches_held_lists_both_in_headline(self, client, monkeypatch):
        ts = datetime.now(timezone.utc) - timedelta(seconds=300)
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(runner, "_no_decision_first_ts", ts)
        monkeypatch.setattr(runner, "_quota_first_ts", ts)
        _, body = _get_json(client, "/api/alarm-latches")
        assert body["verdict"] == "LATCHED"
        # The composite headline names BOTH latches so the operator can see
        # the dual-failure case at a glance instead of only the first.
        assert "CLAUDE BREAKER held" in body["headline"]
        assert "QUOTA latch held" in body["headline"]


class TestAlarmLatchesEndpointNotCached:
    """The endpoint exists specifically to bypass the 20s SWR cache that
    ``/api/runner-heartbeat`` carries. Confirm mutating the runner-side
    latch is reflected on the very next request."""

    def test_mutation_reflected_immediately(self, client, monkeypatch):
        _, before = _get_json(client, "/api/alarm-latches")
        assert before["verdict"] == "CLEAR"
        # Operator scenario: breaker fires between two heartbeat polls.
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        _, after = _get_json(client, "/api/alarm-latches")
        # A cached endpoint would still read CLEAR. Confirm the no-cache
        # contract: the new state is visible without any TTL delay.
        assert after["verdict"] == "LATCHED"
        assert after["breaker_active"] is True


class TestAlarmLatchesEndpointFailurePath:
    """A fault inside ``runner.alarm_latch_state`` must degrade to a
    valid-shaped ERROR envelope so the dashboard panel can still render
    (mirrors ``/api/notify-health``'s failure contract)."""

    def test_exception_returns_500_with_error_envelope(self, client, monkeypatch):
        def _boom():
            raise RuntimeError("simulated latch fault")
        monkeypatch.setattr(runner, "alarm_latch_state", _boom)
        status, body = _get_json(client, "/api/alarm-latches")
        # 500 surfaces the fault upstream without breaking the panel JSON shape.
        assert status == 500
        assert body["verdict"] == "ERROR"
        assert "simulated latch fault" in body["headline"]
        # The safe defaults guarantee the dashboard never reads a missing key.
        assert body["breaker_active"] is False
        assert body["quota_active"] is False
        assert body["any_active"] is False


class TestRunnerHeartbeatLatchesBlock:
    """The same ``alarm_latch_state`` snapshot is nested under ``latches``
    on ``/api/runner-heartbeat`` so panels that already poll the heartbeat
    surface this without a second round-trip (mirrors the ``notify`` and
    ``singleton_lock`` additive blocks)."""

    def test_heartbeat_carries_latches_block(self, client, monkeypatch):
        ts = datetime.now(timezone.utc) - timedelta(seconds=120)
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        monkeypatch.setattr(runner, "_no_decision_first_ts", ts)
        # Disable SWR caching for this test so the heartbeat rebuilds inline
        # (otherwise the test could see a cached pre-mutation snapshot).
        monkeypatch.setattr(dashboard, "_swr_active", lambda: False)
        status, body = _get_json(client, "/api/runner-heartbeat")
        assert status == 200
        assert "latches" in body, (
            "runner-heartbeat dropped the latches block — the additive "
            "wiring regressed")
        assert body["latches"]["breaker_active"] is True
        # Round-trips the in-process counter so the panel and the dedicated
        # endpoint can never tell different stories.
        assert body["latches"]["consecutive_no_decisions"] == 0
        assert abs(body["latches"]["breaker_outage_s"] - 120) <= 2

    def test_heartbeat_latches_block_clean_when_no_outage(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "_swr_active", lambda: False)
        _, body = _get_json(client, "/api/runner-heartbeat")
        assert "latches" in body
        assert body["latches"]["any_active"] is False
        assert body["latches"]["breaker_outage_s"] is None
