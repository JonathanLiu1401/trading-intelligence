"""Tests for the standalone ``/api/notify-health`` endpoint.

Pins the contract that the same ``reporter.notify_health()`` snapshot already
nested inside ``/api/runner-heartbeat`` is *also* available un-cached at a
dedicated route. The whole point of the standalone endpoint is freshness:
``/api/runner-heartbeat`` is SWR-cached (20s TTL but can serve minutes-old
payloads under host-saturation), and an operator who suspects Discord is dark
needs the *current* in-process counter — not a frozen heartbeat snapshot.

These tests assert specific computed values for every distinct verdict
(UNKNOWN / HEALTHY / DEGRADED with ``restart_recommended`` toggling at the
3-consecutive-failure boundary) plus the failure-path envelope shape, so a
regression that breaks the route, the field names, or the
restart-recommended threshold fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard
from paper_trader import reporter


@pytest.fixture
def fresh_notify_state(monkeypatch):
    """Isolate the module-global delivery-health tracker per test.

    Mirrors the same fixture in ``tests/test_core_reporter.py`` —
    a single source of truth would force one test file to import the
    other's fixtures, which pytest discourages. Keep the two copies in
    lock-step.
    """
    monkeypatch.setattr(reporter, "_notify_state", {
        "last_attempt_ts": None, "last_ok_ts": None, "last_result": None,
        "consecutive_failures": 0, "last_error": "",
    })
    return reporter


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


def _get_json(client, path):
    r = client.get(path)
    return r.status_code, r.get_json()


class TestNotifyHealthEndpointContract:
    """Each verdict path produces the documented envelope shape."""

    def test_unknown_before_any_send(self, client, fresh_notify_state):
        status, body = _get_json(client, "/api/notify-health")
        assert status == 200
        assert body["verdict"] == "UNKNOWN"
        assert body["consecutive_failures"] == 0
        assert body["restart_recommended"] is False
        assert body["last_ok_ts"] is None
        assert body["last_attempt_ts"] is None
        # Envelope additions vs the bare reporter.notify_health() dict.
        assert body["service"] == "paper_trader"
        assert "as_of" in body and body["as_of"]

    def test_healthy_after_single_success(self, client, fresh_notify_state):
        reporter._record_send_outcome(True)
        status, body = _get_json(client, "/api/notify-health")
        assert status == 200
        assert body["verdict"] == "HEALTHY"
        assert body["consecutive_failures"] == 0
        assert body["last_ok_ts"] is not None
        assert body["restart_recommended"] is False

    def test_degraded_after_one_failure_no_restart_recommended(self, client, fresh_notify_state):
        reporter._record_send_outcome(False, "boom")
        status, body = _get_json(client, "/api/notify-health")
        assert status == 200
        assert body["verdict"] == "DEGRADED"
        assert body["consecutive_failures"] == 1
        # restart_recommended only flips at >=3 consecutive failures.
        assert body["restart_recommended"] is False
        # The headline carries the count + the original error text.
        assert "1 consecutive send failure," in body["headline"]
        assert body["last_error"] == "boom"

    def test_restart_recommended_at_three_consecutive_failures(self, client, fresh_notify_state):
        """The 3-failure boundary is load-bearing — pin it explicitly so a
        threshold tweak in ``reporter.notify_health`` is caught."""
        for _ in range(3):
            reporter._record_send_outcome(False, "rc=1")
        status, body = _get_json(client, "/api/notify-health")
        assert status == 200
        assert body["verdict"] == "DEGRADED"
        assert body["consecutive_failures"] == 3
        assert body["restart_recommended"] is True
        assert "3 consecutive send failures," in body["headline"]

    def test_success_after_failures_resets_counter_and_clears_recommendation(
            self, client, fresh_notify_state):
        for _ in range(5):  # well past the restart-recommended boundary
            reporter._record_send_outcome(False, "boom")
        # Confirm the state we set up.
        _, body = _get_json(client, "/api/notify-health")
        assert body["restart_recommended"] is True
        # One success clears the streak.
        reporter._record_send_outcome(True)
        _, body = _get_json(client, "/api/notify-health")
        assert body["verdict"] == "HEALTHY"
        assert body["consecutive_failures"] == 0
        assert body["restart_recommended"] is False
        assert body["last_error"] == ""


class TestNotifyHealthEndpointNotCached:
    """The whole reason this endpoint exists is to bypass the SWR cache.
    Confirm that mutating the tracker is reflected on the very next request —
    the caching layer would produce a stale verdict instead."""

    def test_mutation_is_reflected_in_the_next_call(self, client, fresh_notify_state):
        # Fresh tracker → UNKNOWN.
        _, before = _get_json(client, "/api/notify-health")
        assert before["verdict"] == "UNKNOWN"
        # Record a failure between two same-process requests.
        reporter._record_send_outcome(False, "x")
        _, after = _get_json(client, "/api/notify-health")
        # If a cache were live, `after` would still read UNKNOWN. With no
        # cache, the next call must reflect the new DEGRADED state immediately.
        assert after["verdict"] == "DEGRADED"
        assert after["consecutive_failures"] == 1

    def test_mutation_resets_on_success_immediately(self, client, fresh_notify_state):
        reporter._record_send_outcome(False, "x")
        reporter._record_send_outcome(False, "x")
        _, mid = _get_json(client, "/api/notify-health")
        assert mid["consecutive_failures"] == 2
        reporter._record_send_outcome(True)
        _, after = _get_json(client, "/api/notify-health")
        # Counter and verdict are both fresh — never serving a cached stale view.
        assert after["consecutive_failures"] == 0
        assert after["verdict"] == "HEALTHY"


class TestNotifyHealthEndpointFailurePath:
    """A fault inside ``reporter.notify_health`` must degrade to a valid-shaped
    ERROR envelope so the dashboard panel can render and the upstream
    cross-fetch never sees a 500 it would surface as 'endpoint dark'."""

    def test_non_dict_return_degrades_to_error_envelope(self, client, monkeypatch):
        monkeypatch.setattr(reporter, "notify_health", lambda: None)
        status, body = _get_json(client, "/api/notify-health")
        assert status == 200
        assert body["verdict"] == "ERROR"
        assert "non-dict" in body["headline"]
        assert body["restart_recommended"] is False

    def test_exception_returns_error_envelope_with_500(self, client, monkeypatch):
        def _boom():
            raise RuntimeError("simulated tracker fault")
        monkeypatch.setattr(reporter, "notify_health", _boom)
        status, body = _get_json(client, "/api/notify-health")
        # A genuine exception surfaces as 500 — upstream pollers see the fault
        # (not a fake success) but the envelope is still valid-shape JSON so
        # panels self-heal rather than rendering a generic browser 500 page.
        assert status == 500
        assert body["verdict"] == "ERROR"
        assert "simulated tracker fault" in body["last_error"]
        assert body["restart_recommended"] is False
