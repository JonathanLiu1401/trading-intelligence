"""SWR cold-path *failure observability*.

The stale-while-revalidate machinery (dashboard.swr_cached) bounds a slow
endpoint's cold path: the first caller waits at most ``_SWR_COLD_BUDGET_S``
and otherwise gets a ``{"warming": true}`` placeholder while the build
finishes for the next poll.

The gap this file locks: the background ``_run`` rebuild used to
``except Exception: return None`` — so a handler that *reliably raises*
(a None-deref on a corrupted-mark book, a missing column, …) never
populates the cache and **every** poll re-returns the same opaque
``{"warming": true}`` *forever*, with the exception recorded **nowhere**
(no log, no counter, nothing the operator can see). Live user-perspective
testing on 2026-05-18 reproduced ``/api/briefing`` stuck ``warming:true``
for 60s+ with zero diagnostic, and ``/api/scorer-confidence`` (the one
expensive endpoint NOT swr-wrapped) hanging the request thread 30s+.

These assertions pin the diagnostic surface:

* a *failing* build surfaces ``attempts`` (consecutive failures),
  ``last_error`` (exception type + message) and ``stale_for_s`` in the
  warming body — broken is now distinguishable from merely slow;
* a *slow* (non-raising) build keeps ``attempts == 0`` / ``last_error is
  None`` — patience, not breakage;
* a success **resets** the failure counter (a transient blip is not
  reported forever);
* a consecutive failure is logged to stderr (operator signal without
  touching the 7k-line template);
* ``/api/scorer-confidence`` is swr-wrapped so a slow scorer replay can
  never wedge a Flask request thread again.

Mirrors test_dashboard_swr.py's white-box harness: drive the wrapped view
through a real request context with deterministic fake builders, opting
the otherwise-pytest-inert cache in via ``_SWR_TEST_FORCE``.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from flask import jsonify

import paper_trader.dashboard as d


@pytest.fixture
def swr(monkeypatch):
    """Opt the pytest-inert SWR cache in, with a fresh state map, a short
    cold budget and a per-test executor (identical to test_dashboard_swr.py
    so failure-path semantics match the happy-path tests)."""
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-fail-test")
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
    monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 0.3)
    monkeypatch.setattr(d, "_SWR_STATE", {})
    monkeypatch.setattr(d, "_SWR_EXEC", pool)
    yield d
    pool.shutdown(wait=True)


def _call(wrapped, query=""):
    path = "/x" + (("?" + query) if query else "")
    with d.app.test_request_context(path):
        resp = d.app.make_response(wrapped())
    raw = resp.get_data()
    try:
        return resp.status_code, json.loads(raw)
    except Exception:
        return resp.status_code, raw


def _poll(wrapped, predicate, timeout=3.0):
    """Drive the wrapped view until ``predicate(body)`` or deadline."""
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        _s, body = _call(wrapped)
        if isinstance(body, dict) and predicate(body):
            return body
        time.sleep(0.04)
    return body


def test_raising_builder_surfaces_attempts_and_last_error(swr):
    @d.swr_cached("boom", 60.0)
    def handler():
        raise RuntimeError("kaboom-detail")

    # First cold call already records one failure (the bg _run sets the
    # counter before returning None, which is what fut.result() waits on).
    status, body = _call(handler)
    assert status == 200                       # never a 5xx, never a hang
    assert body["warming"] is True
    assert body["cached"] is False
    assert body["attempts"] >= 1
    assert "RuntimeError" in body["last_error"]
    assert "kaboom-detail" in body["last_error"]
    assert isinstance(body["stale_for_s"], (int, float))
    assert body["stale_for_s"] >= 0

    # Repeated polls accumulate — the operator sees it is NOT self-healing.
    final = _poll(handler, lambda b: b.get("attempts", 0) >= 3)
    assert final["attempts"] >= 3
    assert "RuntimeError" in final["last_error"]


def test_slow_builder_warming_is_zero_attempts_and_no_error(swr):
    """A merely-slow (non-raising) build must read as 'be patient', NOT
    'broken' — attempts 0 and no last_error. This is the broken-vs-slow
    discriminator the operator needs."""
    @d.swr_cached("slowok", 60.0)
    def handler():
        time.sleep(1.0)                        # >> cold budget (0.3s)
        return jsonify({"heavy": True})

    status, body = _call(handler)
    assert status == 200
    assert body["warming"] is True
    assert body["attempts"] == 0
    assert body["last_error"] is None
    assert body["stale_for_s"] is None         # no failure has occurred


def test_success_resets_failure_counter(swr):
    """A transient failure run must not be reported forever: once a build
    succeeds the counter clears, and a later fresh failure restarts at 1
    (not the stale pre-success total)."""
    mode = {"v": "raise"}

    @d.swr_cached("flap", 60.0)
    def handler():
        if mode["v"] == "raise":
            raise ValueError("transient")
        return jsonify({"ok": True})

    _poll(handler, lambda b: b.get("attempts", 0) >= 2)
    st = d._SWR_STATE["flap?"]
    assert st["fail_count"] >= 2

    mode["v"] = "ok"
    landed = _poll(handler, lambda b: b.get("ok") is True)
    assert landed["ok"] is True
    st = d._SWR_STATE["flap?"]
    assert st["fail_count"] == 0               # reset on success
    assert st["last_error"] is None
    assert st["last_ok_ts"] > 0

    # Force a cold path again and fail freshly: counter restarts at 1.
    st["data"] = None
    mode["v"] = "raise"
    fresh = _poll(handler, lambda b: b.get("warming") is True
                  and b.get("attempts", 0) >= 1)
    assert fresh["attempts"] == 1              # NOT 3 — no stale carry-over


def test_consecutive_failure_is_logged_to_stderr(swr, capfd):
    """Operator signal without touching the template: a failing background
    build prints a throttled `[swr]` line naming the key and exception."""
    @d.swr_cached("noisy", 60.0)
    def handler():
        raise RuntimeError("logme")

    _poll(handler, lambda b: b.get("attempts", 0) >= 1)
    time.sleep(0.05)                           # let the bg thread flush
    out, err = capfd.readouterr()
    assert "[swr]" in err
    assert "noisy" in err
    assert "RuntimeError" in err
    assert "logme" in err


def test_scorer_confidence_endpoint_is_swr_wrapped():
    """Regression guard: /api/scorer-confidence must stay swr-wrapped so a
    slow scorer replay can never again hang a Flask request thread 30s+
    (live-measured code 000 @ 30s on 2026-05-18). swr_cached uses
    functools.wraps, which sets __wrapped__ on the view function."""
    # Mechanism sanity: a known swr-wrapped sibling exposes __wrapped__.
    assert hasattr(d.briefing_api, "__wrapped__"), (
        "harness invalid — briefing_api should be swr-wrapped")
    assert hasattr(d.scorer_confidence_api, "__wrapped__"), (
        "/api/scorer-confidence is not swr-wrapped — it will hang the "
        "request thread on a cold scorer replay")
