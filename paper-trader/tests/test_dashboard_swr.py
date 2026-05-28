"""Stale-while-revalidate response cache for the slow :8090 endpoints.

Resolves AGENTS.md invariant #7's "remaining genuine concern": several
handlers (/api/briefing, /api/feed-health, /api/thesis-drift, …) do unbounded
yfinance / cross-DB I/O. ``threaded=True`` (d5b8eac, test_dashboard_threaded.py)
removed *cross-request* head-of-line blocking but not *per-endpoint* latency —
live user-perspective testing on 2026-05-17 measured /api/suggestions 5.2s,
/api/data-feed 16s, /api/briefing & /api/feed-health >35s, /api/thesis-drift
outright hung (curl -m15 → 000). ``dashboard.swr_cached`` serves the last good
payload instantly, refreshes single-flight in the background, and bounds the
cold path so the first caller never hangs.

These exercise the SWR machinery directly via ``app.test_request_context``
(the standard way to unit-test a Flask view decorator) with deterministic
fake builders — no DB, no network, no real :8090 bind. They opt the cache in
via ``_SWR_TEST_FORCE`` because it is deliberately inert under pytest so a
module-global response cache can't leak one endpoint test's fixture DB into
the next test's exact-value assertion.
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
    """Opt the otherwise-pytest-inert SWR cache in, with a fresh state map,
    a short cold budget, and a per-test executor (the production one is a
    process-global pool — a slow test's lingering build would otherwise
    starve a later test's cold path within the short budget)."""
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-test")
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
    monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 0.3)
    monkeypatch.setattr(d, "_SWR_STATE", {})
    monkeypatch.setattr(d, "_SWR_EXEC", pool)
    yield d
    # Wait for every background build (incl. the 1s slow-builder test's) so it
    # cannot write into a later test's freshly-swapped state.
    pool.shutdown(wait=True)


def _call(wrapped, query=""):
    """Drive the wrapped view through a real request context and normalize
    its return to (status, json-or-bytes)."""
    path = "/x" + (("?" + query) if query else "")
    with d.app.test_request_context(path):
        resp = d.app.make_response(wrapped())
    raw = resp.get_data()
    try:
        return resp.status_code, json.loads(raw)
    except Exception:
        return resp.status_code, raw


def test_inert_under_pytest_unless_forced(monkeypatch):
    """Default under pytest: the wrapper calls the handler directly every
    time — no caching, no honesty keys. This is what keeps the existing
    exact-value endpoint tests isolated."""
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", False)
    calls = []

    @d.swr_cached("inert", 60.0)
    def handler():
        calls.append(1)
        return jsonify({"n": len(calls)})

    s1, b1 = _call(handler)
    s2, b2 = _call(handler)
    assert s1 == 200 and s2 == 200
    assert b1 == {"n": 1} and b2 == {"n": 2}        # builder ran each time
    assert "cached" not in b1 and "cache_age_s" not in b1


def test_cold_build_then_warm_hit_skips_rebuild(swr):
    calls = []

    @d.swr_cached("warm", 60.0)
    def handler():
        calls.append(1)
        return jsonify({"build": len(calls)})

    s1, b1 = _call(handler)
    assert s1 == 200
    assert b1["build"] == 1
    assert b1["cached"] is False               # cold start served fresh
    assert b1["cache_age_s"] is not None       # command-center honesty keys

    s2, b2 = _call(handler)
    assert s2 == 200
    assert b2["build"] == 1                     # served from cache
    assert b2["cached"] is True
    assert len(calls) == 1                      # builder NOT re-invoked


def test_stale_serves_last_good_then_revalidates(swr):
    state = {"v": 1}

    @d.swr_cached("stale", 0.05)                # tiny TTL
    def handler():
        return jsonify({"v": state["v"]})

    _s, b1 = _call(handler)
    assert b1 == {"v": 1, "cached": False, "cache_age_s": b1["cache_age_s"]}

    state["v"] = 2                              # underlying changes
    time.sleep(0.08)                            # age past TTL

    # Stale-while-revalidate: the *old* value is served instantly (cached),
    # and a background refresh is kicked.
    _s, b2 = _call(handler)
    assert b2["v"] == 1
    assert b2["cached"] is True

    # The background refresh eventually lands the new value.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        _s, b3 = _call(handler)
        if b3["v"] == 2:
            break
        time.sleep(0.03)
    assert b3["v"] == 2


def test_cold_slow_builder_returns_fast_warming_then_real(swr):
    started = threading.Event()

    @d.swr_cached("slow", 60.0)
    def handler():
        started.set()
        time.sleep(1.0)                         # >> _SWR_COLD_BUDGET_S (0.3)
        return jsonify({"heavy": True})

    t0 = time.time()
    status, body = _call(handler)
    elapsed = time.time() - t0

    assert status == 200                        # never a 5xx, never a hang
    assert elapsed < 0.9                        # bounded ≈ cold budget
    assert body["warming"] is True
    assert body["cached"] is False
    assert started.is_set()                     # the build is running in bg

    # Once the slow build finishes it populates the cache for the next poll.
    deadline = time.time() + 2.5
    status_real = None
    while time.time() < deadline:
        status_real, b = _call(handler)
        if b.get("heavy") is True:
            break
        time.sleep(0.05)
    assert b.get("heavy") is True               # real payload eventually lands
    assert status_real == 200
    # (whether that first heavy serve is the cold-direct build or the now-warm
    # cache hit is an inherent race at the fill instant — not the property
    # under test, so it is deliberately not asserted.)


def test_non_200_is_not_cached_and_retried(swr):
    calls = []

    @d.swr_cached("err", 60.0)
    def handler():
        calls.append(1)
        return jsonify({"error": "boom", "n": len(calls)}), 500

    s1, b1 = _call(handler)
    s2, b2 = _call(handler)
    assert s1 == 500 and s2 == 500
    assert b1["n"] == 1 and b2["n"] == 2        # error never pinned; retried
    assert "cached" not in b1                   # passed straight through


def test_query_string_keys_separate_cache_entries(swr):
    @d.swr_cached("param", 60.0)
    def handler():
        from flask import request
        return jsonify({"days": request.args.get("days", "30")})

    _s, b30 = _call(handler, query="days=30")
    _s, b7 = _call(handler, query="days=7")
    assert b30["days"] == "30"
    assert b7["days"] == "7"                     # not a stale "30" cross-hit
    # both keys now independently cached
    _s, b30b = _call(handler, query="days=30")
    assert b30b == {"days": "30", "cached": True,
                    "cache_age_s": b30b["cache_age_s"]}


def test_list_body_served_without_meta_injection(swr):
    """A non-dict JSON body can't carry the honesty keys — it must still be
    cached and served verbatim, never crash _swr_make."""
    @d.swr_cached("listy", 60.0)
    def handler():
        return jsonify([1, 2, 3])

    s1, b1 = _call(handler)
    assert s1 == 200 and b1 == [1, 2, 3]
    s2, b2 = _call(handler)                      # cache hit, still a clean list
    assert s2 == 200 and b2 == [1, 2, 3]


def test_real_slow_routes_are_wrapped():
    """Lock the wiring: every slow yfinance/cross-DB endpoint identified by
    user-perspective testing carries the decorator (a future un-wrap is a
    regression — the panel goes back to hanging)."""
    wrapped = {
        "data_feed_api", "briefing_api", "suggestions_api",
        "scorer_predictions_api", "sector_heatmap_api", "correlation_api",
        "thesis_drift_api", "news_edge_api", "source_edge_api",
        "feed_health_api",
        # Added 2026-05-28 — sector_pulse was the only sector-card endpoint
        # still firing ~17 synchronous yfinance round-trips inline (live: 8s+,
        # browser panel hit its 10s timeout). Same SWR pattern as the others
        # — a future un-wrap is a regression.
        "sector_pulse_api",
    }
    for name in wrapped:
        fn = getattr(d, name)
        assert getattr(fn, "__wrapped__", None) is not None, (
            f"{name} lost its @swr_cached wrapper")
