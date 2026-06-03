"""/api/stats partial-degrade — defense-in-depth on top of the (already in
HEAD) _expect_row cursor-collision fix.

Root cause of the original "500 / dashboard blind" symptom: ``store.stats()``
returning ``fetchone()=None`` on a shared-conn cursor collision is fixed by
``_expect_row`` + ``@_retry_on_lock``. But ``/api/stats`` also makes TWO
SUPPLEMENTARY ``store.stats_since(...)`` calls, each its own retry budget.
Before this change a transient retry-exhaustion on EITHER supplementary
window 500'd the WHOLE payload — the dashboard went fully dark over a slow
secondary tile while the core gauge was perfectly healthy.

These pin the contract with specific behaviour (not "no crash"):
  * core ``stats()`` healthy + supplementary calls failing → 200, core gauge
    intact, ``last_hour``/``last_24h`` present-but-null, ``degraded: true``
    (key present so a client doing ``s.last_hour`` gets null, never an
    undefined-property crash);
  * supplementary calls healthy → no ``degraded`` flag leaks in;
  * core ``stats()`` itself exhausting its budget IS a genuine outage → 500.
"""
from __future__ import annotations

import pytest

from dashboard.web_server import create_app


class _FakeStore:
    def __init__(self, *, stats_ok=True, since_raises=False):
        self._stats_ok = stats_ok
        self._since_raises = since_raises

    def stats(self):
        if not self._stats_ok:
            raise RuntimeError("lock retry exhausted — core gauge down")
        return {"total": 123, "urgent": 4, "unscored": 7,
                "below_threshold": 2, "db_mb": 9.9,
                "lock_retries": 0, "lock_failures": 0}

    def stats_since(self, hours):
        if self._since_raises:
            raise RuntimeError("supplementary window contended")
        return {"total": 5 if hours == 1 else 50,
                "urgent": 1 if hours == 1 else 3}


def _client(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    import dashboard.web_server as _ws
    _ws._dashboard_cache.clear()
    return create_app(store=store).test_client()


def test_happy_path_no_degraded_flag(monkeypatch):
    c = _client(_FakeStore(), monkeypatch)
    r = c.get("/api/stats")
    assert r.status_code == 200
    d = r.get_json()
    assert d["total"] == 123 and d["urgent"] == 4
    assert d["last_hour"] == {"total": 5, "urgent": 1}
    assert d["last_24h"] == {"total": 50, "urgent": 3}
    assert "degraded" not in d, "degraded must not leak on a fully healthy read"


def test_supplementary_failure_degrades_not_500(monkeypatch):
    c = _client(_FakeStore(since_raises=True), monkeypatch)
    r = c.get("/api/stats")
    # The whole point: a transient supplementary failure must NOT 500.
    assert r.status_code == 200
    d = r.get_json()
    # Core gauge survives untouched.
    assert d["total"] == 123 and d["urgent"] == 4 and d["unscored"] == 7
    # Supplementary keys present-but-null (no undefined-property crash for a
    # client doing s.last_hour) and the degrade is flagged.
    assert d["last_hour"] is None and d["last_24h"] is None
    assert d["degraded"] is True


def test_core_stats_failure_is_a_real_500(monkeypatch):
    c = _client(_FakeStore(stats_ok=False), monkeypatch)
    r = c.get("/api/stats")
    assert r.status_code == 500
    assert "error" in r.get_json()
