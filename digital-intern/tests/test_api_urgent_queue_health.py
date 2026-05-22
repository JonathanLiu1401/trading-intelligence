"""/api/urgent-queue-health — wire-up for ArticleStore.urgent_queue_health.

Pins the dashboard endpoint contract (verdict ladder + clamping + 500-on-
error + 503/401) with specific expected values, not "no crash". The
underlying ``urgent_queue_health`` method has its own pinned tests at
``tests/test_urgent_queue_health.py`` (backlog counting, near-reap/overdue
classification, live-only exclusion, per-ticker breakdown); this file owns
the HTTP-layer translation:

  * ``quiet``       — empty backlog, queued=0
  * ``ok``          — queued>0 but nothing near the reap deadline
  * ``near_reap``   — >=1 row within near_reap_hours of the deadline
  * ``items_lost``  — >=1 overdue row (urgent push silently dropped)
  * reap_hours clamp 1..168, near_hours clamp 0..reap
  * underlying method raising → 500 with shape
  * no-store-set → 503; WEB_API_KEY enforced
"""
from __future__ import annotations

from dashboard.web_server import create_app


class _FakeStore:
    """Stub mirroring ArticleStore.urgent_queue_health's exact shape."""

    def __init__(self, payload=None, raises=False):
        self._payload = payload
        self._raises = raises
        self.last_kwargs = None

    def urgent_queue_health(self, tickers=None, reap_age_hours=24,
                            near_reap_hours=3.0):
        self.last_kwargs = {
            "tickers": tickers,
            "reap_age_hours": reap_age_hours,
            "near_reap_hours": near_reap_hours,
        }
        if self._raises:
            raise RuntimeError("lock retry exhausted")
        return dict(self._payload)


def _client(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    return create_app(store=store).test_client()


def _payload(queued=0, oldest=None, near=0, overdue=0, by_ticker=None):
    return {
        "queued": queued,
        "oldest_age_h": oldest,
        "near_reap": near,
        "overdue": overdue,
        "reap_age_hours": 24,
        "near_reap_hours": 3.0,
        "by_ticker": by_ticker or [],
    }


def test_quiet_when_backlog_empty(monkeypatch):
    store = _FakeStore(_payload(queued=0))
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "quiet"
    assert d["queued"] == 0
    assert "as_of" in d


def test_ok_when_queue_has_no_aging_rows(monkeypatch):
    """A non-empty backlog with nothing near the deadline is healthy."""
    store = _FakeStore(_payload(queued=4, oldest=6.0, near=0, overdue=0))
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    d = r.get_json()
    assert d["status"] == "ok"
    assert d["queued"] == 4


def test_near_reap_when_rows_close_to_deadline(monkeypatch):
    store = _FakeStore(_payload(queued=3, oldest=22.0, near=2, overdue=0))
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    d = r.get_json()
    assert d["status"] == "near_reap"
    assert d["near_reap"] == 2


def test_items_lost_when_rows_overdue(monkeypatch):
    """An overdue row is an urgent push the analyst never got — the verdict
    must escalate above near_reap even if near_reap rows also exist."""
    store = _FakeStore(_payload(queued=5, oldest=40.0, near=1, overdue=2))
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    d = r.get_json()
    assert d["status"] == "items_lost"
    assert d["overdue"] == 2


def test_per_ticker_passthrough(monkeypatch):
    store = _FakeStore(_payload(
        queued=2, oldest=20.0, near=1, overdue=0,
        by_ticker=[{"ticker": "NVDA", "queued": 1, "oldest_age_h": 20.0,
                    "near_reap": 1, "overdue": 0}],
    ))
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    d = r.get_json()
    assert d["by_ticker"][0]["ticker"] == "NVDA"


def test_params_clamped(monkeypatch):
    store = _FakeStore(_payload(queued=0))
    client = _client(store, monkeypatch)

    client.get("/api/urgent-queue-health?reap_hours=0")
    assert store.last_kwargs["reap_age_hours"] == 1  # clamped low

    client.get("/api/urgent-queue-health?reap_hours=99999")
    assert store.last_kwargs["reap_age_hours"] == 168  # clamped high

    client.get("/api/urgent-queue-health?reap_hours=bad")
    assert store.last_kwargs["reap_age_hours"] == 24  # default

    # near_hours can never exceed reap_hours.
    client.get("/api/urgent-queue-health?reap_hours=10&near_hours=50")
    assert store.last_kwargs["near_reap_hours"] == 10.0


def test_underlying_failure_yields_500_with_error_shape(monkeypatch):
    store = _FakeStore(raises=True)
    r = _client(store, monkeypatch).get("/api/urgent-queue-health")
    assert r.status_code == 500
    d = r.get_json()
    assert "error" in d
    assert "lock retry exhausted" in d["error"]


def test_no_store_yields_503(monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    import dashboard.web_server as ws
    monkeypatch.setattr(ws, "_store", None)
    app = create_app()
    r = app.test_client().get("/api/urgent-queue-health")
    assert r.status_code == 503
    assert "store unavailable" in r.get_json().get("error", "")


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("WEB_API_KEY", "secret-token")
    store = _FakeStore(_payload(queued=0))
    client = create_app(store=store).test_client()

    assert client.get("/api/urgent-queue-health").status_code == 401
    assert client.get(
        "/api/urgent-queue-health?key=secret-token"
    ).status_code == 200
