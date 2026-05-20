"""/api/urgent-label-split — wire-up for ArticleStore.urgency_label_split.

Pins the dashboard endpoint contract (verdict ladder + clamping + 500-on-
error) with specific expected values, not "no crash". The underlying
``urgency_label_split`` method already has its own pinned tests at
``tests/test_urgency_label_split.py`` (the SQL aggregation, live-only
exclusion, fixed 4-key shape); this file owns the HTTP layer translation:

  * ``quiet``                 — empty window, total=0
  * ``healthy``               — >=50% LLM-vetted
  * ``mostly_unverified``     — <50% LLM with total>=5
  * ``unverified_storm``      — 0% LLM with total>=3 (the live-evidence case)
  * hours clamp 1..168
  * underlying method raising → 500 with shape, never a Flask 500-on-exception
  * no-store-set → 503
"""
from __future__ import annotations

import pytest

from dashboard.web_server import create_app


class _FakeStore:
    """Stub mirroring ArticleStore.urgency_label_split's exact shape."""

    def __init__(self, payload=None, raises=False):
        self._payload = payload
        self._raises = raises
        self.last_hours = None

    def urgency_label_split(self, hours: int = 24):
        self.last_hours = hours
        if self._raises:
            raise RuntimeError("lock retry exhausted")
        return self._payload


def _client(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    return create_app(store=store).test_client()


def _payload(by_source, llm_fraction, total, window_h=24):
    return {
        "window_h": window_h,
        "total": total,
        "by_source": by_source,
        "llm_fraction": llm_fraction,
    }


def test_quiet_when_no_urgent_in_window(monkeypatch):
    """An empty window must collapse to ``quiet`` (no manufactured alarm)."""
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0},
        llm_fraction=0.0,
        total=0,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "quiet"
    assert d["total"] == 0
    assert d["llm_fraction"] == 0.0
    assert d["by_source"]["llm"] == 0
    assert "as_of" in d


def test_healthy_when_llm_majority(monkeypatch):
    """>=50% LLM-vetted is healthy — the calibration path is alive."""
    store = _FakeStore(_payload(
        by_source={"llm": 7, "ml": 3, "briefing_boost": 1, "null": 0},
        llm_fraction=0.7273,
        total=11,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "healthy"
    assert d["total"] == 11
    assert d["llm_fraction"] == pytest.approx(0.7273, rel=1e-3)


def test_unverified_storm_when_zero_llm_and_total_ge_3(monkeypatch):
    """The exact live-evidence pattern: all urgent rows model-only — the
    Sonnet urgency_scorer is dark/throttled. Verdict must escalate so the
    analyst sees the channel is single-headed."""
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 5, "briefing_boost": 0, "null": 0},
        llm_fraction=0.0,
        total=5,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "unverified_storm"
    assert d["total"] == 5
    assert d["llm_fraction"] == 0.0


def test_zero_llm_but_below_minimum_total_is_healthy(monkeypatch):
    """A *single* unverified urgent row is not a storm — needs total>=3 to
    distinguish noise from a calibration failure. 2 with 0% LLM stays
    ``healthy`` (no manufactured alarm)."""
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 2, "briefing_boost": 0, "null": 0},
        llm_fraction=0.0,
        total=2,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    d = r.get_json()
    assert d["status"] == "healthy", (
        "below minimum-total threshold should not escalate to unverified_storm"
    )


def test_mostly_unverified_when_low_llm_with_large_total(monkeypatch):
    """<50% LLM with total>=5 — most urgent rows are model-only but the LLM
    path is producing *some*, so degraded rather than full storm."""
    store = _FakeStore(_payload(
        by_source={"llm": 2, "ml": 8, "briefing_boost": 0, "null": 0},
        llm_fraction=0.2,
        total=10,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    d = r.get_json()
    assert d["status"] == "mostly_unverified"


def test_briefing_boost_counts_as_vetted(monkeypatch):
    """``briefing_boost`` is a real Opus-curated label, NOT model self-prediction
    — same training treatment as ``llm`` in storage/article_store.py. Must
    contribute to the vetted fraction so a window with strong briefing-boost
    coverage doesn't read as ``unverified_storm``."""
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 2, "briefing_boost": 5, "null": 0},
        llm_fraction=5 / 7,
        total=7,
    ))
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    d = r.get_json()
    assert d["status"] == "healthy"


def test_hours_clamped_to_range(monkeypatch):
    """``hours`` must be clamped 1..168; out-of-range values mustn't 500
    and mustn't reach the store as garbage."""
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0},
        llm_fraction=0.0,
        total=0,
    ))
    client = _client(store, monkeypatch)

    r = client.get("/api/urgent-label-split?hours=0")
    assert r.status_code == 200
    assert store.last_hours == 1  # clamped low

    r = client.get("/api/urgent-label-split?hours=10000")
    assert r.status_code == 200
    assert store.last_hours == 168  # clamped high

    r = client.get("/api/urgent-label-split?hours=not-a-number")
    assert r.status_code == 200
    assert store.last_hours == 24  # falls back to default


def test_underlying_method_failure_yields_500_with_error_shape(monkeypatch):
    """A raised exception in the store method must surface as JSON 500 with
    an ``error`` key — never a Flask debug HTML page that breaks a JS
    dashboard consumer."""
    store = _FakeStore(payload=None, raises=True)
    r = _client(store, monkeypatch).get("/api/urgent-label-split")
    assert r.status_code == 500
    d = r.get_json()
    assert "error" in d
    assert "lock retry exhausted" in d["error"]


def test_no_store_yields_503(monkeypatch):
    """When the daemon hasn't wired the store yet (test client built without
    ``store=`` arg), the endpoint must report 503, not crash."""
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    # Reset the module-level _store to None so create_app() doesn't reuse a
    # prior store from a sibling test.
    import dashboard.web_server as ws
    monkeypatch.setattr(ws, "_store", None)
    app = create_app()  # no store
    r = app.test_client().get("/api/urgent-label-split")
    assert r.status_code == 503
    d = r.get_json()
    assert "store unavailable" in d.get("error", "")


def test_api_key_enforced(monkeypatch):
    """When WEB_API_KEY is set, the endpoint must require ?key= — same
    discipline as every other /api/* route in the dashboard."""
    monkeypatch.setenv("WEB_API_KEY", "secret-token")
    store = _FakeStore(_payload(
        by_source={"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0},
        llm_fraction=0.0,
        total=0,
    ))
    client = create_app(store=store).test_client()

    r = client.get("/api/urgent-label-split")
    assert r.status_code == 401

    r = client.get("/api/urgent-label-split?key=secret-token")
    assert r.status_code == 200
