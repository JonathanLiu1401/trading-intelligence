"""/api/source-throughput — wire-up for ArticleStore.source_throughput.

Pins the dashboard endpoint contract (verdict ladder + clamping + 500-on-
error) with specific expected values. The underlying ``source_throughput``
method already has its own tests; this file owns the HTTP layer translation.

A source can decelerate sharply while its newest item is still only minutes
old — ``/api/collector-health`` (1h/24h counts) won't flag it, but this will.
"""
from __future__ import annotations

from dashboard.web_server import create_app


class _FakeStore:
    """Stub mirroring ArticleStore.source_throughput's exact shape."""

    def __init__(self, rows=None, raises=False):
        self._rows = rows or []
        self._raises = raises
        self.last_window_min = None

    def source_throughput(self, window_min: int = 60):
        self.last_window_min = window_min
        if self._raises:
            raise RuntimeError("source_throughput broke")
        return list(self._rows)


def _client(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    return create_app(store=store).test_client()


def test_status_ok_when_no_deceleration(monkeypatch):
    """All sources steady or accelerating — verdict is ``ok``."""
    store = _FakeStore(rows=[
        {"source": "rss", "recent": 100, "prior": 95, "delta": 5, "decel_pct": -5.3},
        {"source": "web", "recent": 50, "prior": 50, "delta": 0, "decel_pct": 0.0},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "ok"
    assert d["n_critical"] == 0
    assert d["n_degraded"] == 0
    assert len(d["sources"]) == 2


def test_status_degraded_when_mid_decline(monkeypatch):
    """A 40-75% decel on a meaningful baseline triggers ``degraded`` —
    investigate before next briefing. Baseline must be >= MIN_PRIOR (5)."""
    store = _FakeStore(rows=[
        {"source": "gdelt", "recent": 30, "prior": 60, "delta": -30, "decel_pct": 50.0},
        {"source": "rss", "recent": 100, "prior": 95, "delta": 5, "decel_pct": -5.3},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    d = r.get_json()
    assert d["status"] == "degraded"
    assert d["n_degraded"] == 1
    assert d["n_critical"] == 0


def test_status_critical_when_sharp_decline(monkeypatch):
    """A >=75% decel on a meaningful baseline triggers ``critical`` — source
    is effectively dark."""
    store = _FakeStore(rows=[
        {"source": "finnhub", "recent": 2, "prior": 40, "delta": -38, "decel_pct": 95.0},
        {"source": "rss", "recent": 100, "prior": 95, "delta": 5, "decel_pct": -5.3},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    d = r.get_json()
    assert d["status"] == "critical"
    assert d["n_critical"] == 1


def test_critical_dominates_degraded(monkeypatch):
    """Mixed critical + degraded must report ``critical`` (the worse signal),
    not collapse to ``degraded``."""
    store = _FakeStore(rows=[
        {"source": "finnhub", "recent": 2, "prior": 40, "delta": -38, "decel_pct": 95.0},
        {"source": "gdelt", "recent": 30, "prior": 60, "delta": -30, "decel_pct": 50.0},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    d = r.get_json()
    assert d["status"] == "critical"
    assert d["n_critical"] == 1
    assert d["n_degraded"] == 1


def test_low_prior_noise_excluded_from_verdict(monkeypatch):
    """Long-tail one-off sub-tag noise — a single article last hour, zero this
    hour — must NOT count as critical. Live evidence (2026-05-20): the
    ``ArticleStore.source_throughput`` 60-min window returned 8+ rows of
    ``prior=1, recent=0, decel_pct=100`` for one-off GDELT/AlphaVantage host
    keys. Without a baseline floor every cycle reported ``critical`` on a
    healthy daemon — false alarm. Sources must have prior >= 5 to count.

    Crucially the rows ARE still returned in ``sources`` so an operator can
    still see them; only the verdict count is gated.
    """
    store = _FakeStore(rows=[
        {"source": "GDELT/longtail1", "recent": 0, "prior": 1, "delta": -1, "decel_pct": 100.0},
        {"source": "GDELT/longtail2", "recent": 0, "prior": 2, "delta": -2, "decel_pct": 100.0},
        {"source": "GDELT/longtail3", "recent": 0, "prior": 3, "delta": -3, "decel_pct": 100.0},
        {"source": "GDELT/longtail4", "recent": 0, "prior": 4, "delta": -4, "decel_pct": 100.0},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    d = r.get_json()
    assert d["status"] == "ok", (
        "low-prior noise rows must not inflate the verdict to critical"
    )
    assert d["n_critical"] == 0
    assert d["n_degraded"] == 0
    # All rows still surfaced in the sources list — the operator can inspect.
    assert len(d["sources"]) == 4


def test_none_decel_does_not_inflate_status(monkeypatch):
    """A brand-new source has ``decel_pct=None`` (no prior baseline). It must
    NOT count as critical/degraded — ``None`` is "no signal", never an alarm."""
    store = _FakeStore(rows=[
        {"source": "brand_new", "recent": 5, "prior": 0, "delta": 5, "decel_pct": None},
    ])
    r = _client(store, monkeypatch).get("/api/source-throughput")
    d = r.get_json()
    assert d["status"] == "ok"
    assert d["n_critical"] == 0
    assert d["n_degraded"] == 0


def test_window_min_clamped(monkeypatch):
    """``window_min`` clamped 5..720; out-of-range mustn't 500 or hit store
    with garbage."""
    store = _FakeStore(rows=[])
    client = _client(store, monkeypatch)

    client.get("/api/source-throughput?window_min=1")
    assert store.last_window_min == 5  # clamped low

    client.get("/api/source-throughput?window_min=10000")
    assert store.last_window_min == 720  # clamped high

    client.get("/api/source-throughput?window_min=garbage")
    assert store.last_window_min == 60  # fallback


def test_limit_applied(monkeypatch):
    """``limit`` truncates the sources list to the requested cap."""
    rows = [
        {"source": f"s{i}", "recent": 10, "prior": 20,
         "delta": -10, "decel_pct": 50.0}
        for i in range(20)
    ]
    store = _FakeStore(rows=rows)
    r = _client(store, monkeypatch).get("/api/source-throughput?limit=5")
    d = r.get_json()
    assert len(d["sources"]) == 5
    # All 20 still count toward verdict (the cap is presentation-only).
    assert d["n_degraded"] == 20


def test_underlying_method_failure_yields_500(monkeypatch):
    store = _FakeStore(raises=True)
    r = _client(store, monkeypatch).get("/api/source-throughput")
    assert r.status_code == 500
    d = r.get_json()
    assert "error" in d
    assert "source_throughput broke" in d["error"]


def test_no_store_yields_503(monkeypatch):
    import dashboard.web_server as ws
    monkeypatch.setattr(ws, "_store", None)
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    app = create_app()
    r = app.test_client().get("/api/source-throughput")
    assert r.status_code == 503


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("WEB_API_KEY", "shh")
    store = _FakeStore(rows=[])
    client = create_app(store=store).test_client()
    assert client.get("/api/source-throughput").status_code == 401
    assert client.get("/api/source-throughput?key=shh").status_code == 200
