"""End-to-end test_client smokes for /api/ticker-velocity and
/api/ticker-comentions.

Thin — the pure builders are tested in test_ticker_velocity.py and
test_ticker_comentions.py. This file pins ONLY the things the pure tests
cannot: the Flask routes are registered, the ``_LIVE_ONLY_SQL`` filter is
actually applied (backtest rows must not leak), and the route returns the
builder's shape verbatim.

Per project-memory: verify via the Flask test client, not a module
``__main__`` smoke that may hit a different/empty DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.web_server import create_app


def _ts(hours_ago: float, ref: datetime | None = None) -> str:
    ref = ref or datetime.now(timezone.utc)
    return (ref - timedelta(hours=hours_ago)).isoformat()


def _insert(store, *, id, url, title, source, hours_ago):
    """Direct article insertion (no pipeline scoring)."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            " urgency, first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, None, 1.0, 0.0, 0,
             _ts(hours_ago), 0, 0.0, "test", None),
        )
        store.conn.commit()


def _seed_breaking_nvda(store, n_recent=7, n_prior=0):
    """Seed a clear BREAKING NVDA pattern: recent burst, empty prior so
    the Laplace-smoothed ratio comfortably exceeds BREAKING_RATIO=4.0."""
    for i in range(n_recent):
        _insert(store, id=f"vrec{i}",
                url=f"https://example.com/vrec{i}",
                title=f"NVDA fresh story {i}",
                source="rss", hours_ago=0.2)
    for i in range(n_prior):
        _insert(store, id=f"vpri{i}",
                url=f"https://example.com/vpri{i}",
                title=f"NVDA prior cycle {i}",
                source="rss", hours_ago=3.0)


def test_ticker_velocity_endpoint_returns_builder_shape(store):
    _seed_breaking_nvda(store)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-velocity")
    assert r.status_code == 200
    payload = r.get_json()
    for key in ("generated_at", "window_min", "top_n", "rows_scanned",
                "rows_in_window", "verdict", "headline",
                "n_breaking", "n_warming", "tickers"):
        assert key in payload, key
    assert payload["verdict"] == "BREAKING"
    nvda = next(t for t in payload["tickers"] if t["ticker"] == "NVDA")
    assert nvda["verdict"] == "BREAKING"
    assert nvda["recent"] >= 5


def test_ticker_velocity_endpoint_excludes_backtest_rows(store):
    """backtest:// URLs / backtest_* / opus_annotation* sources must be
    filtered out — synthetic injection bursts must not look like a
    breaking ticker."""
    for i in range(8):
        _insert(store, id=f"btv{i}",
                url=f"backtest://run_99/ticker/NVDA/{i}",
                title=f"NVDA fake burst {i}",
                source="backtest_run_99_winner", hours_ago=0.2)
    for i in range(4):
        _insert(store, id=f"opv{i}",
                url=f"https://example.com/op-v{i}",
                title=f"AMD synthetic burst {i}",
                source="opus_annotation_cycle_42", hours_ago=0.2)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-velocity")
    assert r.status_code == 200
    payload = r.get_json()
    # Every row is synthetic — must NOT register as live velocity.
    assert payload["verdict"] == "NO_DATA"
    assert payload["tickers"] == []


def test_ticker_velocity_endpoint_respects_window_min(store):
    """Default window=120min. With ?window_min=30, articles >60min ago drop
    out of the recent+prior window entirely."""
    for i in range(8):
        _insert(store, id=f"vwl{i}",
                url=f"https://example.com/vwl{i}",
                title=f"WDC rally {i}",
                source="rss", hours_ago=2.0)  # 120 min ago
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-velocity?window_min=30")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["window_min"] == 30
    # All 8 articles are 120min old → outside 2× 30min = 60min window.
    assert payload["verdict"] == "NO_DATA"


def test_ticker_velocity_endpoint_clamps_window_min(store):
    """?window_min outside 30..720 must clamp."""
    app = create_app(store)
    client = app.test_client()
    # Below floor → clamps to 30.
    r1 = client.get("/api/ticker-velocity?window_min=5")
    assert r1.status_code == 200
    assert r1.get_json()["window_min"] == 30
    # Above ceiling → clamps to 720.
    r2 = client.get("/api/ticker-velocity?window_min=99999")
    assert r2.status_code == 200
    assert r2.get_json()["window_min"] == 720


def _seed_burst_pair(store, n=6):
    """Seed NVDA+AMD pair burst — SECTOR_BURST trigger."""
    for i in range(n):
        _insert(store, id=f"cm{i}",
                url=f"https://example.com/cm{i}",
                title=f"NVDA AMD chip rally {i}",
                source="rss", hours_ago=0.5)


def test_ticker_comentions_endpoint_returns_builder_shape(store):
    _seed_burst_pair(store)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-comentions")
    assert r.status_code == 200
    payload = r.get_json()
    for key in ("generated_at", "window_hours", "top_n", "rows_scanned",
                "rows_in_window", "unique_pairs", "qualified_pairs",
                "min_pair_count", "burst_lift_threshold", "burst_min_co",
                "verdict", "headline", "top"):
        assert key in payload, key
    assert payload["verdict"] == "SECTOR_BURST"
    assert len(payload["top"]) >= 1
    top = payload["top"][0]
    assert sorted(top["pair"]) == ["AMD", "NVDA"]


def test_ticker_comentions_endpoint_excludes_backtest_rows(store):
    """Synthetic rows must not aggregate into the pair graph."""
    for i in range(8):
        _insert(store, id=f"bcm{i}",
                url=f"backtest://run_77/ticker/ZZZZ/{i}",
                title=f"NVDA AMD synthetic pair {i}",
                source="backtest_run_77_winner", hours_ago=0.5)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-comentions")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["verdict"] == "NO_DATA"
    assert payload["top"] == []


def test_ticker_comentions_endpoint_respects_hours(store):
    """All articles 12h old — invisible at default ?hours=2, visible at 24h."""
    for i in range(6):
        _insert(store, id=f"hcm{i}",
                url=f"https://example.com/hcm{i}",
                title=f"NVDA AMD old story {i}",
                source="rss", hours_ago=12.0)
    app = create_app(store)
    client = app.test_client()
    r2 = client.get("/api/ticker-comentions?hours=2")
    assert r2.status_code == 200
    assert r2.get_json()["verdict"] == "NO_DATA"

    r24 = client.get("/api/ticker-comentions?hours=24")
    assert r24.status_code == 200
    p24 = r24.get_json()
    assert p24["window_hours"] == 24
    # 6 co-mentions in window → SECTOR_BURST.
    assert p24["verdict"] == "SECTOR_BURST"


def test_ticker_comentions_endpoint_clamps_hours(store):
    """?hours outside 1..24 must clamp."""
    app = create_app(store)
    client = app.test_client()
    r1 = client.get("/api/ticker-comentions?hours=0")
    assert r1.status_code == 200
    assert r1.get_json()["window_hours"] == 1
    r2 = client.get("/api/ticker-comentions?hours=999")
    assert r2.status_code == 200
    assert r2.get_json()["window_hours"] == 24
