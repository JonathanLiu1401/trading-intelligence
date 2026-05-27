"""End-to-end test_client smokes for /api/ticker-news-burst.

Thin — the pure builder is exhaustively tested in
``test_ticker_news_burst_runner.py``. This file pins ONLY the things the pure
tests cannot: route registration, ``_LIVE_ONLY_SQL`` actually applied,
``?window_h`` / ``?baseline_h`` / ``?tickers`` query-string handling, and
that the route returns the builder's shape verbatim.

Per project memory: verify via the Flask test client, not a module
``__main__`` smoke that may hit a different/empty DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.web_server import create_app


def _ts(hours_ago: float, ref: datetime | None = None) -> str:
    ref = ref or datetime.now(timezone.utc)
    return (ref - timedelta(hours=hours_ago)).isoformat()


def _insert(store, *, id, url, title, source, hours_ago):
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


def _seed_blazing_nvda(store):
    """5 fresh NVDA mentions in the last hour, 1 in the prior 24h →
    spike = 5 / max(1/24, 0.5) = 5 / 0.5 = 10 → BLAZING."""
    for i in range(5):
        _insert(store, id=f"nvda{i}",
                url=f"https://example.com/nvda{i}",
                title=f"NVDA breaking story {i}",
                source="rss", hours_ago=0.3)
    _insert(store, id="nvdabase",
            url="https://example.com/nvda-base",
            title="NVDA earnings preview",
            source="rss", hours_ago=12.0)


def test_endpoint_returns_builder_shape(store):
    _seed_blazing_nvda(store)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-news-burst?tickers=NVDA")
    assert r.status_code == 200
    payload = r.get_json()
    for key in (
        "generated_at", "window_h", "baseline_h", "n_window", "n_baseline",
        "by_ticker", "hottest", "n_hot", "verdict", "headline",
    ):
        assert key in payload, key
    assert payload["verdict"] == "BLAZING"
    assert payload["hottest"] == "NVDA"
    nvda = next(r for r in payload["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["verdict"] == "BLAZING"
    assert nvda["count_window"] == 5
    assert nvda["count_baseline"] == 1


def test_endpoint_excludes_backtest_rows(store):
    """``backtest://`` URLs / ``backtest_*`` / ``opus_annotation*`` sources
    MUST NOT count — same invariant as every other live wire surface."""
    for i in range(8):
        _insert(store, id=f"bt{i}",
                url=f"backtest://run_1/2026-05-26/BUY/NVDA-{i}",
                title=f"NVDA synthetic burst {i}",
                source="backtest_run_1_winner", hours_ago=0.2)
    for i in range(4):
        _insert(store, id=f"op{i}",
                url=f"https://example.com/op-{i}",
                title=f"AMD synthetic annotation {i}",
                source="opus_annotation_cycle_42", hours_ago=0.2)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-news-burst?tickers=NVDA,AMD")
    assert r.status_code == 200
    payload = r.get_json()
    nvda = next(r for r in payload["by_ticker"] if r["ticker"] == "NVDA")
    amd = next(r for r in payload["by_ticker"] if r["ticker"] == "AMD")
    assert nvda["count_window"] == 0
    assert amd["count_window"] == 0
    assert nvda["verdict"] == "COLD"
    assert amd["verdict"] == "COLD"


def test_endpoint_respects_window_h_param(store):
    """A 0.5h window vs the default 1.0h must reshape the spike."""
    # 5 mentions at 0.7h ago — visible in window_h=1.0, NOT in window_h=0.5
    for i in range(5):
        _insert(store, id=f"w{i}", url=f"https://example.com/w{i}",
                title=f"NVDA at edge {i}", source="rss", hours_ago=0.7)
    app = create_app(store)
    client = app.test_client()
    r1 = client.get("/api/ticker-news-burst?tickers=NVDA&window_h=1.0")
    payload1 = r1.get_json()
    nvda1 = next(r for r in payload1["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda1["count_window"] == 5

    r2 = client.get("/api/ticker-news-burst?tickers=NVDA&window_h=0.5")
    payload2 = r2.get_json()
    nvda2 = next(r for r in payload2["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda2["count_window"] == 0


def test_endpoint_clamps_window_h(store):
    """?window_h outside 0.25..12.0 must clamp."""
    app = create_app(store)
    client = app.test_client()
    r1 = client.get("/api/ticker-news-burst?tickers=NVDA&window_h=0.01")
    assert r1.get_json()["window_h"] == 0.25
    r2 = client.get("/api/ticker-news-burst?tickers=NVDA&window_h=999")
    assert r2.get_json()["window_h"] == 12.0


def test_endpoint_clamps_baseline_h_to_window_floor(store):
    """baseline_h must always be ≥ 1.5 × window_h."""
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-news-burst?tickers=NVDA&window_h=4.0&baseline_h=2.0")
    payload = r.get_json()
    # baseline_h floored to 1.5 * window_h = 6.0
    assert payload["baseline_h"] >= 6.0


def test_endpoint_default_tickers_uses_live_portfolio(store, monkeypatch):
    """When ?tickers omitted, endpoint falls back to LIVE_PORTFOLIO_TICKERS
    so the analyst gets the held universe by default."""
    import ml.features as features
    monkeypatch.setattr(features, "LIVE_PORTFOLIO_TICKERS", {"FOO", "BAR"})
    _insert(store, id="x", url="https://example.com/x",
            title="FOO announces deal", source="rss", hours_ago=0.2)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-news-burst")
    assert r.status_code == 200
    payload = r.get_json()
    tickers_seen = {r["ticker"] for r in payload["by_ticker"]}
    assert tickers_seen == {"FOO", "BAR"}


def test_endpoint_with_explicit_csv_tickers(store):
    """?tickers=MU,NVDA must take precedence over the default."""
    _insert(store, id="mu1", url="https://example.com/mu1",
            title="MU news", source="rss", hours_ago=0.2)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-news-burst?tickers=MU,NVDA")
    payload = r.get_json()
    assert {r["ticker"] for r in payload["by_ticker"]} == {"MU", "NVDA"}
