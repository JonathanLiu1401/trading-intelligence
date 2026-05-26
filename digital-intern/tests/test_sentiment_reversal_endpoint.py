"""End-to-end test_client smokes for /api/sentiment-reversal and
/api/ticker-score-dispersion.

These are intentionally thin — the pure builders are deeply tested in
tests/test_sentiment_reversal.py and tests/test_ticker_score_dispersion.py.
This file pins ONLY the things the pure tests cannot: the Flask route is
registered, the _LIVE_ONLY_SQL filter is actually applied in the query
(backtest:// rows must not leak through), and the route returns the
builder's shape verbatim.

Per project-memory: verify via the Flask test client, not a module __main__
smoke that hits a different/empty DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard.web_server import create_app


def _ts(hours_ago: float, ref: datetime | None = None) -> str:
    ref = ref or datetime.now(timezone.utc)
    return (ref - timedelta(hours=hours_ago)).isoformat()


def _insert(store, *, id, url, title, source, ml_score, hours_ago):
    """Insert a single article row directly (we don't need full pipeline)."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            " urgency, first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, None, 1.0, ml_score, 0,
             _ts(hours_ago), 0, ml_score, "test", None),
        )
        store.conn.commit()


def _seed_reversal(store):
    """A clear neg→pos reversal on NVDA: 4 prev-window neg + 4 curr-window pos."""
    for i in range(4):
        _insert(store, id=f"prev{i}",
                url=f"https://example.com/prev{i}",
                title="NVDA crash extends as guide cut",
                source="rss", ml_score=-3.0, hours_ago=3.0)
    for i in range(4):
        _insert(store, id=f"curr{i}",
                url=f"https://example.com/curr{i}",
                title="NVDA rally on HBM breakthrough",
                source="rss", ml_score=+3.0, hours_ago=0.5)


def test_sentiment_reversal_endpoint_returns_builder_shape(store):
    """Endpoint must return the builder's full payload shape and detect
    the same neg→pos NVDA flip the pure builder identifies."""
    _seed_reversal(store)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/sentiment-reversal")
    assert r.status_code == 200
    payload = r.get_json()
    # Shape contract — the chat helper depends on every one of these keys.
    for key in ("generated_at", "window_hours", "rows_scanned",
                "min_articles_per_window", "min_delta",
                "reversals_found", "reversals"):
        assert key in payload, key
    assert payload["reversals_found"] == 1
    rev = payload["reversals"][0]
    assert rev["ticker"] == "NVDA"
    assert rev["direction"] == "neg→pos"
    assert rev["articles_prev"] == 4
    assert rev["articles_curr"] == 4


def test_sentiment_reversal_endpoint_excludes_backtest_rows(store):
    """``backtest://`` URLs and ``backtest_*`` / ``opus_annotation*`` sources
    must be filtered out — otherwise the reversal endpoint would surface
    training-injected synthetic rows as live news flips. Cross-system
    invariant; mirrors paper-trader signals.py."""
    # 4 backtest-injected PREV-window rows (must be filtered out)
    for i in range(4):
        _insert(store, id=f"btprev{i}",
                url=f"backtest://run_99/ticker/AAAA/{i}",
                title="AAAA fake crash",
                source="backtest_run_99_loser", ml_score=-5.0, hours_ago=3.0)
    # 4 backtest-injected CURR-window rows (also filtered)
    for i in range(4):
        _insert(store, id=f"btcurr{i}",
                url=f"backtest://run_99/ticker/AAAA/{i+100}",
                title="AAAA fake surge",
                source="backtest_run_99_winner", ml_score=+5.0, hours_ago=0.5)
    # 4 opus_annotation rows on BBBB (also filtered)
    for i in range(4):
        _insert(store, id=f"opprev{i}",
                url=f"https://example.com/op-prev{i}",
                title="BBBB crash story",
                source="opus_annotation_cycle_42",
                ml_score=-4.0, hours_ago=3.0)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/sentiment-reversal")
    assert r.status_code == 200
    payload = r.get_json()
    # No reversals should be found — every row is synthetic and should be
    # filtered out by _LIVE_ONLY_SQL.
    assert payload["reversals_found"] == 0
    assert payload["reversals"] == []


def test_ticker_score_dispersion_endpoint_returns_builder_shape(store):
    """Endpoint must return the builder's full payload shape and detect
    CONFLICTED on a wide-spread ticker."""
    for i, s in enumerate([9.0, 1.0, 8.5, 1.5, 5.0]):
        _insert(store, id=f"d{i}",
                url=f"https://example.com/d{i}",
                title=f"AAAA contested {i}",
                source="rss", ml_score=s, hours_ago=1.0 + i * 0.1)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-score-dispersion")
    assert r.status_code == 200
    payload = r.get_json()
    for key in ("generated_at", "window_hours", "rows_scanned",
                "rows_in_window", "min_articles_per_ticker",
                "tight_std_threshold", "conflicted_std_threshold",
                "verdict", "n_tickers_qualified", "n_tight", "n_mixed",
                "n_conflicted", "tickers"):
        assert key in payload, key
    assert payload["verdict"] == "CONFLICTED_NEWS"
    assert payload["n_conflicted"] == 1
    t = payload["tickers"][0]
    assert t["ticker"] == "AAAA"
    assert t["verdict"] == "CONFLICTED"


def test_ticker_score_dispersion_endpoint_excludes_backtest_rows(store):
    """Same live-only invariant — synthetic rows must not be aggregated."""
    for i, s in enumerate([9.0, 1.0, 8.5, 1.5, 5.0]):
        _insert(store, id=f"bd{i}",
                url=f"backtest://run_77/ticker/ZZZZ/{i}",
                title=f"ZZZZ synthetic {i}",
                source="backtest_run_77_winner", ml_score=s,
                hours_ago=1.0 + i * 0.1)
    app = create_app(store)
    client = app.test_client()
    r = client.get("/api/ticker-score-dispersion")
    assert r.status_code == 200
    payload = r.get_json()
    # Empty / no-data because every row is synthetic.
    assert payload["verdict"] == "NO_DATA"
    assert payload["tickers"] == []


def test_ticker_score_dispersion_endpoint_respects_hours_param(store):
    """Custom ``?hours=`` must be honoured (clamped 1..168)."""
    # All articles 50h ago — invisible to default 24h window, visible at 72h.
    for i in range(5):
        _insert(store, id=f"old{i}",
                url=f"https://example.com/old{i}",
                title=f"WDC slow news {i}",
                source="rss", ml_score=5.0, hours_ago=50.0)
    app = create_app(store)
    client = app.test_client()
    r24 = client.get("/api/ticker-score-dispersion?hours=24")
    assert r24.status_code == 200
    assert r24.get_json()["verdict"] == "NO_DATA"

    r72 = client.get("/api/ticker-score-dispersion?hours=72")
    assert r72.status_code == 200
    p72 = r72.get_json()
    assert p72["window_hours"] == 72
    # All 5 rows visible — WDC is TIGHT (zero variance), so CONSENSUS.
    assert p72["verdict"] == "CONSENSUS"
