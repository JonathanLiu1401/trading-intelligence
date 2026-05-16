"""Cross-system invariant: paper-trader's live signal reads must exclude
synthetic backtest/opus rows.

`paper_trader/signals.py` is a vendored snapshot of the authoritative file in
`/home/zeph/paper-trader/`. Its live-read queries (`get_top_signals`,
`get_urgent_articles`, `get_ticker_sentiment`, `ticker_sentiments`) read the
shared `articles.db`. AGENTS.md's "Cross-system contract with paper-trader"
mandates every such read inline the `_LIVE_ONLY_CLAUSE` SQL fragment —
otherwise backtest training rows (high ai_score=5.0 BUY winners, urgency=1)
leak straight into the live trader's prompt context as if they were breaking
news.

The vendored copy had drifted out of sync with the authoritative source (which
already carried the filter); these tests pin the invariant so it can't regress
again.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from paper_trader import signals
from storage.article_store import SCHEMA


def _build_db(path):
    """Create a real articles.db with one live row and three synthetic rows,
    all recent + urgent + high-scoring, then close it so signals.py can open a
    read-only connection to it."""
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        # id, url, title, source, published, kw, ai, urg, first_seen, cyc, ml, src
        ("live", "https://reuters.com/x", "MU earnings beat live wire",
         "rss", "", 1.0, 9.0, 1, now, 0, None, "llm"),
        ("bt_url", "backtest://run_1/2026-01-01/BUY/MU",
         "MU synthetic backtest winner", "backtest_run_1", "", 1.0, 5.0, 1,
         now, 0, None, None),
        ("bt_src", "https://example.com/syn",
         "MU backtest source tagged row", "backtest_run_42_winner", "", 1.0,
         5.0, 1, now, 0, None, None),
        ("opus", "https://example.com/op",
         "MU opus annotation lesson row", "opus_annotation_cycle_3", "", 1.0,
         5.0, 1, now, 0, None, None),
    ]
    conn.executemany(
        "INSERT INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        "first_seen, cycle, ml_score, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def signals_db(tmp_path, monkeypatch):
    db = tmp_path / "articles.db"
    _build_db(db)
    monkeypatch.setattr(signals, "_db_path", lambda: db)
    return db


def test_get_top_signals_excludes_synthetic(signals_db):
    out = signals.get_top_signals(n=50, hours=24, min_score=0.0)
    ids = {r["id"] for r in out}
    assert ids == {"live"}, f"synthetic rows leaked into top signals: {ids}"
    for r in out:
        assert not r["url"].startswith("backtest://")
        assert not r["source"].startswith("backtest_")
        assert not r["source"].startswith("opus_annotation")


def test_get_urgent_articles_excludes_synthetic(signals_db):
    out = signals.get_urgent_articles(minutes=60)
    ids = {r["id"] for r in out}
    assert ids == {"live"}, f"synthetic rows leaked into urgent feed: {ids}"
    for r in out:
        assert not r["source"].startswith("backtest_")
        assert not r["source"].startswith("opus_annotation")


def test_get_ticker_sentiment_ignores_synthetic(signals_db):
    """Only the single live MU row must count — three synthetic MU rows would
    triple `n` and skew `avg_score` toward the injected backtest label."""
    s = signals.get_ticker_sentiment("MU", hours=24)
    assert s["n"] == 1, f"synthetic rows inflated ticker count: {s}"
    assert s["avg_score"] == pytest.approx(9.0)


def test_ticker_sentiments_bulk_ignores_synthetic(signals_db):
    res = signals.ticker_sentiments(["MU"], hours=24)
    mu = next(r for r in res if r["ticker"] == "MU")
    assert mu["n"] == 1, f"synthetic rows inflated bulk ticker count: {mu}"
    assert mu["avg_score"] == pytest.approx(9.0)
