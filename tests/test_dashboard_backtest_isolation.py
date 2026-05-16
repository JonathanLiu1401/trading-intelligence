"""Backtest isolation on the *live-facing dashboard surfaces*.

`dashboard/web_server.py` (the daemon's in-thread Flask on :8080) has always
filtered synthetic backtest/opus rows out of its feed. The standalone
`dashboard/server.py` (run by `dashboard.service` via uvicorn on :8765) and
`ml/sentiment_trends.py` (the dashboard's per-ticker panel) are two parallel
implementations of the same surface that did NOT — so backtest:// URLs and
`backtest_*` source rows were rendered to the user as real news, and
synthetic rows inflated the per-ticker sentiment aggregates.

These pin the canonical `_LIVE_ONLY_CLAUSE` invariant on those two paths so a
future divergence between the two dashboards re-fails here instead of silently
leaking training data into the live UI.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                kw_score=1.0, first_seen=None):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, None, None),
        )
        store.conn.commit()


def _seed_live_and_synthetic(store):
    """One live row + the three synthetic shapes, all with ai_score high enough
    to outrank the live row if the filter were missing."""
    _insert_raw(store, id="live1", url="https://reuters.com/x",
                title="Real Reuters market headline here", source="rss",
                ai_score=8.0)
    _insert_raw(store, id="bt_url", url="backtest://run_1/2026-01-01/BUY/MU",
                title="Synthetic backtest BUY MU entry", source="backtest_run_1",
                ai_score=9.9)
    _insert_raw(store, id="bt_src", url="https://example.com/y",
                title="Backtest winner row title here",
                source="backtest_run_42_winner", ai_score=9.8)
    _insert_raw(store, id="opus", url="https://example.com/z",
                title="Opus annotation lesson row here",
                source="opus_annotation_cycle_3", ai_score=9.7)


# ── dashboard/server.py — the standalone uvicorn dashboard (:8765) ───────────
class TestStandaloneDashboardFeed:
    def test_articles_payload_excludes_synthetic(self, store, monkeypatch):
        """_articles_payload feeds the live article list. A backtest:// URL or
        backtest_/opus_annotation source row appearing here is the user being
        shown training data as breaking news."""
        from dashboard import server

        _seed_live_and_synthetic(store)
        monkeypatch.setattr(server, "_store", store)

        rows = server._articles_payload(limit=50, min_score=0.0)
        ids = {r["id"] for r in rows}

        assert ids == {"live1"}, f"synthetic rows leaked into feed: {ids - {'live1'}}"
        for r in rows:
            assert not r["url"].startswith("backtest://")
            assert not r["source"].startswith("backtest_")
            assert not r["source"].startswith("opus_annotation")

    def test_articles_per_hour_excludes_synthetic(self, store, monkeypatch):
        """The 24h activity histogram must count only live rows — otherwise a
        backtest injection burst shows up as a fake spike in collection rate."""
        from dashboard import server

        _seed_live_and_synthetic(store)
        monkeypatch.setattr(server, "_store", store)

        buckets = server._articles_per_hour_24h()
        total = sum(b["count"] for b in buckets)
        assert total == 1, f"synthetic rows inflated the histogram: total={total}"


# ── ml/sentiment_trends.py — the dashboard per-ticker panel ──────────────────
class TestSentimentTrendsIsolation:
    def test_compute_trends_excludes_synthetic(self, store, monkeypatch):
        """A backtest row whose title mentions a tracked ticker must not inflate
        that ticker's count / avg_score on the dashboard sentiment panel."""
        from ml import sentiment_trends

        monkeypatch.setattr(
            sentiment_trends, "_load_tracked_tickers", lambda: ["MU"]
        )

        _insert_raw(store, id="live_mu", url="https://reuters.com/mu",
                    title="MU earnings beat lifts the stock", source="rss",
                    ai_score=8.0)
        # Synthetic rows whose titles also mention MU — must be ignored.
        _insert_raw(store, id="bt_mu", url="backtest://run_1/d/BUY/MU",
                    title="MU backtest BUY winner synthetic row",
                    source="backtest_run_1_winner", ai_score=5.0)
        _insert_raw(store, id="opus_mu", url="https://example.com/opus-mu",
                    title="MU opus annotation GOOD trade lesson",
                    source="opus_annotation_cycle_2", ai_score=2.5)

        data = sentiment_trends.compute_trends(store)
        mu = data["tickers"]["MU"]

        assert mu["count"] == 1, (
            f"synthetic rows counted toward MU: count={mu['count']}"
        )
        # Only the live ai_score=8.0 row should drive the average/max.
        assert mu["avg_score"] == pytest.approx(8.0)
        assert mu["max_score"] == pytest.approx(8.0)
