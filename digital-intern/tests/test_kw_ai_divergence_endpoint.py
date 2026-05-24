"""Verifies /api/kw-ai-divergence and /api/urgency-drought endpoints.

Both endpoints surface previously-dark analyzer modules to the dashboard.
Same shape as the recently-added /api/label-quality / /api/active-learning-
queue endpoints — compute on demand, never raise, always return 200.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _client(store_factory):
    """Flask test client backed by a fresh per-test store."""
    from dashboard import web_server
    store = store_factory()
    app = web_server.create_app(store=store)
    app.testing = True
    return app.test_client(), store


def _seed_articles(store, rows):
    """Insert raw rows into an ArticleStore — bypasses dedup/hashing."""
    now = datetime.now(timezone.utc)
    with store._write_lock:
        for i, r in enumerate(rows):
            fs = (now - timedelta(minutes=i)).isoformat()
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, kw_score, ai_score, urgency, "
                "first_seen, cycle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["id"], r.get("url", f"https://x.com/{r['id']}"),
                    r.get("title", f"title {r['id']}"), r["source"],
                    r["kw_score"], r["ai_score"],
                    r.get("urgency", 0), fs, 0,
                ),
            )
        store.conn.commit()


# ─────────────────────────────────────────────────────────────────────────
#  /api/kw-ai-divergence
# ─────────────────────────────────────────────────────────────────────────

class TestKwAiDivergenceEndpoint:
    def test_returns_payload_on_empty_db(self, monkeypatch, store_factory):
        """An empty DB must not 500 — return a 200 with zero counts. Same
        graceful-degrade discipline as /api/label-quality."""
        client, store = _client(store_factory)
        from analytics import kw_ai_divergence as kad
        # PRAGMA database_list returns (seq, name, file) per attached DB.
        db_file = store.conn.execute(
            "PRAGMA database_list"
        ).fetchone()[2]
        monkeypatch.setattr(kad, "DB_PATH", db_file)

        r = client.get("/api/kw-ai-divergence")
        assert r.status_code == 200
        body = r.get_json()
        assert body["scanned"] == 0
        assert body["false_positives"]["total"] == 0
        assert body["hidden_gems"]["total"] == 0
        assert "as_of" in body
        # Thresholds string must reflect the corrected 0..10 scale
        assert "ai<=1.5" in body["thresholds"]["false_positive"]
        assert "ai>=6.0" in body["thresholds"]["hidden_gem"]

    def test_classifies_seeded_rows(self, monkeypatch, store_factory):
        client, store = _client(store_factory)
        from analytics import kw_ai_divergence as kad
        db_file = store.conn.execute(
            "PRAGMA database_list"
        ).fetchone()[2]
        monkeypatch.setattr(kad, "DB_PATH", db_file)

        _seed_articles(store, [
            # clear false-positive (kw 10, ai 0)
            {"id": "fp", "source": "reddit",
             "kw_score": 10.0, "ai_score": 0.0,
             "title": "loud noise headline a"},
            # clear hidden-gem (ai 8, kw 0.5)
            {"id": "hg", "source": "GN: obscure",
             "kw_score": 0.5, "ai_score": 8.0,
             "title": "quiet but Sonnet liked it"},
            # mid both — neither
            {"id": "mid", "source": "rss",
             "kw_score": 4.0, "ai_score": 4.0,
             "title": "middling general story headline"},
        ])

        r = client.get("/api/kw-ai-divergence")
        assert r.status_code == 200
        body = r.get_json()
        assert body["scanned"] == 3
        assert body["false_positives"]["total"] == 1
        assert body["hidden_gems"]["total"] == 1
        assert body["false_positives"]["top_sources"][0]["source"] == "reddit"
        assert body["hidden_gems"]["top_sources"][0]["source"] == "GN: obscure"

    def test_error_absorbed_into_200(self, monkeypatch, store_factory):
        """If compute() raises, the endpoint must NOT 500 — it returns 200
        with an error key so a dashboard widget degrades gracefully instead
        of breaking the page render. Mirrors /api/ml-status discipline."""
        client, store = _client(store_factory)

        def _boom():
            raise RuntimeError("simulated analyzer failure")

        import analytics.kw_ai_divergence as kad
        monkeypatch.setattr(kad, "compute", _boom)

        r = client.get("/api/kw-ai-divergence")
        assert r.status_code == 200
        body = r.get_json()
        assert "error" in body
        assert "simulated analyzer failure" in body["error"]
        assert body["false_positives"] is None
        assert body["hidden_gems"] is None


# ─────────────────────────────────────────────────────────────────────────
#  /api/urgency-drought
# ─────────────────────────────────────────────────────────────────────────

class TestUrgencyDroughtEndpoint:
    def test_returns_ok_when_recent_urgent_present(
            self, monkeypatch, tmp_path, store_factory):
        client, store = _client(store_factory)
        from analytics import urgency_drought as ud

        db_file = store.conn.execute(
            "PRAGMA database_list"
        ).fetchone()[2]
        monkeypatch.setattr(ud, "_get_db_path", lambda: db_file)
        monkeypatch.setattr(ud, "OUT", tmp_path / "urgency_drought.json")

        _seed_articles(store, [
            {"id": "u2", "source": "rss",
             "kw_score": 5.0, "ai_score": 9.0, "urgency": 2,
             "title": "fresh urgent pushed headline"},
            {"id": "u1", "source": "rss",
             "kw_score": 5.0, "ai_score": 9.0, "urgency": 1,
             "title": "fresh urgent queued headline"},
        ])

        r = client.get("/api/urgency-drought")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "ok"
        assert body["urgency_2"]["status"] == "ok"
        assert body["urgency_1"]["status"] == "ok"
        assert "as_of" in body

    def test_returns_alert_when_long_drought(
            self, monkeypatch, tmp_path, store_factory):
        client, store = _client(store_factory)
        from analytics import urgency_drought as ud

        db_file = store.conn.execute(
            "PRAGMA database_list"
        ).fetchone()[2]
        monkeypatch.setattr(ud, "_get_db_path", lambda: db_file)
        monkeypatch.setattr(ud, "OUT", tmp_path / "urgency_drought.json")

        # urgent rows from 24h ago — well above ALERT_HOURS=12
        old = (
            datetime.now(timezone.utc) - timedelta(hours=24)
        ).isoformat()
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id,url,title,source,kw_score,ai_score,urgency,first_seen,cycle) "
                "VALUES ('old','https://x/old','very old urgent','rss',5.0,9.0,2,?,0)",
                (old,))
            store.conn.execute(
                "INSERT INTO articles "
                "(id,url,title,source,kw_score,ai_score,urgency,first_seen,cycle) "
                "VALUES ('old2','https://x/old2','very old queued','rss',5.0,9.0,1,?,0)",
                (old,))
            store.conn.commit()

        r = client.get("/api/urgency-drought")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "alert"
        assert body["urgency_2"]["status"] == "alert"
        assert body["urgency_2"]["hours_ago"] >= ud.ALERT_HOURS

    def test_error_absorbed_into_200(self, monkeypatch, store_factory):
        client, store = _client(store_factory)

        def _boom():
            raise RuntimeError("drought compute crashed")

        import analytics.urgency_drought as ud
        monkeypatch.setattr(ud, "compute", _boom)

        r = client.get("/api/urgency-drought")
        assert r.status_code == 200
        body = r.get_json()
        assert "error" in body
        assert body["status"] == "unknown"
