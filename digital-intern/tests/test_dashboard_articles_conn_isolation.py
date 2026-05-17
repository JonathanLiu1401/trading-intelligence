"""`/api/articles` must not read through the daemon's shared writer connection.

Observed in production (10×): `Exception on /api/articles [GET]` →
`IndexError: tuple index out of range` at `_articles_from_db`'s `r[6]`.

Root cause: `dashboard/web_server.py` runs `app.run(threaded=True)`, but
`_articles_from_db` queried `store.conn` — the *single* `sqlite3.Connection`
the daemon's ~30 writer threads share (`check_same_thread=False`). sqlite3
connections are not safe for concurrent use; a dashboard read racing a
writer's implicit `conn.execute("SELECT changes()")` (inside `insert_batch`)
yielded a wrong-shaped 1-tuple where the 9-column row was expected, so
`r[6]` blew up. `IndexError` is not a `sqlite3.Error`, so the endpoint's
`except sqlite3.Error: return []` did not catch it and the request 500'd.

The fix reads via a dedicated short-lived `mode=ro` connection (lock-free
WAL reads, fully isolated from the writer connection's cursor state). This
test pins that by poisoning `store.conn` with the exact interleave shape
(a cursor whose `.fetchall()` returns a 1-tuple) and asserting the endpoint
still returns the real rows from the isolated connection — and still excludes
synthetic backtest rows (the existing `_LIVE_ONLY` invariant on this path).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard import web_server


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert(store, *, id, url, title, source, ai_score=0.0, kw_score=2.0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, 0,
             _recent_iso(), 0, None, None),
        )
        store.conn.commit()


class _PoisonedConn:
    """Stands in for the shared writer connection mid-race: any execute()
    returns a cursor whose fetchall() is a 1-tuple — the `SELECT changes()`
    interleave that produced the production IndexError."""

    class _Cur:
        def fetchall(self):
            return [(0,)]

        def fetchone(self):
            return (0,)

    def execute(self, *a, **k):
        return self._Cur()


def test_api_articles_survives_poisoned_shared_connection(store, monkeypatch):
    # Two live rows + one synthetic backtest row in the real (tmp) DB.
    _insert(store, id="live1", url="https://ex.com/a",
            title="MU earnings beat", source="rss", ai_score=8.0)
    _insert(store, id="live2", url="https://ex.com/b",
            title="NVDA guidance raise", source="reuters", ai_score=7.0)
    _insert(store, id="bt1", url="backtest://run_9/2026-01-01/BUY/MU",
            title="SHOULD NEVER SURFACE", source="backtest_run_9_winner",
            ai_score=5.0)

    # The endpoint must ignore the shared writer connection entirely.
    store.conn = _PoisonedConn()
    monkeypatch.setattr(web_server, "_store", store, raising=False)

    app = web_server.create_app(store)
    client = app.test_client()
    resp = client.get("/api/articles?limit=50")

    assert resp.status_code == 200, resp.data
    rows = resp.get_json()
    ids = sorted(r["id"] for r in rows)
    assert ids == ["live1", "live2"], rows          # real rows, not the 1-tuple
    titles = {r["title"] for r in rows}
    assert "SHOULD NEVER SURFACE" not in titles      # backtest isolation intact
    # Effective score still derived correctly from the real columns.
    by_id = {r["id"]: r for r in rows}
    assert by_id["live1"]["score"] == 8.0
    assert by_id["live1"]["ai_score"] == 8.0
