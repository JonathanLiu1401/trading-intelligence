"""Flask wiring tests for /api/news-themes.

Stubs the articles-DB sqlite3.connect and get_store so the test is
offline and deterministic. Asserts the endpoint composes the live-only
SQL filter + builder correctly and returns sensible JSON.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


@pytest.fixture
def client():
    return dashboard.app.test_client()


class _FakeStore:
    def __init__(self, positions=None):
        self._p = positions or []

    def open_positions(self):
        return list(self._p)


def _fake_articles_db(tmp_path, articles):
    """Build a tiny on-disk articles.db with the schema fields the
    endpoint SELECTs. Backtest-row leakage is intentionally exercised
    via the live-only SQL filter."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE articles ("
        "id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, "
        "ai_score REAL, urgency INTEGER, first_seen TEXT, "
        "kw_score REAL, full_text BLOB, time_sensitivity REAL, cycle INTEGER)"
    )
    for art in articles:
        conn.execute(
            "INSERT INTO articles (id, url, title, source, ai_score, "
            "urgency, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (art["id"], art["url"], art["title"], art["source"],
             art["ai_score"], art["urgency"], art["first_seen"]),
        )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def live_db(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    arts = [
        {"id": "a1", "url": "https://x/nvda", "title": "$NVDA earnings beat",
         "source": "Reuters", "ai_score": 9.0, "urgency": 2,
         "first_seen": (now - timedelta(hours=1)).isoformat()},
        {"id": "a2", "url": "https://x/amd", "title": "AMD ships chip",
         "source": "CNBC", "ai_score": 8.0, "urgency": 1,
         "first_seen": (now - timedelta(hours=2)).isoformat()},
        # Synthetic — must be filtered by SQL clause.
        {"id": "a3", "url": "backtest://run_5/x/y/z",
         "title": "$FAKE backtest winner",
         "source": "backtest_run_5_winner", "ai_score": 10.0, "urgency": 2,
         "first_seen": (now - timedelta(hours=1)).isoformat()},
    ]
    db = _fake_articles_db(tmp_path, arts)
    monkeypatch.setattr(dashboard, "_articles_db_path", lambda: str(db))
    return db


class TestNewsThemesEndpoint:
    def test_returns_themes_excluding_synthetic(self, client, monkeypatch, live_db):
        monkeypatch.setattr(
            dashboard, "get_store",
            lambda: _FakeStore(positions=[
                {"ticker": "NVDA", "type": "stock", "qty": 2,
                 "avg_cost": 222.0, "current_price": 220.0}
            ]),
        )
        res = client.get("/api/news-themes")
        assert res.status_code == 200
        j = res.get_json()
        assert j["state"] == "OK"
        tickers = {t["ticker"] for t in j["themes"]}
        # NVDA and AMD surface; FAKE (synthetic) does not.
        assert "NVDA" in tickers
        assert "AMD" in tickers
        assert "FAKE" not in tickers
        # NVDA shows held=True.
        by = {t["ticker"]: t for t in j["themes"]}
        assert by["NVDA"]["held"] is True
        assert by["AMD"]["held"] is False
        # top_unheld is the unheld theme with the highest decayed score.
        assert j["top_unheld_ticker"] == "AMD"

    def test_no_articles_db_degrades_to_no_data(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        j = client.get("/api/news-themes").get_json()
        assert j["state"] == "NO_DATA"

    def test_query_param_clamps(self, client, monkeypatch, live_db):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        # Way-out-of-range params should be clamped (no 500).
        res = client.get("/api/news-themes?hours=9999&max_themes=99999&min_score=-50")
        assert res.status_code == 200
        j = res.get_json()
        assert j["window_hours"] <= 168.0
        assert j["max_themes"] <= 100

    def test_store_failure_does_not_block_endpoint(self, client, monkeypatch, live_db):
        def _boom():
            raise RuntimeError("store unavailable")
        monkeypatch.setattr(dashboard, "get_store", _boom)
        res = client.get("/api/news-themes")
        # Held tickers degrade to [] but the endpoint still returns OK.
        assert res.status_code == 200
        j = res.get_json()
        assert j["state"] == "OK"
        assert all(t["held"] is False for t in j["themes"])
