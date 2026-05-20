"""Flask wiring tests for /api/rising-unheld-themes.

Stubs articles.db sqlite3 + get_store so the test is offline and
deterministic. Asserts the endpoint composes the live-only SQL filter
+ builder correctly, surfaces unheld tickers only, and returns the
expected JSON shape.
"""
from __future__ import annotations

import sqlite3
import sys
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
def wire_db(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    arts = [
        # Held name with loud fresh coverage — must NOT surface in the
        # unheld output (held-theme-decay's territory).
        {"id": "h1", "url": "https://x/nvda", "title": "$NVDA earnings tonight",
         "source": "Reuters", "ai_score": 9.5, "urgency": 2,
         "first_seen": (now - timedelta(hours=0.5)).isoformat()},
        # Unheld name with BREAKING coverage (no prior, loud fresh).
        {"id": "u1", "url": "https://x/mu", "title": "$MU price target raised",
         "source": "CNBC", "ai_score": 8.0, "urgency": 1,
         "first_seen": (now - timedelta(hours=0.5)).isoformat()},
        # Unheld name with BUILDING coverage (prior + accelerating fresh).
        # Prior ai_score >= 2.0 so the endpoint's default min_score
        # SQL filter doesn't drop it (would collapse to BREAKING).
        {"id": "u2p", "url": "https://x/amd1", "title": "$AMD analyst note",
         "source": "Reuters", "ai_score": 2.5, "urgency": 0,
         "first_seen": (now - timedelta(hours=6.1)).isoformat()},
        {"id": "u2f", "url": "https://x/amd2", "title": "$AMD ships next-gen",
         "source": "Reuters", "ai_score": 6.0, "urgency": 1,
         "first_seen": (now - timedelta(hours=0.5)).isoformat()},
        # Synthetic — must be filtered by SQL clause.
        {"id": "syn", "url": "backtest://run_5/x/y",
         "title": "$FAKE backtest winner",
         "source": "backtest_run_5_winner", "ai_score": 10.0, "urgency": 2,
         "first_seen": (now - timedelta(hours=1)).isoformat()},
    ]
    db = _fake_articles_db(tmp_path, arts)
    monkeypatch.setattr(dashboard, "_articles_db_path", lambda: str(db))
    return db


class TestRisingUnheldThemesEndpoint:
    def test_surfaces_unheld_excludes_held_and_synthetic(
        self, client, monkeypatch, wire_db
    ):
        monkeypatch.setattr(
            dashboard, "get_store",
            lambda: _FakeStore(positions=[
                {"ticker": "NVDA", "type": "stock", "qty": 2,
                 "avg_cost": 222.0, "current_price": 220.0}
            ]),
        )
        res = client.get("/api/rising-unheld-themes")
        assert res.status_code == 200
        j = res.get_json()
        assert j["state"] == "OK"
        tickers = {t["ticker"] for t in j["themes"]}
        # NVDA held → excluded; MU/AMD unheld → present; FAKE synthetic
        # → dropped at SQL filter.
        assert "NVDA" not in tickers
        assert "MU" in tickers
        assert "AMD" in tickers
        assert "FAKE" not in tickers
        # MU is BREAKING (no prior + loud fresh); AMD is BUILDING.
        by = {t["ticker"]: t for t in j["themes"]}
        assert by["MU"]["verdict"] == "BREAKING"
        assert by["AMD"]["verdict"] == "BUILDING"
        # Sort puts BREAKING first.
        assert j["themes"][0]["ticker"] == "MU"
        assert j["top_rising"]["ticker"] == "MU"

    def test_no_articles_db_degrades_to_no_data(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        j = client.get("/api/rising-unheld-themes").get_json()
        assert j["state"] == "NO_DATA"
        assert j["themes"] == []

    def test_query_param_clamps(self, client, monkeypatch, wire_db):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        # Way-out-of-range params should be clamped (no 500).
        res = client.get(
            "/api/rising-unheld-themes"
            "?hours=9999&max_themes=99999&min_score=-50"
        )
        assert res.status_code == 200
        j = res.get_json()
        assert j["fresh_window_hours"] <= 72.0
        assert j["max_themes"] <= 100

    def test_store_failure_does_not_block_endpoint(
        self, client, monkeypatch, wire_db
    ):
        def _boom():
            raise RuntimeError("store unavailable")
        monkeypatch.setattr(dashboard, "get_store", _boom)
        res = client.get("/api/rising-unheld-themes")
        # held tickers degrade to [] but endpoint still returns OK.
        # With held=[], NVDA's loud coverage now surfaces as an unheld
        # theme too. Verifies the endpoint stays alive on store failure.
        assert res.status_code == 200
        j = res.get_json()
        assert j["state"] == "OK"
