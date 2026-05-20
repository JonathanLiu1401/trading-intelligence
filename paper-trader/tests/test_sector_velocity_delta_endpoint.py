"""Flask wiring tests for /api/sector-velocity-delta.

Stubs articles.db sqlite3 so the test is offline and deterministic.
Asserts the endpoint composes the live-only SQL filter + builder
correctly and surfaces production HEATMAP_BUCKETS structure.
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
        # memory_core: 3 high-score articles in fresh window — should
        # trigger ACCELERATING (n_tickers=3, floor=3.0; total fresh
        # well above floor with no prior).
        {"id": "u1", "url": "https://x/mu", "title": "$MU price target raised",
         "source": "CNBC", "ai_score": 8.0, "urgency": 1,
         "first_seen": (now - timedelta(hours=0.5)).isoformat()},
        {"id": "u2", "url": "https://x/wdc", "title": "$WDC supply update",
         "source": "Reuters", "ai_score": 7.5, "urgency": 1,
         "first_seen": (now - timedelta(hours=0.3)).isoformat()},
        {"id": "u3", "url": "https://x/stx", "title": "$STX storage demand",
         "source": "Bloomberg", "ai_score": 6.5, "urgency": 0,
         "first_seen": (now - timedelta(hours=0.4)).isoformat()},
        # Synthetic — must be filtered by SQL clause.
        {"id": "syn", "url": "backtest://run_5/x/y",
         "title": "$FAKE backtest winner",
         "source": "backtest_run_5_winner", "ai_score": 10.0, "urgency": 2,
         "first_seen": (now - timedelta(hours=1)).isoformat()},
    ]
    db = _fake_articles_db(tmp_path, arts)
    monkeypatch.setattr(dashboard, "_articles_db_path", lambda: str(db))
    return db


class TestSectorVelocityDeltaEndpoint:
    def test_surfaces_buckets_and_filters_synthetic(self, client, wire_db):
        res = client.get("/api/sector-velocity-delta")
        assert res.status_code == 200
        j = res.get_json()
        assert j["state"] == "OK"
        names = {b["name"] for b in j["buckets"]}
        # Production HEATMAP_BUCKETS includes memory_core / design /
        # etc. — at minimum memory_core must be present (the fixture
        # tickers all live there).
        assert "memory_core" in names
        mc = next(b for b in j["buckets"] if b["name"] == "memory_core")
        # Fresh score = (8 + 7.5 + 6.5) decayed at ~0.3-0.5h ≈ 21+
        # (decay barely takes effect). With n_tickers=3, floor=3.0,
        # no prior → ACCELERATING.
        assert mc["verdict"] == "ACCELERATING"
        assert mc["fresh_n"] == 3
        # Synthetic dropped — no FAKE ticker present anywhere.
        for b in j["buckets"]:
            assert b["top_fresh_ticker"] not in (None, "FAKE") or \
                b["fresh_score"] == 0.0
            # explicit: any bucket with weight must not be FAKE
            if b["top_fresh_ticker"] is not None:
                assert b["top_fresh_ticker"] != "FAKE"
        # rotating_in surfaces memory_core.
        assert "memory_core" in j["rotating_in"]

    def test_no_articles_db_degrades_to_no_data(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        j = client.get("/api/sector-velocity-delta").get_json()
        assert j["state"] == "NO_DATA"
        # All buckets present but DARK on no-data.
        assert all(b["verdict"] == "DARK" for b in j["buckets"])

    def test_query_param_clamps(self, client, wire_db):
        res = client.get(
            "/api/sector-velocity-delta?hours=9999&min_score=-50"
        )
        assert res.status_code == 200
        j = res.get_json()
        assert j["fresh_window_hours"] <= 72.0
