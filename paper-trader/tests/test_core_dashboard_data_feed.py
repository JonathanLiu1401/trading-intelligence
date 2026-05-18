"""Regression lock: /api/data-feed must resolve digital-intern's articles.db
through the freshness-aware single source of truth (`dashboard._articles_db_path()`
→ `signals._db_path()`, AGENTS.md invariant #17) — NOT its own hardcoded
candidate list.

Before this fix `data_feed_api()` did:

    candidates = [Path("/home/zeph/digital-intern/data/articles.db"),     # LOCAL
                  Path("/media/zeph/projects/digital-intern/db/articles.db")]  # USB
    db_path = next((p for p in candidates if p.exists()), None)

Two real defects:
  1. It bypassed the split-brain-safe resolver — this panel could read a stale
     USB mirror while the live trader read fresh LOCAL (the exact failure
     invariant #17 closed for every *other* news endpoint).
  2. The "LOCAL" literal was the **pre-migration** path
     `/home/zeph/digital-intern/...` (the repo now lives under
     `/home/zeph/trading-intelligence/`). It only resolves on the original box
     via a legacy symlink; on a clean checkout the endpoint silently zeroed the
     live news-pulse panel with `error: articles.db not found`.

The discriminating test: with LOCAL fresh and USB stale **and a different
source mix in each**, the old code reads USB (stale source/counts); the fixed
code reads LOCAL (fresh source/counts).
"""
from __future__ import annotations

import sqlite3
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, signals


def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _build_articles_db(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT,
            source TEXT,
            ai_score REAL,
            urgency REAL,
            first_seen TEXT,
            full_text BLOB
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO articles (id, url, title, source, ai_score, urgency, "
            "first_seen, full_text) VALUES (?,?,?,?,?,?,?,?)",
            (
                r.get("id"),
                r.get("url"),
                r.get("title"),
                r.get("source"),
                r.get("ai_score"),
                r.get("urgency"),
                r.get("first_seen"),
                zlib.compress(r.get("body", "").encode("utf-8")) if r.get("body") else None,
            ),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def two_dbs(tmp_path, monkeypatch):
    """USB + LOCAL temp paths wired into the freshness-aware resolver."""
    usb = tmp_path / "usb" / "articles.db"
    local = tmp_path / "local" / "articles.db"
    usb.parent.mkdir()
    local.parent.mkdir()
    monkeypatch.setattr(signals, "USB_DB", usb)
    monkeypatch.setattr(signals, "LOCAL_DB", local)
    signals._reset_resolver_cache()
    yield usb, local
    signals._reset_resolver_cache()


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


def _get_json(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.data
    return r.get_json()


class TestDataFeedFreshnessResolved:
    def test_reads_fresh_local_not_stale_usb(self, two_dbs, client):
        """THE discriminating test. USB newest-live is 5h old (4 rows, source
        ``stale_src``); LOCAL newest-live is 0.5h old (2 rows, source
        ``fresh_src``). The freshness resolver picks LOCAL, so the endpoint
        must report LOCAL's numbers/source — the old hardcoded-USB path would
        report 0 in the 1h window and ``stale_src``."""
        usb, local = two_dbs
        _build_articles_db(usb, [
            {"id": i, "url": f"https://x/u{i}", "title": "t", "source": "stale_src",
             "ai_score": 5.0, "urgency": 0, "first_seen": _iso_ago(5)}
            for i in range(1, 5)
        ])
        _build_articles_db(local, [
            {"id": i, "url": f"https://x/l{i}", "title": "t", "source": "fresh_src",
             "ai_score": 7.0, "urgency": 0, "first_seen": _iso_ago(0.5)}
            for i in range(1, 3)
        ])
        # Sanity: the resolver itself picks LOCAL.
        assert dashboard._articles_db_path() == local

        j = _get_json(client, "/api/data-feed")
        assert j["articles_1h"] == 2
        assert j["articles_24h"] == 2
        assert j["top_sources"] == [{"name": "fresh_src", "count": 2}]
        assert "error" not in j

    def test_fresher_usb_still_wins(self, two_dbs, client):
        """Freshness-aware, not blindly LOCAL-first: a genuinely fresher USB
        is still chosen and its numbers reported."""
        usb, local = two_dbs
        _build_articles_db(usb, [
            {"id": 1, "url": "https://x/u1", "title": "t", "source": "usb_src",
             "ai_score": 5.0, "urgency": 0, "first_seen": _iso_ago(0.2)},
        ])
        _build_articles_db(local, [
            {"id": i, "url": f"https://x/l{i}", "title": "t", "source": "local_src",
             "ai_score": 7.0, "urgency": 0, "first_seen": _iso_ago(40)}
            for i in range(1, 4)
        ])
        assert dashboard._articles_db_path() == usb
        j = _get_json(client, "/api/data-feed")
        assert j["articles_1h"] == 1
        assert j["top_sources"] == [{"name": "usb_src", "count": 1}]

    def test_excludes_backtest_and_opus_rows(self, two_dbs, client):
        """Live-only filter (invariant #1/#3) — synthetic rows must never be
        counted even when they are the freshest thing on disk."""
        usb, local = two_dbs
        _build_articles_db(local, [
            {"id": 1, "url": "https://x/live", "title": "real", "source": "rss",
             "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(0.3)},
            {"id": 2, "url": "backtest://run_1/2026-05-16/BUY/NVDA",
             "title": "synthetic", "source": "backtest_run_1_winner",
             "ai_score": 5.0, "urgency": 0, "first_seen": _iso_ago(0.1)},
            {"id": 3, "url": "https://x/op", "title": "annot",
             "source": "opus_annotation_1", "ai_score": 5.0, "urgency": 0,
             "first_seen": _iso_ago(0.1)},
        ])
        j = _get_json(client, "/api/data-feed")
        assert j["articles_1h"] == 1
        assert j["articles_24h"] == 1
        assert j["top_sources"] == [{"name": "rss", "count": 1}]

    def test_graceful_zero_shape_when_no_db(self, two_dbs, client):
        """Caller/UI contract: no resolvable DB → valid zeroed body (the
        widget still renders), never a 500."""
        # Neither temp DB created → _articles_db_path() is None.
        assert dashboard._articles_db_path() is None
        j = _get_json(client, "/api/data-feed")
        assert j["articles_1h"] == 0
        assert j["articles_24h"] == 0
        assert j["top_sources"] == []
        assert j.get("error")

    def test_one_hour_vs_24h_window_boundary(self, two_dbs, client):
        """The 1h / 24h cutoffs are independent: an article 3h old counts in
        24h but not 1h (off-by-window regression lock)."""
        usb, local = two_dbs
        _build_articles_db(local, [
            {"id": 1, "url": "https://x/recent", "title": "t", "source": "s",
             "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(0.2)},
            {"id": 2, "url": "https://x/old", "title": "t", "source": "s",
             "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(3)},
            {"id": 3, "url": "https://x/ancient", "title": "t", "source": "s",
             "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(30)},
        ])
        j = _get_json(client, "/api/data-feed")
        assert j["articles_1h"] == 1   # only the 0.2h row
        assert j["articles_24h"] == 2  # 0.2h + 3h, not the 30h row
