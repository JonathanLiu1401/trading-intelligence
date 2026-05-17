"""End-to-end test for /api/feed-health + the _feed_db_probe live-only lock.

Drives the real Flask endpoint via the test client against a temp Store and
two temp articles.db files (a stale USB candidate + a fresh local one) so it
exercises the exact split-brain the feature exists to surface, and pins that a
planted fresh ``backtest://`` row never reads as the newest article (the
live-only invariant — if it leaked, the split-brain detector would be defeated
by training data). Per the paper-trader-analytics-verification note, endpoints
are verified through the Flask test client, never a module __main__ smoke.
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

from paper_trader import store as store_mod
from paper_trader import signals as sig_mod
from paper_trader.store import Store

_SCHEMA = """
CREATE TABLE articles (
    id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, published TEXT,
    kw_score REAL, ai_score REAL, urgency REAL, first_seen TEXT,
    cycle INTEGER, full_text BLOB
)
"""


def _iso(hours_ago):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _make_db(path: Path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(_SCHEMA)
    conn.executemany(
        "INSERT INTO articles (id,url,title,source,ai_score,urgency,first_seen) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_feed_db_probe_excludes_backtest_rows():
    """The live-only clause must hold: a fresher backtest:// row is invisible."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "articles.db"
        _make_db(db, [
            ("a1", "https://x/1", "live story", "rss", 5.0, 0, _iso(5.0)),
            # NEWER synthetic rows — must be excluded by every clause arm
            ("b1", "backtest://run_1/x", "synthetic", "rss", 9.0, 0, _iso(0.5)),
            ("b2", "https://x/2", "synthetic", "backtest_run_1_winner", 9.0, 0, _iso(0.4)),
            ("b3", "https://x/3", "synthetic", "opus_annotation_1", 9.0, 0, _iso(0.3)),
        ])
        from paper_trader.dashboard import _feed_db_probe
        out = _feed_db_probe(str(db), want_counts=True)
        assert out["exists"] is True
        # newest is the 5h-old LIVE row, not the 0.3h synthetic ones
        assert out["newest"].startswith(_iso(5.0)[:13])
        assert out["live_24h"] == 1   # only the one live row
        assert out["live_2h"] == 0    # the live row is 5h old; synthetics excluded


@pytest.fixture
def split_brain_client(tmp_path, monkeypatch):
    """Temp Store with a 0-signal streak + a stale USB / fresh LOCAL split."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    for _ in range(5):                       # 5 consecutive 0-signal decisions
        s.record_decision(True, 0, "HOLD NONE → HOLD", "{}", 972.0, 6.0)

    usb = tmp_path / "usb_articles.db"        # the resolved (USB-first) DB
    local = tmp_path / "local_articles.db"    # the fresh one the daemon writes
    _make_db(usb, [
        ("u1", "https://x/u", "old live", "rss", 5.0, 0, _iso(19.8)),
        # a fresher backtest row that must NOT win MAX(first_seen)
        ("ubt", "backtest://run/9", "synthetic", "rss", 9.0, 0, _iso(0.05)),
    ])
    _make_db(local, [("l1", "https://x/l", "fresh live", "rss", 6.0, 0, _iso(0.1))])

    monkeypatch.setattr(sig_mod, "USB_DB", usb)
    monkeypatch.setattr(sig_mod, "LOCAL_DB", local)

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as client:
        yield client
    s.close()
    store_mod._singleton = None


def test_endpoint_flags_blind_split_brain(split_brain_client):
    r = split_brain_client.get("/api/feed-health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "BLIND"               # 5-decision 0-signal streak
    assert d["blind_streak"] == 5
    assert d["split_brain"] is True
    assert d["restart_recommended"] is True
    # Post freshness-aware _db_path(): the trader now resolves the FRESH local
    # DB (the bug fix). The hazard moved to "a process still on the old
    # existence-first resolver reads the stale USB" — surfaced via legacy_*.
    assert d["resolved_path"].endswith("local_articles.db")
    assert d["resolved_newest_age_h"] < 2.0      # fresh local feed
    assert d["legacy_path"].endswith("usb_articles.db")
    # ~19.8h, NOT ~0.05h → the fresher backtest:// row on USB was correctly
    # excluded by the live-only MAX(first_seen) probe.
    assert d["legacy_newest_age_h"] >= 19.0
    assert d["resolved_live_2h"] == 1            # the fresh DB does carry news
    assert "split-brain" in d["headline"]
