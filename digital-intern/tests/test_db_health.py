"""storage/db_health.py — read-only ingestion & write-contention monitor.

These tests pin the behaviour a news analyst relies on: dropped-batch events
(silent article loss) are counted correctly and never inflated by the tty
echo lines; ingestion counts are *live news only* (backtest:// /
``backtest_*`` / ``opus_annotation*`` rows must never be counted as
collection); a source that has gone dark is flagged; and the canonical
backtest-isolation clause has not drifted from article_store's copy.

All DB tests use in-memory SQLite; the log parser uses a tmp file. No
external calls, no real ``articles.db`` touched.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage import db_health

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)

_SCHEMA = """
CREATE TABLE articles (
    id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL, source TEXT,
    published TEXT, kw_score REAL DEFAULT 0, ai_score REAL DEFAULT 0,
    urgency INTEGER DEFAULT 0, full_text BLOB, first_seen TEXT NOT NULL,
    cycle INTEGER DEFAULT 0, time_sensitivity REAL
);
"""


def _iso(minutes_ago: float = 0.0, days_ago: float = 0.0) -> str:
    return (NOW - timedelta(minutes=minutes_ago, days=days_ago)).isoformat()


def _conn(rows: list[tuple[str, str, str]]) -> sqlite3.Connection:
    """rows = list of (source, url, first_seen_iso)."""
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA)
    for i, (src, url, fs) in enumerate(rows):
        c.execute(
            "INSERT INTO articles (id, url, title, source, first_seen) "
            "VALUES (?,?,?,?,?)",
            (f"id{i}", url, f"title {i}", src, fs),
        )
    c.commit()
    return c


# ── canonical-clause drift guard ────────────────────────────────────────────

def test_live_only_clause_matches_canonical():
    """If article_store's clause changes, this duplicate MUST be updated too."""
    from storage import article_store

    assert db_health.LIVE_ONLY_CLAUSE == article_store._LIVE_ONLY_CLAUSE


# ── ingestion_by_source ─────────────────────────────────────────────────────

def test_ingestion_excludes_backtest_and_respects_window():
    c = _conn([
        ("rss", "https://a.com/1", _iso(minutes_ago=10)),     # in window
        ("web", "https://b.com/2", _iso(minutes_ago=50)),     # in window
        ("rss", "https://c.com/3", _iso(minutes_ago=90)),     # OUT of 1h window
        ("rss", "backtest://run_7/2026-05-18/BUY/NVDA", _iso(minutes_ago=5)),   # excluded: url
        ("backtest_run_42_winner", "https://d.com/4", _iso(minutes_ago=5)),     # excluded: source
        ("opus_annotation_cycle_3", "https://e.com/5", _iso(minutes_ago=5)),    # excluded: source
    ])
    by_src = db_health.ingestion_by_source(c, hours=1.0, now=NOW)
    assert by_src == {"rss": 1, "web": 1}
    assert sum(by_src.values()) == 2


# ── newest_live_age_seconds ─────────────────────────────────────────────────

def test_newest_age_ignores_newer_backtest_row():
    c = _conn([
        ("rss", "https://a.com/1", _iso(minutes_ago=10)),                 # newest LIVE
        ("backtest_run_9", "backtest://run_9/x", _iso(minutes_ago=1)),    # newer but synthetic
    ])
    age = db_health.newest_live_age_seconds(c, now=NOW)
    assert age is not None
    assert abs(age - 600) < 2.0  # ~10 min, NOT ~1 min


def test_newest_age_none_when_no_live_rows():
    c = _conn([("backtest_run_1", "backtest://x", _iso(minutes_ago=1))])
    assert db_health.newest_live_age_seconds(c, now=NOW) is None


# ── stale_sources ───────────────────────────────────────────────────────────

def test_stale_sources_flags_gone_dark_only():
    c = _conn([
        ("rss", "https://a.com/1", _iso(minutes_ago=10)),       # in 2h window → healthy
        ("finnhub", "https://b.com/2", _iso(minutes_ago=360)),  # in 48h baseline, silent 6h → STALE
        ("gdelt", "https://c.com/3", _iso(days_ago=5)),         # >48h baseline → not actionable
        ("backtest_run_3", "backtest://x", _iso(minutes_ago=2)),  # synthetic → never listed
    ])
    assert db_health.stale_sources(
        c, window_hours=2.0, baseline_hours=48.0, now=NOW
    ) == ["finnhub"]


# ── count_dropped_batches (the data-loss counter) ───────────────────────────

def test_count_dropped_batches_window_op_split_and_echo_exclusion(tmp_path):
    log = tmp_path / "daemon.log"
    log.write_text(
        "2026-05-18T11:55:00Z [ERROR] article_store: [article_store] insert_batch: "
        "lock retry exhausted after 5 attempts — raising\n"
        "2026-05-18T11:56:00Z [ERROR] article_store: [article_store] insert_batch: "
        "lock retry exhausted after 5 attempts — raising\n"
        "2026-05-18T11:57:00Z [ERROR] article_store: [article_store] "
        "update_ml_scores_batch: lock retry exhausted after 5 attempts — raising\n"
        "2026-05-18T09:00:00Z [ERROR] article_store: [article_store] insert_batch: "
        "lock retry exhausted after 5 attempts — raising\n"               # outside 1h
        "\x1b[33m11:55:00 [E] insert_batch: lock retry exhausted after 5 attempts\n"  # tty echo
        "2026-05-18T11:58:00Z [INFO] daemon: [rss] alive\n"               # unrelated
    )
    out = db_health.count_dropped_batches(log, hours=1.0, now=NOW)
    assert out == {"insert_batch": 2, "update_ml_scores_batch": 1, "_total": 3}


def test_count_dropped_batches_missing_file():
    assert db_health.count_dropped_batches(
        "/no/such/daemon.log", hours=1.0, now=NOW
    ) == {"_total": 0}


# ── wal_status ──────────────────────────────────────────────────────────────

def test_wal_status_sizes(tmp_path):
    db = tmp_path / "articles.db"
    db.write_bytes(b"\0" * 300_000)
    assert db_health.wal_status(db) == {"db_mb": 0.3, "wal_mb": 0.0}
    (tmp_path / "articles.db-wal").write_bytes(b"\0" * 200_000)
    assert db_health.wal_status(db) == {"db_mb": 0.3, "wal_mb": 0.2}


# ── resolve_db_path (no side effects) ───────────────────────────────────────

def test_resolve_db_path_prefers_usb_then_local(tmp_path, monkeypatch):
    usb = tmp_path / "usb"
    usb.mkdir()
    (usb / "articles.db").write_bytes(b"x")
    monkeypatch.setattr(db_health, "_USB_PATH", usb)
    assert db_health.resolve_db_path() == usb / "articles.db"

    monkeypatch.setattr(db_health, "_USB_PATH", tmp_path / "absent")
    monkeypatch.setattr(db_health, "_LOCAL_PATH", tmp_path / "data")
    assert db_health.resolve_db_path() == tmp_path / "data" / "articles.db"
    # strictly read-only: resolving must NOT create the fallback dir
    assert not (tmp_path / "data").exists()


# ── health_report integration ───────────────────────────────────────────────

def test_health_report_degrades_gracefully_without_db(tmp_path):
    log = tmp_path / "daemon.log"
    log.write_text(
        "2026-05-18T11:59:00Z [ERROR] article_store: [article_store] insert_batch: "
        "lock retry exhausted after 5 attempts — raising\n"
    )
    rep = db_health.health_report(
        db_path=tmp_path / "nope.db", log_path=log, hours=1.0, now=NOW
    )
    assert "error" in rep  # DB missing → reported, not crashed
    assert rep["dropped_batches"] == {"insert_batch": 1, "_total": 1}
    assert rep["db_path"].endswith("nope.db")
    assert rep["window_hours"] == 1.0
