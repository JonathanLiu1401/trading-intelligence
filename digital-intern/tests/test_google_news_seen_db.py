"""google_news._ensure_db must harden the seen_articles connection.

Regression pin for the 30+ ``[google_news_worker] error: database is
locked; backing off Ns`` events observed in structured.jsonl. The seen
DB connection was opened with SQLite's default busy_timeout of 0, so any
transient lock raised OperationalError immediately and aborted the whole
pass. The contract: the connection waits out contention (busy_timeout)
and uses WAL, matching the canonical source_health.py / article_store.py
pattern.
"""
from __future__ import annotations

import pytest

from collectors import google_news


@pytest.fixture
def seen_db(tmp_path, monkeypatch):
    """Point google_news at an isolated seen_articles DB."""
    monkeypatch.setattr(google_news, "DB_PATH", tmp_path / "seen_articles.db")
    return google_news


def test_ensure_db_sets_busy_timeout(seen_db):
    conn = seen_db._ensure_db()
    try:
        (busy_timeout,) = conn.execute("PRAGMA busy_timeout").fetchone()
        assert busy_timeout == 30000
    finally:
        conn.close()


def test_ensure_db_uses_wal(seen_db):
    conn = seen_db._ensure_db()
    try:
        (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_ensure_db_roundtrips_seen_marks(seen_db):
    """The hardening must not regress the dedup contract itself."""
    conn = seen_db._ensure_db()
    try:
        assert seen_db._is_seen(conn, "abc") is False
        seen_db._mark_seen(conn, "abc", "http://x", "title", "Google News")
        conn.commit()
        assert seen_db._is_seen(conn, "abc") is True
    finally:
        conn.close()
