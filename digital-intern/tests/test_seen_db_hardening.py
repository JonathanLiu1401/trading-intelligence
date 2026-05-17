"""Fleet-wide pin: every seen_articles.db writer must harden its connection.

Eleven collectors share the single ``data/seen_articles.db`` file, each
running as its own daemon worker thread on its own interval. SQLite's
default ``busy_timeout`` is 0, so any transient lock from a concurrent
collector's write (or a WAL checkpoint) raises
``OperationalError("database is locked")`` *immediately*. In every collector
``_ensure_db`` opens the connection and the surrounding ``collect_*()`` has a
broad ``except`` that returns ``[]`` / trips the worker's 5–300s backoff —
so an unhardened connection silently drops the whole fetched batch on any
cross-writer contention.

``google_news._ensure_db`` was hardened first (commit 76f9baa) with the
canonical ``article_store`` / ``source_health`` pattern (``timeout=30`` +
``PRAGMA journal_mode=WAL`` + ``PRAGMA busy_timeout=30000``). The other ten
writers carried the identical bare-connection bug; this suite pins all
eleven so the canonical pattern can't silently drift back out on any single
collector — the same defense-in-depth discipline as the backtest-isolation
parity suites. ``google_news`` is included as the canonical reference so
this file is the single source of truth for the whole fleet.
"""
from __future__ import annotations

import importlib

import pytest

# Every module that opens data/seen_articles.db via its own _ensure_db().
SEEN_DB_COLLECTORS = [
    "rss_collector",
    "gdelt_collector",
    "finnhub_collector",
    "polygon_collector",
    "newsapi_collector",
    "sec_edgar",
    "massive_collector",
    "yahoo_ticker_rss",
    "wikipedia_collector",
    "alphavantage_collector",
    "google_news",  # canonical reference (76f9baa) — kept here as source of truth
]


@pytest.mark.parametrize("mod_name", SEEN_DB_COLLECTORS)
def test_seen_db_busy_timeout_and_wal(mod_name, tmp_path, monkeypatch):
    """_ensure_db must return a WAL connection with a 30s busy_timeout.

    Without busy_timeout the daemon logged 30+ ``database is locked``
    backoff events (observed in structured.jsonl for google_news before
    76f9baa); the other ten collectors share the same file and the same
    failure mode.
    """
    mod = importlib.import_module(f"collectors.{mod_name}")
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / f"{mod_name}_seen.db")
    conn = mod._ensure_db()
    try:
        (busy_timeout,) = conn.execute("PRAGMA busy_timeout").fetchone()
        (journal_mode,) = conn.execute("PRAGMA journal_mode").fetchone()
        assert busy_timeout == 30000, (
            f"{mod_name}._ensure_db opened with busy_timeout={busy_timeout} "
            f"(expected 30000) — a transient cross-writer lock will abort "
            f"the whole pass instead of waiting it out"
        )
        assert journal_mode.lower() == "wal", (
            f"{mod_name}._ensure_db journal_mode={journal_mode!r} "
            f"(expected 'wal') — does not match the canonical "
            f"article_store/source_health/google_news hardening"
        )
    finally:
        conn.close()


@pytest.mark.parametrize("mod_name", SEEN_DB_COLLECTORS)
def test_seen_db_roundtrips_dedup_mark(mod_name, tmp_path, monkeypatch):
    """The hardening must not regress the seen-articles dedup contract.

    A row written into seen_articles must be visible to a subsequent
    ``SELECT 1 ... WHERE id=?`` on a fresh connection to the same file
    (WAL persists across connections), and absent before it is written.
    """
    mod = importlib.import_module(f"collectors.{mod_name}")
    db_file = tmp_path / f"{mod_name}_seen.db"
    monkeypatch.setattr(mod, "DB_PATH", db_file)

    conn = mod._ensure_db()
    try:
        before = conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", ("aid-1",)
        ).fetchone()
        assert before is None
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles "
            "(id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            ("aid-1", "http://x/a", "Title A", mod_name, "2026-01-01T00:00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    # Fresh connection — proves the write is durable across connections,
    # which is the actual dedup contract the collectors rely on.
    conn2 = mod._ensure_db()
    try:
        after = conn2.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", ("aid-1",)
        ).fetchone()
        assert after is not None, (
            f"{mod_name}: seen_articles row not durable across connections — "
            f"dedup contract broken by the hardening change"
        )
    finally:
        conn2.close()
