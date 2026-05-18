"""Tests for scripts/export_training_data.py.

Regression coverage for the production incident where a corrupt
``paper_trader_signals.db`` (a *derived* artifact) made ``export_worker``
fail every 30 minutes indefinitely: the exporter only did
``CREATE TABLE IF NOT EXISTS`` + ``INSERT OR REPLACE`` into the destination,
so once its B-tree was malformed the corruption could never self-heal and
every subsequent export raised ``database disk image is malformed``.

The fix rebuilds the destination atomically on every run, so corruption
self-heals and stale signals never accumulate.
"""
import sqlite3

import pytest

import scripts.export_training_data as ex


def _make_source_db(path, rows):
    """Create a minimal source articles.db with just the columns export reads."""
    c = sqlite3.connect(str(path))
    c.execute(
        "CREATE TABLE articles ("
        "id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, "
        "published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER, "
        "full_text BLOB, first_seen TEXT, cycle INTEGER, time_sensitivity REAL)"
    )
    c.executemany(
        "INSERT INTO articles (id, title, source, ai_score, full_text, first_seen) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    c.commit()
    c.close()


def _make_corrupt_signals_db(path):
    """Build a valid signals DB, then zero interior pages to corrupt its B-tree.

    This reproduces the real incident: a structurally-corrupt destination DB
    whose header is intact (so ``connect`` succeeds) but whose B-tree pages
    contain invalid page numbers, so the first write raises
    ``database disk image is malformed``.
    """
    c = sqlite3.connect(str(path))
    c.execute("PRAGMA page_size=4096")
    c.execute(
        "CREATE TABLE signals (id TEXT PRIMARY KEY, title TEXT, source TEXT, "
        "ai_score REAL, tickers TEXT, first_seen TEXT, exported_at TEXT)"
    )
    c.executemany(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
        [(f"old{i}", f"stale {i}", "rss", 5.0, "AAPL", "2026-01-01", "2026-01-01")
         for i in range(4000)],
    )
    c.commit()
    c.close()
    # Scribble zeros over interior pages (skip page 1 / the file header) so
    # the B-tree contains invalid page references.
    with open(path, "r+b") as fh:
        fh.seek(4096 * 2)
        fh.write(b"\x00" * 4096 * 20)

    # Sanity: confirm the file is genuinely malformed for writes the way the
    # production exporter exercises it.
    bad = sqlite3.connect(str(path))
    with pytest.raises(sqlite3.DatabaseError, match="malformed"):
        bad.execute(
            "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?)",
            ("x", "x", "x", 5.0, "", "", ""),
        )
        bad.commit()
    bad.close()


@pytest.fixture
def _export_dir(tmp_path, monkeypatch):
    db_path = tmp_path / "articles.db"
    monkeypatch.setattr(ex, "_get_db_path", lambda: db_path)
    return tmp_path, db_path


def test_export_self_heals_corrupt_destination(_export_dir):
    tmp_path, db_path = _export_dir
    _make_source_db(db_path, [
        ("a1", "Apple beats earnings", "rss", 5.0, None, "2026-05-17T00:00:00Z"),
        ("a2", "minor blog post", "web", 1.5, None, "2026-05-17T00:01:00Z"),
    ])
    _make_corrupt_signals_db(tmp_path / "paper_trader_signals.db")

    # Must NOT raise "database disk image is malformed" — the exporter rebuilds
    # the derived destination from scratch.
    result = ex.export_all()

    assert result["json_count"] == 2          # both ai_score > 0 rows go to JSON
    assert result["db_count"] == 1            # only ai_score >= 4.0 -> signals DB

    sig = sqlite3.connect(str(tmp_path / "paper_trader_signals.db"))
    assert sig.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert sig.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1
    assert sig.execute("SELECT id FROM signals").fetchone()[0] == "a1"
    sig.close()


def test_export_is_idempotent_and_drops_stale_rows(_export_dir):
    tmp_path, db_path = _export_dir
    _make_source_db(db_path, [
        ("a1", "Apple beats earnings", "rss", 5.0, None, "2026-05-17T00:00:00Z"),
    ])

    ex.export_all()
    ex.export_all()  # second run must rebuild, not accumulate duplicates

    sig = sqlite3.connect(str(tmp_path / "paper_trader_signals.db"))
    assert sig.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1
    sig.close()

    # Article drops below the 4.0 signal threshold -> rebuilt DB must forget it.
    c = sqlite3.connect(str(db_path))
    c.execute("UPDATE articles SET ai_score=2.0 WHERE id='a1'")
    c.commit()
    c.close()

    result = ex.export_all()
    assert result["db_count"] == 0
    sig = sqlite3.connect(str(tmp_path / "paper_trader_signals.db"))
    assert sig.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0
    sig.close()
