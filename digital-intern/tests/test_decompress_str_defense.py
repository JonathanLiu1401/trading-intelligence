"""Defense + collector fix for the str-in-full_text regression.

Live evidence (2026-05-27): the scorer_worker raised
``a bytes-like object is required, not 'str'`` ~300x/day. Root cause:
``collectors.source_quality_scorer`` wrote a plain str ``summary`` into the
``full_text`` BLOB column. SQLite is dynamically typed, so the str
survived; ``decompress(r[4])`` in ``get_unscored`` then crashed on every
subsequent scorer batch, because the row stays at ai_score=0 /
ml_score=NULL and is re-fetched every 30s.

Two complementary regression pins:
  1. ``decompress(str)`` defense — never raises on a str-typed payload
     (treat as already-decoded). Caps the blast radius of any future
     mis-encoded collector.
  2. ``collectors.source_quality_scorer._write_article`` compresses the
     summary before insert. Root-cause fix; pinned so a future "let's
     just pass the str" refactor fails CI.
"""
from __future__ import annotations

import sqlite3
import zlib

import pytest


def test_decompress_handles_bytes_roundtrip():
    """Sanity: the normal compressed-bytes path round-trips intact."""
    from storage.article_store import compress, decompress
    out = decompress(compress("hello world"))
    assert out == "hello world"


def test_decompress_returns_str_passthrough_unchanged():
    """The defensive branch: a str slipping into the column must NOT raise.
    Treat it as already-decoded plaintext (the only sensible interpretation
    of a TEXT-affinity value in a BLOB-declared column)."""
    from storage.article_store import decompress
    assert decompress("plain summary text") == "plain summary text"
    # Empty string is the noisy edge-case (zlib would raise on b"").
    assert decompress("") == ""


def test_get_unscored_survives_str_full_text(store):
    """End-to-end pin: a row whose ``full_text`` is a TEXT-affinity str
    must NOT crash the scorer worker's ``store.get_unscored`` call.

    Pre-fix this raised ``a bytes-like object is required, not 'str'``
    and the whole batch was lost; the row is unscoreable (ai_score=0 /
    ml_score=NULL) so it re-fetched every 30s, taking the scorer dark
    300+× /day."""
    # Insert a row with a plain str in full_text — the exact shape the
    # buggy source_quality_scorer used to write.
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "src_quality_row",
                "internal://source_quality_report/2026-05-27",
                "[Source Quality] Top: x",
                "source_quality_report",
                "",
                1.0,  # kw_score so it gets pulled by get_unscored
                0.0,
                0,
                "2026-05-27T00:00:00+00:00",
                0,
                None,
                None,
                "plain string summary — not zlib-compressed",  # the bug
            ),
        )
        store.conn.commit()

    # typeof confirms we genuinely planted a str in a BLOB column.
    typ = store.conn.execute(
        "SELECT typeof(full_text) FROM articles WHERE id='src_quality_row'"
    ).fetchone()[0]
    assert typ == "text", f"fixture failed to plant a str: typeof={typ}"

    # The real fix: get_unscored must not raise here.
    rows = store.get_unscored(min_kw=0.0)
    ids = {r["_id"] for r in rows}
    assert "src_quality_row" in ids, "row should be reachable, not raise"
    summary = next(r["summary"] for r in rows if r["_id"] == "src_quality_row")
    assert summary == "plain string summary — not zlib-compressed"


def test_source_quality_scorer_compresses_full_text(tmp_path, monkeypatch):
    """The collector's ``_write_article`` MUST persist ``full_text`` as a
    zlib-compressed BLOB, never a raw str. This is the root-cause fix for
    the live regression. Asserts via ``typeof(full_text)='blob'`` + a
    round-trip decompress check on the actual cell value."""
    from collectors import source_quality_scorer as sqs

    # Build a brand-new articles.db in tmp_path with the minimal schema
    # the collector touches.
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published TEXT,
            kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0,
            urgency INTEGER DEFAULT 0,
            full_text BLOB,
            first_seen TEXT NOT NULL,
            cycle INTEGER DEFAULT 0,
            time_sensitivity REAL DEFAULT NULL,
            ml_score REAL DEFAULT NULL,
            score_source TEXT DEFAULT NULL
        )"""
    )
    conn.commit()
    conn.close()

    # Redirect the module's ARTICLES_DB to this tmp db.
    monkeypatch.setattr(sqs, "ARTICLES_DB", db)

    summary = "Top: reuters | 152 sources analyzed\nLine 2\nLine 3"
    ok = sqs._write_article(
        title="[Source Quality] Top: reuters",
        summary=summary,
        link="internal://source_quality_report/2026-05-27",
    )
    assert ok is True

    conn = sqlite3.connect(str(db))
    typ, blob = conn.execute(
        "SELECT typeof(full_text), full_text FROM articles "
        "WHERE source='source_quality_report'"
    ).fetchone()
    conn.close()

    assert typ == "blob", (
        f"source_quality_scorer wrote full_text as {typ!r}, not blob — "
        f"the exact regression that crashed the scorer 300+× /day"
    )
    # zlib-decompress round-trips back to the original summary.
    assert zlib.decompress(blob).decode("utf-8") == summary
