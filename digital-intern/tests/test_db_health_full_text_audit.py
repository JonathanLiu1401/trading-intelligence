"""Pin the ``full_text_type_audit`` primitive in storage/db_health.py.

Surfaces the bug class fixed by the 2026-05-27 ``decompress`` defense and
``source_quality_scorer`` compression fix: a collector writing a raw str
into the BLOB-declared ``full_text`` column leaves a TEXT-affinity row
that the scorer worker re-fetches and crashes on every cycle.
``full_text_type_audit`` answers "is there a row in this state right
now?" as a single queryable metric, so a future regression is caught by
monitoring rather than by tail-grepping daemon.log for the error string.

Read-only invariants checked here:
  * audit returns "OK" with no DB write when no bad rows exist;
  * audit returns "DEGRADED" with ``scorer_at_risk >= 1`` when at least
    one TEXT-affinity row sits at ai_score=0 / ml_score=NULL;
  * audit IGNORES rows that have already been scored (ai_score > 0)
    even if their full_text is somehow TEXT — those don't crash the
    scorer because get_unscored never fetches them;
  * audit IGNORES backtest / opus_annotation rows (live-only clause);
  * audit ignores typeof='null' rows (real, normal — collectors are
    allowed to insert null full_text for summary-less articles).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage import db_health


_SCHEMA = """
CREATE TABLE articles (
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
    time_sensitivity REAL,
    ml_score REAL DEFAULT NULL,
    score_source TEXT DEFAULT NULL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA)
    c.commit()
    return c


def _insert(
    conn, *, id, url, source, full_text, ai_score=0.0, ml_score=None,
    score_source=None,
):
    conn.execute(
        "INSERT INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        "full_text, first_seen, cycle, ml_score, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            id, url, "t", source, "", 1.0, ai_score, 0, full_text,
            datetime.now(timezone.utc).isoformat(), 0, ml_score, score_source,
        ),
    )
    conn.commit()


def test_audit_ok_when_only_blob_and_null_rows():
    conn = _conn()
    _insert(conn, id="a", url="https://x.com/a", source="rss",
            full_text=b"\x78\x9c\x4a\x4c\x05\x00\x02\x4d\x01\x27")  # bytes
    _insert(conn, id="b", url="https://x.com/b", source="rss",
            full_text=None)
    out = db_health.full_text_type_audit(conn)
    assert out["verdict"] == "OK"
    assert out["scorer_at_risk"] == 0
    assert out["non_blob_live"] == 0
    # Both row types accounted for.
    assert set(out["typeof_counts"].keys()) == {"blob", "null"}
    assert out["typeof_counts"]["blob"] == 1
    assert out["typeof_counts"]["null"] == 1


def test_audit_degraded_on_str_typed_full_text_unscored():
    """The exact regression that fired 300+ scorer warnings/day."""
    conn = _conn()
    _insert(conn, id="bad", url="internal://source_quality_report/x",
            source="source_quality_report", full_text="plain string oops")
    out = db_health.full_text_type_audit(conn)
    assert out["verdict"] == "DEGRADED", out
    assert out["scorer_at_risk"] == 1
    assert out["non_blob_live"] == 1
    assert out["typeof_counts"].get("text") == 1


def test_audit_excludes_already_scored_rows():
    """A TEXT-affinity row that already has ai_score>0 isn't fetched by
    get_unscored (filters on ai_score=0 AND ml_score IS NULL) — so it
    cannot crash the scorer and must not count as scorer_at_risk."""
    conn = _conn()
    _insert(conn, id="scored", url="https://x.com/y", source="rss",
            full_text="legacy str row that's already labeled",
            ai_score=8.0, score_source="llm")
    out = db_health.full_text_type_audit(conn)
    assert out["non_blob_live"] == 1, "must still see the type mismatch"
    assert out["scorer_at_risk"] == 0, (
        "an already-scored row cannot crash the scorer; the audit must "
        "not falsely flag DEGRADED on it"
    )
    assert out["verdict"] == "OK"


def test_audit_excludes_backtest_rows():
    """A backtest:// row with str full_text is an injection artefact —
    the scorer never reads it (live-only clause). The audit must mirror
    the same invariant."""
    conn = _conn()
    _insert(conn, id="bt", url="backtest://run_1/2026-05-21/BUY/MU",
            source="backtest_run_1_winner",
            full_text="synthetic str body")
    out = db_health.full_text_type_audit(conn)
    assert out["verdict"] == "OK"
    assert out["scorer_at_risk"] == 0
    assert out["non_blob_live"] == 0
    assert "text" not in out["typeof_counts"], (
        "backtest rows must NEVER count toward the type audit"
    )


def test_audit_excludes_opus_annotation_rows():
    conn = _conn()
    _insert(conn, id="op", url="https://x.com/op",
            source="opus_annotation_cycle_3",
            full_text="opus label str leak")
    out = db_health.full_text_type_audit(conn)
    assert out["verdict"] == "OK"
    assert out["scorer_at_risk"] == 0


def test_audit_counts_multiple_bad_rows():
    conn = _conn()
    for i in range(5):
        _insert(conn, id=f"bad{i}", url=f"https://x.com/{i}", source="rss",
                full_text=f"row {i}")
    out = db_health.full_text_type_audit(conn)
    assert out["verdict"] == "DEGRADED"
    assert out["non_blob_live"] == 5
    assert out["scorer_at_risk"] == 5


def test_audit_excludes_ml_scored_unscored_status():
    """A row that has ml_score>0 is NOT re-fetched by get_unscored
    (filter is ``ai_score=0 AND ml_score IS NULL``). So a str row with
    ml_score set can't crash the scorer either."""
    conn = _conn()
    _insert(conn, id="mlscored", url="https://x.com/m", source="rss",
            full_text="ml-scored str row", ml_score=7.5, score_source="ml")
    out = db_health.full_text_type_audit(conn)
    assert out["non_blob_live"] == 1
    assert out["scorer_at_risk"] == 0, (
        "ml_score is set → get_unscored skips this row → cannot crash scorer"
    )
    assert out["verdict"] == "OK"


def test_health_report_includes_full_text_type_audit(tmp_path):
    """Sanity: the audit block must be wired into ``health_report`` so
    the dashboard / CLI snapshot surfaces it without a special call."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    report = db_health.health_report(db_path=db, log_path=tmp_path / "nope.log")
    assert "full_text_type_audit" in report, report
    assert report["full_text_type_audit"]["verdict"] in ("OK", "DEGRADED")
