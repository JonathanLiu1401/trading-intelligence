"""Tests for scripts/score_divergence.py — the ml_score vs ai_score divergence
detector.

Discriminating contract: a row carrying a model self-prediction but NO real
LLM ground-truth label (``ai_score=0``, schema default) is NOT divergence —
it is an unlabelled row. The earlier version of this script queried
``ai_score IS NOT NULL``, which is a tautology on a ``REAL DEFAULT 0`` column,
and reported every model-scored row as ``ml_higher`` against an implicit
ai=0 (live evidence: ``divergent=5000  ml_higher_pct=100.0%``).

These tests pin the fix:
  * ``load_rows`` SQL excludes ``ai_score = 0`` and backtest-isolation rows.
  * ``classify_divergent`` returns specific gap values and directions.
  * ``build_summary`` aggregates exactly the rows the classifier returned.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import score_divergence as mod


_SCHEMA = """
CREATE TABLE articles (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT,
    published    TEXT,
    kw_score     REAL DEFAULT 0,
    ai_score     REAL DEFAULT 0,
    urgency      INTEGER DEFAULT 0,
    full_text    BLOB,
    first_seen   TEXT NOT NULL,
    cycle        INTEGER DEFAULT 0,
    time_sensitivity REAL DEFAULT NULL,
    ml_score     REAL DEFAULT NULL,
    score_source TEXT DEFAULT NULL
);
"""


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """DB with: 1 fresh divergent live row, 1 fresh non-divergent live row,
    1 fresh model-only (ai=0) row, 1 fresh backtest synthetic row, and 1
    stale row outside the window."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=10)).isoformat()
    stale = (now - timedelta(hours=72)).isoformat()
    rows = [
        # (id, url, title, source, ai_score, ml_score, urgency, first_seen)
        # Fresh divergent live: ai=2.0, ml=8.0 → gap 6.0, ml_higher.
        ("live-div", "https://wire/a", "Divergent live", "rss",
         2.0, 8.0, 0, fresh),
        # Fresh non-divergent live: ai=6.0, ml=6.1 → gap 0.10 < MIN_GAP 0.30.
        ("live-tight", "https://wire/b", "Tight live", "rss",
         6.0, 6.1, 0, fresh),
        # Fresh model-only: ai=0 default, ml=9.5 → MUST NOT appear (this is
        # the recurring bug: not divergence, just an unlabelled row).
        ("model-only", "https://wire/c", "Model only", "finnhub",
         0.0, 9.5, 0, fresh),
        # Fresh backtest synthetic — has real ai_score+ml_score but is a
        # training row; _LIVE_ONLY_CLAUSE must exclude it.
        ("bt-row", "backtest://run_1/foo", "Synthetic", "backtest_winner",
         5.0, 0.1, 0, fresh),
        # Stale row outside the 24h window.
        ("stale-row", "https://wire/d", "Stale", "rss",
         3.0, 9.0, 0, stale),
    ]
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, ai_score, ml_score, "
        "urgency, first_seen) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def test_load_rows_excludes_unlabelled_and_synthetic(seeded_db: Path):
    rows = mod.load_rows(seeded_db, window_hours=24)
    ids = {r[0] for r in rows}
    # The two live rows with real ai_score + ml_score remain; everything else
    # is filtered at the SQL layer.
    assert ids == {"live-div", "live-tight"}, (
        f"expected live rows only, got {ids} — "
        "ai_score=0 / backtest / stale rows must be excluded"
    )


def test_classify_divergent_excludes_below_threshold(seeded_db: Path):
    rows = mod.load_rows(seeded_db, window_hours=24)
    div = mod.classify_divergent(rows, min_gap=0.30)
    # Only live-div (gap=6.0) survives; live-tight (gap=0.10) is below.
    assert [d["id"] for d in div] == ["live-div"]
    d = div[0]
    assert d["gap"] == 6.0, d
    assert d["direction"] == "ml_higher", d
    assert d["ai_score"] == 2.0 and d["ml_score"] == 8.0


def test_classify_divergent_sorts_descending_by_gap():
    rows = [
        ("a", "t1", "rss", 1.0, 9.0, 0, "now"),  # gap 8.0
        ("b", "t2", "rss", 5.0, 6.0, 0, "now"),  # gap 1.0
        ("c", "t3", "rss", 9.0, 1.0, 0, "now"),  # gap 8.0 — same as a
    ]
    div = mod.classify_divergent(rows, min_gap=0.30)
    gaps = [d["gap"] for d in div]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert {d["direction"] for d in div} == {"ml_higher", "ai_higher"}


def test_classify_direction_ai_higher():
    rows = [("x", "ti", "rss", 9.0, 2.0, 0, "now")]
    div = mod.classify_divergent(rows, min_gap=0.30)
    assert div[0]["direction"] == "ai_higher"
    assert div[0]["gap"] == 7.0


def test_build_summary_no_div_zero_pcts():
    s = mod.build_summary(divergent=[], sampled=42)
    assert s["divergent_count"] == 0
    assert s["sampled"] == 42
    # Both ratios must collapse to 0 without ZeroDivisionError.
    assert s["avg_gap"] == 0.0
    assert s["ml_higher_pct"] == 0.0
    assert s["top"] == []


def test_build_summary_aggregates_correctly():
    divergent = [
        {"id": "a", "title": "t", "source": "rss", "ai_score": 1.0,
         "ml_score": 9.0, "gap": 8.0, "direction": "ml_higher",
         "urgency": 0, "first_seen": "now"},
        {"id": "b", "title": "t", "source": "rss", "ai_score": 9.0,
         "ml_score": 1.0, "gap": 8.0, "direction": "ai_higher",
         "urgency": 0, "first_seen": "now"},
    ]
    s = mod.build_summary(divergent, sampled=2)
    assert s["divergent_count"] == 2
    assert s["avg_gap"] == 8.0
    # One of two divergent rows was ml_higher → 50.0%
    assert s["ml_higher_pct"] == 50.0


def test_end_to_end_no_one_directional_failure(seeded_db: Path):
    """The original bug's smoking gun was ml_higher_pct=100.0%. Verify the
    fixed pipeline (load_rows → classify → build_summary) does NOT degrade
    to a one-directional 100% report when the live row has ai>ml."""
    # Replace the live row's direction so the fixed pipeline shows ai_higher.
    conn = sqlite3.connect(str(seeded_db))
    conn.execute(
        "UPDATE articles SET ai_score=8.0, ml_score=1.0 WHERE id='live-div'"
    )
    conn.commit()
    conn.close()
    rows = mod.load_rows(seeded_db, window_hours=24)
    div = mod.classify_divergent(rows, min_gap=0.30)
    s = mod.build_summary(div, sampled=len(rows))
    # The classifier must see only the one truly divergent live row; that
    # row is ai_higher → ml_higher_pct must be 0.0. The OLD bug would have
    # included the model-only ai=0/ml=9.5 row and reported >50% ml_higher.
    assert s["ml_higher_pct"] == 0.0, (
        f"ml_higher_pct={s['ml_higher_pct']} suggests model-only or synthetic "
        "rows leaked back into the divergence report"
    )
    assert s["divergent_count"] == 1
