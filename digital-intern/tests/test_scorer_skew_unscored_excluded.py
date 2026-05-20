"""Pin the ``ai_score > 0`` filter on the per-source scorer-skew and
source-score-volatility analytics.

``ai_score`` is ``REAL DEFAULT 0`` — never NULL — so the prior
``ai_score IS NOT NULL`` clause was a tautology that swept in every
model-scored-but-LLM-unscored row at ai_score=0. That contaminated:

  * scorer_skew's per-source ai-vs-ml gap (an unlabelled row with
    ml_score=8 reported a fake gap of 8.0 against an implicit ai=0);
  * source_score_volatility's per-source ai_score variance (a source with
    100 LLM-scored rows + 900 zeros looks vastly noisier than its real
    LLM-label spread).

Discriminating contract: with one ai_score>0 row and many ai_score=0 rows
under the same source, the new ``ai_score > 0`` filter must count exactly 1.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


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


def _seed(db: Path) -> None:
    """One labelled row + four unscored zero rows under the SAME source.
    A correct filter should report n=1 for that source; the buggy
    ``IS NOT NULL`` reports n=5 with mean dragged toward 0."""
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    fresh = datetime.now(timezone.utc).isoformat()
    rows = [
        # Labelled: ai_score=8.0, ml_score=7.0 → real divergence sample.
        ("a-labelled", "https://w/1", "labelled", "rss",
         8.0, 7.0, fresh),
    ]
    # Four unscored zero rows under the same source. ai_score defaults to 0;
    # ml_score is set so they pass the IS NOT NULL check that would have
    # included them in the buggy version.
    rows += [
        (f"a-zero-{i}", f"https://w/z{i}", f"zero {i}", "rss",
         0.0, 5.0, fresh)
        for i in range(4)
    ]
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, ai_score, ml_score, "
        "first_seen) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "articles.db"
    _seed(path)
    return path


def test_scorer_skew_excludes_unscored_rows(db: Path, monkeypatch, tmp_path):
    from analytics import scorer_skew as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "skew.json")
    monkeypatch.setattr(mod, "MIN_PER_SOURCE", 1)
    payload = mod.compute()
    # Exactly the one labelled row contributes — the four ai_score=0 rows
    # must not appear. Buggy version produced n=5 and an avg_gap dragged
    # toward (avg_ml − 0) ≈ -5.4.
    assert payload["scanned"] == 1, (
        f"scanned={payload['scanned']} — the four ai_score=0 rows leaked back "
        "into the per-source ai-vs-ml skew aggregate"
    )
    assert len(payload["ranked"]) == 1
    entry = payload["ranked"][0]
    assert entry["n"] == 1
    assert entry["avg_ai"] == 8.0
    assert entry["avg_ml"] == 7.0
    # gap = ml - ai = -1.0
    assert entry["avg_gap_ml_minus_ai"] == -1.0


def test_source_score_volatility_excludes_unscored_rows(
    db: Path, monkeypatch, tmp_path
):
    from analytics import source_score_volatility as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "ssv.json")
    monkeypatch.setattr(mod, "MIN_PER_SOURCE", 1)
    payload = mod.compute()
    # Exactly the one labelled row → std=0 (single sample), mean=8.0. The
    # buggy version included four zeros so it reported n=5, mean=1.6,
    # std≈3.2 — characterising "rss" as vastly noisy on the back of
    # unscored rows alone.
    assert payload["scanned"] == 1, (
        f"scanned={payload['scanned']} — ai_score=0 rows leaked into "
        "per-source variance"
    )
    entry = payload["ranked"][0]
    assert entry["n"] == 1
    assert entry["mean"] == 8.0
    assert entry["std"] == 0.0
