"""KW-vs-AI divergence detector — threshold semantics on real ai_score scale.

Pins the scale fix: ``ai_score`` is the 0..10 LLM relevance scale documented
in CLAUDE.md (set by urgency_scorer + the trainer with the same magnitude as
heuristic ``kw_score``). The previous thresholds (``AI_HIGH=0.50``,
``AI_LOW=0.15``) were leftover from a 0..1 normalisation that never landed:
``AI_HIGH=0.5`` then matched ``ai_score=1.0`` (Sonnet's "engaged at all" floor)
as a hidden gem, so the hidden_gems list became "anything Sonnet rated >=1" —
pure noise the analyst could not action.

These tests fix the thresholds at the right scale and pin the regime semantics
so a future drift back to 0..1 (or any rebalance that loses the >=6 floor)
fails loudly.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from analytics import kw_ai_divergence as kad


def _build_tmp_db(tmp_path, rows):
    """Build a one-row-per-spec temp DB matching ``articles`` schema. Each
    spec is a dict with keys: id, source, kw_score, ai_score, title (default).
    ``first_seen`` is auto-set to a recent timestamp so all rows are scanned.
    """
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL,
            source TEXT, published TEXT, kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0, urgency INTEGER DEFAULT 0,
            full_text BLOB, first_seen TEXT NOT NULL, cycle INTEGER DEFAULT 0,
            time_sensitivity REAL, ml_score REAL, score_source TEXT
        );
        CREATE INDEX idx_first_seen ON articles(first_seen);
    """)
    now = datetime.now(timezone.utc)
    for i, r in enumerate(rows):
        fs = (now - timedelta(minutes=i)).isoformat()
        conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, kw_score, ai_score, first_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                r["id"],
                r.get("url", f"https://x.com/{r['id']}"),
                r.get("title", f"title for {r['id']}"),
                r["source"], r["kw_score"], r["ai_score"], fs,
            ),
        )
    conn.commit()
    conn.close()
    return db


class TestThresholdsAreOnZeroToTenScale:
    """The whole point of the fix: thresholds are interpreted in the 0..10
    ai_score scale, not 0..1."""

    def test_ai_thresholds_pinned(self):
        # AI_HIGH must be at a value that Sonnet would call "relevant"
        # (the urgency prompt's RELEVANT band starts at 5). Anything < 5
        # would re-trigger the previous "any Sonnet engagement = hidden
        # gem" regression.
        assert kad.AI_HIGH >= 5.0, (
            f"AI_HIGH={kad.AI_HIGH} is below Sonnet's 'relevant' floor — "
            "hidden_gem matches Sonnet's 'engaged at all' noise"
        )
        assert kad.AI_HIGH <= 10.0
        # AI_LOW must exclude rows Sonnet engaged with at any meaningful
        # relevance (>=2) — i.e. only "unscored" (0) and "anti-loop floor"
        # (0.01) and small fractional migrated values count as "AI found
        # little signal".
        assert kad.AI_LOW < 2.0
        assert kad.AI_LOW >= 0.0

    def test_kw_thresholds_pinned(self):
        # kw_score is also 0..10 per triage/heuristic_scorer.py; the live
        # max landed in the snapshot is 10.0. KW_HIGH below 4 would
        # over-claim "keyword fired strongly".
        assert kad.KW_HIGH >= 4.0
        assert kad.KW_LOW < kad.KW_HIGH

    def test_threshold_strings_in_report(self, tmp_path, monkeypatch):
        """Roundtrip — the thresholds the report advertises in its
        ``thresholds`` block must match the constants. Catches the
        documented-vs-applied skew the dashboard would surface verbatim."""
        db = _build_tmp_db(tmp_path, [])
        monkeypatch.setattr(kad, "DB_PATH", db)
        report = kad.compute()
        fp = report["thresholds"]["false_positive"]
        hg = report["thresholds"]["hidden_gem"]
        assert f"kw>={kad.KW_HIGH}" in fp
        assert f"ai<={kad.AI_LOW}" in fp
        assert f"ai>={kad.AI_HIGH}" in hg
        assert f"kw<{kad.KW_LOW}" in hg


class TestRegimeClassification:
    """End-to-end behaviour: rows on each side of the threshold boundaries
    are correctly classified — and the previously-buggy 'engaged Sonnet
    counts as hidden gem' regime is NOT re-triggered."""

    def test_clear_false_positive_classified(self, tmp_path, monkeypatch):
        rows = [{
            "id": "fp1", "source": "reddit",
            "kw_score": 10.0, "ai_score": 0.0,  # kw fired hard, ai silent
            "title": "loud reddit noise about ai",
        }]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["false_positives"]["total"] == 1
        assert rep["hidden_gems"]["total"] == 0
        assert rep["false_positives"]["top_sources"][0]["source"] == "reddit"

    def test_clear_hidden_gem_classified(self, tmp_path, monkeypatch):
        rows = [{
            "id": "hg1", "source": "GN: SP500",
            "kw_score": 0.5, "ai_score": 8.0,  # kw missed, ai loved it
            "title": "obscure-headline strong-signal",
        }]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["hidden_gems"]["total"] == 1
        assert rep["false_positives"]["total"] == 0

    def test_low_engagement_sonnet_not_a_hidden_gem(self, tmp_path, monkeypatch):
        """The exact previously-broken case: Sonnet engaged and rated 1.0
        ('looked at it, no signal'). Under the pre-fix AI_HIGH=0.50 this was
        classified as a hidden_gem — the regression the fix exists to prevent."""
        rows = [{
            "id": "skim1", "source": "GN: noise",
            "kw_score": 0.5, "ai_score": 1.0,
            "title": "Sonnet looked, said low signal",
        }]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["hidden_gems"]["total"] == 0, (
            "Sonnet's 'engaged at low relevance' row was mis-classified "
            "as a hidden gem — AI_HIGH threshold regressed below 5.0"
        )

    def test_mid_kw_mid_ai_in_neither_bucket(self, tmp_path, monkeypatch):
        rows = [{
            "id": "mid1", "source": "rss",
            "kw_score": 4.0, "ai_score": 4.0,
            "title": "middle of the road",
        }]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["false_positives"]["total"] == 0
        assert rep["hidden_gems"]["total"] == 0

    def test_rate_computed_against_scanned(self, tmp_path, monkeypatch):
        # 1 false_positive, 1 hidden_gem, 8 noise → fp_rate = 0.10
        rows = [
            {"id": "fp", "source": "s1", "kw_score": 10.0, "ai_score": 0.0,
             "title": "loud"},
            {"id": "hg", "source": "s2", "kw_score": 0.5, "ai_score": 7.0,
             "title": "quiet"},
        ] + [
            {"id": f"n{i}", "source": "s3", "kw_score": 4.0, "ai_score": 4.0,
             "title": "m"}
            for i in range(8)
        ]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["scanned"] == 10
        assert rep["false_positives"]["rate"] == pytest.approx(0.10)
        assert rep["hidden_gems"]["rate"] == pytest.approx(0.10)


class TestBacktestIsolation:
    """The system's most load-bearing invariant: synthetic backtest rows
    must NEVER leak into a live-pool read. Mirrors the discipline pinned
    in test_article_store.TestBacktestIsolation."""

    def test_backtest_url_excluded(self, tmp_path, monkeypatch):
        rows = [
            {"id": "live", "source": "rss", "kw_score": 10.0, "ai_score": 0.0,
             "title": "live noise headline", "url": "https://reuters.com/x"},
            {"id": "bt", "source": "backtest_run_42_winner",
             "kw_score": 10.0, "ai_score": 0.0,
             "title": "backtest noise", "url": "backtest://run_42/2026-05-23/BUY/MU"},
        ]
        db = _build_tmp_db(tmp_path, rows)
        monkeypatch.setattr(kad, "DB_PATH", db)
        rep = kad.compute()
        assert rep["scanned"] == 1, "backtest row leaked into the scan"
        assert all(r["source"] != "backtest_run_42_winner"
                   for r in rep["false_positives"]["top_sources"])
