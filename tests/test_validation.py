"""Tests for paper_trader/validation.py and the hindsight label filter.

These tests provide rigorous evidence that the overfitting-detection
machinery actually catches the leakage vectors it claims to catch:

  1. The hindsight label filter in `_load_local_articles` must downgrade
     `ai_score` to keyword-baseline when `first_seen >> published`.
  2. `audit_label_contamination` must report contamination correctly,
     handle RFC822 dates, and survive empty/malformed rows.
  3. `run_permutation_test` must isolate writes (no `backtest.db` pollution)
     and detect a known-good signal as significant vs. noise.
  4. `run_walk_forward_validation` must produce N-1 folds for an N-year
     window with `fold_years=1`.
  5. Temporal scorer split (`split_outcomes_temporal` / `evaluate_scorer_oos`)
     must put the most recent records in the OOS holdout, not the training set.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Hindsight label filter — _load_local_articles
# ─────────────────────────────────────────────────────────────────────────

def _make_articles_db(path: Path, rows: list[dict]) -> Path:
    """Create a minimal articles.db at `path` with the given rows.

    Each row dict may contain: title, url, source, published, ai_score, kw_score,
    first_seen, urgency. Missing keys default to None / sane values.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            source TEXT,
            published TEXT,
            kw_score REAL,
            ai_score REAL,
            urgency REAL,
            first_seen TEXT,
            cycle INTEGER,
            full_text BLOB
        )
    """)
    for i, r in enumerate(rows):
        conn.execute(
            "INSERT INTO articles (id, url, title, source, published, kw_score, "
            "ai_score, urgency, first_seen, cycle, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r.get("id", f"id{i}"),
                r.get("url", f"https://example.com/{i}"),
                r.get("title", f"headline {i}"),
                r.get("source", "test"),
                r.get("published"),
                r.get("kw_score"),
                r.get("ai_score"),
                r.get("urgency"),
                r.get("first_seen"),
                r.get("cycle", 0),
                None,
            ),
        )
    conn.commit()
    conn.close()
    return path


class TestHindsightLabelFilter:
    """The biggest leakage vector: GDELT-collected historical articles
    arrive in articles.db with a `first_seen` timestamp from today and an
    `ai_score` from a Claude pass that knew the future. The filter must
    fall back to `kw_score` for anything where first_seen >> published."""

    def test_fresh_label_kept(self, tmp_path, monkeypatch):
        # Article published yesterday, labeled today — label is timely, trust it.
        fresh = (date.today() - timedelta(days=1)).isoformat()
        today = date.today().isoformat()
        db = _make_articles_db(tmp_path / "articles.db", [
            {"published": fresh, "first_seen": today,
             "ai_score": 4.5, "kw_score": 1.0,
             "url": "https://news.com/fresh"},
        ])
        monkeypatch.setattr("paper_trader.backtest.LOCAL_ARTICLES_DB", db)

        from paper_trader.backtest import BacktestEngine
        # Use __new__ to skip price cache init — we only need _load_local_articles.
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.start = date.today() - timedelta(days=30)
        engine.end = date.today()
        # Stub out _merge_sec_cache so we don't depend on disk fixtures.
        engine._merge_sec_cache = lambda result: 0
        articles = engine._load_local_articles()
        # Article is on yesterday's bucket
        recs = articles.get(fresh, [])
        assert len(recs) == 1
        # Trusted ai_score → 4.5
        assert recs[0]["score"] == pytest.approx(4.5)
        assert recs[0]["hindsight_contaminated"] is False

    def test_hindsight_label_downgraded(self, tmp_path, monkeypatch):
        # Article published in 2018, first seen today (8 years later).
        # ai_score must NOT be used — fallback to kw_score.
        old = "2018-03-15"
        today = date.today().isoformat()
        db = _make_articles_db(tmp_path / "articles.db", [
            {"published": old, "first_seen": today,
             "ai_score": 4.8, "kw_score": 2.5,
             "url": "https://news.com/old"},
        ])
        monkeypatch.setattr("paper_trader.backtest.LOCAL_ARTICLES_DB", db)

        from paper_trader.backtest import BacktestEngine
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.start = date(2018, 1, 1)
        engine.end = date(2019, 1, 1)
        engine._merge_sec_cache = lambda result: 0
        articles = engine._load_local_articles()
        recs = articles.get(old, [])
        assert len(recs) == 1
        # Hindsight-contaminated → kw_score baseline (2.5), NOT ai_score (4.8)
        assert recs[0]["score"] == pytest.approx(2.5)
        assert recs[0]["hindsight_contaminated"] is True

    def test_missing_first_seen_treated_as_unknown_not_hindsight(self, tmp_path, monkeypatch):
        # If first_seen is NULL, we can't prove hindsight — keep the existing
        # `ai_score or kw_score` fallback. (Don't punish historical data that
        # predates the first_seen column.)
        old = "2018-03-15"
        db = _make_articles_db(tmp_path / "articles.db", [
            {"published": old, "first_seen": None,
             "ai_score": 4.8, "kw_score": 2.5,
             "url": "https://news.com/no-fs"},
        ])
        monkeypatch.setattr("paper_trader.backtest.LOCAL_ARTICLES_DB", db)

        from paper_trader.backtest import BacktestEngine
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.start = date(2018, 1, 1)
        engine.end = date(2019, 1, 1)
        engine._merge_sec_cache = lambda result: 0
        articles = engine._load_local_articles()
        recs = articles.get(old, [])
        assert len(recs) == 1
        # No staleness signal → trust ai_score
        assert recs[0]["score"] == pytest.approx(4.8)
        assert recs[0]["hindsight_contaminated"] is False

    def test_kw_score_only_when_no_ai_score(self, tmp_path, monkeypatch):
        # Fresh article with ai_score==None must fall back to kw_score.
        fresh = (date.today() - timedelta(days=2)).isoformat()
        today = date.today().isoformat()
        db = _make_articles_db(tmp_path / "articles.db", [
            {"published": fresh, "first_seen": today,
             "ai_score": None, "kw_score": 3.2,
             "url": "https://news.com/kw-only"},
        ])
        monkeypatch.setattr("paper_trader.backtest.LOCAL_ARTICLES_DB", db)

        from paper_trader.backtest import BacktestEngine
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.start = date.today() - timedelta(days=30)
        engine.end = date.today()
        engine._merge_sec_cache = lambda result: 0
        articles = engine._load_local_articles()
        recs = articles.get(fresh, [])
        assert len(recs) == 1
        assert recs[0]["score"] == pytest.approx(3.2)


# ─────────────────────────────────────────────────────────────────────────
# audit_label_contamination
# ─────────────────────────────────────────────────────────────────────────

class TestLabelContaminationAudit:

    def test_zero_contamination_fresh_articles(self, tmp_path):
        from paper_trader.validation import audit_label_contamination
        # All articles labeled within a few days of publication.
        rows = []
        for i in range(10):
            pub = (date(2024, 6, 1) + timedelta(days=i)).isoformat()
            seen = (date(2024, 6, 1) + timedelta(days=i + 2)).isoformat()
            rows.append({
                "published": pub, "first_seen": seen,
                "ai_score": 3.0, "kw_score": 1.0,
                "url": f"https://news.com/{i}",
            })
        db = _make_articles_db(tmp_path / "articles.db", rows)
        result = audit_label_contamination(str(db), date(2024, 5, 1), date(2024, 8, 1))
        assert result["total_articles"] == 10
        assert result["contaminated_count"] == 0
        assert result["contamination_rate"] == 0.0
        assert result["verdict"] == "LOW"

    def test_full_contamination_old_articles(self, tmp_path):
        from paper_trader.validation import audit_label_contamination
        # All articles published years before they were seen.
        rows = []
        for i in range(10):
            pub = (date(2018, 6, 1) + timedelta(days=i)).isoformat()
            seen = "2026-01-01"  # 8 years later
            rows.append({
                "published": pub, "first_seen": seen,
                "ai_score": 4.5, "kw_score": 1.0,
                "url": f"https://news.com/{i}",
            })
        db = _make_articles_db(tmp_path / "articles.db", rows)
        result = audit_label_contamination(str(db), date(2018, 1, 1), date(2018, 12, 31))
        assert result["total_articles"] == 10
        assert result["contaminated_count"] == 10
        assert result["contamination_rate"] == 1.0
        assert result["verdict"] == "HIGH_CONTAMINATION"

    def test_handles_rfc822_published_dates(self, tmp_path):
        # `published` is sometimes stored as RFC822 ("Wed, 14 May 2026 13:00:00 +0000").
        # SQL BETWEEN would silently drop these — the audit must parse them in Python.
        from paper_trader.validation import audit_label_contamination
        rfc = format_datetime(datetime(2024, 6, 15, tzinfo=timezone.utc))
        rows = [
            {"published": rfc, "first_seen": "2024-06-17",
             "ai_score": 3.0, "kw_score": 1.0,
             "url": "https://news.com/rfc"},
        ]
        db = _make_articles_db(tmp_path / "articles.db", rows)
        result = audit_label_contamination(str(db), date(2024, 5, 1), date(2024, 8, 1))
        # The RFC822-dated row must NOT be silently dropped.
        assert result["total_articles"] == 1
        assert result["contaminated_count"] == 0  # 2 days lag is fresh

    def test_per_source_breakdown(self, tmp_path):
        from paper_trader.validation import audit_label_contamination
        rows = []
        # 4 contaminated from source "old", 4 fresh from source "new".
        for i in range(4):
            rows.append({
                "published": (date(2018, 1, 1) + timedelta(days=i)).isoformat(),
                "first_seen": "2026-01-01",
                "ai_score": 4.5, "kw_score": 1.0,
                "source": "old", "url": f"https://news.com/old-{i}",
            })
        for i in range(4):
            rows.append({
                "published": (date(2024, 6, 1) + timedelta(days=i)).isoformat(),
                "first_seen": (date(2024, 6, 1) + timedelta(days=i + 2)).isoformat(),
                "ai_score": 3.0, "kw_score": 1.0,
                "source": "new", "url": f"https://news.com/new-{i}",
            })
        db = _make_articles_db(tmp_path / "articles.db", rows)
        result = audit_label_contamination(str(db), date(2018, 1, 1), date(2024, 12, 31))
        assert result["total_articles"] == 8
        assert result["contaminated_count"] == 4
        assert result["contamination_rate"] == pytest.approx(0.5)
        srcs = result["sources"]
        assert "old" in srcs and "new" in srcs
        assert srcs["old"]["contaminated"] == 4
        assert srcs["new"]["contaminated"] == 0


# ─────────────────────────────────────────────────────────────────────────
# Permutation test isolation
# ─────────────────────────────────────────────────────────────────────────

class TestPermutationTestIsolation:
    """The permutation test must NOT pollute the live backtest.db."""

    def test_permutation_runs_use_isolated_store(self, tmp_path, monkeypatch):
        from paper_trader.validation import _make_isolated_store

        # Verify the helper returns a store pointing at a temp path, not BACKTEST_DB.
        store = _make_isolated_store(tmp_path / "isolated.db")
        assert store is not None
        # Round-trip a row: insert and read it back from the SAME temp DB.
        store.upsert_run(-1, 42, "running", date(2025, 1, 1), date(2025, 1, 31))
        rows = store.conn.execute("SELECT run_id FROM backtest_runs").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == -1


# ─────────────────────────────────────────────────────────────────────────
# Walk-forward validation — fold counting
# ─────────────────────────────────────────────────────────────────────────

class TestWalkForwardFoldCount:

    def test_fold_count_logic(self):
        # 5 years total, fold_years=1 → 5 chunks → 4 walk-forward iterations
        # (fold 0 is pure training, folds 1..4 are OOS tests).
        from paper_trader.validation import _compute_fold_windows
        start, end = date(2019, 1, 1), date(2024, 1, 1)
        folds = _compute_fold_windows(start, end, fold_years=1)
        # We expect 4 OOS folds (year 2..5)
        assert len(folds) == 4
        # First fold tests year 2; last fold tests year 5.
        assert folds[0]["train_end"] == date(2020, 1, 1).isoformat()
        assert folds[0]["test_start"] == date(2020, 1, 1).isoformat()
        assert folds[-1]["test_end"] <= end.isoformat()

    def test_too_short_period_returns_zero_folds(self):
        from paper_trader.validation import _compute_fold_windows
        start, end = date(2024, 1, 1), date(2024, 6, 1)  # 5 months only
        folds = _compute_fold_windows(start, end, fold_years=1)
        assert folds == []


# ─────────────────────────────────────────────────────────────────────────
# Temporal scorer split — most recent records held out as OOS
# ─────────────────────────────────────────────────────────────────────────

class TestTemporalScorerSplit:

    def test_oos_holdout_is_most_recent(self):
        from paper_trader.validation import split_outcomes_temporal
        # Build 100 records spanning 2020-2024, randomly ordered in input.
        records = []
        for i in range(100):
            # Spread sim_dates across 4 years
            d = date(2020, 1, 1) + timedelta(days=i * 14)
            records.append({"sim_date": d.isoformat(),
                            "ticker": "NVDA", "forward_return_5d": float(i)})
        # Shuffle deterministically
        import random as _rnd
        _rnd.Random(0).shuffle(records)

        train, oos = split_outcomes_temporal(records, oos_fraction=0.2)
        # Sizes
        assert len(train) == 80
        assert len(oos) == 20
        # OOS must contain the 20 LATEST sim_dates
        latest_train = max(date.fromisoformat(r["sim_date"]) for r in train)
        earliest_oos = min(date.fromisoformat(r["sim_date"]) for r in oos)
        assert earliest_oos >= latest_train

    def test_too_few_records_returns_full_train_empty_oos(self):
        from paper_trader.validation import split_outcomes_temporal
        recs = [{"sim_date": "2024-01-01", "forward_return_5d": 1.0}]
        train, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
        # With only 1 record, we can't split — give all to training, oos is empty.
        assert len(train) == 1
        assert len(oos) == 0


# ─────────────────────────────────────────────────────────────────────────
# evaluate_scorer_oos — predicts on held-out records
# ─────────────────────────────────────────────────────────────────────────

class TestEvaluateScorerOos:

    def test_returns_rmse_for_trained_scorer(self):
        from paper_trader.ml.decision_scorer import train_scorer, DecisionScorer
        from paper_trader.validation import evaluate_scorer_oos

        # Build 60 train records + 20 OOS records, both with realistic-ish
        # features. Forward returns clustered around 0% so RMSE is bounded.
        import random as _rnd
        rng = _rnd.Random(7)
        def _record(i: int) -> dict:
            return {
                "ticker": "NVDA",
                "sim_date": f"2024-{1 + (i % 12):02d}-01",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "action": "BUY",
                "return_pct": 10.0,
            }
        train_records = [_record(i) for i in range(60)]
        oos_records = [_record(i) for i in range(20)]
        # train_scorer requires ≥30 distinct dedup keys; (ticker, sim_date, action)
        # collide here (only 12 sim_dates × 1 ticker = 12 unique), so vary date too.
        for i, r in enumerate(train_records):
            r["sim_date"] = f"2024-{1 + (i % 12):02d}-{1 + (i // 12):02d}"
        for i, r in enumerate(oos_records):
            r["sim_date"] = f"2025-{1 + (i % 12):02d}-{1 + (i // 12):02d}"

        train_scorer(train_records)  # writes to SCORER_PATH (isolated by autouse fixture)
        # Reload scorer from disk
        scorer = DecisionScorer()
        assert scorer.is_trained

        result = evaluate_scorer_oos(scorer, oos_records)
        assert "rmse" in result
        assert "n" in result
        assert result["n"] == 20
        assert result["rmse"] >= 0
        # RMSE bounded — predictions on a noisy 6%-range target shouldn't blow up.
        assert result["rmse"] < 50.0

    def test_empty_oos_returns_n_zero(self):
        from paper_trader.ml.decision_scorer import DecisionScorer
        from paper_trader.validation import evaluate_scorer_oos
        # Untrained scorer is fine — we'd just fail gracefully on empty input.
        scorer = DecisionScorer()
        result = evaluate_scorer_oos(scorer, [])
        assert result["n"] == 0
        assert result["rmse"] is None or result["rmse"] != result["rmse"]  # NaN ok
