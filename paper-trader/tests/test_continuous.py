"""Tests for run_continuous_backtests.py.

Covers the pure functions: window picking, history trimming, top-decision
appending, outcome computation, and the live-only filter for news context.
The cycle loop itself isn't exercised — it requires the BacktestEngine
which depends on yfinance.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestRun


# ─────────────────────── _pick_window ───────────────────────────

class TestPickWindow:
    def test_returns_two_dates(self):
        start, end = rcb._pick_window(seed=42)
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert start < end

    def test_duration_within_range(self):
        for seed in range(20):
            start, end = rcb._pick_window(seed=seed)
            days = (end - start).days
            # 1-5 years × 365 days/year
            assert rcb.MIN_WINDOW_YEARS * 365 <= days <= rcb.MAX_WINDOW_YEARS * 365

    def test_window_ends_before_buffer(self):
        # Critical invariant: end date must be at least WINDOW_END_BUFFER_DAYS
        # in the past so we never train on data with insufficient forward-return
        # ground truth.
        for seed in range(50):
            start, end = rcb._pick_window(seed=seed)
            days_back = (date.today() - end).days
            assert days_back >= rcb.WINDOW_END_BUFFER_DAYS, \
                f"seed={seed} end={end} is only {days_back}d before today"

    def test_window_starts_after_earliest(self):
        for seed in range(20):
            start, _ = rcb._pick_window(seed=seed)
            assert start >= rcb.EARLIEST_WINDOW_START

    def test_deterministic_for_same_seed(self):
        # Same seed → same window. Critical for reproducibility of historical runs.
        a = rcb._pick_window(seed=1234)
        b = rcb._pick_window(seed=1234)
        assert a == b

    def test_different_seeds_differ(self):
        windows = {rcb._pick_window(seed=s) for s in range(20)}
        # 20 seeds should give at least 5 distinct windows (probabilistically near-certain).
        assert len(windows) > 1


# ─────────────────────── _trim_history ───────────────────────────

def _make_engine_with_runs(tmp_path, n_runs):
    """Build a real BacktestStore in tmp_path with n_runs fake runs."""
    from paper_trader.backtest import BacktestStore
    db_path = tmp_path / "bt.db"
    store = BacktestStore(path=db_path)
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    for i in range(1, n_runs + 1):
        store.upsert_run(i, seed=i, status="complete", start=start, end=end)
        store.record_trade(i, "2025-01-01", "NVDA", "BUY", 1.0, 100.0, "test")
        store.record_decision(i, "2025-01-01",
                              {"action": "BUY", "ticker": "NVDA", "qty": 1.0,
                               "reasoning": "score=2.5 regime=bull"},
                              "FILLED", "ok", 0.0, 0.0, 0)
    engine = MagicMock()
    engine.store = store
    return engine


class TestTrimHistory:
    def test_no_op_when_below_threshold(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=5)
        deleted = rcb._trim_history(eng, keep=10)
        assert deleted == 0
        # All 5 still present.
        rows = eng.store.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        assert rows[0] == 5

    def test_trims_oldest_runs(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=20)
        deleted = rcb._trim_history(eng, keep=10)
        # Should drop runs 1..10, keeping 11..20.
        assert deleted == 10
        rows = eng.store.conn.execute(
            "SELECT run_id FROM backtest_runs ORDER BY run_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        assert ids == list(range(11, 21))

    def test_cascades_to_trades_and_decisions(self, tmp_path):
        """Critical: trades / decisions belonging to trimmed runs must also be deleted —
        otherwise the DB grows unbounded with orphaned rows."""
        eng = _make_engine_with_runs(tmp_path, n_runs=15)
        rcb._trim_history(eng, keep=5)
        n_trades = eng.store.conn.execute(
            "SELECT COUNT(*) FROM backtest_trades WHERE run_id <= 10"
        ).fetchone()[0]
        n_decs = eng.store.conn.execute(
            "SELECT COUNT(*) FROM backtest_decisions WHERE run_id <= 10"
        ).fetchone()[0]
        assert n_trades == 0
        assert n_decs == 0


# ─────────────────────── _append_top_decisions ───────────────────────────

class TestAppendTopDecisions:
    def test_writes_per_decision_line(self, tmp_path, monkeypatch):
        eng = _make_engine_with_runs(tmp_path, n_runs=3)
        jsonl_path = tmp_path / "winners.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

        runs = [
            BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=20.0),
            BacktestRun(run_id=2, seed=2, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=10.0),
            BacktestRun(run_id=3, seed=3, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=5.0),
        ]
        written = rcb._append_top_decisions(eng, runs, cycle=7)
        # Each fake run has exactly 1 BUY decision.
        assert written == 3
        # File must exist and contain valid JSON lines.
        lines = jsonl_path.read_text().splitlines()
        assert len(lines) == 3
        recs = [json.loads(l) for l in lines]
        assert all(r["cycle"] == 7 for r in recs)
        # Top-ranked run should have higher ai_score than bottom-ranked.
        rank1 = next(r for r in recs if r["run_id"] == 1)
        rank3 = next(r for r in recs if r["run_id"] == 3)
        assert rank1["ai_score"] > rank3["ai_score"]

    def test_append_not_overwrite(self, tmp_path, monkeypatch):
        """Old results must accumulate, not be clobbered — historical runs are
        irreplaceable training data."""
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        jsonl_path = tmp_path / "winners.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31", total_return_pct=5.0)]
        rcb._append_top_decisions(eng, runs, cycle=1)
        rcb._append_top_decisions(eng, runs, cycle=2)
        rcb._append_top_decisions(eng, runs, cycle=3)
        lines = jsonl_path.read_text().splitlines()
        # 1 decision × 3 cycles = 3 lines, all preserved.
        assert len(lines) == 3
        cycles = [json.loads(l)["cycle"] for l in lines]
        assert sorted(cycles) == [1, 2, 3]


# ─────────────────────── _compute_decision_outcomes ───────────────────────────

class TestComputeDecisionOutcomes:
    def test_empty_runs(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=0)
        # Trading days is empty here — should produce no outcomes.
        eng.prices = MagicMock()
        eng.prices.trading_days = []
        outs = rcb._compute_decision_outcomes(eng, [])
        assert outs == []

    def test_skips_decisions_past_price_horizon(self, tmp_path, synthetic_prices):
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        # Insert a decision on the last trading day — its 5d forward window
        # extends past available data, so it must be skipped (not silently zeroed).
        last_day = synthetic_prices.trading_days[-1].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "reasoning) VALUES (?, ?, ?, ?, ?)",
            (1, last_day, "BUY", "NVDA", "score=2.5 regime=bull"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        # Last-day decision must be dropped (target idx >= len(trading_days)).
        # Initial fixture also had a BUY on 2025-01-01 — that one has 5d future
        # and a valid price for NVDA (from synthetic_prices), so it survives.
        for o in outs:
            assert o["sim_date"] != last_day


# ─────────────────────── _parse_published_date ───────────────────────────

class TestParsePublishedDate:
    def test_iso_string(self):
        assert rcb._parse_published_date("2025-03-15") == date(2025, 3, 15)

    def test_iso_with_time(self):
        assert rcb._parse_published_date("2025-03-15T10:30:00Z") == date(2025, 3, 15)

    def test_rfc822(self):
        assert rcb._parse_published_date("Wed, 14 May 2025 12:00:00 +0000") == date(2025, 5, 14)

    def test_none(self):
        assert rcb._parse_published_date(None) is None
        assert rcb._parse_published_date("") is None

    def test_garbage_returns_none(self):
        # Critical: garbage timestamps must not crash, must return None — caller
        # treats None as "don't apply date filter".
        assert rcb._parse_published_date("not a date") is None
        assert rcb._parse_published_date("xxxxxxx") is None


# ─────────────────────── _query_news_context ───────────────────────────

class TestQueryNewsContext:
    def test_filters_backtest_synthetic_articles(self, tmp_path, empty_articles_db,
                                                  monkeypatch):
        """The live-only filter is load-bearing. Backtest-injected articles must
        never leak into the LLM annotation news context — that's training
        contamination."""
        conn = sqlite3.connect(str(empty_articles_db))
        # Insert one real article and one backtest-injected article matching same ticker.
        conn.execute(
            "INSERT INTO articles (id, url, title, source, published, ai_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("real1", "https://real.com/x", "NVDA beats earnings",
             "reuters", "2025-05-01", 4.0),
        )
        conn.execute(
            "INSERT INTO articles (id, url, title, source, published, ai_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bt1", "backtest://run_5/2025-05-01/BUY/NVDA",
             "NVDA backtest decision", "backtest_run_5", "2025-05-01", 5.0),
        )
        conn.commit()
        conn.close()

        # Point ROOT to a parent containing digital-intern/data/articles.db
        fake_di = tmp_path / "digital-intern" / "data"
        fake_di.mkdir(parents=True)
        # Move the articles.db to where _query_news_context expects.
        import shutil
        shutil.copy(empty_articles_db, fake_di / "articles.db")

        # Monkeypatch ROOT so ROOT.parent / "digital-intern" / ... resolves correctly.
        # ROOT.parent must equal tmp_path. So ROOT = tmp_path / "anything".
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)

        titles = rcb._query_news_context("NVDA", "2025-05-02", n=5)
        # Real article should appear; backtest-injected one must NOT.
        assert any("beats earnings" in t for t in titles)
        assert not any("backtest decision" in t for t in titles)

    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        # No DB file at all — must return [] without crashing.
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)
        assert rcb._query_news_context("NVDA", "2025-05-01") == []

    def test_invalid_date_returns_empty(self, tmp_path, monkeypatch):
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)
        assert rcb._query_news_context("NVDA", "not-a-date") == []


# ─────────────────────── _train_decision_scorer wrapper ──────────────────

class TestTrainDecisionScorer:
    def test_no_records_returns_message(self):
        assert "no outcome records" in rcb._train_decision_scorer([])

    def test_insufficient_data_status(self):
        status = rcb._train_decision_scorer([{"ticker": "NVDA", "sim_date": "2025-01-01",
                                              "action": "BUY", "forward_return_5d": 1.0}])
        # 1 record → insufficient_after_dedup
        assert "insufficient" in status

    def test_temporal_split_reports_oos_rmse(self):
        # Happy path: enough distinct (ticker, sim_date, action) keys to clear
        # train_scorer's >=30 dedup gate, plus enough range that the temporal
        # 80/20 split leaves a non-empty OOS set the scorer is evaluated on.
        import random as _rnd
        rng = _rnd.Random(11)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })
        status = rcb._train_decision_scorer(records)
        # Status string should report both train and OOS metrics.
        assert "train_n=" in status
        assert "oos_n=" in status
        assert "oos_rmse=" in status
        # OOS holdout must be non-empty (~20% of 80)
        assert "oos_n=0" not in status

    def test_oos_eval_failure_does_not_mask_successful_train(self, monkeypatch):
        """A post-training OOS-eval crash must NOT be reported as a training
        failure.

        ``train_scorer`` pickles the model to ``SCORER_PATH`` and returns
        ``status="ok"`` *before* the OOS diagnostic runs. If the OOS step then
        raises (transient pickle/IO race, validation-module change, …) the
        scorer is in fact trained and gets deployed (the singleton is reset and
        reloads it next cycle) — but a single broad ``except`` around both the
        train call and the diagnostic would surface ``scorer err`` to the
        operator-facing log/Discord, falsely signalling a broken scorer and the
        gate never engaging. The status must stay truthful: training succeeded.
        """
        import random as _rnd
        import paper_trader.validation as _val

        rng = _rnd.Random(13)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated OOS-eval crash after pickling")

        monkeypatch.setattr(_val, "evaluate_scorer_oos", _boom)

        status = rcb._train_decision_scorer(records)
        # Training succeeded and was pickled — the status must reflect that,
        # not a generic "scorer err".
        assert not status.startswith("scorer err"), status
        assert "scorer ok" in status, status
        assert "train_n=" in status, status
        # OOS metric degrades gracefully to n/a rather than killing the report.
        assert "oos_rmse=n/a" in status, status
        # And the model the next cycle will load is genuinely trained.
        from paper_trader.ml.decision_scorer import DecisionScorer
        assert DecisionScorer().is_trained is True

    def test_temporal_split_failure_still_trains(self, monkeypatch):
        """A *pre-training* split failure must NOT skip training.

        The temporal holdout (split_outcomes_temporal) is a diagnostic
        refinement, not the essential operation. Before the fix it sat in the
        same try/except as ``train_scorer``, so a split crash (or an
        unavailable validation module) returned ``scorer err:`` and the model
        was never pickled — silently freezing the per-cycle retrain invariant
        (CLAUDE.md §6) and the conviction gate (#5). After the fix the split
        failure degrades to "train on all records, no OOS" and the scorer is
        still retrained and deployed.
        """
        import random as _rnd
        import paper_trader.validation as _val

        rng = _rnd.Random(17)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated split crash before training")

        monkeypatch.setattr(_val, "split_outcomes_temporal", _boom)

        status = rcb._train_decision_scorer(records)
        # Training proceeded despite the split crash — status is truthful.
        assert not status.startswith("scorer err"), status
        assert "scorer ok" in status, status
        # All 80 records used for training (no holdout carved out).
        assert "train_n=80" in status, status
        assert "oos_n=0" in status, status
        # The model the next cycle reloads is genuinely trained and pickled.
        from paper_trader.ml.decision_scorer import DecisionScorer
        assert DecisionScorer().is_trained is True
