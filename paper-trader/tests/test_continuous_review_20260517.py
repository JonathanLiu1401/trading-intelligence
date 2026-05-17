"""Regression locks (2026-05-17 review) for run_continuous_backtests.py:

1. ``_reap_orphaned_runs`` — a ``running`` row older than the age guard is
   failed; a fresh ``running`` row and a ``complete`` row are untouched
   (the CLAUDE.md §11 "Backtest dashboard shows running forever" fix: a
   hard-killed run thread never reaches finalize_run *or* the
   ``upsert_run("failed")`` fallback, so the row stays running forever).

2. ``_oos_rank_metrics`` — out-of-sample directional accuracy and tie-aware
   rank-IC: exact values on deterministic fixtures, the SELL sign-flip
   convention (mirrors ``validation.evaluate_scorer_oos``), the
   constant-predictor anti-fabrication property (proves the reused
   ``calibration._spearman``, not a rank-skill-fabricating argsort), the
   never-raises contract, and the ``_train_decision_scorer`` status wiring
   (the new tokens coexist with the pre-existing ``oos_rmse=`` token that
   ``test_continuous.TestTrainDecisionScorer`` locks verbatim).

Offline & deterministic — conftest redirects BACKTEST_DB / SCORER_PATH to
tmp; the OOS metric uses a fake scorer with full control over (pred,
realized) pairs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import paper_trader.backtest as bt
import run_continuous_backtests as rcb


# ───────────────────────── orphaned-run reaper ─────────────────────────────

class TestReapOrphanedRuns:
    def _seed(self, store, run_id, status, started_at, completed_at=None):
        with store._lock:
            store.conn.execute(
                "INSERT INTO backtest_runs (run_id,seed,start_date,end_date,"
                "start_value,status,started_at,completed_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (run_id, 0, "2024-01-01", "2024-06-01", 1000.0,
                 status, started_at, completed_at),
            )
            store.conn.commit()

    def test_stale_running_failed_fresh_and_complete_untouched(self):
        store = bt.BacktestStore()   # conftest points BACKTEST_DB at tmp
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(hours=30)).isoformat()
        fresh = (now - timedelta(hours=1)).isoformat()
        self._seed(store, 1, "running", stale)
        self._seed(store, 2, "running", fresh)
        self._seed(store, 3, "complete", stale, stale)

        n = rcb._reap_orphaned_runs(max_age_hours=6.0)
        assert n == 1

        ro = sqlite3.connect(f"file:{bt.BACKTEST_DB}?mode=ro", uri=True)
        rows = dict(ro.execute(
            "SELECT run_id, status FROM backtest_runs").fetchall())
        ro.close()
        assert rows[1] == "failed"     # stale running → failed
        assert rows[2] == "running"    # fresh running untouched
        assert rows[3] == "complete"   # complete untouched

    def test_no_db_returns_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "BACKTEST_DB", tmp_path / "nonexistent.db")
        assert rcb._reap_orphaned_runs() == 0

    def test_no_running_rows_is_a_noop(self):
        store = bt.BacktestStore()
        self._seed(store, 5, "complete",
                   datetime.now(timezone.utc).isoformat())
        assert rcb._reap_orphaned_runs(max_age_hours=0.0) == 0


# ───────────────────────── OOS rank metrics ────────────────────────────────

class _FakeScorer:
    """Returns the record's ml_score verbatim as the prediction, so a test
    fully controls the (pred, realized) pairs the metric sees."""
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None, news_urgency=None,
                news_article_count=None):
        return float(ml_score)


def _rec(pred, realized, action="BUY"):
    return {"ml_score": pred, "forward_return_5d": realized,
            "action": action, "ticker": "X", "regime_mult": 1.0}


class TestOosRankMetrics:
    def test_perfect_rank_and_direction(self):
        recs = [_rec(v, v) for v in range(1, 11)]   # pred == realized
        m = rcb._oos_rank_metrics(_FakeScorer(), recs)
        assert m["n"] == 10
        assert m["rank_ic"] == 1.0
        assert m["dir_acc"] == 1.0

    def test_constant_predictor_cannot_fabricate_rank_skill(self):
        # Constant pred (+1), realized mostly +1 with two -1: dir_acc=0.8,
        # but a constant predictor has NO rank skill → rank_ic must be 0.0
        # (proves the reused tie-aware calibration._spearman, not argsort).
        realized = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]
        recs = [_rec(1.0, r) for r in realized]
        m = rcb._oos_rank_metrics(_FakeScorer(), recs)
        assert m["dir_acc"] == 0.8
        assert m["rank_ic"] == 0.0

    def test_sell_sign_flip_matches_evaluate_scorer_oos_convention(self):
        # SELL with pred>0 and forward_return_5d<0: after the flip the
        # realized target is +, so sign(pred)==sign(realized) → a hit.
        recs = [_rec(5.0, -3.0, action="SELL"),
                _rec(4.0, -2.0, action="SELL"),
                _rec(3.0, -1.0, action="SELL")]
        m = rcb._oos_rank_metrics(_FakeScorer(), recs)
        assert m["dir_acc"] == 1.0

    def test_untrained_scorer_returns_none_metrics(self):
        class _Untrained:
            is_trained = False

            def predict(self, **kw):
                return 0.0

        m = rcb._oos_rank_metrics(_Untrained(), [_rec(1.0, 1.0)])
        assert m == {"dir_acc": None, "rank_ic": None, "n": 0}

    def test_never_raises_on_predict_failure(self):
        class _Broken:
            is_trained = True

            def predict(self, **kw):
                raise RuntimeError("boom")

        m = rcb._oos_rank_metrics(_Broken(), [_rec(1.0, 1.0), _rec(2.0, 2.0)])
        # Every prediction failed → no usable pairs, but no exception.
        assert m["n"] == 0
        assert m["dir_acc"] is None and m["rank_ic"] is None

    def test_zero_realized_pairs_excluded_from_dir_acc(self):
        # realized==0 carries no directional truth; rank_ic still computable.
        recs = [_rec(1.0, 0.0), _rec(2.0, 0.0), _rec(3.0, 0.0)]
        m = rcb._oos_rank_metrics(_FakeScorer(), recs)
        assert m["n"] == 3
        assert m["dir_acc"] is None        # no directional pairs

    def test_train_decision_scorer_status_reports_oos_dir_metrics(self):
        """End-to-end wiring: the status string carries the new tokens
        alongside the pre-existing oos_rmse= token (locked verbatim by
        test_continuous.TestTrainDecisionScorer)."""
        import random as _rnd
        rng = _rnd.Random(23)
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
        assert "oos_rmse=" in status          # pre-existing token preserved
        assert "oos_diracc=" in status
        assert "oos_ic=" in status
        assert "oos_diracc=n/a" not in status   # real value computed
