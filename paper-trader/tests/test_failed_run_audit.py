"""Tests for paper_trader.ml.failed_run_audit.

Pins the full verdict ladder and the bias-shift math on synthetic
backtest.db fixtures. Every test is offline (no yfinance / no real DB) —
the conftest fixture redirects bt.BACKTEST_DB into a tmp.

These tests catch real bug classes the analyzer is designed for:
  * misclassifying an OOM-reaped row as GENUINE_FAILURE (or vice versa)
  * forgetting to read `notes` for the `[reaped]` marker
  * a median that's silently 0 when no input is finite
  * a bias_shift that swallows the dashboard's actual overstatement
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import paper_trader.backtest as bt
from paper_trader.ml import failed_run_audit as fra


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Build a synthetic backtest.db with the given rows.

    Each row dict has keys: run_id, status, n_trades (default 0),
    vs_spy_pct (default None), total_return_pct (default 0),
    notes (default ''), completed_at (default None — i.e. the
    finalize_run-never-ran state; pass any non-None string to simulate
    a row that DID get finalized so its vs_spy_pct is a real value).
    """
    db = tmp_path / "backtest.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE backtest_runs ("
        "  run_id INTEGER PRIMARY KEY, "
        "  seed INTEGER, "
        "  start_date TEXT, "
        "  end_date TEXT, "
        "  start_value REAL, "
        "  final_value REAL, "
        "  total_return_pct REAL, "
        "  spy_return_pct REAL, "
        "  vs_spy_pct REAL, "
        "  n_trades INTEGER, "
        "  n_decisions INTEGER, "
        "  status TEXT, "
        "  started_at TEXT, "
        "  completed_at TEXT, "
        "  equity_curve_json TEXT, "
        "  notes TEXT"
        ")"
    )
    for r in rows:
        # status='complete' implies finalize_run executed, so default
        # completed_at to a sentinel when the caller didn't override.
        # status='failed' defaults to None unless overridden.
        default_completed = ("2026-01-01T01:00:00Z"
                             if r.get("status") == "complete" else None)
        conn.execute(
            "INSERT INTO backtest_runs (run_id, status, n_trades, "
            "vs_spy_pct, total_return_pct, notes, completed_at, seed, "
            "start_date, end_date, start_value, final_value, "
            "spy_return_pct, n_decisions, started_at, equity_curve_json) "
            "VALUES (?,?,?,?,?,?,?,1,'2025-01-01','2026-01-01',1000,1000,"
            "0,0,'2026-01-01T00:00:00Z','[]')",
            (r["run_id"], r["status"],
             r.get("n_trades", 0),
             r.get("vs_spy_pct"),
             r.get("total_return_pct", 0.0),
             r.get("notes", ""),
             r.get("completed_at", default_completed)),
        )
    conn.commit()
    conn.close()
    return db


class TestClassifier:
    def test_reaped_marker_wins_immediately(self):
        """A row with `[reaped]` in notes is OOM-reaped regardless of
        trade count — covers reaped rows that had only a few trades."""
        assert fra._classify_failed_row({
            "n_trades": 0,
            "vs_spy_pct": None,
            "notes": " [reaped: orphaned running row]"
        }) == "LIKELY_OOM_REAPED"

    def test_high_trade_count_is_oom_reaped(self):
        """The pass #21 footprint: 1000+ trades, no notes marker —
        productive run that was killed. vs_spy_pct alone is NOT a
        reliable signal (schema default 0.0 fires false-positive), so
        the classifier leans on trade count."""
        assert fra._classify_failed_row({
            "n_trades": 1000,
            "vs_spy_pct": None,
            "notes": ""
        }) == "LIKELY_OOM_REAPED"

    def test_high_trade_count_with_placeholder_vs_spy_is_oom_reaped(self):
        """Live state: failed rows carry vs_spy_pct=0.0 from the schema
        default, NOT from a real benchmark. The classifier must still
        flag them as OOM-reaped via trade count alone."""
        assert fra._classify_failed_row({
            "n_trades": 1000,
            "vs_spy_pct": 0.0,  # placeholder, not real
            "completed_at": None,
            "notes": ""
        }) == "LIKELY_OOM_REAPED"

    def test_genuine_failure_no_trades(self):
        """Engine crash before any decisions executed."""
        assert fra._classify_failed_row({
            "n_trades": 0,
            "vs_spy_pct": None,
            "notes": ""
        }) == "GENUINE_FAILURE"

    def test_just_below_min_trades_threshold(self):
        """A row with MIN_TRADES_FOR_REAL_RUN - 1 trades is NOT yet
        considered OOM-reaped — locks the threshold boundary."""
        assert fra._classify_failed_row({
            "n_trades": fra.MIN_TRADES_FOR_REAL_RUN - 1,
            "vs_spy_pct": None,
            "notes": ""
        }) == "GENUINE_FAILURE"

    def test_at_min_trades_threshold(self):
        """At MIN_TRADES_FOR_REAL_RUN, classification flips to OOM-reaped."""
        assert fra._classify_failed_row({
            "n_trades": fra.MIN_TRADES_FOR_REAL_RUN,
            "vs_spy_pct": None,
            "notes": ""
        }) == "LIKELY_OOM_REAPED"

    def test_non_int_trades_defaults_to_zero(self):
        """Garbage n_trades shouldn't crash — defaults to 0 → GENUINE."""
        assert fra._classify_failed_row({
            "n_trades": "junk",
            "vs_spy_pct": None,
            "notes": ""
        }) == "GENUINE_FAILURE"


class TestHasRealVsSpy:
    def test_completed_at_none_is_placeholder(self):
        """completed_at IS NULL ⇒ finalize_run never ran ⇒ vs_spy_pct
        is the schema's NOT NULL DEFAULT 0 placeholder, NOT a real value."""
        assert fra._has_real_vs_spy({
            "completed_at": None,
            "vs_spy_pct": 0.0,  # placeholder
        }) is False

    def test_completed_at_set_is_real(self):
        """A row that DID complete (however briefly) carries a real
        finalize_run-written vs_spy_pct, even if it's 0.0."""
        assert fra._has_real_vs_spy({
            "completed_at": "2026-01-01T01:00:00Z",
            "vs_spy_pct": 0.0,  # honest 0% alpha
        }) is True

    def test_completed_at_set_with_non_zero_is_real(self):
        assert fra._has_real_vs_spy({
            "completed_at": "2026-01-01T01:00:00Z",
            "vs_spy_pct": 42.5,
        }) is True

    def test_completed_at_set_but_vs_spy_null_is_not_real(self):
        """Defensive: a completed_at set but vs_spy_pct=None (shouldn't
        happen with current schema, but if it ever does, treat as
        unreliable)."""
        assert fra._has_real_vs_spy({
            "completed_at": "2026-01-01T01:00:00Z",
            "vs_spy_pct": None,
        }) is False


class TestMedian:
    def test_odd_count(self):
        assert fra._median([1.0, 3.0, 5.0]) == 3.0

    def test_even_count(self):
        assert fra._median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_unsorted_input(self):
        assert fra._median([5.0, 1.0, 3.0]) == 3.0

    def test_empty(self):
        assert fra._median([]) is None


class TestAnalyze:
    def test_missing_db_yields_insufficient_data(self, tmp_path):
        nonexistent = tmp_path / "nope.db"
        out = fra.analyze(nonexistent)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_failed"] == 0

    def test_db_with_no_failed_rows(self, tmp_path):
        db = _make_db(tmp_path, [
            {"run_id": 1, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 12.5},
            {"run_id": 2, "status": "complete",
             "n_trades": 800, "vs_spy_pct": -5.0},
        ])
        out = fra.analyze(db)
        assert out["verdict"] == "NO_FAILED_RUNS"
        assert out["n_failed"] == 0
        assert out["n_oom_reaped"] == 0
        # complete_median_vs_spy is NOT reported on NO_FAILED_RUNS
        # (early return) — that's fine, no failed slice to compare with

    def test_all_genuine_failure(self, tmp_path):
        db = _make_db(tmp_path, [
            {"run_id": 1, "status": "failed",
             "n_trades": 0, "vs_spy_pct": None, "notes": ""},
            {"run_id": 2, "status": "failed",
             "n_trades": 0, "vs_spy_pct": None, "notes": ""},
            {"run_id": 3, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 20.0},
        ])
        out = fra.analyze(db)
        assert out["verdict"] == "ALL_GENUINE_FAILURE"
        assert out["n_failed"] == 2
        assert out["n_oom_reaped"] == 0
        assert out["n_genuine"] == 2
        assert out["oom_reaped_pct"] == 0.0

    def test_mostly_oom_reaped_pass21_footprint_realistic(self, tmp_path):
        """Reproduces the pass #21 footprint REALISTICALLY: the orphan
        reaper marks status='failed' on rows whose finalize_run never
        ran, so vs_spy_pct is the schema's NOT NULL DEFAULT 0 placeholder
        (NOT a real value). The classifier should still flag MOSTLY_OOM_REAPED
        via trade count, but hidden_median_vs_spy and bias_shift_pct
        should be None — the realized alpha of the hidden slice is
        UNKNOWN because finalize_run never wrote it.

        Live evidence: 26/26 production failed rows match this footprint.
        """
        db = _make_db(tmp_path, [
            # 5 OOM-reaped: 1000+ trades, never finalized (completed_at=None)
            # so vs_spy_pct=0.0 is the schema default, not real alpha.
            {"run_id": 5981, "status": "failed", "n_trades": 1500,
             "vs_spy_pct": 0.0, "completed_at": None, "notes": ""},
            {"run_id": 5982, "status": "failed", "n_trades": 1200,
             "vs_spy_pct": 0.0, "completed_at": None, "notes": ""},
            {"run_id": 5983, "status": "failed", "n_trades": 1100,
             "vs_spy_pct": 0.0, "completed_at": None, "notes": ""},
            {"run_id": 5984, "status": "failed", "n_trades": 1050,
             "vs_spy_pct": 0.0, "completed_at": None, "notes": ""},
            {"run_id": 5985, "status": "failed", "n_trades": 1300,
             "vs_spy_pct": 0.0, "completed_at": None, "notes": ""},
            # 3 complete runs with median 20%.
            {"run_id": 10, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 10.0},
            {"run_id": 11, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 20.0},
            {"run_id": 12, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 30.0},
        ])
        out = fra.analyze(db)
        assert out["verdict"] == "MOSTLY_OOM_REAPED"
        assert out["n_failed"] == 5
        assert out["n_oom_reaped"] == 5
        assert out["n_genuine"] == 0
        assert out["oom_reaped_pct"] == 1.0
        # The realistic-footprint critical assertion: none of these
        # carry real vs_spy_pct, so hidden_median and bias_shift are
        # honestly None.
        assert out["n_oom_with_real_vs_spy"] == 0
        assert out["hidden_median_vs_spy"] is None
        assert out["hidden_max_vs_spy"] is None
        assert out["hidden_min_vs_spy"] is None
        assert out["bias_shift_pct"] is None
        # Complete median is still reported for context.
        assert out["complete_median_vs_spy"] == 20.0
        # And the suspect run_ids list is populated.
        assert out["suspect_run_ids"] == [5981, 5982, 5983, 5984, 5985]
        # Hint explicitly mentions the placeholder situation.
        assert "schema's vs_spy_pct=0.0 placeholder" in out["hint"]

    def test_mostly_oom_reaped_with_real_vs_spy_hypothetical(self, tmp_path):
        """HYPOTHETICAL: if reaped rows DID carry real finalize_run-written
        vs_spy_pct (impossible under current reaper logic which only fires
        on status='running' — but kept as a future-proofing test in case
        the reap path is extended), the bias_shift computation should
        actually run. completed_at populated ⇒ vs_spy_pct is real."""
        db = _make_db(tmp_path, [
            # Hypothetical: reaped but finalized first.
            {"run_id": 5981, "status": "failed", "n_trades": 1500,
             "vs_spy_pct": 101.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 5982, "status": "failed", "n_trades": 1200,
             "vs_spy_pct": -4.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 5983, "status": "failed", "n_trades": 1100,
             "vs_spy_pct": 47.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 5984, "status": "failed", "n_trades": 1050,
             "vs_spy_pct": 33.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 5985, "status": "failed", "n_trades": 1300,
             "vs_spy_pct": 31.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            # Complete runs with median 20%.
            {"run_id": 10, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 10.0},
            {"run_id": 11, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 20.0},
            {"run_id": 12, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 30.0},
        ])
        out = fra.analyze(db)
        assert out["verdict"] == "MOSTLY_OOM_REAPED"
        assert out["n_oom_with_real_vs_spy"] == 5
        # Hidden vs_spy distribution: [-4, 31, 33, 47, 101] → median 33
        assert out["hidden_median_vs_spy"] == 33.0
        assert out["hidden_max_vs_spy"] == 101.0
        assert out["hidden_min_vs_spy"] == -4.0
        # Complete median = 20 (sorted [10, 20, 30]).
        assert out["complete_median_vs_spy"] == 20.0
        # Merged [-4, 10, 20, 30, 31, 33, 47, 101] median = (30+31)/2 = 30.5
        # Shift = 30.5 - 20.0 = +10.5pp.
        assert out["bias_shift_pct"] == pytest.approx(10.5, abs=1e-3)

    def test_mixed_reaped_and_failure(self, tmp_path):
        """About half OOM-reaped, half genuine — verdict
        MIXED_REAPED_AND_FAILURE."""
        rows = []
        # 3 genuine failures (no trades, no completed_at)
        for i in range(1, 4):
            rows.append({
                "run_id": i, "status": "failed",
                "n_trades": 0, "vs_spy_pct": None,
                "completed_at": None, "notes": ""
            })
        # 3 OOM-reaped (via [reaped] marker); placeholder vs_spy
        for i in range(4, 7):
            rows.append({
                "run_id": i, "status": "failed",
                "n_trades": 5, "vs_spy_pct": 0.0,
                "completed_at": None,
                "notes": " [reaped: orphaned running row]"
            })
        # 5 complete with vs_spy [0, 10, 20, 30, 40]
        for i, v in enumerate([0.0, 10.0, 20.0, 30.0, 40.0], start=10):
            rows.append({
                "run_id": i, "status": "complete",
                "n_trades": 500, "vs_spy_pct": v
            })
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["verdict"] == "MIXED_REAPED_AND_FAILURE"
        assert out["n_failed"] == 6
        assert out["n_oom_reaped"] == 3
        assert out["n_genuine"] == 3
        assert out["oom_reaped_pct"] == 0.5
        # Reaped rows carry placeholder vs_spy ⇒ no bias-shift computable.
        assert out["bias_shift_pct"] is None

    def test_oom_reaped_pct_just_above_low_threshold(self, tmp_path):
        """20% OOM-reaped is at the boundary — verdict should be
        MIXED_REAPED_AND_FAILURE since the bucket is [LOW, HIGH)."""
        rows = []
        # 1 OOM-reaped, 4 genuine = 20% reaped
        rows.append({
            "run_id": 1, "status": "failed",
            "n_trades": 500, "vs_spy_pct": None,
            "completed_at": None, "notes": ""
        })
        for i in range(2, 6):
            rows.append({
                "run_id": i, "status": "failed",
                "n_trades": 0, "vs_spy_pct": None,
                "completed_at": None, "notes": ""
            })
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["oom_reaped_pct"] == 0.2
        # At exactly 0.20 (== LOW threshold), MIXED bucket fires.
        assert out["verdict"] == "MIXED_REAPED_AND_FAILURE"

    def test_oom_reaped_pct_below_low_threshold_with_real_vs_spy(self, tmp_path):
        """10% OOM-reaped is below LOW — verdict ALL_GENUINE_FAILURE
        (the middle bucket honest about most failures being real).
        With a real vs_spy on the one reaped row, hidden_median is
        reported."""
        rows = []
        # 1 OOM-reaped with REAL finalize_run-written vs_spy,
        # 9 genuine = 10% reaped
        rows.append({
            "run_id": 1, "status": "failed",
            "n_trades": 500, "vs_spy_pct": 50.0,
            "completed_at": "2026-01-01T01:00:00Z", "notes": ""
        })
        for i in range(2, 11):
            rows.append({
                "run_id": i, "status": "failed",
                "n_trades": 0, "vs_spy_pct": None,
                "completed_at": None, "notes": ""
            })
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["oom_reaped_pct"] == 0.1
        # Below LOW threshold: ALL_GENUINE_FAILURE honest bucket.
        assert out["verdict"] == "ALL_GENUINE_FAILURE"
        # But the hidden slice IS reported — same dict carries the truth.
        assert out["n_oom_reaped"] == 1
        assert out["hidden_median_vs_spy"] == 50.0

    def test_bias_shift_negative_when_hidden_underperforms(self, tmp_path):
        """If OOM-reaped runs UNDERPERFORM the complete distribution,
        the dashboard OVERSTATES alpha — bias_shift is negative.

        Uses completed_at on the reaped rows to simulate the hypothetical
        scenario where vs_spy is real (the realistic OOM-reaped case has
        placeholder vs_spy and bias_shift is not computable; see
        test_mostly_oom_reaped_pass21_footprint_realistic)."""
        rows = [
            # 3 OOM-reaped at -20%, -10%, -5% — with completed_at set,
            # vs_spy_pct is treated as real.
            {"run_id": 1, "status": "failed",
             "n_trades": 200, "vs_spy_pct": -20.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 2, "status": "failed",
             "n_trades": 200, "vs_spy_pct": -10.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            {"run_id": 3, "status": "failed",
             "n_trades": 200, "vs_spy_pct": -5.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
            # 3 complete at 50%, 60%, 70%
            {"run_id": 10, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 50.0},
            {"run_id": 11, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 60.0},
            {"run_id": 12, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 70.0},
        ]
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["verdict"] == "MOSTLY_OOM_REAPED"
        assert out["complete_median_vs_spy"] == 60.0
        # Merged: [-20, -10, -5, 50, 60, 70] sorted → median (-5+50)/2 = 22.5
        # Shift = 22.5 - 60 = -37.5 — dashboard overstates by 37.5pp.
        assert out["bias_shift_pct"] == pytest.approx(-37.5, abs=1e-3)

    def test_corrupt_db_yields_insufficient_data(self, tmp_path):
        bad = tmp_path / "bad.db"
        bad.write_text("not a sqlite database")
        out = fra.analyze(bad)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_failed"] == 0

    def test_suspect_run_ids_capped_at_50(self, tmp_path):
        """A long failed-run list should cap suspect_run_ids at 50."""
        rows = []
        # 60 OOM-reaped, all with [reaped] markers
        for i in range(1, 61):
            rows.append({
                "run_id": i, "status": "failed",
                "n_trades": 200, "vs_spy_pct": None,
                "completed_at": None, "notes": " [reaped]"
            })
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["verdict"] == "MOSTLY_OOM_REAPED"
        assert out["n_oom_reaped"] == 60
        assert len(out["suspect_run_ids"]) == 50
        # First 50 by run_id ASC.
        assert out["suspect_run_ids"] == list(range(1, 51))

    def test_no_complete_no_bias_shift(self, tmp_path):
        """If there are no complete rows, bias_shift_pct cannot be
        computed — stays None (no false 0). Hidden_median is still
        reported when the reaped row carries a REAL vs_spy."""
        rows = [
            {"run_id": 1, "status": "failed",
             "n_trades": 500, "vs_spy_pct": 25.0,
             "completed_at": "2026-01-01T01:00:00Z", "notes": ""},
        ]
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["bias_shift_pct"] is None
        assert out["complete_median_vs_spy"] is None
        # Hidden_median is reported because vs_spy is real (completed_at set).
        assert out["hidden_median_vs_spy"] == 25.0

    def test_live_production_footprint_placeholder_only(self, tmp_path):
        """Live state snapshot: 26 failed rows all carrying the schema's
        vs_spy_pct=0.0 placeholder because finalize_run never executed.

        Pins the analyzer's correct behaviour: VERDICT='MOSTLY_OOM_REAPED'
        AND n_oom_with_real_vs_spy=0 AND hidden_median_vs_spy=None AND
        bias_shift_pct=None. The hint must explicitly call out the
        placeholder situation so an operator knows the dashboard's
        complete-only aggregate's bias is UNKNOWN, not zero.
        """
        rows = []
        for i in range(1, 27):
            rows.append({
                "run_id": 5980 + i, "status": "failed",
                "n_trades": 1000, "vs_spy_pct": 0.0,  # placeholder
                "completed_at": None, "notes": ""
            })
        # And a few complete rows so the dashboard has something to median.
        for i, v in enumerate([35.0, 40.0, 45.0, 50.0], start=100):
            rows.append({
                "run_id": i, "status": "complete",
                "n_trades": 500, "vs_spy_pct": v
            })
        db = _make_db(tmp_path, rows)
        out = fra.analyze(db)
        assert out["verdict"] == "MOSTLY_OOM_REAPED"
        assert out["n_oom_reaped"] == 26
        assert out["n_oom_with_real_vs_spy"] == 0
        assert out["hidden_median_vs_spy"] is None
        assert out["bias_shift_pct"] is None
        # Hint mentions the placeholder situation.
        assert "schema's vs_spy_pct=0.0 placeholder" in out["hint"]
        # Complete median is reported for context.
        assert out["complete_median_vs_spy"] == 42.5  # median of [35,40,45,50]


class TestIsFailedRunsHidden:
    def test_returns_true_on_mostly_oom_reaped(self, tmp_path, monkeypatch):
        rows = [
            {"run_id": 1, "status": "failed",
             "n_trades": 200, "vs_spy_pct": None,
             "completed_at": None, "notes": ""},
            {"run_id": 2, "status": "failed",
             "n_trades": 200, "vs_spy_pct": None,
             "completed_at": None, "notes": ""},
        ]
        db = _make_db(tmp_path, rows)
        assert fra.is_failed_runs_hidden(db) is True

    def test_returns_false_on_all_genuine(self, tmp_path):
        rows = [
            {"run_id": 1, "status": "failed",
             "n_trades": 0, "vs_spy_pct": None, "notes": ""},
        ]
        db = _make_db(tmp_path, rows)
        assert fra.is_failed_runs_hidden(db) is False

    def test_returns_false_on_no_failed_runs(self, tmp_path):
        rows = [
            {"run_id": 1, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 10.0},
        ]
        db = _make_db(tmp_path, rows)
        assert fra.is_failed_runs_hidden(db) is False

    def test_returns_none_on_insufficient_data(self, tmp_path):
        nonexistent = tmp_path / "nope.db"
        assert fra.is_failed_runs_hidden(nonexistent) is None


class TestCli:
    def test_cli_exits_2_on_adverse_verdict(self, tmp_path, capsys):
        rows = [
            {"run_id": 1, "status": "failed",
             "n_trades": 200, "vs_spy_pct": None,
             "completed_at": None, "notes": ""},
            {"run_id": 2, "status": "failed",
             "n_trades": 200, "vs_spy_pct": None,
             "completed_at": None, "notes": ""},
        ]
        db = _make_db(tmp_path, rows)
        rc = fra._cli(["--db", str(db)])
        assert rc == 2
        captured = capsys.readouterr()
        assert "MOSTLY_OOM_REAPED" in captured.out

    def test_cli_exits_0_on_no_failed_runs(self, tmp_path, capsys):
        rows = [
            {"run_id": 1, "status": "complete",
             "n_trades": 500, "vs_spy_pct": 10.0},
        ]
        db = _make_db(tmp_path, rows)
        rc = fra._cli(["--db", str(db)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "NO_FAILED_RUNS" in captured.out

    def test_cli_json_output(self, tmp_path, capsys):
        rows = [
            {"run_id": 1, "status": "failed",
             "n_trades": 0, "vs_spy_pct": None, "notes": ""},
        ]
        db = _make_db(tmp_path, rows)
        rc = fra._cli(["--json", "--db", str(db)])
        assert rc == 0
        captured = capsys.readouterr()
        import json
        parsed = json.loads(captured.out)
        assert parsed["verdict"] == "ALL_GENUINE_FAILURE"
        assert parsed["n_failed"] == 1
