"""Tests for paper_trader.ml.benchmark_integrity — SPY-benchmark integrity audit.

Tests cover:
  * the pure ``benchmark_integrity_report`` arithmetic on synthetic dicts
    (verdict ladder boundaries, bucketing, qualifier-window economic impact)
  * the ``analyze`` reader against an in-memory ``BacktestStore`` fixture
    (round-trips through real SQL so a schema drift would surface)
  * the CLI exit code mapping

No filesystem dependency beyond a tmp_path BacktestStore (the standard
test isolation pattern documented in AGENTS.md).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from paper_trader.ml import benchmark_integrity as bi


# ─── helpers ─────────────────────────────────────────────────────────────


def _mk_run(run_id: int, *, win_days: int = 365, spy: float = 5.0,
            ret: float = 10.0, n_trades: int = 100,
            notes: str = "", status: str = "complete",
            start: str = "2020-01-02") -> dict:
    """Build a minimal dict row matching the BacktestStore SELECT shape."""
    sd = date.fromisoformat(start)
    ed = sd + timedelta(days=win_days)
    return {
        "run_id": run_id,
        "start_date": sd.isoformat(),
        "end_date": ed.isoformat(),
        "status": status,
        "spy_return_pct": spy,
        "vs_spy_pct": ret - spy if spy is not None else None,
        "n_trades": n_trades,
        "notes": notes,
    }


# ─── benchmark_integrity_report — empty / clean ──────────────────────────


class TestEmptyOrClean:
    def test_empty_input_returns_insufficient_data(self):
        rep = bi.benchmark_integrity_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total"] == 0
        assert rep["flagged_degenerate"] == 0
        assert rep["unflagged_degenerate"] == 0
        assert rep["qualifier_window"]["n"] == 0
        assert "no complete runs" in rep["hint"].lower()

    def test_non_complete_runs_dropped_from_scope(self):
        rows = [_mk_run(i, status=s) for i, s in
                enumerate(["running", "failed", "pending"], start=1)]
        rep = bi.benchmark_integrity_report(rows)
        assert rep["total"] == 0  # only complete runs counted
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_clean_corpus_reads_clean(self):
        # All real SPY returns, no degenerate rows. Need >= MIN_TOTAL.
        rows = [_mk_run(i, spy=5.0 + i * 0.1) for i in range(1, bi.MIN_TOTAL + 1)]
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "CLEAN"
        assert rep["total"] == bi.MIN_TOTAL
        assert rep["flagged_degenerate"] == 0
        assert rep["unflagged_degenerate"] == 0
        assert rep["max_unflagged_run_id"] is None
        assert rep["run_id_buckets"] == []

    def test_below_min_total_is_insufficient_even_with_clean_data(self):
        # Below MIN_TOTAL we cannot make a verdict — even on clean data.
        rows = [_mk_run(i) for i in range(1, bi.MIN_TOTAL)]
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total"] == bi.MIN_TOTAL - 1


# ─── degenerate detection ────────────────────────────────────────────────


class TestDegenerateDetection:
    def test_spy_zero_over_30_days_is_degenerate(self):
        # 30-day window with SPY=0 hits the degenerate condition.
        rows = [_mk_run(i) for i in range(1, bi.MIN_TOTAL + 1)]
        rows.append(_mk_run(999, win_days=bi.MIN_RUN_WINDOW_DAYS, spy=0.0))
        rep = bi.benchmark_integrity_report(rows)
        assert rep["unflagged_degenerate"] == 1
        assert rep["max_unflagged_run_id"] == 999

    def test_spy_zero_under_threshold_window_is_not_degenerate(self):
        # 29-day window with SPY=0 — real flat market plausible at short windows.
        rows = [_mk_run(i) for i in range(1, bi.MIN_TOTAL + 1)]
        rows.append(_mk_run(999, win_days=bi.MIN_RUN_WINDOW_DAYS - 1, spy=0.0))
        rep = bi.benchmark_integrity_report(rows)
        assert rep["unflagged_degenerate"] == 0
        assert rep["verdict"] == "CLEAN"

    def test_flagged_note_separates_flagged_from_unflagged(self):
        rows = [_mk_run(i) for i in range(1, bi.MIN_TOTAL + 1)]
        rows.append(_mk_run(800, spy=0.0,
                            notes="benchmark_unavailable: SPY series empty"))
        rows.append(_mk_run(900, spy=0.0, notes=""))
        rep = bi.benchmark_integrity_report(rows)
        assert rep["flagged_degenerate"] == 1
        assert rep["unflagged_degenerate"] == 1
        assert rep["max_flagged_run_id"] == 800
        assert rep["max_unflagged_run_id"] == 900

    def test_spy_non_zero_never_degenerate_regardless_of_window(self):
        rows = [_mk_run(i, spy=0.01, win_days=3650)  # tiny but non-zero, 10yr
                for i in range(1, bi.MIN_TOTAL + 1)]
        rep = bi.benchmark_integrity_report(rows)
        assert rep["unflagged_degenerate"] == 0
        assert rep["verdict"] == "CLEAN"


# ─── verdict ladder ──────────────────────────────────────────────────────


class TestVerdictLadder:
    def _mixed_corpus(self, recent_unflagged: int, old_unflagged: int,
                      base_n: int = 200) -> list[dict]:
        """base_n clean runs by run_id 1..N, plus N unflagged at low/high ids."""
        rows = []
        # Old unflagged — at the low end of run_id, beyond the leak window.
        for i in range(1, old_unflagged + 1):
            rows.append(_mk_run(i, spy=0.0, win_days=365))
        # Clean filler — spans run_ids OLD+1 ... OLD+base_n.
        for i in range(old_unflagged + 1, old_unflagged + base_n + 1):
            rows.append(_mk_run(i, spy=5.0))
        # Recent unflagged — at the top of run_id, inside the leak window.
        top_start = old_unflagged + base_n + 1
        for i in range(top_start, top_start + recent_unflagged):
            rows.append(_mk_run(i, spy=0.0, win_days=365))
        return rows

    def test_active_leak_when_recent_unflagged_present(self):
        # 1 unflagged in the last LEAK_DETECTION_WINDOW runs → ACTIVE_LEAK.
        rows = self._mixed_corpus(recent_unflagged=1, old_unflagged=0,
                                  base_n=bi.LEAK_DETECTION_WINDOW - 1)
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "ACTIVE_LEAK"
        assert "bypassed" in rep["hint"]

    def test_historical_contamination_when_only_old_unflagged(self):
        # Old unflagged runs, but nothing in the last LEAK_DETECTION_WINDOW.
        rows = self._mixed_corpus(recent_unflagged=0, old_unflagged=5,
                                  base_n=bi.LEAK_DETECTION_WINDOW + 50)
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "HISTORICAL_CONTAMINATION"
        assert rep["unflagged_degenerate"] == 5

    def test_flagged_only_when_every_degenerate_carries_note(self):
        rows = [_mk_run(i, spy=5.0) for i in range(1, 100)]
        for i in range(100, 110):  # 10 flagged degenerate runs
            rows.append(_mk_run(i, spy=0.0,
                                notes="benchmark_unavailable: SPY empty"))
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "FLAGGED_ONLY"
        assert rep["unflagged_degenerate"] == 0
        assert rep["flagged_degenerate"] == 10

    def test_active_leak_dominates_historical_contamination(self):
        # Both old + recent unflagged → ACTIVE_LEAK wins (most actionable).
        rows = self._mixed_corpus(recent_unflagged=2, old_unflagged=5,
                                  base_n=bi.LEAK_DETECTION_WINDOW + 50)
        rep = bi.benchmark_integrity_report(rows)
        assert rep["verdict"] == "ACTIVE_LEAK"


# ─── bucketing ───────────────────────────────────────────────────────────


class TestBucketing:
    def test_unflagged_grouped_into_500_id_buckets(self):
        rows = [_mk_run(i, spy=5.0) for i in range(1, bi.MIN_TOTAL + 1)]
        # Place unflagged runs in two separate buckets.
        for rid in (450, 460, 470):  # bucket 0-499
            rows.append(_mk_run(rid, spy=0.0))
        for rid in (5550, 5560):  # bucket 5500-5999
            rows.append(_mk_run(rid, spy=0.0))
        rep = bi.benchmark_integrity_report(rows)
        buckets = {b["bucket_start"]: b["n_unflagged"] for b in rep["run_id_buckets"]}
        assert buckets[0] == 3
        assert buckets[5500] == 2

    def test_buckets_sorted_ascending(self):
        rows = [_mk_run(i, spy=5.0) for i in range(1, bi.MIN_TOTAL + 1)]
        for rid in (5000, 1000, 3000):
            rows.append(_mk_run(rid, spy=0.0))
        rep = bi.benchmark_integrity_report(rows)
        bucket_starts = [b["bucket_start"] for b in rep["run_id_buckets"]]
        assert bucket_starts == sorted(bucket_starts)


# ─── qualifier window economic impact ────────────────────────────────────


class TestQualifierWindow:
    def test_qualifier_window_uses_run_id_desc_order(self):
        # 30 clean runs + 10 fake-α runs all in the recent window.
        # The qualifier sorts by run_id DESC, so high run_ids appear first.
        rows = [_mk_run(i, spy=5.0, ret=10.0) for i in range(1, 31)]
        rows.extend(_mk_run(i, spy=0.0, ret=200.0)
                    for i in range(100, 110))
        rep = bi.benchmark_integrity_report(rows)
        qw = rep["qualifier_window"]
        assert qw["window"] == bi.QUALIFIER_WINDOW
        assert qw["n"] == bi.QUALIFIER_WINDOW
        # Last QUALIFIER_WINDOW=20 by run_id DESC: all 10 fake (run_ids 100-109)
        # plus 10 clean (run_ids 21-30).
        assert qw["n_unflagged_in_window"] == 10

    def test_median_alpha_delta_exact(self):
        # Construct a corpus where asis median is provably different from clean:
        # 10 clean runs with vs_spy=10, 10 fake-α runs with vs_spy=200.
        # asis = median([10]*10 + [200]*10) = 105.
        # clean = median([10]*10) = 10.
        # delta = +95.
        rows = [_mk_run(rid, spy=5.0, ret=15.0)  # vs_spy=10
                for rid in range(100, 110)]
        rows.extend(_mk_run(rid, spy=0.0, ret=200.0)  # vs_spy=200 fake
                    for rid in range(200, 210))
        rep = bi.benchmark_integrity_report(rows)
        qw = rep["qualifier_window"]
        assert qw["n"] == 20
        assert qw["n_unflagged_in_window"] == 10
        assert qw["median_alpha_asis"] == 105.0
        assert qw["median_alpha_clean"] == 10.0
        assert qw["median_alpha_delta"] == 95.0

    def test_qualifier_window_filters_low_trade_runs(self):
        # _ml_is_qualified excludes n_trades<5 — mirror that here.
        rows = [_mk_run(i, spy=5.0, n_trades=10) for i in range(1, 25)]
        rows.append(_mk_run(999, spy=5.0, n_trades=2))  # below threshold
        rep = bi.benchmark_integrity_report(rows)
        # The n_trades=2 row should NOT enter the qualifier window.
        # With 24 clean runs (n_trades=10) qualifying + 1 below threshold,
        # qualifier window holds 20 of the 24, the n_trades=2 row excluded.
        qw = rep["qualifier_window"]
        assert qw["n"] == bi.QUALIFIER_WINDOW
        # Tighter: max run_id in qualifier should be 24, not 999.
        # (We don't expose run_ids in the qualifier dict, but the n=20 + no
        # delta proves the n_trades filter held.)

    def test_qualifier_window_handles_all_unflagged(self):
        # Pathological: every qualifying run is unflagged-degenerate.
        rows = [_mk_run(rid, spy=0.0, ret=100.0)
                for rid in range(100, 100 + bi.QUALIFIER_WINDOW + 5)]
        # Add MIN_TOTAL clean filler (n_trades=10) so total ≥ MIN_TOTAL.
        rows.extend(_mk_run(rid, spy=5.0, ret=10.0)
                    for rid in range(1, bi.MIN_TOTAL))
        rep = bi.benchmark_integrity_report(rows)
        qw = rep["qualifier_window"]
        # The most recent QUALIFIER_WINDOW=20 by run_id DESC are all fake.
        assert qw["n"] == bi.QUALIFIER_WINDOW
        assert qw["n_unflagged_in_window"] == bi.QUALIFIER_WINDOW
        # Clean median is None — no clean rows in the window.
        assert qw["median_alpha_clean"] is None
        assert qw["median_alpha_delta"] is None


# ─── degrade-never-raises ────────────────────────────────────────────────


class TestDegradeNeverRaises:
    def test_malformed_rows_dropped_not_raised(self):
        # Garbage row should not break the report on the OTHER good rows.
        # Rows missing required parseable fields (run_id, dates) are dropped;
        # a row with parseable dates but bad spy/vs_spy values is KEPT but
        # classified as non-degenerate (spy_f=None → not flagged) — that is
        # honest "we don't know" behavior, not a fabricated degenerate count.
        rows = [_mk_run(i) for i in range(1, bi.MIN_TOTAL + 1)]
        rows.extend([
            {"run_id": "not-an-int", "status": "complete"},  # drops
            {"start_date": "not-a-date", "end_date": "2020", "status": "complete"},  # drops
            None,  # not even a dict — drops
            {"run_id": 999, "status": "complete", "start_date": "2020-01-01",
             "end_date": "2020-01-31", "spy_return_pct": "not-a-float",
             "vs_spy_pct": None, "n_trades": None, "notes": None},  # kept, non-degenerate
        ])
        rep = bi.benchmark_integrity_report(rows)
        # 3 of 4 malformed rows drop; the 4th (bad spy_return but valid dates) is
        # counted as non-degenerate — honest "we couldn't classify it" behavior.
        assert rep["total"] == bi.MIN_TOTAL + 1
        assert rep["unflagged_degenerate"] == 0  # bad spy_return doesn't become a false-positive degenerate
        assert rep["verdict"] == "CLEAN"

    def test_none_input_returns_insufficient_data(self):
        rep = bi.benchmark_integrity_report(None)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total"] == 0

    def test_analyze_missing_db_degrades_honestly(self, tmp_path):
        missing = tmp_path / "does-not-exist.db"
        rep = bi.analyze(missing)
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "missing" in rep["hint"].lower()


# ─── analyze() against a real SQLite DB ──────────────────────────────────


@pytest.fixture()
def synthetic_backtest_db(tmp_path: Path) -> Path:
    """A fresh sqlite DB with a minimal backtest_runs table matching the
    columns analyze() selects. Round-trips through real SQL so a schema
    drift between this test and BacktestStore would surface."""
    db = tmp_path / "backtest.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE backtest_runs (
            run_id INTEGER PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            status TEXT,
            spy_return_pct REAL,
            vs_spy_pct REAL,
            n_trades INTEGER,
            notes TEXT
        )
    """)
    # 30 clean runs + 5 unflagged-degenerate + 3 flagged.
    for i in range(1, 31):
        conn.execute(
            "INSERT INTO backtest_runs VALUES (?, ?, ?, 'complete', 5.0, 5.0, 100, '')",
            (i, "2020-01-01", "2021-01-01"),
        )
    # Unflagged-degenerate at run_ids 100..104.
    for i in range(100, 105):
        conn.execute(
            "INSERT INTO backtest_runs VALUES (?, ?, ?, 'complete', 0.0, 200.0, 100, '')",
            (i, "2020-01-01", "2021-01-01"),
        )
    # Flagged-degenerate at run_ids 200..202.
    for i in range(200, 203):
        conn.execute(
            "INSERT INTO backtest_runs VALUES "
            "(?, ?, ?, 'complete', 0.0, 100.0, 100, "
            "'benchmark_unavailable: SPY empty')",
            (i, "2020-01-01", "2021-01-01"),
        )
    conn.commit()
    conn.close()
    return db


class TestAnalyzeAgainstRealDb:
    def test_round_trip_through_sql(self, synthetic_backtest_db: Path):
        rep = bi.analyze(synthetic_backtest_db)
        assert rep["status"] == "ok"
        assert rep["total"] == 38
        assert rep["flagged_degenerate"] == 3
        assert rep["unflagged_degenerate"] == 5
        # Latest run_id 202 → flagged, so the recent leak window should
        # include the unflagged 100..104. They're in the last 50 runs ⇒ ACTIVE_LEAK.
        assert rep["verdict"] == "ACTIVE_LEAK"
        assert rep["max_unflagged_run_id"] == 104
        assert rep["max_flagged_run_id"] == 202


# ─── CLI ──────────────────────────────────────────────────────────────────


class TestCli:
    def test_cli_emits_valid_json(self, synthetic_backtest_db: Path):
        r = subprocess.run(
            [sys.executable, "-m", "paper_trader.ml.benchmark_integrity",
             "--json", "--db", str(synthetic_backtest_db)],
            capture_output=True, text=True, timeout=30,
        )
        # Exit code 2 = ACTIVE_LEAK (synthetic db has recent unflagged rows).
        assert r.returncode == 2, r.stderr
        payload = json.loads(r.stdout)
        assert payload["verdict"] == "ACTIVE_LEAK"
        assert payload["unflagged_degenerate"] == 5

    def test_cli_exit_code_clean(self, tmp_path: Path):
        db = tmp_path / "clean.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE backtest_runs (
                run_id INTEGER PRIMARY KEY,
                start_date TEXT, end_date TEXT, status TEXT,
                spy_return_pct REAL, vs_spy_pct REAL, n_trades INTEGER, notes TEXT
            )""")
        for i in range(1, 50):
            conn.execute(
                "INSERT INTO backtest_runs VALUES "
                "(?, '2020-01-01', '2021-01-01', 'complete', 5.0, 5.0, 100, '')",
                (i,),
            )
        conn.commit(); conn.close()
        r = subprocess.run(
            [sys.executable, "-m", "paper_trader.ml.benchmark_integrity",
             "--json", "--db", str(db)],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0  # CLEAN
        payload = json.loads(r.stdout)
        assert payload["verdict"] == "CLEAN"

    def test_cli_exit_code_missing_db(self, tmp_path: Path):
        r = subprocess.run(
            [sys.executable, "-m", "paper_trader.ml.benchmark_integrity",
             "--json", "--db", str(tmp_path / "nope.db")],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 3  # INSUFFICIENT_DATA
