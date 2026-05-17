"""2026-05-16 ML/backtest review pass — regression locks for three
load-bearing seams with **zero** prior direct test coverage, found by
grepping every backtest/continuous symbol against `tests/`:

1. ``run_continuous_backtests._next_run_id`` — the monotonic run-id
   allocator. It is the *only* thing that stops a new continuous cycle
   from re-using a run_id that already exists; a wrong value here makes
   the next ``upsert_run`` take its UPDATE branch and silently overwrite
   a completed historical run (the "old results must not be overwritten"
   guarantee the review brief asks for, at the loop layer). Discriminating
   assertions: COALESCE guard on an empty table (→ 1, never a crash on
   ``int(None)+1``) and ``MAX(run_id)+1`` — *not* ``COUNT(*)+1`` — on a
   non-contiguous table.

2. ``BacktestStore.upsert_run`` INSERT-vs-UPDATE branch — re-calling it
   for an existing run_id must change **only** ``status`` and must
   preserve the original ``seed`` / ``start_date`` / ``end_date`` /
   ``start_value`` / ``started_at``. This is the same "results not
   overwritten" guarantee one layer down: the continuous loop calls
   ``upsert_run(i, ..., "running")`` then ``run_one`` finalises; if the
   UPDATE branch clobbered the seed or window the dashboard would show a
   completed run stamped with the wrong metadata. ``upsert_run`` is used
   as a setup helper in 12 test files but its UPDATE-branch *semantics*
   were never asserted.

3. ``update_partial_progress`` vs ``finalize_run`` arithmetic — both
   feed the dashboard's ``total_return_pct``. ``update_partial_progress``
   is what renders a run's live equity % *during* the multi-minute run;
   it shares the ``(value - INITIAL_CASH) / INITIAL_CASH * 100`` formula
   with ``finalize_run`` but must **not** touch ``spy_return_pct`` /
   ``vs_spy_pct`` / ``status`` / ``completed_at`` (those only become
   meaningful at finalize). A regression that divided by ``current_value``
   instead of ``INITIAL_CASH``, flipped the sign, or moved the
   ``vs_spy = total - spy`` subtraction into the partial path would show
   wrong returns on the live backtest dashboard for the entire duration
   of every run. ``update_partial_progress`` had zero references in the
   whole suite; ``finalize_run``'s exact %/vs-SPY arithmetic was never
   pinned either.

Exact-value, not ranges — per the AGENTS.md test-inventory convention,
a formula change must update the literals deliberately. All offline; a
real ``BacktestStore`` on a tmp sqlite file (conftest's autouse fixture
redirects ``BACKTEST_DB`` anyway, but these pass an explicit path).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestStore, INITIAL_CASH


# ─────────────────────── _next_run_id ───────────────────────────


def _engine_with_store(tmp_path):
    store = BacktestStore(path=tmp_path / "bt.db")
    engine = MagicMock()
    engine.store = store
    return engine, store


class TestNextRunId:
    def test_empty_table_returns_one(self, tmp_path):
        """COALESCE(MAX(run_id), 0) + 1 on an empty table → 1.

        A regression that dropped the COALESCE would do int(None)+1 and
        raise TypeError on the very first cycle of a fresh deployment.
        """
        engine, store = _engine_with_store(tmp_path)
        try:
            assert rcb._next_run_id(engine) == 1
        finally:
            store.conn.close()

    def test_populated_table_returns_max_plus_one(self, tmp_path):
        engine, store = _engine_with_store(tmp_path)
        try:
            for i in range(1, 6):  # runs 1..5
                store.upsert_run(i, seed=i, status="complete",
                                 start=date(2025, 1, 1), end=date(2025, 12, 31))
            assert rcb._next_run_id(engine) == 6
        finally:
            store.conn.close()

    def test_non_contiguous_uses_max_not_count(self, tmp_path):
        """Runs 3 and 9 present → next id is 10 (MAX+1), NOT 3 (COUNT+1).

        After _trim_history deletes the oldest runs the table is sparse;
        a COUNT(*)+1 implementation would collide with a surviving run_id
        and the next upsert_run would silently overwrite it.
        """
        engine, store = _engine_with_store(tmp_path)
        try:
            for i in (3, 9):
                store.upsert_run(i, seed=i, status="complete",
                                 start=date(2025, 1, 1), end=date(2025, 12, 31))
            assert rcb._next_run_id(engine) == 10
        finally:
            store.conn.close()


# ─────────────────── upsert_run INSERT vs UPDATE ────────────────


class TestUpsertRunBranch:
    def test_insert_then_update_preserves_metadata(self, tmp_path):
        """First upsert_run INSERTs full metadata; a second call for the
        SAME run_id changes ONLY status — seed/dates/start_value/started_at
        survive untouched even when the second call passes different ones.

        This is the store-layer 'old results are not overwritten' lock:
        the continuous loop calls upsert_run(i, ..., 'running') then later
        finalises the same i; the intermediate status flip must not rewrite
        the run's identity.
        """
        store = BacktestStore(path=tmp_path / "bt.db")
        try:
            store.upsert_run(7, seed=42, status="running",
                             start=date(2021, 3, 1), end=date(2021, 9, 1))
            row1 = store.conn.execute(
                "SELECT seed, start_date, end_date, start_value, status, "
                "started_at FROM backtest_runs WHERE run_id=7"
            ).fetchone()
            assert row1["seed"] == 42
            assert row1["start_date"] == "2021-03-01"
            assert row1["end_date"] == "2021-09-01"
            assert row1["start_value"] == INITIAL_CASH  # 1000.0 default
            assert row1["status"] == "running"
            original_started_at = row1["started_at"]
            assert original_started_at  # non-null ISO timestamp

            # Second call: SAME run_id, deliberately DIFFERENT seed/window.
            store.upsert_run(7, seed=999, status="complete",
                             start=date(2099, 1, 1), end=date(2099, 12, 31))
            row2 = store.conn.execute(
                "SELECT seed, start_date, end_date, start_value, status, "
                "started_at FROM backtest_runs WHERE run_id=7"
            ).fetchone()
            # Only status moved.
            assert row2["status"] == "complete"
            # Everything else is the ORIGINAL — not the second call's args.
            assert row2["seed"] == 42
            assert row2["start_date"] == "2021-03-01"
            assert row2["end_date"] == "2021-09-01"
            assert row2["start_value"] == INITIAL_CASH
            assert row2["started_at"] == original_started_at

            # Still exactly one row for run_id 7 (UPDATE, not a 2nd INSERT).
            n = store.conn.execute(
                "SELECT COUNT(*) FROM backtest_runs WHERE run_id=7"
            ).fetchone()[0]
            assert n == 1
        finally:
            store.conn.close()


# ──────────── update_partial_progress vs finalize_run ───────────


class TestPartialVsFinalizeArithmetic:
    def test_partial_progress_pct_and_no_spy_touch(self, tmp_path):
        """update_partial_progress writes total_return_pct from the
        (value - 1000)/1000*100 formula and leaves spy/vs_spy/status/
        completed_at exactly as upsert_run left them."""
        store = BacktestStore(path=tmp_path / "bt.db")
        try:
            store.upsert_run(1, seed=1, status="running",
                             start=date(2025, 1, 1), end=date(2025, 12, 31))
            store.update_partial_progress(
                1, current_value=1500.0, n_trades=4, n_decisions=9,
                equity_curve=[{"date": "2025-06-01", "value": 1500.0}],
            )
            r = store.conn.execute(
                "SELECT final_value, total_return_pct, spy_return_pct, "
                "vs_spy_pct, n_trades, n_decisions, status, completed_at, "
                "equity_curve_json FROM backtest_runs WHERE run_id=1"
            ).fetchone()
            assert r["final_value"] == 1500.0
            # (1500 - 1000) / 1000 * 100 == 50.0  (exactly representable)
            assert r["total_return_pct"] == 50.0
            assert r["n_trades"] == 4
            assert r["n_decisions"] == 9
            # Untouched by the partial path:
            assert r["spy_return_pct"] == 0.0
            assert r["vs_spy_pct"] == 0.0
            assert r["status"] == "running"
            assert r["completed_at"] is None
            assert '"value": 1500.0' in r["equity_curve_json"]
        finally:
            store.conn.close()

    def test_partial_progress_handles_loss(self, tmp_path):
        """Negative path: a sub-$1000 mark gives a negative pct (sign lock)."""
        store = BacktestStore(path=tmp_path / "bt.db")
        try:
            store.upsert_run(2, seed=2, status="running",
                             start=date(2025, 1, 1), end=date(2025, 12, 31))
            store.update_partial_progress(
                2, current_value=975.0, n_trades=1, n_decisions=1,
                equity_curve=[],
            )
            pct = store.conn.execute(
                "SELECT total_return_pct FROM backtest_runs WHERE run_id=2"
            ).fetchone()[0]
            # (975 - 1000) / 1000 * 100 == -2.5  (exactly representable)
            assert pct == -2.5
        finally:
            store.conn.close()

    def test_finalize_run_pct_vs_spy_and_terminal_fields(self, tmp_path):
        """finalize_run shares the % formula but ALSO computes
        vs_spy = total - spy, stamps status + completed_at, and persists
        the SPY benchmark. The vs_spy subtraction lives ONLY here — never
        in update_partial_progress."""
        store = BacktestStore(path=tmp_path / "bt.db")
        try:
            store.upsert_run(3, seed=3, status="running",
                             start=date(2025, 1, 1), end=date(2025, 12, 31))
            store.finalize_run(
                3, final_value=1500.0, spy_return_pct=12.5,
                n_trades=7, n_decisions=20,
                equity_curve=[{"date": "2025-12-31", "value": 1500.0}],
            )
            r = store.conn.execute(
                "SELECT final_value, total_return_pct, spy_return_pct, "
                "vs_spy_pct, n_trades, n_decisions, status, completed_at "
                "FROM backtest_runs WHERE run_id=3"
            ).fetchone()
            assert r["final_value"] == 1500.0
            assert r["total_return_pct"] == 50.0          # same formula
            assert r["spy_return_pct"] == 12.5            # benchmark persisted
            assert r["vs_spy_pct"] == 37.5                # 50.0 - 12.5, here only
            assert r["n_trades"] == 7
            assert r["n_decisions"] == 20
            assert r["status"] == "complete"              # default terminal status
            assert r["completed_at"]                       # non-null timestamp
        finally:
            store.conn.close()

    def test_finalize_run_negative_alpha(self, tmp_path):
        """A run that made +50% while SPY made +80% has NEGATIVE alpha
        (vs_spy = -30.0). Pins the subtraction direction (total - spy,
        not spy - total)."""
        store = BacktestStore(path=tmp_path / "bt.db")
        try:
            store.upsert_run(4, seed=4, status="running",
                             start=date(2025, 1, 1), end=date(2025, 12, 31))
            store.finalize_run(
                4, final_value=1500.0, spy_return_pct=80.0,
                n_trades=1, n_decisions=1, equity_curve=[],
            )
            r = store.conn.execute(
                "SELECT total_return_pct, vs_spy_pct FROM backtest_runs "
                "WHERE run_id=4"
            ).fetchone()
            assert r["total_return_pct"] == 50.0
            assert r["vs_spy_pct"] == -30.0
        finally:
            store.conn.close()
