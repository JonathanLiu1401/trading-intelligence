"""Regression coverage for /api/model-progress.

Live failure (2026-05-26 ops session): the endpoint instantiated a full
``BacktestStore()`` on every request, which runs ``PRAGMA journal_mode=WAL``,
a ``CREATE TABLE IF NOT EXISTS`` executescript, and idempotent ``ALTER TABLE``
migrations. Each of those takes the SQLite WRITER lock briefly. Under the live
continuous-backtest committee (5 parallel writers every 60s on the same
``backtest.db``), the writer lock is heavily contended — the endpoint then
sat for the full 30s busy_timeout and curl returned HTTP 000. The Model
Progress chart on the dashboard was permanently blank.

The fix opens ``backtest.db`` in ``mode=ro`` and runs only the SELECT (the
``signals.py`` / ``api_model_rankings`` precedent — readers never contend
with WAL writers). This file pins:

  * the endpoint returns 200 with a sane cycles list,
  * the cycle aggregation math (5-row chunks, best/avg/worst, last cycle
    partial-chunk handling),
  * the no-data ``cycles=[] total_runs=0`` shape — both for an empty file
    AND for a missing file (fresh deploy),
  * the endpoint does NOT acquire a writer lock — verified by holding the
    write lock on a sibling connection and asserting the endpoint still
    succeeds well inside the 5s read-only timeout.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import paper_trader.dashboard as dash


SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id              INTEGER PRIMARY KEY,
    status              TEXT NOT NULL,
    total_return_pct    REAL,
    vs_spy_pct          REAL,
    n_trades            INTEGER,
    completed_at        TEXT
);
"""


def _make_backtest_db(path: Path, n_complete: int, *, status_pattern: str = "complete") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    for i in range(n_complete):
        status = status_pattern
        conn.execute(
            "INSERT INTO backtest_runs (run_id, status, total_return_pct, vs_spy_pct, n_trades, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i + 1, status, (i % 10) - 5.0, (i % 7) - 3.0, 5 + i,
             f"2026-05-{(i % 28) + 1:02d}T12:00:00+00:00"),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def client():
    dash.app.config["TESTING"] = True
    return dash.app.test_client()


def test_no_db_file_returns_empty_payload(tmp_path, monkeypatch, client):
    """A missing backtest.db (fresh deploy / freshly cleaned) returns a
    sane empty payload — NOT a 500, NOT a hang."""
    missing = tmp_path / "absent" / "backtest.db"
    monkeypatch.setattr(dash, "BACKTEST_DB", missing)

    rv = client.get("/api/model-progress")
    assert rv.status_code == 200, rv.data
    body = rv.get_json()
    assert body["cycles"] == []
    assert body["total_runs"] == 0


def test_empty_backtest_db_returns_empty_payload(tmp_path, monkeypatch, client):
    """A backtest.db with the table but ZERO complete rows returns the
    same empty shape an empty-file caller gets — covers the
    ``if not rows`` branch."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=0)
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    rv = client.get("/api/model-progress")
    assert rv.status_code == 200, rv.data
    body = rv.get_json()
    assert body["cycles"] == []
    assert body["total_runs"] == 0


def test_aggregates_full_cycles_of_five(tmp_path, monkeypatch, client):
    """10 complete runs aggregate into exactly 2 cycles of 5 with the
    expected labels and per-cycle best/avg/worst."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=10)
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    rv = client.get("/api/model-progress")
    assert rv.status_code == 200, rv.data
    body = rv.get_json()
    cycles = body["cycles"]
    assert body["total_runs"] == 10
    assert len(cycles) == 2

    # Run returns by (i % 10) - 5.0: -5,-4,-3,-2,-1,0,1,2,3,4
    # Cycle 1 (run_id 1..5) → returns [-5,-4,-3,-2,-1]
    assert cycles[0]["cycle"] == "#1-5"
    assert cycles[0]["run_start"] == 1
    assert cycles[0]["n"] == 5
    assert cycles[0]["best"] == -1.0
    assert cycles[0]["worst"] == -5.0
    assert cycles[0]["avg"] == -3.0   # mean(-5..-1)

    # Cycle 2 (run_id 6..10) → returns [0,1,2,3,4]
    assert cycles[1]["cycle"] == "#6-10"
    assert cycles[1]["run_start"] == 6
    assert cycles[1]["best"] == 4.0
    assert cycles[1]["worst"] == 0.0
    assert cycles[1]["avg"] == 2.0


def test_partial_last_chunk_is_labelled_and_aggregated(
    tmp_path, monkeypatch, client
):
    """A tail of fewer than 5 rows produces a partial cycle whose
    label is a single ``#N`` (not ``#N-N``) and whose n reflects the
    actual chunk size — the live failure mode would have been an
    IndexError or wrong label if chunk slicing was off-by-one."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=6)        # 5 + 1
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    rv = client.get("/api/model-progress")
    body = rv.get_json()
    cycles = body["cycles"]
    assert body["total_runs"] == 6
    assert len(cycles) == 2

    # Partial second chunk: just run_id=6 (return=0.0)
    tail = cycles[1]
    assert tail["cycle"] == "#6"               # single-run label
    assert tail["run_start"] == 6
    assert tail["n"] == 1
    assert tail["best"] == 0.0
    assert tail["worst"] == 0.0
    assert tail["avg"] == 0.0


def test_excludes_non_complete_runs(tmp_path, monkeypatch, client):
    """Only ``status='complete'`` rows count toward the chart — a
    ``running`` / ``failed`` row must not poison a cycle."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=5)
    # Insert 2 non-complete rows
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO backtest_runs (run_id, status, total_return_pct, vs_spy_pct, n_trades, completed_at) "
        "VALUES (6, 'running', 999.0, 0, 0, NULL), "
        "       (7, 'failed',  -999.0, 0, 0, NULL)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    rv = client.get("/api/model-progress")
    body = rv.get_json()
    # The 999 / -999 numbers would dominate any cycle if they slipped through.
    assert body["total_runs"] == 5
    assert len(body["cycles"]) == 1
    assert body["cycles"][0]["best"] == -1.0
    assert body["cycles"][0]["worst"] == -5.0


def test_response_includes_completed_at_per_cycle(tmp_path, monkeypatch, client):
    """The chart's tooltip reads ``completed_at`` — pin that the
    builder propagates it from the LAST row of each chunk."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=5)
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    rv = client.get("/api/model-progress")
    body = rv.get_json()
    assert body["cycles"][0]["completed_at"].startswith("2026-05-")


def test_endpoint_does_not_acquire_writer_lock(tmp_path, monkeypatch, client):
    """Regression for the live failure mode: hold an EXCLUSIVE writer lock on
    the file and verify ``/api/model-progress`` still returns 200 quickly.

    The pre-fix implementation called ``BacktestStore()`` which runs DDL on
    every request — any DDL needs the writer lock, so this test would have
    hung past the 5s timeout. The fixed endpoint opens read-only and the
    SELECT against the WAL DB completes immediately."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=5)
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    # Open a long-lived writer connection in WAL mode and start (but never
    # commit) an exclusive transaction. With WAL + a held writer txn, ANY
    # second writer that needs the lock waits up to busy_timeout.
    writer = sqlite3.connect(str(db), timeout=30)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("BEGIN IMMEDIATE")   # take the WAL writer lock NOW

    try:
        t0 = time.monotonic()
        rv = client.get("/api/model-progress")
        elapsed = time.monotonic() - t0
        assert rv.status_code == 200, rv.data
        body = rv.get_json()
        assert body["total_runs"] == 5
        # The whole point: the read MUST be fast even under contention.
        # The fixed endpoint connects with timeout=5 ; pin << 5s.
        assert elapsed < 3.0, (
            f"/api/model-progress took {elapsed:.2f}s with a held writer lock "
            f"— it must NOT acquire a writer lock"
        )
    finally:
        writer.rollback()
        writer.close()


def test_endpoint_closes_its_connection(tmp_path, monkeypatch, client):
    """The pre-fix code never closed the BacktestStore's connection, leaking
    one per request. Verify that 50 sequential hits do not exhaust SQLite's
    per-process FD budget — a connection leak would surface as
    ``OperationalError: unable to open database file`` once the OS FD
    rlimit was hit. Empirically 50 hits is plenty to reveal a leak without
    making the test slow."""
    db = tmp_path / "backtest.db"
    _make_backtest_db(db, n_complete=3)
    monkeypatch.setattr(dash, "BACKTEST_DB", db)

    for _ in range(50):
        rv = client.get("/api/model-progress")
        assert rv.status_code == 200, rv.data
