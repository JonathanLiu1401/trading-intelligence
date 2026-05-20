import sqlite3
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def make_store(tmp_path):
    from paper_trader.backtest import BacktestStore
    return BacktestStore(path=tmp_path / "test.db")

def test_backtest_runs_has_model_id_column(tmp_path):
    store = make_store(tmp_path)
    cols = [row[1] for row in store.conn.execute("PRAGMA table_info(backtest_runs)").fetchall()]
    assert "model_id" in cols

def test_backtest_runs_model_id_defaults_to_ml_quant(tmp_path):
    store = make_store(tmp_path)
    store.conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, start_value, status, started_at) "
        "VALUES (1, 42, '2025-01-01', '2026-01-01', 1000.0, 'running', '2026-01-01T00:00:00Z')"
    )
    store.conn.commit()
    row = store.conn.execute("SELECT model_id FROM backtest_runs WHERE run_id=1").fetchone()
    assert row[0] == "ml_quant"
