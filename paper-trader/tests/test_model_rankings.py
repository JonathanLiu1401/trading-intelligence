import sqlite3
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
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


def _build_synthetic_prices(days: int = 21):
    """Mirror tests/test_integration_backtest._build_synthetic_prices in miniature."""
    from paper_trader.backtest import PriceCache
    start = date(2024, 1, 2)
    seq = []
    d = start
    while len(seq) < days:
        if d.weekday() < 5:
            seq.append(d)
        d += timedelta(days=1)
    tickers = ["SPY", "NVDA"]
    cache = PriceCache.__new__(PriceCache)
    cache.tickers = tickers
    cache.start = seq[0]
    cache.end = seq[-1]
    step = 50.0 / max(days - 1, 1)
    cache.prices = {
        t: {dd.isoformat(): 100.0 + i * step for i, dd in enumerate(seq)}
        for t in tickers
    }
    cache.trading_days = seq
    return cache


def _make_engine_no_net(prices, tmp_path, model_id="ml_quant"):
    """Construct BacktestEngine without triggering yfinance/GDELT init."""
    from paper_trader.backtest import BacktestEngine, BacktestStore
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.start = prices.trading_days[0]
    engine.end = prices.trading_days[-1]
    engine.store = BacktestStore(path=tmp_path / "bt.db")
    engine.prices = prices
    engine.gdelt = None
    engine.av_news = None
    engine._local_news = {}
    engine.model_id = model_id
    return engine


def test_backtest_engine_stores_model_id(tmp_path):
    """BacktestEngine with model_id='claude-opus-4-7' stores that value in backtest_runs."""
    import paper_trader.backtest as bt

    prices = _build_synthetic_prices(days=11)
    engine = _make_engine_no_net(prices, tmp_path, model_id="claude-opus-4-7")

    # _fetch_signals returns one dummy article so the loop has something to chew on
    def fake_signals(d, seed, rng, portfolio=None):
        return [{"title": f"news {d.isoformat()}", "url": "", "score": 1.0,
                 "tickers": ["SPY"]}]

    # LLM path goes through _llm_call. Return a HOLD JSON so no trade fires.
    def fake_llm_call(model_id, prompt, *a, **kw):
        return '{"action":"HOLD","ticker":"","qty":0,"reasoning":"test"}'

    with patch.object(engine, "_fetch_signals", side_effect=fake_signals), \
         patch.object(bt, "_llm_call", side_effect=fake_llm_call):
        engine.run_one(run_id=1, seed=42)

    row = engine.store.conn.execute(
        "SELECT model_id FROM backtest_runs WHERE run_id=1"
    ).fetchone()
    assert row[0] == "claude-opus-4-7"


def test_backtest_engine_rejects_invalid_model_id(tmp_path):
    """BacktestEngine.__init__ raises on a model_id that is not in the valid set."""
    import paper_trader.backtest as bt

    # Stub yfinance so __init__ doesn't go to network; the validation should run
    # AFTER attribute setup. Easiest: patch PriceCache to skip the load.
    class _StubPriceCache:
        def __init__(self, tickers, start, end):
            self.tickers = tickers
            self.start = start
            self.end = end
            self.prices = {t: {} for t in tickers}
            self.trading_days = [start]
    with patch.object(bt, "PriceCache", _StubPriceCache):
        with pytest.raises(ValueError, match="Invalid model_id"):
            bt.BacktestEngine(start=date(2024, 1, 2), end=date(2024, 1, 5),
                              model_id="gpt-4")
