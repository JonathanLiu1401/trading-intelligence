"""Pytest fixtures shared across ML/backtest tests.

The fixtures in this file deliberately avoid touching the network or the
live paper_trader.db / backtest.db. Every test must be runnable offline.
"""
from __future__ import annotations

import sys
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Make the project root importable so `paper_trader` resolves no matter
# where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Redirect paths that ml/backtest modules write to into a temp dir.

    Without this, tests would clobber the real data/ml/decision_scorer.pkl
    and the real backtest.db. Tests should be idempotent and side-effect free.
    """
    data_dir = tmp_path / "data"
    (data_dir / "ml").mkdir(parents=True, exist_ok=True)
    (data_dir / "backtest_cache").mkdir(parents=True, exist_ok=True)

    import paper_trader.ml.decision_scorer as ds
    monkeypatch.setattr(ds, "SCORER_PATH", data_dir / "ml" / "decision_scorer.pkl")
    # The scorer's process-wide load cache persists across tests in a session;
    # clear it so a prior test's model can never leak into this one (keys are
    # path+mtime+size, so collisions are already implausible — this is belt
    # and braces, mirroring the backtest cache resets below).
    if hasattr(ds, "_LOAD_CACHE"):
        ds._LOAD_CACHE.clear()

    import paper_trader.backtest as bt
    # raising=False so attribute renames in backtest.py don't break this whole
    # fixture — best-effort isolation. The caller still gets the data_dir.
    monkeypatch.setattr(bt, "CACHE_DIR", data_dir / "backtest_cache", raising=False)
    monkeypatch.setattr(bt, "PRICE_CACHE_PATH", data_dir / "backtest_cache" / "prices.json", raising=False)
    monkeypatch.setattr(bt, "GDELT_CACHE", data_dir / "backtest_cache" / "gdelt", raising=False)
    monkeypatch.setattr(bt, "AV_CACHE_DIR", data_dir / "backtest_cache" / "alphavantage", raising=False)
    monkeypatch.setattr(bt, "AV_QUOTA_PATH", data_dir / "backtest_cache" / "av_quota.json", raising=False)
    monkeypatch.setattr(bt, "BACKTEST_DB", data_dir / "backtest.db", raising=False)
    monkeypatch.setattr(bt, "_VOLUME_CACHE_PATH", data_dir / "backtest_cache" / "volumes.json", raising=False)

    # Reset module-level caches if they exist. Names have shifted as backtest.py
    # evolved; just clear whatever is present today.
    if hasattr(bt, "_VOLUME_CACHE"):
        bt._VOLUME_CACHE = {}
    if hasattr(bt, "_VOLUME_CACHE_LOADED"):
        bt._VOLUME_CACHE_LOADED = False
    if hasattr(bt, "_VOLUME_CACHE_DISK_LOADED"):
        bt._VOLUME_CACHE_DISK_LOADED = set()

    yield data_dir


@pytest.fixture
def synthetic_prices():
    """Build a tiny PriceCache populated with deterministic synthetic data.

    The series goes from 100.00 to ~150.00 monotonically over 60 weekdays,
    starting 2025-01-02. Tests can assert exact returns against this curve.
    """
    from paper_trader.backtest import PriceCache

    start = date(2025, 1, 2)  # a Thursday
    days = []
    d = start
    while len(days) < 60:
        if d.weekday() < 5:  # weekdays only
            days.append(d)
        d += timedelta(days=1)

    cache = PriceCache.__new__(PriceCache)
    cache.tickers = ["SPY", "NVDA"]
    cache.start = days[0]
    cache.end = days[-1]
    # Build monotonically rising closes. SPY: 100 → 150. NVDA: 100 → 200.
    cache.prices = {
        "SPY": {d.isoformat(): 100.0 + i for i, d in enumerate(days[:51])},
        "NVDA": {d.isoformat(): 100.0 + i * 2 for i, d in enumerate(days[:51])},
    }
    cache.trading_days = days[:51]
    return cache


@pytest.fixture
def empty_articles_db(tmp_path):
    """Provide a minimal articles.db with the schema paper-trader expects."""
    db_path = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db_path))
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
    conn.commit()
    conn.close()
    return db_path
