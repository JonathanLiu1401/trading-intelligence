"""A transient SQLite lock on the news DB must NEVER abort the decision cycle.

Live-observed bug (runner.log): `signals.get_top_signals()` let a transient
`sqlite3.OperationalError: database is locked` (the digital-intern daemon
mid-WAL-checkpoint) propagate up through `strategy.decide()`, which
`runner._cycle` only catches generically — so the WHOLE cycle was lost: no
decision recorded, no equity point. A *news* DB hiccup must degrade to "no
signals this cycle" (identical to a missing DB), not silently freeze trading.

These tests assert the ACTUAL degradation contract for every signals reader
that `decide()` calls, plus that the connection is still closed on the error
path (no fd leak), and that a non-sqlite error is NOT swallowed.
"""
from __future__ import annotations

import sqlite3

import pytest

from paper_trader import signals


class _LockingConn:
    """Stands in for a read-only sqlite connection whose query raises a
    transient lock error. Records whether close() was called."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.closed = False

    def execute(self, *_a, **_k):
        raise self._exc

    def close(self):
        self.closed = True


@pytest.fixture
def locked_conn(monkeypatch):
    conn = _LockingConn(sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(signals, "_connect_ro", lambda: conn)
    return conn


def test_get_top_signals_degrades_to_empty(locked_conn):
    out = signals.get_top_signals(20, hours=2, min_score=4.0)
    assert out == []                       # safe default, not a crash
    assert locked_conn.closed is True      # connection still released


def test_get_urgent_articles_degrades_to_empty(locked_conn):
    out = signals.get_urgent_articles(minutes=30)
    assert out == []
    assert locked_conn.closed is True


def test_get_ticker_sentiment_degrades_to_zeroed_dict(locked_conn):
    out = signals.get_ticker_sentiment("NVDA", hours=4)
    assert out == {"ticker": "NVDA", "avg_score": 0.0,
                   "max_score": 0.0, "n": 0, "urgent": 0}
    assert locked_conn.closed is True


def test_ticker_sentiments_degrades_to_zeroed_list(locked_conn):
    tickers = ["NVDA", "AMD", "MU"]
    out = signals.ticker_sentiments(tickers, hours=4)
    assert [r["ticker"] for r in out] == tickers     # order preserved
    assert all(r["n"] == 0 and r["avg_score"] == 0.0 and r["urgent"] == 0
               for r in out)
    assert locked_conn.closed is True


def test_decide_path_survives_a_locked_news_db(monkeypatch):
    """End-to-end: the four readers `decide()` calls all degrade, so a fully
    locked news DB yields *empty news* — never a raised exception."""
    conn = _LockingConn(sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(signals, "_connect_ro", lambda: conn)
    # Exactly the calls strategy.decide() makes against signals.py.
    top = signals.get_top_signals(20, hours=2, min_score=4.0)
    urgent = signals.get_urgent_articles(minutes=30)
    sents = signals.ticker_sentiments(["NVDA", "AMD"], hours=4)
    assert top == [] and urgent == []
    assert all(s["n"] == 0 for s in sents)
    # The merge `strategy.decide()` performs must not blow up on the defaults.
    seen = {s["id"] for s in top}
    merged = [a for a in urgent if a["id"] not in seen] + top
    assert merged == []


def test_non_sqlite_error_is_not_swallowed(monkeypatch):
    """The guard is `except sqlite3.Error` — a genuine bug (e.g. a
    programming error) must still surface, not be masked as 'no news'."""
    class _BoomConn:
        def execute(self, *_a, **_k):
            raise ValueError("genuine bug, not a lock")

        def close(self):
            pass

    monkeypatch.setattr(signals, "_connect_ro", lambda: _BoomConn())
    with pytest.raises(ValueError):
        signals.get_top_signals(20, hours=2, min_score=4.0)
