"""Tests for paper_trader.store — portfolio bookkeeping, trades, positions.

These tests use a real sqlite store backed by a temp DB. Each test gets a
fresh Store via the ``tmp_store`` fixture (see conftest.py), so writes from
one test do not leak into another.

The goal is to catch logic bugs in the store's invariants:
- cash bookkeeping after BUY / SELL
- position upserts (open, add to lot, partial close, full close)
- trade ordering (recent_trades returns most-recent first)
- equity curve ordering (ascending after the inversion in store.equity_curve)
"""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import store as store_mod
from paper_trader.store import INITIAL_CASH, Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Return a brand-new Store with its DB in tmp_path."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


class TestPortfolioInitialization:
    def test_initial_cash_is_default(self, fresh_store):
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == INITIAL_CASH
        assert pf["total_value"] == INITIAL_CASH
        assert pf["positions"] == []

    def test_no_open_positions_initially(self, fresh_store):
        assert fresh_store.open_positions() == []

    def test_no_trades_initially(self, fresh_store):
        assert fresh_store.recent_trades() == []


class TestCashBookkeeping:
    def test_update_portfolio_persists_cash(self, fresh_store):
        fresh_store.update_portfolio(cash=750.0, total_value=900.0, positions=[])
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == 750.0
        assert pf["total_value"] == 900.0

    def test_buy_then_update_portfolio_decreases_cash(self, fresh_store):
        # Simulate the cash flow of buying 10 shares at $50 (total $500):
        fresh_store.record_trade("NVDA", "BUY", qty=10, price=50.0, reason="t1")
        fresh_store.upsert_position("NVDA", "stock", qty=10, avg_cost=50.0)
        fresh_store.update_portfolio(
            cash=INITIAL_CASH - 500.0, total_value=INITIAL_CASH, positions=[])
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == INITIAL_CASH - 500.0
        assert pf["total_value"] == INITIAL_CASH  # mark-to-market not yet applied


class TestUpsertPosition:
    def test_first_buy_creates_position(self, fresh_store):
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        positions = fresh_store.open_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "AMD"
        assert positions[0]["qty"] == 5
        assert positions[0]["avg_cost"] == 100.0

    def test_second_buy_blends_avg_cost(self, fresh_store):
        # First lot: 10 @ 100.  Second lot: 10 @ 120.  Blended = 110.
        fresh_store.upsert_position("AMD", "stock", qty=10, avg_cost=100.0)
        fresh_store.upsert_position("AMD", "stock", qty=10, avg_cost=120.0)
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        assert pos[0]["qty"] == 20
        assert pos[0]["avg_cost"] == pytest.approx(110.0)

    def test_partial_sell_keeps_avg_cost(self, fresh_store):
        # Open 10 @ 100.  Sell 4 @ 130 → 6 remain at avg_cost 100 (cost basis unchanged).
        fresh_store.upsert_position("AMD", "stock", qty=10, avg_cost=100.0)
        fresh_store.upsert_position("AMD", "stock", qty=-4, avg_cost=130.0)
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        assert pos[0]["qty"] == 6
        # avg_cost should NOT change on a sell — the blended formula bypasses
        # for qty <= 0 specifically to preserve cost basis.
        assert pos[0]["avg_cost"] == pytest.approx(100.0)

    def test_full_sell_closes_position(self, fresh_store):
        fresh_store.upsert_position("AMD", "stock", qty=10, avg_cost=100.0)
        fresh_store.upsert_position("AMD", "stock", qty=-10, avg_cost=110.0)
        # open_positions() filters closed_at IS NULL AND qty > 0; this should be empty.
        assert fresh_store.open_positions() == []

    def test_overselling_closes_position(self, fresh_store):
        # Defensive: even if we slip past the pre-trade check, an oversell
        # should NOT leave a negative quantity dangling.
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        fresh_store.upsert_position("AMD", "stock", qty=-10, avg_cost=110.0)
        assert fresh_store.open_positions() == []

    def test_options_and_stock_are_separate_positions(self, fresh_store):
        # Same ticker, different type → distinct rows.
        fresh_store.upsert_position("NVDA", "stock", qty=10, avg_cost=500.0)
        fresh_store.upsert_position("NVDA", "call", qty=2, avg_cost=15.0,
                                    expiry="2026-12-19", strike=600.0)
        positions = fresh_store.open_positions()
        types = {p["type"] for p in positions}
        assert types == {"stock", "call"}

    def test_different_strikes_are_separate_positions(self, fresh_store):
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=10.0,
                                    expiry="2026-12-19", strike=600.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=20.0,
                                    expiry="2026-12-19", strike=700.0)
        positions = fresh_store.open_positions()
        assert len(positions) == 2
        strikes = sorted(p["strike"] for p in positions)
        assert strikes == [600.0, 700.0]

    def test_reopen_after_close_reactivates_row(self, fresh_store):
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        fresh_store.upsert_position("AMD", "stock", qty=-5, avg_cost=110.0)
        # Re-buy after a full close: the same-key closed row is reactivated
        # (fresh qty/avg/opened_at, closed_at cleared) rather than orphaned.
        fresh_store.upsert_position("AMD", "stock", qty=3, avg_cost=120.0)
        positions = fresh_store.open_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == 3
        assert positions[0]["avg_cost"] == 120.0
        # Reactivation, not a new row: exactly one row exists for the key.
        n_rows = fresh_store.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE ticker='AMD' AND type='stock'"
        ).fetchone()[0]
        assert n_rows == 1

    def test_reopen_option_after_close_does_not_crash(self, fresh_store):
        # Regression: the table-wide UNIQUE(ticker,type,expiry,strike) made a
        # plain INSERT raise IntegrityError when re-entering a previously
        # fully-closed OPTION (non-NULL strike/expiry). That crashed the live
        # cycle mid-_execute, leaving a recorded trade with no position and
        # skipping the cash debit + decision/equity write.
        fresh_store.upsert_position("NVDA", "call", qty=2, avg_cost=5.0,
                                    expiry="2026-05-30", strike=900.0)
        fresh_store.upsert_position("NVDA", "call", qty=-2, avg_cost=6.0,
                                    expiry="2026-05-30", strike=900.0)  # full close
        assert fresh_store.open_positions() == []
        # Re-buy the exact same contract — must not raise.
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=7.0,
                                    expiry="2026-05-30", strike=900.0)
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        assert pos[0]["type"] == "call"
        assert pos[0]["qty"] == 1
        assert pos[0]["avg_cost"] == 7.0
        assert pos[0]["strike"] == 900.0
        assert pos[0]["expiry"] == "2026-05-30"
        # current_price / unrealized_pl reset on reactivation (no stale marks).
        assert pos[0]["current_price"] == 0
        assert pos[0]["unrealized_pl"] == 0
        # A subsequent add still blends into the reactivated lot, not a 3rd row.
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=9.0,
                                    expiry="2026-05-30", strike=900.0)
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        assert pos[0]["qty"] == 2
        assert pos[0]["avg_cost"] == pytest.approx(8.0)  # (1*7 + 1*9)/2
        n_rows = fresh_store.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE ticker='NVDA' AND type='call'"
        ).fetchone()[0]
        assert n_rows == 1


class TestTradesOrdering:
    def test_recent_trades_most_recent_first(self, fresh_store):
        # Insert in known order; recent_trades should reverse.
        fresh_store.record_trade("AAA", "BUY", 1, 1.0, "first")
        fresh_store.record_trade("BBB", "BUY", 1, 2.0, "second")
        fresh_store.record_trade("CCC", "BUY", 1, 3.0, "third")
        trades = fresh_store.recent_trades(limit=10)
        assert [t["ticker"] for t in trades] == ["CCC", "BBB", "AAA"]

    def test_recent_trades_respects_limit(self, fresh_store):
        for i in range(5):
            fresh_store.record_trade(f"T{i}", "BUY", 1, 10.0, "")
        assert len(fresh_store.recent_trades(limit=2)) == 2

    def test_record_trade_value_for_stock(self, fresh_store):
        fresh_store.record_trade("AMD", "BUY", qty=4, price=25.0, reason="")
        t = fresh_store.recent_trades(1)[0]
        # qty * price for stocks (no 100x multiplier).
        assert t["value"] == 100.0

    def test_record_trade_value_for_option(self, fresh_store):
        # qty * price * 100 for options.
        fresh_store.record_trade("NVDA", "BUY_CALL", qty=2, price=5.0, reason="",
                                 option_type="call", strike=600.0, expiry="2026-12-19")
        t = fresh_store.recent_trades(1)[0]
        assert t["value"] == 2 * 5.0 * 100  # = 1000.0


class TestEquityCurve:
    def test_equity_curve_returns_ascending(self, fresh_store):
        # Recording 3 points in chronological order; equity_curve should
        # return them oldest-first (DESC then reversed).
        fresh_store.record_equity_point(1000.0, 1000.0, None)
        fresh_store.record_equity_point(1010.0, 990.0, None)
        fresh_store.record_equity_point(1020.0, 980.0, None)
        eq = fresh_store.equity_curve(limit=10)
        assert len(eq) == 3
        # Total values should be monotonically increasing (matches insert order).
        values = [p["total_value"] for p in eq]
        assert values == sorted(values)

    def test_equity_curve_limit(self, fresh_store):
        for i in range(7):
            fresh_store.record_equity_point(1000.0 + i, 1000.0, None)
        eq = fresh_store.equity_curve(limit=3)
        assert len(eq) == 3
        # The 3 most recent are the highest values 1004, 1005, 1006 ascending.
        assert [p["total_value"] for p in eq] == [1004.0, 1005.0, 1006.0]


class TestDecisions:
    def test_record_decision_returns_id(self, fresh_store):
        rid = fresh_store.record_decision(True, 5, "BUY AMD → FILLED", "{}", 1000.0, 500.0)
        assert rid == 1
        rid2 = fresh_store.record_decision(True, 5, "HOLD → HOLD", "{}", 1000.0, 500.0)
        assert rid2 == 2

    def test_recent_decisions_ordering(self, fresh_store):
        fresh_store.record_decision(True, 1, "first", "", 0, 0)
        fresh_store.record_decision(True, 2, "second", "", 0, 0)
        recs = fresh_store.recent_decisions(limit=5)
        actions = [r["action_taken"] for r in recs]
        assert actions == ["second", "first"]


class TestUpdatePositionMarks:
    def test_marks_persist(self, fresh_store):
        fresh_store.upsert_position("NVDA", "stock", qty=10, avg_cost=100.0)
        pid = fresh_store.open_positions()[0]["id"]
        fresh_store.update_position_marks({pid: (120.0, 200.0)})
        pos = fresh_store.open_positions()[0]
        assert pos["current_price"] == 120.0
        assert pos["unrealized_pl"] == 200.0


class TestGetPortfolioResilience:
    """get_portfolio() must never 500 the dashboard on a transient bad row.

    The sqlite connection is shared (check_same_thread=False) between the
    runner's writer thread and the Flask dashboard threads. Even with the
    read lock in place, that connection intermittently yields a None row or a
    row whose columns read back NULL (documented in store.py and observed as
    28x /api/state 500s in runner.log over 2 days:
      - TypeError: 'NoneType' object is not subscriptable   (row is None)
      - the JSON object must be str ... not NoneType         (positions_json None)
    get_portfolio() must absorb both instead of crashing /api/state.
    """

    def test_self_heals_when_portfolio_row_missing(self, fresh_store):
        # Reproduces the "row is None" 500: the id=1 row is gone.
        with fresh_store._lock:
            fresh_store.conn.execute("DELETE FROM portfolio")
            fresh_store.conn.commit()
        # Must NOT raise 'NoneType object is not subscriptable'.
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == INITIAL_CASH
        assert pf["total_value"] == INITIAL_CASH
        assert pf["positions"] == []
        # The row must have been re-created so subsequent writes persist.
        row = fresh_store.conn.execute(
            "SELECT COUNT(*) FROM portfolio WHERE id=1"
        ).fetchone()
        assert row[0] == 1

    def test_tolerates_null_positions_json(self, fresh_store, monkeypatch):
        # Reproduces the "json.loads(None)" 500: the row comes back with a
        # NULL positions_json (corrupted read off the shared connection). The
        # NOT NULL schema constraint prevents writing this directly, so we
        # stand in a fake connection that returns exactly that row shape.
        class _FakeCursor:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        class _FakeConn:
            def __init__(self, row):
                self._row = row

            def execute(self, *a, **k):
                return _FakeCursor(self._row)

            def close(self):  # fixture teardown calls Store.close() -> conn.close()
                pass

        corrupt = {
            "cash": 950.0,
            "total_value": 980.0,
            "positions_json": None,
            "last_updated": "2026-05-16T00:00:00Z",
        }
        monkeypatch.setattr(fresh_store, "conn", _FakeConn(corrupt))
        pf = fresh_store.get_portfolio()  # must NOT raise on json.loads(None)
        assert pf["cash"] == 950.0
        assert pf["total_value"] == 980.0
        assert pf["positions"] == []


class TestReadWriteThreadSafety:
    """Regression lock: every public read must serialize on Store._lock.

    The sqlite connection is created check_same_thread=False and shared
    between the runner's writer thread and the Flask dashboard thread(s).
    All writes already hold self._lock; before the fix the read methods
    executed on the shared connection WITHOUT the lock, so a dashboard read
    whose execute() interleaved with a runner write raised
    `sqlite3.InterfaceError: bad parameter or other API misuse` or returned
    a corrupted/None row — the /api/state 500s observed in runner.log
    (11x 'NoneType object is not subscriptable' + InterfaceError).

    Each test deterministically proves the read now blocks while another
    thread holds Store._lock. Pre-fix the unlocked read returned almost
    instantly even with the lock held, so these would fail.
    """

    READ_METHODS = [
        ("get_portfolio", ()),
        ("recent_trades", ()),
        ("open_positions", ()),
        ("recent_decisions", ()),
        ("equity_curve", ()),
    ]

    @pytest.mark.parametrize("method_name,args", READ_METHODS)
    def test_read_serializes_on_lock(self, fresh_store, method_name, args):
        started = threading.Event()
        completed = threading.Event()

        def worker():
            started.set()
            getattr(fresh_store, method_name)(*args)
            completed.set()

        with fresh_store._lock:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            assert started.wait(timeout=2.0), "worker thread never started"
            # While we hold the lock the read must NOT be able to complete.
            assert not completed.wait(timeout=0.5), (
                f"Store.{method_name}() completed while another thread held "
                f"Store._lock — it is reading the shared sqlite connection "
                f"without serializing against writers (concurrent execute() "
                f"-> sqlite3.InterfaceError / None row)."
            )
        # Lock released -> the read must now finish promptly.
        assert completed.wait(timeout=3.0), (
            f"Store.{method_name}() did not complete after the lock released"
        )
        t.join(timeout=2.0)
