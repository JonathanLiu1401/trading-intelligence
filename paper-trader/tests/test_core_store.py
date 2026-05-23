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

    def test_update_portfolio_positions_none_preserves_existing_json(self, fresh_store):
        """``positions=None`` updates cash/total_value but does NOT touch
        ``positions_json``. Used by strategy._execute mid-trade so the
        positions list written at the start of the cycle (by
        ``_portfolio_snapshot``) is not overwritten with a stale snapshot."""
        seeded = [
            {"ticker": "NVDA", "type": "stock", "qty": 5, "avg_cost": 100.0,
             "current_price": 110.0, "unrealized_pl": 50.0, "market_value": 550.0},
        ]
        fresh_store.update_portfolio(
            cash=450.0, total_value=1000.0, positions=seeded)
        # Cash-only update via positions=None (omitted)
        fresh_store.update_portfolio(cash=320.0, total_value=1000.0)
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == 320.0
        assert pf["total_value"] == 1000.0
        # positions_json must still carry the seeded list — NOT cleared to []
        assert pf["positions"] == seeded

    def test_update_portfolio_positions_none_default(self, fresh_store):
        """The default of `positions=None` (omitted) preserves positions_json
        — back-compat guard for new callers that forget to pass positions."""
        seeded = [{"ticker": "MU", "type": "stock", "qty": 2,
                   "avg_cost": 50.0, "current_price": 55.0,
                   "unrealized_pl": 10.0, "market_value": 110.0}]
        fresh_store.update_portfolio(cash=890.0, total_value=1000.0,
                                     positions=seeded)
        fresh_store.update_portfolio(cash=800.0, total_value=1000.0)
        pf = fresh_store.get_portfolio()
        assert pf["positions"] == seeded
        assert pf["cash"] == 800.0

    def test_update_portfolio_positions_empty_list_clears_json(self, fresh_store):
        """Empty list `[]` MUST clear positions_json — distinct from None.
        A None silently preserves; an empty list explicitly empties."""
        seeded = [{"ticker": "AMD", "type": "stock", "qty": 3,
                   "avg_cost": 100.0, "current_price": 110.0,
                   "unrealized_pl": 30.0, "market_value": 330.0}]
        fresh_store.update_portfolio(cash=670.0, total_value=1000.0,
                                     positions=seeded)
        fresh_store.update_portfolio(cash=1000.0, total_value=1000.0,
                                     positions=[])
        pf = fresh_store.get_portfolio()
        assert pf["positions"] == []


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


class TestLastRealDecision:
    """``store.last_real_decision`` answers a trader's question
    ``recent_decisions(1)[0]`` cannot: "when did the engine actually decide
    something, not just cycle?" Without this, a 24h NO_DECISION storm reads as
    "last decision 6m ago" — a green light on the operator's primary surface
    while the engine has not produced a parseable Claude response for days.
    These tests pin the filter, the ordering, and the None contract."""

    def test_none_when_empty(self, fresh_store):
        """A fresh book with no decisions returns None — not an exception,
        not a fabricated empty row."""
        assert fresh_store.last_real_decision() is None

    def test_filters_out_no_decision(self, fresh_store):
        """A book whose entire decision history is NO_DECISION rows returns
        None — the exact 24h IDLE_STORM regime the method is built to
        diagnose."""
        for _ in range(10):
            fresh_store.record_decision(
                False, 0, "NO_DECISION",
                "skipped claude call — host saturated", 1000.0, 1000.0,
            )
        assert fresh_store.last_real_decision() is None

    def test_returns_most_recent_filled(self, fresh_store):
        """Walk through a typical sequence: a real BUY → many NO_DECISIONs
        → the real BUY is the one returned (not the most recent
        NO_DECISION row that ``recent_decisions(1)`` would surface)."""
        # Real BUY (action verb + → status — the canonical record_decision
        # shape strategy._execute writes).
        fresh_store.record_decision(
            True, 5, "BUY NVDA → FILLED", "earnings", 1000.0, 500.0,
        )
        # Many NO_DECISION cycles after the real decision.
        for _ in range(20):
            fresh_store.record_decision(
                False, 0, "NO_DECISION",
                "skipped claude call — host saturated", 990.0, 500.0,
            )
        # recent_decisions(1) returns a NO_DECISION row — the documented
        # IDLE_STORM "green light" failure mode.
        assert fresh_store.recent_decisions(1)[0]["action_taken"] == "NO_DECISION"
        # last_real_decision returns the actual BUY.
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "BUY NVDA → FILLED"
        assert last["reasoning"] == "earnings"

    def test_includes_hold_and_blocked(self, fresh_store):
        """HOLD and BLOCKED are *real* decisions — a deliberate hold or a
        risk-rejected trade are meaningfully different from a cycle that
        couldn't even produce a parseable Claude response. The filter must
        keep them; only the literal NO_DECISION cycle is excluded."""
        fresh_store.record_decision(True, 5, "HOLD MU → HOLD", "wait", 1000.0, 500.0)
        # NO_DECISION newer than the HOLD — must be filtered out so the
        # HOLD is what's returned.
        fresh_store.record_decision(False, 0, "NO_DECISION", "skipped", 1000.0, 500.0)
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "HOLD MU → HOLD"

        # BLOCKED — same contract.
        fresh_store.record_decision(
            True, 5, "SELL NVDA → BLOCKED", "oversell", 1000.0, 500.0,
        )
        fresh_store.record_decision(False, 0, "NO_DECISION", "skipped", 1000.0, 500.0)
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "SELL NVDA → BLOCKED"

    def test_picks_newest_real_decision(self, fresh_store):
        """When multiple real decisions exist, returns the most recent —
        same ``ORDER BY timestamp DESC, id DESC`` semantics as
        ``recent_decisions``."""
        fresh_store.record_decision(True, 5, "BUY AMD → FILLED", "first", 1000.0, 500.0)
        fresh_store.record_decision(True, 5, "BUY NVDA → FILLED", "second", 1000.0, 500.0)
        fresh_store.record_decision(True, 5, "HOLD MU → HOLD", "third", 1000.0, 500.0)
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "HOLD MU → HOLD"
        assert last["reasoning"] == "third"

    def test_filters_no_decision_prefix_rows(self, fresh_store):
        """Some historical rows carry a NO_DECISION-prefixed action_taken
        (e.g. ``"NO_DECISION (host saturated)"`` — defensive against the
        free-text column shape; AGENTS.md invariant #11). The filter's
        ``NOT LIKE 'NO_DECISION%'`` arm catches these too."""
        fresh_store.record_decision(True, 5, "BUY NVDA → FILLED", "real", 1000.0, 500.0)
        # Prefix variant - must still be filtered.
        fresh_store.record_decision(
            False, 0, "NO_DECISION (host saturated)", "", 1000.0, 500.0,
        )
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "BUY NVDA → FILLED"

    def test_empty_string_action_filtered(self, fresh_store):
        """A row with an empty ``action_taken`` (defensive — never written
        by strategy.py, but the schema allows NULL/empty so the test pins
        the filter) is treated as not-a-real-decision."""
        # Real BUY.
        fresh_store.record_decision(True, 5, "BUY NVDA → FILLED", "real", 1000.0, 500.0)
        # Empty action_taken — must be filtered out.
        fresh_store.record_decision(False, 0, "", "", 1000.0, 500.0)
        last = fresh_store.last_real_decision()
        assert last is not None
        assert last["action_taken"] == "BUY NVDA → FILLED"


class TestUpdatePositionMarks:
    def test_marks_persist(self, fresh_store):
        fresh_store.upsert_position("NVDA", "stock", qty=10, avg_cost=100.0)
        pid = fresh_store.open_positions()[0]["id"]
        fresh_store.update_position_marks({pid: (120.0, 200.0)})
        pos = fresh_store.open_positions()[0]
        assert pos["current_price"] == 120.0
        assert pos["unrealized_pl"] == 200.0


class TestClosedPositionsRealizedPL:
    """``store.closed_positions`` rolls up realized P/L per closed lot by
    summing matching trade rows inside the lot's [opened_at, closed_at]
    window. Each of these locks a specific behaviour a trader relies on
    when reading ``/api/closed-positions`` after the bell."""

    def test_stock_round_trip_realized_pl(self, fresh_store):
        """A simple BUY → SELL on a stock at a higher price.

        Cost  = 10 * 50 = 500
        Proceeds = 10 * 60 = 600
        Realized P/L = +100. The endpoint's % framing is gross P/L / cost.
        """
        fresh_store.record_trade("AMD", "BUY", qty=10, price=50.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=10, avg_cost=50.0)
        fresh_store.record_trade("AMD", "SELL", qty=10, price=60.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=-10, avg_cost=60.0)
        closed = fresh_store.closed_positions()
        assert len(closed) == 1
        c = closed[0]
        assert c["ticker"] == "AMD"
        assert c["realized_pl"] == 100.0
        assert c["cost"] == 500.0
        assert c["proceeds"] == 600.0
        assert c["realized_pl_pct"] == 20.0  # 100 / 500 * 100

    def test_stock_losing_round_trip(self, fresh_store):
        """A BUY → SELL at a lower price → negative realized."""
        fresh_store.record_trade("AMD", "BUY", qty=5, price=100.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        fresh_store.record_trade("AMD", "SELL", qty=5, price=80.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=-5, avg_cost=80.0)
        c = fresh_store.closed_positions()[0]
        # Loss: 5 * (80 - 100) = -100
        assert c["realized_pl"] == -100.0
        assert c["realized_pl_pct"] == -20.0

    def test_option_round_trip_realized_pl_is_not_zero(self, fresh_store):
        """REGRESSION: an option round-trip used to report $0 realized.

        The OLD code matched only ``("SELL","CLOSE","SELL_TO_CLOSE")`` and
        ``("BUY","OPEN","BUY_TO_OPEN")`` — exact strings — so the live
        trader's ``BUY_CALL`` / ``SELL_CALL`` actions silently fell out of
        both arms and every option close on /api/closed-positions read as
        breakeven. The fix uses ``startswith("BUY")`` / ``startswith("SELL")``
        which matches every documented entry/exit action.

        Cost  = 2 * 5  * 100 = 1000  (the contract multiplier)
        Proceeds = 2 * 8  * 100 = 1600
        Realized P/L = +600.
        """
        fresh_store.record_trade(
            "NVDA", "BUY_CALL", qty=2, price=5.0, reason="bullish",
            option_type="call", strike=600.0, expiry="2026-12-19",
        )
        fresh_store.upsert_position(
            "NVDA", "call", qty=2, avg_cost=5.0,
            expiry="2026-12-19", strike=600.0,
        )
        fresh_store.record_trade(
            "NVDA", "SELL_CALL", qty=2, price=8.0, reason="profit-take",
            option_type="call", strike=600.0, expiry="2026-12-19",
        )
        fresh_store.upsert_position(
            "NVDA", "call", qty=-2, avg_cost=8.0,
            expiry="2026-12-19", strike=600.0,
        )
        closed = fresh_store.closed_positions()
        assert len(closed) == 1, "the call lot must appear in closed_positions"
        c = closed[0]
        assert c["type"] == "call"
        assert c["ticker"] == "NVDA"
        assert c["strike"] == 600.0
        assert c["expiry"] == "2026-12-19"
        # The bug pinning: this MUST NOT be zero.
        assert c["realized_pl"] != 0, (
            "option round-trip realized_pl reported $0 — the action filter "
            "is missing BUY_CALL / SELL_CALL (regression)"
        )
        assert c["realized_pl"] == 600.0
        assert c["cost"] == 1000.0
        assert c["proceeds"] == 1600.0
        assert c["realized_pl_pct"] == 60.0  # 600 / 1000 * 100

    def test_put_round_trip_losing(self, fresh_store):
        """SAME regression on the PUT path — BUY_PUT/SELL_PUT must also
        contribute to realized."""
        fresh_store.record_trade(
            "SPY", "BUY_PUT", qty=1, price=3.0, reason="hedge",
            option_type="put", strike=500.0, expiry="2026-06-19",
        )
        fresh_store.upsert_position(
            "SPY", "put", qty=1, avg_cost=3.0,
            expiry="2026-06-19", strike=500.0,
        )
        fresh_store.record_trade(
            "SPY", "SELL_PUT", qty=1, price=1.0, reason="bleeding",
            option_type="put", strike=500.0, expiry="2026-06-19",
        )
        fresh_store.upsert_position(
            "SPY", "put", qty=-1, avg_cost=1.0,
            expiry="2026-06-19", strike=500.0,
        )
        c = fresh_store.closed_positions()[0]
        # Cost = 1 * 3 * 100 = 300; Proceeds = 1 * 1 * 100 = 100; PL = -200.
        assert c["realized_pl"] == -200.0
        assert c["realized_pl_pct"] == pytest.approx(-66.67, abs=0.01)

    def test_partial_close_then_full_close(self, fresh_store):
        """Two SELLs covering a 10-share BUY: realized aggregates across
        them once the lot fully closes."""
        fresh_store.record_trade("MU", "BUY", qty=10, price=100.0, reason="")
        fresh_store.upsert_position("MU", "stock", qty=10, avg_cost=100.0)
        fresh_store.record_trade("MU", "SELL", qty=4, price=110.0, reason="")
        fresh_store.upsert_position("MU", "stock", qty=-4, avg_cost=110.0)
        # Still open — should not appear in closed_positions yet.
        assert fresh_store.closed_positions() == []
        fresh_store.record_trade("MU", "SELL", qty=6, price=120.0, reason="")
        fresh_store.upsert_position("MU", "stock", qty=-6, avg_cost=120.0)
        c = fresh_store.closed_positions()[0]
        # Cost = 1000. Proceeds = 4*110 + 6*120 = 440 + 720 = 1160.
        assert c["realized_pl"] == 160.0
        assert c["proceeds"] == 1160.0

    def test_hold_duration_fields_present(self, fresh_store):
        """closed_positions surfaces a hold_days / hold_seconds pair so a
        trader can answer 'I made $X in Y days' without re-parsing
        timestamps. Both are computed from the lot's opened_at/closed_at
        which the store maintains for every position."""
        fresh_store.record_trade("AMD", "BUY", qty=1, price=100.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=1, avg_cost=100.0)
        fresh_store.record_trade("AMD", "SELL", qty=1, price=110.0, reason="")
        fresh_store.upsert_position("AMD", "stock", qty=-1, avg_cost=110.0)
        c = fresh_store.closed_positions()[0]
        # Both fields exist on the row (the additive contract).
        assert "hold_days" in c and "hold_seconds" in c
        # The lot was opened and closed in the same test method, so hold time
        # is small but defined and non-negative.
        assert c["hold_seconds"] is not None and c["hold_seconds"] >= 0
        assert c["hold_days"] is not None and c["hold_days"] >= 0

    def test_hold_duration_helper_round_trip(self):
        """``_hold_duration`` is pure; pin its arithmetic deterministically
        so the API-shape test above doesn't have to depend on wall clock."""
        from paper_trader.store import _hold_duration
        secs, days = _hold_duration(
            "2026-05-19T10:00:00+00:00", "2026-05-22T22:00:00+00:00"
        )
        # 3d 12h = 302_400s = 3.5 days exactly.
        assert secs == 302_400
        assert days == 3.5

    def test_hold_duration_helper_handles_bad_input(self):
        """Unparseable / missing endpoints → ``(None, None)``; closed_at
        strictly before opened_at (a non-physical wall-clock step-back) →
        clamps to zero, never returns a negative figure."""
        from paper_trader.store import _hold_duration
        assert _hold_duration(None, "2026-05-19T10:00:00+00:00") == (None, None)
        assert _hold_duration("2026-05-19T10:00:00+00:00", None) == (None, None)
        assert _hold_duration("garbage", "garbage") == (None, None)
        # Reverse order (clock step-back).
        secs, days = _hold_duration(
            "2026-05-22T22:00:00+00:00", "2026-05-19T10:00:00+00:00"
        )
        assert secs == 0
        assert days == 0.0

    def test_summary_win_rate_via_endpoint_shape(self, fresh_store):
        """closed_positions returns newest-closed first — the same order the
        endpoint's summary section relies on for win/loss bucketing."""
        # Winner first (closed earlier).
        fresh_store.record_trade("A", "BUY", qty=1, price=10.0, reason="")
        fresh_store.upsert_position("A", "stock", qty=1, avg_cost=10.0)
        fresh_store.record_trade("A", "SELL", qty=1, price=20.0, reason="")
        fresh_store.upsert_position("A", "stock", qty=-1, avg_cost=20.0)
        # Loser second (closed later).
        fresh_store.record_trade("B", "BUY", qty=1, price=30.0, reason="")
        fresh_store.upsert_position("B", "stock", qty=1, avg_cost=30.0)
        fresh_store.record_trade("B", "SELL", qty=1, price=10.0, reason="")
        fresh_store.upsert_position("B", "stock", qty=-1, avg_cost=10.0)
        lots = fresh_store.closed_positions()
        # Newest-closed first.
        assert [p["ticker"] for p in lots] == ["B", "A"]
        # Per-lot sign matches the price direction.
        assert lots[0]["realized_pl"] < 0
        assert lots[1]["realized_pl"] > 0


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

    def test_transient_null_cash_recovers_real_values(self, fresh_store,
                                                      monkeypatch):
        # The OTHER documented corruption mode the docstring promises to
        # absorb but the original code silently dropped: a present row whose
        # `cash`/`total_value` read back NULL on a transient shared-connection
        # blip. The re-read must recover the real, well-formed values — NOT
        # return None (which TypeErrors strategy._portfolio_snapshot's
        # `None + open_value` and aborts the whole decide() cycle).
        corrupt = {"cash": None, "total_value": None,
                   "positions_json": "[]",
                   "last_updated": "2026-05-16T00:00:00Z"}
        good = {"cash": 712.34, "total_value": 905.10,
                "positions_json": '[{"ticker": "MU"}]',
                "last_updated": "2026-05-16T00:01:00Z"}

        class _SeqCursor:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        class _SeqConn:
            """Returns a truthy row for _init_portfolio's `SELECT id` (so it
            never INSERT/resets a live book) and pops the queued cash-row
            sequence for get_portfolio's read + re-read."""
            def __init__(self, rows):
                self._rows = list(rows)

            def execute(self, sql, *a, **k):
                if "SELECT id FROM portfolio" in sql:
                    return _SeqCursor({"id": 1})
                row = self._rows.pop(0) if self._rows else good
                return _SeqCursor(row)

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(fresh_store, "conn",
                            _SeqConn([corrupt, good]))
        pf = fresh_store.get_portfolio()
        assert pf["cash"] == 712.34          # recovered, not None
        assert pf["total_value"] == 905.10
        assert pf["positions"] == [{"ticker": "MU"}]
        # The exact arithmetic that aborted the cycle pre-fix:
        assert isinstance(pf["cash"] + 0.0, float)

    def test_persistent_null_cash_degrades_safely(self, fresh_store,
                                                  monkeypatch):
        # If the corruption never clears, get_portfolio must still return an
        # arithmetic-safe portfolio (the documented "degrade to in-memory
        # default, never crash" contract — same terminal fallback as the
        # row-is-None path), never None cash/total_value.
        corrupt = {"cash": None, "total_value": None,
                   "positions_json": None,
                   "last_updated": None}

        class _AlwaysCursor:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        class _AlwaysConn:
            def __init__(self, row):
                self._row = row

            def execute(self, sql, *a, **k):
                if "SELECT id FROM portfolio" in sql:
                    return _AlwaysCursor({"id": 1})
                return _AlwaysCursor(self._row)

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(fresh_store, "conn", _AlwaysConn(corrupt))
        pf = fresh_store.get_portfolio()  # must NOT raise
        assert pf["cash"] == INITIAL_CASH
        assert pf["total_value"] == INITIAL_CASH
        assert pf["positions"] == []
        # Proves the live-loop symptom is gone: pre-fix this was `None + x`.
        assert isinstance(pf["total_value"] + 1.0, float)


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
