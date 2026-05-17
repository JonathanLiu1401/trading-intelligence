"""Locking tests for two load-bearing properties of the live trader.

1. Deterministic ordering — store.recent_trades / recent_decisions /
   equity_curve must return newest-first (or insertion-ascending for the
   curve) even when several rows share the *exact same* timestamp string.
   Two writes inside the same microsecond collide on `timestamp` alone, and
   `runner._cycle` calls `recent_trades(1)` right after `_execute` records a
   trade — a wrong-row result there means `send_trade_alert` posts a stale
   trade to Discord. The fix is a `, id DESC` tiebreaker; these tests force
   the collision (by freezing store._now) and pin the behavior so it can't
   silently regress.

2. No hard risk limits (CLAUDE.md / AGENTS.md invariant #2) — the live
   trader intentionally has NO position-size / leverage / stop-loss cap.
   `_enforce_risk_pre_trade` must permit an arbitrarily large BUY and an
   all-in (100%-of-cash) BUY must FILL. The ONLY constraints are: cash may
   not go negative, sells may not exceed held qty. These tests fail loudly
   if a future "fixer" quietly bolts on a cap (which would otherwise pass
   every existing test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import market, strategy
from paper_trader import store as store_mod
from paper_trader.store import Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


# ─────────────────────── deterministic ordering on timestamp collision ───────────────────────

class TestSameTimestampOrdering:
    def test_recent_trades_breaks_tie_on_id(self, fresh_store, monkeypatch):
        # Freeze the clock so every insert gets an identical timestamp string.
        monkeypatch.setattr(store_mod, "_now", lambda: "2026-05-15T12:00:00+00:00")
        fresh_store.record_trade("AAA", "BUY", 1, 1.0, "first")
        fresh_store.record_trade("BBB", "BUY", 1, 2.0, "second")
        fresh_store.record_trade("CCC", "BUY", 1, 3.0, "third")

        # Even with colliding timestamps, newest-inserted must come first.
        assert [t["ticker"] for t in fresh_store.recent_trades(10)] == ["CCC", "BBB", "AAA"]
        # The single-row read used by runner._cycle must be the just-recorded one.
        assert fresh_store.recent_trades(1)[0]["ticker"] == "CCC"

    def test_recent_decisions_breaks_tie_on_id(self, fresh_store, monkeypatch):
        monkeypatch.setattr(store_mod, "_now", lambda: "2026-05-15T12:00:00+00:00")
        fresh_store.record_decision(True, 1, "first", "", 0, 0)
        fresh_store.record_decision(True, 2, "second", "", 0, 0)
        fresh_store.record_decision(True, 3, "third", "", 0, 0)
        assert [d["action_taken"] for d in fresh_store.recent_decisions(10)] == [
            "third", "second", "first"]

    def test_equity_curve_preserves_insertion_order_on_tie(self, fresh_store, monkeypatch):
        monkeypatch.setattr(store_mod, "_now", lambda: "2026-05-15T12:00:00+00:00")
        fresh_store.record_equity_point(1000.0, 1000.0, None)
        fresh_store.record_equity_point(1010.0, 990.0, None)
        fresh_store.record_equity_point(1005.0, 995.0, None)
        # Ascending by (timestamp, id) → exactly the order they were inserted,
        # NOT sorted by value (1005 must stay last, not be reordered before 1010).
        eq = fresh_store.equity_curve(10)
        assert [p["total_value"] for p in eq] == [1000.0, 1010.0, 1005.0]
        # Returned shape must not leak the internal `id` column.
        assert set(eq[0].keys()) == {"timestamp", "total_value", "cash", "sp500_price"}

    def test_equity_curve_limit_with_collision(self, fresh_store, monkeypatch):
        monkeypatch.setattr(store_mod, "_now", lambda: "2026-05-15T12:00:00+00:00")
        for i in range(7):
            fresh_store.record_equity_point(1000.0 + i, 1000.0, None)
        # The 3 most-recently inserted (by id) — 1004, 1005, 1006 — ascending.
        assert [p["total_value"] for p in fresh_store.equity_curve(3)] == [1004.0, 1005.0, 1006.0]


# ─────────────────────── "no hard risk limits" invariant ───────────────────────

class TestNoHardRiskLimits:
    def test_enforce_allows_arbitrarily_large_buy(self):
        """The pre-trade gate must NOT cap position size — only the cash
        check inside _execute can block a BUY (invariant #2)."""
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        ok, why = strategy._enforce_risk_pre_trade(
            {"action": "BUY", "ticker": "TQQQ", "qty": 1_000_000_000}, snap)
        assert ok is True
        assert why == ""

    def test_all_in_100pct_cash_buy_fills(self, fresh_store, monkeypatch):
        """A BUY that consumes 100% of cash is allowed (no position cap).
        notional == cash exactly → not < 0 → FILLED."""
        monkeypatch.setattr(market, "get_price", lambda t: 10.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "BUY", "ticker": "SOXL", "qty": 100, "reasoning": "all-in"},
            snap, fresh_store)
        assert status == "FILLED"
        assert fresh_store.get_portfolio()["cash"] == 0.0

    def test_buy_one_cent_over_cash_is_the_only_block(self, fresh_store, monkeypatch):
        """The sole BUY constraint is "cash must not go negative" — a notional
        strictly greater than cash is blocked; nothing else is."""
        monkeypatch.setattr(market, "get_price", lambda t: 10.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "BUY", "ticker": "SOXL", "qty": 100.001, "reasoning": ""},
            snap, fresh_store)
        assert status == "BLOCKED"
        assert "insufficient cash" in detail

    def test_no_auto_exit_path_in_execute(self, fresh_store):
        """REBALANCE degrades to HOLD and there is no stop-loss / auto-exit
        branch — _execute never force-closes a position on its own."""
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "REBALANCE", "ticker": "NVDA", "qty": 1}, snap, fresh_store)
        assert status == "HOLD"
        assert "REBALANCE" in detail
