"""Tests for the hard stop-loss auto-exit feature
(commit 3176d2f, 2026-05-24).

This feature carries a load-bearing live-trading invariant: every BUY
auto-stamps SL/TP, but only stop-loss is mechanically executed BEFORE Opus
sees the prompt. Take-profit is advisory context for dynamic LLM exits. Two
distinct surfaces must agree:

  1. ``store.positions_needing_hard_exit`` — pure SQL that surfaces
     lots eligible for forced exit (qty>0, SL/TP set, mark current).
  2. ``strategy._check_and_execute_hard_exits`` — the runner-side
     executor that records SELL trades + closes the lots + credits cash.

These tests assert SPECIFIC outcomes (cash delta, lot closure, reason
string) rather than "didn't crash" — they would catch:
  * SL/TP percentage being set wrong (e.g. 5% standard vs 10% leveraged)
  * Off-by-one on the breach predicate (price strictly below SL vs <=)
  * Cash math wrong on the SELL (price * qty)
  * The lot not being marked closed
  * The reason string missing critical operator-facing context
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import strategy
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


# ─────────────────────────── constants sanity ───────────────────────────


class TestSLTPConstants:
    """Lock the literal SL/TP percentages — SL is mechanically enforced and TP
    is advisory context exposed to Opus.
    settings exposed verbatim to Opus via SYSTEM_PROMPT ('5% below entry',
    '10% for leveraged ETFs', etc.). A typo here silently re-prices every
    auto-exit. SYSTEM_PROMPT documents 5/15 stocks · 10/25 leveraged."""

    def test_standard_stop_loss_is_five_pct(self):
        assert strategy._SL_PCT_STANDARD == 0.05

    def test_standard_take_profit_is_fifteen_pct(self):
        assert strategy._TP_PCT_STANDARD == 0.15

    def test_leveraged_stop_loss_is_ten_pct(self):
        assert strategy._SL_PCT_LEVERAGED == 0.10

    def test_leveraged_take_profit_is_twenty_five_pct(self):
        assert strategy._TP_PCT_LEVERAGED == 0.25

    def test_advisory_tp_is_materially_wider_than_stop(self):
        """TP is advisory now; keep it far enough away that it cannot become
        a scalp target in prompt context."""
        assert (strategy._TP_PCT_STANDARD / strategy._SL_PCT_STANDARD) == pytest.approx(3.0)
        assert (strategy._TP_PCT_LEVERAGED / strategy._SL_PCT_LEVERAGED) == pytest.approx(2.5)

    def test_known_leveraged_etfs_in_set(self):
        # Spot-check the headline 3x bulls / 3x bears.
        for tk in ("TQQQ", "SOXL", "UPRO", "SPXL", "SQQQ", "SPXS"):
            assert tk in strategy._LEVERAGED_ETFS_SL, f"{tk} missing from leveraged set"

    def test_non_leveraged_not_in_set(self):
        # A real spot-stock / non-leveraged ETF must not be in the leveraged
        # bucket — would otherwise get the wider 10/25 stops by accident.
        for tk in ("NVDA", "AAPL", "SPY", "QQQ", "MU"):
            assert tk not in strategy._LEVERAGED_ETFS_SL, (
                f"{tk} incorrectly in leveraged set — would get 10/25 instead "
                f"of 5/15 stops")


# ────────────────────── store.positions_needing_hard_exit ───────────────────────


class TestPositionsNeedingHardExit:
    """Pure SQL — surfaces the lots a cycle SHOULD auto-exit."""

    def test_returns_empty_on_fresh_store(self, fresh_store):
        assert fresh_store.positions_needing_hard_exit() == []

    def test_returns_lot_breaching_stop_loss(self, fresh_store):
        # Buy NVDA at $100, mark at $97 (below SL=$98 → breach)
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        rows = fresh_store.positions_needing_hard_exit()
        assert len(rows) == 1
        assert rows[0]["ticker"] == "NVDA"

    def test_take_profit_breach_is_advisory_not_hard_exit(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (104.0, 4.0)})
        rows = fresh_store.positions_needing_hard_exit()
        assert rows == []

    def test_skips_lot_within_band(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        # Mark at $101 — between SL and TP, no exit
        fresh_store.update_position_marks({1: (101.0, 1.0)})
        assert fresh_store.positions_needing_hard_exit() == []

    def test_exact_stop_loss_boundary_breaches(self, fresh_store):
        """The predicate is `current_price <= stop_loss_price` (inclusive).
        Exactly AT the SL must trigger — otherwise an exact-tick SL gets
        held forever."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (98.0, -2.0)})
        rows = fresh_store.positions_needing_hard_exit()
        assert len(rows) == 1, "exact SL price must trigger exit"

    def test_exact_take_profit_boundary_is_not_hard_exit(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (103.0, 3.0)})
        assert fresh_store.positions_needing_hard_exit() == []

    def test_skips_lot_with_null_sl(self, fresh_store):
        """A lot without SL set (e.g. legacy pre-migration) must NEVER
        be auto-exited — the SQL guard requires stop_loss_price."""
        fresh_store.upsert_position("NVDA", "stock", 1.0, 100.0)  # no SL/TP
        fresh_store.update_position_marks({1: (50.0, -50.0)})  # huge loss
        assert fresh_store.positions_needing_hard_exit() == []

    def test_stop_loss_fires_even_without_take_profit(self, fresh_store):
        """TP is advisory now; a missing TP marker must not disable stop-loss."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=None,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        rows = fresh_store.positions_needing_hard_exit()
        assert len(rows) == 1
        assert rows[0]["ticker"] == "NVDA"

    def test_skips_lot_with_stale_mark(self, fresh_store):
        """current_price == 0 is the snapshot-freshness proxy. A lot
        with no fresh mark must not fire (would otherwise instantly
        exit every fresh BUY since current_price defaults to 0 < SL)."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        # No update_position_marks → current_price stays 0
        assert fresh_store.positions_needing_hard_exit() == []

    def test_skips_option_position(self, fresh_store):
        """Hard exits are stock-only (options need different logic). An
        option lot with SL/TP set must not fire."""
        fresh_store.upsert_position(
            "NVDA", "call", 1.0, 5.0,
            expiry="2026-06-15", strike=200.0,
            stop_loss_price=4.9, take_profit_price=5.15,
        )
        fresh_store.update_position_marks({1: (4.0, -100.0)})  # breach
        assert fresh_store.positions_needing_hard_exit() == []

    def test_skips_closed_lot(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        fresh_store.close_position(1)
        assert fresh_store.positions_needing_hard_exit() == []


# ────────────────── strategy._check_and_execute_hard_exits ─────────────────


class TestCheckAndExecuteHardExits:
    """The runner-side executor. Verifies cash delta, lot closure, the SELL
    trade row, and the human-readable reason string fired into the trade."""

    def test_no_exits_when_no_breaches(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (101.0, 1.0)})
        snap = {"cash": 900.0, "total_value": 1001.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == []
        assert snap["cash"] == 900.0  # unchanged

    def test_sl_breach_closes_lot_and_credits_cash(self, fresh_store):
        # Pre: $900 cash, 1 NVDA at avg 100, mark 97 (-3% breach of SL=98)
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        snap = {"cash": 900.0, "total_value": 997.0}

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == ["NVDA"]
        # cash credited 1 * 97 = $97 → $997
        assert snap["cash"] == pytest.approx(997.0)
        # lot closed (no longer open)
        assert fresh_store.open_positions() == []
        # SELL trade recorded with HARD_SL reason
        trades = fresh_store.recent_trades(1)
        assert len(trades) == 1
        assert trades[0]["action"] == "SELL"
        assert trades[0]["ticker"] == "NVDA"
        assert trades[0]["qty"] == 1.0
        assert trades[0]["price"] == 97.0
        assert "HARD_SL" in trades[0]["reason"]
        assert "97.00" in trades[0]["reason"]
        assert "98.00" in trades[0]["reason"]

    def test_sl_breach_waits_when_market_closed(self, fresh_store):
        """Premarket/after-hours marks are noisy; hard stops must not
        force-sell outside regular trading hours."""
        fresh_store.upsert_position(
            "MU", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=115.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        snap = {"cash": 900.0, "total_value": 997.0}

        exits = strategy._check_and_execute_hard_exits(
            fresh_store,
            snap,
            market_open=False,
        )

        assert exits == []
        assert snap["cash"] == pytest.approx(900.0)
        assert fresh_store.open_positions()[0]["ticker"] == "MU"
        assert fresh_store.recent_trades(1) == []

    def test_tp_breach_does_not_close_lot_or_credit_cash(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 2.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (104.0, 8.0)})
        snap = {"cash": 800.0, "total_value": 1008.0}

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == []
        assert snap["cash"] == pytest.approx(800.0)
        assert fresh_store.open_positions()[0]["ticker"] == "NVDA"
        assert fresh_store.recent_trades(1) == []

    def test_multiple_breaches_in_one_cycle(self, fresh_store):
        """Only stop-loss breaches are mechanical; TP breaches stay open for
        dynamic LLM review."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.upsert_position(
            "AMD", "stock", 5.0, 50.0,
            stop_loss_price=49.0, take_profit_price=51.5,
        )
        fresh_store.update_position_marks({
            1: (97.0, -3.0),     # NVDA SL breach
            2: (52.0, 10.0),     # AMD TP marker, advisory only
        })
        snap = {"cash": 650.0, "total_value": 1009.0}

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == ["NVDA"]
        assert snap["cash"] == pytest.approx(747.0)
        open_positions = fresh_store.open_positions()
        assert len(open_positions) == 1
        assert open_positions[0]["ticker"] == "AMD"

    def test_skips_lot_with_zero_qty(self, fresh_store):
        """An ALREADY-flat lot (qty=0) must not generate a phantom exit."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        fresh_store.close_position(1)  # qty=0, closed_at set
        snap = {"cash": 900.0, "total_value": 900.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == []

    def test_non_fatal_on_store_exception(self, fresh_store, monkeypatch):
        """A diagnostics fault must NEVER abort the cycle — returns []."""
        def _raise(*a, **k):
            raise RuntimeError("simulated DB blow-up")
        monkeypatch.setattr(
            fresh_store, "positions_needing_hard_exit", _raise
        )
        snap = {"cash": 900.0, "total_value": 900.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap, market_open=True)
        assert exits == []


# ────────────────── BUY path stamps SL/TP correctly ─────────────────


class TestBuyStampsSLTP:
    """The live BUY path computes SL/TP off the just-traded price and stamps
    them onto the lot via a follow-up metadata-only upsert. These tests
    pin that the stamped values are CORRECT (right percentage, right
    rounding, right leveraged-vs-standard discrimination)."""

    def test_buy_stock_stamps_5_15_pct(self, fresh_store, monkeypatch):
        # Avoid touching network
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 100.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "NVDA", "qty": 1.0,
                    "reasoning": "test"}
        status, _detail = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        # 100 * 0.95 = 95.00, 100 * 1.15 = 115.00
        assert pos[0]["stop_loss_price"] == pytest.approx(95.0)
        assert pos[0]["take_profit_price"] == pytest.approx(115.0)

    def test_buy_blocks_nan_price(self, fresh_store, monkeypatch):
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: float("nan"))
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": 1.0, "reasoning": "bad mark"},
            snap,
            fresh_store,
        )
        assert status == "BLOCKED"
        assert detail == "no price for NVDA"
        assert fresh_store.recent_trades(10) == []

    def test_buy_blocks_infinite_price(self, fresh_store, monkeypatch):
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: float("inf"))
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": 1.0, "reasoning": "bad mark"},
            snap,
            fresh_store,
        )
        assert status == "BLOCKED"
        assert detail == "no price for NVDA"
        assert fresh_store.recent_trades(10) == []

    def test_buy_blocks_nan_qty(self, fresh_store, monkeypatch):
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 100.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": float("nan"), "reasoning": "bad qty"},
            snap,
            fresh_store,
        )
        assert status == "BLOCKED"
        assert detail == "qty not finite: nan"
        assert fresh_store.recent_trades(10) == []

    def test_buy_leveraged_etf_stamps_10_25_pct(self, fresh_store, monkeypatch):
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 50.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "SOXL", "qty": 1.0,
                    "reasoning": "test"}
        status, _detail = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        # 50 * 0.90 = 45.00, 50 * 1.25 = 62.50
        assert pos[0]["stop_loss_price"] == pytest.approx(45.0)
        assert pos[0]["take_profit_price"] == pytest.approx(62.5)

    def test_buy_add_reanchors_sl_tp_to_latest_price(self, fresh_store, monkeypatch):
        """On an add, SL/TP must be re-anchored to the NEW price so an add
        at a higher price doesn't leave SL absurdly close to the original
        entry (or, on an averaging-down add, leave TP unreachable)."""
        # First BUY at $100
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 100.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": 1.0, "reasoning": "i"},
            snap, fresh_store,
        )
        pos = fresh_store.open_positions()
        assert pos[0]["stop_loss_price"] == pytest.approx(95.0)
        assert pos[0]["take_profit_price"] == pytest.approx(115.0)

        # Second BUY at $110 — add
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 110.0)
        snap2 = {"cash": 900.0, "total_value": 1010.0,
                 "positions": [{
                     "ticker": "NVDA", "type": "stock", "qty": 1.0,
                     "avg_cost": 100.0,
                 }]}
        strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": 1.0, "reasoning": "add"},
            snap2, fresh_store,
        )
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        # Re-anchored to $110: SL=104.50, TP=126.50
        assert pos[0]["stop_loss_price"] == pytest.approx(104.5)
        assert pos[0]["take_profit_price"] == pytest.approx(126.5)
        # Blended avg_cost is 105 (1*100 + 1*110) / 2
        assert pos[0]["avg_cost"] == pytest.approx(105.0)

    def test_buy_then_immediate_breach_can_auto_exit(self, fresh_store, monkeypatch):
        """End-to-end: BUY → mark falls to SL → auto-exit fires."""
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 100.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        strategy._execute(
            {"action": "BUY", "ticker": "NVDA", "qty": 1.0, "reasoning": "i"},
            snap, fresh_store,
        )
        pos = fresh_store.open_positions()
        pid = pos[0]["id"]
        # Simulate the mark dropping to the SL (95.0)
        fresh_store.update_position_marks({pid: (95.0, -5.0)})

        snap2 = {"cash": 900.0, "total_value": 995.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap2, market_open=True)
        assert exits == ["NVDA"]
        # Cash should now be 900 + 1*95 = $995
        assert snap2["cash"] == pytest.approx(995.0)


# ────────────────── store.upsert_position SL/TP semantics ─────────────────


class TestUpsertPositionSLTPSemantics:
    """The store-level contract around the SL/TP fields — the metadata-only
    path (qty=0), reactivation of a closed lot, and partial-update
    behaviour."""

    def test_qty_zero_metadata_only_path_sets_only_sl_tp(self, fresh_store):
        """qty=0 must NOT change the size; only the SL/TP fields move."""
        fresh_store.upsert_position("NVDA", "stock", 1.0, 100.0)
        # Stamp SL/TP only
        fresh_store.upsert_position(
            "NVDA", "stock", 0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        pos = fresh_store.open_positions()
        assert pos[0]["qty"] == 1.0
        assert pos[0]["avg_cost"] == 100.0
        assert pos[0]["stop_loss_price"] == 98.0
        assert pos[0]["take_profit_price"] == 103.0

    def test_qty_zero_with_no_sl_tp_is_pure_noop(self, fresh_store):
        """qty=0 + no SL/TP args = no-op (doesn't clear existing SL/TP)."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.upsert_position("NVDA", "stock", 0, 100.0)
        pos = fresh_store.open_positions()
        assert pos[0]["stop_loss_price"] == 98.0
        assert pos[0]["take_profit_price"] == 103.0

    def test_reactivation_with_sl_tp(self, fresh_store):
        """A previously fully-closed lot reactivates on the next BUY with the
        passed SL/TP (covers a partial-close → re-entry sequence)."""
        fresh_store.upsert_position("NVDA", "stock", 1.0, 100.0)
        fresh_store.upsert_position("NVDA", "stock", -1.0, 100.0)  # close
        assert fresh_store.open_positions() == []
        # Re-enter with SL/TP
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 110.0,
            stop_loss_price=107.8, take_profit_price=113.3,
        )
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        assert pos[0]["qty"] == 1.0
        assert pos[0]["avg_cost"] == 110.0
        assert pos[0]["stop_loss_price"] == 107.8
        assert pos[0]["take_profit_price"] == 113.3
