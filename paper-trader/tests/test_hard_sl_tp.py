"""Tests for the hard stop-loss / take-profit auto-exit feature
(commit 3176d2f, 2026-05-24).

This feature carries a load-bearing live-trading invariant: every BUY
auto-stamps SL/TP, and every cycle BEFORE Opus sees the prompt the
runner executes mandatory exits against any lot whose mark has breached
its threshold. Two distinct surfaces must agree:

  1. ``store.positions_needing_hard_exit`` — pure SQL that surfaces
     lots eligible for forced exit (qty>0, SL/TP set, mark current).
  2. ``strategy._check_and_execute_hard_exits`` — the runner-side
     executor that records SELL trades + closes the lots + credits cash.

These tests assert SPECIFIC outcomes (cash delta, lot closure, reason
string) rather than "didn't crash" — they would catch:
  * SL/TP percentage being set wrong (e.g. 2% standard vs 4% leveraged)
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
    """Lock the literal SL/TP percentages — these are the load-bearing risk
    settings exposed verbatim to Opus via SYSTEM_PROMPT ('2% below entry',
    '4% for leveraged ETFs', etc.). A typo here silently re-prices every
    auto-exit. SYSTEM_PROMPT documents 2/3 stocks · 4/6 leveraged."""

    def test_standard_stop_loss_is_two_pct(self):
        assert strategy._SL_PCT_STANDARD == 0.02

    def test_standard_take_profit_is_three_pct(self):
        assert strategy._TP_PCT_STANDARD == 0.03

    def test_leveraged_stop_loss_is_four_pct(self):
        assert strategy._SL_PCT_LEVERAGED == 0.04

    def test_leveraged_take_profit_is_six_pct(self):
        assert strategy._TP_PCT_LEVERAGED == 0.06

    def test_15to1_rr_ratio_intact(self):
        """The whole strategy hangs on TP/SL == 1.5 (MACD-strategy convention).
        If someone tweaks SL or TP individually and breaks the ratio, this
        test catches it."""
        assert (strategy._TP_PCT_STANDARD / strategy._SL_PCT_STANDARD) == pytest.approx(1.5)
        assert (strategy._TP_PCT_LEVERAGED / strategy._SL_PCT_LEVERAGED) == pytest.approx(1.5)

    def test_known_leveraged_etfs_in_set(self):
        # Spot-check the headline 3x bulls / 3x bears.
        for tk in ("TQQQ", "SOXL", "UPRO", "SPXL", "SQQQ", "SPXS"):
            assert tk in strategy._LEVERAGED_ETFS_SL, f"{tk} missing from leveraged set"

    def test_non_leveraged_not_in_set(self):
        # A real spot-stock / non-leveraged ETF must not be in the leveraged
        # bucket — would otherwise get the wider 4/6 stops by accident.
        for tk in ("NVDA", "AAPL", "SPY", "QQQ", "MU"):
            assert tk not in strategy._LEVERAGED_ETFS_SL, (
                f"{tk} incorrectly in leveraged set — would get 4/6 instead "
                f"of 2/3 stops")


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

    def test_returns_lot_breaching_take_profit(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (104.0, 4.0)})
        rows = fresh_store.positions_needing_hard_exit()
        assert len(rows) == 1

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

    def test_exact_take_profit_boundary_breaches(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (103.0, 3.0)})
        assert len(fresh_store.positions_needing_hard_exit()) == 1

    def test_skips_lot_with_null_sl(self, fresh_store):
        """A lot without SL/TP set (e.g. legacy pre-migration) must NEVER
        be auto-exited — the SQL guard requires both NOT NULL."""
        fresh_store.upsert_position("NVDA", "stock", 1.0, 100.0)  # no SL/TP
        fresh_store.update_position_marks({1: (50.0, -50.0)})  # huge loss
        assert fresh_store.positions_needing_hard_exit() == []

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
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
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

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
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

    def test_tp_breach_closes_lot_and_credits_cash(self, fresh_store):
        fresh_store.upsert_position(
            "NVDA", "stock", 2.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (104.0, 8.0)})
        snap = {"cash": 800.0, "total_value": 1008.0}

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
        assert exits == ["NVDA"]
        # 2 * 104 = $208 → cash $1008
        assert snap["cash"] == pytest.approx(1008.0)
        trades = fresh_store.recent_trades(1)
        assert "HARD_TP" in trades[0]["reason"]
        assert trades[0]["qty"] == 2.0

    def test_multiple_breaches_in_one_cycle(self, fresh_store):
        """Two lots breach simultaneously; both must be exited and snap
        cash must reflect BOTH credits in order."""
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
            2: (52.0, 10.0),     # AMD TP breach
        })
        snap = {"cash": 650.0, "total_value": 1009.0}

        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
        assert set(exits) == {"NVDA", "AMD"}
        # NVDA +$97, AMD +5*52=$260 → cash should be 650 + 97 + 260 = $1007
        assert snap["cash"] == pytest.approx(1007.0)
        assert fresh_store.open_positions() == []

    def test_skips_lot_with_zero_qty(self, fresh_store):
        """An ALREADY-flat lot (qty=0) must not generate a phantom exit."""
        fresh_store.upsert_position(
            "NVDA", "stock", 1.0, 100.0,
            stop_loss_price=98.0, take_profit_price=103.0,
        )
        fresh_store.update_position_marks({1: (97.0, -3.0)})
        fresh_store.close_position(1)  # qty=0, closed_at set
        snap = {"cash": 900.0, "total_value": 900.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
        assert exits == []

    def test_non_fatal_on_store_exception(self, fresh_store, monkeypatch):
        """A diagnostics fault must NEVER abort the cycle — returns []."""
        def _raise(*a, **k):
            raise RuntimeError("simulated DB blow-up")
        monkeypatch.setattr(
            fresh_store, "positions_needing_hard_exit", _raise
        )
        snap = {"cash": 900.0, "total_value": 900.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap)
        assert exits == []


# ────────────────── BUY path stamps SL/TP correctly ─────────────────


class TestBuyStampsSLTP:
    """The live BUY path computes SL/TP off the just-traded price and stamps
    them onto the lot via a follow-up metadata-only upsert. These tests
    pin that the stamped values are CORRECT (right percentage, right
    rounding, right leveraged-vs-standard discrimination)."""

    def test_buy_stock_stamps_2_3_pct(self, fresh_store, monkeypatch):
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
        # 100 * 0.98 = 98.00, 100 * 1.03 = 103.00
        assert pos[0]["stop_loss_price"] == pytest.approx(98.0)
        assert pos[0]["take_profit_price"] == pytest.approx(103.0)

    def test_buy_leveraged_etf_stamps_4_6_pct(self, fresh_store, monkeypatch):
        monkeypatch.setattr(strategy.market, "get_price",
                            lambda *a, **k: 50.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "SOXL", "qty": 1.0,
                    "reasoning": "test"}
        status, _detail = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        pos = fresh_store.open_positions()
        assert len(pos) == 1
        # 50 * 0.96 = 48.00, 50 * 1.06 = 53.00
        assert pos[0]["stop_loss_price"] == pytest.approx(48.0)
        assert pos[0]["take_profit_price"] == pytest.approx(53.0)

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
        assert pos[0]["stop_loss_price"] == pytest.approx(98.0)
        assert pos[0]["take_profit_price"] == pytest.approx(103.0)

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
        # Re-anchored to $110: SL=107.80, TP=113.30
        assert pos[0]["stop_loss_price"] == pytest.approx(107.8)
        assert pos[0]["take_profit_price"] == pytest.approx(113.3)
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
        # Simulate the mark dropping to the SL (98.0)
        fresh_store.update_position_marks({pid: (98.0, -2.0)})

        snap2 = {"cash": 900.0, "total_value": 998.0}
        exits = strategy._check_and_execute_hard_exits(fresh_store, snap2)
        assert exits == ["NVDA"]
        # Cash should now be 900 + 1*98 = $998
        assert snap2["cash"] == pytest.approx(998.0)


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
