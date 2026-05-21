"""Tests for paper_trader.analytics.cost_basis_ladder.

Pins:
* the FIFO lot reconstruction (BUY/SELL ordering, partial fills,
  multi-position isolation, options vs stock disambiguation)
* the per-position verdict ladder
  (LADDER_ALL_GREEN / LADDER_ALL_RED / LADDER_WIDE / LADDER_STACKED /
   LADDER_SINGLE_LOT / NO_LOTS)
* the aggregate verdict (HARVESTABLE_LOTS / UNDERWATER_BOOK /
  MIXED_BOOK / NO_DATA)
* per-lot pl_pct + pl_usd math from current_price
* defensive: malformed rows, missing fields, REBALANCE / non-BUY-non-SELL
  actions are all skipped (never raise)
* threshold-override forwarding (wide_spread_pct / harvest_pct_floor)
* harvestable list is sorted by pl_pct descending
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.cost_basis_ladder import (
    HARVESTABLE_LOTS,
    UNDERWATER_BOOK,
    MIXED_BOOK,
    NO_DATA,
    LADDER_ALL_GREEN,
    LADDER_ALL_RED,
    LADDER_WIDE,
    LADDER_STACKED,
    LADDER_SINGLE_LOT,
    NO_LOTS,
    build_cost_basis_ladder,
)


_NOW = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)


def _ts(offset_minutes: float) -> str:
    return (_NOW - timedelta(minutes=offset_minutes)).isoformat()


def _trade(
    tid: int,
    ticker: str,
    action: str,
    qty: float,
    price: float,
    offset_minutes: float,
    *,
    type_: str = "stock",
    reason: str = "",
    expiry: str | None = None,
    strike: float | None = None,
):
    return {
        "id": tid,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "timestamp": _ts(offset_minutes),
        "reason": reason,
        "type": type_,
        "expiry": expiry,
        "strike": strike,
    }


def _position(
    ticker: str,
    *,
    current_price: float,
    type_: str = "stock",
    expiry: str | None = None,
    strike: float | None = None,
):
    return {
        "ticker": ticker,
        "type": type_,
        "current_price": current_price,
        "expiry": expiry,
        "strike": strike,
        "qty": None,  # builder ignores; FIFO reconstructs
        "avg_cost": None,
        "opened_at": _ts(120),
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "n_positions",
    "n_lots_total", "positions", "harvestable", "thresholds",
}


class TestEnvelopeStability:
    def test_no_positions(self):
        out = build_cost_basis_ladder([], [], now=_NOW)
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == NO_DATA
        assert out["n_positions"] == 0
        assert out["positions"] == []
        assert out["harvestable"] == []

    def test_none_inputs(self):
        out = build_cost_basis_ladder(None, None, now=_NOW)
        assert out["verdict"] == NO_DATA
        assert out["n_positions"] == 0

    def test_position_with_no_trades(self):
        positions = [_position("NVDA", current_price=220.0)]
        out = build_cost_basis_ladder(positions, [], now=_NOW)
        assert out["n_positions"] == 1
        assert out["positions"][0]["verdict"] == NO_LOTS
        assert out["verdict"] == NO_DATA


class TestSingleLot:
    def test_single_buy_above_mark(self):
        positions = [_position("NVDA", current_price=230.0)]
        trades = [_trade(1, "NVDA", "BUY", 2.0, 220.0, 60)]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_SINGLE_LOT
        assert p["n_lots"] == 1
        assert p["lots"][0]["pl_pct"] == 4.55
        assert p["lots"][0]["pl_usd"] == 20.0
        assert p["lots"][0]["trade_id"] == 1
        # +4.55% > harvest_pct_floor (3.0)
        assert out["verdict"] == HARVESTABLE_LOTS
        assert len(out["harvestable"]) == 1
        assert out["harvestable"][0]["ticker"] == "NVDA"

    def test_single_buy_below_mark(self):
        positions = [_position("NVDA", current_price=210.0)]
        trades = [_trade(1, "NVDA", "BUY", 1.0, 220.0, 60)]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_SINGLE_LOT
        assert p["lots"][0]["pl_pct"] == -4.55
        # No harvestable lot; book has only red → UNDERWATER_BOOK
        assert out["verdict"] == UNDERWATER_BOOK


class TestMultiLotLadder:
    def test_wide_spread_triggers_wide_verdict(self):
        positions = [_position("NVDA", current_price=220.0)]
        # Spread: lot1 +10%, lot2 -5% → 15pp spread > 5pp default
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 120),  # +10%
            _trade(2, "NVDA", "BUY", 1.0, 231.5, 60),   # -5%
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_WIDE
        assert p["n_lots"] == 2
        assert p["spread_pct"] >= 14.0
        # The +10% lot harvestable
        assert out["verdict"] == HARVESTABLE_LOTS
        assert out["harvestable"][0]["pl_pct"] == 10.0

    def test_stacked_ladder_within_floor(self):
        positions = [_position("NVDA", current_price=220.0)]
        # Both lots within 5pp spread → STACKED
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 218.0, 120),  # +0.92%
            _trade(2, "NVDA", "BUY", 1.0, 222.0, 60),   # -0.90%
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_STACKED
        assert p["spread_pct"] < 5.0
        # Neither lot clears 3% harvest floor → MIXED_BOOK (green+red)
        assert out["verdict"] == MIXED_BOOK

    def test_all_green_ladder(self):
        positions = [_position("NVDA", current_price=240.0)]
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 180),  # +20%
            _trade(2, "NVDA", "BUY", 1.0, 215.0, 120),  # +11.6%
            _trade(3, "NVDA", "BUY", 1.0, 230.0, 60),   # +4.3%
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_ALL_GREEN
        assert all(l["pl_pct"] > 0 for l in p["lots"])
        # All 3 lots are in profit
        assert p["lots"][0]["pl_pct"] == 20.0
        assert p["lots"][1]["pl_pct"] == 11.63
        assert p["lots"][2]["pl_pct"] == 4.35
        # Harvestable is the SINGLE best lot per position; top is +20%
        assert len(out["harvestable"]) == 1
        assert out["harvestable"][0]["pl_pct"] == 20.0
        assert out["harvestable"][0]["trade_id"] == 1

    def test_all_red_ladder(self):
        positions = [_position("NVDA", current_price=200.0)]
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 220.0, 180),  # -9.1%
            _trade(2, "NVDA", "BUY", 1.0, 215.0, 60),   # -7.0%
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == LADDER_ALL_RED
        assert all(l["pl_pct"] < 0 for l in p["lots"])
        assert out["verdict"] == UNDERWATER_BOOK

    def test_fifo_sell_consumes_oldest_first(self):
        positions = [_position("NVDA", current_price=220.0)]
        # Buy lot1 @ $200 (oldest), lot2 @ $230, then sell 1 share
        # FIFO should consume lot1, leaving only lot2 @ $230
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 180),
            _trade(2, "NVDA", "BUY", 1.0, 230.0, 120),
            _trade(3, "NVDA", "SELL", 1.0, 215.0, 60),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["n_lots"] == 1
        assert p["lots"][0]["price"] == 230.0
        assert p["lots"][0]["trade_id"] == 2

    def test_fifo_partial_sell(self):
        positions = [_position("NVDA", current_price=220.0)]
        # Buy 3 @ $200, sell 1 → 2 left at $200
        trades = [
            _trade(1, "NVDA", "BUY", 3.0, 200.0, 120),
            _trade(2, "NVDA", "SELL", 1.0, 215.0, 60),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["n_lots"] == 1
        assert p["lots"][0]["qty"] == 2.0
        assert p["lots"][0]["price"] == 200.0

    def test_fifo_sell_spanning_multiple_lots(self):
        positions = [_position("NVDA", current_price=220.0)]
        # 2 @ $200, 2 @ $230, sell 3 → 1 @ $230 left
        trades = [
            _trade(1, "NVDA", "BUY", 2.0, 200.0, 180),
            _trade(2, "NVDA", "BUY", 2.0, 230.0, 120),
            _trade(3, "NVDA", "SELL", 3.0, 215.0, 60),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["n_lots"] == 1
        assert p["lots"][0]["price"] == 230.0
        assert p["lots"][0]["qty"] == 1.0


class TestMultiPositionIsolation:
    def test_trades_segregated_by_ticker(self):
        positions = [
            _position("NVDA", current_price=220.0),
            _position("TSLA", current_price=200.0),
        ]
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 120),
            _trade(2, "TSLA", "BUY", 1.0, 220.0, 60),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out["n_positions"] == 2
        by_ticker = {p["ticker"]: p for p in out["positions"]}
        # NVDA: +10%, TSLA: -9.09%
        assert by_ticker["NVDA"]["lots"][0]["pl_pct"] == 10.0
        assert by_ticker["TSLA"]["lots"][0]["pl_pct"] == -9.09
        # Mixed book: green + red, NVDA harvestable
        assert out["verdict"] == HARVESTABLE_LOTS

    def test_options_dont_collide_with_stock_lots(self):
        positions = [
            _position("NVDA", current_price=220.0),
            _position(
                "NVDA", current_price=5.0, type_="call",
                expiry="2026-06-21", strike=225.0,
            ),
        ]
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 120),  # stock
            _trade(
                2, "NVDA", "BUY_CALL", 1.0, 3.5, 60,
                type_="call", expiry="2026-06-21", strike=225.0,
            ),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        # Each position got exactly its own trade
        by_type = {p["type"]: p for p in out["positions"]}
        assert by_type["stock"]["n_lots"] == 1
        assert by_type["stock"]["lots"][0]["price"] == 200.0
        assert by_type["call"]["n_lots"] == 1
        assert by_type["call"]["lots"][0]["price"] == 3.5


class TestDefensiveParse:
    def test_malformed_rows_skipped(self):
        positions = [_position("NVDA", current_price=220.0)]
        trades = [
            None,  # type: ignore[list-item]
            {"id": "garbage"},
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 60),
            _trade(2, "NVDA", "REBALANCE", 1.0, 200.0, 30),  # non-BUY/SELL
            _trade(3, "NVDA", "BUY", -1.0, 200.0, 20),       # negative qty
            _trade(4, "NVDA", "BUY", 1.0, 0.0, 10),          # zero price
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        # Only the one valid BUY survives
        assert out["positions"][0]["n_lots"] == 1
        assert out["positions"][0]["lots"][0]["trade_id"] == 1

    def test_unparseable_timestamps_skipped(self):
        positions = [_position("NVDA", current_price=220.0)]
        trades = [
            {"id": 1, "ticker": "NVDA", "action": "BUY", "qty": 1.0,
             "price": 200.0, "timestamp": "not-a-date"},
            _trade(2, "NVDA", "BUY", 1.0, 210.0, 30),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out["positions"][0]["n_lots"] == 1
        assert out["positions"][0]["lots"][0]["trade_id"] == 2

    def test_sell_more_than_held_just_drops(self):
        positions = [_position("NVDA", current_price=220.0)]
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 120),
            _trade(2, "NVDA", "SELL", 5.0, 215.0, 60),  # over-sell
        ]
        # Builder must not raise — over-sell just empties the queue
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out["positions"][0]["n_lots"] == 0
        assert out["positions"][0]["verdict"] == NO_LOTS

    def test_position_missing_current_price(self):
        positions = [{
            "ticker": "NVDA", "type": "stock",
            "expiry": None, "strike": None,
            # No current_price
        }]
        trades = [_trade(1, "NVDA", "BUY", 1.0, 200.0, 60)]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        p = out["positions"][0]
        # Lots reconstructed but pl_pct is None
        assert p["n_lots"] == 1
        assert p["lots"][0]["pl_pct"] is None
        # NO_LOTS verdict because we can't classify without a mark
        assert p["verdict"] == NO_LOTS


class TestThresholdOverrides:
    def test_wide_spread_pct_threshold(self):
        positions = [_position("NVDA", current_price=220.0)]
        # 3pp spread: +1% / -2%
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 217.82, 60),  # +1.0%
            _trade(2, "NVDA", "BUY", 1.0, 224.49, 30),  # -2.0%
        ]
        # Default 5pp → STACKED
        out_default = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out_default["positions"][0]["verdict"] == LADDER_STACKED
        # Tighten to 2pp → WIDE
        out_tight = build_cost_basis_ladder(
            positions, trades, now=_NOW, wide_spread_pct=2.0,
        )
        assert out_tight["positions"][0]["verdict"] == LADDER_WIDE

    def test_harvest_pct_floor_threshold(self):
        positions = [_position("NVDA", current_price=220.0)]
        # +2% lot — below default 3% but above a 1% override
        trades = [_trade(1, "NVDA", "BUY", 1.0, 215.69, 60)]
        out_default = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out_default["harvestable"] == []
        out_low = build_cost_basis_ladder(
            positions, trades, now=_NOW, harvest_pct_floor=1.0,
        )
        assert len(out_low["harvestable"]) == 1
        assert out_low["verdict"] == HARVESTABLE_LOTS


class TestThresholdsExposed:
    def test_thresholds_in_envelope(self):
        out = build_cost_basis_ladder(
            [], [], now=_NOW,
            wide_spread_pct=7.5, harvest_pct_floor=4.2,
        )
        assert out["thresholds"]["wide_spread_pct"] == 7.5
        assert out["thresholds"]["harvest_pct_floor"] == 4.2


class TestHarvestableOrdering:
    def test_harvestable_sorted_by_pl_pct_desc(self):
        positions = [
            _position("NVDA", current_price=220.0),
            _position("TSLA", current_price=300.0),
        ]
        # NVDA lot +10%, TSLA lot +5%
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 120),
            _trade(2, "TSLA", "BUY", 1.0, 285.71, 60),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        assert out["harvestable"][0]["ticker"] == "NVDA"
        assert out["harvestable"][1]["ticker"] == "TSLA"


class TestReasonExcerpt:
    def test_long_reason_truncated(self):
        positions = [_position("NVDA", current_price=220.0)]
        long_reason = "x" * 500
        trades = [
            _trade(1, "NVDA", "BUY", 1.0, 200.0, 60, reason=long_reason),
        ]
        out = build_cost_basis_ladder(positions, trades, now=_NOW)
        excerpt = out["positions"][0]["lots"][0]["reason_excerpt"]
        assert excerpt.endswith("…")
        assert len(excerpt) < 100
