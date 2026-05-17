"""Unit tests for paper_trader.analytics.round_trips.build_round_trips.

These assert *exact* hand-computed round-trip aggregates, not just "it ran".
The function is the single source of truth behind /api/analytics'
win_rate / profit_factor / avg_holding_days, so its arithmetic is
load-bearing.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.round_trips import build_round_trips


def _trade(tid, ts, ticker, action, qty, price, *, option_type=None,
           strike=None, expiry=None):
    """Mirror a Store.recent_trades() row. value = qty*price*(100 if option)."""
    mult = 100 if option_type in ("call", "put") else 1
    return {
        "id": tid,
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price * mult,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
    }


class TestSimpleRoundTrip:
    def test_single_buy_sell_winner(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "AAPL", "BUY", 10, 10.0),
            _trade(2, "2026-05-03T10:00:00+00:00", "AAPL", "SELL", 10, 12.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        rt = rts[0]
        assert rt["ticker"] == "AAPL"
        assert rt["type"] == "stock"
        assert rt["cost"] == 100.0
        assert rt["proceeds"] == 120.0
        assert rt["pnl_usd"] == 20.0
        assert rt["pnl_pct"] == 20.0  # 20/100*100
        assert rt["qty"] == 10
        assert rt["n_buys"] == 1
        assert rt["n_sells"] == 1
        assert rt["entry_trade_ids"] == [1]
        assert rt["exit_trade_ids"] == [2]
        # 2 calendar days between entry and exit
        assert rt["hold_days"] == 2.0

    def test_single_buy_sell_loser(self):
        trades = [
            _trade(1, "2026-05-01T00:00:00+00:00", "MSFT", "BUY", 5, 20.0),
            _trade(2, "2026-05-01T12:00:00+00:00", "MSFT", "SELL", 5, 16.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        assert rts[0]["pnl_usd"] == -20.0
        assert rts[0]["pnl_pct"] == -20.0
        assert rts[0]["hold_days"] == 0.5  # 12h


class TestPartialAndReentry:
    def test_partial_sells_close_one_round_trip(self):
        # BUY 2 @ 50, SELL 1 @ 60 (held=1, no close), SELL 1 @ 70 (held=0).
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "NVDA", "BUY", 2, 50.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "NVDA", "SELL", 1, 60.0),
            _trade(3, "2026-05-04T10:00:00+00:00", "NVDA", "SELL", 1, 70.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        rt = rts[0]
        assert rt["cost"] == 100.0
        assert rt["proceeds"] == 130.0
        assert rt["pnl_usd"] == 30.0
        assert rt["n_buys"] == 1
        assert rt["n_sells"] == 2
        # entry = first BUY ts, exit = last (closing) SELL ts
        assert rt["entry_ts"] == "2026-05-01T10:00:00+00:00"
        assert rt["exit_ts"] == "2026-05-04T10:00:00+00:00"
        assert rt["hold_days"] == 3.0

    def test_rebuy_after_close_is_a_new_round_trip(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "TQQQ", "BUY", 1, 100.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "TQQQ", "SELL", 1, 110.0),
            _trade(3, "2026-05-05T10:00:00+00:00", "TQQQ", "BUY", 1, 90.0),
            _trade(4, "2026-05-06T10:00:00+00:00", "TQQQ", "SELL", 1, 80.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 2
        assert rts[0]["pnl_usd"] == 10.0
        assert rts[0]["entry_trade_ids"] == [1]
        assert rts[1]["pnl_usd"] == -10.0
        # second round-trip must NOT carry over the first's ids/cost
        assert rts[1]["entry_trade_ids"] == [3]
        assert rts[1]["cost"] == 90.0
        assert rts[1]["n_buys"] == 1

    def test_add_to_open_lot_blends_into_one_round_trip(self):
        # BUY 1 @ 100, BUY 1 @ 200 (avg in), SELL 2 @ 180 -> one round-trip.
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "AMD", "BUY", 1, 100.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "AMD", "BUY", 1, 200.0),
            _trade(3, "2026-05-03T10:00:00+00:00", "AMD", "SELL", 2, 180.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        rt = rts[0]
        assert rt["cost"] == 300.0
        assert rt["proceeds"] == 360.0
        assert rt["pnl_usd"] == 60.0
        assert rt["qty"] == 2
        assert rt["n_buys"] == 2


class TestOptionsAndKeys:
    def test_option_value_uses_x100_via_trade_value(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "TSLA", "BUY_CALL", 1, 2.0,
                   option_type="call", strike=100.0, expiry="2026-06-19"),
            _trade(2, "2026-05-10T10:00:00+00:00", "TSLA", "SELL_CALL", 1, 3.0,
                   option_type="call", strike=100.0, expiry="2026-06-19"),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        rt = rts[0]
        assert rt["type"] == "call"
        assert rt["strike"] == 100.0
        assert rt["expiry"] == "2026-06-19"
        assert rt["cost"] == 200.0      # 1 * 2 * 100
        assert rt["proceeds"] == 300.0  # 1 * 3 * 100
        assert rt["pnl_usd"] == 100.0

    def test_stock_and_option_legs_do_not_conflate(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "NVDA", "BUY", 1, 100.0),
            _trade(2, "2026-05-01T10:00:01+00:00", "NVDA", "BUY_CALL", 1, 5.0,
                   option_type="call", strike=120.0, expiry="2026-06-19"),
            _trade(3, "2026-05-02T10:00:00+00:00", "NVDA", "SELL", 1, 110.0),
            _trade(4, "2026-05-02T10:00:01+00:00", "NVDA", "SELL_CALL", 1, 7.0,
                   option_type="call", strike=120.0, expiry="2026-06-19"),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 2
        by_type = {rt["type"]: rt for rt in rts}
        assert by_type["stock"]["pnl_usd"] == 10.0
        assert by_type["call"]["pnl_usd"] == 200.0  # (7-5)*100

    def test_different_strikes_are_distinct_round_trips(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "SPY", "BUY_CALL", 1, 1.0,
                   option_type="call", strike=500.0, expiry="2026-06-19"),
            _trade(2, "2026-05-01T10:00:01+00:00", "SPY", "BUY_CALL", 1, 2.0,
                   option_type="call", strike=510.0, expiry="2026-06-19"),
            _trade(3, "2026-05-02T10:00:00+00:00", "SPY", "SELL_CALL", 1, 4.0,
                   option_type="call", strike=500.0, expiry="2026-06-19"),
            _trade(4, "2026-05-02T10:00:01+00:00", "SPY", "SELL_CALL", 1, 1.0,
                   option_type="call", strike=510.0, expiry="2026-06-19"),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 2
        by_strike = {rt["strike"]: rt["pnl_usd"] for rt in rts}
        assert by_strike[500.0] == 300.0   # (4-1)*100
        assert by_strike[510.0] == -100.0  # (1-2)*100


class TestEdgeCases:
    def test_open_position_is_not_a_closed_round_trip(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "MU", "BUY", 10, 5.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "MU", "SELL", 4, 6.0),
        ]
        # held = 6, never returns to 0 -> nothing emitted
        assert build_round_trips(trades) == []

    def test_orphan_sell_with_no_prior_buy_emits_nothing(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "GME", "SELL", 5, 20.0),
        ]
        # held = -5; abs(held) is not ~0 so no round-trip is emitted
        assert build_round_trips(trades) == []

    def test_zero_cost_round_trip_has_none_pnl_pct(self):
        # Pathological: a recorded BUY with value 0 then a SELL. pnl_pct
        # must be None (no divide-by-zero), pnl_usd still computed.
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "ZZZ", "BUY", 1, 0.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "ZZZ", "SELL", 1, 5.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        assert rts[0]["cost"] == 0.0
        assert rts[0]["pnl_usd"] == 5.0
        assert rts[0]["pnl_pct"] is None

    def test_negative_hold_days_become_none(self):
        # Closing SELL stamped *before* the entry BUY (clock skew / bad data).
        trades = [
            _trade(1, "2026-05-05T10:00:00+00:00", "X", "BUY", 1, 10.0),
            _trade(2, "2026-05-01T10:00:00+00:00", "X", "SELL", 1, 11.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        assert rts[0]["hold_days"] is None
        assert rts[0]["pnl_usd"] == 1.0

    def test_unparseable_timestamps_yield_none_hold_days(self):
        trades = [
            _trade(1, "not-a-date", "Y", "BUY", 1, 10.0),
            _trade(2, "also-bad", "Y", "SELL", 1, 12.0),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        assert rts[0]["hold_days"] is None
        assert rts[0]["pnl_usd"] == 2.0

    def test_subcent_pnl_rounds_to_zero(self):
        # pnl_usd is rounded to 4 decimals. A 1e-6 "profit" rounds to 0.0,
        # so the /api/analytics `> 0` win classifier treats it as a non-win.
        # Pin this so a future reviewer doesn't reintroduce raw-float drift.
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "Q", "BUY", 1, 100.0),
            _trade(2, "2026-05-02T10:00:00+00:00", "Q", "SELL", 1, 100.000001),
        ]
        rts = build_round_trips(trades)
        assert len(rts) == 1
        assert rts[0]["pnl_usd"] == 0.0
        assert not (rts[0]["pnl_usd"] > 0)
