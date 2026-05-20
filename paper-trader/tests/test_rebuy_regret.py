"""Unit tests for paper_trader.analytics.rebuy_regret.

The builder composes round_trips and walks the trade stream for the next
same-key BUY to measure the close→re-buy price-delta regret. Tests assert
the sign convention (positive = lost, negative = saved), shared-quantity
math, per-event classification thresholds, the verdict ladder, option ×100
multiplier honored, and degrade-never-raise on garbage rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.rebuy_regret import build_rebuy_regret


def _trade(tid, ts, ticker, action, qty, price, *, option_type=None,
           strike=None, expiry=None):
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


class TestStateLadder:
    def test_no_trades_no_data(self):
        out = build_rebuy_regret([])
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "NO_DATA"
        assert out["n_events"] == 0
        assert out["total_regret_usd"] == 0.0
        assert out["recent_events"] == []
        assert out["per_ticker"] == []

    def test_only_closed_round_trips_no_rebuys(self):
        """1 closed round-trip + no subsequent re-buy → NO_REBUYS, not NO_DATA."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 101.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["state"] == "NO_REBUYS"
        assert out["verdict"] == "NO_REBUYS"
        assert out["n_round_trips"] == 1
        assert out["n_events"] == 0
        assert "no re-entries" in out["headline"]

    def test_open_position_no_close_no_data_event(self):
        """An open BUY-only position with no close → no round-trip, no event."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["state"] == "NO_REBUYS"
        assert out["n_round_trips"] == 0
        assert out["n_events"] == 0


class TestSignConvention:
    def test_sold_low_bought_higher_is_regret(self):
        """Sold $100 → re-bought $105 = positive regret (lost $5/share)."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 105.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["state"] == "OK"
        assert out["n_events"] == 1
        ev = out["recent_events"][0]
        # Sold at 100, re-bought at 105 → +5 × 10 = +$50 regret.
        assert abs(ev["regret_usd"] - 50.0) < 1e-6
        assert ev["classification"] == "REGRET_HIGH"
        assert out["verdict"] == "REGRETTING"
        assert "REGRET" in out["headline"]

    def test_sold_high_bought_lower_is_savings(self):
        """Sold $105 → re-bought $100 = negative regret (saved $5/share)."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 105.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 100.0),
        ]
        out = build_rebuy_regret(trades)
        ev = out["recent_events"][0]
        assert abs(ev["regret_usd"] - (-50.0)) < 1e-6
        assert ev["classification"] == "SAVED_HIGH"
        assert out["verdict"] == "SAVINGS"
        assert "SAVINGS" in out["headline"]

    def test_neutral_below_noise_floor(self):
        """A $0.30 regret on a 1-share trade is below the NEUTRAL floor."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 1, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 1, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 1, 100.30),
        ]
        out = build_rebuy_regret(trades)
        ev = out["recent_events"][0]
        assert ev["classification"] == "NEUTRAL"
        assert out["verdict"] == "NET_NEUTRAL"


class TestSharedQuantity:
    def test_shared_qty_is_min_of_sell_and_rebuy(self):
        """Sold 10 shares, re-bought only 5 → shared_qty=5, regret on 5."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 5, 110.0),
        ]
        out = build_rebuy_regret(trades)
        ev = out["recent_events"][0]
        assert ev["shared_qty"] == 5.0
        # (110 - 100) * 5 = 50
        assert abs(ev["regret_usd"] - 50.0) < 1e-6

    def test_shared_qty_works_when_rebuy_larger_than_sell(self):
        """Sold 5 shares, re-bought 10 → shared_qty=5."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 5, 100.0),
            _trade(3, "2026-05-19T11:30:00+00:00", "NVDA", "SELL", 5, 100.0),
            _trade(4, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 110.0),
        ]
        out = build_rebuy_regret(trades)
        # Only one round-trip closes (when the second SELL drops qty to 0);
        # its exit_trade is the second SELL (id=3).
        assert out["n_events"] == 1
        ev = out["recent_events"][0]
        # Last SELL was qty=5, re-buy was qty=10 → shared=5, delta=10
        assert ev["shared_qty"] == 5.0


class TestOptionMultiplier:
    def test_option_uses_100x_multiplier(self):
        """An option close→re-buy uses the ×100 contract multiplier."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY_CALL", 1, 4.0,
                   option_type="call", strike=220.0, expiry="2026-05-30"),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL_CALL", 1, 4.0,
                   option_type="call", strike=220.0, expiry="2026-05-30"),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY_CALL", 1, 5.0,
                   option_type="call", strike=220.0, expiry="2026-05-30"),
        ]
        out = build_rebuy_regret(trades)
        ev = out["recent_events"][0]
        # (5 - 4) * 1 * 100 = $100 regret
        assert abs(ev["regret_usd"] - 100.0) < 1e-6
        assert ev["type"] == "call"


class TestKeyIsolation:
    def test_different_tickers_dont_count_as_rebuys(self):
        """Sell NVDA → buy AMD is NOT a rebuy."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 105.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "AMD", "BUY", 10, 200.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["n_events"] == 0
        assert out["state"] == "NO_REBUYS"

    def test_different_option_strikes_dont_count(self):
        """Sell NVDA $220 call → buy NVDA $230 call is a different key."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY_CALL", 1, 4.0,
                   option_type="call", strike=220.0, expiry="2026-05-30"),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL_CALL", 1, 4.0,
                   option_type="call", strike=220.0, expiry="2026-05-30"),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY_CALL", 1, 5.0,
                   option_type="call", strike=230.0, expiry="2026-05-30"),
        ]
        out = build_rebuy_regret(trades)
        assert out["n_events"] == 0


class TestMultipleEvents:
    def test_per_ticker_rollup_sums_correctly(self):
        """Two regret events on NVDA sum into per_ticker.net_regret_usd."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 110.0),
            _trade(4, "2026-05-19T13:00:00+00:00", "NVDA", "SELL", 10, 110.0),
            _trade(5, "2026-05-19T14:00:00+00:00", "NVDA", "BUY", 10, 120.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["n_events"] == 2
        # Event 1: sold 100, re-bought 110 → +100 regret
        # Event 2: sold 110, re-bought 120 → +100 regret
        # Total: +200
        assert abs(out["total_regret_usd"] - 200.0) < 1e-6
        nvda = next(p for p in out["per_ticker"] if p["ticker"] == "NVDA")
        assert nvda["n_events"] == 2
        assert abs(nvda["net_regret_usd"] - 200.0) < 1e-6
        assert out["verdict"] == "REGRETTING"

    def test_mixed_regret_and_savings_nets_correctly(self):
        """One $50 regret + one $30 savings = $20 net regret."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 105.0),
            _trade(4, "2026-05-19T13:00:00+00:00", "NVDA", "SELL", 10, 105.0),
            _trade(5, "2026-05-19T14:00:00+00:00", "NVDA", "BUY", 10, 102.0),
        ]
        out = build_rebuy_regret(trades)
        # Event 1: 100 → 105 = +50 (REGRET_HIGH)
        # Event 2: 105 → 102 = -30 × 10 = -30  (SAVED_HIGH)
        # Total: +20 net regret
        assert out["n_events"] == 2
        assert abs(out["total_regret_usd"] - 20.0) < 1e-6
        # Worst regret reported
        assert abs(out["worst_regret_usd"] - 50.0) < 1e-6
        # Best savings reported (most negative)
        assert abs(out["best_savings_usd"] - (-30.0)) < 1e-6
        assert out["regret_event_count"] == 1
        assert out["saved_event_count"] == 1


class TestSorting:
    def test_recent_events_newest_first(self):
        """recent_events sorted by rebought_at DESC."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 105.0),
            _trade(4, "2026-05-19T13:00:00+00:00", "NVDA", "SELL", 10, 105.0),
            _trade(5, "2026-05-19T14:00:00+00:00", "NVDA", "BUY", 10, 110.0),
        ]
        out = build_rebuy_regret(trades)
        evts = out["recent_events"]
        assert len(evts) == 2
        # Newest re-buy first.
        assert evts[0]["rebought_at"] > evts[1]["rebought_at"]

    def test_per_ticker_worst_offender_first(self):
        """per_ticker sorted by net_regret_usd DESC."""
        # NVDA: +100 regret. AMD: -50 savings.
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 110.0),
            _trade(4, "2026-05-19T13:00:00+00:00", "AMD", "BUY", 10, 200.0),
            _trade(5, "2026-05-19T14:00:00+00:00", "AMD", "SELL", 10, 200.0),
            _trade(6, "2026-05-19T15:00:00+00:00", "AMD", "BUY", 10, 195.0),
        ]
        out = build_rebuy_regret(trades)
        assert out["per_ticker"][0]["ticker"] == "NVDA"
        assert out["per_ticker"][1]["ticker"] == "AMD"

    def test_recent_limit_caps_output(self):
        """recent_limit controls slice size."""
        trades = []
        for i in range(0, 6):
            t = f"2026-05-19T{10+i:02d}:00:00+00:00"
            trades.append(_trade(2*i+1, t, "NVDA", "BUY", 10, 100.0 + i))
            t2 = f"2026-05-19T{10+i:02d}:30:00+00:00"
            trades.append(_trade(2*i+2, t2, "NVDA", "SELL", 10, 100.0 + i))
        # Last buy with no subsequent sell — not a closed round-trip.
        # We have 6 closed round-trips, 5 re-buys → 5 events.
        out = build_rebuy_regret(trades, recent_limit=2)
        assert out["n_events"] == 5
        assert len(out["recent_events"]) == 2


class TestGapHours:
    def test_gap_hours_computed(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T13:00:00+00:00", "NVDA", "BUY", 10, 105.0),
        ]
        out = build_rebuy_regret(trades)
        ev = out["recent_events"][0]
        # 11:00 → 13:00 = 2h
        assert abs(ev["gap_hours"] - 2.0) < 1e-6


class TestDegradeNeverRaise:
    def test_none_trades(self):
        out = build_rebuy_regret(None)
        assert out["state"] == "NO_DATA"

    def test_non_dict_rows_skipped(self):
        out = build_rebuy_regret([None, "garbage", 123, {}])
        # None of these have parseable timestamps → NO_DATA.
        assert out["state"] in ("NO_DATA", "NO_REBUYS")
        # We tolerate either depending on whether the dict round-trips through.

    def test_garbage_timestamps_skip(self):
        trades = [
            {"id": 1, "timestamp": "not-a-date", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 100.0,
             "value": 1000.0, "option_type": None, "strike": None,
             "expiry": None},
        ]
        # Should not raise.
        out = build_rebuy_regret(trades)
        assert "state" in out

    def test_zero_price_rebuy_skipped(self):
        """A rebuy with price <= 0 doesn't produce a misleading event."""
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 0.0),
        ]
        out = build_rebuy_regret(trades)
        # Zero-price re-buy is filtered → NO_REBUYS, not a $-1000 fake regret.
        assert out["state"] == "NO_REBUYS"


class TestInputOrderTolerance:
    def test_newest_first_input_still_works(self):
        """Caller can pass either oldest→newest or newest→oldest."""
        oldest_first = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 10, 110.0),
        ]
        newest_first = list(reversed(oldest_first))
        out_a = build_rebuy_regret(oldest_first)
        out_b = build_rebuy_regret(newest_first)
        assert out_a["n_events"] == out_b["n_events"]
        assert out_a["total_regret_usd"] == out_b["total_regret_usd"]


class TestVerdictBoundaries:
    def test_neutral_floor_boundary_exact(self):
        """A regret == NEUTRAL_USD_FLOOR is NEUTRAL (not REGRET)."""
        # $0.50 regret at the boundary
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 1, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 1, 100.0),
            _trade(3, "2026-05-19T12:00:00+00:00", "NVDA", "BUY", 1, 100.50),
        ]
        out = build_rebuy_regret(trades)
        # Exactly $0.50 — _classify uses strict >, so boundary is NEUTRAL.
        assert out["recent_events"][0]["classification"] == "NEUTRAL"
