"""Tests for analytics/liquidity.py — capital deployment & liquidity.

Deterministic arithmetic: every number below is hand-computed, so a wrong
weight, status threshold, or option multiplier fails the assertion.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.liquidity import build_liquidity

NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


def _pos(ticker, qty, avg, cur, type_="stock"):
    return {"ticker": ticker, "type": type_, "qty": qty,
            "avg_cost": avg, "current_price": cur}


def _trade(action, ticker, days_ago):
    return {"timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
            "action": action, "ticker": ticker, "qty": 1, "price": 1}


class TestPinnedBook:
    """The observed live case: ~0% cash, all positions red, no rotation."""

    def setup_method(self):
        self.r = build_liquidity(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("NVDA", 1.0, 200.0, 180.0),
             _pos("LITE", 1.0, 800.0, 790.0)],
            [_trade("SELL", "NVDA", 1.0), _trade("BUY", "LITE", 3.0)],
            now=NOW,
        )

    def test_status_is_no_dry_powder(self):
        assert self.r["status"] == "NO_DRY_POWDER"

    def test_cash_and_deployed_pct(self):
        assert self.r["cash_pct"] == 0.6
        assert self.r["deployed_pct"] == 99.4
        assert self.r["invested_value"] == 994.0

    def test_cannot_act_on_signal(self):
        # cash $6 ≥ $1 but only 0.6% of book → effectively no buying power.
        assert self.r["can_act_on_signal"] is False

    def test_largest_position_and_weight(self):
        assert self.r["largest_position"] == "LITE"
        assert self.r["top_weight_pct"] == 79.0

    def test_unrealized_pl(self):
        # (180-200) + (790-800) = -30 on a $1000 cost basis = -3%
        assert self.r["unrealized_pl"] == -30.0
        assert self.r["unrealized_pl_pct"] == -3.0
        assert self.r["n_losers"] == 2
        assert self.r["n_winners"] == 0

    def test_days_since_entry_skips_sell(self):
        # newest trade is a SELL (1d); the last *opening* trade is BUY (3d).
        assert self.r["days_since_last_trade"] == 1.0
        assert self.r["days_since_last_entry"] == 3.0

    def test_flags_present(self):
        joined = " | ".join(self.r["flags"])
        assert "deployed" in joined
        assert "no dry powder" in joined
        assert "all 2 open positions underwater" in joined
        assert "LITE is 79% of the book" in joined
        assert "no new position opened in 3.0d" in joined


class TestStatusThresholds:
    def _status(self, cash, total, npos=1):
        positions = [_pos("X", 1.0, 100.0, 100.0)] * npos
        return build_liquidity({"cash": cash, "total_value": total},
                               positions, [], now=NOW)["status"]

    def test_balanced(self):
        assert self._status(400, 1000) == "BALANCED"

    def test_cash_heavy_over_60(self):
        assert self._status(800, 1000) == "CASH_HEAVY"

    def test_dry_powder_low_under_5(self):
        assert self._status(30, 1000) == "DRY_POWDER_LOW"

    def test_no_dry_powder_under_2_with_positions(self):
        assert self._status(15, 1000, npos=2) == "NO_DRY_POWDER"

    def test_under_2_but_no_positions_is_dry_powder_low(self):
        # The "pinned" status only applies when capital is actually tied up
        # in positions; flat-but-broke is just low dry powder.
        r = build_liquidity({"cash": 15, "total_value": 1000}, [], [], now=NOW)
        assert r["status"] == "DRY_POWDER_LOW"

    def test_no_data_when_empty(self):
        r = build_liquidity({"cash": 0, "total_value": 0}, [], [], now=NOW)
        assert r["status"] == "NO_DATA"
        assert r["headline"] == "No portfolio data"


class TestMarketValueMath:
    def test_option_value_uses_100_multiplier(self):
        r = build_liquidity(
            {"cash": 100.0, "total_value": 1300.0},
            [_pos("NVDA", 2.0, 5.0, 6.0, type_="call")],
            [], now=NOW,
        )
        p = r["positions"][0]
        assert p["market_value"] == 1200.0      # 6 * 2 * 100
        assert p["unrealized_pl"] == 200.0      # (6-5) * 2 * 100

    def test_unmarked_price_falls_back_to_avg_cost(self):
        # current_price 0 must not count the position as $0 deployed.
        r = build_liquidity(
            {"cash": 50.0, "total_value": 150.0},
            [_pos("MU", 1.0, 100.0, 0.0)],
            [], now=NOW,
        )
        assert r["positions"][0]["market_value"] == 100.0
        assert r["positions"][0]["unrealized_pl"] == 0.0

    def test_positions_sorted_by_value_desc(self):
        r = build_liquidity(
            {"cash": 0.0, "total_value": 600.0},
            [_pos("A", 1.0, 100.0, 100.0), _pos("B", 1.0, 500.0, 500.0)],
            [], now=NOW,
        )
        assert [p["ticker"] for p in r["positions"]] == ["B", "A"]
        assert r["largest_position"] == "B"


class TestEntryTracking:
    def test_no_opening_trade_flag(self):
        r = build_liquidity(
            {"cash": 50.0, "total_value": 100.0},
            [_pos("X", 1.0, 50.0, 50.0)],
            [_trade("SELL", "X", 2.0)],   # only a SELL on record
            now=NOW,
        )
        assert r["days_since_last_entry"] is None
        assert any("no opening trade on record" in f for f in r["flags"])

    def test_buy_call_counts_as_entry(self):
        r = build_liquidity(
            {"cash": 50.0, "total_value": 100.0},
            [_pos("X", 1.0, 50.0, 50.0)],
            [_trade("BUY_CALL", "X", 0.5)],
            now=NOW,
        )
        assert r["days_since_last_entry"] == 0.5
