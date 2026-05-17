"""Tests for analytics/open_attribution.py — open-book selection vs SPY.

Hand-computed arithmetic: a wrong entry anchor (must be S&P at-or-after
``opened_at``), a wrong alpha sign, an option that isn't skipped, or an
unanchored position polluting the book aggregate all fail an assertion.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.open_attribution import build_open_attribution

NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


def _pos(ticker, qty, avg, cur, mins_ago_opened, type_="stock"):
    return {"ticker": ticker, "type": type_, "qty": qty,
            "avg_cost": avg, "current_price": cur,
            "opened_at": (NOW - timedelta(minutes=mins_ago_opened)).isoformat()}


def _eq(sp500, mins_ago):
    return {"timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
            "total_value": 1000.0, "cash": 6.0, "sp500_price": sp500}


# SPY curve: 100 @ -100m, 110 @ -50m, 121 @ -10m (latest → now_spy = 121).
EQUITY = [_eq(100.0, 100), _eq(110.0, 50), _eq(121.0, 10)]


class TestSelectionDragBook:
    def setup_method(self):
        # NVDA opened -60m → first SPY at/after = 110 (the -50m point).
        #   pos +10% (110/100), SPY +10% (121/110) → alpha 0, excess $0.
        # LITE opened -120m → first SPY at/after = 100 (the -100m point).
        #   pos -10% (180/200), SPY +21% (121/100) → alpha -31%, excess -$62.
        self.r = build_open_attribution(
            [_pos("NVDA", 2.0, 100.0, 110.0, 60),
             _pos("LITE", 1.0, 200.0, 180.0, 120),
             _pos("SOXL", 1.0, 5.0, 6.0, 30, type_="call")],
            EQUITY, now=NOW,
        )

    def test_option_is_skipped_not_attributed(self):
        assert [s["ticker"] for s in self.r["skipped"]] == ["SOXL"]
        assert "option" in self.r["skipped"][0]["reason"]
        assert all(p["ticker"] != "SOXL" for p in self.r["positions"])

    def test_nvda_anchor_and_alpha(self):
        nvda = next(p for p in self.r["positions"] if p["ticker"] == "NVDA")
        assert nvda["position_return_pct"] == 10.0
        assert nvda["spy_return_pct"] == 10.0
        assert nvda["alpha_pct"] == 0.0
        assert nvda["unrealized_usd"] == 20.0
        assert nvda["spy_equivalent_usd"] == 20.0
        assert nvda["excess_usd"] == 0.0
        assert nvda["anchored"] is True

    def test_lite_anchor_uses_at_or_after_entry(self):
        lite = next(p for p in self.r["positions"] if p["ticker"] == "LITE")
        assert lite["position_return_pct"] == -10.0
        assert lite["spy_return_pct"] == 21.0          # 121/100 - 1
        assert lite["alpha_pct"] == -31.0
        assert lite["unrealized_usd"] == -20.0
        assert lite["spy_equivalent_usd"] == 42.0      # 200 * 0.21
        assert lite["excess_usd"] == -62.0

    def test_biggest_drag_sorted_first(self):
        assert self.r["positions"][0]["ticker"] == "LITE"

    def test_book_aggregate(self):
        # cost 400, unreal 0, spy_equiv 62 → net -62, book alpha -15.5%.
        assert self.r["total_cost_basis_usd"] == 400.0
        assert self.r["total_unrealized_usd"] == 0.0
        assert self.r["total_spy_equivalent_usd"] == 62.0
        assert self.r["net_excess_usd"] == -62.0
        assert self.r["book_open_alpha_pct"] == -15.5
        assert self.r["status"] == "SELECTION_DRAG"
        assert self.r["n_anchored"] == 2

    def test_headline_states_drag_and_worst(self):
        h = self.r["headline"]
        assert "dragging 15.50% alpha" in h
        assert "LITE" in h


class TestSelectionAdding:
    def test_positive_excess_flags_adding(self):
        # AAPL opened -100m → SPY at/after = 100; SPY +21%.
        # pos +50% (150/100) → alpha +29%, excess = 50 - 21 = +$29 on $100.
        r = build_open_attribution(
            [_pos("AAPL", 1.0, 100.0, 150.0, 100)], EQUITY, now=NOW,
        )
        p = r["positions"][0]
        assert p["alpha_pct"] == 29.0
        assert p["excess_usd"] == 29.0
        assert r["status"] == "SELECTION_ADDING"
        assert r["book_open_alpha_pct"] == 29.0
        assert "adding 29.00% alpha" in r["headline"]


class TestEdgeCases:
    def test_no_benchmark_when_equity_has_no_spy(self):
        eq = [{"timestamp": (NOW - timedelta(minutes=10)).isoformat(),
               "total_value": 1000.0, "cash": 6.0, "sp500_price": None}]
        r = build_open_attribution([_pos("NVDA", 1.0, 100.0, 110.0, 30)],
                                   eq, now=NOW)
        assert r["status"] == "NO_BENCHMARK"
        assert r["book_open_alpha_pct"] is None

    def test_position_opened_after_all_equity_is_unanchored_not_polluting(self):
        # Opened -5m, but the last SPY sample is -10m → no point at/after.
        r = build_open_attribution(
            [_pos("LATE", 1.0, 100.0, 130.0, 5),
             _pos("NVDA", 2.0, 100.0, 110.0, 60)],
            EQUITY, now=NOW,
        )
        late = next(p for p in r["positions"] if p["ticker"] == "LATE")
        assert late["anchored"] is False
        assert late["alpha_pct"] is None
        assert late["excess_usd"] is None
        # The book aggregate must exclude the unanchored row entirely:
        # only NVDA (cost 200) counts, not LATE (cost 100).
        assert r["total_cost_basis_usd"] == 200.0
        assert r["n_anchored"] == 1

    def test_unmarked_price_is_skipped(self):
        r = build_open_attribution(
            [_pos("NVDA", 1.0, 100.0, 0.0, 60)], EQUITY, now=NOW,
        )
        assert r["positions"] == []
        assert r["skipped"][0]["ticker"] == "NVDA"
        assert "unmarked" in r["skipped"][0]["reason"]

    def test_empty_positions(self):
        r = build_open_attribution([], EQUITY, now=NOW)
        assert r["status"] == "NO_DATA"
        assert r["positions"] == []
