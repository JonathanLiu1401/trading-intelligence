"""Tests for analytics.buying_power — the lean deployable-cash advisory fed
into the live Opus decision prompt (the prompt-facing complement to the
dashboard-only capital_paralysis).

Every assertion pins a *specific* expected value so a wrong comparison
operator, off-by-one floor, or broken state transition fails loudly. The
discriminating locks:

* the live CASH_CONSTRAINED shape (the documented $18.49-of-$972 pathology)
  → every affordable whole-share count is 0, the cheapest in-play name is
  correctly identified, and the unlock fact names the most-underwater
  position (the capital_paralysis "biggest loser first" cut-priority);
* the whole-share count is a strict floor (``int(cash // px)``) — a 999.99
  cash / 500 price → 1 share, never 2;
* ``_position_mark_value`` consumes the enriched ``market_value`` (option
  ×100 already baked in) and never re-derives it — a regression that
  re-multiplies fails;
* the block is observational: it carries the autonomy preamble and no
  imperative trade verb (the event_calendar #2/#12 contract);
* ``_build_payload`` renders it last in the advisory stack (after
  event_calendar, before WATCHLIST PRICES); ``None`` renders no stray text;
* it never raises on garbage (the _safe contract — a diagnostics fault must
  not sink a live decision cycle).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.buying_power import (build_buying_power,
                                                 _position_mark_value)


def _snap(cash, total, positions=None):
    return {"cash": cash, "total_value": total,
            "positions": positions or []}


def _pos(ticker, qty, avg, cur, upl, type_="stock", market_value=None):
    p = {"ticker": ticker, "type": type_, "qty": qty, "avg_cost": avg,
         "current_price": cur, "unrealized_pl": upl}
    if market_value is not None:
        p["market_value"] = market_value
    return p


# ───────────────────────── CASH_CONSTRAINED (the live pathology) ─────────

class TestCashConstrained:
    def test_live_pinned_book_shape(self):
        """The documented finding: $18.49 free of a $972.69 book, two
        underwater fractional names. No in-play name is affordable whole;
        the unlock fact names the most-underwater position."""
        snap = _snap(18.49, 972.69, [
            _pos("MU", 0.5, 724.12, 724.12, 0.0),
            _pos("LITE", 0.61, 980.90, 970.71, -6.21),
        ])
        wp = {"SPY": 500.0, "QQQ": 480.0, "MU": 92.0, "LITE": 55.0,
              "SOXL": 30.0}
        names = {"SPY", "QQQ", "MU", "LITE", "SOXL"}
        out = build_buying_power(snap, wp, names)
        assert out["state"] == "CASH_CONSTRAINED"
        # Every affordable count is 0 — cash below every in-play price.
        assert all(a["whole_shares"] == 0 for a in out["affordable"])
        # Cheapest in-play priced name is SOXL @ $30.
        assert out["cheapest_name"] == "SOXL"
        assert out["cheapest_price"] == 30.0
        assert out["cash"] == 18.49
        assert round(out["deployed_pct"], 2) == round(
            (972.69 - 18.49) / 972.69 * 100, 2)
        # Unlock = most-underwater position (LITE −6.21, not MU 0.00).
        assert out["unlock"]["ticker"] == "LITE"
        assert out["unlock"]["unrealized_pl"] == -6.21
        pb = out["prompt_block"]
        assert "$18.49" in pb
        assert "SOXL" in pb and "$30.00" in pb
        assert "LITE" in pb  # the unlock candidate is surfaced

    def test_constrained_boundary_is_strict(self):
        """cash == cheapest price is NOT constrained (you can afford exactly
        one whole share); cash one cent below IS constrained."""
        names = {"SPY"}
        at = build_buying_power(_snap(500.0, 1000.0), {"SPY": 500.0}, names)
        assert at["state"] == "DEPLOYABLE"
        assert at["affordable"][0]["whole_shares"] == 1
        below = build_buying_power(_snap(499.99, 1000.0),
                                   {"SPY": 500.0}, names)
        assert below["state"] == "CASH_CONSTRAINED"
        assert below["affordable"][0]["whole_shares"] == 0


# ───────────────────────── DEPLOYABLE ─────────────────────────

class TestDeployable:
    def test_affordable_whole_shares_are_a_strict_floor(self):
        out = build_buying_power(
            _snap(999.99, 999.99), {"SPY": 500.0, "QQQ": 333.34},
            {"SPY", "QQQ"})
        assert out["state"] == "DEPLOYABLE"
        aff = {a["ticker"]: a["whole_shares"] for a in out["affordable"]}
        assert aff["SPY"] == 1          # 999.99 // 500   = 1 (not 2)
        assert aff["QQQ"] == 2          # 999.99 // 333.34 = 2 (not 3)
        pb = out["prompt_block"]
        assert "SPY 1" in pb and "QQQ 2" in pb

    def test_zero_negative_and_missing_prices_excluded(self):
        out = build_buying_power(
            _snap(1000.0, 1000.0),
            {"SPY": 500.0, "DEAD": 0.0, "NEG": -5.0, "NULLP": None},
            {"SPY", "DEAD", "NEG", "NULLP"})
        tickers = {a["ticker"] for a in out["affordable"]}
        assert tickers == {"SPY"}

    def test_name_not_in_play_is_excluded(self):
        out = build_buying_power(
            _snap(1000.0, 1000.0), {"SPY": 500.0, "TSLL": 10.0},
            {"SPY"})  # TSLL priced but not in play
        assert {a["ticker"] for a in out["affordable"]} == {"SPY"}

    def test_affordable_list_capped(self):
        wp = {f"T{i}": 1.0 for i in range(20)}
        out = build_buying_power(_snap(1000.0, 1000.0), wp, set(wp))
        assert out["state"] == "DEPLOYABLE"
        # All 20 affordable, but the prompt list is capped to 6 names.
        assert len(out["affordable"]) == 20
        assert out["prompt_block"].count(" · ") <= 5


# ───────────────────────── unlock fact ─────────────────────────

class TestUnlock:
    def test_picks_most_underwater_loser(self):
        snap = _snap(18.0, 1000.0, [
            _pos("A", 1, 100, 90, -10.0),
            _pos("B", 1, 100, 70, -30.0),   # worst
            _pos("C", 1, 100, 95, -5.0),
        ])
        out = build_buying_power(snap, {"A": 9999.0}, {"A"})
        assert out["unlock"]["ticker"] == "B"
        assert out["unlock"]["unrealized_pl"] == -30.0

    def test_no_losers_picks_largest_mark_value(self):
        snap = _snap(5.0, 1000.0, [
            _pos("A", 1, 100, 110, 10.0, market_value=110.0),
            _pos("B", 1, 100, 800, 700.0, market_value=800.0),  # biggest
        ])
        out = build_buying_power(snap, {"A": 9999.0}, {"A"})
        assert out["unlock"]["ticker"] == "B"
        assert out["unlock"]["frees_usd"] == 800.0
        # Positive-PL unlock line uses the "largest position" phrasing.
        assert "Largest position by mark value" in out["prompt_block"]


# ───────────────────────── _position_mark_value (single source) ──────────

class TestPositionMarkValue:
    def test_prefers_enriched_market_value_not_rederived(self):
        # An option with market_value already ×100-baked: must NOT be
        # re-multiplied. 2 contracts marked at $3 → market_value 600.
        p = _pos("NVDA", 2, 2.0, 3.0, 200.0, type_="call",
                 market_value=600.0)
        assert _position_mark_value(p) == 600.0

    def test_falls_back_to_derivation_for_plain_row(self):
        # No market_value key (a plain open_positions() row) → derive with
        # the ×100 option multiplier.
        p = _pos("NVDA", 2, 2.0, 3.0, 200.0, type_="call")
        assert _position_mark_value(p) == 3.0 * 2 * 100

    def test_stock_row_without_market_value(self):
        p = _pos("MU", 0.5, 724.12, 92.0, -316.06)
        assert _position_mark_value(p) == 92.0 * 0.5


# ───────────────────────── honesty / _safe ─────────────────────────

class TestHonestyAndSafe:
    def test_no_data_when_total_zero(self):
        out = build_buying_power(_snap(0.0, 0.0), {"SPY": 500.0}, {"SPY"})
        assert out["state"] == "NO_DATA"
        assert "unavailable" in out["summary"]
        assert isinstance(out["prompt_block"], str) and out["prompt_block"]

    def test_no_priced_names(self):
        out = build_buying_power(_snap(50.0, 1000.0, [_pos("MU", 1, 1, 1, 0)]),
                                 {}, {"MU"})
        assert out["state"] == "NO_PRICED_NAMES"
        assert "$50.00" in out["prompt_block"]

    def test_never_raises_on_garbage(self):
        for snap in (None, {}, {"cash": "x", "total_value": None},
                     {"cash": 10, "total_value": 100,
                      "positions": [{"ticker": None, "unrealized_pl": "bad"}]}):
            out = build_buying_power(snap, {"SPY": "oops"}, None)
            assert isinstance(out, dict) and "state" in out
            assert isinstance(out["prompt_block"], str)

    def test_block_is_observational_not_directive(self):
        snap = _snap(18.49, 972.69, [_pos("LITE", 0.61, 980, 970, -6.21)])
        out = build_buying_power(snap, {"SPY": 500.0}, {"SPY"})
        low = out["prompt_block"].lower()
        assert "autonomy" in low
        for directive in ("you should sell", "you must ", "do not buy",
                          "sell this", "exit the", "reduce your"):
            assert directive not in low


# ───────────────────────── _build_payload wiring ─────────────────────────

class TestBuildPayloadWiring:
    def test_renders_last_in_advisory_stack(self):
        from paper_trader import strategy

        snap = {"positions": [], "cash": 1000.0,
                "open_value": 0.0, "total_value": 1000.0}
        ev = "EVENT-CAL-MARKER earnings NVDA"
        bp = "BUYING-POWER-MARKER deployable"
        payload = strategy._build_payload(
            snap, [], [], {}, {}, None, True,
            quant_signals={},
            event_calendar_block=ev,
            buying_power_block=bp,
        )
        assert ev in payload and bp in payload
        # event_calendar < buying_power < WATCHLIST PRICES
        assert payload.index(ev) < payload.index(bp)
        assert payload.index(bp) < payload.index("WATCHLIST PRICES")

    def test_none_renders_no_stray_text(self):
        from paper_trader import strategy

        snap = {"positions": [], "cash": 1000.0,
                "open_value": 0.0, "total_value": 1000.0}
        payload = strategy._build_payload(
            snap, [], [], {}, {}, None, True,
            quant_signals={}, buying_power_block=None)
        assert "BUYING POWER" not in payload
