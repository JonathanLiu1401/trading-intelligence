"""Tests for the per-position upside profit ladder.

Two layers:

* **Builder** — ``paper_trader.analytics.profit_ladder.build_profit_ladder``,
  symmetric to ``position_blowup`` (DOWN shocks) but UP. These tests pin
  the exact shock arithmetic, the trim-yield realized/unrealized split,
  the most-upside-first sort, the options ×100 multiplier, and the
  never-raises / NO_DATA degradation contract.

* **Endpoint** — ``/api/profit-ladder`` (plain GET, not @swr_cached).
  Round-trips the builder via the live store; asserts byte-identical
  parity with the direct builder call (the SSOT invariant).

No live process, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.analytics.profit_ladder as pl
from paper_trader.analytics.profit_ladder import (
    build_profit_ladder,
    UPSIDE_SHOCK_PCT,
    TRIM_FRACTIONS,
    BIG_WINNER_AT_25PCT_THRESHOLD,
)


# ───────────────────────────── builder: degradation ────────────────────────
def test_no_positions_is_no_data():
    res = build_profit_ladder([], 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []
    assert res["n_positions"] == 0
    assert "no priced book" in res["headline"]


def test_none_positions_is_no_data():
    res = build_profit_ladder(None, 1000.0)
    assert res["state"] == "NO_DATA"


def test_zero_total_value_is_no_data():
    res = build_profit_ladder(
        [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}], 0.0,
    )
    assert res["state"] == "NO_DATA"


def test_garbage_total_value_is_no_data():
    res = build_profit_ladder(
        [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}],
        "not-a-number",
    )
    assert res["state"] == "NO_DATA"


def test_all_rows_zero_value_is_no_data():
    """Every row prices to 0 — NO_DATA, not BIG_WINNERS on phantom rows."""
    rows = [
        {"ticker": "NVDA", "current_price": 0.0, "qty": 0, "avg_cost": 0.0},
        {"ticker": "AMD", "current_price": None, "qty": None,
         "avg_cost": None},
    ]
    res = build_profit_ladder(rows, 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []


# ───────────────────────────── builder: shock arithmetic ───────────────────
def test_shock_arithmetic_is_exact():
    """A $600 single name (3 shares @ $200, avg_cost $180) in a $1000 book.

    +10 %: shocked price $220, mv $660, gain $60 (6% of book),
           unrealized at shock = $660 - $540 cost basis = $120.
    +25 %: shocked $250, mv $750, gain $150 (15% of book), unreal $210.
    +50 %: shocked $300, mv $900, gain $300 (30% of book), unreal $360.
    +100 %: shocked $400, mv $1200, gain $600 (60% of book), unreal $660.
    """
    rows = [{
        "ticker": "NVDA",
        "type": "stock",
        "current_price": 200.0,
        "avg_cost": 180.0,
        "qty": 3,
    }]
    res = build_profit_ladder(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["market_value_usd"] == 600.0
    assert pos["cost_basis_usd"] == 540.0
    assert pos["unrealized_pl_usd_now"] == 60.0
    assert pos["weight_pct"] == 60.0
    by_mag = {s["shock_pct"]: s for s in pos["shocks"]}

    assert by_mag[10.0]["shocked_market_value_usd"] == 660.0
    assert by_mag[10.0]["gain_above_current_usd"] == 60.0
    assert by_mag[10.0]["gain_above_current_pct_of_book"] == 6.0
    assert by_mag[10.0]["unrealized_pl_at_shock_usd"] == 120.0

    assert by_mag[25.0]["gain_above_current_usd"] == 150.0
    assert by_mag[25.0]["gain_above_current_pct_of_book"] == 15.0
    assert by_mag[25.0]["unrealized_pl_at_shock_usd"] == 210.0

    assert by_mag[50.0]["gain_above_current_usd"] == 300.0
    assert by_mag[50.0]["gain_above_current_pct_of_book"] == 30.0
    assert by_mag[50.0]["unrealized_pl_at_shock_usd"] == 360.0

    assert by_mag[100.0]["gain_above_current_usd"] == 600.0
    assert by_mag[100.0]["gain_above_current_pct_of_book"] == 60.0
    assert by_mag[100.0]["unrealized_pl_at_shock_usd"] == 660.0


def test_shocks_carry_all_five_magnitudes():
    rows = [{"ticker": "NVDA", "current_price": 100.0, "avg_cost": 100.0,
             "qty": 1}]
    res = build_profit_ladder(rows, 100.0)
    shock_pcts = [s["shock_pct"] for s in res["positions"][0]["shocks"]]
    assert shock_pcts == [5.0, 10.0, 25.0, 50.0, 100.0]


def test_max_gain_equals_position_value():
    """+100 % == doubling, so max_gain_usd MUST equal current market value."""
    rows = [{"ticker": "NVDA", "current_price": 100.0, "avg_cost": 90.0,
             "qty": 4}]
    res = build_profit_ladder(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["market_value_usd"] == 400.0
    assert pos["max_gain_usd"] == 400.0
    assert pos["max_gain_pct_of_book"] == 40.0


# ───────────────────────────── builder: trim schedule ──────────────────────
def test_trim_schedule_realized_unrealized_split():
    """Position: 4 shares @ avg_cost $100, current $100, $1000 book.
    At +25 % rung shocked price = $125 / mv $500.

    Trim 25 % (qty 1): cash freed $125, cost freed $100, realized +$25.
        remaining 3 shares × $125 = $375 mv, unrealized = 3×($125-$100) = $75.
    Trim 50 % (qty 2): cash $250, cost $200, realized +$50.
        remaining 2 × $125 = $250, unrealized = 2×$25 = $50.
    Trim 100 % (qty 4): cash $500, cost $400, realized +$100.
        remaining 0, value 0, unrealized 0.
    """
    rows = [{"ticker": "X", "current_price": 100.0, "avg_cost": 100.0,
             "qty": 4}]
    res = build_profit_ladder(rows, 1000.0)
    shocks = {s["shock_pct"]: s for s in res["positions"][0]["shocks"]}
    s25 = shocks[25.0]
    trims = {t["trim_pct"]: t for t in s25["trim_schedule"]}

    assert trims[25.0]["trim_qty"] == 1.0
    assert trims[25.0]["cash_freed_usd"] == 125.0
    assert trims[25.0]["realized_pl_usd"] == 25.0
    assert trims[25.0]["remaining_qty"] == 3.0
    assert trims[25.0]["remaining_value_usd"] == 375.0
    assert trims[25.0]["remaining_unrealized_pl_usd"] == 75.0

    assert trims[50.0]["trim_qty"] == 2.0
    assert trims[50.0]["realized_pl_usd"] == 50.0
    assert trims[50.0]["remaining_unrealized_pl_usd"] == 50.0

    assert trims[100.0]["trim_qty"] == 4.0
    assert trims[100.0]["realized_pl_usd"] == 100.0
    assert trims[100.0]["remaining_qty"] == 0.0
    assert trims[100.0]["remaining_value_usd"] == 0.0
    assert trims[100.0]["remaining_unrealized_pl_usd"] == 0.0


def test_trim_schedule_on_underwater_lot_can_lock_loss():
    """avg_cost $200, current $100 — underwater. At +10 % shocked $110,
    trim 100 % realizes 4×($110-$200) = -$360 (a locked loss, not a gain).
    The trim-schedule must report negative realized P&L honestly."""
    rows = [{"ticker": "X", "current_price": 100.0, "avg_cost": 200.0,
             "qty": 4}]
    res = build_profit_ladder(rows, 1000.0)
    shocks = {s["shock_pct"]: s for s in res["positions"][0]["shocks"]}
    s10 = shocks[10.0]
    trims = {t["trim_pct"]: t for t in s10["trim_schedule"]}
    assert trims[100.0]["realized_pl_usd"] == -360.0
    assert trims[100.0]["cash_freed_usd"] == 440.0  # 4 × $110


# ───────────────────────────── builder: options ×100 multiplier ────────────
def test_options_multiplier_100x():
    """One call contract at $5 premium (×100 mult) = $500 notional.
    +25 % → shocked premium $6.25, mv $625, gain $125, trim 50 % qty 0.5
    contracts: cash $6.25×0.5×100 = $312.50."""
    rows = [{
        "ticker": "NVDA",
        "type": "call",
        "current_price": 5.0,
        "avg_cost": 4.0,
        "qty": 1,
    }]
    res = build_profit_ladder(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["market_value_usd"] == 500.0
    assert pos["cost_basis_usd"] == 400.0
    assert pos["unrealized_pl_usd_now"] == 100.0
    by_mag = {s["shock_pct"]: s for s in pos["shocks"]}
    assert by_mag[25.0]["shocked_market_value_usd"] == 625.0
    assert by_mag[25.0]["gain_above_current_usd"] == 125.0
    trims = {t["trim_pct"]: t for t in by_mag[25.0]["trim_schedule"]}
    assert trims[50.0]["cash_freed_usd"] == 312.5
    # realized: cash 312.5 - cost 0.5×4×100=200 = 112.5
    assert trims[50.0]["realized_pl_usd"] == 112.5


# ───────────────────────────── builder: sort + verdicts ────────────────────
def test_rows_sorted_most_upside_first():
    rows = [
        {"ticker": "SMALL", "current_price": 10.0, "avg_cost": 10.0,
         "qty": 1},   # $10 mv
        {"ticker": "BIG", "current_price": 100.0, "avg_cost": 100.0,
         "qty": 5},   # $500 mv
        {"ticker": "MID", "current_price": 50.0, "avg_cost": 50.0,
         "qty": 2},   # $100 mv
    ]
    res = build_profit_ladder(rows, 1000.0)
    tickers = [r["ticker"] for r in res["positions"]]
    assert tickers == ["BIG", "MID", "SMALL"]


def test_recovery_book_verdict_all_underwater():
    """Every position underwater → RECOVERY_BOOK regardless of upside."""
    rows = [
        {"ticker": "A", "current_price": 80.0, "avg_cost": 100.0, "qty": 1},
        {"ticker": "B", "current_price": 90.0, "avg_cost": 100.0, "qty": 1},
    ]
    res = build_profit_ladder(rows, 1000.0)
    assert res["state"] == "RECOVERY_BOOK"


def test_in_profit_verdict_all_green_small():
    """All green but no rung exceeds the BIG_WINNERS threshold."""
    rows = [{"ticker": "X", "current_price": 110.0, "avg_cost": 100.0,
             "qty": 1}]  # $110 mv in a $10000 book
    res = build_profit_ladder(rows, 10000.0)
    assert res["state"] == "IN_PROFIT"


def test_big_winners_verdict_at_25pct_rung():
    """Single name where +25% rung ≥ BIG_WINNER_AT_25PCT_THRESHOLD %
    of book → BIG_WINNERS."""
    # 4 shares @ $100 = $400 mv, $1000 book. +25 % gain = $100 = 10 % of book.
    rows = [{"ticker": "X", "current_price": 100.0, "avg_cost": 100.0,
             "qty": 4}]
    res = build_profit_ladder(rows, 1000.0)
    assert res["state"] == "BIG_WINNERS"


def test_mixed_book_verdict_green_and_red():
    rows = [
        {"ticker": "GREEN", "current_price": 110.0, "avg_cost": 100.0,
         "qty": 1},
        {"ticker": "RED", "current_price": 90.0, "avg_cost": 100.0, "qty": 1},
    ]
    res = build_profit_ladder(rows, 1000.0)
    assert res["state"] == "MIXED_BOOK"


# ───────────────────────────── builder: edge cases ─────────────────────────
def test_missing_avg_cost_degrades_to_current_mark():
    """A row without avg_cost should not raise; cost basis degrades to
    current market value so unrealized_pl_at_shock equals the gain rung."""
    rows = [{"ticker": "X", "current_price": 100.0, "qty": 1,
             "avg_cost": None}]
    res = build_profit_ladder(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["cost_basis_usd"] == 100.0
    assert pos["unrealized_pl_usd_now"] == 0.0
    by_mag = {s["shock_pct"]: s for s in pos["shocks"]}
    # +25 % rung: unrealized_at_shock == gain_above_current since basis ==
    # current mark.
    assert by_mag[25.0]["unrealized_pl_at_shock_usd"] == by_mag[25.0][
        "gain_above_current_usd"]


def test_negative_zero_folded_to_zero():
    """``_z`` MUST fold a signed -0.0 to +0.0 so JSON serialization
    never emits ``-0.0`` (matches the ``position_blowup._z`` /
    ``stress_scenarios._z`` precedent). Direct check on the helper +
    a representative builder run that produces a zero rung."""
    assert pl._z(-0.0) == 0.0
    assert repr(pl._z(-0.0)) == "0.0"
    # A trim-100% rung on a flat-basis position must report exactly
    # 0.0 remaining_unrealized (no signed zero leaking through).
    rows = [{"ticker": "X", "current_price": 100.0, "avg_cost": 100.0,
             "qty": 1}]
    res = build_profit_ladder(rows, 100.0)
    for s in res["positions"][0]["shocks"]:
        full_trim = [t for t in s["trim_schedule"] if t["trim_pct"] == 100.0][0]
        assert full_trim["remaining_unrealized_pl_usd"] == 0.0
        assert repr(full_trim["remaining_unrealized_pl_usd"]) == "0.0"
        assert full_trim["remaining_qty"] == 0.0
        assert full_trim["remaining_value_usd"] == 0.0


def test_headline_mentions_top_position_ticker_and_verdict():
    rows = [{"ticker": "NVDA", "current_price": 100.0, "avg_cost": 100.0,
             "qty": 4}]
    res = build_profit_ladder(rows, 1000.0)
    assert "NVDA" in res["headline"]
    assert res["state"] in res["headline"]


def test_thresholds_are_constants_not_magic():
    """Lock the published constants so a re-tune is a visible code edit."""
    assert UPSIDE_SHOCK_PCT == (5.0, 10.0, 25.0, 50.0, 100.0)
    assert TRIM_FRACTIONS == (0.25, 0.50, 1.00)
    assert BIG_WINNER_AT_25PCT_THRESHOLD == 10.0


# ───────────────────────────── endpoint parity ─────────────────────────────
def test_endpoint_returns_builder_output_byte_for_byte(monkeypatch):
    """The /api/profit-ladder route MUST be a pure passthrough — no
    extra keys, no shape drift from the builder. SSOT invariant."""
    fake_positions = [{
        "ticker": "X",
        "type": "stock",
        "current_price": 100.0,
        "avg_cost": 90.0,
        "qty": 2,
    }]
    fake_portfolio = {"total_value": 500.0}

    class _FakeStore:
        def open_positions(self):
            return fake_positions

        def get_portfolio(self):
            return fake_portfolio

    monkeypatch.setattr(dash, "get_store", lambda: _FakeStore())

    expected = build_profit_ladder(fake_positions, 500.0)
    with dash.app.test_client() as client:
        rv = client.get("/api/profit-ladder")
    assert rv.status_code == 200
    got = rv.get_json()
    # as_of will drift by milliseconds between the two calls; mask it.
    got.pop("as_of", None)
    expected.pop("as_of", None)
    assert got == expected
