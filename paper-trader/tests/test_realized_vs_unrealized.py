"""Tests for analytics/realized_vs_unrealized.py — pure, deterministic.

Pins:

* The algebraic invariant ``realized + unrealized == total − starting``
  for every point in the time series (the discriminating constraint
  that catches any sign / double-count drift no scenario test catches).
* The verdict ladder boundaries.
* Partial-sell realized accounting (proceeds − running-avg-cost·qty).
* The cost-basis walk's reset on full close (re-BUY uses the new price,
  not a stale blend).
* Option vs stock parity via ``trade.value`` directly so the option
  ×100 multiplier never bleeds into realized P&L.
* Robustness — garbage rows, naked-short degradation, empty inputs,
  unparseable timestamps, out-of-order curve points.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.realized_vs_unrealized import (
    DD_PCT,
    LEAK_PCT,
    MAX_SERIES_POINTS,
    MIN_NET_PCT,
    PAPER_HEAVY_SHARE,
    _compress_series,
    _walk_realized,
    build_realized_vs_unrealized,
)


NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


def _trade(ts: datetime, ticker: str, action: str, qty: float, price: float,
           strike=None, expiry=None, option_type=None) -> dict:
    """Helper that mirrors a Store.recent_trades row including the
    pre-computed ``value`` (price·qty for stocks, price·qty·100 for
    options — same convention as the live engine)."""
    mult = 100.0 if option_type else 1.0
    return {
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": price * qty * mult,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
    }


def _curve(*pts) -> list[dict]:
    """Helper: each ``pt`` is ``(ts, total_value)`` or
    ``(ts, total_value, cash)``."""
    out = []
    for p in pts:
        ts, tv = p[0], p[1]
        cash = p[2] if len(p) > 2 else None
        out.append({"timestamp": ts.isoformat(), "total_value": tv, "cash": cash})
    return out


class TestEmptyAndDegradedInputs:
    def test_no_trades_no_curve(self):
        r = build_realized_vs_unrealized([], [], starting_value=1000.0, now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["n_trades_walked"] == 0
        assert r["n_curve_points"] == 0
        assert r["series"] == []
        assert r["realized_pnl_usd"] == 0.0
        assert r["unrealized_pnl_usd"] == 0.0

    def test_trades_but_no_curve_still_no_data(self):
        # The curve is the spine of the time-series; without it we can't
        # attribute any P&L to a point in time.
        trades = [_trade(NOW - timedelta(days=1), "NVDA", "BUY", 1, 100)]
        r = build_realized_vs_unrealized(trades, [], 1000.0, now=NOW)
        assert r["verdict"] == "NO_DATA"

    def test_curve_only_no_trades(self):
        # Curve at exactly starting value ⇒ BALANCED (no gain, no loss).
        curve = _curve((NOW - timedelta(minutes=10), 1000.0),
                       (NOW, 1000.0))
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert r["verdict"] == "BALANCED"
        assert r["realized_pnl_usd"] == 0.0
        assert r["unrealized_pnl_usd"] == 0.0
        assert r["n_curve_points"] == 2

    def test_unparseable_curve_ts_skipped(self):
        curve = [{"timestamp": "not-a-date", "total_value": 1000.0},
                 {"timestamp": NOW.isoformat(), "total_value": 1000.0}]
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert r["n_curve_points"] == 1

    def test_unparseable_total_value_skipped(self):
        curve = [{"timestamp": NOW.isoformat(), "total_value": "nope"},
                 {"timestamp": NOW.isoformat(), "total_value": 1000.0}]
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert r["n_curve_points"] == 1

    def test_garbage_trade_rows_dont_raise(self):
        trades = [
            {"timestamp": None, "ticker": None, "action": None, "qty": None,
             "value": None},
            {"timestamp": "garbage", "action": "BUY", "qty": "x", "value": "y",
             "ticker": "NVDA"},
            _trade(NOW - timedelta(minutes=5), "NVDA", "BUY", 1, 100),
        ]
        curve = _curve((NOW, 1100.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        # Garbage rows are walked safely (zero qty / zero value → no
        # state change); the one real BUY is recorded.
        assert isinstance(r["unrealized_pnl_usd"], float)
        assert r["n_trades_walked"] == 3


class TestAlgebraicInvariant:
    """The single discriminating constraint that catches any sign /
    double-count bug: at EVERY point in the series, realized +
    unrealized must equal total − starting."""

    def test_invariant_simple_buy_sell(self):
        starting = 1000.0
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),   # cost 200
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 2, 110),  # +20 realized
        ]
        curve = _curve(
            (NOW - timedelta(hours=5), 1000.0),                       # before any trade
            (NOW - timedelta(hours=3), 1020.0),                       # mid-hold (mark up)
            (NOW - timedelta(hours=1), 1020.0),                       # after sell
        )
        r = build_realized_vs_unrealized(trades, curve, starting, now=NOW)
        for p in r["series"]:
            assert abs(p["realized"] + p["unrealized"] - (p["total"] - starting)) < 1e-6

    def test_invariant_partial_sell(self):
        starting = 1000.0
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 10, 100),  # cost 1000
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 4, 120),  # partial: realized = (120-100)*4 = 80
        ]
        curve = _curve(
            (NOW - timedelta(hours=5), 1000.0),
            (NOW - timedelta(hours=3), 1200.0),                       # mid-hold mark up 20%
            (NOW - timedelta(hours=1), 80.0 + (6 * 120) + 0.0),       # post-sell: cash 80 + 6 sh @ 120
        )
        r = build_realized_vs_unrealized(trades, curve, starting, now=NOW)
        for p in r["series"]:
            assert abs(p["realized"] + p["unrealized"] - (p["total"] - starting)) < 1e-6
        # Realized at the end should be exactly +$80 from the partial sell.
        assert r["realized_pnl_usd"] == 80.0

    def test_invariant_round_trip_then_reentry(self):
        starting = 1000.0
        trades = [
            _trade(NOW - timedelta(hours=6), "NVDA", "BUY", 2, 100),     # 200
            _trade(NOW - timedelta(hours=4), "NVDA", "SELL", 2, 110),    # +20 realized
            _trade(NOW - timedelta(hours=3), "NVDA", "BUY", 1, 105),     # re-entry @ 105
        ]
        curve = _curve(
            (NOW - timedelta(hours=5), 1020.0),                          # after 1st sell, before re-buy
            (NOW - timedelta(hours=2), 1020.0),                          # after re-buy: 915 cash + 105 mark
            (NOW - timedelta(hours=1), 1030.0),                          # mark up to 115
        )
        r = build_realized_vs_unrealized(trades, curve, starting, now=NOW)
        for p in r["series"]:
            assert abs(p["realized"] + p["unrealized"] - (p["total"] - starting)) < 1e-6
        # Final realized: only the +$20 from the closed round-trip.
        assert r["realized_pnl_usd"] == 20.0
        # Final unrealized: $10 mark-up on the new lot.
        assert r["unrealized_pnl_usd"] == 10.0


class TestCostBasisWalk:
    def test_running_avg_cost_blends_on_subsequent_buys(self):
        # 2 sh @ $100 then 2 sh @ $120 → avg cost $110.
        # Sell 4 sh @ $130 → realized = (130-110)*4 = $80.
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=3), "NVDA", "BUY", 2, 120),
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 4, 130),
        ]
        timeline = _walk_realized(trades)
        assert timeline[-1][1] == 80.0

    def test_partial_sell_keeps_avg_cost_unchanged(self):
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 10, 100),
            _trade(NOW - timedelta(hours=3), "NVDA", "SELL", 4, 130),     # +120
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 6, 90),      # -60
        ]
        timeline = _walk_realized(trades)
        # First sell realized: (130-100)*4 = 120. Cost basis stays $100
        # on the remaining 6 shares, so second sell: (90-100)*6 = -60.
        # Final cumulative: +60.
        assert timeline[-1][1] == 60.0

    def test_full_close_then_rebuy_resets_avg_cost(self):
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=3), "NVDA", "SELL", 2, 110),     # +20, FULL close
            _trade(NOW - timedelta(hours=2), "NVDA", "BUY", 1, 200),      # NEW lot @ 200
            _trade(NOW - timedelta(hours=1), "NVDA", "SELL", 1, 220),     # +20 (NOT +120 — blend would be wrong)
        ]
        timeline = _walk_realized(trades)
        assert timeline[-1][1] == 40.0   # 20 + 20, NOT 20 + 120

    def test_naked_short_degrades_to_full_proceeds(self):
        # No prior BUY — degrade gracefully (don't raise). Treat as full
        # realization of proceeds (the engine's risk gate already blocks
        # this; this is a never-raise contract test).
        trades = [_trade(NOW - timedelta(hours=1), "NVDA", "SELL", 1, 100)]
        timeline = _walk_realized(trades)
        assert timeline[-1][1] == 100.0

    def test_separate_tickers_dont_cross_contaminate(self):
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=3), "TQQQ", "BUY", 5, 70),
            _trade(NOW - timedelta(hours=2), "TQQQ", "SELL", 5, 75),       # +25
            _trade(NOW - timedelta(hours=1), "NVDA", "SELL", 2, 120),      # +40
        ]
        timeline = _walk_realized(trades)
        assert timeline[-1][1] == 65.0  # 25 + 40, never blended

    def test_option_realized_uses_value_directly(self):
        # Option contracts use value = price·qty·100. The walker should
        # not re-multiply, and stock+option keys must be distinct so an
        # option SELL never reads a stock's avg_cost.
        trades = [
            _trade(NOW - timedelta(hours=3), "NVDA", "BUY_CALL", 1, 5.0,
                   strike=220, expiry="2026-06-20", option_type="call"),
            _trade(NOW - timedelta(hours=1), "NVDA", "SELL_CALL", 1, 8.0,
                   strike=220, expiry="2026-06-20", option_type="call"),
        ]
        timeline = _walk_realized(trades)
        # Realized = (8 - 5) * 1 * 100 = 300
        assert timeline[-1][1] == 300.0


class TestVerdictLadder:
    """Pin each verdict branch boundary against a hand-crafted scenario."""

    def test_drawing_down(self):
        # Net P&L worse than -DD_PCT% of starting.
        trades = [
            _trade(NOW - timedelta(hours=3), "NVDA", "BUY", 10, 100),  # cost 1000
            _trade(NOW - timedelta(hours=1), "NVDA", "SELL", 10, 90),  # -100 realized
        ]
        # Final book value: starting 1000 - 100 loss = 900
        curve = _curve((NOW, 900.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["verdict"] == "DRAWING_DOWN"
        assert r["net_pnl_pct"] == -10.0

    def test_leaking_paper(self):
        # Realized > 0, unrealized worse than -LEAK_PCT% of starting.
        # First round-trip banks +$20, then a fresh buy goes underwater $10.
        trades = [
            _trade(NOW - timedelta(hours=6), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=4), "NVDA", "SELL", 2, 110),  # +20 banked
            _trade(NOW - timedelta(hours=2), "TQQQ", "BUY", 1, 100),
        ]
        # Total = realized(+20) + unrealized(-30) ⇒ total = 990
        # Unrealized -30 on $1000 starting = -3% (below LEAK_PCT=0.25%).
        curve = _curve((NOW, 990.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["verdict"] == "LEAKING_PAPER"

    def test_paper_heavy(self):
        # All gain is unrealized — one open position marked up.
        trades = [_trade(NOW - timedelta(hours=3), "NVDA", "BUY", 1, 100)]
        # Total = starting + $50 unrealized markup
        curve = _curve((NOW, 1050.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["verdict"] == "PAPER_HEAVY"
        assert r["realized_pnl_usd"] == 0.0
        assert r["unrealized_pnl_usd"] == 50.0

    def test_banked(self):
        # All gain is realized — closed round-trip, open book flat.
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 2, 110),  # +20 banked
        ]
        curve = _curve((NOW, 1020.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["verdict"] == "BANKED"
        assert r["realized_pnl_usd"] == 20.0
        assert r["unrealized_pnl_usd"] == 0.0

    def test_balanced_below_noise_floor(self):
        # Net P&L within ±MIN_NET_PCT of starting.
        curve = _curve((NOW, 1001.0))  # +0.1% — below 0.5%
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert r["verdict"] == "BALANCED"

    def test_paper_heavy_silenced_when_below_min_net(self):
        # Even if unrealized share is high, if net% is below MIN_NET_PCT
        # the verdict should collapse to BALANCED (noise floor).
        trades = [_trade(NOW - timedelta(hours=3), "NVDA", "BUY", 1, 100)]
        curve = _curve((NOW, 1003.0))  # +0.3% — below 0.5%
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["verdict"] == "BALANCED"

    def test_drawdown_threshold_constant_drives_boundary(self):
        # Below threshold ⇒ BALANCED; at-or-past ⇒ DRAWING_DOWN.
        below = _curve((NOW, 1000.0 - (DD_PCT - 0.1) * 10))  # net% > -DD_PCT
        past = _curve((NOW, 1000.0 - (DD_PCT + 0.1) * 10))   # net% < -DD_PCT
        r_below = build_realized_vs_unrealized([], below, 1000.0, now=NOW)
        r_past = build_realized_vs_unrealized([], past, 1000.0, now=NOW)
        assert r_below["verdict"] != "DRAWING_DOWN"
        assert r_past["verdict"] == "DRAWING_DOWN"


class TestSeriesAttribution:
    def test_realized_attaches_to_correct_curve_point(self):
        # Curve point BEFORE the sell should see 0 realized;
        # curve point AFTER should see +20.
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 2, 110),
        ]
        curve = _curve(
            (NOW - timedelta(hours=3), 1020.0),  # pre-sell, unrealized $20
            (NOW - timedelta(hours=1), 1020.0),  # post-sell, realized $20
        )
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert r["series"][0]["realized"] == 0.0
        assert r["series"][0]["unrealized"] == 20.0
        assert r["series"][1]["realized"] == 20.0
        assert r["series"][1]["unrealized"] == 0.0

    def test_out_of_order_curve_is_sorted(self):
        curve = _curve(
            (NOW, 1010.0),
            (NOW - timedelta(hours=1), 1005.0),
            (NOW - timedelta(hours=2), 1000.0),
        )
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert [p["total"] for p in r["series"]] == [1000.0, 1005.0, 1010.0]


class TestSeriesCompression:
    def test_short_series_passes_through(self):
        s = [{"i": i} for i in range(10)]
        assert _compress_series(s, 100) == s

    def test_long_series_caps_to_max(self):
        s = [{"i": i} for i in range(2000)]
        out = _compress_series(s, MAX_SERIES_POINTS)
        assert len(out) <= MAX_SERIES_POINTS
        assert out[0]["i"] == 0       # very first row always kept
        assert out[-1]["i"] == 1999   # very last row always kept

    def test_long_series_preserves_tail(self):
        # The last half of the budget should be the most recent rows.
        s = [{"i": i} for i in range(2000)]
        out = _compress_series(s, MAX_SERIES_POINTS)
        tail_n = MAX_SERIES_POINTS // 2
        tail_is = [r["i"] for r in out[-tail_n:]]
        assert tail_is == list(range(2000 - tail_n, 2000))

    def test_endpoint_payload_is_bounded(self):
        # Synthesize a long curve and confirm the wire payload caps to
        # MAX_SERIES_POINTS even though the engine consumes more.
        curve = [
            {"timestamp": (NOW - timedelta(minutes=i)).isoformat(),
             "total_value": 1000.0 + i * 0.01}
            for i in range(1500)
        ]
        r = build_realized_vs_unrealized([], curve, 1000.0, now=NOW)
        assert len(r["series"]) <= MAX_SERIES_POINTS
        assert r["n_curve_points"] == 1500


class TestThresholdContract:
    def test_thresholds_exposed_in_payload(self):
        # External docs / panels read these — pin the contract.
        r = build_realized_vs_unrealized([], [_curve((NOW, 1000.0))[0]],
                                         1000.0, now=NOW)
        t = r["thresholds"]
        assert t["min_net_pct"] == MIN_NET_PCT
        assert t["drawdown_pct"] == DD_PCT
        assert t["paper_heavy_share"] == PAPER_HEAVY_SHARE
        assert t["leak_pct"] == LEAK_PCT

    def test_starting_value_is_load_bearing(self):
        # Non-default starting → percentages scale accordingly.
        curve = _curve((NOW, 5050.0))
        r = build_realized_vs_unrealized([], curve, 5000.0, now=NOW)
        assert r["starting_value"] == 5000.0
        assert r["unrealized_pnl_usd"] == 50.0
        # +50/5000 = +1.00%, comfortably ≥ MIN_NET_PCT but the share
        # check still applies — entire gain is unrealized ⇒ PAPER_HEAVY
        assert r["verdict"] == "PAPER_HEAVY"


class TestHeadlineSanity:
    def test_drawing_down_headline_mentions_amount(self):
        trades = [_trade(NOW - timedelta(hours=3), "NVDA", "BUY", 10, 100),
                  _trade(NOW - timedelta(hours=1), "NVDA", "SELL", 10, 90)]
        curve = _curve((NOW, 900.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert "DRAWING_DOWN" in r["headline"]
        assert "-10" in r["headline"] or "-$100" in r["headline"] or "-100" in r["headline"]

    def test_paper_heavy_headline_mentions_share(self):
        trades = [_trade(NOW - timedelta(hours=3), "NVDA", "BUY", 1, 100)]
        curve = _curve((NOW, 1050.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert "PAPER_HEAVY" in r["headline"]
        assert "%" in r["headline"]

    def test_banked_headline_mentions_locked_in(self):
        trades = [
            _trade(NOW - timedelta(hours=4), "NVDA", "BUY", 2, 100),
            _trade(NOW - timedelta(hours=2), "NVDA", "SELL", 2, 110),
        ]
        curve = _curve((NOW, 1020.0))
        r = build_realized_vs_unrealized(trades, curve, 1000.0, now=NOW)
        assert "BANKED" in r["headline"]
        assert "locked-in" in r["headline"] or "realized" in r["headline"]


class TestLiveDataParity:
    """One smoke test against the actual paper_trader.db so a real
    schema / column-rename regression surfaces here, never raises."""

    def test_smokes_against_live_db(self):
        import sqlite3
        from pathlib import Path
        db = Path(__file__).resolve().parent.parent / "data" / "paper_trader.db"
        if not db.exists():
            return  # CI without seeded DB — degrade silently
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        trades = [dict(r) for r in conn.execute(
            "SELECT id, timestamp, ticker, action, qty, price, value, "
            "strike, expiry, option_type FROM trades ORDER BY id ASC"
        ).fetchall()]
        curve = [dict(r) for r in conn.execute(
            "SELECT timestamp, total_value, cash FROM equity_curve "
            "ORDER BY id ASC LIMIT 1000"
        ).fetchall()]
        conn.close()
        r = build_realized_vs_unrealized(trades, curve, 1000.0)
        # The discriminating invariant must hold even on real data
        # shapes (decimal noise, microsecond timestamps, etc.).
        starting = 1000.0
        for p in r["series"]:
            assert abs(p["realized"] + p["unrealized"]
                       - (p["total"] - starting)) < 1e-3
