"""Exact-value regression guard for build_round_trips P&L math.

`reporter._realized_pl_today` ("what did I lock in today?", shown at the
daily close) and `/api/trade-asymmetry` / `/api/churn` all consume this as
the single source of truth. A sign flip or a missed scale-in here silently
mis-reports a trader's realized P&L, so pin the arithmetic on known ledgers.
"""
from __future__ import annotations

from paper_trader.analytics.round_trips import build_round_trips


def _trade(i, ticker, action, qty, price, ts, **kw):
    return {
        "id": i, "ticker": ticker, "action": action, "qty": qty,
        "price": price, "value": qty * price, "timestamp": ts,
        "option_type": kw.get("option_type"), "strike": kw.get("strike"),
        "expiry": kw.get("expiry"),
    }


def test_simple_winning_round_trip():
    trades = [
        _trade(1, "NVDA", "BUY", 2, 100.0, "2026-05-01T10:00:00+00:00"),
        _trade(2, "NVDA", "SELL", 2, 120.0, "2026-05-03T10:00:00+00:00"),
    ]
    rts = build_round_trips(trades)
    assert len(rts) == 1
    rt = rts[0]
    assert rt["cost"] == 200.0
    assert rt["proceeds"] == 240.0
    assert rt["pnl_usd"] == 40.0          # +20%/share * 2 shares
    assert rt["pnl_pct"] == 20.0
    assert rt["qty"] == 2
    assert rt["hold_days"] == 2.0
    assert rt["entry_ts"].startswith("2026-05-01")
    assert rt["exit_ts"].startswith("2026-05-03")


def test_scale_in_partial_close_then_full_close_nets_correctly():
    """Two BUYs, a partial SELL, a re-BUY, then a full SELL — one round-trip
    whose pnl is total proceeds − total cost, entry = the FIRST buy."""
    trades = [
        _trade(1, "MU", "BUY", 10, 50.0, "2026-05-01T10:00:00+00:00"),   # 500
        _trade(2, "MU", "BUY", 10, 60.0, "2026-05-02T10:00:00+00:00"),   # 600
        _trade(3, "MU", "SELL", 4, 70.0, "2026-05-03T10:00:00+00:00"),   # 280
        _trade(4, "MU", "BUY", 5, 65.0, "2026-05-04T10:00:00+00:00"),    # 325
        _trade(5, "MU", "SELL", 21, 72.0, "2026-05-05T10:00:00+00:00"),  # 1512
    ]
    rts = build_round_trips(trades)
    assert len(rts) == 1
    rt = rts[0]
    assert rt["cost"] == 500.0 + 600.0 + 325.0           # 1425
    assert rt["proceeds"] == 280.0 + 1512.0              # 1792
    assert rt["pnl_usd"] == round(1792.0 - 1425.0, 4)    # +367.0
    assert rt["qty"] == 25                               # 10+10+5 opened
    assert rt["n_buys"] == 3 and rt["n_sells"] == 2
    assert rt["entry_ts"].startswith("2026-05-01")       # FIRST buy
    assert rt["exit_ts"].startswith("2026-05-05")


def test_fractional_residue_closes_round_trip():
    """0.1 + 0.2 BUY then 0.3 SELL leaves held ≈ 4e-17 — the 1e-4 epsilon
    must still close the trip (else `_realized_pl_today` would never pair
    fractional-share scalps)."""
    trades = [
        _trade(1, "TQQQ", "BUY", 0.1, 100.0, "2026-05-01T10:00:00+00:00"),
        _trade(2, "TQQQ", "BUY", 0.2, 100.0, "2026-05-01T11:00:00+00:00"),
        _trade(3, "TQQQ", "SELL", 0.3, 110.0, "2026-05-02T10:00:00+00:00"),
        # A fresh BUY after the close must start a NEW round-trip.
        _trade(4, "TQQQ", "BUY", 0.5, 100.0, "2026-05-03T10:00:00+00:00"),
    ]
    rts = build_round_trips(trades)
    assert len(rts) == 1                       # only the closed one
    rt = rts[0]
    assert round(rt["pnl_usd"], 2) == 3.0      # 0.3 * (110-100)
    assert rt["entry_ts"].startswith("2026-05-01")


def test_open_position_produces_no_round_trip():
    trades = [_trade(1, "SPY", "BUY", 1, 400.0, "2026-05-01T10:00:00+00:00")]
    assert build_round_trips(trades) == []


def test_options_value_already_carries_x100_multiplier():
    """trades.value for an option row is qty*price*100 (store.record_trade);
    build_round_trips must not re-multiply — it sums value verbatim."""
    trades = [
        _trade(1, "NVDA", "BUY_CALL", 1, 5.0, "2026-05-01T10:00:00+00:00",
               option_type="call", strike=900, expiry="2026-06-19"),
        _trade(2, "NVDA", "SELL_CALL", 1, 8.0, "2026-05-02T10:00:00+00:00",
               option_type="call", strike=900, expiry="2026-06-19"),
    ]
    # Caller passes store-shaped value (already ×100).
    trades[0]["value"] = 1 * 5.0 * 100
    trades[1]["value"] = 1 * 8.0 * 100
    rts = build_round_trips(trades)
    assert len(rts) == 1
    assert rts[0]["pnl_usd"] == 300.0          # (8-5)*100, not *10000
    assert rts[0]["type"] == "call"
