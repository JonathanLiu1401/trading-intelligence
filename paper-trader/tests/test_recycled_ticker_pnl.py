"""Tests for paper_trader.analytics.recycled_ticker_pnl.

The verdict surfaces whether the bot's recycling habit pays — wrong
grouping, wrong sign on P&L, or mis-classified wins/losses would point
the operator at the wrong remediation. Every assertion below pins a
specific expected value, count, or verdict.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import recycled_ticker_pnl as rtp  # noqa: E402

_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _trade(action: str, ticker: str, hours_ago: float,
           qty: float = 1.0, price: float = 100.0, trade_id: int | None = None) -> dict:
    """One trade row in chronological feed (oldest first by caller convention).

    ``value`` is computed as qty*price to match the store.recent_trades shape.
    The trade ``id`` is optional but exercised by ``build_round_trips``.
    """
    ts = (_NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        "id": trade_id,
        "action": action,
        "ticker": ticker,
        "timestamp": ts,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "expiry": None,
        "strike": None,
        "option_type": None,
    }


# ─── NO_DATA / NO_RECYCLED_NAMES ──────────────────────────────────────

def test_empty_trades_returns_no_data():
    out = rtp.build_recycled_ticker_pnl([], now=_NOW)
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["n_round_trips"] == 0
    assert out["recycled_tickers"] == []
    assert out["headline"]


def test_none_trades_returns_no_data():
    out = rtp.build_recycled_ticker_pnl(None, now=_NOW)
    assert out["state"] == "NO_DATA"


def test_open_position_no_closed_round_trips_returns_no_data():
    # BUY but never SELL — no closed round-trips.
    trades = [_trade("BUY", "NVDA", 100.0, qty=2, price=200, trade_id=1)]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"


def test_one_closed_round_trip_is_not_recycled():
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=2, price=200, trade_id=1),
        _trade("SELL", "NVDA", 50.0, qty=2, price=220, trade_id=2),
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    # 1 round-trip — too few to call recycled.
    assert out["state"] == "OK"
    assert out["verdict"] == "NO_RECYCLED_NAMES"
    assert out["n_round_trips"] == 1
    assert out["n_recycled_tickers"] == 0


def test_two_round_trips_across_two_distinct_tickers_no_recycle():
    # Each name traded once ⇒ no recycling.
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=2, price=200, trade_id=1),
        _trade("SELL", "NVDA", 90.0, qty=2, price=220, trade_id=2),
        _trade("BUY", "AMD", 80.0, qty=4, price=100, trade_id=3),
        _trade("SELL", "AMD", 70.0, qty=4, price=110, trade_id=4),
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "NO_RECYCLED_NAMES"
    assert out["n_round_trips"] == 2
    assert out["n_distinct_tickers"] == 2
    assert out["n_recycled_tickers"] == 0
    # Both ended up in one_shot bucket.
    assert len(out["one_shot_tickers"]) == 2


# ─── WORTH_THE_CHURN (profitable recycling) ───────────────────────────

def test_profitable_recycle_worth_the_churn():
    # Two profitable NVDA round-trips ⇒ PROFITABLE_RECYCLE per name,
    # WORTH_THE_CHURN overall.
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=2, price=200, trade_id=1),  # cost 400
        _trade("SELL", "NVDA", 90.0, qty=2, price=220, trade_id=2),  # proceeds 440 → +40
        _trade("BUY", "NVDA", 50.0, qty=1, price=210, trade_id=3),   # cost 210
        _trade("SELL", "NVDA", 20.0, qty=1, price=230, trade_id=4),  # proceeds 230 → +20
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "WORTH_THE_CHURN"
    assert out["n_round_trips"] == 2
    assert out["n_recycled_tickers"] == 1
    nvda = out["recycled_tickers"][0]
    assert nvda["ticker"] == "NVDA"
    assert nvda["n_round_trips"] == 2
    assert nvda["n_wins"] == 2
    assert nvda["n_losses"] == 0
    assert nvda["realized_pnl_usd"] == 60.0
    assert nvda["verdict"] == "PROFITABLE_RECYCLE"
    assert out["net_realized_pnl_usd"] == 60.0


# ─── CHURN_DRAG (loss recycling) ──────────────────────────────────────

def test_drag_recycle_churn_drag():
    # Three losing NVDA round-trips — classic "trying to make it work" pattern.
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=2, price=200, trade_id=1),  # cost 400
        _trade("SELL", "NVDA", 180.0, qty=2, price=190, trade_id=2),  # proceeds 380 → -20
        _trade("BUY", "NVDA", 150.0, qty=1, price=210, trade_id=3),  # cost 210
        _trade("SELL", "NVDA", 120.0, qty=1, price=195, trade_id=4),  # proceeds 195 → -15
        _trade("BUY", "NVDA", 80.0, qty=1, price=200, trade_id=5),   # cost 200
        _trade("SELL", "NVDA", 40.0, qty=1, price=190, trade_id=6),   # proceeds 190 → -10
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "CHURN_DRAG"
    nvda = out["recycled_tickers"][0]
    assert nvda["realized_pnl_usd"] == -45.0
    assert nvda["n_wins"] == 0
    assert nvda["n_losses"] == 3
    assert nvda["verdict"] == "DRAG_RECYCLE"
    assert out["net_realized_pnl_usd"] == -45.0


def test_drag_recycle_by_win_rate_even_with_small_net_loss():
    # 4 trips, 1 win 3 losses but small dollars → DRAG_RECYCLE because
    # win_rate 0.25 < 0.34 AND net < 0.
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=1, price=200, trade_id=1),   # cost 200
        _trade("SELL", "NVDA", 195.0, qty=1, price=210, trade_id=2),  # proceeds 210 → +10
        _trade("BUY", "NVDA", 180.0, qty=1, price=200, trade_id=3),   # cost 200
        _trade("SELL", "NVDA", 175.0, qty=1, price=196, trade_id=4),  # -4
        _trade("BUY", "NVDA", 150.0, qty=1, price=200, trade_id=5),
        _trade("SELL", "NVDA", 145.0, qty=1, price=196, trade_id=6),  # -4
        _trade("BUY", "NVDA", 100.0, qty=1, price=200, trade_id=7),
        _trade("SELL", "NVDA", 95.0, qty=1, price=196, trade_id=8),   # -4
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    nvda = out["recycled_tickers"][0]
    assert nvda["n_wins"] == 1
    assert nvda["n_losses"] == 3
    assert nvda["realized_pnl_usd"] == -2.0
    # net = -2 → not in PROFITABLE zone, not >= -DRAG_USD ($1) threshold by net alone …
    # net is -2.0, which IS ≤ -_DRAG_USD ($1) → DRAG_RECYCLE on net.
    assert nvda["verdict"] == "DRAG_RECYCLE"


# ─── CHURN_NEUTRAL ────────────────────────────────────────────────────

def test_recycled_in_wash_band_is_neutral():
    # Two NVDA trips: small win + small loss, net ≈ 0 within ±$1 band.
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 90.0, qty=1, price=200.40, trade_id=2),  # +0.40 wash
        _trade("BUY", "NVDA", 50.0, qty=1, price=200, trade_id=3),
        _trade("SELL", "NVDA", 40.0, qty=1, price=199.80, trade_id=4),  # -0.20 wash
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    nvda = out["recycled_tickers"][0]
    # Both trips are washes (|pnl| ≤ $0.50).
    assert nvda["n_washes"] == 2
    assert nvda["n_wins"] == 0
    assert nvda["n_losses"] == 0
    assert nvda["verdict"] == "NEUTRAL_RECYCLE"
    assert out["verdict"] == "CHURN_NEUTRAL"


# ─── multi-ticker mix ─────────────────────────────────────────────────

def test_multi_ticker_only_recycled_count_in_net():
    # NVDA recycled +30 net; AMD one-shot -50 (NOT counted in net).
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 180.0, qty=1, price=220, trade_id=2),   # +20
        _trade("BUY", "NVDA", 150.0, qty=1, price=200, trade_id=3),
        _trade("SELL", "NVDA", 120.0, qty=1, price=210, trade_id=4),   # +10
        _trade("BUY", "AMD", 100.0, qty=1, price=100, trade_id=5),
        _trade("SELL", "AMD", 80.0, qty=1, price=50, trade_id=6),     # -50 (one-shot only)
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["n_round_trips"] == 3
    assert out["n_distinct_tickers"] == 2
    assert out["n_recycled_tickers"] == 1
    # net counts only NVDA's +30.
    assert out["net_realized_pnl_usd"] == 30.0
    assert out["verdict"] == "WORTH_THE_CHURN"
    assert len(out["recycled_tickers"]) == 1
    assert out["recycled_tickers"][0]["ticker"] == "NVDA"
    # AMD lives in the one-shot bucket.
    assert len(out["one_shot_tickers"]) == 1
    assert out["one_shot_tickers"][0]["ticker"] == "AMD"


def test_recycled_sorted_worst_first():
    # NVDA -50, AMD +30. Worst-realized-first ordering.
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 180.0, qty=1, price=180, trade_id=2),  # -20
        _trade("BUY", "NVDA", 150.0, qty=1, price=200, trade_id=3),
        _trade("SELL", "NVDA", 120.0, qty=1, price=170, trade_id=4),  # -30
        _trade("BUY", "AMD", 100.0, qty=1, price=100, trade_id=5),
        _trade("SELL", "AMD", 90.0, qty=1, price=120, trade_id=6),    # +20
        _trade("BUY", "AMD", 80.0, qty=1, price=100, trade_id=7),
        _trade("SELL", "AMD", 70.0, qty=1, price=110, trade_id=8),    # +10
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["n_recycled_tickers"] == 2
    # Worst first.
    assert out["recycled_tickers"][0]["ticker"] == "NVDA"
    assert out["recycled_tickers"][0]["realized_pnl_usd"] == -50.0
    assert out["recycled_tickers"][1]["ticker"] == "AMD"
    assert out["recycled_tickers"][1]["realized_pnl_usd"] == 30.0
    # Overall: -50 + 30 = -20 → CHURN_DRAG.
    assert out["net_realized_pnl_usd"] == -20.0
    assert out["verdict"] == "CHURN_DRAG"


# ─── per-ticker shape ─────────────────────────────────────────────────

def test_per_ticker_record_shape():
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 50.0, qty=1, price=220, trade_id=2),   # +20
        _trade("BUY", "NVDA", 40.0, qty=2, price=200, trade_id=3),
        _trade("SELL", "NVDA", 10.0, qty=2, price=215, trade_id=4),   # +30
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    nvda = out["recycled_tickers"][0]
    assert set(nvda.keys()) >= {
        "ticker", "n_round_trips", "n_wins", "n_losses", "n_washes",
        "win_rate", "total_cost_usd", "total_proceeds_usd",
        "realized_pnl_usd", "realized_pnl_pct", "avg_pnl_per_trip_usd",
        "best_trip_usd", "worst_trip_usd", "avg_hold_days",
        "first_entry_ts", "last_exit_ts", "verdict",
    }
    assert nvda["total_cost_usd"] == 600.0
    assert nvda["total_proceeds_usd"] == 650.0
    assert nvda["realized_pnl_usd"] == 50.0
    assert nvda["best_trip_usd"] == 30.0
    assert nvda["worst_trip_usd"] == 20.0
    assert nvda["avg_pnl_per_trip_usd"] == 25.0
    # win_rate computed as wins/n_round_trips.
    assert nvda["win_rate"] == 1.0
    # avg_hold_days populated.
    assert nvda["avg_hold_days"] is not None


def test_first_entry_and_last_exit_bookends():
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=1, price=200, trade_id=1),  # FIRST entry
        _trade("SELL", "NVDA", 180.0, qty=1, price=220, trade_id=2),
        _trade("BUY", "NVDA", 100.0, qty=1, price=200, trade_id=3),
        _trade("SELL", "NVDA", 50.0, qty=1, price=210, trade_id=4),   # LAST exit
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    nvda = out["recycled_tickers"][0]
    # first_entry should be 200h ago, last_exit should be 50h ago.
    assert nvda["first_entry_ts"] == trades[0]["timestamp"]
    assert nvda["last_exit_ts"] == trades[-1]["timestamp"]


# ─── option round-trips count toward ticker recycling ────────────────

def test_option_and_stock_round_trips_share_ticker_bucket():
    # One stock round-trip on NVDA + one option call round-trip on NVDA
    # ⇒ 2 round-trips against NVDA ⇒ recycled.
    trades = [
        _trade("BUY", "NVDA", 200.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 180.0, qty=1, price=220, trade_id=2),  # +20
        # Option call round-trip on NVDA.
        {"id": 3, "action": "BUY_CALL", "ticker": "NVDA",
         "timestamp": (_NOW - timedelta(hours=150.0)).isoformat(),
         "qty": 1, "price": 5.0, "value": 500.0, "expiry": "2026-06-19",
         "strike": 220.0, "option_type": "call"},
        {"id": 4, "action": "SELL_CALL", "ticker": "NVDA",
         "timestamp": (_NOW - timedelta(hours=140.0)).isoformat(),
         "qty": 1, "price": 6.0, "value": 600.0, "expiry": "2026-06-19",
         "strike": 220.0, "option_type": "call"},   # +100
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    nvda = out["recycled_tickers"][0]
    assert nvda["n_round_trips"] == 2
    assert nvda["realized_pnl_usd"] == 120.0


# ─── stability / never-raises discipline ──────────────────────────────

def test_malformed_rows_do_not_raise():
    trades = [
        _trade("BUY", "NVDA", 100.0, qty=1, price=200, trade_id=1),
        _trade("SELL", "NVDA", 90.0, qty=1, price=210, trade_id=2),
        {"action": None, "ticker": None, "qty": None,
         "price": None, "value": None, "timestamp": None},
        {"action": "BUY", "ticker": "AMD", "qty": "abc", "price": "xyz",
         "value": 0, "timestamp": "garbage"},
    ]
    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    # Doesn't raise.
    assert "verdict" in out


def test_thresholds_surfaced_in_response():
    out = rtp.build_recycled_ticker_pnl([], now=_NOW)
    assert out["thresholds"]["profitable_usd"] == rtp._PROFITABLE_USD
    assert out["thresholds"]["drag_usd"] == rtp._DRAG_USD
    assert out["thresholds"]["min_trips_to_recycle"] == rtp._MIN_TRIPS_TO_RECYCLE


# ─── live-shape: 3 NVDA tirps, drag — current live ledger reproduction ───

def test_live_shape_three_recycled_names_with_mixed_outcomes():
    # Reproduces the spirit of the 2026-05-24 live ledger (3 distinct
    # recycled names DRAM/NVDA/TQQQ, mixed P&L). Asserts the per-ticker
    # decomposition matches: 3 recycled names, sorted worst-first.
    trades = []
    tid = 0

    def _add(act, tk, h, q, p):
        nonlocal tid
        tid += 1
        trades.append(_trade(act, tk, h, qty=q, price=p, trade_id=tid))

    # DRAM 2 trips: +10, +5 ⇒ net +15 PROFITABLE
    _add("BUY", "DRAM", 200.0, 5, 20)
    _add("SELL", "DRAM", 190.0, 5, 22)   # +10
    _add("BUY", "DRAM", 150.0, 5, 20)
    _add("SELL", "DRAM", 140.0, 5, 21)   # +5
    # NVDA 2 trips: -20, -30 ⇒ net -50 DRAG
    _add("BUY", "NVDA", 130.0, 1, 200)
    _add("SELL", "NVDA", 120.0, 1, 180)  # -20
    _add("BUY", "NVDA", 100.0, 1, 200)
    _add("SELL", "NVDA", 80.0, 1, 170)   # -30
    # TQQQ 2 trips: +2, -1 ⇒ net +1 NEUTRAL band edge
    _add("BUY", "TQQQ", 60.0, 1, 100)
    _add("SELL", "TQQQ", 50.0, 1, 102)   # +2
    _add("BUY", "TQQQ", 40.0, 1, 100)
    _add("SELL", "TQQQ", 30.0, 1, 99)    # -1

    out = rtp.build_recycled_ticker_pnl(trades, now=_NOW)
    assert out["n_round_trips"] == 6
    assert out["n_recycled_tickers"] == 3
    # Sorted worst-first → NVDA, TQQQ, DRAM.
    tickers = [r["ticker"] for r in out["recycled_tickers"]]
    assert tickers == ["NVDA", "TQQQ", "DRAM"]
    # Net = +15 - 50 + 1 = -34 → CHURN_DRAG (well past -$1).
    assert out["net_realized_pnl_usd"] == -34.0
    assert out["verdict"] == "CHURN_DRAG"
    # Worst-record headline cites NVDA.
    assert "NVDA" in out["verdict_detail"]
