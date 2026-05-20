"""Today's action tape — chronological timeline of every TRADE and every
DECISION since UTC midnight.

Sibling to /api/session-delta (ranked materiality feed, arbitrary
``since=T``), /api/daily-recap (aggregate totals, no per-cycle rows),
and /api/decision-forensics (behavioural, decisions-only). This one is
the literal calendar-anchored chronological tape.

Pinned behaviour:
  * Pure / no DB / no LLM / no yfinance
  * Default since = today's UTC midnight
  * Rows include both TRADE and DECISION (HOLD / NO_DECISION / BLOCKED)
  * Chronological oldest → newest
  * Trade rows sort before decisions on same timestamp (chronology pin)
  * Aggregate counts match the tape contents
  * Never raises — per-class faults drop that class only
  * Out-of-window rows excluded
  * net_cash_flow_usd = notional_out − notional_in
  * Endpoint exists via Flask test client + clamps minutes
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trader.analytics.today_action_tape import (  # noqa: E402
    build_today_action_tape,
)

NOW = datetime(2026, 5, 20, 14, 0, 0, tzinfo=timezone.utc)
TODAY_MIDNIGHT = datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc)


def _trade(ts: datetime, ticker: str, action: str,
           qty: float = 1.0, price: float = 100.0, reason: str = "",
           option_type: str | None = None) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price * (100 if option_type in ("call", "put") else 1),
        "reason": reason,
        "expiry": None,
        "strike": None,
        "option_type": option_type,
    }


def _decision(ts: datetime, action_taken: str,
              reasoning: str = "", market_open: bool = True,
              signal_count: int = 0,
              portfolio_value: float = 1000.0,
              cash: float = 500.0) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "market_open": 1 if market_open else 0,
        "signal_count": signal_count,
        "action_taken": action_taken,
        "reasoning": reasoning,
        "portfolio_value": portfolio_value,
        "cash": cash,
    }


# ── pure shape ─────────────────────────────────────────────────────────────


def test_empty_input_returns_well_formed_envelope():
    out = build_today_action_tape([], [], now=NOW)
    assert out["n_trades"] == 0
    assert out["n_decisions"] == 0
    assert out["n_holds"] == 0
    assert out["n_no_decisions"] == 0
    assert out["tape"] == []
    assert out["as_of"] == NOW.isoformat()
    assert out["since"] == TODAY_MIDNIGHT.isoformat()
    assert out["net_cash_flow_usd"] == 0.0


def test_default_since_is_utc_midnight():
    """Anchoring is UTC-midnight when no explicit since."""
    out = build_today_action_tape([], [], now=NOW)
    assert out["since"] == TODAY_MIDNIGHT.isoformat()
    assert out["window_minutes"] == round(14.0 * 60.0, 2)


# ── windowing ──────────────────────────────────────────────────────────────


def test_yesterday_rows_excluded_by_default():
    """A trade from 23:59 UTC yesterday is outside today's window."""
    yest = TODAY_MIDNIGHT - timedelta(minutes=1)
    today = TODAY_MIDNIGHT + timedelta(hours=2)
    trades = [
        _trade(yest, "NVDA", "BUY", qty=1, price=100.0),
        _trade(today, "NVDA", "SELL", qty=1, price=110.0),
    ]
    out = build_today_action_tape(trades, [], now=NOW)
    assert out["n_trades"] == 1
    assert out["n_sells"] == 1
    assert out["n_buys"] == 0
    assert len(out["tape"]) == 1
    assert out["tape"][0]["action"] == "SELL"


def test_custom_since_overrides_default():
    """``since=`` lets the caller widen the window into prior days."""
    yest = TODAY_MIDNIGHT - timedelta(hours=4)
    today = TODAY_MIDNIGHT + timedelta(hours=2)
    trades = [
        _trade(yest, "NVDA", "BUY", qty=1, price=100.0),
        _trade(today, "NVDA", "SELL", qty=1, price=110.0),
    ]
    custom_since = TODAY_MIDNIGHT - timedelta(hours=6)
    out = build_today_action_tape(trades, [], now=NOW, since=custom_since)
    assert out["n_trades"] == 2
    assert len(out["tape"]) == 2


# ── chronological ordering ─────────────────────────────────────────────────


def test_tape_is_chronological_oldest_to_newest():
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    t1 = TODAY_MIDNIGHT + timedelta(hours=5)
    t2 = TODAY_MIDNIGHT + timedelta(hours=9)
    decisions = [
        _decision(t2, "HOLD"),
        _decision(t0, "BUY NVDA → FILLED"),
        _decision(t1, "NO_DECISION"),
    ]
    out = build_today_action_tape([], decisions, now=NOW)
    ts_list = [r["ts"] for r in out["tape"]]
    assert ts_list == [t0.isoformat(), t1.isoformat(), t2.isoformat()]


def test_trade_sorts_before_decision_on_tie():
    """Live runner records trade microseconds before decision; if they
    share a timestamp the trade row sorts first in the tape."""
    ts = TODAY_MIDNIGHT + timedelta(hours=3)
    trades = [_trade(ts, "NVDA", "BUY", qty=1, price=200.0)]
    decisions = [_decision(ts, "BUY NVDA → FILLED")]
    out = build_today_action_tape(trades, decisions, now=NOW)
    assert out["tape"][0]["kind"] == "TRADE"
    assert out["tape"][1]["kind"] == "DECISION"


# ── decision-class counting ────────────────────────────────────────────────


def test_decision_verbs_are_counted_correctly():
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    t1 = TODAY_MIDNIGHT + timedelta(hours=2)
    t2 = TODAY_MIDNIGHT + timedelta(hours=3)
    t3 = TODAY_MIDNIGHT + timedelta(hours=4)
    t4 = TODAY_MIDNIGHT + timedelta(hours=5)
    decisions = [
        _decision(t0, "HOLD"),
        _decision(t1, "NO_DECISION"),
        _decision(t2, "BLOCKED"),
        _decision(t3, "BUY NVDA → FILLED"),
        _decision(t4, "SELL NVDA → FILLED"),
    ]
    out = build_today_action_tape([], decisions, now=NOW)
    assert out["n_holds"] == 1
    assert out["n_no_decisions"] == 1
    assert out["n_blocked"] == 1
    assert out["n_other_decisions"] == 2  # BUY + SELL
    assert out["n_decisions"] == 5


def test_decision_action_parses_ticker():
    """Free-text action_taken like 'BUY NVDA → FILLED' yields ticker=NVDA."""
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    decisions = [_decision(t0, "BUY NVDA → FILLED")]
    out = build_today_action_tape([], decisions, now=NOW)
    row = out["tape"][0]
    assert row["verb"] == "BUY"
    assert row["ticker"] == "NVDA"


def test_decision_pseudo_ticker_collapses_to_none():
    """CASH / NONE pseudo-tickers don't pollute the ticker column."""
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    decisions = [_decision(t0, "HOLD CASH")]
    out = build_today_action_tape([], decisions, now=NOW)
    assert out["tape"][0]["verb"] == "HOLD"
    assert out["tape"][0]["ticker"] is None


# ── trade-class counting + net cash flow ───────────────────────────────────


def test_buy_sell_tallies_and_net_cash_flow():
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    t1 = TODAY_MIDNIGHT + timedelta(hours=2)
    t2 = TODAY_MIDNIGHT + timedelta(hours=3)
    trades = [
        _trade(t0, "NVDA", "BUY", qty=2, price=100.0),  # in: 200
        _trade(t1, "NVDA", "BUY", qty=1, price=110.0),  # in: 110
        _trade(t2, "NVDA", "SELL", qty=3, price=120.0),  # out: 360
    ]
    out = build_today_action_tape(trades, [], now=NOW)
    assert out["n_trades"] == 3
    assert out["n_buys"] == 2
    assert out["n_sells"] == 1
    assert out["notional_in_usd"] == 310.0
    assert out["notional_out_usd"] == 360.0
    # Net cash flow = proceeds - spend = 50
    assert out["net_cash_flow_usd"] == 50.0


def test_option_trade_notional_uses_value_field():
    """Builder must use the trade row's pre-computed ``value`` field
    (100×multiplier for options), not recompute."""
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    # _trade applies the 100x multiplier for options
    trades = [_trade(t0, "AAPL", "BUY_CALL", qty=2, price=5.0,
                     option_type="call")]
    out = build_today_action_tape(trades, [], now=NOW)
    # value = 2 * 5 * 100 = 1000
    assert out["notional_in_usd"] == 1000.0
    assert out["tape"][0]["option_type"] == "call"


# ── robustness ─────────────────────────────────────────────────────────────


def test_garbage_timestamp_is_skipped_not_raised():
    """A row with an unparseable timestamp drops from the tape, not the
    whole report. The session_delta precedent (per-class drop, never
    sink the whole feed)."""
    t0 = TODAY_MIDNIGHT + timedelta(hours=1)
    trades = [
        {"timestamp": "not-a-timestamp", "ticker": "NVDA",
         "action": "BUY", "qty": 1, "price": 100, "value": 100},
        _trade(t0, "AAPL", "BUY", qty=1, price=150.0),
    ]
    out = build_today_action_tape(trades, [], now=NOW)
    assert out["n_trades"] == 1
    assert out["tape"][0]["ticker"] == "AAPL"


def test_future_dated_row_is_excluded():
    """A row with ts > now is impossible normal-path; defensively excluded."""
    future = NOW + timedelta(hours=1)
    trades = [_trade(future, "NVDA", "BUY", qty=1, price=100.0)]
    out = build_today_action_tape(trades, [], now=NOW)
    assert out["n_trades"] == 0


# ── route ──────────────────────────────────────────────────────────────────


def test_route_via_flask_test_client():
    """The endpoint loads through Flask and returns a JSON envelope.

    Uses Flask test_client (the AGENTS.md verification pattern — the
    module __main__ smoke would hit an empty DB)."""
    from paper_trader.dashboard import app
    client = app.test_client()
    r = client.get("/api/today-action-tape")
    assert r.status_code == 200
    data = r.get_json()
    assert "as_of" in data
    assert "since" in data
    assert "tape" in data
    assert "n_trades" in data
    assert isinstance(data["tape"], list)


def test_route_accepts_minutes_param():
    """?minutes= overrides UTC-midnight to "last N minutes"."""
    from paper_trader.dashboard import app
    client = app.test_client()
    r = client.get("/api/today-action-tape?minutes=120")
    assert r.status_code == 200
    data = r.get_json()
    # since should now be ~120 min ago, not UTC midnight.
    since_dt = datetime.fromisoformat(data["since"].replace("Z", "+00:00"))
    as_of_dt = datetime.fromisoformat(data["as_of"].replace("Z", "+00:00"))
    diff_min = (as_of_dt - since_dt).total_seconds() / 60.0
    # Allow a small tolerance for the test_client wall-clock difference.
    assert 100 <= diff_min <= 140
