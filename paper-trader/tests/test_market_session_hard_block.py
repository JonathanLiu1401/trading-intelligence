"""Hard block: no paper fills when no equity trading session is open.

RTH / pre-market / after-hours / weekday overnight may fill.
WEEKEND and HOLIDAY must never mint a stock or option fill, even if the
model emits BUY/SELL. Prompt advisory is not enough — _execute enforces it.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from paper_trader import market, strategy
from paper_trader import store as store_mod
from paper_trader.store import Store

NY = ZoneInfo("America/New_York")


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


def _snap(cash=10_000.0, total=10_000.0, positions=None):
    return {
        "cash": cash,
        "total_value": total,
        "positions": positions or [],
        "positions_json": "[]",
    }


@pytest.mark.parametrize(
    "phase",
    ["WEEKEND", "HOLIDAY"],
)
@pytest.mark.parametrize(
    "action,extra",
    [
        ("BUY", {"ticker": "PLD", "qty": 1}),
        ("SELL", {"ticker": "PLD", "qty": 1}),
        ("SHORT", {"ticker": "PLD", "qty": 1}),
        ("COVER", {"ticker": "PLD", "qty": 1}),
        ("BUY_CALL", {"ticker": "PLD", "qty": 1, "strike": 100, "expiry": "2026-08-21"}),
        ("SELL_PUT", {"ticker": "PLD", "qty": 1, "strike": 100, "expiry": "2026-08-21"}),
    ],
)
def test_execute_blocks_fills_outside_any_session(fresh_store, monkeypatch, phase, action, extra):
    # Override the autouse "session always open" pin from conftest.
    monkeypatch.setattr(market, "is_any_trading_session_open", lambda now=None: False)
    monkeypatch.setattr(market, "market_phase", lambda now=None: phase)
    monkeypatch.setattr(market, "get_price", lambda t: 150.0)

    decision = {
        "action": action,
        "reasoning": "should never fill on weekend/holiday",
        **extra,
    }
    status, detail = strategy._execute(decision, _snap(), fresh_store)
    assert status == "BLOCKED", (status, detail)
    assert "no trading session open" in detail
    assert phase in detail
    assert fresh_store.recent_trades(5) == []


@pytest.mark.parametrize(
    "phase",
    [
        "PRE_MARKET",
        "OPENING_BELL",
        "MID_SESSION",
        "CLOSING_HALF_HOUR",
        "AFTER_CLOSE",
        "OVERNIGHT",
    ],
)
def test_execute_allows_stock_buy_in_live_sessions(fresh_store, monkeypatch, phase):
    monkeypatch.setattr(market, "is_any_trading_session_open", lambda now=None: True)
    monkeypatch.setattr(market, "market_phase", lambda now=None: phase)
    monkeypatch.setattr(market, "get_price", lambda t: 50.0)

    decision = {
        "action": "BUY",
        "ticker": "AAA",
        "qty": 2,
        "reasoning": f"allowed in {phase}",
    }
    status, detail = strategy._execute(decision, _snap(), fresh_store)
    assert status == "FILLED", (status, detail)
    trades = fresh_store.recent_trades(1)
    assert trades and trades[0]["ticker"] == "AAA"


def test_hold_still_ok_on_weekend(fresh_store, monkeypatch):
    monkeypatch.setattr(market, "is_any_trading_session_open", lambda now=None: False)
    monkeypatch.setattr(market, "market_phase", lambda now=None: "WEEKEND")
    status, _ = strategy._execute(
        {"action": "HOLD", "reasoning": "wait"},
        _snap(),
        fresh_store,
    )
    assert status == "HOLD"


def test_helper_matches_real_sunday_pld_bug_time(monkeypatch):
    # Exact class of bug: 2026-07-19 06:21 ET Sunday PLD buy.
    monkeypatch.setattr(
        market,
        "is_any_trading_session_open",
        lambda now=None: market.market_phase(now) in market._ANY_TRADING_SESSION_PHASES,
    )
    ts = datetime(2026, 7, 19, 6, 21, tzinfo=NY)
    assert market.market_phase(ts) == "WEEKEND"
    assert market.is_any_trading_session_open(ts) is False
    assert market.is_tradable_window_open(ts) is False
    assert market.is_market_open(ts) is False
