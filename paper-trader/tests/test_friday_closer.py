from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from paper_trader.analytics import friday_closer as fc
from paper_trader.analytics import options_desk
from paper_trader.analytics import options_risk

ET = ZoneInfo("America/New_York")


def _pos(ticker, typ, qty, expiry, strike, avg=1.0, mark=1.0):
    return {
        "ticker": ticker,
        "type": typ,
        "qty": qty,
        "expiry": expiry,
        "strike": strike,
        "avg_cost": avg,
        "current_price": mark,
    }


def test_leftover_sep_mark_does_not_consume_new_debit_cap():
    positions = [
        _pos("NVDA", "call", 3, "2026-09-18", 220, avg=7.25, mark=7.50),
        _pos("NVDA", "call", -3, "2026-09-18", 225, avg=5.45, mark=5.70),
        _pos("SK", "call", 3, "2026-09-18", 32, avg=1.05, mark=0.90),
        _pos("SK", "call", -3, "2026-09-18", 35, avg=0.40, mark=0.32),
    ]
    leftover = fc.leftover_option_mark(positions)
    assert leftover > 800
    assert fc.new_debit_budget_remaining(positions) == 800.0
    assert fc.has_friday_weekly(positions) is False
    assert len(fc.wrong_issuer_lots(positions)) == 2


def test_friday_weekly_consumes_new_debit_cap():
    positions = [
        _pos("NVDA", "call", 3, "2026-08-14", 185, avg=2.0, mark=2.0),
        _pos("NVDA", "call", -3, "2026-08-14", 190, avg=1.0, mark=1.0),
    ]
    used = 3 * 2.0 * 100 + 0  # long mark only
    assert fc.new_debit_budget_remaining(positions) == 800.0 - used
    assert fc.has_friday_weekly(positions) is True


def test_build_call_spread_when_tape_is_not_hot():
    snap = {"positions": [], "cash": 6000}
    d = fc.build_friday_weekly_decision(snap, spot=184.0, prior_close=183.0, ticker="NVDA")
    assert d is not None
    assert d["action"] == "BUY_CALL_SPREAD"
    assert d["expiry"] == "2026-08-14"
    assert d["ticker"] == "NVDA"
    assert d["qty"] == 3
    assert d["short_strike"] == d["long_strike"] + 5


def test_build_put_spread_when_tape_is_hot():
    snap = {"positions": [], "cash": 6000}
    d = fc.build_friday_weekly_decision(snap, spot=180.0, prior_close=190.0, ticker="NVDA", hot=True)
    assert d is not None
    assert d["action"] == "BUY_PUT_SPREAD"
    assert d["short_strike"] == d["long_strike"] - 5


def test_ppi_window_waits_until_thursday_0845():
    before = datetime(2026, 8, 13, 8, 44, tzinfo=ET)
    after = datetime(2026, 8, 13, 8, 45, tzinfo=ET)
    wed = datetime(2026, 8, 12, 16, 0, tzinfo=ET)
    assert fc.ppi_window_open(wed) is False
    assert fc.ppi_window_open(before) is False
    assert fc.ppi_window_open(after) is True


def test_desk_snapshots_friday_weekly_dte():
    src = Path(__file__).resolve().parents[1] / "paper_trader" / "analytics" / "options_desk.py"
    text = src.read_text()
    assert "for target in (2, 45, 365)" in text
    tickers = options_desk._candidate_tickers(
        [{"ticker": "SK", "qty": 1, "type": "call"}],
        ["MU"],
        {},
        limit=4,
    )
    assert tickers[:2] == ["NVDA", "QQQ"]


def test_options_risk_separates_leftover_from_new_debit():
    positions = [
        _pos("NVDA", "call", 3, "2026-09-18", 220, avg=7.25, mark=7.50),
        _pos("NVDA", "call", -3, "2026-09-18", 225, avg=5.45, mark=5.70),
    ]
    block = options_risk.build_options_risk(
        {"positions": positions, "cash": 6000, "total_value": 9231}
    )["prompt_block"]
    assert "leftover_long_option_mark" in block
    assert "new 8/14 debit remaining=$800.00" in block
    assert "NOT the new-debit cap" in block


def test_maybe_execute_does_nothing_when_market_closed():
    notes = fc.maybe_execute_friday_closer(
        store=None,
        snapshot={"positions": []},
        market_mod=None,
        execute_fn=lambda *a, **k: ("filled", "x"),
        market_open=False,
    )
    assert notes == []
