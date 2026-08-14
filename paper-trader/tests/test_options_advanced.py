"""Advanced options: spreads, covered calls, CSPs, buy-to-cover."""
from __future__ import annotations

from paper_trader.analytics import options_structures as osx
from paper_trader.analytics import options_risk


class _Mkt:
    def __init__(self, prices):
        self.prices = prices

    def get_option_price(self, ticker, expiry, strike, otype):
        return self.prices.get((ticker, expiry, float(strike), otype))


def test_parse_debit_call_spread_orientation():
    spec = osx.parse_vertical_spread(
        {
            "action": "BUY_CALL_SPREAD",
            "ticker": "AAPL",
            "qty": 1,
            "expiry": "2026-12-18",
            "long_strike": 180,
            "short_strike": 190,
        }
    )
    assert spec["ok"] is True
    assert spec["is_debit"] is True
    assert spec["width"] == 10


def test_parse_debit_call_spread_bad_orientation():
    spec = osx.parse_vertical_spread(
        {
            "action": "BUY_CALL_SPREAD",
            "ticker": "AAPL",
            "qty": 1,
            "expiry": "2026-12-18",
            "long_strike": 190,
            "short_strike": 180,
        }
    )
    assert spec["ok"] is False


def test_parse_spread_strike_width():
    spec = osx.parse_vertical_spread(
        {
            "action": "BUY_CALL_SPREAD",
            "ticker": "NVDA",
            "qty": 2,
            "expiry": "2026-09-18",
            "strike": 120,
            "width": 10,
        }
    )
    assert spec["ok"] is True
    assert spec["long_strike"] == 120
    assert spec["short_strike"] == 130


def test_price_debit_spread():
    m = _Mkt(
        {
            ("AAPL", "2026-12-18", 180.0, "call"): 8.0,
            ("AAPL", "2026-12-18", 190.0, "call"): 3.0,
        }
    )
    spec = osx.parse_vertical_spread(
        {
            "action": "BUY_CALL_SPREAD",
            "ticker": "AAPL",
            "qty": 1,
            "expiry": "2026-12-18",
            "long_strike": 180,
            "short_strike": 190,
        }
    )
    priced = osx.price_vertical_spread(m, spec)
    assert priced["ok"] is True
    assert abs(priced["net_per_share"] - 5.0) < 1e-9
    assert abs(priced["cash_delta"] + 500.0) < 1e-9
    assert abs(priced["max_loss"] - 500.0) < 1e-9
    assert abs(priced["max_gain"] - 500.0) < 1e-9


def test_price_credit_put_spread():
    m = _Mkt(
        {
            ("AAPL", "2026-12-18", 180.0, "put"): 4.0,  # short
            ("AAPL", "2026-12-18", 170.0, "put"): 1.5,  # long
        }
    )
    spec = osx.parse_vertical_spread(
        {
            "action": "SELL_PUT_SPREAD",
            "ticker": "AAPL",
            "qty": 1,
            "expiry": "2026-12-18",
            "short_strike": 180,
            "long_strike": 170,
        }
    )
    priced = osx.price_vertical_spread(m, spec)
    assert priced["ok"] is True
    assert priced["cash_delta"] > 0
    assert priced["max_loss"] > 0


def test_classify_covered_call_and_naked_block():
    snap = {
        "cash": 10000,
        "positions": [
            {"ticker": "AAPL", "type": "stock", "qty": 100, "avg_cost": 180},
        ],
    }
    ok = osx.classify_sell_option(
        {
            "action": "SELL_CALL",
            "ticker": "AAPL",
            "qty": 1,
            "strike": 200,
            "expiry": "2026-12-18",
        },
        snap,
    )
    assert ok["mode"] == "open_covered_call"

    naked = osx.classify_sell_option(
        {
            "action": "SELL_CALL",
            "ticker": "MSFT",
            "qty": 1,
            "strike": 400,
            "expiry": "2026-12-18",
        },
        {"cash": 50000, "positions": []},
    )
    assert naked["mode"] == "blocked"


def test_classify_close_long_call():
    snap = {
        "cash": 10000,
        "positions": [
            {
                "ticker": "AAPL",
                "type": "call",
                "qty": 2,
                "strike": 200,
                "expiry": "2026-12-18",
                "avg_cost": 5,
            }
        ],
    }
    cls = osx.classify_sell_option(
        {
            "action": "SELL_CALL",
            "ticker": "AAPL",
            "qty": 1,
            "strike": 200,
            "expiry": "2026-12-18",
        },
        snap,
    )
    assert cls["mode"] == "close_long"


def test_classify_buy_to_cover():
    snap = {
        "cash": 10000,
        "positions": [
            {
                "ticker": "AAPL",
                "type": "put",
                "qty": -1,
                "strike": 170,
                "expiry": "2026-12-18",
                "avg_cost": 3,
            }
        ],
    }
    cls = osx.classify_buy_option(
        {
            "action": "BUY_PUT",
            "ticker": "AAPL",
            "qty": 1,
            "strike": 170,
            "expiry": "2026-12-18",
        },
        snap,
    )
    assert cls["mode"] == "close_short"
    assert cls["held_short"] == 1


def test_csp_collateral():
    # strike 100, premium 2 => collateral 9800 per contract
    assert abs(osx.csp_collateral_required(100, 1, 2) - 9800) < 1e-9


def test_options_risk_short_and_long():
    snap = {
        "cash": 20000,
        "total_value": 30000,
        "positions": [
            {
                "ticker": "AAPL",
                "type": "call",
                "qty": 1,
                "strike": 200,
                "expiry": "2099-01-01",
                "avg_cost": 5,
                "current_price": 6,
                "unrealized_pl": 100,
            },
            {
                "ticker": "AAPL",
                "type": "put",
                "qty": -1,
                "strike": 170,
                "expiry": "2099-01-01",
                "avg_cost": 3,
                "current_price": 2,
                "unrealized_pl": 100,
            },
        ],
    }
    out = options_risk.build_options_risk(snap)
    assert out["state"] == "ACTIVE"
    assert out["prompt_block"]
    assert out["long_premium"] > 0
    assert out["short_premium"] > 0
    assert out["short_put_collateral"] > 0


def test_execute_covered_call_and_spread(tmp_path, monkeypatch):
    from paper_trader import store as store_mod
    from paper_trader.store import Store
    from paper_trader import strategy

    db = tmp_path / "t.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    store = Store()
    store.update_portfolio(50000.0, 50000.0)
    store.upsert_position("AAPL", "stock", 100, 180.0)

    def fake_opt_price(ticker, expiry, strike, otype):
        table = {
            (200.0, "call"): 3.5,
            (180.0, "call"): 8.0,
            (190.0, "call"): 3.0,
        }
        return table.get((float(strike), otype))

    monkeypatch.setattr(strategy.market, "get_option_price", fake_opt_price)

    snap = strategy._portfolio_snapshot(store)
    st, detail = strategy._execute(
        {
            "action": "SELL_CALL",
            "ticker": "AAPL",
            "qty": 1,
            "strike": 200,
            "expiry": "2026-12-18",
            "reasoning": "cc test",
        },
        snap,
        store,
    )
    assert st == "FILLED", detail
    assert "COVERED_CALL" in detail
    pos = store.open_positions()
    shorts = [p for p in pos if p["type"] == "call" and float(p["qty"]) < 0]
    assert len(shorts) == 1

    snap2 = strategy._portfolio_snapshot(store)
    st2, detail2 = strategy._execute(
        {
            "action": "BUY_CALL_SPREAD",
            "ticker": "AAPL",
            "qty": 1,
            "expiry": "2026-12-18",
            "long_strike": 180,
            "short_strike": 190,
            "reasoning": "spread test",
        },
        snap2,
        store,
    )
    assert st2 == "FILLED", detail2
    assert "BUY_CALL_SPREAD" in detail2
    pos2 = store.open_positions()
    legs = [p for p in pos2 if p["type"] == "call" and float(p.get("strike") or 0) in (180.0, 190.0)]
    # one long 180 and one short 190 (plus prior short 200)
    qs = {(float(p["strike"]), float(p["qty"])) for p in legs}
    assert (180.0, 1.0) in qs
    assert (190.0, -1.0) in qs
