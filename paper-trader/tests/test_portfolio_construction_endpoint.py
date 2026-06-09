from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d  # noqa: E402
from paper_trader import market as market_mod  # noqa: E402
from paper_trader import signals as signals_mod  # noqa: E402
from paper_trader import store as store_mod  # noqa: E402
from paper_trader import strategy as strategy_mod  # noqa: E402
from paper_trader.store import Store  # noqa: E402


@pytest.fixture
def client_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)

    prices = {"SOXX": 571.45, "NVDA": 208.64, "SPY": 740.0, "QQQ": 520.0}
    monkeypatch.setattr(strategy_mod, "WATCHLIST",
                        ["SOXX", "NVDA", "SPY", "QQQ"])
    monkeypatch.setattr(strategy_mod, "QUANT_TICKERS_LIVE",
                        ["SPY", "QQQ", "NVDA"])

    def _get_price(ticker):
        return prices.get(ticker)

    def _get_prices(tickers):
        return {t: prices.get(t) for t in tickers}

    def _quant(tickers):
        data = {
            "NVDA": {
                "RSI": 46.0, "MACD": "bullish", "MA_cross": "bullish",
                "mom_5d": -3.0, "mom_20d": 6.0,
            },
            "SOXX": {
                "RSI": 55.0, "MACD": "bearish", "mom_20d": 3.0,
            },
            "SPY": {
                "RSI": 58.0, "MACD": "bullish", "mom_20d": 2.0,
            },
            "QQQ": {
                "RSI": 57.0, "MACD": "bullish", "mom_20d": 4.0,
            },
        }
        return {t: data[t] for t in tickers if t in data}

    monkeypatch.setattr(market_mod, "get_price", _get_price)
    monkeypatch.setattr(market_mod, "get_prices", _get_prices)
    monkeypatch.setattr(strategy_mod, "get_quant_signals_live", _quant)
    monkeypatch.setattr(signals_mod, "get_top_signals", lambda **_kw: [
        {
            "id": 1,
            "ai_score": 9.5,
            "urgency": 1,
            "title": "NVDA AI demand strengthens",
            "tickers": ["NVDA"],
        },
        {
            "id": 2,
            "ai_score": 7.0,
            "urgency": 0,
            "title": "QQQ breadth improves",
            "tickers": ["QQQ"],
        },
    ])

    s = Store()
    s.record_trade("SOXX", "BUY", 10, 570.84, reason="seed")
    s.upsert_position("SOXX", "stock", 10, 570.84)
    s.record_trade("NVDA", "BUY", 20, 208.61, reason="seed")
    s.upsert_position("NVDA", "stock", 20, 208.61)
    s.update_portfolio(cash=119.40, total_value=10000.0, positions=[])

    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        s.close()


def test_endpoint_returns_portfolio_construction_target(client_store):
    client, _store = client_store
    r = client.get(
        "/api/portfolio-construction?risk_tolerance=low&time_horizon=6m"
    )
    assert r.status_code == 200
    j = r.get_json()

    assert j["state"] in {"OK", "REBALANCE_AWARE", "CONSTRAINED"}
    assert j["risk_tolerance"] == "low"
    assert j["cash_target_pct"] >= 20.0
    assert j["n_signals_input"] == 2
    assert j["portfolio_expected_return_pct"] is not None
    assert j["portfolio_volatility_pct"] is not None
    assert j["portfolio_drawdown_pct"] < 0
    assert "PORTFOLIO CONSTRUCTION" in j["prompt_block"]

    by_ticker = {a["ticker"]: a for a in j["allocations"]}
    assert {"NVDA", "SOXX", "SPY", "QQQ"} <= set(by_ticker)
    assert by_ticker["NVDA"]["why_included"]
    assert by_ticker["NVDA"]["expected_return_pct"] is not None
    assert by_ticker["NVDA"]["max_drawdown_pct"] is not None
