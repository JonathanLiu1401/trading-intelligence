from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.portfolio_construction import (  # noqa: E402
    build_portfolio_construction,
)
from paper_trader.analytics.sector_exposure import classify  # noqa: E402
from paper_trader.analytics.stress_scenarios import (  # noqa: E402
    _LEVERAGE_BETA,
)


_NOW = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _snap():
    return {
        "cash": 120.0,
        "total_value": 10000.0,
        "positions": [
            {
                "ticker": "SOXX", "type": "stock", "qty": 10,
                "avg_cost": 570.84, "current_price": 571.45,
                "market_value": 5714.5,
            },
            {
                "ticker": "NVDA", "type": "stock", "qty": 20,
                "avg_cost": 208.61, "current_price": 208.64,
                "market_value": 4172.8,
            },
        ],
    }


def _signals():
    return [
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
    ]


def _quant():
    return {
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


def _prices():
    return {"NVDA": 208.64, "SOXX": 571.45, "SPY": 740.0, "QQQ": 520.0}


class TestPortfolioConstructionBuilder:
    def test_medium_risk_builds_all_requested_fields_and_prompt(self):
        out = build_portfolio_construction(
            _snap(), _signals(), _quant(), _prices(), classify, _LEVERAGE_BETA,
            risk_tolerance="medium", time_horizon="1-3 years", now=_NOW,
        )

        assert out["state"] in {"OK", "REBALANCE_AWARE", "CONSTRAINED"}
        assert out["risk_tolerance"] == "medium"
        assert out["cash_target_pct"] >= 5.0
        assert out["portfolio_expected_return_pct"] is not None
        assert out["portfolio_volatility_pct"] is not None
        assert out["portfolio_drawdown_pct"] < 0
        assert "PORTFOLIO CONSTRUCTION" in out["prompt_block"]

        by_ticker = {a["ticker"]: a for a in out["allocations"]}
        assert {"NVDA", "SOXX", "SPY", "QQQ"} <= set(by_ticker)
        for row in by_ticker.values():
            assert row["target_allocation_pct"] is not None
            assert row["expected_return_pct"] is not None
            assert row["volatility_pct"] is not None
            assert row["max_drawdown_pct"] is not None
            assert row["why_included"]

        # Current book is ~99% semis; medium target must cap the target semis
        # cluster at the configured 65% cap and leave room for broad exposure.
        sectors = out["diversification"]["target_sector_pct"]
        assert sectors["semis"] <= out["constraints"]["max_sector_pct"]
        assert sectors["broad"] > 0

    def test_low_risk_short_horizon_is_more_conservative(self):
        out = build_portfolio_construction(
            _snap(), _signals(), _quant(), _prices(), classify, _LEVERAGE_BETA,
            risk_tolerance="low", time_horizon="6 months", now=_NOW,
        )

        assert out["risk_tolerance"] == "low"
        assert out["constraints"]["cash_buffer_pct"] == pytest.approx(25.0)
        assert out["constraints"]["max_single_name_pct"] == pytest.approx(15.0)
        assert out["constraints"]["max_sector_pct"] == pytest.approx(30.0)
        for row in out["allocations"]:
            assert row["target_allocation_pct"] <= 15.0 + 1e-6
        assert out["cash_target_pct"] >= 20.0

    def test_sector_cap_is_not_overfilled_by_many_same_sector_names(self):
        semis = ["NVDA", "AMD", "MU", "MRVL", "KLAC", "SOXX"]
        signals = [
            {"id": i, "ai_score": 9.0, "urgency": 1,
             "title": f"{tk} catalyst", "tickers": [tk]}
            for i, tk in enumerate(semis, start=1)
        ]
        quant = {
            tk: {"MACD": "bullish", "mom_20d": 8.0, "RSI": 55.0}
            for tk in semis
        }
        prices = {tk: 100.0 for tk in semis}
        out = build_portfolio_construction(
            {"cash": 10000.0, "total_value": 10000.0, "positions": []},
            signals, quant, prices, classify, _LEVERAGE_BETA,
            risk_tolerance="medium", time_horizon="1-3 years", now=_NOW,
        )

        sectors = out["diversification"]["target_sector_pct"]
        assert sectors["semis"] <= out["constraints"]["max_sector_pct"]
        assert out["cash_target_pct"] > out["constraints"]["cash_buffer_pct"]
        assert out["state"] == "CONSTRAINED"

    def test_no_value_returns_no_data(self):
        out = build_portfolio_construction(
            {"cash": 0, "total_value": 0, "positions": []},
            [], {}, {}, classify, _LEVERAGE_BETA, now=_NOW,
        )
        assert out["state"] == "NO_DATA"
        assert out["allocations"] == []
        assert out["prompt_block"] is None
