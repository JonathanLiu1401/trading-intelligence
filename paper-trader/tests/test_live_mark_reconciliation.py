from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.live_mark_reconciliation import (
    build_live_mark_reconciliation,
    held_stock_tickers,
)


def test_flat_cached_mark_diverges_from_independent_quote():
    positions = [
        {
            "ticker": "SOXX",
            "type": "stock",
            "qty": 10,
            "avg_cost": 570.84,
            "current_price": 570.84,
            "market_value": 5708.40,
            "unrealized_pl": 0.0,
            "stale_mark": False,
        },
        {
            "ticker": "NVDA",
            "type": "stock",
            "qty": 20,
            "avg_cost": 208.61,
            "current_price": 208.61,
            "market_value": 4172.20,
            "unrealized_pl": 0.0,
            "stale_mark": False,
        },
    ]

    out = build_live_mark_reconciliation(
        positions,
        {"SOXX": 571.45, "NVDA": 208.64},
    )

    assert out["verdict"] == "DIVERGED"
    assert out["book_delta_usd"] == pytest.approx(6.7)
    assert out["worst"]["ticker"] == "SOXX"
    assert out["positions"][0]["expected_unrealized_pl"] == pytest.approx(6.1)
    assert "P/L display may be stale" in out["headline"]


def test_clean_when_marks_match_independent_quotes_within_tolerance():
    positions = [
        {
            "ticker": "NVDA",
            "type": "stock",
            "qty": 1,
            "avg_cost": 100.0,
            "current_price": 100.00,
            "unrealized_pl": 0.0,
        }
    ]

    out = build_live_mark_reconciliation(positions, {"NVDA": 100.01})

    assert out["verdict"] == "CLEAN"
    assert out["book_delta_usd"] == pytest.approx(0.01)
    assert out["n_divergent"] == 0


def test_degraded_when_some_independent_quotes_missing():
    positions = [
        {"ticker": "NVDA", "type": "stock", "qty": 1, "current_price": 100},
        {"ticker": "SOXX", "type": "stock", "qty": 1, "current_price": 50},
    ]

    out = build_live_mark_reconciliation(positions, {"NVDA": 100.0})

    assert out["verdict"] == "DEGRADED"
    assert out["n_missing_quotes"] == 1


def test_no_quotes_when_all_independent_reads_fail():
    positions = [
        {"ticker": "NVDA", "type": "stock", "qty": 1, "current_price": 100},
    ]

    out = build_live_mark_reconciliation(positions, {"NVDA": None})

    assert out["verdict"] == "NO_QUOTES"
    assert out["n_missing_quotes"] == 1


def test_held_stock_tickers_skips_options_and_pseudo_tickers():
    positions = [
        {"ticker": "NVDA", "type": "stock"},
        {"ticker": "NVDA", "type": "call"},
        {"ticker": "CASH", "type": "stock"},
        {"ticker": "SOXX", "type": "stock"},
    ]

    assert held_stock_tickers(positions) == ["NVDA", "SOXX"]
