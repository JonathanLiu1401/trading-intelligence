"""Behaviour lock for paper_trader.analytics.leverage_exposure and the
/api/leverage-exposure Flask endpoint.

The pure function ``build_leverage_exposure`` classifies the current book
AND the scorer opportunity slate by leverage factor (+/- 2x, +/- 3x).
These tests assert the EXACT factor bucketing and the verdict ladder —
not just "the call returned 200". They would fail if:

  * a 3x long ETF were ever silently classified as 1x or as inverse,
  * inverse-leveraged dollars leaked into the long-leveraged bucket,
  * the verdict said ALIGNED when the book was overweight in a bear tape,
  * the verdict said UNDER_LEV when the slate had zero leveraged names,
  * the factor map drifted from the strategy._LEVERAGED_ETFS_LIVE
    membership set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.leverage_exposure import (
    LEVERAGE_FACTOR,
    classify,
    build_leverage_exposure,
)


# ───────────── pure-function classifier ─────────────────────────────────

def test_classify_known_long_3x():
    assert classify("SOXL") == {"factor": 3, "direction": "long"}
    assert classify("TQQQ") == {"factor": 3, "direction": "long"}
    assert classify("NAIL") == {"factor": 3, "direction": "long"}


def test_classify_known_inverse_3x():
    assert classify("SOXS") == {"factor": -3, "direction": "inverse"}
    assert classify("SQQQ") == {"factor": -3, "direction": "inverse"}
    assert classify("FNGD") == {"factor": -3, "direction": "inverse"}


def test_classify_known_long_2x():
    assert classify("NVDU") == {"factor": 2, "direction": "long"}
    assert classify("TSLL") == {"factor": 2, "direction": "long"}
    assert classify("AMZU") == {"factor": 2, "direction": "long"}


def test_classify_unlev_default():
    # A vanilla stock falls through to 1x long.
    assert classify("NVDA") == {"factor": 1, "direction": "long"}
    assert classify("AAPL") == {"factor": 1, "direction": "long"}


def test_classify_unknown_ticker_is_unlev():
    assert classify("ZZZZ_NOT_A_REAL_TICKER") == {"factor": 1, "direction": "long"}


def test_classify_case_insensitive():
    assert classify("soxl") == {"factor": 3, "direction": "long"}
    assert classify("Sqqq") == {"factor": -3, "direction": "inverse"}


def test_classify_garbage_input_degrades_to_unlev():
    # Non-string input must not raise.
    assert classify(None) == {"factor": 1, "direction": "long"}
    assert classify(123) == {"factor": 1, "direction": "long"}


def test_factor_map_mirrors_leveraged_etfs_live():
    """The factor map must cover EVERY member of strategy._LEVERAGED_ETFS_LIVE
    so a planner reading the membership set doesn't see an unmapped ticker
    and silently treat it as 1x."""
    from paper_trader.strategy import _LEVERAGED_ETFS_LIVE as live_set
    unmapped = sorted(t for t in live_set if t not in LEVERAGE_FACTOR)
    assert unmapped == [], f"LEVERAGE_FACTOR is missing: {unmapped}"


# ───────────── book classification ──────────────────────────────────────

def _pos(ticker, qty, price, type_="stock"):
    return {"ticker": ticker, "type": type_, "qty": qty,
            "current_price": price, "market_value": qty * price}


def test_empty_book_zero_lev():
    out = build_leverage_exposure(
        positions=[],
        total_value=10000.0,
        opportunities=[],
        regime="bull",
    )
    assert out["current_book"]["by_direction"]["long_lev_pct"] == 0.0
    assert out["current_book"]["by_direction"]["inverse_lev_pct"] == 0.0
    assert out["current_book"]["by_direction"]["unlev_pct"] == 0.0
    assert out["current_book"]["n_positions"] == 0


def test_book_split_by_factor():
    # 50% in SOXL (+3x), 30% in SOXS (-3x), 20% in NVDA (1x)
    positions = [
        _pos("SOXL", 10, 500.0),   # $5000
        _pos("SOXS", 10, 300.0),   # $3000
        _pos("NVDA", 10, 200.0),   # $2000
    ]
    out = build_leverage_exposure(
        positions=positions,
        total_value=10000.0,
        opportunities=[],
        regime="bull",
    )
    bd = out["current_book"]["by_direction"]
    assert bd["long_lev_pct"] == 50.0
    assert bd["inverse_lev_pct"] == 30.0
    assert bd["unlev_pct"] == 20.0
    by_factor = out["current_book"]["by_factor"]
    assert by_factor["+3"]["tickers"] == ["SOXL"]
    assert by_factor["+3"]["pct"] == 50.0
    assert by_factor["-3"]["tickers"] == ["SOXS"]
    assert by_factor["-3"]["pct"] == 30.0
    assert by_factor["1"]["tickers"] == ["NVDA"]
    assert by_factor["1"]["pct"] == 20.0


def test_book_uses_market_value_when_present():
    # market_value overrides qty*current_price.
    positions = [
        {"ticker": "SOXL", "type": "stock", "qty": 1, "current_price": 1.0,
         "market_value": 1000.0},
    ]
    out = build_leverage_exposure(
        positions=positions, total_value=1000.0,
        opportunities=[], regime="bull",
    )
    assert out["current_book"]["by_factor"]["+3"]["usd"] == 1000.0


# ───────────── slate classification ─────────────────────────────────────

def _opp(t, pred):
    return {"ticker": t, "pred_5d_return_pct": pred, "verdict": "STRONG_HOLD"}


def test_slate_count_and_direction_split():
    # 4 opps: 2 long-lev, 1 inverse-lev, 1 unlev.
    slate = [_opp("SOXL", 8.0), _opp("TQQQ", 6.0),
             _opp("SOXS", 17.0), _opp("NVDA", 3.0)]
    out = build_leverage_exposure(
        positions=[], total_value=1000.0,
        opportunities=slate, regime="bull",
    )
    sd = out["opportunity_slate"]["by_direction"]
    assert sd["long_lev_pct"] == 50.0
    assert sd["inverse_lev_pct"] == 25.0
    assert sd["unlev_pct"] == 25.0
    assert out["opportunity_slate"]["n_opportunities"] == 4
    # Blended pred is equal-weighted across all 4: mean(8,6,17,3) = 8.5
    assert out["opportunity_slate"]["blended_pred_5d_return_pct"] == 8.5


def test_slate_empty_blends_to_none():
    out = build_leverage_exposure(
        positions=[], total_value=1000.0,
        opportunities=[], regime="bull",
    )
    assert out["opportunity_slate"]["n_opportunities"] == 0
    assert out["opportunity_slate"]["blended_pred_5d_return_pct"] is None


# ───────────── verdict ladder ───────────────────────────────────────────

def test_verdict_no_slate_when_slate_empty():
    out = build_leverage_exposure(
        positions=[], total_value=1000.0,
        opportunities=[], regime="bull",
    )
    assert out["verdict"] == "NO_SLATE"


def test_verdict_under_lev_in_bull_with_lev_slate():
    # All cash book + heavily leveraged slate in bull tape → UNDER_LEV.
    slate = [_opp("SOXL", 8), _opp("TQQQ", 6), _opp("NAIL", 5)]
    out = build_leverage_exposure(
        positions=[], total_value=1000.0,
        opportunities=slate, regime="bull",
    )
    assert out["verdict"] == "UNDER_LEV"


def test_verdict_over_lev_in_bear():
    # 50% leveraged book in a bear tape → OVER_LEV.
    positions = [_pos("SOXL", 10, 500.0)]
    out = build_leverage_exposure(
        positions=positions, total_value=1000.0,
        opportunities=[_opp("SOXS", 5)], regime="bear",
    )
    assert out["verdict"] == "OVER_LEV"


def test_verdict_aligned_neutral():
    # Modest book lev + modest slate lev in sideways tape → ALIGNED.
    positions = [_pos("SOXL", 1, 100.0), _pos("NVDA", 9, 100.0)]
    out = build_leverage_exposure(
        positions=positions, total_value=1000.0,
        opportunities=[_opp("AAPL", 2)], regime="sideways",
    )
    assert out["verdict"] == "ALIGNED"


# ───────────── Flask endpoint wiring ────────────────────────────────────

@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    """Offline test client — stubs store + scorer-opportunities + spy
    regime so the endpoint runs deterministically."""
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    # Seed one held leveraged position so the book breakdown is non-zero.
    s.record_trade("SOXL", "BUY", 10.0, 500.0)
    s.upsert_position("SOXL", "stock", 10.0, 500.0)
    pid = s.open_positions()[0]["id"]
    s.update_position_marks({pid: (500.0, 0.0)})

    from paper_trader import dashboard
    monkeypatch.setattr(dashboard, "get_store", lambda: s)

    # Stub the inline scorer-opportunities helper so we don't run the
    # actual scorer (and don't depend on a trained model).
    monkeypatch.setattr(dashboard, "_scorer_opportunities_inline", lambda: [
        {"ticker": "TQQQ", "pred_5d_return_pct": 8.0, "verdict": "STRONG_HOLD"},
        {"ticker": "SOXS", "pred_5d_return_pct": 17.0, "verdict": "STRONG_HOLD"},
        {"ticker": "NVDA", "pred_5d_return_pct": 3.0, "verdict": "STRONG_HOLD"},
    ])
    monkeypatch.setattr(dashboard, "_spy_regime_label", lambda: ("bull", 5.0))

    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client()


def test_endpoint_returns_breakdown_and_verdict(stub_client):
    r = stub_client.get("/api/leverage-exposure")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("as_of", "verdict", "headline", "regime",
                "current_book", "opportunity_slate", "thresholds"):
        assert key in body, f"missing {key}"
    # The seeded SOXL position must show in the +3 factor bucket.
    assert "SOXL" in body["current_book"]["by_factor"]["+3"]["tickers"]
    # The slate must include +3 (TQQQ), -3 (SOXS), and 1 (NVDA).
    assert "+3" in body["opportunity_slate"]["by_factor"]
    assert "-3" in body["opportunity_slate"]["by_factor"]
    assert "1" in body["opportunity_slate"]["by_factor"]
    assert body["regime"] == "bull"
