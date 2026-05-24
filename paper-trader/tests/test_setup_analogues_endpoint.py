"""End-to-end Flask-client tests for /api/setup-analogues — the empirical
conditional-return-distribution panel.

The pure builder is locked by tests/test_setup_analogues.py (43 unit tests).
This file pins the dashboard contract: the route exists, is a faithful thin
wrapper over ``build_setup_analogues``, never raises into a panel, and
carries every key the operator / chat sub-fetch would render.

Convention mirrors tests/test_baseline_compare_endpoint.py — real Flask app,
real module math, deterministic offline outcomes; no :8090 bind, no live DB
read, no yfinance hit.  ``get_quant_signals_live`` is monkeypatched so the
test runs offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d


def _synthetic_outcomes(n: int = 200) -> list[dict]:
    """Two cell-distinct slices the matcher can pick apart:
      - 100 BUYs in (mid_high, strong, sideways) all +4% ⇒ STRONG_EDGE.
      - 100 BUYs in (oversold, deep_neg, bear) all -3% ⇒ STRONG_HEADWIND.

    The endpoint queries the *current* ticker's quant signals (monkeypatched
    below) and should hit the first slice cleanly.
    """
    rows = []
    for i in range(n // 2):
        rows.append({
            "run_id": 1000 + i, "sim_date": "2025-06-01",
            "ticker": "NVDA", "action": "BUY",
            "ml_score": 1.0, "rsi": 60.0, "macd": 0.1,
            "mom5": 2.0, "mom20": 12.0, "regime_mult": 1.0,
            "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": 4.0,
        })
    for i in range(n // 2):
        rows.append({
            "run_id": 2000 + i, "sim_date": "2025-08-01",
            "ticker": "SOXL", "action": "BUY",
            "ml_score": -1.0, "rsi": 25.0, "macd": -0.1,
            "mom5": -2.0, "mom20": -15.0, "regime_mult": 0.7,
            "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": -3.0,
        })
    return rows


@pytest.fixture
def client(monkeypatch):
    outcomes = _synthetic_outcomes()
    monkeypatch.setattr(d, "_load_decision_outcomes",
                        lambda *a, **k: outcomes)

    # Stub out the live quant fetch — yfinance must not be hit in tests.
    # The current setup mirrors the FIRST slice (mid_high RSI, strong mom20).
    def _stub_quant(tickers):
        out = {}
        for t in tickers:
            if t == "SPY":
                out[t] = {"mom_5d": 0.0}      # regime ⇒ 1.0 (sideways)
            else:
                out[t] = {"rsi": 60.0, "mom_20d": 12.0,
                          "macd_signal": 0.1, "vol_ratio": 1.0,
                          "bb_position": 0.0}
        return out

    import paper_trader.strategy as strat
    monkeypatch.setattr(strat, "get_quant_signals_live", _stub_quant)

    # Force an empty held set so the endpoint falls back to WATCHLIST[0].
    class _EmptyStore:
        def open_positions(self):
            return []
    monkeypatch.setattr(d, "get_store", lambda: _EmptyStore())

    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        yield c, outcomes


_REQUIRED_KEYS = (
    "as_of", "ticker", "action", "current_features", "current_buckets",
    "min_matches", "n_outcomes", "n_action_only_matches", "n_matches",
    "stats", "trader_median_pct", "win_rate", "verdict", "headline",
)


def test_route_exists_and_returns_required_keys(client):
    c, _ = client
    r = c.get("/api/setup-analogues")
    assert r.status_code == 200
    j = r.get_json()
    assert "error" not in j, j
    for k in _REQUIRED_KEYS:
        assert k in j, f"missing {k!r} in {sorted(j)}"


def test_default_setup_lands_in_strong_edge_cell(client):
    """The stubbed quant signals (rsi=60, mom20=12, regime=sideways) match
    the first synthetic slice perfectly — verdict must be STRONG_EDGE."""
    c, _ = client
    j = c.get("/api/setup-analogues").get_json()
    assert j["verdict"] == "STRONG_EDGE"
    assert j["n_matches"] == 100
    assert j["stats"]["p50"] == 4.0
    assert j["win_rate"] == 1.0


def test_ticker_query_param_overrides_default(client):
    c, _ = client
    j = c.get("/api/setup-analogues?ticker=AMD").get_json()
    assert j["ticker"] == "AMD"


def test_action_query_param_routes_to_other_cell(client):
    """A SELL action with no matching SELL rows in the corpus must return
    INSUFFICIENT_DATA — and crucially must NOT degrade to a BUY verdict."""
    c, _ = client
    j = c.get("/api/setup-analogues?action=SELL").get_json()
    assert j["action"] == "SELL"
    assert j["n_action_only_matches"] == 0
    assert j["verdict"] == "INSUFFICIENT_DATA"


def test_min_matches_query_param_honored(client):
    """Bump the floor above n_matches=100 and the panel must collapse to
    INSUFFICIENT_DATA — proves the query param wires through the builder."""
    c, _ = client
    j = c.get("/api/setup-analogues?min_matches=500").get_json()
    assert j["verdict"] == "INSUFFICIENT_DATA"
    assert j["min_matches"] == 500


def test_cors_header_present_for_cross_fetch(client):
    """Digital-intern's chat / the unified dashboard cross-read this; the
    global _cors after_request must stamp it like every sibling endpoint."""
    c, _ = client
    r = c.get("/api/setup-analogues")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_never_raises_on_load_failure(monkeypatch):
    """A read fault must degrade to a verdict-keyed body, never a 500 with a
    bare stack — the card/chat must always find ``verdict`` to render."""
    def _boom(*a, **k):
        raise RuntimeError("decision_outcomes.jsonl unreadable")
    monkeypatch.setattr(d, "_load_decision_outcomes", _boom)
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        r = c.get("/api/setup-analogues")
        j = r.get_json()
        assert "verdict" in j and j["verdict"] == "INSUFFICIENT_DATA"
        assert "error" in j


def test_endpoint_uses_pure_builder_faithfully(client, monkeypatch):
    """The endpoint must call ``build_setup_analogues`` with the exact
    matched feature inputs — no re-derivation of buckets or stats inside the
    Flask handler.  Spy on the builder and compare what flows in."""
    seen = {}
    from paper_trader.analytics import setup_analogues as sa_mod
    real = sa_mod.build_setup_analogues

    def _spy(outcomes, **kw):
        seen.update(kw)
        return real(outcomes, **kw)

    monkeypatch.setattr(sa_mod, "build_setup_analogues", _spy)
    c, _ = client
    c.get("/api/setup-analogues?ticker=NVDA&action=BUY")
    # The endpoint passed our stubbed quant features through unchanged.
    assert seen["ticker"] == "NVDA"
    assert seen["action"] == "BUY"
    assert seen["rsi"] == 60.0
    assert seen["mom20"] == 12.0
    # regime_mult was derived from SPY mom_5d=0 ⇒ 1.0.
    assert abs(seen["regime_mult"] - 1.0) < 0.001
