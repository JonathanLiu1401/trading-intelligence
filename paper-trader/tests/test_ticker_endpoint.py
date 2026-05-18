"""Flask wiring tests for the per-ticker drill-down + opportunity radar.

These assert the endpoints actually call the analytics SSOT and shape JSON
correctly (the pure arithmetic is pinned in test_ticker_dossier /
test_watchlist_opportunities). The store + signals layer is faked so the
test is offline and deterministic.

Key regression guard: /api/ticker/<sym> is deliberately NOT @swr_cached
because that decorator keys on the query string only — two different path
tickers would otherwise collide. ``test_two_tickers_do_not_collide`` fails
if someone "helpfully" adds @swr_cached back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, signals


class _FakeStore:
    def __init__(self, positions=None, trades=None, decisions=None):
        self._p, self._t, self._d = positions or [], trades or [], decisions or []

    def open_positions(self):
        return list(self._p)

    def recent_trades(self, limit=50):
        return list(self._t)[:limit]

    def recent_decisions(self, limit=20):
        return list(self._d)[:limit]


@pytest.fixture
def client():
    return dashboard.app.test_client()


@pytest.fixture
def wired(monkeypatch):
    """Wire a fake store + canned news into the endpoints."""
    store = _FakeStore(
        positions=[{"ticker": "MU", "type": "stock", "qty": 10,
                    "avg_cost": 5.0, "current_price": 6.0,
                    "unrealized_pl": 10.0}],
        trades=[
            {"id": 1, "timestamp": "2026-05-01T10:00:00+00:00", "ticker": "MU",
             "action": "BUY", "qty": 10, "price": 5.0, "value": 50.0,
             "reason": "entry"},
            {"id": 2, "timestamp": "2026-05-03T10:00:00+00:00", "ticker": "MU",
             "action": "SELL", "qty": 10, "price": 7.0, "value": 70.0,
             "reason": "take profit"},
        ],
        decisions=[{"timestamp": "2026-05-03T10:00:00+00:00",
                    "action_taken": "SELL MU → FILLED",
                    "reasoning": "lock the DRAM gain"}],
    )
    monkeypatch.setattr(dashboard, "get_store", lambda: store)

    arts = [
        {"id": "a1", "title": "Micron DRAM surge", "source": "Reuters",
         "ai_score": 9.0, "urgency": 1, "first_seen": "2026-05-18T01:00:00+00:00",
         "url": "http://x/mu", "summary": "body", "tickers": ["MU"]},
        {"id": "a2", "title": "Nvidia keynote", "source": "CNBC",
         "ai_score": 8.0, "urgency": 0, "first_seen": "2026-05-18T02:00:00+00:00",
         "url": "http://x/nv", "summary": "body", "tickers": ["NVDA"]},
    ]
    monkeypatch.setattr(signals, "get_top_signals",
                        lambda n=20, hours=2, min_score=4.0: list(arts))
    monkeypatch.setattr(signals, "get_ticker_sentiment",
                        lambda t, hours=4: {"ticker": t, "avg_score": 9.0,
                                            "max_score": 9.0, "n": 1,
                                            "urgent": 1})
    return store


class TestTickerApi:
    def test_dossier_shape_and_values(self, client, wired):
        j = client.get("/api/ticker/mu").get_json()
        assert j["symbol"] == "MU"
        assert j["held"] is True
        assert j["position"]["unrealized_pl_total"] == 10.0
        assert j["realized"]["n_round_trips"] == 1
        assert j["realized"]["total_pnl_usd"] == 20.0      # 70 - 50
        assert [d["verb"] for d in j["decisions"]] == ["SELL"]
        assert [a["source"] for a in j["news"]["articles"]] == ["Reuters"]
        assert j["news"]["sentiment"]["max_score"] == 9.0
        assert j["has_coverage"] is True

    def test_unknown_ticker_is_clean_not_error(self, client, wired):
        j = client.get("/api/ticker/ZZZZ").get_json()
        assert j["symbol"] == "ZZZZ"
        assert j["held"] is False
        assert j["has_coverage"] is False
        assert j["news"]["articles"] == []

    def test_two_tickers_do_not_collide(self, client, wired):
        """Regression: a path-keyed SWR cache would serve MU for NVDA."""
        mu = client.get("/api/ticker/MU").get_json()
        nv = client.get("/api/ticker/NVDA").get_json()
        assert mu["symbol"] == "MU" and mu["held"] is True
        assert nv["symbol"] == "NVDA" and nv["held"] is False


class TestTickerPage:
    def test_page_renders_html_with_symbol(self, client):
        r = client.get("/ticker/mu")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "text/html" in r.headers["Content-Type"]
        assert '"MU"' in body          # injected via {{ symbol|tojson }}
        assert "/api/ticker/" in body  # page fetches the dossier client-side


class TestWatchlistOpportunitiesApi:
    def test_surfaces_unheld_news_heat(self, client, wired):
        # MU is held (fake store) + only MU/NVDA have articles; MU must be
        # excluded, NVDA surfaced (NVDA is in the real WATCHLIST).
        j = client.get("/api/watchlist-opportunities").get_json()
        tickers = [o["ticker"] for o in j["opportunities"]]
        assert "MU" not in tickers
        assert "NVDA" in tickers
        nv = next(o for o in j["opportunities"] if o["ticker"] == "NVDA")
        assert nv["n_articles"] == 1
        assert nv["max_score"] == 8.0
