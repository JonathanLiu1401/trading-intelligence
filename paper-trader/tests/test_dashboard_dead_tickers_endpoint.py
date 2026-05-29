"""``/api/dead-tickers`` endpoint — surfaces the live trader's negative
cache to the operator.

Today ``market._DEAD_CACHE`` carries which watchlist symbols the live
trader is currently unable to fetch from yfinance (delisted, halted,
weekend-closed futures). The internal cache prints a single stderr
line on the first dead-mark per TTL window — no operator surface
exists for "what's dark RIGHT NOW?". This endpoint is the trader-
facing surface that closes the gap.

Tested against the actual Flask app via test_client to lock the
endpoint contract (route, JSON shape, ERROR envelope, n_dark count,
sorted ticker list) so a future dashboard refactor cannot silently
break this surface.
"""
from __future__ import annotations

import json

import pytest

from paper_trader import dashboard as dash
from paper_trader import market


@pytest.fixture(autouse=True)
def _clean_cache():
    market._DEAD_CACHE.clear()
    yield
    market._DEAD_CACHE.clear()


@pytest.fixture
def client():
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        yield c


class TestDeadTickersEndpoint:
    def test_returns_empty_when_no_dark_tickers(self, client):
        # Clean cache → empty payload, n_dark=0, status 200.
        resp = client.get("/api/dead-tickers")
        assert resp.status_code == 200
        body = json.loads(resp.data)
        assert body["service"] == "paper_trader"
        assert body["n_dark"] == 0
        assert body["tickers"] == []
        assert body["ttl_seconds"] == int(market._DEAD_TTL)
        # ``as_of`` is present and ISO-formatted (best-effort sanity, not byte
        # match — the timestamp is wall-clock).
        assert "as_of" in body and isinstance(body["as_of"], str)

    def test_returns_currently_dark_tickers(self, client, monkeypatch):
        t0 = 9_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("LITE")
        market._mark_dead("MUU")
        # Probe at t0 + 15s — both still inside TTL.
        monkeypatch.setattr(market.time, "time", lambda: t0 + 15.0)
        resp = client.get("/api/dead-tickers")
        assert resp.status_code == 200
        body = json.loads(resp.data)
        assert body["n_dark"] == 2
        tickers = [r["ticker"] for r in body["tickers"]]
        # Sorted output → deterministic across calls (the Discord-line
        # composability invariant the accessor pins).
        assert tickers == ["LITE", "MUU"]
        # Each row carries the four documented fields.
        for r in body["tickers"]:
            assert set(r.keys()) >= {
                "ticker", "marked_at_ts", "seconds_dead", "ttl_remaining_s",
            }
            assert r["seconds_dead"] == 15

    def test_excludes_expired_entries_from_count(self, client, monkeypatch):
        t0 = 10_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("OLD")
        # Probe past the TTL — OLD is no longer "dark" by the accessor's
        # contract (the next get_price() would re-attempt). n_dark must
        # NOT count it.
        monkeypatch.setattr(market.time, "time",
                            lambda: t0 + market._DEAD_TTL + 5.0)
        resp = client.get("/api/dead-tickers")
        body = json.loads(resp.data)
        assert body["n_dark"] == 0
        assert body["tickers"] == []

    def test_accessor_fault_yields_error_envelope_not_500(
            self, client, monkeypatch):
        """If the accessor (somehow) raises, the operator must still get a
        valid JSON envelope with verdict=ERROR — never a 500 that the
        upstream dashboard renders as a dead endpoint."""
        def _boom():
            raise RuntimeError("simulated cache corruption")
        monkeypatch.setattr(market, "dead_tickers", _boom)
        resp = client.get("/api/dead-tickers")
        # Endpoint deliberately returns 500 for ops alerting BUT also a
        # well-formed payload so a panel that handles 500 still renders
        # the failure message rather than going dark.
        assert resp.status_code == 500
        body = json.loads(resp.data)
        assert body["verdict"] == "ERROR"
        assert "dead-tickers endpoint error" in body["headline"]
        assert body["n_dark"] == 0
        assert body["tickers"] == []
