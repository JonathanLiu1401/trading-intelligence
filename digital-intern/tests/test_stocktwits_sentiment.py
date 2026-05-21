"""Tests for stocktwits_sentiment collector (agent5 review of 61ec87e)."""
from unittest.mock import MagicMock, patch

import pytest

from collectors import stocktwits_sentiment as ss


def _msg(basic, body="x"):
    return {"entities": {"sentiment": {"basic": basic}}, "body": body}


def test_fetch_sentiment_extreme_bullish():
    msgs = [_msg("Bullish")] * 8 + [_msg("Bearish")] * 1 + [_msg(None)] * 1
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"messages": msgs}
    with patch.object(ss.requests, "get", return_value=resp):
        s = ss._fetch_sentiment("NVDA")
    assert s["total"] == 10
    assert s["bull"] == 8 and s["bear"] == 1 and s["neutral"] == 1
    assert s["ratio"] == pytest.approx(8 / 9)


def test_fetch_sentiment_handles_empty_messages():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"messages": []}
    with patch.object(ss.requests, "get", return_value=resp):
        assert ss._fetch_sentiment("AAPL") is None


def test_fetch_sentiment_handles_429():
    resp = MagicMock(status_code=429)
    with patch.object(ss.requests, "get", return_value=resp):
        assert ss._fetch_sentiment("AAPL") is None


def test_fetch_sentiment_no_bull_no_bear_ratio_is_half():
    msgs = [_msg(None)] * 5
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"messages": msgs}
    with patch.object(ss.requests, "get", return_value=resp):
        s = ss._fetch_sentiment("AAPL")
    assert s["ratio"] == 0.5


def test_article_id_is_stable_and_unique():
    a = ss._article_id("NVDA", "Bullish", "2026-05-20T19")
    b = ss._article_id("NVDA", "Bullish", "2026-05-20T19")
    c = ss._article_id("NVDA", "Bearish", "2026-05-20T19")
    assert a == b
    assert a != c


def test_load_tickers_dedupes_and_filters():
    pf = {
        "positions": [{"ticker": "nvda"}, {"ticker": "NVDA"}, {"ticker": "BRK.B"}],
        "options": [{"underlying": "AAPL"}],
        "sector_watchlist": ["^VIX", "MSFT"],
    }
    wl = {"memory_core": ["MU"], "etfs": ["SOXX"], "korean": []}
    with patch("builtins.open"), \
         patch.object(ss.json, "load", side_effect=[pf, wl]):
        tickers = ss._load_tickers()
    assert "NVDA" in tickers and "AAPL" in tickers and "MSFT" in tickers
    assert "MU" in tickers and "SOXX" in tickers
    assert "BRK.B" not in tickers
    assert "^VIX" not in tickers
    assert tickers.count("NVDA") == 1
