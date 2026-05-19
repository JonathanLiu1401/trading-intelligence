"""Tests for the market_movers collector.

Pins two recently-fixed properties:

1. ``_fetch_screener`` now logs at WARNING when the upstream request raises
   (formerly a bare ``except: return []`` made every network outage
   indistinguishable from an empty result set). The contract: the function
   STILL returns an empty list (the worker downstream handles that gracefully),
   but it MUST log so source_health diagnostics surface the outage.

2. ``collect_market_movers`` filters out small moves below MIN_GAINER_PCT
   and MIN_LOSER_PCT — a defensive threshold the source helper requires.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from collectors import market_movers


def _quote(symbol="NVDA", name="NVIDIA Corp", price=100.0, chg_pct=5.0,
           volume=5_000_000, avg_vol=10_000_000):
    return {
        "symbol": symbol,
        "shortName": name,
        "regularMarketPrice": price,
        "regularMarketChangePercent": chg_pct,
        "regularMarketVolume": volume,
        "averageDailyVolume3Month": avg_vol,
    }


class TestFetchScreenerLogging:
    """The silent-exception fix: a network error MUST surface in the log
    even though the return value is still an empty list (the worker
    treats that as 'no new articles this cycle')."""

    def test_returns_empty_and_logs_on_network_error(self, monkeypatch, caplog):
        # requests.get raises a generic exception (e.g. timeout / DNS / SSL).
        def boom(*a, **kw):
            raise RuntimeError("simulated network outage")
        monkeypatch.setattr(market_movers.requests, "get", boom)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("day_gainers")

        # Returns an empty list — worker downstream handles this as no data.
        assert out == []
        # A WARNING-level record must be emitted, naming the screener and
        # the exception type so an operator tailing daemon.log can attribute
        # the outage.
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "day_gainers" in messages
        assert "RuntimeError" in messages

    def test_returns_empty_and_logs_on_bad_status(self, monkeypatch, caplog):
        # Yahoo occasionally returns 401/429 — raise_for_status must surface
        # them and the wrapper must log.
        resp = MagicMock()
        # Raising in raise_for_status mirrors requests' real behavior on >=400.
        resp.raise_for_status.side_effect = RuntimeError("HTTP 429")
        monkeypatch.setattr(market_movers.requests, "get", lambda *a, **k: resp)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("most_actives")

        assert out == []
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "most_actives" in messages

    def test_happy_path_returns_quotes_no_warning(self, monkeypatch, caplog):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "finance": {"result": [{"quotes": [_quote(), _quote(symbol="MU")]}]}
        }
        monkeypatch.setattr(market_movers.requests, "get", lambda *a, **k: resp)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("day_gainers")

        assert len(out) == 2
        # On the happy path NO warning should be emitted — the
        # silent-network-error fix must not turn a healthy fetch into log noise.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []


class TestMoverThresholds:
    """Defensive thresholds — sub-3% gainers / -3% losers are filtered out
    so the collector does not flood the briefing on low-volatility days."""

    def _setup_screener(self, monkeypatch, scr_id_to_quotes, tmp_path):
        # Redirect the seen_articles.db to a tmp file so the test doesn't
        # write into the repo's data/ dir.
        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")

        def fake_fetch(scr_id):
            return scr_id_to_quotes.get(scr_id, [])

        monkeypatch.setattr(market_movers, "_fetch_screener", fake_fetch)

    def test_small_gainer_below_threshold_is_filtered(self, monkeypatch, tmp_path):
        # 2.0% < MIN_GAINER_PCT=3.0 — must be dropped.
        small = _quote(chg_pct=2.0)
        big = _quote(symbol="MU", chg_pct=8.5)
        self._setup_screener(monkeypatch, {"day_gainers": [small, big]}, tmp_path)

        out = market_movers.collect_market_movers()

        # Only the >3% mover (MU) makes it through.
        symbols = [a["symbol"] for a in out]
        assert "MU" in symbols
        assert "NVDA" not in symbols

    def test_small_loser_above_threshold_is_filtered(self, monkeypatch, tmp_path):
        # -1.5% > MIN_LOSER_PCT=-3.0 — must be dropped.
        small_loss = _quote(symbol="AAPL", chg_pct=-1.5)
        big_loss = _quote(symbol="TSLA", chg_pct=-7.2)
        self._setup_screener(monkeypatch, {"day_losers": [small_loss, big_loss]},
                             tmp_path)

        out = market_movers.collect_market_movers()

        symbols = [a["symbol"] for a in out]
        assert "TSLA" in symbols
        assert "AAPL" not in symbols

    def test_skips_quotes_with_no_symbol_or_price(self, monkeypatch, tmp_path):
        bad1 = {"symbol": "", "regularMarketPrice": 10.0,
                "regularMarketChangePercent": 5.0}
        bad2 = {"symbol": "GOOG", "regularMarketPrice": None,
                "regularMarketChangePercent": 5.0}
        good = _quote(symbol="MSFT", chg_pct=4.0)
        self._setup_screener(monkeypatch,
                             {"day_gainers": [bad1, bad2, good]}, tmp_path)

        out = market_movers.collect_market_movers()
        symbols = [a["symbol"] for a in out]
        assert symbols == ["MSFT"]


class TestDedup:
    """Re-running the collector with the same upstream payload must NOT
    produce duplicate articles — seen_articles.db is the cross-cycle gate."""

    def test_second_run_returns_no_new_articles(self, monkeypatch, tmp_path):
        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")
        quotes = [_quote(symbol="NVDA", chg_pct=6.0)]
        monkeypatch.setattr(market_movers, "_fetch_screener",
                            lambda scr_id: quotes if scr_id == "day_gainers" else [])

        first = market_movers.collect_market_movers()
        second = market_movers.collect_market_movers()

        assert len(first) == 1
        # Second run sees the exact same article and must skip it via the
        # seen_articles dedup index.
        assert second == []
