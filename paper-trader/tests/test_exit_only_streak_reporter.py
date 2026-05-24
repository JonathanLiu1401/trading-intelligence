"""Tests for `reporter._exit_only_streak_line` — Discord surface for the
new `build_exit_only_streak` analytic.

The hourly summary line surfaces only the two actionable verdicts
(DEFENSIVE_TRIM and DEFENSIVE_LIQUIDATION); everything else is silent so
the hourly does not become its own lying green light (the
`_streak_line` suppression precedent).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from paper_trader import reporter


class _FakeStore:
    """Minimal store stub. Only `recent_trades` matters for this line."""
    def __init__(self, recent_trades=None):
        self._recent_trades = recent_trades or []

    def recent_trades(self, limit=50):
        return list(self._recent_trades[:limit])


def _t(action: str, ticker: str = "NVDA", ts: str | None = None) -> dict:
    """Single trade row in store-newest-first shape."""
    return {
        "action": action,
        "ticker": ticker,
        "timestamp": ts or "2026-05-23T12:00:00+00:00",
        "qty": 1.0, "price": 100.0, "value": 100.0,
        "option_type": None, "strike": None, "expiry": None,
    }


class TestExitOnlyStreakLineSuppression:
    """A book whose newest fill is an entry (or whose tail is below the
    trim floor) must produce NO Discord line."""

    def test_empty_book_returns_empty(self):
        store = _FakeStore(recent_trades=[])
        assert reporter._exit_only_streak_line(store) == ""

    def test_most_recent_is_entry_returns_empty(self):
        # Store hands back newest-first. Newest = BUY → silent.
        store = _FakeStore(recent_trades=[
            _t("BUY", "NVDA"),
            _t("SELL", "MU"),
        ])
        assert reporter._exit_only_streak_line(store) == ""

    def test_two_trailing_exits_returns_empty(self):
        # 2 SELLs trailing — below DEFENSIVE_TRIM_MIN (=3). Silent.
        store = _FakeStore(recent_trades=[
            _t("SELL", "MU"),  # newest (store-newest-first)
            _t("SELL", "AMD"),
            _t("BUY", "NVDA"),  # oldest
        ])
        assert reporter._exit_only_streak_line(store) == ""


class TestExitOnlyStreakLineSurfaces:
    """DEFENSIVE_TRIM and DEFENSIVE_LIQUIDATION must surface the
    builder's own headline verbatim (invariant #10)."""

    def test_defensive_trim_surfaces_verbatim(self):
        # 3 trailing SELLs after a BUY → DEFENSIVE_TRIM
        store = _FakeStore(recent_trades=[
            _t("SELL", "TSLA"),  # newest
            _t("SELL", "MU"),
            _t("SELL", "AMD"),
            _t("BUY", "NVDA"),  # oldest
        ])
        out = reporter._exit_only_streak_line(store)
        assert out.startswith("**EXIT-ONLY** ◈ DEFENSIVE_TRIM")
        # Builder headline appears verbatim after the "> " prefix.
        assert "DEFENSIVE_TRIM" in out

    def test_defensive_liquidation_surfaces_verbatim(self):
        # 6 trailing SELLs after a BUY → DEFENSIVE_LIQUIDATION
        ledger = [_t("SELL", f"T{i}") for i in range(6)] + [_t("BUY", "X")]
        store = _FakeStore(recent_trades=ledger)
        out = reporter._exit_only_streak_line(store)
        assert out.startswith("**EXIT-ONLY** ◈ DEFENSIVE_LIQUIDATION")
        assert "liquidating" in out.lower()


class TestExitOnlyStreakLineFailureContract:
    """A builder/store fault must degrade to "" — never raise — so the
    hourly summary still ships even if this line cannot be computed."""

    def test_store_raise_returns_empty(self):
        class _RaiseStore:
            def recent_trades(self, limit=50):
                raise RuntimeError("simulated store fault")
        assert reporter._exit_only_streak_line(_RaiseStore()) == ""

    def test_builder_raise_returns_empty(self):
        store = _FakeStore(recent_trades=[_t("SELL")])
        with patch("paper_trader.analytics.exit_only_streak."
                   "build_exit_only_streak",
                   side_effect=RuntimeError("simulated builder fault")):
            assert reporter._exit_only_streak_line(store) == ""

    def test_non_dict_result_returns_empty(self):
        store = _FakeStore(recent_trades=[_t("SELL")])
        with patch("paper_trader.analytics.exit_only_streak."
                   "build_exit_only_streak", return_value="oops"):
            assert reporter._exit_only_streak_line(store) == ""

    def test_missing_headline_returns_empty(self):
        store = _FakeStore(recent_trades=[_t("SELL")])
        with patch("paper_trader.analytics.exit_only_streak."
                   "build_exit_only_streak",
                   return_value={"verdict": "DEFENSIVE_TRIM",
                                 "headline": ""}):
            assert reporter._exit_only_streak_line(store) == ""


class TestExitOnlyEndpoint:
    """The dashboard endpoint must compose the same builder verbatim."""

    def test_endpoint_returns_builder_output(self):
        from paper_trader import dashboard
        client = dashboard.app.test_client()
        with patch("paper_trader.dashboard.get_store") as gs:
            gs.return_value.recent_trades.return_value = [
                _t("SELL", "TSLA"),
                _t("SELL", "MU"),
                _t("SELL", "AMD"),
                _t("BUY", "NVDA"),
            ]
            resp = client.get("/api/exit-only-streak")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["verdict"] == "DEFENSIVE_TRIM"
        assert body["exit_run_length"] == 3
        assert "TSLA" in body["exit_run_tickers"]
