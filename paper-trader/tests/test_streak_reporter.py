"""Tests for `reporter._streak_line` — Discord surface for the existing
`build_streak` analytic.

`build_streak` (paper_trader/analytics/streak.py) is wired only to
`/api/streak` on the dashboard. The operator who lives in Discord never
sees HOT_HAND (overconfidence risk after a 4+ win run) or TILT_RISK
(loss-cluster bias after a 4+ loss run). This module covers the new
`_streak_line` that routes the builder's own verdict to the surface the
operator actually reads — the exact dashboard→Discord trajectory
`_capital_pulse_line` / `_hold_discipline_line` each followed.

Tests verify:
  * NEUTRAL / EMERGING / NO_DATA states are silent (no hourly noise).
  * HOT_HAND verdict surfaces verbatim from the builder (invariant #10).
  * TILT_RISK verdict surfaces verbatim from the builder.
  * Builder fault degrades to "" (additive failure contract).
  * Wired into both send_hourly_summary and send_daily_close.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from paper_trader import reporter


class _FakeStore:
    """Minimal store stub — every reporter call goes through these methods."""
    def __init__(self, **kw):
        self._portfolio = kw.get("portfolio", {
            "cash": 500.0, "total_value": 1000.0, "positions": [],
            "last_updated": "2026-05-20T12:00:00+00:00",
        })
        self._open_positions = kw.get("open_positions", [])
        self._recent_trades = kw.get("recent_trades", [])
        self._recent_decisions = kw.get("recent_decisions", [])
        self._equity_curve = kw.get("equity_curve", [])

    def get_portfolio(self):
        return dict(self._portfolio)

    def open_positions(self):
        return list(self._open_positions)

    def recent_trades(self, limit=50):
        return list(self._recent_trades[:limit])

    def recent_decisions(self, limit=20):
        return list(self._recent_decisions[:limit])

    def equity_curve(self, limit=500):
        return list(self._equity_curve)


def _make_round_trip_trades(outcomes: list[str], ticker: str = "NVDA"
                            ) -> list[dict]:
    """Build a synthetic trade ledger that produces the given W/L outcome
    sequence (chronologically) when passed (newest-first) to ``recent_trades``.

    Each outcome generates one BUY then one SELL with the appropriate
    PnL sign. ``W`` → +$1 win, ``L`` → -$1 loss, ``F`` → flat. Trades come
    back NEWEST-FIRST (the store contract — ``reversed`` in the consumer
    yields chronological order that ``build_round_trips`` requires).
    """
    chronological: list[dict] = []
    for i, o in enumerate(outcomes):
        buy_ts = f"2026-05-10T09:{30 + i * 2:02d}:00+00:00"
        sell_ts = f"2026-05-10T09:{31 + i * 2:02d}:00+00:00"
        buy_px = 100.0
        if o == "W":
            sell_px = 101.0
        elif o == "L":
            sell_px = 99.0
        else:
            sell_px = 100.0
        # Append in chronological order: BUY first (oldest), then SELL.
        chronological.append({
            "timestamp": buy_ts, "ticker": ticker, "action": "BUY",
            "qty": 1.0, "price": buy_px, "value": buy_px,
            "option_type": None, "strike": None, "expiry": None,
        })
        chronological.append({
            "timestamp": sell_ts, "ticker": ticker, "action": "SELL",
            "qty": 1.0, "price": sell_px, "value": sell_px,
            "option_type": None, "strike": None, "expiry": None,
        })
    # Store contract: newest-first.
    return list(reversed(chronological))


class TestStreakLineSuppression:
    """A balanced book or insufficient sample must produce no Discord line.

    The summary must never become its own lying green light — the
    `_hold_discipline_line` NO_DATA / `_capital_pulse_line` FREE
    suppression precedent."""

    def test_empty_book_returns_empty(self):
        store = _FakeStore(recent_trades=[])
        assert reporter._streak_line(store) == ""

    def test_emerging_below_stable_returns_empty(self):
        # 3 round-trips < STABLE_MIN_ROUND_TRIPS (8) → EMERGING → silent
        store = _FakeStore(recent_trades=_make_round_trip_trades(
            ["W", "L", "W"]))
        assert reporter._streak_line(store) == ""

    def test_neutral_streak_returns_empty(self):
        # 8 alternating W/L → NEUTRAL (no run >= threshold)
        store = _FakeStore(recent_trades=_make_round_trip_trades(
            ["W", "L", "W", "L", "W", "L", "W", "L"]))
        assert reporter._streak_line(store) == ""


class TestStreakLineSurfaces:
    """HOT_HAND and TILT_RISK must surface the builder's own headline
    verbatim (invariant #10 — no re-derivation). The Discord line and
    `/api/streak` can never tell different stories."""

    def test_hot_hand_surfaces_verbatim(self):
        # 4 wins in a row at the end of an 8-trip series → HOT_HAND
        outcomes = ["L", "L", "L", "L", "W", "W", "W", "W"]
        trades = _make_round_trip_trades(outcomes)
        store = _FakeStore(recent_trades=trades)
        out = reporter._streak_line(store)
        assert out, "should surface HOT_HAND verdict"
        assert "STREAK" in out
        assert "HOT_HAND" in out
        # Builder's headline includes "on a 4-win run"
        assert "4-win run" in out

    def test_tilt_risk_surfaces_verbatim(self):
        # 4 losses in a row at the end → TILT_RISK
        outcomes = ["W", "W", "W", "W", "L", "L", "L", "L"]
        trades = _make_round_trip_trades(outcomes)
        store = _FakeStore(recent_trades=trades)
        out = reporter._streak_line(store)
        assert out, "should surface TILT_RISK verdict"
        assert "STREAK" in out
        assert "TILT_RISK" in out
        assert "4-loss run" in out

    def test_longer_loss_run_still_surfaces(self):
        # 6 losses in a row — well above threshold
        outcomes = ["W", "W"] + ["L"] * 6
        trades = _make_round_trip_trades(outcomes)
        store = _FakeStore(recent_trades=trades)
        out = reporter._streak_line(store)
        assert "TILT_RISK" in out
        assert "6-loss run" in out


class TestStreakLineFailureContract:
    """Any builder/store fault degrades to ``""`` — never an exception
    that takes down the whole hourly/daily summary."""

    def test_store_raise_returns_empty(self):
        class BadStore:
            def recent_trades(self, limit=50):
                raise RuntimeError("store down")
        out = reporter._streak_line(BadStore())
        assert out == ""

    def test_builder_raise_returns_empty(self):
        store = _FakeStore(recent_trades=_make_round_trip_trades(
            ["W", "W", "W", "W"] * 2))
        with patch(
            "paper_trader.analytics.streak.build_streak",
            side_effect=RuntimeError("builder boom"),
        ):
            assert reporter._streak_line(store) == ""

    def test_non_dict_result_returns_empty(self):
        store = _FakeStore(recent_trades=_make_round_trip_trades(
            ["W"] * 8))
        with patch(
            "paper_trader.analytics.streak.build_streak",
            return_value=None,
        ):
            assert reporter._streak_line(store) == ""

    def test_missing_headline_returns_empty(self):
        store = _FakeStore(recent_trades=_make_round_trip_trades(
            ["W"] * 8))
        with patch(
            "paper_trader.analytics.streak.build_streak",
            return_value={"state": "STABLE", "verdict": "HOT_HAND",
                          "headline": ""},
        ):
            assert reporter._streak_line(store) == ""


class TestStreakLineWiredIntoSummaries:
    """The line must be wired into both `send_hourly_summary` and
    `send_daily_close` — the operator must see streak signals on the
    surface they actually read (the `_capital_pulse_line` /
    `_hold_discipline_line` precedent)."""

    def test_hourly_summary_surfaces_tilt_risk(self, monkeypatch):
        """A real fake store with a 4-loss streak → hourly body must
        contain the TILT_RISK line."""
        outcomes = ["W", "W", "W", "W", "L", "L", "L", "L"]
        trades = _make_round_trip_trades(outcomes)

        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)

        # Use the live get_store path but feed it a fake
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)

        # Neutralise yfinance benchmark
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)

        ok = reporter.send_hourly_summary()
        assert ok
        assert sent, "send must have been called"
        body = sent[0]
        assert "TILT_RISK" in body, (
            "TILT_RISK streak must appear in hourly summary; got:\n" + body
        )

    def test_hourly_summary_silent_when_neutral(self, monkeypatch):
        outcomes = ["W", "L"] * 5
        trades = _make_round_trip_trades(outcomes)
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)
        reporter.send_hourly_summary()
        body = sent[0]
        assert "STREAK" not in body, (
            "NEUTRAL must not surface a STREAK line; got:\n" + body
        )

    def test_daily_close_surfaces_hot_hand(self, monkeypatch):
        outcomes = ["L", "L", "L", "L", "W", "W", "W", "W"]
        trades = _make_round_trip_trades(outcomes)
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)
        reporter.send_daily_close()
        body = sent[0]
        assert "HOT_HAND" in body, (
            "HOT_HAND streak must appear in daily close; got:\n" + body
        )

    def test_summary_still_sends_when_streak_builder_faults(
            self, monkeypatch):
        """Builder explodes → hourly still ships, streak line just absent."""
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=[])
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)
        with patch(
            "paper_trader.analytics.streak.build_streak",
            side_effect=RuntimeError("builder explode"),
        ):
            ok = reporter.send_hourly_summary()
        assert ok, "hourly send must still succeed"
        assert "STREAK" not in sent[0]
