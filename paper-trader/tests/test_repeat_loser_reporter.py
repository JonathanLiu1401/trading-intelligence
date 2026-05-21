"""Tests for paper_trader.reporter._repeat_loser_line and its wiring
into send_hourly_summary / send_daily_close.

Mirrors the structure of test_streak_reporter.py since this is the
per-ticker sibling of the aggregate streak surface.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paper_trader.reporter import (
    _repeat_loser_line,
    send_daily_close,
    send_hourly_summary,
)


def _store_returning(trades):
    """Build a MagicMock store whose recent_trades(N) returns ``trades``
    newest-first (matching the store contract; the reporter reverses it
    to feed build_repeat_loser oldest-first).

    Other methods get safe defaults so the reporter helper itself doesn't
    blow up — only ``recent_trades`` is exercised here.
    """
    store = MagicMock()
    store.recent_trades.return_value = list(trades)
    return store


def _pair(ticker: str, day: str, *, win: bool, pnl: float | None = None):
    """Same helper as test_repeat_loser, but emits newest-first ordering
    so the store mock returns the SELL first (the recent_trades contract).
    """
    if pnl is None:
        pnl = 5.0 if win else -5.0
    cost = 100.0
    proceeds = cost + pnl
    # Reporter does `reversed(store.recent_trades(...))` so we hand back
    # newest-first; here SELL is newer than BUY.
    return [
        {
            "ticker": ticker, "action": "SELL", "qty": 1.0,
            "price": proceeds, "value": proceeds,
            "timestamp": f"{day}T15:00:00+00:00",
            "option_type": None, "strike": None, "expiry": None, "id": None,
        },
        {
            "ticker": ticker, "action": "BUY", "qty": 1.0,
            "price": cost, "value": cost,
            "timestamp": f"{day}T09:00:00+00:00",
            "option_type": None, "strike": None, "expiry": None, "id": None,
        },
    ]


class TestRepeatLoserLineSuppression:
    def test_empty_book_returns_empty(self):
        store = _store_returning([])
        assert _repeat_loser_line(store) == ""

    def test_ok_book_returns_empty(self):
        # one winning trip → state OK, no surfacing
        trades = _pair("NVDA", "2026-05-01", win=True)
        store = _store_returning(trades)
        assert _repeat_loser_line(store) == ""

    def test_single_losing_trip_below_threshold_returns_empty(self):
        # 1 loss is below the 2-loss threshold
        trades = _pair("LITE", "2026-05-01", win=False)
        store = _store_returning(trades)
        assert _repeat_loser_line(store) == ""


class TestRepeatLoserLineSurfaces:
    def test_two_consecutive_losses_surfaces_block(self):
        # Need newest-first ordering: day 2 SELL/BUY, then day 1 SELL/BUY
        trades = (
            _pair("LITE", "2026-05-02", win=False, pnl=-6.0)
            + _pair("LITE", "2026-05-01", win=False, pnl=-4.0)
        )
        store = _store_returning(trades)
        line = _repeat_loser_line(store)
        assert "REPEAT_LOSER" in line
        assert "LITE" in line
        assert "2-loss run" in line
        # The visual flag is present
        assert line.startswith("⚠️")

    def test_three_loss_run_surfaces_correct_count(self):
        trades = (
            _pair("MU", "2026-05-03", win=False)
            + _pair("MU", "2026-05-02", win=False)
            + _pair("MU", "2026-05-01", win=False)
        )
        store = _store_returning(trades)
        line = _repeat_loser_line(store)
        assert "3-loss run" in line
        assert "MU" in line


class TestRepeatLoserLineFailureContract:
    def test_store_raise_returns_empty(self):
        store = MagicMock()
        store.recent_trades.side_effect = RuntimeError("boom")
        assert _repeat_loser_line(store) == ""

    def test_builder_raise_returns_empty(self):
        store = _store_returning([])
        with patch(
            "paper_trader.analytics.repeat_loser.build_repeat_loser",
            side_effect=RuntimeError("builder went bad"),
        ):
            assert _repeat_loser_line(store) == ""

    def test_non_dict_result_returns_empty(self):
        store = _store_returning([])
        with patch(
            "paper_trader.analytics.repeat_loser.build_repeat_loser",
            return_value="not a dict",
        ):
            assert _repeat_loser_line(store) == ""

    def test_missing_headline_returns_empty(self):
        store = _store_returning([])
        with patch(
            "paper_trader.analytics.repeat_loser.build_repeat_loser",
            return_value={"verdict": "REPEAT_LOSER", "headline": ""},
        ):
            assert _repeat_loser_line(store) == ""

    def test_no_verdict_returns_empty(self):
        store = _store_returning([])
        with patch(
            "paper_trader.analytics.repeat_loser.build_repeat_loser",
            return_value={"verdict": None, "headline": "anything"},
        ):
            assert _repeat_loser_line(store) == ""


class TestRepeatLoserWiredIntoSummaries:
    """End-to-end: when build_repeat_loser flags REPEAT_LOSER, the line
    actually appears in the hourly / daily-close body that goes to _send."""

    def test_hourly_summary_surfaces_repeat_loser(self):
        sent_bodies = []
        with (
            patch(
                "paper_trader.reporter._send",
                side_effect=lambda body: (sent_bodies.append(body), True)[1],
            ),
            patch(
                "paper_trader.analytics.repeat_loser.build_repeat_loser",
                return_value={
                    "state": "REPEAT_LOSER",
                    "verdict": "REPEAT_LOSER",
                    "headline": "REPEAT_LOSER — LITE on a 3-loss run.",
                    "offenders": [{"ticker": "LITE", "streak": 3}],
                    "n_offenders": 1,
                    "n_round_trips": 8,
                    "threshold": 2,
                    "per_ticker": {},
                    "as_of": "2026-05-20T14:00:00",
                },
            ),
        ):
            send_hourly_summary()
        assert sent_bodies, "expected _send to be called"
        body = sent_bodies[0]
        assert "REPEAT_LOSER" in body
        assert "LITE" in body

    def test_hourly_summary_silent_when_ok(self):
        sent_bodies = []
        with (
            patch(
                "paper_trader.reporter._send",
                side_effect=lambda body: (sent_bodies.append(body), True)[1],
            ),
            patch(
                "paper_trader.analytics.repeat_loser.build_repeat_loser",
                return_value={
                    "state": "OK",
                    "verdict": None,
                    "headline": "OK — no offenders.",
                    "offenders": [],
                    "n_offenders": 0,
                    "n_round_trips": 4,
                    "threshold": 2,
                    "per_ticker": {},
                    "as_of": "2026-05-20T14:00:00",
                },
            ),
        ):
            send_hourly_summary()
        assert sent_bodies
        # REPEAT_LOSER block should NOT appear when verdict is None
        # (other unrelated mentions of the word are tolerated in the
        # commit_msg / docstrings — but the specific block prefix is)
        assert "**REPEAT_LOSER**" not in sent_bodies[0]

    def test_daily_close_surfaces_repeat_loser(self):
        sent_bodies = []
        with (
            patch(
                "paper_trader.reporter._send",
                side_effect=lambda body: (sent_bodies.append(body), True)[1],
            ),
            patch(
                "paper_trader.analytics.repeat_loser.build_repeat_loser",
                return_value={
                    "state": "REPEAT_LOSER",
                    "verdict": "REPEAT_LOSER",
                    "headline": "REPEAT_LOSER — MU on a 4-loss run.",
                    "offenders": [{"ticker": "MU", "streak": 4}],
                    "n_offenders": 1,
                    "n_round_trips": 10,
                    "threshold": 2,
                    "per_ticker": {},
                    "as_of": "2026-05-20T16:05:00",
                },
            ),
        ):
            send_daily_close()
        assert sent_bodies
        body = sent_bodies[0]
        assert "REPEAT_LOSER" in body
        assert "MU" in body

    def test_summary_still_sends_when_builder_faults(self):
        # the additive failure contract: a builder fault drops the line,
        # never the whole report
        sent_bodies = []
        with (
            patch(
                "paper_trader.reporter._send",
                side_effect=lambda body: (sent_bodies.append(body), True)[1],
            ),
            patch(
                "paper_trader.analytics.repeat_loser.build_repeat_loser",
                side_effect=RuntimeError("kaboom"),
            ),
        ):
            send_hourly_summary()
        assert sent_bodies, "hourly summary must still send when builder faults"
