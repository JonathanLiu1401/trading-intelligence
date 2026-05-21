"""Tests for paper_trader.reporter._pos_alpha_token and its wiring into
_portfolio_lines.

Pure-function tests over the alpha-vs-SPY computation:
  alpha = (cur - avg) / avg * 100  -  (sp_now / sp_base - 1) * 100

Validates the suppression contract: missing data degrades to "" (drop
the token), never raises and never emits a misleading number.
"""
from __future__ import annotations

import pytest

from paper_trader import reporter
from paper_trader.reporter import _pos_alpha_token, _portfolio_lines


def _pos(ticker="NVDA", avg=100.0, cur=110.0, opened_at="2026-05-20T10:00:00+00:00",
         **extra):
    return {
        "ticker": ticker, "type": "stock", "qty": 1.0,
        "avg_cost": avg, "current_price": cur,
        "unrealized_pl": (cur - avg),
        "opened_at": opened_at, **extra,
    }


def _eq_point(ts: str, sp500: float | None, total: float = 1000.0):
    return {"timestamp": ts, "total_value": total, "cash": 0.0, "sp500_price": sp500}


class TestPosAlphaTokenSuppression:
    def test_empty_equity_curve_returns_empty(self):
        assert _pos_alpha_token(_pos(), [], 5000.0) == ""

    def test_none_equity_curve_returns_empty(self):
        assert _pos_alpha_token(_pos(), None, 5000.0) == ""

    def test_missing_sp500_now_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        assert _pos_alpha_token(_pos(), eq, None) == ""
        assert _pos_alpha_token(_pos(), eq, 0.0) == ""

    def test_stale_mark_returns_empty(self):
        # A stale_mark position has current_price == avg_cost; the alpha
        # number would lie next to the STALE flag.
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(stale_mark=True)
        assert _pos_alpha_token(p, eq, 5050.0) == ""

    def test_missing_opened_at_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(opened_at=None)
        assert _pos_alpha_token(p, eq, 5050.0) == ""
        p = _pos(opened_at="")
        assert _pos_alpha_token(p, eq, 5050.0) == ""

    def test_no_baseline_point_returns_empty(self):
        # Position opened AFTER every equity point in the curve.
        eq = [_eq_point("2026-05-20T09:00:00+00:00", 5000.0)]
        p = _pos(opened_at="2026-05-21T10:00:00+00:00")
        assert _pos_alpha_token(p, eq, 5050.0) == ""

    def test_base_sp500_missing_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", None)]
        assert _pos_alpha_token(_pos(), eq, 5050.0) == ""

    def test_base_sp500_zero_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 0.0)]
        assert _pos_alpha_token(_pos(), eq, 5050.0) == ""

    def test_avg_cost_zero_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=0.0)
        assert _pos_alpha_token(p, eq, 5050.0) == ""

    def test_current_price_zero_returns_empty(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(cur=0.0)
        assert _pos_alpha_token(p, eq, 5050.0) == ""


class TestPosAlphaTokenComputation:
    def test_outperforming_position_shows_positive_alpha(self):
        # Position: +10% (100 → 110)
        # SPY: +1% (5000 → 5050)
        # Alpha: +9.0%
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=110.0)
        token = _pos_alpha_token(p, eq, 5050.0)
        assert token == "  α +9.0%"

    def test_underperforming_position_shows_negative_alpha(self):
        # Position: +1% (100 → 101)
        # SPY: +5% (5000 → 5250)
        # Alpha: -4.0%
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=101.0)
        token = _pos_alpha_token(p, eq, 5250.0)
        assert token == "  α -4.0%"

    def test_position_tracking_spy_shows_zero_alpha(self):
        # Position: +2%; SPY: +2%; alpha: 0%
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=102.0)
        token = _pos_alpha_token(p, eq, 5100.0)
        assert token == "  α +0.0%"

    def test_picks_first_point_at_or_after_open(self):
        # Two equity points; the position opened between them — the
        # baseline must be the one at-or-after opened_at (NOT the earlier
        # one, which would pre-date the entry).
        eq = [
            _eq_point("2026-05-20T09:00:00+00:00", 4900.0),
            _eq_point("2026-05-20T11:00:00+00:00", 5000.0),
        ]
        p = _pos(avg=100.0, cur=110.0,
                 opened_at="2026-05-20T10:00:00+00:00")
        # Should use sp_base=5000 (the 11:00 point), so SPY +1% (5000→5050),
        # position +10% → alpha +9.0%
        token = _pos_alpha_token(p, eq, 5050.0)
        assert token == "  α +9.0%"

    def test_negative_pos_against_flat_spy(self):
        # Position: -5% (100 → 95)
        # SPY flat (5000 → 5000)
        # Alpha: -5.0%
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=95.0)
        token = _pos_alpha_token(p, eq, 5000.0)
        assert token == "  α -5.0%"

    def test_falling_position_in_rising_market(self):
        # Position: -2% (100 → 98)
        # SPY: +3% (5000 → 5150)
        # Alpha: -5.0%
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=98.0)
        token = _pos_alpha_token(p, eq, 5150.0)
        assert token == "  α -5.0%"


class TestPosAlphaTokenWiredIntoPortfolioLines:
    def test_portfolio_lines_emits_alpha_token_when_data_present(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=110.0)
        lines = _portfolio_lines([p], total_value=1110.0,
                                  equity_asc=eq, sp500_now=5050.0)
        assert len(lines) == 1
        assert "α +9.0%" in lines[0]

    def test_portfolio_lines_silent_without_alpha_kwargs(self):
        # Byte-compatibility: existing callers that don't pass equity_asc
        # / sp500_now get NO alpha token.
        p = _pos(avg=100.0, cur=110.0)
        lines = _portfolio_lines([p], total_value=1110.0)
        assert len(lines) == 1
        assert " α " not in lines[0]

    def test_portfolio_lines_alpha_silent_on_stale_mark(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        p = _pos(avg=100.0, cur=100.0, stale_mark=True)
        lines = _portfolio_lines([p], total_value=1100.0,
                                  equity_asc=eq, sp500_now=5050.0)
        assert " α " not in lines[0]
        # But the STALE marker is still there
        assert "STALE" in lines[0]

    def test_portfolio_lines_alpha_works_for_option_position(self):
        eq = [_eq_point("2026-05-20T10:00:00+00:00", 5000.0)]
        # Options multiplier is in market_value, but alpha is a % over
        # avg/cur prices, NOT $ — so the option row uses the same
        # arithmetic as stock rows.
        p = {
            "ticker": "NVDA", "type": "call", "qty": 1.0,
            "avg_cost": 5.0, "current_price": 5.5,
            "strike": 200.0, "expiry": "2026-06-19",
            "unrealized_pl": 50.0,
            "opened_at": "2026-05-20T10:00:00+00:00",
        }
        lines = _portfolio_lines([p], total_value=1100.0,
                                  equity_asc=eq, sp500_now=5050.0)
        # Option +10% (5→5.5); SPY +1%; alpha +9.0%
        assert "α +9.0%" in lines[0]
