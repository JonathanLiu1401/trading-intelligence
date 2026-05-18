"""Unit tests for analytics.ticker_dossier.build_ticker_dossier.

These pin *exact* hand-computed aggregates — a wrong filter, a missed
option leg, a decision-ticker mismatch, or a P&L sign error must fail here,
not surface silently on the /ticker/<sym> page. The function is the single
source of truth behind /api/ticker/<sym> and the standalone page.

The contract is "pure, never raises": the malformed-input tests assert it
degrades bad rows to skips rather than throwing (a drill-down panel must
fail soft like every other analytics module).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.ticker_dossier import (
    articles_mentioning, build_ticker_dossier)


def _parse(s):
    """Tiny stand-in for dashboard._parse_action_ticker (BUY MU → FILLED)."""
    if not s or s in ("NO_DECISION", "BLOCKED"):
        return (s or "", None)
    head = s.split("→")[0].split()
    return (head[0].upper(), head[1].upper() if len(head) > 1 else None)


def _trade(tid, ts, ticker, action, qty, price):
    return {"id": tid, "timestamp": ts, "ticker": ticker, "action": action,
            "qty": qty, "price": price, "value": qty * price,
            "strike": None, "expiry": None, "option_type": None,
            "reason": f"{action} {ticker} thesis"}


class TestPositionLegs:
    def test_stock_and_option_legs_only_for_symbol(self):
        positions = [
            {"ticker": "MU", "type": "stock", "qty": 10, "avg_cost": 5.0,
             "current_price": 6.0, "unrealized_pl": 10.0},
            {"ticker": "MU", "type": "call", "qty": 2, "avg_cost": 1.0,
             "current_price": 1.5, "unrealized_pl": 100.0,
             "strike": 130.0, "expiry": "2026-06-19"},
            {"ticker": "NVDA", "type": "stock", "qty": 3, "avg_cost": 100.0,
             "current_price": 110.0, "unrealized_pl": 30.0},
        ]
        d = build_ticker_dossier("mu", positions=positions, trades=[],
                                 decisions=[], signals_list=[], sentiment={},
                                 parse_action_ticker=_parse)
        assert d["symbol"] == "MU"
        assert d["held"] is True
        assert len(d["position"]["legs"]) == 2          # NVDA excluded
        assert d["position"]["unrealized_pl_total"] == 110.0  # 10 + 100
        assert d["has_coverage"] is True

    def test_zero_qty_leg_is_not_held(self):
        d = build_ticker_dossier("MU", positions=[
            {"ticker": "MU", "type": "stock", "qty": 0, "avg_cost": 5.0}],
            trades=[], decisions=[], signals_list=[], sentiment={},
            parse_action_ticker=_parse)
        assert d["held"] is False
        assert d["position"] is None


class TestRealizedThisNameOnly:
    def test_round_trips_filtered_and_pnl_exact(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "MU", "BUY", 10, 5.0),
            _trade(2, "2026-05-03T10:00:00+00:00", "MU", "SELL", 10, 7.0),  # +20
            _trade(3, "2026-05-04T10:00:00+00:00", "MU", "BUY", 5, 10.0),
            _trade(4, "2026-05-05T10:00:00+00:00", "MU", "SELL", 5, 8.0),   # -10
            _trade(5, "2026-05-01T10:00:00+00:00", "NVDA", "BUY", 1, 100.0),
            _trade(6, "2026-05-02T10:00:00+00:00", "NVDA", "SELL", 1, 200.0),  # excluded
        ]
        d = build_ticker_dossier("MU", positions=[], trades=trades,
                                 decisions=[], signals_list=[], sentiment={},
                                 parse_action_ticker=_parse)
        r = d["realized"]
        assert r["n_round_trips"] == 2          # NVDA RT excluded
        assert r["n_wins"] == 1
        assert r["n_losses"] == 1
        assert r["win_rate_pct"] == 50.0
        assert r["total_pnl_usd"] == 10.0       # +20 -10
        assert r["avg_hold_days"] == pytest.approx(1.5)  # 2d and 1d
        assert len(d["round_trips"]) == 2
        assert {rt["ticker"] for rt in d["round_trips"]} == {"MU"}

    def test_no_history_is_zeroed_not_errored(self):
        d = build_ticker_dossier("ZZZZ", positions=[], trades=[], decisions=[],
                                 signals_list=[], sentiment={},
                                 parse_action_ticker=_parse)
        assert d["realized"]["n_round_trips"] == 0
        assert d["realized"]["win_rate_pct"] is None
        assert d["realized"]["total_pnl_usd"] == 0.0
        assert d["has_coverage"] is False


class TestDecisionTrail:
    def test_only_decisions_touching_symbol(self):
        decisions = [
            {"timestamp": "2026-05-05T09:00:00+00:00",
             "action_taken": "BUY MU → FILLED", "reasoning": "DRAM catalyst"},
            {"timestamp": "2026-05-05T09:01:00+00:00",
             "action_taken": "SELL NVDA → FILLED", "reasoning": "trim"},
            {"timestamp": "2026-05-05T09:02:00+00:00",
             "action_taken": "NO_DECISION", "reasoning": "no edge"},
        ]
        d = build_ticker_dossier("MU", positions=[], trades=[],
                                 decisions=decisions, signals_list=[],
                                 sentiment={}, parse_action_ticker=_parse)
        assert len(d["decisions"]) == 1
        assert d["decisions"][0]["verb"] == "BUY"
        assert d["decisions"][0]["reasoning"] == "DRAM catalyst"

    def test_parser_that_raises_skips_row_not_dossier(self):
        def angry(s):
            if "BOOM" in s:
                raise ValueError("bad row")
            return _parse(s)
        decisions = [
            {"action_taken": "BOOM", "reasoning": "x"},
            {"action_taken": "BUY MU → FILLED", "reasoning": "ok"},
        ]
        d = build_ticker_dossier("MU", positions=[], trades=[],
                                 decisions=decisions, signals_list=[],
                                 sentiment={}, parse_action_ticker=angry)
        assert len(d["decisions"]) == 1   # the BOOM row was skipped, not raised


class TestNewsFlow:
    def test_articles_mentioning_uses_tickers_field(self):
        sigs = [
            {"title": "Micron pops", "tickers": ["MU"], "ai_score": 9.0},
            {"title": "Nvidia only", "tickers": ["NVDA"], "ai_score": 8.0},
            {"title": "MU + AMD", "tickers": ["MU", "AMD"], "ai_score": 7.0},
        ]
        assert {a["title"] for a in articles_mentioning("MU", sigs)} == {
            "Micron pops", "MU + AMD"}
        assert articles_mentioning("", sigs) == []

    def test_sentiment_passthrough_and_article_subset(self):
        sigs = [
            {"title": "Micron pops", "tickers": ["MU"], "ai_score": 9.0,
             "source": "Reuters", "urgency": 1, "url": "u",
             "first_seen": "2026-05-18T01:00:00+00:00", "summary": "body"},
            {"title": "Nvidia", "tickers": ["NVDA"], "ai_score": 8.0},
        ]
        d = build_ticker_dossier(
            "MU", positions=[], trades=[], decisions=[], signals_list=sigs,
            sentiment={"avg_score": 7.5, "max_score": 9.0, "n": 4, "urgent": 2},
            parse_action_ticker=_parse)
        assert len(d["news"]["articles"]) == 1
        assert d["news"]["articles"][0]["source"] == "Reuters"
        assert d["news"]["sentiment"] == {"avg_score": 7.5, "max_score": 9.0,
                                          "n": 4, "urgent": 2}
        assert d["has_coverage"] is True


class TestNeverRaises:
    def test_garbage_inputs_degrade_gracefully(self):
        d = build_ticker_dossier(
            "MU",
            positions=["not-a-dict", {"ticker": "MU", "qty": "abc"}],
            trades=["x", {"ticker": "MU", "action": "BUY"}],
            decisions=["y", {"action_taken": None}],
            signals_list=["z", {"tickers": "MU"}],   # tickers not a list
            sentiment="not-a-dict",
            parse_action_ticker=_parse)
        assert d["symbol"] == "MU"
        assert d["news"]["sentiment"]["avg_score"] == 0.0
        assert isinstance(d["round_trips"], list)
