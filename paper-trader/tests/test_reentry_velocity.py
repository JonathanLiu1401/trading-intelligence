"""Unit tests for paper_trader.analytics.reentry_velocity.

The builder composes round_trips and walks each key's exits to the next
entry. Tests assert classification thresholds, ordering, the open-after-
close path, and the verdict ladder — everything operators key off of.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.reentry_velocity import build_reentry_velocity


def _trade(tid, ts, ticker, action, qty, price, *, option_type=None,
           strike=None, expiry=None):
    mult = 100 if option_type in ("call", "put") else 1
    return {
        "id": tid,
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price * mult,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
    }


class TestEmpty:
    def test_no_trades_sparse(self):
        out = build_reentry_velocity([])
        assert out["verdict"] == "SPARSE"
        assert out["n_round_trips"] == 0
        assert out["n_gaps"] == 0
        assert out["median_gap_hours"] is None
        assert out["recent_gaps"] == []
        assert out["per_ticker"] == []

    def test_single_open_round_trip_no_gap(self):
        # One BUY, no SELL — no closed round-trip, no gap.
        trades = [_trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0)]
        out = build_reentry_velocity(trades)
        assert out["n_round_trips"] == 0
        assert out["n_gaps"] == 0
        assert out["verdict"] == "SPARSE"


class TestClassifications:
    def test_immediate_under_1h(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 101.0),
            _trade(3, "2026-05-19T11:30:00+00:00", "NVDA", "BUY", 10, 102.0),
            _trade(4, "2026-05-19T12:30:00+00:00", "NVDA", "SELL", 10, 103.0),
        ]
        out = build_reentry_velocity(trades)
        assert out["n_round_trips"] == 2
        assert out["n_gaps"] == 1
        gap = out["recent_gaps"][0]
        assert gap["ticker"] == "NVDA"
        assert gap["classification"] == "IMMEDIATE"
        assert abs(gap["gap_hours"] - 0.5) < 1e-6

    def test_same_day_1_to_24h(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "AAPL", "BUY", 5, 200.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "AAPL", "SELL", 5, 201.0),
            _trade(3, "2026-05-19T17:00:00+00:00", "AAPL", "BUY", 5, 202.0),
            _trade(4, "2026-05-19T18:00:00+00:00", "AAPL", "SELL", 5, 203.0),
        ]
        out = build_reentry_velocity(trades)
        gap = out["recent_gaps"][0]
        assert gap["classification"] == "SAME_DAY"
        assert abs(gap["gap_hours"] - 6.0) < 1e-6

    def test_quick_1_to_3_days(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "TSLA", "BUY", 1, 300.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "TSLA", "SELL", 1, 301.0),
            _trade(3, "2026-05-21T11:00:00+00:00", "TSLA", "BUY", 1, 302.0),
            _trade(4, "2026-05-21T12:00:00+00:00", "TSLA", "SELL", 1, 303.0),
        ]
        out = build_reentry_velocity(trades)
        gap = out["recent_gaps"][0]
        assert gap["classification"] == "QUICK"

    def test_normal_3_to_14_days(self):
        trades = [
            _trade(1, "2026-05-01T10:00:00+00:00", "META", "BUY", 1, 400.0),
            _trade(2, "2026-05-01T11:00:00+00:00", "META", "SELL", 1, 401.0),
            _trade(3, "2026-05-08T11:00:00+00:00", "META", "BUY", 1, 402.0),
            _trade(4, "2026-05-08T12:00:00+00:00", "META", "SELL", 1, 403.0),
        ]
        out = build_reentry_velocity(trades)
        gap = out["recent_gaps"][0]
        assert gap["classification"] == "NORMAL"

    def test_rare_over_14_days(self):
        trades = [
            _trade(1, "2026-04-01T10:00:00+00:00", "GOOG", "BUY", 1, 100.0),
            _trade(2, "2026-04-01T11:00:00+00:00", "GOOG", "SELL", 1, 101.0),
            _trade(3, "2026-05-01T11:00:00+00:00", "GOOG", "BUY", 1, 102.0),
            _trade(4, "2026-05-01T12:00:00+00:00", "GOOG", "SELL", 1, 103.0),
        ]
        out = build_reentry_velocity(trades)
        gap = out["recent_gaps"][0]
        assert gap["classification"] == "RARE"


class TestPerTickerAndKeyIsolation:
    def test_separate_keys_dont_pair(self):
        # NVDA stock and an NVDA call are different position keys —
        # closing the stock then opening the call is NOT a re-entry.
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 101.0),
            _trade(3, "2026-05-19T11:30:00+00:00", "NVDA", "BUY_CALL", 1, 5.0,
                   option_type="call", strike=110, expiry="2026-06-20"),
        ]
        out = build_reentry_velocity(trades)
        assert out["n_gaps"] == 0

    def test_per_ticker_aggregates_two_gaps(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "X", "BUY", 1, 10.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "X", "SELL", 1, 11.0),
            _trade(3, "2026-05-19T15:00:00+00:00", "X", "BUY", 1, 12.0),
            _trade(4, "2026-05-19T16:00:00+00:00", "X", "SELL", 1, 13.0),
            _trade(5, "2026-05-20T16:00:00+00:00", "X", "BUY", 1, 14.0),
            _trade(6, "2026-05-20T17:00:00+00:00", "X", "SELL", 1, 15.0),
        ]
        out = build_reentry_velocity(trades)
        assert out["n_gaps"] == 2
        x_row = next(p for p in out["per_ticker"] if p["ticker"] == "X")
        assert x_row["n_gaps"] == 2
        # Gaps: 11:00→15:00 = 4h, 16:00→D+1 16:00 = 24h.
        # Median of {4h, 24h} = 14h.
        assert abs(x_row["median_gap_hours"] - 14.0) < 1e-6
        assert abs(x_row["min_gap_hours"] - 4.0) < 1e-6


class TestOpenAfterClose:
    def test_open_position_gap_from_last_close(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "NVDA", "BUY", 10, 100.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "NVDA", "SELL", 10, 101.0),
        ]
        open_pos = [{
            "ticker": "NVDA",
            "type": "stock",
            "qty": 5,
            "strike": None,
            "expiry": None,
            "opened_at": "2026-05-19T13:00:00+00:00",
        }]
        out = build_reentry_velocity(trades, open_positions=open_pos)
        assert out["n_gaps"] == 1
        g = out["recent_gaps"][0]
        assert g["open_after_close"] is True
        assert g["ticker"] == "NVDA"
        assert abs(g["gap_hours"] - 2.0) < 1e-6

    def test_open_with_no_prior_close_no_gap(self):
        open_pos = [{
            "ticker": "AAPL", "type": "stock", "qty": 5,
            "strike": None, "expiry": None,
            "opened_at": "2026-05-19T13:00:00+00:00",
        }]
        out = build_reentry_velocity([], open_positions=open_pos)
        assert out["n_gaps"] == 0
        assert out["verdict"] == "SPARSE"


class TestVerdictLadder:
    def test_churn_risk_when_median_under_24h(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "A", "BUY", 1, 10.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "A", "SELL", 1, 11.0),
            _trade(3, "2026-05-19T13:00:00+00:00", "A", "BUY", 1, 12.0),
            _trade(4, "2026-05-19T14:00:00+00:00", "A", "SELL", 1, 13.0),
            _trade(5, "2026-05-19T16:00:00+00:00", "A", "BUY", 1, 14.0),
            _trade(6, "2026-05-19T17:00:00+00:00", "A", "SELL", 1, 15.0),
        ]
        out = build_reentry_velocity(trades)
        assert out["verdict"] == "CHURN_RISK"
        assert out["median_gap_hours"] is not None
        assert out["median_gap_hours"] < 24.0

    def test_stable_when_all_normal(self):
        trades = [
            _trade(1, "2026-04-01T10:00:00+00:00", "B", "BUY", 1, 10.0),
            _trade(2, "2026-04-01T11:00:00+00:00", "B", "SELL", 1, 11.0),
            _trade(3, "2026-04-08T11:00:00+00:00", "B", "BUY", 1, 12.0),
            _trade(4, "2026-04-08T12:00:00+00:00", "B", "SELL", 1, 13.0),
            _trade(5, "2026-04-15T12:00:00+00:00", "B", "BUY", 1, 14.0),
            _trade(6, "2026-04-15T13:00:00+00:00", "B", "SELL", 1, 15.0),
        ]
        out = build_reentry_velocity(trades)
        assert out["verdict"] == "STABLE"

    def test_buckets_sum_to_n_gaps(self):
        trades = [
            _trade(1, "2026-05-19T10:00:00+00:00", "C", "BUY", 1, 10.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "C", "SELL", 1, 11.0),
            _trade(3, "2026-05-19T11:30:00+00:00", "C", "BUY", 1, 12.0),
            _trade(4, "2026-05-19T12:30:00+00:00", "C", "SELL", 1, 13.0),
            _trade(5, "2026-06-01T12:30:00+00:00", "C", "BUY", 1, 14.0),
            _trade(6, "2026-06-01T13:30:00+00:00", "C", "SELL", 1, 15.0),
        ]
        out = build_reentry_velocity(trades)
        assert sum(out["buckets"].values()) == out["n_gaps"]


class TestInputOrderTolerance:
    def test_newest_first_input_still_works(self):
        trades_oldest_first = [
            _trade(1, "2026-05-19T10:00:00+00:00", "D", "BUY", 1, 10.0),
            _trade(2, "2026-05-19T11:00:00+00:00", "D", "SELL", 1, 11.0),
            _trade(3, "2026-05-19T15:00:00+00:00", "D", "BUY", 1, 12.0),
            _trade(4, "2026-05-19T16:00:00+00:00", "D", "SELL", 1, 13.0),
        ]
        out_old = build_reentry_velocity(trades_oldest_first)
        out_new = build_reentry_velocity(list(reversed(trades_oldest_first)))
        assert out_old["n_gaps"] == out_new["n_gaps"] == 1
        assert out_old["recent_gaps"][0]["gap_hours"] == out_new["recent_gaps"][0]["gap_hours"]


class TestRecentLimit:
    def test_recent_gaps_capped(self):
        ts_pairs = [
            ("2026-05-01T10:00:00+00:00", "2026-05-01T11:00:00+00:00"),
            ("2026-05-02T10:00:00+00:00", "2026-05-02T11:00:00+00:00"),
            ("2026-05-03T10:00:00+00:00", "2026-05-03T11:00:00+00:00"),
            ("2026-05-04T10:00:00+00:00", "2026-05-04T11:00:00+00:00"),
        ]
        trades = []
        tid = 1
        for buy_ts, sell_ts in ts_pairs:
            trades.append(_trade(tid, buy_ts, "E", "BUY", 1, 10.0)); tid += 1
            trades.append(_trade(tid, sell_ts, "E", "SELL", 1, 11.0)); tid += 1
        out = build_reentry_velocity(trades, recent_limit=2)
        assert out["n_gaps"] == 3
        assert len(out["recent_gaps"]) == 2
        # Newest first: re-entry into round-trip #4 (2026-05-04) should be top.
        assert out["recent_gaps"][0]["reentered_at"].startswith("2026-05-04")
