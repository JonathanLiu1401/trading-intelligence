"""Tests for analytics/thesis_drift.py — entry-thesis vs current reality.

Deterministic: the verdict is driven only by P/L since entry, hold time
and (optional) supplied quant/news. The opening BUY reason is surfaced
**verbatim**. A health verdict that ignores the −8% pain line, a
WEAKENING that doesn't fire on a hot RSI / MACD flip, a wrong opening-fill
selection on a re-entered lot (invariant #8), or a mangled entry reason
all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.thesis_drift import (
    PAIN_PCT,
    build_thesis_drift,
)

_NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


def _ts(days_before: float) -> str:
    return (_NOW - timedelta(days=days_before)).isoformat()


def _pos(ticker, avg, cur, opened_days_ago, qty=1.0,
         typ="stock", strike=None, expiry=None):
    return {
        "id": 1, "ticker": ticker, "type": typ, "qty": qty,
        "avg_cost": avg, "current_price": cur, "strike": strike,
        "expiry": expiry, "opened_at": _ts(opened_days_ago),
        "closed_at": None, "unrealized_pl": (cur - avg) * qty,
    }


def _buy(ticker, reason, days_before, price, tid=1,
         typ=None, strike=None, expiry=None):
    return {
        "id": tid, "timestamp": _ts(days_before), "ticker": ticker,
        "action": "BUY", "qty": 1.0, "price": price, "value": price,
        "reason": reason, "strike": strike, "expiry": expiry,
        "option_type": typ,
    }


class TestNoData:
    def test_empty_is_no_data(self):
        r = build_thesis_drift([], [])
        assert r["state"] == "NO_DATA"
        assert r["n_positions"] == 0
        assert "no open positions" in r["headline"].lower()


class TestHealthVerdict:
    def test_intact_when_up_and_signals_benign(self):
        pos = _pos("NVDA", avg=100.0, cur=105.0, opened_days_ago=4)
        trades = [_buy("NVDA", "golden cross + RSI room to run", 4, 100.0)]
        sig = {"NVDA": {"rsi": 60.0, "macd": "bullish",
                        "mom_5d": 3.2, "news_count": 2}}
        r = build_thesis_drift([pos], trades, sig, now=_NOW)
        c = r["positions"][0]
        assert c["health"] == "INTACT"
        assert c["pl_pct"] == 5.0
        assert c["entry_reason"] == "golden cross + RSI room to run"
        assert c["days_held"] == 4.0
        assert c["signals_present"] is True

    def test_broken_via_pain_threshold_regardless_of_signals(self):
        pos = _pos("LITE", avg=1000.0, cur=900.0, opened_days_ago=1)  # -10%
        trades = [_buy("LITE", "earnings beat, urgent 8.0", 1, 1000.0)]
        sig = {"LITE": {"rsi": 55.0, "macd": "bullish", "mom_5d": 1.0,
                        "news_count": 3}}
        r = build_thesis_drift([pos], trades, sig, now=_NOW)
        c = r["positions"][0]
        assert c["pl_pct"] == -10.0
        assert c["pl_pct"] <= PAIN_PCT
        assert c["health"] == "BROKEN"

    def test_broken_via_macd_flip_plus_negative_mom_and_loss(self):
        # only -1% (not past WEAK), but MACD bearish + 5d negative + losing.
        pos = _pos("AMD", avg=100.0, cur=99.0, opened_days_ago=2)
        trades = [_buy("AMD", "bullish MACD breakout", 2, 100.0)]
        sig = {"AMD": {"rsi": 50.0, "macd": "bearish", "mom_5d": -2.0,
                       "news_count": 1}}
        r = build_thesis_drift([pos], trades, sig, now=_NOW)
        assert r["positions"][0]["health"] == "BROKEN"

    def test_weakening_via_soft_loss_no_signals(self):
        pos = _pos("MU", avg=100.0, cur=96.0, opened_days_ago=3)  # -4%
        trades = [_buy("MU", "DRAM upcycle", 3, 100.0)]
        r = build_thesis_drift([pos], trades, signals=None, now=_NOW)
        c = r["positions"][0]
        assert c["pl_pct"] == -4.0
        assert c["health"] == "WEAKENING"
        assert c["signals_present"] is False

    def test_weakening_via_hot_rsi_even_when_green(self):
        pos = _pos("TQQQ", avg=50.0, cur=51.0, opened_days_ago=1)  # +2%
        trades = [_buy("TQQQ", "momentum continuation", 1, 50.0)]
        sig = {"TQQQ": {"rsi": 82.0, "macd": "bullish", "mom_5d": 4.0,
                        "news_count": 1}}
        r = build_thesis_drift([pos], trades, sig, now=_NOW)
        c = r["positions"][0]
        assert c["health"] == "WEAKENING"
        assert any("overextended" in s for s in c["drift_reasons"])

    def test_weakening_via_news_catalyst_gone_cold(self):
        # entry rationale cited a catalyst; live news_count now 0.
        pos = _pos("LNOK", avg=10.0, cur=10.1, opened_days_ago=2)  # +1%
        trades = [_buy("LNOK", "urgent earnings beat headline", 2, 10.0)]
        sig = {"LNOK": {"rsi": 55.0, "macd": "bullish", "mom_5d": 0.5,
                        "news_count": 0}}
        r = build_thesis_drift([pos], trades, sig, now=_NOW)
        c = r["positions"][0]
        assert c["health"] == "WEAKENING"
        assert any("news catalyst" in s for s in c["drift_reasons"])


class TestOpeningFillSelection:
    def test_picks_opener_nearest_opened_at_not_prior_closed_lot(self):
        # Invariant #8: a prior fully-closed NVDA lot's BUY is far earlier;
        # opened_at was reset to the re-entry. The opener of *this* lot is
        # the BUY whose timestamp is nearest opened_at.
        pos = _pos("NVDA", avg=900.0, cur=950.0, opened_days_ago=2)
        trades = [
            _buy("NVDA", "OLD prior-lot thesis", days_before=30,
                 price=400.0, tid=1),
            _buy("NVDA", "RE-ENTRY: confirmed catalyst, golden cross",
                 days_before=2, price=900.0, tid=9),
        ]
        r = build_thesis_drift([pos], trades, now=_NOW)
        c = r["positions"][0]
        assert c["entry_reason"] == "RE-ENTRY: confirmed catalyst, golden cross"
        assert c["entry_price"] == 900.0

    def test_entry_reason_surfaced_verbatim(self):
        long_reason = (
            "LITE posted an earnings beat with the top-scored urgent signal "
            "(8.0) and 8.0 news sentiment, yet dipped premarket — a buyable "
            "overreaction. RSI 56, MACD bullish, golden cross, mom_5d +8%.")
        pos = _pos("LITE", avg=970.0, cur=975.0, opened_days_ago=1)
        trades = [_buy("LITE", long_reason, 1, 970.0)]
        r = build_thesis_drift([pos], trades, now=_NOW)
        assert r["positions"][0]["entry_reason"] == long_reason

    def test_missing_opening_trade_degrades_not_errors(self):
        pos = _pos("ASML", avg=800.0, cur=820.0, opened_days_ago=5)
        r = build_thesis_drift([pos], [], now=_NOW)  # no ledger rows
        c = r["positions"][0]
        assert c["entry_reason"] is None
        assert c["entry_price"] == 800.0   # falls back to avg_cost
        assert c["health"] == "INTACT"     # +2.5%, no adverse signal


class TestAggregation:
    def test_sorted_worst_first_and_counts(self):
        positions = [
            _pos("A", 100.0, 105.0, 4),   # +5%  INTACT
            _pos("B", 100.0, 88.0, 2),    # -12% BROKEN
            _pos("C", 100.0, 96.0, 3),    # -4%  WEAKENING
        ]
        trades = [
            _buy("A", "thesis A", 4, 100.0, tid=1),
            _buy("B", "thesis B", 2, 100.0, tid=3),
            _buy("C", "thesis C", 3, 100.0, tid=5),
        ]
        r = build_thesis_drift(positions, trades, now=_NOW)
        assert r["n_positions"] == 3
        assert r["counts"] == {"INTACT": 1, "WEAKENING": 1, "BROKEN": 1}
        assert r["positions"][0]["ticker"] == "B"   # broken, worst P/L first
        assert r["positions"][-1]["ticker"] == "A"  # intact last
