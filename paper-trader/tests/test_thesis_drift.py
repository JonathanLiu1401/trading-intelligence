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
    REASON_CAP,
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


class TestPromptBlock:
    """The prompt_block is the live-trader-facing render — INTACT positions
    are silent (the chat-enrichment silence precedent), WEAKENING/BROKEN
    entries surface with verbatim entry_reason capped at REASON_CAP."""

    def test_empty_book_block_is_none(self):
        r = build_thesis_drift([], [], now=_NOW)
        assert r["prompt_block"] is None

    def test_all_intact_block_is_none(self):
        # +2% on both holdings, no quant adverse signals — INTACT.
        positions = [_pos("A", 100.0, 102.0, 3), _pos("B", 50.0, 51.0, 2)]
        trades = [_buy("A", "thesis A", 3, 100.0, tid=1),
                  _buy("B", "thesis B", 2, 50.0, tid=3)]
        r = build_thesis_drift(positions, trades, now=_NOW)
        assert r["counts"] == {"INTACT": 2, "WEAKENING": 0, "BROKEN": 0}
        assert r["prompt_block"] is None

    def test_weakening_surfaces_ticker_and_drift_reasons(self):
        # -5% since entry → WEAKENING; intact sibling must NOT appear.
        positions = [_pos("WEAK", 100.0, 95.0, 2),
                      _pos("FINE", 100.0, 103.0, 2)]
        trades = [_buy("WEAK", "bought WEAK on catalyst X", 2, 100.0, tid=1),
                  _buy("FINE", "bought FINE on catalyst Y", 2, 100.0, tid=3)]
        block = build_thesis_drift(positions, trades, now=_NOW)["prompt_block"]
        assert block is not None
        assert "WEAK" in block
        assert "WEAKENING" in block
        assert "bought WEAK on catalyst X" in block
        # The INTACT sibling must NOT leak into the lean prompt block.
        assert "FINE" not in block
        # Drift reasons must surface so Opus sees the *why*, not just a label.
        assert "P/L since entry" in block

    def test_broken_leads_and_overrides_weakening_in_block_order(self):
        # Broken (worst), Weakening, Intact. Block must lead with BROKEN
        # — mirroring the dict's positions[] ordering.
        positions = [
            _pos("INT", 100.0, 102.0, 2),
            _pos("BRK", 100.0, 88.0, 2),     # -12% BROKEN
            _pos("WK",  100.0, 96.0, 2),     # -4% WEAKENING
        ]
        trades = [
            _buy("INT", "thesis intact", 2, 100.0, tid=1),
            _buy("BRK", "thesis broken", 2, 100.0, tid=3),
            _buy("WK",  "thesis weakening", 2, 100.0, tid=5),
        ]
        block = build_thesis_drift(positions, trades, now=_NOW)["prompt_block"]
        assert block is not None
        # BRK must appear before WK in the block; INT must not appear at all.
        i_brk = block.find("BRK")
        i_wk = block.find("WK ")  # space to disambiguate from WEAKENING
        assert 0 <= i_brk < i_wk
        assert "INT " not in block
        assert "BROKEN" in block

    def test_entry_reason_truncated_past_cap(self):
        long_reason = "Y" * (REASON_CAP + 50)
        positions = [_pos("LONG", 100.0, 92.0, 2)]  # WEAKENING (-8 boundary)
        trades = [_buy("LONG", long_reason, 2, 100.0, tid=1)]
        block = build_thesis_drift(positions, trades, now=_NOW)["prompt_block"]
        assert block is not None
        assert long_reason not in block
        assert "…" in block
        # Cap-1 chars + ellipsis (the track_record truncation contract).
        assert ("Y" * (REASON_CAP - 1)) in block

    def test_block_carries_no_prescriptive_language(self):
        # Same silence-precedent as track_record / chat_lines: factual, not
        # directive. Reject the most common directive verbs the rest of the
        # codebase already takes pains to keep out of advisory blocks.
        positions = [_pos("X", 100.0, 90.0, 2)]  # -10% BROKEN
        trades = [_buy("X", "broken thesis X", 2, 100.0, tid=1)]
        block = build_thesis_drift(positions, trades, now=_NOW)["prompt_block"]
        assert block is not None
        low = block.lower()
        assert "you must" not in low
        assert "you should" not in low
        assert " sell " not in low.replace("\n", " ")  # no SELL directive
        # Autonomy preamble is required so Opus does not read it as a cap.
        assert "complete autonomy" in block.lower()
