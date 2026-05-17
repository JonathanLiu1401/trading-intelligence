"""Tests for strategy._ml_live_opinion — the gated ML advisory opinion.

This function had ZERO direct coverage and shipped a silent correctness bug:
it read ``a.get("score")`` from live articles, but signals.get_top_signals /
get_urgent_articles emit ``ai_score`` (never ``score``). So every article was
skipped (raw_score==0.0 < 1.0) and the news-sentiment half of the advisory was
dead — it degraded to quant-only, contradicting CLAUDE.md §15.

The first test is the red→green lock: a single bullish high-ai_score article,
no quant signals. Pre-fix it returned HOLD (news ignored); post-fix it returns
a sentiment-directed BUY. A test that didn't strip quant input would pass
either way and leave the bug uncovered.
"""
from __future__ import annotations

import paper_trader.strategy as strategy


def _snap():
    return {"cash": 1000.0, "total_value": 1000.0, "open_value": 0.0,
            "positions": []}


class TestNewsKeyRegression:
    """The core bug: live articles carry ai_score, not score."""

    def test_bullish_ai_score_article_drives_a_buy(self):
        # NVDA is on WATCHLIST. Bullish words: surges, record, strong.
        articles = [{
            "id": 1,
            "title": "NVDA surges to record high on strong AI demand",
            "ai_score": 8.0,           # the LIVE key — NOT "score"
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"NVDA": 900.0, "TQQQ": 80.0},
        )
        assert op is not None
        # Pre-fix this was HOLD (article skipped because a.get("score") is None).
        assert op["action"] == "BUY"
        assert op["ticker"] == "NVDA"

    def test_bearish_article_does_not_trigger_a_buy(self):
        # Sentiment direction must be respected: a bearish catalyst yields a
        # negative ticker score, which is below the buy threshold → HOLD.
        articles = [{
            "id": 2,
            "title": "AMD plunges on weak guidance miss",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": ["AMD"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"AMD": 150.0},
        )
        assert op is not None
        assert op["action"] == "HOLD"

    def test_score_key_still_works_as_fallback(self):
        # Backtest-shaped input (carries "score", no "ai_score") must still
        # score — the fix prefers ai_score but keeps score as a fallback.
        articles = [{
            "id": 3,
            "title": "NVDA rallies on record bookings",
            "score": 7.0,              # legacy/backtest key only
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"NVDA": 900.0},
        )
        assert op is not None
        assert op["action"] == "BUY"
        assert op["ticker"] == "NVDA"

    def test_low_score_article_is_skipped(self):
        # raw_score < 1.0 is filtered. With nothing else, no high-conviction
        # signal → HOLD (not a crash, not a spurious BUY).
        articles = [{
            "id": 4,
            "title": "NVDA surges on record demand",
            "ai_score": 0.4,
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"NVDA": 900.0},
        )
        assert op is not None
        assert op["action"] == "HOLD"


class TestRobustness:
    def test_empty_articles_no_quant_holds_with_regime(self):
        op = strategy._ml_live_opinion(
            [], quant_sigs={}, snap=_snap(), watch_px={},
        )
        assert op is not None
        assert op["action"] == "HOLD"
        assert "regime=" in op["reasoning"]

    def test_non_numeric_ai_score_is_tolerated(self):
        # A corrupt ai_score must not raise (the function never raises — it
        # returns None on failure, but a coercible-skip is the right behaviour).
        articles = [{
            "id": 5,
            "title": "NVDA surges record strong",
            "ai_score": "not-a-number",
            "urgency": "weird",
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"NVDA": 900.0},
        )
        assert op is not None
        assert op["action"] == "HOLD"   # bad score skipped, nothing else scores

    def test_quant_only_path_still_works(self):
        # Even with no news, a strongly oversold + positive-momentum quant
        # profile in a bull regime can clear the buy threshold. Locks that the
        # quant arm is independent of the (now-fixed) news arm.
        quant = {
            "NVDA": {"rsi": 25.0, "macd_signal": 0.5, "mom_5d": 6.0,
                     "mom_20d": 8.0, "bb_position": -1.5},
            "SPY": {"mom_20d": 5.0},   # bull regime (mom_20d > 3)
        }
        op = strategy._ml_live_opinion(
            [], quant_sigs=quant, snap=_snap(),
            watch_px={"NVDA": 900.0, "SPY": 500.0},
        )
        assert op is not None
        assert op["action"] == "BUY"
        assert op["ticker"] == "NVDA"
        assert "regime=bull" in op["reasoning"]
