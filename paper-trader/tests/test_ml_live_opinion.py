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


class TestPunctuationTokenization:
    """The 2026-05-17 bug: raw str.split() left punctuation on tokens, so a
    sentence-final / comma-trailed sentiment word never exact-matched the
    vocab and the news-sentiment half of the advisory silently zeroed for
    the majority of real headlines (which almost always end on their catalyst
    verb). Fixed by tokenizing on [a-z]+ word boundaries (CLAUDE.md §15 —
    mirror the punctuation-tolerant backtest scorer)."""

    def test_trailing_punctuation_bullish_still_scores(self):
        # Every bullish word here carries trailing punctuation. Pre-fix:
        # {"nvda","surges,","hits","record."} → 0 vocab hits → sentiment 0.0
        # → score 0.0 ≤ threshold → HOLD. Post-fix: surges + record → BUY.
        articles = [{
            "id": 10,
            "title": "NVDA surges, hits record.",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"NVDA": 900.0},
        )
        assert op is not None
        assert op["action"] == "BUY"
        assert op["ticker"] == "NVDA"

    def test_trailing_punctuation_bearish_is_respected(self):
        # Mirror image: punctuation-trailed bearish words must now register a
        # negative sentiment so the name stays below the buy threshold.
        articles = [{
            "id": 11,
            "title": "AMD plunges! Guidance disappoints, analysts warn.",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": ["AMD"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"AMD": 150.0},
        )
        assert op is not None
        assert op["action"] == "HOLD"

    def test_exact_match_not_prefix_no_false_positive(self):
        # The fix tokenizes but keeps EXACT membership (not the backtest's
        # prefix match), so "mission"/"missionary" must NOT read as the
        # bearish "miss". A bullish-punctuated NVDA headline with such a word
        # still nets bullish → BUY (mission did not cancel it as bearish).
        articles = [{
            "id": 12,
            "title": "NVDA surges; mission-critical AI wins, record demand!",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(), watch_px={"NVDA": 900.0},
        )
        assert op is not None
        assert op["action"] == "BUY"
        assert op["ticker"] == "NVDA"


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


class TestRegimeAndUniverseGuards:
    """Locks the regime multiplier and watch_px universe guards — the two
    paths that *suppress* an otherwise-buyable signal."""

    def test_bear_regime_suppresses_borderline_buy(self):
        # A moderate bullish quant signal (RSI oversold alone → +1.5 adj)
        # clears the >1.0 buy threshold under bull regime_mult=1.0
        # (1.5 > 1.0 → BUY) but collapses under bear regime_mult=0.3
        # (1.5 × 0.3 = 0.45 < 1.0 → HOLD). Locks that regime_mult is
        # actually applied to the score, not just rendered in the label.
        quant = {
            "NVDA": {"rsi": 25.0},     # only oversold RSI → adj=+1.5
            "SPY": {"mom_20d": -5.0},  # bear regime (< -3.0)
        }
        op = strategy._ml_live_opinion(
            [], quant_sigs=quant, snap=_snap(),
            watch_px={"NVDA": 900.0, "SPY": 500.0},
        )
        assert op is not None
        assert op["action"] == "HOLD"
        assert "regime=bear" in op["reasoning"]

    def test_keyword_mapping_picks_up_unticked_article(self):
        # An article whose `tickers` list is empty but whose title contains a
        # mapped keyword (`_WORD_TO_TICKER_LIVE`) must still drive a BUY —
        # the keyword→ticker fallback IS the value-add of that map. "nvidia"
        # → NVDA is the canonical mapping; the bullish words `surges` /
        # `record` net bullish; raw_score 8.0 × +1.0 sentiment well clears
        # the 1.0 threshold.
        articles = [{
            "id": 100,
            "title": "nvidia surges to record on chip demand",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": [],  # the test: tickers extractor missed it
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"NVDA": 900.0, "SOXL": 30.0},
        )
        assert op is not None
        assert op["action"] == "BUY"
        # The mapping puts NVDA AND SOXL ("chip") in scope. Whichever wins,
        # it must be one of the keyword-mapped tickers — locks the path.
        assert op["ticker"] in ("NVDA", "SOXL")

    def test_unpriced_ticker_cannot_be_chosen(self):
        # `watch_px` is the live universe gate: a ticker with no live price
        # (yfinance dead / delisted / off-hours) must NOT be picked as best
        # even when its score is the highest. Without the `px and px > 0`
        # guard the engine would emit a BUY recommendation for a name the
        # trader cannot actually transact in.
        articles = [{
            "id": 200,
            "title": "NVDA surges to record on strong AI demand",
            "ai_score": 9.0,
            "urgency": 0,
            "tickers": ["NVDA"],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"NVDA": None, "AMD": 0.0},  # both unpriced
        )
        assert op is not None
        assert op["action"] == "HOLD"


class TestKeywordSubstringFalsePositives:
    """Locks that ``_WORD_TO_TICKER_LIVE`` matching uses word boundaries.

    A bare ``keyword in title`` substring match false-positively mapped
    short keys to their tickers on irrelevant articles:

      * "ai" → TQQQ matched "China" / "rain" / "Spain" / "trail"
      * "gold" → GLD matched "Goldman" (very common in finance news)
      * "intel" → INTC matched "intelligence" (and "artificial intelligence",
        double-counted with the "ai" → TQQQ map)

    Each silently boosted an unrelated ticker's score on every article
    containing the substring — a real signal-quality regression on the
    advisor (CLAUDE.md §15). The canonical recovery case in
    ``test_keyword_mapping_picks_up_unticked_article`` ("nvidia surges to
    record on chip demand") still matches under ``\\bkeyword\\b`` because
    the keyword appears as a standalone token — both paths must hold.
    """

    def test_rain_in_title_does_not_alias_to_tqqq_via_ai(self):
        # "rain" contains the substring "ai". Pre-fix: ``"ai" in "rain..."``
        # is True → TQQQ gets the article's raw_score * sentiment added.
        # Post-fix: ``\bai\b`` requires the standalone token "ai".
        # The article has no other watchlist ticker, no quant signal, no
        # mapped keyword that matches at a word boundary → expect HOLD.
        articles = [{
            "id": 300,
            "title": "Heavy rain surges to record level in strong storm",
            "ai_score": 9.0,
            "urgency": 0,
            "tickers": [],  # extractor would not surface RAIN as a ticker
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"TQQQ": 80.0, "NVDA": 900.0},
        )
        assert op is not None
        # Pre-fix this returned BUY TQQQ (substring "ai" in "rain" bullishly
        # routed the high-score article to TQQQ). Post-fix → HOLD.
        assert op["action"] == "HOLD", op

    def test_pain_in_title_does_not_alias_to_tqqq_via_ai(self):
        # Additional "ai" substring false-positive: "pain" → TQQQ pre-fix,
        # HOLD post-fix. Two different stems is the regression-locking
        # value-add over relying on one word's letter pattern alone.
        articles = [{
            "id": 302,
            "title": "Inflation pain surges to record strong levels",
            "ai_score": 9.0,
            "urgency": 0,
            "tickers": [],
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"TQQQ": 80.0},
        )
        assert op is not None
        assert op["action"] == "HOLD", op

    def test_standalone_ai_token_still_maps_to_tqqq(self):
        # The fix must NOT regress the canonical "AI" → TQQQ recovery — the
        # whole point of the keyword map. A title with "AI" as a standalone
        # token (after lowering) is exactly what the map exists to surface.
        articles = [{
            "id": 303,
            "title": "AI demand surges to record on strong outlook",
            "ai_score": 8.0,
            "urgency": 0,
            "tickers": [],  # extractor missed it (2-char "AI" filtered)
        }]
        op = strategy._ml_live_opinion(
            articles, quant_sigs={}, snap=_snap(),
            watch_px={"TQQQ": 80.0},
        )
        assert op is not None
        # Standalone "ai" → TQQQ (the recovery path the keyword map exists
        # for) — locks the keyword-mapping picks-up regression. Pre-fix
        # and post-fix BOTH match here, by design.
        assert op["action"] == "BUY"
        assert op["ticker"] == "TQQQ"
