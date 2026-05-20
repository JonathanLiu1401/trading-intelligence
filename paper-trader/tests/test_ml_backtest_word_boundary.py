"""Regression locks for the backtest-side word-boundary keyword mapping fix.

`paper_trader.backtest._ml_decide` used to do a naive `keyword in title_lower`
substring match for every entry in `_WORD_TO_TICKER`. That false-positively
mapped short keys to their tickers on irrelevant articles in exactly the same
way `strategy._ml_live_opinion` was previously broken (see
`tests/test_ml_live_opinion.TestKeywordSubstringFalsePositives`):

  * "ai"    → TQQQ matched "rain" / "training" / "blockchain" / "captain"
  * "gold"  → GLD matched "Goldman" (very common in finance headlines)
  * "intel" → INTC matched "intelligence" (and "artificial intelligence",
              which then double-counted with the "ai" → TQQQ map)
  * "oil"   → USO matched "spoiled" / "coil"

Each false positive silently boosted an unrelated ticker's `ticker_scores`
weight on every article containing the substring — distorting which ticker
the backtest's `_ml_decide` picked AND poisoning the `decision_outcomes.jsonl`
training corpus that retrains the DecisionScorer (the gate the live trader
eventually relies on). The fix mirrors `strategy._WORD_TO_TICKER_LIVE_PATTERNS`:
compile `\\bkw\\b` once and match via `Pattern.search`.

These tests lock the behaviour at the LOWEST level (the keyword pattern
dict) AND at the integration level (`_ml_decide`'s recommendation) so a
regression in either layer fails loudly.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

import paper_trader.backtest as bt
from paper_trader.backtest import (
    SimPortfolio,
    _WORD_TO_TICKER,
    _WORD_TO_TICKER_PATTERNS,
    _ml_decide,
)


class TestWordToTickerPatterns:
    """Pin the compiled patterns. A regression to naive `in` substring matching
    would trip these directly — independent of any _ml_decide change."""

    def test_patterns_exist_for_every_key(self):
        # Every entry in the source dict gets a compiled pattern.
        assert set(_WORD_TO_TICKER_PATTERNS.keys()) == set(_WORD_TO_TICKER.keys())

    def test_ai_does_not_match_rain(self):
        # Pre-fix: `"ai" in "rain..."` is True → false positive TQQQ.
        pat = _WORD_TO_TICKER_PATTERNS["ai"]
        assert pat.search("heavy rain surges to record levels") is None

    def test_ai_does_not_match_training(self):
        # The most common false positive — every ML/training news article.
        pat = _WORD_TO_TICKER_PATTERNS["ai"]
        assert pat.search("training data improves model accuracy") is None

    def test_ai_does_not_match_blockchain(self):
        # The crypto crossover false positive.
        pat = _WORD_TO_TICKER_PATTERNS["ai"]
        assert pat.search("blockchain raises new concerns over scaling") is None

    def test_ai_does_not_match_captain(self):
        pat = _WORD_TO_TICKER_PATTERNS["ai"]
        assert pat.search("us captain raises ire of regulators") is None

    def test_ai_matches_standalone_token(self):
        # Standalone token "ai" is exactly what the map exists to surface —
        # the recovery path for "AI demand surges" headlines that the
        # ticker extractor (2-char filter) misses.
        pat = _WORD_TO_TICKER_PATTERNS["ai"]
        assert pat.search("ai demand surges to record on strong outlook") is not None

    def test_gold_does_not_match_goldman(self):
        # "Goldman" is everywhere in finance news — pre-fix this aliased GLD
        # to every Goldman Sachs article.
        pat = _WORD_TO_TICKER_PATTERNS["gold"]
        assert pat.search("goldman sachs raises gdp forecast") is None

    def test_gold_matches_standalone_token(self):
        pat = _WORD_TO_TICKER_PATTERNS["gold"]
        assert pat.search("gold rallies as dollar weakens") is not None

    def test_intel_does_not_match_intelligence(self):
        # "intelligence" → "intel" substring → INTC false positive.
        # And "artificial intelligence" would double-trigger (also via "ai").
        pat = _WORD_TO_TICKER_PATTERNS["intel"]
        assert pat.search(
            "artificial intelligence drives chip demand higher"
        ) is None

    def test_oil_does_not_match_spoiled(self):
        pat = _WORD_TO_TICKER_PATTERNS["oil"]
        assert pat.search("spoiled investment thesis sinks stock") is None

    def test_oil_matches_standalone_token(self):
        pat = _WORD_TO_TICKER_PATTERNS["oil"]
        assert pat.search("oil prices spike on OPEC cut") is not None

    def test_multi_word_key_matches_phrase(self):
        # Multi-word keys still work because \b matches at every word/non-word
        # transition — including the space between tokens.
        pat = _WORD_TO_TICKER_PATTERNS["natural gas"]
        assert pat.search("natural gas inventories drop") is not None
        # And NOT inside an unrelated longer word.
        assert pat.search("supernatural gases (a 90s grunge band)") is None


class TestMlDecideKeywordIntegration:
    """End-to-end pins on `_ml_decide`'s decision when the keyword extractor
    is the only signal source. These exercise the full prompt-to-pick flow
    (_WORD_TO_TICKER → ticker_scores → buy_ticker)."""

    def test_rain_headline_does_not_buy_tqqq(self, synthetic_prices, monkeypatch):
        """Pre-fix this returned BUY TQQQ because "rain" contained "ai".
        Post-fix → HOLD (no recognized ticker, no quant history, no persona
        pick at this score)."""
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)
        p = SimPortfolio(cash=10_000.0)
        rng = random.Random(0)
        d = synthetic_prices.trading_days[-1]
        articles = [{
            "title": "Heavy rain delays harvests across midwest",
            "score": 9.0,
            "tickers": [],  # extractor would not surface RAIN as a ticker
        }]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        # The "rain" → "ai" → TQQQ false positive is removed.
        assert decision["ticker"] != "TQQQ", (
            "rain article must not alias to TQQQ via 'ai' substring "
            f"(pre-fix bug); got {decision}"
        )

    def test_blockchain_headline_does_not_buy_tqqq(
            self, synthetic_prices, monkeypatch):
        """Crypto crossover false positive: 'blockchain' contains 'ai'."""
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)
        p = SimPortfolio(cash=10_000.0)
        rng = random.Random(0)
        d = synthetic_prices.trading_days[-1]
        articles = [{
            "title": "Blockchain layer-2 protocol raises $50M Series B",
            "score": 9.0,
            "tickers": [],
        }]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        assert decision["ticker"] != "TQQQ", (
            "blockchain article must not alias to TQQQ via 'ai' substring; "
            f"got {decision}"
        )

    def test_goldman_headline_does_not_buy_gld(
            self, synthetic_prices, monkeypatch):
        """'Goldman' → 'gold' substring → GLD false positive."""
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)
        p = SimPortfolio(cash=10_000.0)
        rng = random.Random(0)
        d = synthetic_prices.trading_days[-1]
        articles = [{
            "title": "Goldman Sachs raises forecast for earnings season",
            "score": 9.0,
            "tickers": [],
        }]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        assert decision["ticker"] != "GLD", (
            "goldman article must not alias to GLD via 'gold' substring; "
            f"got {decision}"
        )

    def test_intelligence_headline_does_not_buy_intc(
            self, synthetic_prices, monkeypatch):
        """'intelligence' → 'intel' substring → INTC false positive."""
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)
        p = SimPortfolio(cash=10_000.0)
        rng = random.Random(0)
        d = synthetic_prices.trading_days[-1]
        articles = [{
            "title": "Business intelligence platform raises funding round",
            "score": 9.0,
            "tickers": [],
        }]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        assert decision["ticker"] != "INTC", (
            "intelligence article must not alias to INTC via 'intel' "
            f"substring; got {decision}"
        )

    def test_standalone_nvidia_token_still_maps_to_nvda(
            self, synthetic_prices, monkeypatch):
        """Regression guard: the canonical "nvidia" → NVDA recovery path
        must continue to work — this is the WHOLE POINT of the keyword map.
        Confirming the fix doesn't break legitimate matches."""
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)
        p = SimPortfolio(cash=10_000.0)
        rng = random.Random(0)
        d = synthetic_prices.trading_days[-1]
        articles = [{
            "title": "Nvidia surges to record on AI compute demand",
            "score": 8.0,
            "tickers": [],  # extractor missed it — keyword map must recover
        }]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        # Both "nvidia" → NVDA and "ai" → TQQQ are word-boundary matches here,
        # both legitimate. Either is acceptable; the test catches the case
        # where NEITHER is picked (which would indicate the patterns broke).
        assert decision["action"] == "BUY"
        assert decision["ticker"] in ("NVDA", "TQQQ"), (
            f"Expected NVDA or TQQQ from standalone-token map, got {decision}"
        )
