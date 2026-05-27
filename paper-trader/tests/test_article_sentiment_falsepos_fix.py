"""Agent 2 (ML+backtests) HYBRID pass — 2026-05-28 Phase 1.

Locks the ``_article_sentiment`` false-positive fix.

The bug:
    ``_article_sentiment`` used per-word ``startswith`` against the bullish /
    bearish stem sets. That silently flagged unrelated tokens as bearish:
    ``mission`` against ``miss``, ``missile`` against ``miss``, ``cute``
    against ``cut``. Verified live: ``"NVDA upgrade mission critical AI"``
    scored 0.0 instead of bullish because the upgrade vote was cancelled by
    a fake ``mission→miss`` bearish vote. Because ``_ml_decide`` multiplies
    per-article ``raw_score * sentiment`` directly into the per-ticker
    aggregate, the false negatives poisoned both the daily decision AND the
    ``decision_outcomes.jsonl`` row that retrains the DecisionScorer the
    gate eventually relies on.

The fix:
    Replaced ``startswith`` with a word-boundary regex that allows only a
    closed safe-suffix set; the legacy ``acqui`` stem (no safe-suffix match
    for ``-sition``) was replaced with explicit variants. The headline-common
    forms ``cutting/selling/buying/rallying`` (tails outside the safe-suffix
    set) were enumerated explicitly so the regex tail rejection doesn't
    lose those legitimate matches.
"""
from __future__ import annotations

import unittest


class TestArticleSentimentFalsePositives(unittest.TestCase):
    """Lock the documented false positives are NO LONGER reported as bearish."""

    def setUp(self):
        from paper_trader.backtest import _article_sentiment
        self.sentiment = _article_sentiment

    def test_mission_does_not_match_bearish_miss(self):
        # The article is bullish (upgrade). Pre-fix scored 0.0 because
        # ``mission`` was silently flagged bearish via ``startswith("miss")``.
        v = self.sentiment("NVDA upgrade mission critical AI")
        self.assertGreater(v, 0.0)

    def test_cute_does_not_match_bearish_cut(self):
        # Article is bullish (rally). Pre-fix scored 0.0 because ``cute``
        # flagged bearish via ``startswith("cut")``.
        v = self.sentiment("cute kitten gold rally")
        self.assertGreater(v, 0.0)

    def test_missile_does_not_match_bearish_miss(self):
        # Bullish article (higher). Pre-fix scored 0.0 because
        # ``missile`` flagged bearish via ``startswith("miss")``.
        v = self.sentiment("missile threat sends defense higher")
        self.assertGreater(v, 0.0)


class TestArticleSentimentLegitimateMatches(unittest.TestCase):
    """Lock that the regex rewrite preserves the legitimate semantic matches
    the legacy ``startswith`` approach captured."""

    def setUp(self):
        from paper_trader.backtest import _article_sentiment
        self.sentiment = _article_sentiment

    def test_beats_surges_higher_is_strongly_bullish(self):
        # Three bullish stems hit, no bearish — sentiment should be +1.0.
        v = self.sentiment("NVDA beats earnings, surges higher")
        self.assertEqual(v, 1.0)

    def test_misses_plunges_lower_is_strongly_bearish(self):
        v = self.sentiment("NVDA misses earnings, plunges lower")
        self.assertEqual(v, -1.0)

    def test_acquisition_explicit_variants_still_match(self):
        # The legacy ``"acqui"`` stem matched ``acquisition`` via startswith;
        # the regex word boundary rejects it (``acqui`` + non-safe ``sition``)
        # so the explicit variants must be enumerated.
        for w in ("acquisition", "acquisitions", "acquire",
                  "acquires", "acquired", "acquiring"):
            v = self.sentiment(f"Apple {w} startup announced")
            self.assertGreater(v, 0.0, f"failed for word {w!r}")

    def test_cutting_buying_selling_explicit_variants_match(self):
        # Tails ``-ting`` / ``-ying`` lie outside the safe-suffix set
        # (``mission`` / ``cute`` defense). Explicit enumeration keeps
        # these common forms producing the correct sentiment.
        self.assertLess(self.sentiment("Fed cutting rates"), 0.0)
        self.assertLess(self.sentiment("Hedge fund selling tech"), 0.0)
        self.assertGreater(self.sentiment("Hedge fund buying tech"), 0.0)

    def test_legacy_existing_tests_still_pass(self):
        # The three legacy ``TestArticleSentiment`` cases from
        # ``test_backtest.py`` must continue to hold under the rewrite.
        self.assertGreater(self.sentiment(
            "NVDA beats earnings, surges higher"), 0.0)
        self.assertLess(self.sentiment(
            "NVDA misses earnings, plunges lower"), 0.0)
        self.assertEqual(self.sentiment("some random headline"), 0.0)


class TestArticleSentimentEdgeCases(unittest.TestCase):
    """Defensive contract: never crashes on missing / pathological input."""

    def setUp(self):
        from paper_trader.backtest import _article_sentiment
        self.sentiment = _article_sentiment

    def test_empty_string_returns_zero(self):
        self.assertEqual(self.sentiment(""), 0.0)

    def test_none_returns_zero(self):
        self.assertEqual(self.sentiment(None), 0.0)

    def test_only_punctuation_returns_zero(self):
        self.assertEqual(self.sentiment("...!!!"), 0.0)

    def test_punctuation_attached_to_word_still_matched(self):
        # "beats!" should still match — the regex's ``\b`` boundary handles
        # punctuation correctly (whereas the original ``set(title.split())``
        # would have left "beats!" as the token and ``startswith`` would have
        # matched against the comma-suffixed variant).
        self.assertGreater(self.sentiment("NVDA beats!"), 0.0)

    def test_duplicate_words_dedup_to_one_vote(self):
        # set-based dedup ensures the same stem repeated in the title doesn't
        # multi-vote. ``"beats beats beats"`` should sentiment the same as
        # ``"beats"``.
        v_one = self.sentiment("NVDA beats earnings")
        v_three = self.sentiment("NVDA beats beats beats earnings")
        self.assertEqual(v_one, v_three)


class TestSentimentRegexInternals(unittest.TestCase):
    """Lock the structural properties of the compiled regex so a future
    refactor cannot silently reintroduce the ``startswith``-class bug."""

    def test_bullish_regex_has_word_boundaries(self):
        from paper_trader.backtest import _BULLISH_RE
        self.assertIn(r"\b", _BULLISH_RE.pattern)

    def test_bearish_regex_has_word_boundaries(self):
        from paper_trader.backtest import _BEARISH_RE
        self.assertIn(r"\b", _BEARISH_RE.pattern)

    def test_bullish_regex_alternatives_sorted_longest_first(self):
        # Python's ``re`` is leftmost-first. Longest-first alternation
        # prevents a shorter prefix (``buy``) from committing on a longer
        # token (``buying``) when the backtrack to add the optional suffix
        # is unnecessary. Verified by checking that ``buying`` matches
        # ``buying`` (not just ``buy``).
        from paper_trader.backtest import _BULLISH_RE
        match = _BULLISH_RE.search("hedge fund buying tech")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(0), "buying")

    def test_legacy_acqui_stem_was_replaced_with_explicit_variants(self):
        from paper_trader.backtest import _BULLISH_WORDS
        self.assertNotIn("acqui", _BULLISH_WORDS)
        for w in ("acquire", "acquires", "acquired", "acquiring",
                  "acquisition", "acquisitions"):
            self.assertIn(w, _BULLISH_WORDS)


class TestArticleSentimentMlScoreScale(unittest.TestCase):
    """The sentiment value is used as a multiplier on the per-article
    ``raw_score`` inside ``_ml_decide``. The fix matters because the
    prior false-positive ``mission/cute`` matches silently zeroed out
    otherwise-positive contributions, poisoning per-ticker score
    aggregation and downstream DecisionScorer training data."""

    def test_strongly_bullish_article_multiplies_score_positively(self):
        from paper_trader.backtest import _article_sentiment
        v = _article_sentiment("NVDA upgrade mission critical AI surges")
        self.assertGreater(v, 0.0)
        raw_score = 4.0
        product = raw_score * v
        self.assertGreater(product, 0.0)

    def test_strongly_bearish_article_multiplies_score_negatively(self):
        from paper_trader.backtest import _article_sentiment
        v = _article_sentiment("TSLA missing guidance, plunges crashing")
        self.assertLess(v, 0.0)
        raw_score = 4.0
        product = raw_score * v
        self.assertLess(product, 0.0)


if __name__ == "__main__":
    unittest.main()
