"""Order-independent near-duplicate detection for syndicated articles.

``watchers/alert_dedup.py`` collapses the *urgent-alert batch* with an exact
first-8-token prefix signature — fast, but blind to word reordering and only
applied at alert time. ``ml/dedup.py`` is the complementary general-purpose
detector: a tunable token-set **Jaccard** similarity over normalized titles,
so two paraphrases of the same wire story collide even when their leading
words differ. Pure functions over the standard article-dict shape; no DB, no
LLM, no mutation of the input.

These tests pin specific computed values (not "no crash"): exact Jaccard
arithmetic, attribution/wire-prefix normalization, the order-independence that
distinguishes this from the prefix-key dedup, threshold behaviour, the
highest-score representative invariant, survivor order, and input immutability.
"""
from __future__ import annotations

import pytest

from ml.dedup import (
    dedupe_articles,
    is_near_duplicate,
    jaccard_similarity,
    normalize_title,
    title_tokens,
)


class TestNormalizeTitle:
    def test_strips_trailing_source_attribution(self):
        base = normalize_title("Federal Reserve cuts interest rates 25 basis points")
        assert normalize_title("Federal Reserve cuts interest rates 25 basis points - CNBC") == base
        assert normalize_title("Federal Reserve cuts interest rates 25 basis points | Bloomberg") == base
        assert normalize_title("Federal Reserve cuts interest rates 25 basis points (Reuters)") == base

    def test_strips_leading_wire_prefix(self):
        base = normalize_title("Micron shares surge after Q3 earnings blowout")
        assert normalize_title("UPDATE 2-Micron shares surge after Q3 earnings blowout") == base
        assert normalize_title("RPT-BREAKING: Micron shares surge after Q3 earnings blowout") == base

    def test_none_and_empty_are_empty_string(self):
        assert normalize_title(None) == ""
        assert normalize_title("") == ""
        assert normalize_title("   ") == ""


class TestTitleTokens:
    def test_drops_stopwords_and_short_tokens(self):
        # "on"/"a" are stopwords; single-char "x" is below min_len.
        assert title_tokens("Oil prices fall on a x supply concern") == {
            "oil",
            "prices",
            "fall",
            "supply",
            "concern",
        }

    def test_empty_title_has_no_tokens(self):
        assert title_tokens(None) == set()
        assert title_tokens("") == set()


class TestJaccardSimilarity:
    def test_identical_sets_is_one(self):
        assert jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets_is_zero(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap_exact_ratio(self):
        # |{b}| / |{a,b,c}| = 1/3
        assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_two_empty_sets_is_zero_not_error(self):
        assert jaccard_similarity(set(), set()) == 0.0


class TestIsNearDuplicate:
    def test_attribution_only_variants_are_duplicates(self):
        a = "Nvidia shares surge on record quarterly earnings"
        b = "Nvidia shares surge on record quarterly earnings - Reuters"
        assert is_near_duplicate(a, b) is True

    def test_word_reorder_is_caught_unlike_prefix_key(self):
        # Different leading words: a first-8-token prefix signature would NOT
        # collapse these. Token-set Jaccard does.
        a = "Apple beats Q2 earnings expectations"
        b = "Q2 earnings expectations beaten by Apple"
        # tokens(a)={apple,beats,q2,earnings,expectations}
        # tokens(b)={q2,earnings,expectations,beaten,apple}  ("by" is a stopword)
        # |inter|=4 ({apple,q2,earnings,expectations}); |union|=6 -> 0.667
        assert is_near_duplicate(a, b, threshold=0.6) is True

    def test_unrelated_headlines_not_duplicates(self):
        a = "Apple beats Q2 earnings expectations"
        b = "Oil prices fall on OPEC supply concerns"
        assert is_near_duplicate(a, b) is False

    def test_threshold_is_respected(self):
        a = "Apple beats Q2 earnings expectations"
        b = "Q2 earnings expectations beaten by Apple"  # ~0.667 similarity
        assert is_near_duplicate(a, b, threshold=0.9) is False
        assert is_near_duplicate(a, b, threshold=0.6) is True

    def test_empty_titles_are_never_duplicates(self):
        assert is_near_duplicate("", "") is False
        assert is_near_duplicate(None, "Apple beats Q2 earnings") is False
        assert is_near_duplicate("Apple beats Q2 earnings", None) is False


class TestDedupeArticles:
    def _fixture(self):
        return [
            {"title": "Nvidia shares surge on record earnings", "ai_score": 4.0},
            {"title": "Nvidia shares surge on record earnings - Reuters", "ai_score": 7.5},
            {"title": "Oil prices fall on OPEC supply concerns", "ai_score": 6.0},
            {"title": "Nvidia shares surge on record earnings (Bloomberg)", "ai_score": 5.0},
        ]

    def test_keeps_highest_score_representative_per_cluster(self):
        out = dedupe_articles(self._fixture())
        assert len(out) == 2
        # Cluster {0,1,3} collapses to the ai_score=7.5 copy.
        assert out[0]["ai_score"] == 7.5
        assert out[0]["title"] == "Nvidia shares surge on record earnings - Reuters"
        # Unrelated article survives untouched.
        assert out[1]["title"] == "Oil prices fall on OPEC supply concerns"

    def test_survivor_order_follows_first_cluster_appearance(self):
        out = dedupe_articles(self._fixture())
        # Nvidia cluster first appears at index 0, oil at index 2 -> Nvidia first.
        assert [a["title"].split()[0] for a in out] == ["Nvidia", "Oil"]

    def test_does_not_mutate_input(self):
        articles = self._fixture()
        dedupe_articles(articles)
        assert len(articles) == 4
        assert articles[0]["ai_score"] == 4.0

    def test_high_threshold_keeps_paraphrases_separate(self):
        articles = [
            {"title": "Apple beats Q2 earnings expectations", "ai_score": 3.0},
            {"title": "Q2 earnings expectations beaten by Apple", "ai_score": 9.0},
        ]
        # ~0.667 similarity < 0.99 -> both kept, original order preserved.
        out = dedupe_articles(articles, threshold=0.99)
        assert len(out) == 2
        assert [a["ai_score"] for a in out] == [3.0, 9.0]

    def test_empty_input_returns_empty_list(self):
        assert dedupe_articles([]) == []

    def test_missing_score_key_defaults_to_zero(self):
        articles = [
            {"title": "Tesla recalls 2 million vehicles over autopilot"},
            {"title": "Tesla recalls 2 million vehicles over autopilot - AP", "ai_score": 8.0},
        ]
        out = dedupe_articles(articles)
        assert len(out) == 1
        # The scored copy (8.0) wins over the unscored (treated as 0.0).
        assert out[0]["ai_score"] == 8.0
