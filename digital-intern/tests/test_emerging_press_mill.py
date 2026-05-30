"""Unit tests for analytics/emerging_press_mill.py.

Every test asserts SPECIFIC values (not just "no crash") — predicate truthiness,
bucket counts, verdict, mean_ml_score arithmetic, top-N source ordering.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analytics.emerging_press_mill import (
    EMERGING_PREDICATES,
    build_emerging_press_mill,
    is_foreign_pr_newswire,
    is_ownership_disclosed_press_mill,
)


# ─────────────────────────────────────────────────────────────────────────────
# Predicate 1: ownership_disclosed_press_mill
# ─────────────────────────────────────────────────────────────────────────────

class TestOwnershipDisclosedPredicate:
    """The Stock Titan 13D/13G filing-summary template:
    "FMR LLC (COHR) reports 22.6M shares, 12.1% ownership disclosed - Stock Titan"
    Must match. Real news with overlapping tokens must NOT match.
    """

    def test_fmr_live_evidence_matches(self):
        # Verbatim live-DB title from 2026-05-29 23:42:26Z (urgency=2 fire).
        art = {
            "title": "FMR LLC (COHR) reports 22.6M shares, 12.1% ownership "
                     "disclosed - Stock Titan",
            "source": "GoogleNews/Stock Titan",
        }
        assert is_ownership_disclosed_press_mill(art) is True

    def test_variant_with_billion_shares_matches(self):
        art = {
            "title": "Vanguard Group Inc (NVDA) reports 1.2B shares, 5.3% "
                     "ownership disclosed - Stock Titan",
            "source": "GoogleNews/Stock Titan",
        }
        assert is_ownership_disclosed_press_mill(art) is True

    def test_no_parens_does_not_match(self):
        # Same verbs but no parenthesised ticker — different headline class.
        art = {
            "title": "FMR reports 22M shares of new ownership disclosed",
            "source": "GoogleNews/Stock Titan",
        }
        assert is_ownership_disclosed_press_mill(art) is False

    def test_no_disclosed_keyword_does_not_match(self):
        # Parenthesised ticker + reports + shares but no "ownership disclosed".
        art = {
            "title": "Apple Inc (AAPL) reports 220M shares outstanding",
            "source": "GoogleNews/Bloomberg",
        }
        assert is_ownership_disclosed_press_mill(art) is False

    def test_real_news_with_disclosed_word_does_not_match(self):
        # "ownership disclosed" appears in non-template context — but the
        # discriminator requires parens-ticker + reports + shares + the
        # "ownership disclosed" trailer together.
        art = {
            "title": "Insider ownership disclosed in 13G filing for chip "
                     "stocks",
            "source": "Reuters",
        }
        assert is_ownership_disclosed_press_mill(art) is False

    def test_missing_title_does_not_raise(self):
        # Defensive: never raise on a malformed row.
        assert is_ownership_disclosed_press_mill({}) is False
        assert is_ownership_disclosed_press_mill({"title": None}) is False
        assert is_ownership_disclosed_press_mill({"title": ""}) is False

    def test_non_dict_input_does_not_raise(self):
        # The builder feeds dicts but defensive predicate must tolerate junk.
        assert is_ownership_disclosed_press_mill(None) is False
        assert is_ownership_disclosed_press_mill("string") is False
        assert is_ownership_disclosed_press_mill(123) is False


# ─────────────────────────────────────────────────────────────────────────────
# Predicate 2: foreign_pr_newswire
# ─────────────────────────────────────────────────────────────────────────────

class TestForeignPrNewswirePredicate:
    """Multi-language wire-aggregator releases — Spanish/German/French/CJK.
    Must match. English wire releases and non-PR-Newswire foreign content
    must NOT match (preserves real reporting on foreign topics)."""

    def test_spanish_arasan_live_evidence_matches(self):
        art = {
            "title": "Arasan Chip Systems anuncia la primera solución IP "
                     "Sureboot Total de 16 bits xSPI + PSRAM",
            "source": "PR Newswire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_german_arasan_live_evidence_matches(self):
        art = {
            "title": "Arasan Chip Systems kündigt branchenweit erste "
                     "Sureboot Total 16-bit xSPI + PSRAM IP-Lösung",
            "source": "PR Newswire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_french_arasan_live_evidence_matches(self):
        art = {
            "title": "Arasan Chip Systems annonce la première solution IP "
                     "Sureboot Total 16 bits xSPI + PSRAM.",
            "source": "PR Newswire Tech",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_italian_announce_verb_matches(self):
        art = {
            "title": "Acme Corp annuncia il nuovo prodotto",
            "source": "GlobeNewswire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_cjk_title_matches(self):
        art = {
            "title": "アラサン・チップ・システムズが新製品を発表",
            "source": "PR Newswire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_english_pr_newswire_does_not_match(self):
        # The English version of the same release — must pass through (this is
        # the canonical headline the analyst SHOULD see).
        art = {
            "title": "Arasan Chip Systems announces first Sureboot Total "
                     "16-bit xSPI + PSRAM IP solution",
            "source": "PR Newswire",
        }
        assert is_foreign_pr_newswire(art) is False

    def test_foreign_topic_from_reuters_does_not_match(self):
        # Real reporting on a foreign topic carries non-ASCII characters but
        # NOT a wire-aggregator source — must pass through.
        art = {
            "title": "Macron meets Xi at G20 summit on trade tensions",
            "source": "Reuters",
        }
        assert is_foreign_pr_newswire(art) is False

    def test_non_aggregator_source_with_foreign_markers_does_not_match(self):
        # Hypothetical non-English wire — but the source-prefix gate is what
        # SOURCEs the false-positive risk. A Reuters article in Spanish would
        # NOT be a press-mill duplicate; it would be real reporting.
        art = {
            "title": "Acme Corp anuncia un nuevo producto",
            "source": "Reuters Espanol",
        }
        assert is_foreign_pr_newswire(art) is False

    def test_source_case_insensitive(self):
        # The wire-aggregator prefix match is case-insensitive.
        art = {
            "title": "Acme Corp anuncia un nuevo producto",
            "source": "pr newswire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_business_wire_variant_matches(self):
        art = {
            "title": "Empresa kündigt nuevo Produkt an",
            "source": "BusinessWire",
        }
        assert is_foreign_pr_newswire(art) is True

    def test_missing_title_or_source_does_not_raise(self):
        assert is_foreign_pr_newswire({}) is False
        assert is_foreign_pr_newswire({"title": None, "source": None}) is False
        assert is_foreign_pr_newswire(
            {"title": "anuncia algo", "source": ""}
        ) is False

    def test_non_dict_input_does_not_raise(self):
        assert is_foreign_pr_newswire(None) is False
        assert is_foreign_pr_newswire("string") is False

    def test_announced_english_past_tense_does_not_match(self):
        # The English "announced" word starts with "ann-" like "annonce" but
        # is a different stem — word-boundary match prevents conflation.
        art = {
            "title": "Acme Corp announced new product release",
            "source": "PR Newswire",
        }
        assert is_foreign_pr_newswire(art) is False


# ─────────────────────────────────────────────────────────────────────────────
# Builder: build_emerging_press_mill
# ─────────────────────────────────────────────────────────────────────────────

class TestBuilder:

    def test_empty_input_returns_no_data(self):
        result = build_emerging_press_mill([])
        assert result["verdict"] == "NO_DATA"
        assert result["n_audited"] == 0
        assert result["n_emerging_caught"] == 0
        assert result["n_uncaught"] == 0
        assert result["by_predicate"]["ownership_disclosed_press_mill"]["count"] == 0
        assert result["by_predicate"]["foreign_pr_newswire"]["count"] == 0
        assert result["by_uncaught_source"] == []

    def test_all_gated_when_no_predicate_fires(self):
        rows = [
            {"title": "Apple Q1 beats", "source": "Reuters", "ml_score": 9.0},
            {"title": "Fed cuts rates 25bp", "source": "Bloomberg", "ml_score": 9.5},
            {"title": "MU upgraded to buy", "source": "Finnhub", "ml_score": 8.5},
        ]
        result = build_emerging_press_mill(rows)
        assert result["verdict"] == "ALL_GATED"
        assert result["n_audited"] == 3
        assert result["n_emerging_caught"] == 0
        assert result["n_uncaught"] == 3
        # Reuters/Bloomberg/Finnhub each carry 1 uncaught row.
        sources = {s["source"]: s["count"] for s in result["by_uncaught_source"]}
        assert sources == {"Reuters": 1, "Bloomberg": 1, "Finnhub": 1}

    def test_emerging_noise_verdict_with_ownership_match(self):
        rows = [
            {
                "title": "FMR LLC (COHR) reports 22.6M shares, 12.1% "
                         "ownership disclosed - Stock Titan",
                "source": "GoogleNews/Stock Titan",
                "ml_score": 9.96,
            },
            {"title": "Apple Q1 beats", "source": "Reuters", "ml_score": 9.0},
        ]
        result = build_emerging_press_mill(rows)
        assert result["verdict"] == "EMERGING_NOISE"
        assert result["n_audited"] == 2
        assert result["n_emerging_caught"] == 1
        assert result["n_uncaught"] == 1
        pm = result["by_predicate"]["ownership_disclosed_press_mill"]
        assert pm["count"] == 1
        assert pm["mean_ml_score"] == 9.96
        assert len(pm["sample_titles"]) == 1
        assert "FMR LLC (COHR)" in pm["sample_titles"][0]

    def test_three_arasan_cross_language_duplicates_all_caught(self):
        # The live-DB scenario: 3 Arasan press releases in Spanish/German/French
        # all fired urgency=2 within a minute. Predicate must catch all three.
        rows = [
            {
                "title": "Arasan Chip Systems anuncia la primera solución",
                "source": "PR Newswire",
                "ml_score": 9.97,
            },
            {
                "title": "Arasan Chip Systems kündigt branchenweit erste",
                "source": "PR Newswire",
                "ml_score": 9.9,
            },
            {
                "title": "Arasan Chip Systems annonce la première solution",
                "source": "PR Newswire Tech",
                "ml_score": 9.95,
            },
        ]
        result = build_emerging_press_mill(rows)
        assert result["verdict"] == "EMERGING_NOISE"
        assert result["n_emerging_caught"] == 3
        assert result["n_uncaught"] == 0
        fpr = result["by_predicate"]["foreign_pr_newswire"]
        assert fpr["count"] == 3
        # (9.97 + 9.9 + 9.95) / 3 = 9.940
        assert fpr["mean_ml_score"] == pytest.approx(9.94, abs=1e-3)
        assert len(fpr["sample_titles"]) == 3

    def test_mixed_batch_both_predicates_fire(self):
        rows = [
            # Ownership-disclosed press mill (1 row).
            {
                "title": "FMR LLC (COHR) reports 22.6M shares, 12.1% "
                         "ownership disclosed",
                "source": "Stock Titan",
                "ml_score": 9.0,
            },
            # Foreign PR Newswire (2 rows, different languages).
            {
                "title": "Empresa anuncia nuevo producto",
                "source": "PR Newswire",
                "ml_score": 8.0,
            },
            {
                "title": "Firma kündigt Produkt an",
                "source": "PR Newswire",
                "ml_score": 9.0,
            },
            # Real news (1 row — uncaught by either predicate).
            {
                "title": "Fed cuts rates 50bp emergency",
                "source": "Bloomberg",
                "ml_score": 9.5,
            },
        ]
        result = build_emerging_press_mill(rows)
        assert result["verdict"] == "EMERGING_NOISE"
        assert result["n_audited"] == 4
        assert result["n_emerging_caught"] == 3
        assert result["n_uncaught"] == 1
        assert result["by_predicate"]["ownership_disclosed_press_mill"]["count"] == 1
        assert result["by_predicate"]["foreign_pr_newswire"]["count"] == 2
        # The single uncaught row is the Bloomberg Fed cut.
        assert len(result["by_uncaught_source"]) == 1
        assert result["by_uncaught_source"][0]["source"] == "Bloomberg"

    def test_uncaught_sources_ranked_by_count_descending(self):
        rows = [
            {"title": "Apple beats", "source": "Reuters", "ml_score": 8.0},
            {"title": "MU upgrades", "source": "Reuters", "ml_score": 7.0},
            {"title": "Fed action", "source": "Reuters", "ml_score": 9.0},
            {"title": "NVDA earnings", "source": "Bloomberg", "ml_score": 8.5},
            {"title": "Macro print", "source": "WSJ", "ml_score": 8.0},
        ]
        result = build_emerging_press_mill(rows)
        # Reuters=3, Bloomberg=1, WSJ=1.
        srcs = result["by_uncaught_source"]
        assert srcs[0] == {"source": "Reuters", "count": 3, "mean_ml_score": 8.0}
        assert srcs[1]["count"] == 1
        assert srcs[2]["count"] == 1
        # Ties broken alphabetically: Bloomberg before WSJ.
        assert srcs[1]["source"] == "Bloomberg"
        assert srcs[2]["source"] == "WSJ"

    def test_max_uncaught_sources_caps_output(self):
        rows = [
            {"title": f"News {i}", "source": f"src{i}", "ml_score": 5.0}
            for i in range(10)
        ]
        result = build_emerging_press_mill(rows, max_uncaught_sources=3)
        assert len(result["by_uncaught_source"]) == 3

    def test_max_samples_per_pattern_caps_titles(self):
        # Use letter-only tickers — real Stock Titan tags follow this convention
        # (FMR / COHR / NVDA / ...). The discriminator regex matches
        # `\([A-Z]{1,6}\)` so a digit-bearing test ticker (TKR0) would not
        # match the predicate at all.
        suffixes = ["A", "B", "C", "D", "E"]
        rows = [
            {
                "title": f"Fund LLC (TKR{sfx}) reports 1M shares, "
                         f"5% ownership disclosed",
                "source": "Stock Titan",
                "ml_score": 9.0,
            }
            for sfx in suffixes
        ]
        result = build_emerging_press_mill(rows, max_samples_per_pattern=2)
        pm = result["by_predicate"]["ownership_disclosed_press_mill"]
        assert pm["count"] == 5
        assert len(pm["sample_titles"]) == 2

    def test_non_dict_rows_silently_skipped(self):
        rows = [
            {"title": "Apple beats", "source": "Reuters", "ml_score": 8.0},
            None,           # skipped
            "not a dict",  # skipped
            123,           # skipped
        ]
        result = build_emerging_press_mill(rows)
        assert result["n_audited"] == 1

    def test_malformed_ml_score_does_not_raise(self):
        rows = [
            {"title": "Apple beats", "source": "Reuters", "ml_score": "junk"},
            {"title": "MU upgrades", "source": "Reuters", "ml_score": None},
            {"title": "NVDA news", "source": "Bloomberg"},  # no ml_score
        ]
        result = build_emerging_press_mill(rows)
        assert result["n_audited"] == 3
        # All three uncaught, ml scores defaulted to 0.0 → mean 0.0.
        srcs = {s["source"]: s["mean_ml_score"] for s in result["by_uncaught_source"]}
        assert srcs["Reuters"] == 0.0
        assert srcs["Bloomberg"] == 0.0

    def test_now_override_in_as_of(self):
        fixed = datetime(2026, 5, 29, 17, 0, 0, tzinfo=timezone.utc)
        result = build_emerging_press_mill([], now=fixed)
        assert result["as_of"] == "2026-05-29T17:00:00+00:00"

    def test_naive_now_normalised_to_utc(self):
        # A naive datetime must be treated as UTC.
        naive = datetime(2026, 5, 29, 17, 0, 0)
        result = build_emerging_press_mill([], now=naive)
        assert result["as_of"].endswith("+00:00")

    def test_predicate_count_matches_constant_length(self):
        # Drift-lock: EMERGING_PREDICATES has exactly the predicates the
        # envelope's by_predicate map enumerates. Any added predicate must
        # update both EMERGING_PREDICATES and tests/test_emerging_press_mill.
        result = build_emerging_press_mill([])
        assert set(result["by_predicate"].keys()) == {
            label for label, _ in EMERGING_PREDICATES
        }
        assert len(EMERGING_PREDICATES) == 2

    def test_first_match_rule(self):
        # If a row could match more than one predicate, the FIRST predicate in
        # EMERGING_PREDICATES wins. Construct an ambiguous row (parenthesised
        # ticker + 'reports' + ... + 'ownership disclosed' AND foreign markers
        # + PR Newswire source). The first predicate ('ownership_disclosed_
        # press_mill') wins.
        art = {
            "title": "FMR LLC (NVDA) reports 1M shares, 5% ownership "
                     "disclosed — anuncia",
            "source": "PR Newswire",
            "ml_score": 9.0,
        }
        result = build_emerging_press_mill([art])
        assert result["by_predicate"]["ownership_disclosed_press_mill"]["count"] == 1
        assert result["by_predicate"]["foreign_pr_newswire"]["count"] == 0
