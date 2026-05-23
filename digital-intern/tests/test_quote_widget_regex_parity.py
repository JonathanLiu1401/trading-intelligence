"""Quote-widget fingerprint regex parity across the triple-gate defense layers.

Five quote-widget patterns must be byte-identical across
``watchers.alert_agent``, ``analysis.claude_analyst``, and
``collectors.web_scraper`` — the documented anti-import-cycle lockstep
discipline (each layer carries its own copy because the analysis layer must
not pull the watchers+ml graph, and the watchers layer must not pull the
collectors/aiohttp graph). A regex tightened in one place silently goes out
of sync in the other two without this test pinning parity.

Two additional patterns (``_QW_STOCKTWITS_SENTIMENT`` / ``_QW_SCREENER_TAPE``)
live in only ``alert_agent`` + ``claude_analyst`` because their source rows
arrive from different collectors (``collectors/stocktwits_sentiment.py``,
``collectors/market_movers.py``) and never pass through ``web_scraper``;
they are pinned to two-way parity.

Also pins the post-fix behavior of ``_QW_IMAGE_CREDIT`` — the hyphen-aware
name-token alternation that catches Asian/French photographer names
("I-Hwa Cheng/Bloomberg", "O-Lin Wong/Reuters", "Jean-Pierre Dupont/AFP").
The prior regex required ``[A-Z][a-zA-Z]+`` for the first token, so a
hyphenated first token ("I-Hwa") failed at the second character (``-`` not in
``[a-zA-Z]``) and the credit silently leaked through every defense layer.
"""
from __future__ import annotations

import pytest

from watchers import alert_agent as A
from analysis import claude_analyst as C
from collectors import web_scraper as W


# ── Three-way parity (must exist in all three modules) ──────────────────────
TRIPLE_PATTERNS = (
    "_QW_PRICE_GLUE",
    "_QW_PCT_PAREN",
    "_QW_LISTING",
    "_QW_IMAGE_CREDIT",
    "_QW_QUOTE_PATH",
)

# ── Two-way parity (alert_agent + claude_analyst only) ─────────────────────
DUAL_PATTERNS = (
    "_QW_STOCKTWITS_SENTIMENT",
    "_QW_SCREENER_TAPE",
)


class TestTripleGateRegexParity:
    """The five core quote-widget fingerprints MUST be byte-identical across
    the three defense layers. Drift here is silent and catastrophic — a
    tightening on the alert path leaves the briefing path (the analyst's
    primary consumed product) showing the noise the alert path suppressed.
    """

    @pytest.mark.parametrize("name", TRIPLE_PATTERNS)
    def test_pattern_present_in_all_three_modules(self, name):
        for mod_name, mod in (("alert_agent", A),
                              ("claude_analyst", C),
                              ("web_scraper", W)):
            assert hasattr(mod, name), (
                f"{mod_name}.{name} missing — the triple-gate lockstep "
                f"requires every quote-widget fingerprint in all three modules"
            )

    @pytest.mark.parametrize("name", TRIPLE_PATTERNS)
    def test_pattern_byte_identical(self, name):
        pa = getattr(A, name).pattern
        pc = getattr(C, name).pattern
        pw = getattr(W, name).pattern
        assert pa == pc == pw, (
            f"{name} has DRIFTED across the triple-gate layers:\n"
            f"  alert_agent     : {pa!r}\n"
            f"  claude_analyst  : {pc!r}\n"
            f"  web_scraper     : {pw!r}"
        )

    @pytest.mark.parametrize("name", TRIPLE_PATTERNS)
    def test_pattern_flags_identical(self, name):
        # Same compiled regex flags too — a flags drift (IGNORECASE on one
        # layer but not another) is the silent kind that fails only on the
        # one title with a lowercased token.
        fa = getattr(A, name).flags
        fc = getattr(C, name).flags
        fw = getattr(W, name).flags
        assert fa == fc == fw, (
            f"{name} regex flags drifted: alert={fa} analyst={fc} web={fw}"
        )


class TestDualGateRegexParity:
    """``_QW_STOCKTWITS_SENTIMENT`` and ``_QW_SCREENER_TAPE`` are pinned at
    two-way parity (alert_agent + claude_analyst) — their source collectors
    don't pipe through web_scraper, so the third layer doesn't apply."""

    @pytest.mark.parametrize("name", DUAL_PATTERNS)
    def test_pattern_present_in_both_modules(self, name):
        for mod_name, mod in (("alert_agent", A), ("claude_analyst", C)):
            assert hasattr(mod, name), (
                f"{mod_name}.{name} missing — the dual-gate lockstep requires "
                f"both alert and briefing paths to share the fingerprint"
            )

    @pytest.mark.parametrize("name", DUAL_PATTERNS)
    def test_pattern_byte_identical(self, name):
        pa = getattr(A, name).pattern
        pc = getattr(C, name).pattern
        assert pa == pc, (
            f"{name} drifted between alert_agent and claude_analyst:\n"
            f"  alert_agent     : {pa!r}\n"
            f"  claude_analyst  : {pc!r}"
        )


class TestImageCreditHyphenatedNames:
    """The fix: ``_QW_IMAGE_CREDIT`` must catch hyphenated photographer names
    (Asian / French given-name conventions). Live evidence (2026-05-23
    urgency=2 set): ``I-Hwa Cheng/Bloomberg`` from
    ``scraped/www.bloomberg.com`` reached alerted state un-suppressed because
    the prior name-token regex required ``[A-Z][a-zA-Z]+`` and hit ``-`` at
    the second character. The fix adds an explicit hyphenated branch to the
    name-token alternation and is applied byte-identically across all three
    defense layers (test pinned by TestTripleGateRegexParity above)."""

    HYPHENATED_CREDITS = [
        "I-Hwa Cheng/Bloomberg",        # live failure (2026-05-23)
        "O-Lin Wong/Reuters",
        "Jean-Pierre Dupont/AFP",
        "Marie-Claire Vasse/Getty Images",
    ]

    CANONICAL_CREDITS = [
        "Tomohiro Ohsumi/Getty Images",  # live precedent (2026-05-21)
        "Angela Weiss/AFP/Getty Images",
        "Sang Tan/AP/Bloomberg",
        "Timothy A. Clary/AFP/Getty Images",
    ]

    MUST_SURVIVE = [
        # Real headlines that look superficially credit-shaped but are prose:
        "Reuters/Yahoo Finance reports earnings",
        "Sam Altman/OpenAI says GPT-5 coming",
        "MU drops 5%/Yahoo",
        "AFP/Getty Images launches new service",
        "NVDA breaks out on Reuters report",
        "Nvidia Q1 revenue rises 22%",
        "Fed cuts rates 50bp",
        "MU earnings blow past estimates",
        "Trump signs executive order on chips",
        # Hyphen in a hostname/domain context, not a credit:
        "Korea-Times reports MU news",
    ]

    @pytest.mark.parametrize("title", HYPHENATED_CREDITS)
    @pytest.mark.parametrize("mod_name,mod", [
        ("alert_agent", A), ("claude_analyst", C), ("web_scraper", W),
    ])
    def test_hyphenated_credit_caught(self, mod_name, mod, title):
        rx = getattr(mod, "_QW_IMAGE_CREDIT")
        assert rx.search(title), (
            f"{mod_name}._QW_IMAGE_CREDIT failed to catch {title!r} — "
            f"the hyphen-aware name-token branch regressed"
        )

    @pytest.mark.parametrize("title", CANONICAL_CREDITS)
    def test_canonical_credit_still_caught(self, title):
        """The original (non-hyphenated) credits must still match — the
        alternation rewrite did not regress the prior behavior."""
        assert A._QW_IMAGE_CREDIT.search(title), title

    @pytest.mark.parametrize("title", MUST_SURVIVE)
    def test_real_headlines_unaffected(self, title):
        """Real news headlines must NOT be falsely suppressed by the new
        hyphen branch."""
        assert not A._QW_IMAGE_CREDIT.search(title), (
            f"false positive on real headline: {title!r}"
        )

    def test_full_gate_drops_hyphenated_credit(self):
        """End-to-end: ``_looks_like_quote_widget`` (the public predicate
        the alert formatter consults) must drop a hyphenated image credit."""
        for title in self.HYPHENATED_CREDITS:
            assert A._looks_like_quote_widget({"title": title}), title
