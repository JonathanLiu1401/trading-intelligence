"""Zacks SEO blog-highlights recap gate — alert + briefing lockstep.

Live evidence (2026-05-26, articles.db 30-day urgency scan): the Zacks
"Analyst Blog Highlights" / "featured highlights include" template was firing
real standalone 🚨 BREAKING pushes (4 urgency=2 rows today alone, 101 same-
template rows at urgency=0 across 30 days). The ML urgency head over-scored
them because the title is dense with held tickers (NVDA + MU + KLA + LITE),
cross-cycle dedup didn't merge them because the source-attribution suffix
varied between syndications, and no existing recap fingerprint caught the
Zacks-specific lead.

This test pins:
  - Positive corpus: live failure-case titles + canonical Zacks SEO variants.
  - Must-survive corpus: real wires that happen to mention Zacks mid-sentence;
    forward-looking Zacks rating actions ("MU added to Zacks Rank 1 list").
  - Alert/briefing parity on the same titles (byte-identical regex).

Read-only / pure-function tests — no DB, no IO, no LLM. The gate is a text
drop at the formatter chokepoint, so unit-testing the regex is the spec.
"""
from analysis.claude_analyst import (
    _BRIEFING_RECAP_TEMPLATE_PATTERNS,
    _looks_like_recap_template as briefing_looks_like_recap,
)
from watchers.alert_agent import (
    _RECAP_TEMPLATE_PATTERNS,
    _RT_ZACKS_HIGHLIGHTS,
    _looks_like_recap_template as alert_looks_like_recap,
)


# Live failure cases — the exact titles articles.db urgency=2 captured today.
LIVE_FAILURE_TITLES = [
    # 2026-05-26T10:35:20Z Finnhub/Yahoo ml=9.9 urgency=2
    "Zacks.com featured highlights include Micron Technology, Murphy USA and Vertiv",
    # 2026-05-26T09:40:31Z YahooFinance/NVDA ml=9.9 urgency=2
    "The Zacks Analyst Blog Highlights NVDA, FTEC, VGT, SMH, IYW and XLK",
    # 2026-05-26T09:35:40Z yfinance/Zacks ml=9.83 urgency=2 (same blog post)
    "The Zacks Analyst Blog Highlights NVDA, FTEC, VGT, SMH, IYW and XLK",
    # 2026-05-26T08:54:55Z yfinance/Zacks ml=9.5 urgency=2
    "Zacks.com featured highlights include Micron Technology, Murphy USA and Vertiv",
]

# Same template seen in the 30-day urgency=0 leak set — different ticker
# composition, same SEO mill signature.
TEMPLATE_VARIANTS = [
    "The Zacks Analyst Blog Highlights Applied Materials, Shell and KLA",
    "The Zacks Analyst Blog Highlights Johnson & Johnson, Oracle, Netflix and Espey",
    "The Zacks Analyst Blog Highlights D-Wave Quantum and Rigetti Computing",
    "Zacks.com featured highlights include NVIDIA and Micron Technology",
    "Zacks.com featured highlights include Dow, Arrow Electronics and Lumentum",
    # Syndicated source-attribution suffix variants (cross-cycle dedup misses
    # these because the suffix differs, but the gate fires regardless because
    # it's anchored on the lead).
    "Zacks.com featured highlights include NVIDIA and Micron Technology - The Globe and Mail",
    "Zacks.com featured highlights include Micron Technology, Murphy USA and Vertiv - The Globe and Mail",
    "The Zacks Analyst Blog Highlights NVDA, FTEC, VGT, SMH, IYW and XLK - Yahoo Finance",
    # Tolerance for the "Zacks featured highlights" form (no .com) — same
    # template, occasionally republished without the .com.
    "Zacks featured highlights include Apple and Microsoft",
]

# Real wire copy that legitimately mentions Zacks but is NOT this SEO template —
# must NOT match. Includes forward-looking analyst actions, mid-sentence Zacks
# references, and real breaking headlines that have no Zacks lead at all.
MUST_SURVIVE = [
    # Forward-looking Zacks rating actions (the real Zacks signal, not the
    # blog-mill output).
    "MU added to Zacks Rank #1 (Strong Buy) list",
    "Nvidia upgraded to Zacks Rank #2 by analyst team",
    "Zacks raises MU price target to $200 on memory demand",
    # Mid-sentence Zacks reference inside a real wire headline.
    "Bank of America says Zacks highlights NVDA in their year-end note",
    # Real breaking news — no Zacks template anywhere.
    "Nvidia Q1 revenue rises 22% to $35.1 billion",
    "Fed cuts rates by 50bp on labor weakness",
    "MU earnings blow past Q3 estimates",
    "Tesla shares jump after earnings beat",
    "Reuters reports Nvidia $80B buyback announcement",
    # Synthetic PORTFOLIO/OPTIONS snapshot rows the daemon prepends to the
    # digest (no link/url; must always pass through untouched, same discipline
    # as every other recap gate's must-survive corpus).
    "PORTFOLIO P&L SNAPSHOT",
    "OPTIONS SNAPSHOT",
]


class TestAlertGateCatchesZacksTemplate:
    """Pin every live failure case fires the alert-path recap gate."""

    def test_live_failure_titles_all_caught(self):
        for title in LIVE_FAILURE_TITLES:
            hit, name = alert_looks_like_recap({"title": title})
            assert hit, f"alert gate missed live failure {title!r}"
            assert name == "zacks_highlights", (
                f"alert gate caught {title!r} with wrong fingerprint {name!r}"
            )

    def test_template_variants_all_caught(self):
        for title in TEMPLATE_VARIANTS:
            hit, name = alert_looks_like_recap({"title": title})
            assert hit, f"alert gate missed template variant {title!r}"
            assert name == "zacks_highlights", (
                f"alert gate caught {title!r} with wrong fingerprint {name!r}"
            )


class TestBriefingGateCatchesZacksTemplate:
    """Lockstep mirror — every live failure must also fire the briefing gate."""

    def test_live_failure_titles_all_caught(self):
        for title in LIVE_FAILURE_TITLES:
            hit, name = briefing_looks_like_recap({"title": title})
            assert hit, f"briefing gate missed live failure {title!r}"
            assert name == "zacks_highlights", (
                f"briefing gate caught {title!r} with wrong fingerprint {name!r}"
            )

    def test_template_variants_all_caught(self):
        for title in TEMPLATE_VARIANTS:
            hit, name = briefing_looks_like_recap({"title": title})
            assert hit, f"briefing gate missed template variant {title!r}"


class TestZacksGateMustSurvive:
    """No real wire / forward Zacks rating action gets falsely caught."""

    def test_alert_path_lets_real_news_survive(self):
        for title in MUST_SURVIVE:
            hit, name = alert_looks_like_recap({"title": title})
            assert not hit, (
                f"alert gate over-caught {title!r} with fingerprint {name!r}"
            )

    def test_briefing_path_lets_real_news_survive(self):
        for title in MUST_SURVIVE:
            hit, name = briefing_looks_like_recap({"title": title})
            assert not hit, (
                f"briefing gate over-caught {title!r} with fingerprint {name!r}"
            )


class TestAlertBriefingParity:
    """The two gates must agree on every live test row."""

    def test_alert_and_briefing_agree_on_all_cases(self):
        all_titles = LIVE_FAILURE_TITLES + TEMPLATE_VARIANTS + MUST_SURVIVE
        for title in all_titles:
            a_hit, _ = alert_looks_like_recap({"title": title})
            b_hit, _ = briefing_looks_like_recap({"title": title})
            assert a_hit == b_hit, (
                f"alert/briefing disagree on {title!r}: alert={a_hit}, "
                f"briefing={b_hit}"
            )

    def test_zacks_highlights_appears_in_both_tuples(self):
        alert_names = {name for name, _ in _RECAP_TEMPLATE_PATTERNS}
        briefing_names = {name for name, _ in _BRIEFING_RECAP_TEMPLATE_PATTERNS}
        assert "zacks_highlights" in alert_names, (
            "zacks_highlights missing from alert _RECAP_TEMPLATE_PATTERNS"
        )
        assert "zacks_highlights" in briefing_names, (
            "zacks_highlights missing from briefing _BRIEFING_RECAP_TEMPLATE_PATTERNS"
        )


class TestZacksRegexAnchoring:
    """The regex must be anchored ^ — a mid-sentence Zacks reference inside a
    real wire headline must NEVER match."""

    def test_regex_anchored_at_start(self):
        # The mid-sentence reference puts "Zacks" 12+ chars in; the anchored
        # regex must not match.
        assert _RT_ZACKS_HIGHLIGHTS.search(
            "Bank of America says Zacks.com featured highlights include NVDA"
        ) is None

    def test_regex_matches_leading_whitespace(self):
        # Leading whitespace tolerance — feeds occasionally emit a leading
        # space or tab.
        assert _RT_ZACKS_HIGHLIGHTS.search(
            "  The Zacks Analyst Blog Highlights NVDA"
        ) is not None

    def test_regex_case_insensitive(self):
        # Real feeds emit "The Zacks Analyst Blog Highlights" with the canonical
        # Title-Case; check the lowercase / all-caps variants would also fire
        # (defensive — a syndicating publisher could lower-case the prefix).
        assert _RT_ZACKS_HIGHLIGHTS.search(
            "the zacks analyst blog highlights nvda"
        ) is not None
        assert _RT_ZACKS_HIGHLIGHTS.search(
            "ZACKS.COM FEATURED HIGHLIGHTS INCLUDE NVDA"
        ) is not None
