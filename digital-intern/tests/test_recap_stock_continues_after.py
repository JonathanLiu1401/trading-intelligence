"""Pin the ``stock_continues_after`` recap fingerprint.

Live failure (2026-05-22..25, alert_recency.db push audit + 7d articles.db
scan): the exact title "Nvidia stock continues to struggle after earnings,
but analysts remain firmly bullish" pushed 4 distinct Discord BREAKING
alerts in 3 days across 4 syndication channels (Invezz, CryptoRank, MSN,
TradingView), every copy ``score_source='ml'`` ml_score 9.81-9.99 — exactly
the analyst's #1 noise complaint (duplicate breaking pushes for a
retrospective post-event recap) on a fingerprint no existing sibling
caught:

  * ``subject_pct_after`` requires ``\\d+(?:\\.\\d+)?\\s*%`` — this title
    has no explicit move magnitude.
  * ``why_stock_is_after`` / ``why_pct_after`` require a leading ``^Why``.
  * ``whats_next_after`` requires "what's/is next after" phrasing.

Empirical false-positive bar (2026-05-25 audit, 55,488 titles in last 7
days): zero false positives — the 30-day urgency=2 set (1340 titles)
matched only the 4 canonical syndication copies, exactly the target
failure and nothing else.

This test file pins:
  * the verbatim 4-copy live failure case matches
  * the fingerprint name appears in both ``_RECAP_TEMPLATE_PATTERNS`` (alert
    side) AND ``_BRIEFING_RECAP_TEMPLATE_PATTERNS`` (briefing side) — the
    documented lockstep / anti-drift discipline (sibling structural test
    ``test_alert_and_briefing_recap_tuples_have_same_length`` enforces the
    tuple length match)
  * SSOT parity: same input → same ``(hit, name)`` from both
    ``alert_agent._looks_like_recap_template`` AND the briefing-side
    ``analysis.claude_analyst._looks_like_recap_template``
  * a must-survive corpus of real recent pushes / breaking headlines stays
    un-flagged — verbatim from the recent ``alert_recency.db`` push history
    so we cannot silently flag a real wire as recap
"""
from watchers import alert_agent
from analysis import claude_analyst


# The exact verbatim title that fired 4 distinct Discord pushes in 3 days,
# one syndication tail per source. Each must match the gate.
LIVE_FAILURE_TITLES = [
    "Nvidia stock continues to struggle after earnings, but analysts remain firmly bullish - MSN",
    "Nvidia stock continues to struggle after earnings, but analysts remain firmly bullish - CryptoRank",
    "Nvidia stock continues to struggle after earnings, but analysts remain firmly bullish - Invezz",
    "Nvidia stock continues to struggle after earnings, but analysts remain firmly bullish - TradingView",
]

# Plausible same-family titles — same recap shape with the other state-
# continuation verbs and other event-noun terminators. All must match.
SAME_FAMILY_TITLES = [
    "Apple stock keeps falling after Q1 miss",
    "AMD stock remains weak after disappointing guidance",
    "MSFT stock stays in the red after report",
    "Tesla stock continues sliding after the Q3 results",
    "INTC stock keeps tumbling after the analyst downgrade",
    "AAPL Stock Continues To Slide After Q4 Earnings Miss",
]

# Real recent push / wire titles that share similar tokens but are NOT
# retrospective recap. ZERO of these may match — they are real news the
# analyst NEEDS. Sourced verbatim from the 2026-05 alert_recency.db push
# log.
MUST_NOT_MATCH_REAL_PUSHES = [
    "JPMorgan lifts Nvidia target to $280 after record quarter - MSN",
    "Bank of America revamps Nvidia stock price target after earnings - TheStreet",
    "All eyes on NVDA stock as Q1 report looms: Morgan Stanley believes 'typical beat and raise pattern is a likely outcome'",
    "Nvidia Hits Record $81.6B Revenue — So Why Is the Stock Down? - Moomoo",
    "NVIDIA Stock Nears Key Price As Huang Pitches $200 Billion CPU Market",
    "Arm stock extends rally on Nvidia's $20B CPU forecast - MSN",
    "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN",
    "Citi resets Micron stock price target after an anomaly",
]

# Generic forward-looking / un-related must-survives.
MUST_NOT_MATCH_FORWARD = [
    "Apple Stock Could Soar After Earnings Beat",
    "Nvidia stock outlook bullish after fresh guidance",
    "MU stock to watch after earnings preview",
    "Apple stock continues to dominate the market",  # no after-earnings
    "Nvidia stock continues rallying on AI demand",  # no after-earnings
    "Apple stock still has room to run after Q1 earnings",  # 'still' deliberately not in verb list
    "Nvidia Q1 revenue rises 22% to $35.1 billion",
    "Fed cuts rates 50bp",
    "MU shares halted",
    "PORTFOLIO P&L SNAPSHOT",
]


class TestStockContinuesAfterGateCatchesLiveFailure:
    """The exact 4 syndication copies of the canonical live failure title MUST
    match — this is the gate's whole reason to exist."""

    def test_all_four_syndication_copies_match_alert_side(self):
        for title in LIVE_FAILURE_TITLES:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert hit, f"alert gate missed live failure: {title!r}"
            assert name == "stock_continues_after", (
                f"alert gate caught {title!r} with wrong fingerprint {name!r}"
            )

    def test_all_four_syndication_copies_match_briefing_side(self):
        for title in LIVE_FAILURE_TITLES:
            hit, name = claude_analyst._looks_like_recap_template({"title": title})
            assert hit, f"briefing gate missed live failure: {title!r}"
            assert name == "stock_continues_after", (
                f"briefing gate caught {title!r} with wrong fingerprint {name!r}"
            )


class TestStockContinuesAfterCatchesFamilyTemplates:
    """All four state-continuation verbs (continues/keeps/stays/remains) and
    the established event-noun terminators must match. This pins the gate's
    intended generality without overreaching."""

    def test_all_family_variants_match_alert_side(self):
        for title in SAME_FAMILY_TITLES:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert hit, f"alert gate missed family variant: {title!r}"
            assert name == "stock_continues_after"

    def test_all_family_variants_match_briefing_side(self):
        for title in SAME_FAMILY_TITLES:
            hit, name = claude_analyst._looks_like_recap_template({"title": title})
            assert hit, f"briefing gate missed family variant: {title!r}"
            assert name == "stock_continues_after"


class TestStockContinuesAfterMustNotOverCatch:
    """Real recent pushes / breaking wires must NOT be flagged. A single false
    positive on a real breaking story is a missed alert — far worse than a
    duplicate, per the alert_recency.py module's documented bar."""

    def test_real_recent_pushes_survive_alert_side(self):
        for title in MUST_NOT_MATCH_REAL_PUSHES:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert not hit, (
                f"alert gate FALSE POSITIVE on real recent push: "
                f"{title!r} caught by {name!r}"
            )

    def test_real_recent_pushes_survive_briefing_side(self):
        for title in MUST_NOT_MATCH_REAL_PUSHES:
            hit, name = claude_analyst._looks_like_recap_template({"title": title})
            assert not hit, (
                f"briefing gate FALSE POSITIVE on real recent push: "
                f"{title!r} caught by {name!r}"
            )

    def test_forward_looking_survives_alert_side(self):
        for title in MUST_NOT_MATCH_FORWARD:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert not hit, (
                f"alert gate FALSE POSITIVE on forward-looking: "
                f"{title!r} caught by {name!r}"
            )

    def test_forward_looking_survives_briefing_side(self):
        for title in MUST_NOT_MATCH_FORWARD:
            hit, name = claude_analyst._looks_like_recap_template({"title": title})
            assert not hit, (
                f"briefing gate FALSE POSITIVE on forward-looking: "
                f"{title!r} caught by {name!r}"
            )


class TestStockContinuesAfterIsRegistered:
    """Anti-drift: the fingerprint name MUST appear in both the alert and
    briefing pattern tuples. The structural-length sibling
    ``test_alert_and_briefing_recap_tuples_have_same_length`` already pins
    the count match; this explicitly pins our specific name on both sides."""

    def test_registered_alert_side(self):
        names = [n for n, _ in alert_agent._RECAP_TEMPLATE_PATTERNS]
        assert "stock_continues_after" in names, (
            f"'stock_continues_after' missing from _RECAP_TEMPLATE_PATTERNS — "
            f"have {names!r}"
        )

    def test_registered_briefing_side(self):
        names = [n for n, _ in claude_analyst._BRIEFING_RECAP_TEMPLATE_PATTERNS]
        assert "stock_continues_after" in names, (
            f"'stock_continues_after' missing from _BRIEFING_RECAP_TEMPLATE_PATTERNS"
            f" — have {names!r}"
        )

    def test_alert_and_briefing_agree_on_live_failure(self):
        """SSOT parity: same input → same (hit, name) from both sides. A
        layered-defense gate that disagrees with its lockstep mirror means
        one side has drifted, which is exactly what
        ``test_alert_and_briefing_gates_agree_on_must_survive_corpus`` enforces
        on the survivor side — we add the catch-side parity here."""
        for title in LIVE_FAILURE_TITLES + SAME_FAMILY_TITLES:
            a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
            b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
            assert (a_hit, a_name) == (b_hit, b_name), (
                f"alert/briefing gate disagreement on {title!r}: "
                f"alert={(a_hit, a_name)!r}, briefing={(b_hit, b_name)!r}"
            )


class TestStockContinuesAfterEmptyAndDefensiveInput:
    """Pure / total contract — empty / missing / non-string title must never
    raise, mirrors the silence-on-bad-input discipline of every other gate."""

    def test_empty_title_no_match(self):
        hit, name = alert_agent._looks_like_recap_template({"title": ""})
        assert hit is False and name == ""
        hit, name = claude_analyst._looks_like_recap_template({"title": ""})
        assert hit is False and name == ""

    def test_missing_title_no_match(self):
        hit, name = alert_agent._looks_like_recap_template({})
        assert hit is False and name == ""
        hit, name = claude_analyst._looks_like_recap_template({})
        assert hit is False and name == ""

    def test_none_title_no_match(self):
        hit, name = alert_agent._looks_like_recap_template({"title": None})
        assert hit is False and name == ""
        hit, name = claude_analyst._looks_like_recap_template({"title": None})
        assert hit is False and name == ""
