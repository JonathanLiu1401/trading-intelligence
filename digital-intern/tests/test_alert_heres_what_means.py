"""Recap gate: "Here's What [It|That|This] [Really] Means" SEO trailer.

Present-tense sibling of ``heres_what_happened`` (which only matches the
past-tense "happened" form). Same SEO retrospective-trailer template family,
distinct regex.

Live evidence (2026-05-25, alert_recency.db pushed-alert audit — the canonical
record of REAL Discord pushes): two distinct titles fired standalone 🚨
BREAKING pushes within a 90-minute window of NVDA earnings afterglow:

  - "Nvidia's Board Just Authorized an Additional $80 Billion Buyback.
     Here's What That Really Means" at 00:17:28Z (GN: dividend buyback,
     ml_score=9.75 score_source='ml').
  - "Jensen Huang just made a surprise announcement. Here's what it means
     for Nvidia investors." at 02:03:48Z (GN: Nvidia, ai_score=8.0
     score_source='llm' — Sonnet itself ALSO mis-labeled this as urgent
     because the news lead before the SEO trailer reads real, even though
     the trailer adds zero actionable information).

Both publishers sit above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so
the source-authority gate does NOT catch them; content type IS the failure.

Tests verify:
  1. The new fingerprint catches each apostrophe variant + each closed-set
     pronoun (it/that/this), with and without the optional ``really``.
  2. The discriminator is the CLOSED PRONOUN SET + plural ``means`` (verb)
     — the singular noun ``mean`` is NOT matched, and a leading ``what``
     instead of ``here what`` is NOT matched. Real wires that use ``means``
     mid-sentence or that ask "what X means" survive.
  3. The lockstep mirror in ``analysis.claude_analyst`` catches the same
     set (anti-drift discipline with the alert-path gate).
  4. End-to-end ``_filter_recap_template_noise`` tags the suppressed rows
     with ``_recap_fingerprint='heres_what_means'``.
"""
from __future__ import annotations

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. The new fingerprint catches the live noise ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # Live row #1 — fired 00:17:28Z.
        "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Means",
        # Live row #2 — fired 02:03:48Z.
        "Jensen Huang just made a surprise announcement. Here's what it means for Nvidia investors.",
        # Variants the regex must also catch (closed pronoun set covered).
        "Fed cuts rates by 50bp. Here is what it means for stocks",
        "Here's what this means for the dollar",
        "Trump signs export ban. Here's what that means for Nvidia",
        # Curly-apostrophe (Unicode) variant — feeds occasionally normalise to ’.
        "MU earnings beat. Here’s what it means for the AI trade",
        # No-apostrophe form (matches the ``[s'’]+`` alternation's bare ``s``).
        "Powell speaks. Heres what that really means for inflation",
        # "Here is" long form (matches ``\s+is`` alternation).
        "AAPL announces buyback. Here is what it means for shareholders",
    ],
)
def test_alert_gate_catches_live_noise(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit, f"missed SEO trailer title: {title!r}"
    assert name == "heres_what_means", (
        f"expected heres_what_means fingerprint, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Means",
        "Jensen Huang just made a surprise announcement. Here's what it means for Nvidia investors.",
        "Here's what this means for the dollar",
        "MU earnings beat. Here’s what it means for the AI trade",
    ],
)
def test_briefing_gate_catches_live_noise(title: str) -> None:
    """Lockstep mirror in ``analysis.claude_analyst`` must catch the same
    titles — otherwise the briefing's top-50 digest can admit rows the alert
    path correctly suppresses (the documented cross-product drift class)."""
    hit, name = claude_analyst._looks_like_recap_template({"title": title})
    assert hit, f"briefing gate missed: {title!r}"
    assert name == "heres_what_means"


# ── 2. Must-survive corpus — real headlines are NEVER caught ───────────────


@pytest.mark.parametrize(
    "title",
    [
        # No leading ``here`` — "what X means" mid-sentence is real prose.
        "Powell explained what the cut means for inflation",
        "Analysts weigh in on what the buyback means",
        "Bank of America says rate cut means recession risk eases",
        # Leading ``what`` (not ``here what``) — different template family.
        "What it means when the Fed pauses",
        "What does this mean for traders?",
        # Singular ``mean`` (noun/verb without -s) — the discriminator excludes it.
        "Powell on what tariffs mean for the labor market",
        "Here is what it could mean for AI investors",  # "mean" not "means"
        # ``Here's why`` is a DIFFERENT SEO surface, not caught by this gate.
        "Here's why Nvidia could double from here",
        "Here's how to position for the Fed decision",
        # Real breaking wire copy — never names an event with this trailer.
        "Fed surprises with 50bp emergency rate cut",
        "Nvidia Q1 revenue rises 22% to $44.06 billion, beats estimates",
        "MU earnings blow past estimates; shares jump 8%",
        "Trump signs executive order on semiconductor exports",
        "Apple announces $100B buyback",
    ],
)
def test_alert_gate_must_survive(title: str) -> None:
    """Real headlines must NOT be caught by the new fingerprint specifically.
    A title may legitimately be caught by a DIFFERENT recap fingerprint
    (the codebase has 25+); this test only pins that ``heres_what_means``
    does not over-fire on the must-survive corpus."""
    _, name = alert_agent._looks_like_recap_template({"title": title})
    if name == "heres_what_means":
        pytest.fail(f"heres_what_means false-positive: {title!r}")


# ── 3. End-to-end filter integration ────────────────────────────────────────


def test_filter_recap_template_noise_separates_correctly() -> None:
    """The new fingerprint integrates into ``_filter_recap_template_noise``:
    recap rows go to ``suppressed`` and carry ``_recap_fingerprint``, real
    rows go to ``kept``, caller's row is NOT mutated."""
    recap = {
        "_id": "r1",
        "title": "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Means",
        "link": "https://msn.com/x",
        "source": "GN: dividend buyback",
    }
    real = {
        "_id": "r2",
        "title": "Fed cuts rates 50bp in emergency move",
        "link": "https://reuters.com/x",
        "source": "rss",
    }
    kept, suppressed = alert_agent._filter_recap_template_noise([recap, real])
    assert [a["_id"] for a in kept] == ["r2"]
    assert len(suppressed) == 1
    assert suppressed[0]["_id"] == "r1"
    assert suppressed[0]["_recap_fingerprint"] == "heres_what_means"
    # Pure: caller's input row must not be mutated.
    assert "_recap_fingerprint" not in recap


# ── 4. Lockstep parity — both gates name the same fingerprint ───────────────


def test_lockstep_parity_on_canonical_jensen_huang_row() -> None:
    """A test failure here means the alert and briefing recap gates drifted —
    the same headline must be caught with the same fingerprint name on both
    paths, mirroring ``test_alert_heres_what_happened`` / the documented
    structural ``test_alert_and_briefing_recap_tuples_have_same_length``."""
    title = (
        "Jensen Huang just made a surprise announcement. "
        "Here's what it means for Nvidia investors."
    )
    a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
    b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
    assert a_hit and b_hit
    assert a_name == b_name == "heres_what_means"
