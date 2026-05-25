"""Recap gate: "Here's What [It|That|This] [Really] Signals" SEO trailer.

Variant-verb sibling of ``heres_what_means`` (which catches the same SEO
trailer template with the verb ``means``). Same retrospective-trailer
template family, distinct verb.

Live evidence (2026-05-25, articles.db 24h urgent scan with the
heres_what_means gate active): two NEW rows reached urgency=1 with
ml_score 9.5-9.8 carrying the same SEO trailer template but with
``signals`` instead of ``means``:

  - "Nvidia's Board Just Authorized an Additional $80 Billion Buyback.
     Here's What That Really Signals to Investors. - The Globe and Mail"
     (ml_score=9.83, score_source='ml', GN: dividend buyback).
  - "Nvidia's board just authorized an additional $80 billion buyback.
     Here's what that really signals to investors. - MSN"
     (ml_score=9.53, score_source='ml', GN: Nvidia).

Both publishers sit above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so
the source-authority gate does NOT catch them; content type IS the
failure, identical class to ``heres_what_means``.

Tests verify:
  1. The new fingerprint catches each apostrophe variant + each closed-set
     pronoun (it/that/this), with and without the optional ``really``.
  2. The discriminator is the CLOSED PRONOUN SET + plural ``signals``
     (verb) — singular ``signal`` (noun) is NOT matched, and a leading
     ``Powell signals`` (subject + verb, no ``here what`` lead-in) is NOT
     matched. Real wires that use ``signals`` mid-sentence survive.
  3. The lockstep mirror in ``analysis.claude_analyst`` catches the same
     set (anti-drift discipline with the alert-path gate).
  4. End-to-end ``_filter_recap_template_noise`` tags suppressed rows
     with ``_recap_fingerprint='heres_what_signals'``.
"""
from __future__ import annotations

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. The new fingerprint catches the live noise ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # Live row #1 — Globe and Mail variant (ml_score=9.83).
        "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Signals to Investors. - The Globe and Mail",
        # Live row #2 — MSN variant (ml_score=9.53).
        "Nvidia's board just authorized an additional $80 billion buyback. Here's what that really signals to investors. - MSN",
        # Closed pronoun set coverage.
        "Fed cut rates 50bp. Here's what it signals for the market",
        "Trump signs trade order. Here's what this signals for tech",
        # Curly-apostrophe (Unicode) variant.
        "MU earnings beat. Here’s what it signals for the AI trade",
        # No-apostrophe form.
        "Powell speaks. Heres what that really signals for inflation",
        # "Here is" long form.
        "AAPL announces buyback. Here is what this signals for shareholders",
    ],
)
def test_alert_gate_catches_live_noise(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit, f"missed SEO trailer title: {title!r}"
    assert name == "heres_what_signals", (
        f"expected heres_what_signals fingerprint, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Signals to Investors. - The Globe and Mail",
        "Nvidia's board just authorized an additional $80 billion buyback. Here's what that really signals to investors. - MSN",
        "Fed cut rates 50bp. Here's what it signals for the market",
        "MU earnings beat. Here’s what it signals for the AI trade",
    ],
)
def test_briefing_gate_catches_live_noise(title: str) -> None:
    """Lockstep mirror in ``analysis.claude_analyst`` must catch the same
    titles — otherwise the briefing's top-50 digest can admit rows the
    alert path correctly suppresses."""
    hit, name = claude_analyst._looks_like_recap_template({"title": title})
    assert hit, f"briefing gate missed: {title!r}"
    assert name == "heres_what_signals"


# ── 2. Must-survive corpus — real headlines are NEVER caught ───────────────


@pytest.mark.parametrize(
    "title",
    [
        # "Subject signals X" real wire copy — no leading ``here what``.
        "Powell signals more cuts after rate decision",
        "Fed signals dovish tilt after CPI miss",
        "BOJ signals shift in policy outlook",
        # No leading ``here`` — "what X signals" mid-sentence.
        "Analysts wonder what this signals for the AI trade",
        "Investors debate what the Fed pause signals",
        # Singular ``signal`` (noun/verb without -s) — discriminator excludes it.
        "Earnings results signal continued growth",
        "MU price action gives bullish signal",
        # Leading ``what`` not ``here what`` — different template.
        "What this signals for the AI trade",
        "What does this signal for traders?",
        # Real breaking wire copy.
        "Fed surprises with 50bp emergency rate cut",
        "Nvidia Q1 revenue rises 22% to $44.06 billion, beats estimates",
        "Apple announces $100B buyback",
    ],
)
def test_alert_gate_must_survive(title: str) -> None:
    """Real headlines must NOT be caught by ``heres_what_signals`` specifically.
    Mirrors the discipline of ``test_alert_heres_what_means`` — a title may
    legitimately be caught by a DIFFERENT recap fingerprint."""
    _, name = alert_agent._looks_like_recap_template({"title": title})
    if name == "heres_what_signals":
        pytest.fail(f"heres_what_signals false-positive: {title!r}")


# ── 3. End-to-end filter integration ────────────────────────────────────────


def test_filter_recap_template_noise_separates_correctly() -> None:
    """The new fingerprint integrates into ``_filter_recap_template_noise``."""
    recap = {
        "_id": "r1",
        "title": "Nvidia's Board Just Authorized an Additional $80 Billion Buyback. Here's What That Really Signals to Investors. - The Globe and Mail",
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
    assert suppressed[0]["_recap_fingerprint"] == "heres_what_signals"
    assert "_recap_fingerprint" not in recap


# ── 4. Lockstep parity ───────────────────────────────────────────────────────


def test_lockstep_parity_on_canonical_globe_row() -> None:
    """A test failure here means the alert and briefing recap gates drifted."""
    title = (
        "Nvidia's board just authorized an additional $80 billion buyback. "
        "Here's what that really signals to investors. - MSN"
    )
    a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
    b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
    assert a_hit and b_hit
    assert a_name == b_name == "heres_what_signals"
