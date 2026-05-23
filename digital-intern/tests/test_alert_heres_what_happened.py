"""Recap gate: "Here's What Happened" SEO retrospective tail.

The Motley Fool / MarketBeat / tickerreport.com SEO-mill template trails its
headline with a generic "Here's What Happened" hook ("Nvidia Just Crushed
Earnings Estimates, but the Stock Fell. Here's What Happened (and What Comes
Next)"). It is by definition past-tense recap — the move was already in the
market by the time the explainer was written.

Live evidence (2026-05-23, 24h articles.db scan): the Motley Fool variant
syndicated across SIX sources (Motley Fool, yfinance/Motley Fool,
scraped/finance.yahoo.com, YahooFinance/NVDA, GN: earnings, GDELT/fool.com)
reached urgency=1 with ml_score 9.22-9.41 — every copy a queued 🚨 BREAKING
push the analyst was about to receive on retrospective content.

Tests verify three things:
  1. The new fingerprint catches each apostrophe variant + the bare "Here What
     Happened" / "Here is What Happened" forms;
  2. The discriminator is the past-tense "happened" — the present-continuous
     "Here's What's Happening" (often forward-looking market wrap) is NOT
     matched;
  3. The same lockstep mirror in analysis.claude_analyst catches the same
     titles (anti-drift discipline with the alert-path gate).
"""
from __future__ import annotations

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. The new fingerprint catches the live noise ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # The exact live row that reached urgency=1 syndicated 6×.
        "Nvidia Just Crushed Earnings Estimates, but the Stock Fell. Here's What Happened (and What Comes Next)",
        # GDELT/fool.com no-apostrophe variant (live).
        "Nvidia Just Crushed Earnings Estimates , but the Stock Fell . Here What Happened ( and What",
        # MarketBeat variant (live).
        "Costco Wholesale (NASDAQ:COST) Stock Price Down 2.1% - Here's What Happened - MarketBeat",
        "Alphabet (NASDAQ:GOOGL) Trading Down 1.2% - Here's What Happened - MarketBeat",
        # tickerreport.com variant (live).
        "Lam Research ( NASDAQ : LRCX ) Shares Down 1 . 6 % – Here What Happened",
        # Curly-apostrophe (Unicode) variant — feeds occasionally normalise to ’.
        "Nvidia Earnings Crushed. Here’s What Happened",
    ],
)
def test_alert_gate_catches_live_noise(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit, f"missed live recap title: {title!r}"
    assert name == "heres_what_happened", (
        f"expected heres_what_happened fingerprint, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Nvidia Just Crushed Earnings Estimates, but the Stock Fell. Here's What Happened (and What Comes Next)",
        "Costco Wholesale (NASDAQ:COST) Stock Price Down 2.1% - Here's What Happened - MarketBeat",
        "Lam Research ( NASDAQ : LRCX ) Shares Down 1 . 6 % – Here What Happened",
        "Nvidia Earnings Crushed. Here’s What Happened",
    ],
)
def test_briefing_gate_catches_live_noise(title: str) -> None:
    """Lockstep mirror in analysis.claude_analyst must catch the same set —
    otherwise the briefing's top-50 digest can still admit rows the alert
    path correctly suppresses, the recurring cross-product drift class."""
    hit, name = claude_analyst._looks_like_recap_template({"title": title})
    assert hit, f"briefing gate missed: {title!r}"
    assert name == "heres_what_happened"


# ── 2. Must-survive corpus — real headlines are NEVER caught ───────────────


@pytest.mark.parametrize(
    "title",
    [
        # Real breaking wire copy — never names an event with "Here's What Happened".
        "Fed surprises with 50bp emergency rate cut",
        "Nvidia Q1 revenue rises 22% to $44.06 billion, beats estimates",
        "MU earnings blow past estimates; shares jump 8%",
        "Trump signs executive order on semiconductor exports",
        "Apple announces $100B buyback",
        # Present-continuous "What's Happening" (often live market wrap, can be
        # forward-looking) — the past-tense discriminator means this is NOT recap.
        "Here's What's Happening in Markets Today",
        "Here's what's happening with Nvidia stock",
        # Other "Here's" headlines that aren't recap tails.
        "Here's Why Nvidia Could Double from Here",
        "Here's How To Position For The Fed Decision",
        # Edge: "happened" in different context.
        "What Happened in 2008: Lessons for Today's Investors",
        # Real headline with "happened" mid-prose but no "here" hook.
        "Tesla shares jumped after Musk's tweet — what happened next",
    ],
)
def test_alert_gate_must_survive(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    if name == "heres_what_happened":
        pytest.fail(f"false-positive on must-survive title: {title!r}")
    # Note: the title may still be caught by a DIFFERENT recap fingerprint;
    # this test only pins that the NEW fingerprint does not over-fire.


# ── 3. End-to-end filter integration ────────────────────────────────────────


def test_filter_recap_template_noise_separates_correctly() -> None:
    """The new fingerprint must integrate into the partition so the recap
    rows are suppressed and tagged with ``_recap_fingerprint``."""
    recap = {
        "_id": "r1",
        "title": "Nvidia Just Crushed Earnings Estimates, but the Stock Fell. Here's What Happened",
        "link": "https://fool.com/x",
        "source": "yfinance/Motley Fool",
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
    assert suppressed[0]["_recap_fingerprint"] == "heres_what_happened"
    # Pure: caller's input rows must not be mutated.
    assert "_recap_fingerprint" not in recap


# ── 4. Lockstep parity — both gates name the same fingerprint ───────────────


def test_lockstep_parity_on_canonical_motley_fool_row() -> None:
    """A test failure here means the alert and briefing recap gates drifted —
    the same headline must be caught with the same fingerprint name on both
    paths, or downstream noise/parity tooling will mis-attribute."""
    title = (
        "Nvidia Just Crushed Earnings Estimates, but the Stock Fell. "
        "Here's What Happened (and What Comes Next)"
    )
    a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
    b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
    assert a_hit and b_hit
    assert a_name == b_name == "heres_what_happened"
