"""Recap gate: "<Fund> LLC Makes New $X [Million|Billion] Investment in <Co>"
— the MarketBeat / AlphaVantage / americanbankingnews 13F press-mill leading-
LLC sibling of ``holdings_by_fund`` (trailing-LLC) and ``shares_bought_by``
("Shares ... bought by"). Same 13F filing-recap template family, distinct
phrasing — the LLC is the SUBJECT (announces the investment), not the trailer.

Live evidence (2026-05-25, articles.db 24h urgent scan): "Torren Management
LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA" reached
urgency=1 (ml_score=9.71, score_source='ml', from AlphaVantage/MarketBeat) —
pure 13F filing recap, by definition retrospective (the SEC filing was
already public weeks before this headline). The MarketBeat credibility tier
(0.68) sits above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so the
source-authority gate does NOT catch it; content type IS the failure,
identical class to the existing two 13F-mill gates.

Tests verify:
  1. The new fingerprint catches each verb (``Makes|Made|Acquires|Acquired|
     Takes|Took``) with and without ``New``, both ``Million`` and ``Billion``
     magnitude qualifiers, both ``Position`` and ``Investment`` and ``Stake``.
  2. The discriminator is the leading ``<Fund> LLC`` + a dollar-prefixed
     magnitude figure — real news that mentions a fund taking a stake
     WITHOUT the LLC anchor + dollar-magnitude template (sovereign-fund
     deals, insider buys, "Holdings now N shares") is NOT caught.
  3. The lockstep mirror in ``analysis.claude_analyst`` catches the same
     set (anti-drift discipline).
  4. End-to-end ``_filter_recap_template_noise`` tags suppressed rows
     with ``_recap_fingerprint='fund_makes_investment'``.
"""
from __future__ import annotations

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. The new fingerprint catches the live noise ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # The exact live row that reached urgency=1.
        "Torren Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA",
        # Verb alternation coverage.
        "Berkshire Hathaway LLC Makes New $100 Million Investment in Apple Inc",
        "Acme Capital LLC Acquired $5 Million Position in Tesla",
        "BlackRock Asset Management LLC Took $50 Million Stake in Microsoft",
        # Million / Billion magnitude variants.
        "Vanguard Group LLC Made $1.2 Billion Position in NVIDIA",
        # Bare dollar (no Million/Billion qualifier).
        "Small Cap LLC Acquires $500,000 Investment in AMD",
        # Multi-word fund-name prefix.
        "Cohen and Steers Capital Management LLC Makes New $25 Million Investment in Vornado",
    ],
)
def test_alert_gate_catches_live_noise(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit, f"missed MarketBeat 13F mill title: {title!r}"
    assert name == "fund_makes_investment", (
        f"expected fund_makes_investment fingerprint, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Torren Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA",
        "Berkshire Hathaway LLC Makes New $100 Million Investment in Apple Inc",
        "Acme Capital LLC Acquired $5 Million Position in Tesla",
        "Vanguard Group LLC Made $1.2 Billion Position in NVIDIA",
    ],
)
def test_briefing_gate_catches_live_noise(title: str) -> None:
    """Lockstep mirror — same titles must be caught by the briefing path
    gate or the per-domain cap admits the recap into the top-50 digest."""
    hit, name = claude_analyst._looks_like_recap_template({"title": title})
    assert hit, f"briefing gate missed: {title!r}"
    assert name == "fund_makes_investment"


# ── 2. Must-survive corpus — real headlines NEVER caught ───────────────────


@pytest.mark.parametrize(
    "title",
    [
        # No LLC suffix — sovereign / individual / insider buys.
        "Berkshire Hathaway takes new stake in Apple",
        "Saudi fund makes $5B investment in semis",
        "Tesla insiders bought 100,000 shares",
        "BlackRock Holdings now 12.3M shares of NVDA",
        # LLC mentioned but NOT in the leading-subject + verb position.
        "Apple Inc reports earnings beat; major LLC holders take note",
        "MU Q3 earnings beat consensus by 12% — LLC funds rotate",
        # Forward-looking news with $X figures.
        "Federal Reserve announces $500B liquidity facility",
        "Apple unveils $100B buyback program",
        # Other 13F-mill forms — already caught by sibling fingerprints.
        # (heres_what_means / shares_bought_by / holdings_by_fund); not this gate.
    ],
)
def test_alert_gate_must_survive(title: str) -> None:
    """Real headlines must NOT match ``fund_makes_investment``."""
    _, name = alert_agent._looks_like_recap_template({"title": title})
    if name == "fund_makes_investment":
        pytest.fail(f"fund_makes_investment false-positive: {title!r}")


# ── 3. End-to-end filter integration ────────────────────────────────────────


def test_filter_recap_template_noise_separates_correctly() -> None:
    recap = {
        "_id": "r1",
        "title": "Torren Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA",
        "link": "https://marketbeat.com/x",
        "source": "AlphaVantage/MarketBeat",
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
    assert suppressed[0]["_recap_fingerprint"] == "fund_makes_investment"
    assert "_recap_fingerprint" not in recap


# ── 4. Lockstep parity ───────────────────────────────────────────────────────


def test_lockstep_parity_on_canonical_torren_row() -> None:
    title = "Torren Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA"
    a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
    b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
    assert a_hit and b_hit
    assert a_name == b_name == "fund_makes_investment"
