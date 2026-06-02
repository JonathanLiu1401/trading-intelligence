"""Recap gate: "<Fund> LLC (Has|Owns|Grows|Trims|Sells N Shares of|...)
(Stock )?Holdings/Shares/Position/Stake in <Co>" — the MarketBeat /
AlphaVantage 13F QUARTERLY-CHANGE press-mill sibling of
``fund_makes_investment``. Same retrospective non-event as the existing
``holdings_by_fund`` (trailing-LLC) / ``shares_bought_by`` ("Shares ... by
LLC") / ``fund_makes_investment`` (leading-LLC, initial position) gates, but
with the closed set of DELTA-ACTION verbs (Grows/Trims/Sells/Buys/Has ...)
instead of initial-position verbs (Makes/Acquires/Takes). The two
``fund_makes_investment`` and ``fund_stake_delta`` siblings have disjoint
verb sets, so a single title is fingerprinted by exactly one of them.

Live evidence (2026-05-28..29, articles.db urgency>=1 scan):
  - "Kingsview Wealth Management LLC Has $31.47 Million Stock Holdings in
     Oracle Corporation $O" (ml_score=9.0, urgency=1, GoogleNews/MarketBeat)
  - "Leeward Financial Partners LLC Grows Stock Holdings in Oracle
     Corporation $ORCL" (ml_score=9.82, urgency=2 — FIRED A REAL BREAKING
     ALERT, GoogleNews/MarketBeat)
  - "Jackson Creek Investment Advisors LLC Sells 3,338 Shares of Lam
     Research Corporation $LRCX" (ml_score=9.64, urgency=1,
     GoogleNews/MarketBeat)

Tests verify:
  1. The new fingerprint catches the live noise variants — Has/Grows/Sells
     across "Stock Holdings"/"Shares"/"Position"/"Stake" nouns, with and
     without dollar magnitudes.
  2. The discriminator is the leading ``<Fund> LLC`` + delta-verb +
     stake-noun + in/of — real news that mentions a fund + stake WITHOUT
     the precise LLC-anchored template is NOT caught.
  3. The lockstep mirror in ``analysis.claude_analyst`` catches the same
     set (anti-drift discipline).
  4. End-to-end ``_filter_recap_template_noise`` tags suppressed rows with
     ``_recap_fingerprint='fund_stake_delta'``.
  5. The two siblings (``fund_makes_investment`` vs ``fund_stake_delta``)
     have disjoint verb sets — a Makes/Acquires title never fingerprints as
     ``fund_stake_delta`` and a Grows/Trims title never as
     ``fund_makes_investment``.
"""
from __future__ import annotations

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. The new fingerprint catches the live noise ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # The three canonical live rows that reached urgency>=1.
        "Kingsview Wealth Management LLC Has $31.47 Million Stock Holdings in Oracle Corporation $O",
        "Leeward Financial Partners LLC Grows Stock Holdings in Oracle Corporation $ORCL",
        "Jackson Creek Investment Advisors LLC Sells 3,338 Shares of Lam Research Corporation $LRCX",
        # Delta-verb coverage — same template across the closed verb list.
        "Vanguard Group LLC Buys 1,234 Shares of NVIDIA Corporation $NVDA",
        "BlackRock Asset Management LLC Trims Stock Holdings in Apple Inc",
        "Citadel Advisors LLC Boosted Position in Tesla",
        "Renaissance Technologies LLC Cuts Stake in Microsoft",
        "Bridgewater Associates LLC Reduced Holdings in MU",
        "Two Sigma Investments LLC Increases Position in AMD",
        "Acme Capital LLC Owns $5 Million Position in Tesla",
        "Bar Asset Management LLC Initiated Position in Lumentum",
        # Bare number (no $, no Million/Billion qualifier).
        "Alpha Beta LLC Buys 500 Shares of NVDA",
        # Multi-word fund-name prefix (up to 5 tokens).
        "Cohen and Steers Capital Management LLC Boosts Position in Vornado",
    ],
)
def test_alert_gate_catches_live_noise(title: str) -> None:
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit, f"missed MarketBeat 13F-delta mill title: {title!r}"
    assert name == "fund_stake_delta", (
        f"expected fund_stake_delta fingerprint, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Kingsview Wealth Management LLC Has $31.47 Million Stock Holdings in Oracle Corporation $O",
        "Leeward Financial Partners LLC Grows Stock Holdings in Oracle Corporation $ORCL",
        "Jackson Creek Investment Advisors LLC Sells 3,338 Shares of Lam Research Corporation $LRCX",
        "BlackRock Asset Management LLC Trims Stock Holdings in Apple Inc",
    ],
)
def test_briefing_gate_catches_live_noise(title: str) -> None:
    """Lockstep mirror — same titles must be caught by the briefing path
    gate or the per-domain cap admits the recap into the top-50 digest."""
    hit, name = claude_analyst._looks_like_recap_template({"title": title})
    assert hit, f"briefing gate missed: {title!r}"
    assert name == "fund_stake_delta"


# ── 2. Must-survive corpus — real headlines NEVER caught ───────────────────


@pytest.mark.parametrize(
    "title",
    [
        # No LLC suffix — sovereign / individual / insider buys.
        "Berkshire Hathaway takes new stake in Apple",
        "Saudi fund grows stake in semis",
        "Tesla insiders bought 100,000 shares",
        "BlackRock Holdings now 12.3M shares of NVDA",
        # LLC mentioned mid-sentence WITHOUT a verb in the closed delta-action
        # list (these survived the existing fund_makes_investment gate, must
        # also survive this one).
        "Apple Inc reports earnings beat; major LLC holders take note",
        "MU Q3 earnings beat consensus by 12% — LLC funds rotate",
        # Forward-looking news with $X figures — no LLC.
        "Federal Reserve announces $500B liquidity facility",
        "Apple unveils $100B buyback program",
        # Real "stake in" wire copy without the LLC-template pattern.
        "Government considers stake in struggling utility",
        "Activist investor builds stake in laggard retailer",
        # Briefing-survivor corpus from
        # test_alert_and_briefing_gates_agree_on_must_survive_corpus.
        "MU earnings blow past Q3 estimates sharply",
        "Fed cuts rates by 50bp, citing labor weakness",
        "MU shares halted on pending news",
        "Why investors are bullish on Nvidia ahead of earnings",
        "Nvidia Q1 earnings preview: all eyes on data center",
        "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
        "PORTFOLIO P&L SNAPSHOT",
    ],
)
def test_alert_gate_must_survive(title: str) -> None:
    """Real headlines must NOT match ``fund_stake_delta``."""
    _, name = alert_agent._looks_like_recap_template({"title": title})
    if name == "fund_stake_delta":
        pytest.fail(f"fund_stake_delta false-positive: {title!r}")


# ── 3. End-to-end filter integration ────────────────────────────────────────


def test_filter_recap_template_noise_separates_correctly() -> None:
    recap = {
        "_id": "r1",
        "title": "Leeward Financial Partners LLC Grows Stock Holdings in Oracle Corporation $ORCL",
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
    assert suppressed[0]["_recap_fingerprint"] == "fund_stake_delta"
    # Pure helper must not mutate the caller's row.
    assert "_recap_fingerprint" not in recap


# ── 4. Lockstep parity ───────────────────────────────────────────────────────


def test_lockstep_parity_on_canonical_leeward_row() -> None:
    title = "Leeward Financial Partners LLC Grows Stock Holdings in Oracle Corporation $ORCL"
    a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
    b_hit, b_name = claude_analyst._looks_like_recap_template({"title": title})
    assert a_hit and b_hit
    assert a_name == b_name == "fund_stake_delta"


# ── 5. Disjoint verb sets — siblings never collide ──────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        # Initial-position verbs — must fingerprint as fund_makes_investment.
        "Torren Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation $NVDA",
        "Acme Capital LLC Acquired $5 Million Position in Tesla",
        "BlackRock Asset Management LLC Took $50 Million Stake in Microsoft",
        "Vanguard Group LLC Made $1.2 Billion Position in NVIDIA",
    ],
)
def test_makes_investment_titles_do_not_fingerprint_as_stake_delta(title: str) -> None:
    """``fund_makes_investment`` titles must not be re-labeled as
    ``fund_stake_delta`` (disjoint verb sets — Makes/Acquires/Takes vs
    Grows/Trims/Sells/Has). Tested via the alert path; the briefing mirror is
    byte-identical."""
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit
    assert name == "fund_makes_investment", (
        f"expected fund_makes_investment, got {name!r}"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Leeward Financial Partners LLC Grows Stock Holdings in Oracle Corporation $ORCL",
        "Citadel Advisors LLC Boosted Position in Tesla",
        "Renaissance Technologies LLC Cuts Stake in Microsoft",
    ],
)
def test_stake_delta_titles_do_not_fingerprint_as_makes_investment(title: str) -> None:
    """Same disjoint-set check, the other direction."""
    hit, name = alert_agent._looks_like_recap_template({"title": title})
    assert hit
    assert name == "fund_stake_delta"


# ── 6. Structural anti-drift (alert ↔ briefing tuple parity) ────────────────


def test_alert_and_briefing_recap_tuples_have_same_length_after_addition() -> None:
    """Re-state the global structural anti-drift check: adding the new
    fingerprint to one tuple but not the other must fail to import. This
    duplicates ``test_briefing_recap_template.test_alert_and_briefing_recap_
    tuples_have_same_length`` deliberately — having a per-feature copy makes
    a future revert of one side immediately obvious in the per-fingerprint
    test file rather than only in the global anti-drift test."""
    from watchers.alert_agent import _RECAP_TEMPLATE_PATTERNS
    from analysis.claude_analyst import _BRIEFING_RECAP_TEMPLATE_PATTERNS
    assert len(_RECAP_TEMPLATE_PATTERNS) == len(_BRIEFING_RECAP_TEMPLATE_PATTERNS)
    assert "fund_stake_delta" in {name for name, _ in _RECAP_TEMPLATE_PATTERNS}
    assert "fund_stake_delta" in {name for name, _ in _BRIEFING_RECAP_TEMPLATE_PATTERNS}
