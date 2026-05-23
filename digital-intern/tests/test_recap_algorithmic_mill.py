"""Recap-template gates v2 — algorithmic-mill fingerprints.

Three new fingerprints added on top of the existing 17 in
``_RECAP_TEMPLATE_PATTERNS`` based on the 2026-05-23 live noise audit
(48h articles.db scan of urgency=2 rows with ``score_source='ml'``):

  1. ``holdings_by_fund`` — MarketBeat / americanbankingnews 13F press mill
     ("Applied Materials, Inc. $AMAT Holdings Raised by Global Retirement
     Partners LLC - MarketBeat"). Fires once per 13F filing change against
     any tracked ticker.

  2. ``futures_why_today`` — TipRanks daily pre-market futures-state recap
     mill ("Why Are Stock Market Futures Down Today, 5/21/26? - TipRanks").
     Same retrospective shape as ``why_trading_today`` but the subject is
     index futures.

  3. ``daily_price_city`` — Business Today (India) daily city-by-city
     commodity-price feed ("Gold Rate Today in Kolkata 21st May 2026 :
     22 & 24 Carat, Gold Price in Kolkata - Business Today"). Not even
     US-market relevant.

Each gate is high-precision: distinctive structural anchors validated
against the live noise corpus AND a must-survive corpus of real wire
headlines that legitimately mention the same tokens mid-sentence
(e.g. "Berkshire trims AAPL holdings", "Stock futures edge higher
ahead of NVDA earnings", "Gold prices rally on Fed minutes").
"""
from __future__ import annotations

import pytest

from watchers.alert_agent import (
    _looks_like_recap_template,
    _filter_recap_template_noise,
    _RECAP_TEMPLATE_PATTERNS,
)


def _art(title):
    return {"title": title}


class TestHoldingsByFundFingerprint:
    """MarketBeat 13F institutional-holdings press mill."""

    @pytest.mark.parametrize("title", [
        "Applied Materials, Inc. $AMAT Holdings Raised by Global Retirement Partners LLC - MarketBeat",
        "NVIDIA Corporation $NVDA Holdings Cut by Vanguard Group LLC",
        "Micron Technology Holdings Boosted by State Street Corp LLC - MarketBeat",
        "Microsoft Holdings Trimmed by Capital Research and Management LLC",
        "Tesla Inc $TSLA Holdings Lowered by JP Morgan Asset Management LLC",
        "ORACLE Holdings Increased by Renaissance Technologies LLC",
    ])
    def test_holdings_press_mill_titles_match(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert hit, f"holdings-by-fund mill leaked: {title!r}"
        assert fp == "holdings_by_fund"

    @pytest.mark.parametrize("title", [
        "Berkshire Hathaway trims AAPL holdings",            # no "by ... LLC"
        "Saudi fund increases Tesla stake",                  # no "holdings"
        "Fund's holdings now 12.3M shares",                  # no "<verb> by ... LLC"
        "Nvidia holdings reach record high after Q1 beat",   # no fund + LLC
        "Bank of America holdings questioned by analysts",   # "by analysts" not fund LLC
    ])
    def test_must_survive_holdings_mentions(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert not hit, f"holdings_by_fund regex false-positive on {title!r} (matched={fp})"


class TestFuturesWhyTodayFingerprint:
    """TipRanks daily pre-market futures-state recap mill."""

    @pytest.mark.parametrize("title", [
        "Why Are Stock Market Futures Down Today, 5/21/26? - TipRanks",
        "Why Are Stock Market Futures Up Today",
        "Why Are Stock Market Futures Higher Today, May 22?",
        "Why Are Stock Market Futures Mixed Today, 5/23/26",
        "Why Are Stock Market Futures Sliding Today",
    ])
    def test_futures_recap_titles_match(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert hit, f"futures-why-today mill leaked: {title!r}"
        assert fp == "futures_why_today"

    @pytest.mark.parametrize("title", [
        "Stock futures edge higher ahead of NVDA earnings",  # not "Why Are"
        "Pre-market: Futures rally 0.8%",                    # not "Why Are"
        "Fed minutes due; futures rise",                     # mid-sentence
        "Why investors are buying futures today",            # different "Why" frame
        "Why are NVDA shares falling today",                 # not market futures
        "Why are bond yields up today",                      # bonds not stock market futures
    ])
    def test_must_survive_futures_mentions(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert not hit, f"futures_why_today false-positive on {title!r} (matched={fp})"


class TestDailyPriceCityFingerprint:
    """Business Today (India) daily city-by-city commodity-price feed."""

    @pytest.mark.parametrize("title", [
        "Gold Rate Today in Kolkata 21st May 2026 : 22 & 24 Carat, Gold Price in Kolkata - Business Today",
        "Gold Rate Today in Kanpur 21st May 2026 : 22 & 24 Carat",
        "Silver Price Today in Mumbai",
        "Petrol Price Today in Delhi 22nd May 2026",
        "Diesel Rate Today in Bengaluru",
        "Crude Oil Price Today in Chennai",
    ])
    def test_daily_price_mill_titles_match(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert hit, f"daily-price-city mill leaked: {title!r}"
        assert fp == "daily_price_city"

    @pytest.mark.parametrize("title", [
        "Gold prices rally on Fed minutes",                  # no "Today in <city>"
        "Oil futures gap higher",                            # different shape
        "Silver surges 8% to 14-year high",                  # no "Today in <city>"
        "Petrol prices ease as crude tumbles",               # no "Today in <city>"
        "Why is gold rallying today",                        # different lead word
    ])
    def test_must_survive_commodity_mentions(self, title):
        hit, fp = _looks_like_recap_template(_art(title))
        assert not hit, f"daily_price_city false-positive on {title!r} (matched={fp})"


class TestNewFingerprintsRegistered:
    """SSOT pin: the new fingerprint names must appear in the canonical
    ``_RECAP_TEMPLATE_PATTERNS`` tuple, otherwise ``_filter_recap_template_
    noise`` cannot see them and the gates are inert."""

    def test_holdings_by_fund_registered(self):
        assert "holdings_by_fund" in [n for n, _ in _RECAP_TEMPLATE_PATTERNS]

    def test_futures_why_today_registered(self):
        assert "futures_why_today" in [n for n, _ in _RECAP_TEMPLATE_PATTERNS]

    def test_daily_price_city_registered(self):
        assert "daily_price_city" in [n for n, _ in _RECAP_TEMPLATE_PATTERNS]


class TestFilterFunctionPartitions:
    """End-to-end: ``_filter_recap_template_noise`` correctly partitions a
    mixed urgent batch into (kept, suppressed) with the new fingerprints
    AND tags the suppressed rows with ``_recap_fingerprint`` for log clarity
    (same shape as the existing gate tagging discipline)."""

    def test_mixed_batch_partitions_correctly(self):
        urgent = [
            _art("Fed surprises with 50bp emergency cut"),                # real
            _art("MU earnings blow past Q3 estimates"),                   # real
            _art("Applied Materials, Inc. $AMAT Holdings Raised by Global Retirement Partners LLC - MarketBeat"),  # mill
            _art("Why Are Stock Market Futures Down Today, 5/21/26? - TipRanks"),  # mill
            _art("Gold Rate Today in Kolkata 21st May 2026"),             # mill
            _art("Nvidia announces $80B buyback"),                        # real
        ]
        kept, suppressed = _filter_recap_template_noise(urgent)
        kept_titles = {a["title"] for a in kept}
        sup_titles = {a["title"] for a in suppressed}
        assert kept_titles == {
            "Fed surprises with 50bp emergency cut",
            "MU earnings blow past Q3 estimates",
            "Nvidia announces $80B buyback",
        }
        assert len(suppressed) == 3
        for a in suppressed:
            assert a.get("_recap_fingerprint"), (
                "suppressed row missing _recap_fingerprint tag — "
                "operator can't see which template fired"
            )
        # Each suppressed title pinned to its expected fingerprint.
        fp_by_title = {a["title"]: a["_recap_fingerprint"] for a in suppressed}
        assert fp_by_title["Applied Materials, Inc. $AMAT Holdings Raised by Global Retirement Partners LLC - MarketBeat"] == "holdings_by_fund"
        assert fp_by_title["Why Are Stock Market Futures Down Today, 5/21/26? - TipRanks"] == "futures_why_today"
        assert fp_by_title["Gold Rate Today in Kolkata 21st May 2026"] == "daily_price_city"

    def test_filter_does_not_mutate_input(self):
        original = _art("Gold Rate Today in Kolkata 21st May 2026")
        original_copy = dict(original)
        kept, suppressed = _filter_recap_template_noise([original])
        # The caller's row must be byte-unchanged (shallow-copy discipline).
        assert original == original_copy
        # The suppressed list carries a tagged copy, not the same object.
        assert suppressed[0] is not original
        assert "_recap_fingerprint" in suppressed[0]
        assert "_recap_fingerprint" not in original


class TestUrgencyScorerCrossGateConsistency:
    """The urgency_scorer pre-floor and the alert-side gate use the SAME
    helper (``_looks_like_recap_template`` imported into urgency_scorer from
    alert_agent). If a recap template is added in one place but the import
    is removed from the other, the pre-floor would let the row through and
    the alert path would also miss it. This pins the cross-module SSOT."""

    def test_urgency_scorer_imports_same_helper(self):
        from watchers import urgency_scorer
        from watchers import alert_agent
        assert urgency_scorer._looks_like_recap_template is \
            alert_agent._looks_like_recap_template, (
            "urgency_scorer and alert_agent must use the SAME recap-template "
            "helper — drift would silently disable the pre-floor"
        )
