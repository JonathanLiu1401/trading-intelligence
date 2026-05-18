"""Unit tests for analytics.watchlist_opportunities.build_watchlist_opportunities.

Pins exact heat arithmetic, the held-exclusion, and the threshold gates.
A wrong heat formula or a leaked held name would mis-rank the
missed-opportunity radar; these assertions catch that deterministically.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.watchlist_opportunities import (
    build_watchlist_opportunities)


def _a(ticker, score, urgency=0):
    return {"tickers": [ticker], "ai_score": score, "urgency": urgency,
            "title": f"{ticker} headline {score}", "source": "Reuters",
            "url": f"http://x/{ticker}/{score}"}


def _heat(max_score, n, urgent):
    return max_score * (1.0 + math.log1p(n) / 3.0) * (1.0 + 0.25 * urgent)


class TestRankingAndExclusion:
    def test_held_excluded_and_ranked_by_heat(self):
        sigs = [_a("NVDA", 9.0, 1), _a("NVDA", 8.0), _a("NVDA", 7.0),
                _a("AMD", 5.0),
                _a("MU", 9.9, 1),          # held → must not appear
                _a("SPY", 3.0)]            # avg below default 4.0 → filtered
        out = build_watchlist_opportunities(
            ["MU", "NVDA", "AMD", "SPY"], {"MU"}, sigs)
        opps = out["opportunities"]
        assert [o["ticker"] for o in opps] == ["NVDA", "AMD"]
        assert out["n_scanned"] == 3        # NVDA, AMD, SPY (MU held, dropped)
        assert out["n_surfaced"] == 2       # SPY filtered by min_avg_score

        nvda = opps[0]
        assert nvda["n_articles"] == 3
        assert nvda["avg_score"] == pytest.approx((9 + 8 + 7) / 3)
        assert nvda["max_score"] == 9.0
        assert nvda["urgent"] == 1
        assert nvda["heat"] == pytest.approx(round(_heat(9.0, 3, 1), 3))
        assert nvda["top_headline"] == "NVDA headline 9.0"

        amd = opps[1]
        assert amd["heat"] == pytest.approx(round(_heat(5.0, 1, 0), 3))
        assert nvda["heat"] > amd["heat"]

    def test_thresholds_and_limit(self):
        sigs = [_a("NVDA", 9.0), _a("NVDA", 9.0),
                _a("AMD", 4.5), _a("TSM", 4.1)]
        out = build_watchlist_opportunities(
            ["NVDA", "AMD", "TSM"], set(), sigs,
            min_articles=2, limit=1)
        # min_articles=2 drops AMD/TSM (1 each); limit=1 caps the list
        assert [o["ticker"] for o in out["opportunities"]] == ["NVDA"]
        assert out["n_surfaced"] == 1

    def test_no_signals_yields_empty(self):
        out = build_watchlist_opportunities(["NVDA", "AMD"], set(), [])
        assert out["opportunities"] == []
        assert out["n_surfaced"] == 0
        assert out["n_scanned"] == 2


class TestNeverRaises:
    def test_malformed_signal_rows_are_ignored(self):
        sigs = ["junk", {"tickers": "NVDA"}, {"tickers": ["NVDA"],
                                              "ai_score": 8.0}]
        out = build_watchlist_opportunities(["NVDA"], set(), sigs,
                                            min_avg_score=0.0)
        assert out["opportunities"][0]["ticker"] == "NVDA"
        assert out["opportunities"][0]["n_articles"] == 1  # the string-tickers row ignored
