"""Unit tests for analytics.off_watchlist_mentions.build_off_watchlist_mentions.

Pins the WATCHLIST + held exclusion, exact heat arithmetic (mirrors
watchlist_opportunities so the two surfaces stay directly comparable),
threshold gates, and the malformed-input degrade-soft contract.

A leaked watchlist ticker would falsely flag the operator that they're
missing a name they're already watching — these assertions catch that.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.off_watchlist_mentions import (
    build_off_watchlist_mentions)


def _a(tickers, score, urgency=0, title=None, source="Reuters", url=None):
    tk_list = [tickers] if isinstance(tickers, str) else list(tickers)
    name = tk_list[0] if tk_list else "X"
    return {
        "tickers": tk_list,
        "ai_score": score,
        "urgency": urgency,
        "title": title or f"{name} headline {score}",
        "source": source,
        "url": url or f"http://x/{name}/{score}",
    }


def _heat(max_score, n, urgent):
    return max_score * (1.0 + math.log1p(n) / 3.0) * (1.0 + 0.25 * urgent)


class TestExclusionRules:
    def test_watchlist_tickers_dropped(self):
        sigs = [_a("NVDA", 9.0, 1),       # on watchlist → must NOT appear
                _a("KWEB", 8.5, 1),
                _a("BIDU", 7.5)]
        out = build_off_watchlist_mentions(
            ["NVDA", "AMD", "SPY"], set(), sigs)
        tickers = [d["ticker"] for d in out["discoveries"]]
        assert "NVDA" not in tickers
        assert "KWEB" in tickers
        assert "BIDU" in tickers
        assert out["watchlist_size"] == 3

    def test_held_tickers_dropped(self):
        # A held off-watchlist ADR is already position-tracked; should not
        # be surfaced as a "discovery".
        sigs = [_a("KWEB", 9.0, 1), _a("BIDU", 8.0, 1)]
        out = build_off_watchlist_mentions(
            ["NVDA"], {"KWEB"}, sigs)
        tickers = [d["ticker"] for d in out["discoveries"]]
        assert tickers == ["BIDU"]
        assert out["held_size"] == 1

    def test_case_insensitivity_for_exclusions(self):
        sigs = [_a("kweb", 9.0, 1)]  # lowercase input
        out = build_off_watchlist_mentions(
            ["nvda"], {"kweb"}, sigs)
        assert out["discoveries"] == []
        # The article's lowercase ticker was upper-cased; held set was too.

    def test_surface_specific_noise_filter_strips_known_false_positives(self):
        # MSN / TSX / EUV / CAPEX / EPYC / NA / AI are SSOT extraction
        # false-positives on the unbounded off-watchlist path. A KWEB
        # mention in the same batch must still surface — the filter is
        # additive, not a kill-switch.
        sigs = [
            _a("MSN", 9.0, 1),     # "…- MSN" article-suffix pollution
            _a("TSX", 9.0, 1),     # "$BB.TSX" exchange suffix split
            _a("EUV", 9.0, 1),     # extreme-UV lithography
            _a("CAPEX", 9.0, 1),   # capital expenditure word
            _a("EPYC", 9.0, 1),    # AMD chip line
            _a("NA", 9.0, 1),      # not applicable
            _a("AI", 9.0, 1),      # very high collision
            _a("KWEB", 9.0, 1),
        ]
        out = build_off_watchlist_mentions(["NVDA"], set(), sigs)
        tickers = [d["ticker"] for d in out["discoveries"]]
        assert tickers == ["KWEB"]


class TestRanking:
    def test_heat_formula_matches_watchlist_opportunities(self):
        # Same formula is the explicit design — operator can compare the
        # two surfaces directly. This is the regression pin.
        sigs = [_a("KWEB", 9.0, 1), _a("KWEB", 8.0), _a("KWEB", 7.0),
                _a("BIDU", 5.0)]
        out = build_off_watchlist_mentions(["NVDA"], set(), sigs)
        kweb = out["discoveries"][0]
        bidu = out["discoveries"][1]
        assert kweb["ticker"] == "KWEB"
        assert kweb["n_articles"] == 3
        assert kweb["urgent"] == 1
        assert kweb["avg_score"] == pytest.approx((9 + 8 + 7) / 3)
        assert kweb["max_score"] == 9.0
        assert kweb["heat"] == pytest.approx(round(_heat(9.0, 3, 1), 3))
        assert bidu["heat"] == pytest.approx(round(_heat(5.0, 1, 0), 3))
        assert kweb["heat"] > bidu["heat"]
        assert kweb["top_headline"] == "KWEB headline 9.0"

    def test_multi_ticker_article_credits_all(self):
        # A China-AI rotation headline mentioning both KWEB + BIDU lifts
        # BOTH tickers — same anti-duplication discipline.
        sigs = [_a(["KWEB", "BIDU"], 9.0, 1, title="China AI rotates")]
        out = build_off_watchlist_mentions(["NVDA"], set(), sigs,
                                            min_avg_score=0.0)
        tickers = {d["ticker"] for d in out["discoveries"]}
        assert tickers == {"KWEB", "BIDU"}


class TestThresholds:
    def test_min_articles_floor(self):
        sigs = [_a("KWEB", 9.0), _a("KWEB", 9.0), _a("BIDU", 9.0)]
        out = build_off_watchlist_mentions(
            ["NVDA"], set(), sigs, min_articles=2)
        assert [d["ticker"] for d in out["discoveries"]] == ["KWEB"]

    def test_min_avg_score_floor(self):
        sigs = [_a("KWEB", 3.0), _a("BIDU", 9.0)]
        out = build_off_watchlist_mentions(
            ["NVDA"], set(), sigs, min_avg_score=4.0)
        assert [d["ticker"] for d in out["discoveries"]] == ["BIDU"]

    def test_limit_caps_result(self):
        sigs = [_a("KWEB", 9.0), _a("BIDU", 8.0), _a("BABA", 7.0)]
        out = build_off_watchlist_mentions(
            ["NVDA"], set(), sigs, limit=2)
        assert len(out["discoveries"]) == 2

    def test_zero_signals_returns_empty(self):
        out = build_off_watchlist_mentions(["NVDA"], set(), [])
        assert out["discoveries"] == []
        assert out["n_scanned_articles"] == 0
        assert out["n_unique_off_watch"] == 0


class TestNeverRaises:
    def test_malformed_rows_are_skipped(self):
        sigs = [
            "junk",                             # not a dict
            None,
            {"tickers": "KWEB"},                # tickers is str not list
            {"tickers": ["KWEB"], "ai_score": "bad"},  # bad numeric
            {"tickers": ["BIDU"], "ai_score": 8.0},
        ]
        out = build_off_watchlist_mentions(
            ["NVDA"], set(), sigs, min_avg_score=0.0)
        tickers = {d["ticker"] for d in out["discoveries"]}
        # KWEB row with bad ai_score still credits (score floors to 0.0);
        # BIDU is the clean row; junk/None/str-tickers are dropped.
        assert "BIDU" in tickers

    def test_empty_watchlist_treats_everything_as_discoverable(self):
        sigs = [_a("AAPL", 9.0)]
        out = build_off_watchlist_mentions([], set(), sigs)
        assert out["discoveries"][0]["ticker"] == "AAPL"

    def test_none_inputs_degrade_to_empty(self):
        out = build_off_watchlist_mentions(None, None, None)
        assert out["discoveries"] == []
        assert out["watchlist_size"] == 0
        assert out["held_size"] == 0
