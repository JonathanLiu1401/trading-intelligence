"""Pure-builder tests for paper_trader/analytics/actionable_opportunities.py.

The composite ranker is pure — it accepts the three input payloads and
returns a stable shape. Every untrusted axis must degrade to "no
contribution" rather than raise. The verdict ladder must match the
docstring contract exactly so the chat helper can rely on it.

The route-level wiring (intern fetch guard, SWR, scorer/persistent sub-
fetch) is exercised in test_actionable_opportunities_endpoint.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.actionable_opportunities import (  # noqa: E402
    MIN_PERSISTENT_H,
    MIN_PRED_PP,
    STRONG_PRED_PP,
    _classify_actionability,
    _composite_score,
    build_actionable_opportunities,
)


def _scorer(opps, is_trained=True, n_train=3914, gate=500):
    return {
        "is_trained": is_trained, "n_train": n_train,
        "gate_threshold": gate, "opportunities": opps,
    }


def _burst(by_ticker):
    return {"by_ticker": by_ticker}


def _persistent(rows):
    return {"opportunities": rows}


class TestClassification:
    def test_high_conviction_requires_both_strong_pred_and_news_heat(self):
        # Strong pred (15) + WARMING (lowest heat tier) → HIGH_CONVICTION
        assert _classify_actionability(15.0, "WARMING", 0.0) == "HIGH_CONVICTION"
        # Strong pred (15) + COLD → SCORER_ONLY (NOT HIGH_CONVICTION)
        assert _classify_actionability(15.0, "COLD", 0.0) == "SCORER_ONLY"

    def test_news_confirmed_is_mid_pred_with_heat(self):
        # Pred at MIN_PRED_PP (5) + HOT → NEWS_CONFIRMED
        assert _classify_actionability(MIN_PRED_PP, "HOT", 0.0) == "NEWS_CONFIRMED"
        # Same heat but pred BELOW MIN → NEWS_ONLY
        assert _classify_actionability(MIN_PRED_PP - 0.1, "HOT", 0.0) == "NEWS_ONLY"

    def test_persistent_followup_needs_min_pred_and_min_hours(self):
        # MIN_PRED + MIN_PERSISTENT + cold → PERSISTENT_FOLLOWUP
        assert (
            _classify_actionability(
                MIN_PRED_PP, "NORMAL", MIN_PERSISTENT_H)
            == "PERSISTENT_FOLLOWUP"
        )
        # MIN_PRED but persistent_hours below threshold → WEAK
        assert (
            _classify_actionability(
                MIN_PRED_PP, "NORMAL", MIN_PERSISTENT_H - 0.5)
            == "WEAK"
        )

    def test_news_only_requires_strong_heat(self):
        """News-only fires on BLAZING / HOT, NOT WARMING (lower-tier heat
        without quant support is too weak to surface as actionable)."""
        assert _classify_actionability(1.0, "HOT", 0.0) == "NEWS_ONLY"
        assert _classify_actionability(1.0, "BLAZING", 0.0) == "NEWS_ONLY"
        assert _classify_actionability(1.0, "WARMING", 0.0) == "WEAK"

    def test_weak_default(self):
        assert _classify_actionability(0.0, "NORMAL", 0.0) == "WEAK"
        assert _classify_actionability(2.0, "COLD", 1.0) == "WEAK"


class TestCompositeScore:
    def test_composite_combines_all_three_axes(self):
        # 10pp pred + HOT (numeric 4.0) + 8h → 10*1 + 4*2 + 8*0.5 = 22
        s = _composite_score(10.0, "HOT", 8.0)
        assert s == 22.0

    def test_zero_axes_yield_zero(self):
        assert _composite_score(0.0, "NORMAL", 0.0) == 0.0

    def test_negative_pred_dominates_when_no_burst(self):
        # A scorer EXIT (negative pred) must NOT be hidden by a small
        # positive burst — sorting by composite must surface it last.
        s_exit = _composite_score(-5.0, "NORMAL", 0.0)
        s_strong = _composite_score(15.0, "NORMAL", 0.0)
        assert s_exit < s_strong

    def test_negative_persistent_clamped(self):
        # Negative persistent_hours (garbage upstream) must clamp at 0 —
        # cannot subtract from composite.
        s = _composite_score(5.0, "NORMAL", -100.0)
        assert s == _composite_score(5.0, "NORMAL", 0.0)


class TestBuilderGate:
    def test_untrained_scorer_yields_insufficient_data(self):
        out = build_actionable_opportunities(
            _scorer([], is_trained=False, n_train=0),
            _burst([]), _persistent([]),
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["by_ticker"] == []
        assert out["n_high_conviction"] == 0

    def test_below_gate_threshold_yields_insufficient_data(self):
        out = build_actionable_opportunities(
            _scorer([], is_trained=True, n_train=100, gate=500),
            _burst([]), _persistent([]),
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_none_payloads_collapse_to_insufficient(self):
        out = build_actionable_opportunities(None, None, None)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["is_trained"] is False
        assert out["n_train"] == 0

    def test_error_scorer_payload_collapses_to_insufficient(self):
        out = build_actionable_opportunities(
            {"error": "intern down"}, _burst([]), _persistent([]),
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"


class TestVerdictLadder:
    def test_high_conviction_when_strong_pred_meets_heat(self):
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": 26.1,
             "verdict": "STRONG_HOLD"},
        ])
        burst = _burst([{
            "ticker": "AMD", "verdict": "HOT", "spike": 6.0,
            "count_window": 5, "count_baseline": 1,
        }])
        out = build_actionable_opportunities(scorer, burst, _persistent([]))
        assert out["verdict"] == "HIGH_CONVICTION_FOUND"
        assert "AMD" in out["headline"]
        amd = next(r for r in out["by_ticker"] if r["ticker"] == "AMD")
        assert amd["actionability"] == "HIGH_CONVICTION"
        assert "scorer +26.1% predicted 5d return" in amd["reasons"][0]

    def test_scorer_but_no_news_is_the_documented_live_state(self):
        """Live live (2026-05-27 02:49 ET): scorer says STRONG_HOLD on MU,
        AMD, SMH, QQQ but news_burst is COLD on every one. This must surface
        as SCORER_BUT_NO_NEWS, not ALL_QUIET — the operator needs to see
        the disagreement, not a silence."""
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": 26.1,
             "verdict": "STRONG_HOLD"},
            {"ticker": "MU", "pred_5d_return_pct": 24.7,
             "verdict": "STRONG_HOLD"},
        ])
        # Every ticker COLD on the wire.
        burst = _burst([
            {"ticker": "AMD", "verdict": "COLD", "spike": None,
             "count_window": 0, "count_baseline": 0},
            {"ticker": "MU", "verdict": "COLD", "spike": None,
             "count_window": 0, "count_baseline": 0},
        ])
        out = build_actionable_opportunities(scorer, burst, _persistent([]))
        assert out["verdict"] == "SCORER_BUT_NO_NEWS"
        assert "AMD" in out["headline"] or "MU" in out["headline"]
        # AMD and MU both classify SCORER_ONLY.
        actionabilities = {r["ticker"]: r["actionability"]
                            for r in out["by_ticker"]}
        assert actionabilities["AMD"] == "SCORER_ONLY"
        assert actionabilities["MU"] == "SCORER_ONLY"

    def test_news_but_no_scorer_when_only_news_heat(self):
        scorer = _scorer([
            {"ticker": "FOO", "pred_5d_return_pct": 1.5, "verdict": "HOLD"},
        ])
        burst = _burst([
            {"ticker": "FOO", "verdict": "HOT", "spike": 6.0,
             "count_window": 5, "count_baseline": 1},
        ])
        out = build_actionable_opportunities(scorer, burst, _persistent([]))
        assert out["verdict"] == "NEWS_BUT_NO_SCORER"

    def test_all_quiet_when_nothing_qualifies(self):
        scorer = _scorer([
            {"ticker": "FOO", "pred_5d_return_pct": 1.0, "verdict": "HOLD"},
        ])
        burst = _burst([
            {"ticker": "FOO", "verdict": "NORMAL", "spike": 1.0,
             "count_window": 1, "count_baseline": 24},
        ])
        out = build_actionable_opportunities(scorer, burst, _persistent([]))
        assert out["verdict"] == "ALL_QUIET"

    def test_persistent_followup_verdict_when_quant_and_persistence(self):
        scorer = _scorer([
            {"ticker": "FOO", "pred_5d_return_pct": 6.0, "verdict": "STRONG_HOLD"},
        ])
        burst = _burst([
            {"ticker": "FOO", "verdict": "NORMAL", "spike": 1.0,
             "count_window": 1, "count_baseline": 12},
        ])
        persistent = _persistent([
            {"ticker": "FOO", "current_run_hours": 10.0},
        ])
        out = build_actionable_opportunities(scorer, burst, persistent)
        assert out["verdict"] == "PERSISTENT_FOLLOWUP"
        foo = next(r for r in out["by_ticker"] if r["ticker"] == "FOO")
        assert foo["actionability"] == "PERSISTENT_FOLLOWUP"
        assert foo["persistent_hours"] == 10.0


class TestSortAndShape:
    def test_sort_by_composite_descending(self):
        scorer = _scorer([
            {"ticker": "LOWPRED", "pred_5d_return_pct": 5.0, "verdict": "STRONG_HOLD"},
            {"ticker": "HIGHPRED", "pred_5d_return_pct": 25.0, "verdict": "STRONG_HOLD"},
            {"ticker": "MIDPRED", "pred_5d_return_pct": 12.0, "verdict": "STRONG_HOLD"},
        ])
        out = build_actionable_opportunities(scorer, _burst([]), _persistent([]))
        order = [r["ticker"] for r in out["by_ticker"]]
        assert order == ["HIGHPRED", "MIDPRED", "LOWPRED"]

    def test_top_n_cap(self):
        scorer = _scorer([
            {"ticker": f"T{i:02d}", "pred_5d_return_pct": 5.0,
             "verdict": "STRONG_HOLD"}
            for i in range(30)
        ])
        out = build_actionable_opportunities(
            scorer, _burst([]), _persistent([]), top_n=5,
        )
        assert len(out["by_ticker"]) == 5

    def test_stable_shape_always_present(self):
        out = build_actionable_opportunities(
            _scorer([]), _burst([]), _persistent([]),
        )
        for key in (
            "generated_at", "verdict", "headline", "is_trained",
            "n_train", "gate_threshold", "n_scored", "n_high_conviction",
            "n_news_confirmed", "by_ticker",
        ):
            assert key in out, key


class TestDegradationOfPartialAxes:
    def test_news_burst_missing_collapses_to_cold(self):
        """If intern is down (burst_payload=None), every ticker should still
        be ranked — just with burst_verdict=COLD. SCORER_BUT_NO_NEWS verdict
        should fire so the operator can see the source-availability issue."""
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": 26.1,
             "verdict": "STRONG_HOLD"},
        ])
        out = build_actionable_opportunities(scorer, None, _persistent([]))
        amd = next(r for r in out["by_ticker"] if r["ticker"] == "AMD")
        assert amd["news_burst_verdict"] == "COLD"
        assert amd["actionability"] == "SCORER_ONLY"
        assert out["verdict"] == "SCORER_BUT_NO_NEWS"

    def test_persistent_missing_yields_zero_hours(self):
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": 6.0,
             "verdict": "STRONG_HOLD"},
        ])
        out = build_actionable_opportunities(scorer, _burst([]), None)
        amd = next(r for r in out["by_ticker"] if r["ticker"] == "AMD")
        assert amd["persistent_hours"] == 0.0

    def test_garbage_pred_string_yields_zero_not_raise(self):
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": "not-a-number",
             "verdict": "STRONG_HOLD"},
        ])
        out = build_actionable_opportunities(
            scorer, _burst([]), _persistent([]),
        )
        amd = next(r for r in out["by_ticker"] if r["ticker"] == "AMD")
        assert amd["scorer_pred_5d_pct"] == 0.0

    def test_off_distribution_flag_passed_through(self):
        scorer = _scorer([
            {"ticker": "AMD", "pred_5d_return_pct": 15.0,
             "verdict": "STRONG_HOLD", "off_distribution": True},
        ])
        out = build_actionable_opportunities(
            scorer, _burst([]), _persistent([]),
        )
        amd = next(r for r in out["by_ticker"] if r["ticker"] == "AMD")
        assert amd["scorer_off_distribution"] is True
