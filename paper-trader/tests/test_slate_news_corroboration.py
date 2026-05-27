"""Tests for the slate × news corroboration builder.

The builder is pure (no DB / no clock) so the tests are deterministic
fixtures over the same input shapes the live ``/api/scorer-opportunities``
and ``dashboard._ticker_news_pulse`` produce.

Coverage:

* Per-name verdict ladder: HOT_CONVERGENT / CONVERGENT / THIN_NEWS /
  QUANT_ONLY / SUB_THRESHOLD with the exact threshold pinned.
* Cohort verdict ladder: STRONG_CORROBORATION / QUANT_LEAD / THIN /
  MIXED_CORROBORATION / NO_SLATE.
* Garbage-input never-raises contract (None, [], non-dicts, missing keys).
* Pred-floor honoured against the cohort verdict.
* Buy-candidate filter excludes EXIT/NEUTRAL scorer verdicts.
* Flask endpoint contract via the test_client (matches the
  analytics-verification memory: live endpoint shape needs the in-process
  Flask app, not a module __main__).
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.slate_news_corroboration import (
    build_slate_news_corroboration,
    DEFAULT_MIN_PRED_PCT,
    DEFAULT_HOT_MIN_COUNT,
    DEFAULT_HOT_MIN_SCORE,
    DEFAULT_CONVERGENT_MIN_COUNT,
    DEFAULT_CONVERGENT_MIN_SCORE,
    _classify_one,
)


def _opp(ticker: str, pred: float = 5.0, verdict: str = "STRONG_HOLD") -> dict:
    return {
        "ticker": ticker,
        "pred_5d_return_pct": pred,
        "verdict": verdict,
    }


def _pulse(n: int = 0, urgent: int = 0, max_score: float = 0.0,
           title: str | None = None, url: str | None = None) -> dict:
    return {
        "n": n,
        "urgent": urgent,
        "top_score": max_score,
        "top_title": title,
        "top_url": url,
    }


class TestClassifyOne:
    def test_sub_threshold_below_floor(self):
        assert _classify_one(
            pred_pct=0.5, n=10, urgent=5, max_score=9.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "SUB_THRESHOLD"

    def test_quant_only_zero_news(self):
        assert _classify_one(
            pred_pct=5.0, n=0, urgent=0, max_score=0.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "QUANT_ONLY"

    def test_hot_convergent_urgent(self):
        # One urgent article alone qualifies as HOT (urgency=1 is itself a
        # high-signal flag — see the dashboard alert pipeline).
        assert _classify_one(
            pred_pct=5.0, n=1, urgent=1, max_score=5.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "HOT_CONVERGENT"

    def test_hot_convergent_count_and_score(self):
        assert _classify_one(
            pred_pct=5.0, n=4, urgent=0, max_score=7.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "HOT_CONVERGENT"

    def test_convergent_count_and_moderate_score(self):
        assert _classify_one(
            pred_pct=5.0, n=2, urgent=0, max_score=4.5,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "CONVERGENT"

    def test_thin_news_single_article(self):
        assert _classify_one(
            pred_pct=5.0, n=1, urgent=0, max_score=3.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "THIN_NEWS"

    def test_thin_news_count_ok_but_score_below_floor(self):
        # 2 articles but max score 3.0 < convergent_min_score 4.0 → THIN
        assert _classify_one(
            pred_pct=5.0, n=2, urgent=0, max_score=3.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "THIN_NEWS"

    def test_floor_boundary_inclusive(self):
        # pred == floor → still SUB_THRESHOLD per strict less-than gate?
        # No — the gate is `pred < floor`, so pred == floor is NOT sub.
        assert _classify_one(
            pred_pct=1.0, n=0, urgent=0, max_score=0.0,
            min_pred_pct=1.0,
            hot_min_count=3, hot_min_score=6.0,
            convergent_min_count=2, convergent_min_score=4.0,
        ) == "QUANT_ONLY"


class TestOverallVerdict:
    def test_no_slate_empty(self):
        r = build_slate_news_corroboration([], {})
        assert r["verdict"] == "NO_SLATE"
        assert r["n_total"] == 0
        assert r["headline"] == "no scorer opportunities — nothing to corroborate"
        # Every count bucket present and zeroed.
        for k in ("HOT_CONVERGENT", "CONVERGENT", "THIN_NEWS",
                  "QUANT_ONLY", "SUB_THRESHOLD"):
            assert r["counts"][k] == 0

    def test_no_slate_when_all_sub_threshold(self):
        # Every opportunity below the pred floor → no buy candidates
        opps = [_opp("AAA", pred=0.3), _opp("BBB", pred=0.5)]
        r = build_slate_news_corroboration(opps, {})
        assert r["verdict"] == "NO_SLATE"
        assert r["n_total"] == 2
        assert r["n_buy_candidates"] == 0
        assert r["counts"]["SUB_THRESHOLD"] == 2

    def test_strong_corroboration_when_majority_hot_or_convergent(self):
        opps = [
            _opp("AAA", pred=5.0),
            _opp("BBB", pred=4.0),
            _opp("CCC", pred=3.0),
        ]
        pulse = {
            "AAA": _pulse(n=4, urgent=1, max_score=8.0, title="up", url="u"),  # HOT
            "BBB": _pulse(n=3, urgent=0, max_score=7.0, title="b", url="u"),   # HOT
            "CCC": _pulse(n=2, urgent=0, max_score=5.0, title="c", url="u"),   # CONVERGENT
        }
        r = build_slate_news_corroboration(opps, pulse)
        assert r["verdict"] == "STRONG_CORROBORATION"
        assert r["cohort_counts"]["HOT_CONVERGENT"] == 2
        assert r["cohort_counts"]["CONVERGENT"] == 1
        assert r["n_buy_candidates"] == 3

    def test_quant_lead_when_majority_quant_only(self):
        opps = [_opp(t) for t in ("A", "B", "C", "D")]
        pulse = {}  # all silent
        r = build_slate_news_corroboration(opps, pulse)
        assert r["verdict"] == "QUANT_LEAD"
        assert r["cohort_counts"]["QUANT_ONLY"] == 4

    def test_thin_when_majority_thin_news(self):
        opps = [_opp(t) for t in ("A", "B", "C")]
        pulse = {
            "A": _pulse(n=1, max_score=2.0),
            "B": _pulse(n=1, max_score=3.0),
            "C": _pulse(n=1, max_score=2.5),
        }
        r = build_slate_news_corroboration(opps, pulse)
        assert r["verdict"] == "THIN"
        assert r["cohort_counts"]["THIN_NEWS"] == 3

    def test_mixed_corroboration_no_majority(self):
        opps = [_opp(t) for t in ("A", "B", "C", "D")]
        pulse = {
            "A": _pulse(n=2, max_score=5.0),   # CONVERGENT
            "B": _pulse(n=1, max_score=2.0),   # THIN
            # C, D silent → QUANT_ONLY
        }
        r = build_slate_news_corroboration(opps, pulse)
        # No HOT and < 50% quant-only and < 50% thin → MIXED.
        # cohort: 0 hot, 1 conv, 1 thin, 2 quant_only — quant_only at 50%
        # exactly → QUANT_LEAD per the >= 50% gate.
        # Test the boundary explicitly:
        assert r["verdict"] == "QUANT_LEAD"

    def test_mixed_corroboration_with_no_majority(self):
        # 5 buy candidates: 1 HOT, 1 CONV (combined 2/5 = 40%, < 50%),
        # 1 THIN (1/5 = 20%), 2 QUANT_ONLY (2/5 = 40%). No bucket hits 50%.
        opps = [_opp(t) for t in ("A", "B", "C", "D", "E")]
        pulse = {
            "A": _pulse(n=4, urgent=1, max_score=8.0),
            "B": _pulse(n=2, max_score=5.0),
            "C": _pulse(n=1, max_score=2.0),
        }
        r = build_slate_news_corroboration(opps, pulse)
        assert r["verdict"] == "MIXED_CORROBORATION"


class TestBuyCandidateFilter:
    def test_exit_verdict_excluded_from_cohort(self):
        # NEUTRAL / TRIM / EXIT scorer verdicts shouldn't drive the
        # cohort ladder.
        opps = [
            _opp("AAA", pred=5.0, verdict="STRONG_HOLD"),
            _opp("BBB", pred=5.0, verdict="EXIT"),
            _opp("CCC", pred=5.0, verdict="TRIM"),
        ]
        r = build_slate_news_corroboration(opps, {})
        # Only AAA in cohort. Its classification: QUANT_ONLY → QUANT_LEAD.
        assert r["n_buy_candidates"] == 1
        assert r["verdict"] == "QUANT_LEAD"

    def test_hold_verdict_in_cohort(self):
        opps = [
            _opp("AAA", pred=5.0, verdict="HOLD"),
            _opp("BBB", pred=5.0, verdict="HOLD"),
        ]
        r = build_slate_news_corroboration(opps, {})
        assert r["n_buy_candidates"] == 2


class TestRobustness:
    def test_none_inputs_no_raise(self):
        r = build_slate_news_corroboration(None, None)  # type: ignore[arg-type]
        assert r["verdict"] == "NO_SLATE"

    def test_garbage_opportunity_rows(self):
        opps = [None, "garbage", 42, {}, {"ticker": ""},
                _opp("AAA", pred=5.0)]
        r = build_slate_news_corroboration(opps, {})
        # Only AAA survives. cohort 1, all QUANT_ONLY → QUANT_LEAD.
        assert r["n_total"] == 1
        assert r["verdict"] == "QUANT_LEAD"

    def test_garbage_pulse_rows(self):
        opps = [_opp("AAA", pred=5.0)]
        # Non-dict pulse rows are skipped; AAA gets default zero pulse.
        pulse = {"AAA": "garbage", "BBB": None}
        r = build_slate_news_corroboration(opps, pulse)  # type: ignore[arg-type]
        # AAA pulse defaulted → QUANT_ONLY → cohort majority → QUANT_LEAD.
        assert r["verdict"] == "QUANT_LEAD"
        assert r["by_name"][0]["news_n"] == 0

    def test_string_pred_in_opportunity(self):
        opps = [{"ticker": "AAA", "pred_5d_return_pct": "garbage",
                 "verdict": "STRONG_HOLD"}]
        r = build_slate_news_corroboration(opps, {})
        # pred coerced to 0.0 → SUB_THRESHOLD against floor 1.0.
        assert r["counts"]["SUB_THRESHOLD"] == 1

    def test_extreme_floor_clamping(self):
        opps = [_opp("AAA", pred=50.0)]
        # min_pred_pct negative → clamped to 0.0 → AAA passes floor.
        r = build_slate_news_corroboration(opps, {}, min_pred_pct=-99.0)
        assert r["constraints"]["min_pred_pct"] == 0.0

    def test_extreme_hot_min_score_clamping(self):
        opps = [_opp("AAA", pred=5.0)]
        pulse = {"AAA": _pulse(n=5, max_score=9.99)}
        r = build_slate_news_corroboration(opps, pulse, hot_min_score=99.0)
        # hot_min_score clamped to MAX_HOT_MIN_SCORE (10.0). 9.99 < 10.0 →
        # falls to CONVERGENT (count ≥ 2, score ≥ 4).
        assert r["by_name"][0]["verdict"] == "CONVERGENT"


class TestHeadlineAndCounts:
    def test_headline_describes_cohort(self):
        opps = [_opp(t) for t in ("A", "B", "C", "D")]
        pulse = {"A": _pulse(n=4, urgent=1, max_score=8.0)}  # HOT, others QUANT_ONLY
        r = build_slate_news_corroboration(opps, pulse)
        assert "4 buy-candidate" in r["headline"]
        assert "1 HOT" in r["headline"]
        assert "3 QUANT_ONLY" in r["headline"]

    def test_totals_aggregate_across_full_slate(self):
        opps = [_opp("A"), _opp("B")]
        pulse = {
            "A": _pulse(n=3, urgent=1, max_score=8.0),
            "B": _pulse(n=2, urgent=0, max_score=5.0),
        }
        r = build_slate_news_corroboration(opps, pulse)
        assert r["totals"]["articles"] == 5
        assert r["totals"]["urgent"] == 1


class TestFlaskEndpointContract:
    def test_endpoint_returns_json_shape(self):
        # Live endpoint contract: the route handler must accept a request
        # with no positions and an empty articles.db and return a clean
        # NO_SLATE envelope rather than a 500. Matches the
        # ``paper-trader analytics verification`` memory: module __main__
        # smoke hits the empty data/ DB and would never trip the live
        # cache; the test_client is the verifiable contract.
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/slate-news-corroboration?hours=24")
        assert resp.status_code in (200,)
        data = resp.get_json()
        assert isinstance(data, dict)
        # Verdict is always one of the documented ladder values.
        assert data.get("verdict") in (
            "NO_SLATE", "STRONG_CORROBORATION", "QUANT_LEAD",
            "THIN", "MIXED_CORROBORATION", "ERROR",
        )
        assert "window_hours" in data
        # Constraints echoed when the builder runs.
        if data["verdict"] != "ERROR":
            assert "constraints" in data
            c = data["constraints"]
            assert "min_pred_pct" in c
            assert "hot_min_count" in c

    def test_endpoint_honours_query_params(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get(
            "/api/slate-news-corroboration?hours=2&min_pred_pct=0.5"
            "&hot_min_count=5&hot_min_score=7.0"
            "&convergent_min_count=3&convergent_min_score=5.0"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        if data.get("verdict") != "ERROR":
            assert data["window_hours"] == 2
            assert data["constraints"]["min_pred_pct"] == 0.5
            assert data["constraints"]["hot_min_count"] == 5
            assert data["constraints"]["convergent_min_count"] == 3

    def test_endpoint_clamps_hours(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        # hours=99999 must clamp to the documented 168 cap.
        resp = client.get("/api/slate-news-corroboration?hours=99999")
        assert resp.status_code == 200
        data = resp.get_json()
        if data.get("verdict") != "ERROR":
            assert data["window_hours"] == 168
        # hours=0 must clamp up to the documented 1 floor.
        resp = client.get("/api/slate-news-corroboration?hours=0")
        assert resp.status_code == 200
        data = resp.get_json()
        if data.get("verdict") != "ERROR":
            assert data["window_hours"] == 1


class TestDefaultsArePinned:
    """Lock the public defaults so a downstream consumer (the panel JS, or
    a sibling endpoint) doesn't silently see them shift."""
    def test_defaults_pinned(self):
        assert DEFAULT_MIN_PRED_PCT == 1.0
        assert DEFAULT_HOT_MIN_COUNT == 3
        assert DEFAULT_HOT_MIN_SCORE == 6.0
        assert DEFAULT_CONVERGENT_MIN_COUNT == 2
        assert DEFAULT_CONVERGENT_MIN_SCORE == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
