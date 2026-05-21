"""Tests for paper_trader.analytics.news_age_at_decision_skill.

Pins:
* the FRESH_NEWS_BETTER × STALE_NEWS_BETTER × NO_PATTERN ×
  INSUFFICIENT_DATA verdict matrix
* exact bucket boundary semantics (lo inclusive, hi exclusive — 60.0 min
  bumps from FRESH_LT_60M to HOURS_1_TO_6, 1440.0 bumps from HOURS_6_TO_24
  to STALE_GT_24H)
* NO_NEWS sample assignment when freshest_article_age_min is None
* malformed sample silent-drop (no realized_pct, no action, wrong types)
* envelope key stability across every verdict
* per-bucket aggregate maths (mean / median / win_rate) at exact values
* HOLD / NO_DECISION samples are dropped (only entries/exits valid)
* threshold-override forwarding
* negative age_min defensively treated as NO_NEWS
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.news_age_at_decision_skill import (
    BUCKETS,
    FRESH_NEWS_BETTER,
    INSUFFICIENT_DATA,
    NO_NEWS_BUCKET,
    NO_PATTERN,
    STALE_NEWS_BETTER,
    _bucket_for,
    build_news_age_at_decision_skill,
)


def _now():
    return datetime(2026, 5, 21, 7, 0, 0, tzinfo=timezone.utc)


def _s(age_min, realized, *, action="BUY", ticker="NVDA", closed=False, tid=1):
    return {
        "trade_id": tid,
        "trade_ts": "2026-05-21T06:00:00+00:00",
        "ticker": ticker,
        "action": action,
        "freshest_article_age_min": age_min,
        "realized_pct": realized,
        "closed": closed,
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "n_samples", "n_with_news",
    "n_no_news", "buckets", "thresholds", "samples",
}


class TestBucketBoundaries:
    def test_age_zero_is_fresh(self):
        assert _bucket_for(0.0) == "FRESH_LT_60M"

    def test_age_59m_is_fresh(self):
        assert _bucket_for(59.999) == "FRESH_LT_60M"

    def test_age_exactly_60m_rolls_to_hours(self):
        # lo inclusive, hi exclusive — 60.0 belongs to HOURS_1_TO_6.
        assert _bucket_for(60.0) == "HOURS_1_TO_6"

    def test_age_359m_is_hours_1_to_6(self):
        assert _bucket_for(359.0) == "HOURS_1_TO_6"

    def test_age_exactly_360m_rolls_to_day(self):
        assert _bucket_for(360.0) == "HOURS_6_TO_24"

    def test_age_1439m_is_day(self):
        assert _bucket_for(1439.99) == "HOURS_6_TO_24"

    def test_age_exactly_1440m_rolls_to_stale(self):
        assert _bucket_for(1440.0) == "STALE_GT_24H"

    def test_age_5_days_is_stale(self):
        assert _bucket_for(60 * 24 * 5) == "STALE_GT_24H"

    def test_none_age_is_no_news(self):
        assert _bucket_for(None) == NO_NEWS_BUCKET

    def test_negative_age_is_no_news(self):
        # Defensive — a negative age means the joiner attached a future
        # article, which is a bug. The sample must not pollute a bucket.
        assert _bucket_for(-1.0) == NO_NEWS_BUCKET

    def test_all_bucket_labels_are_unique(self):
        labels = [b[0] for b in BUCKETS]
        assert len(labels) == len(set(labels))


class TestEnvelopeStability:
    def test_empty_input(self):
        out = build_news_age_at_decision_skill([], now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_samples"] == 0
        assert out["n_with_news"] == 0
        assert out["n_no_news"] == 0

    def test_none_input(self):
        out = build_news_age_at_decision_skill(None, now=_now())
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_samples"] == 0

    def test_envelope_keys_present_on_every_verdict(self):
        # FRESH_NEWS_BETTER case
        fresh = [_s(10, 5.0, tid=i) for i in range(5)]
        stale = [_s(2000, 0.5, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == FRESH_NEWS_BETTER

    def test_all_buckets_in_output(self):
        out = build_news_age_at_decision_skill([_s(10, 1.0)], now=_now())
        assert "FRESH_LT_60M" in out["buckets"]
        assert "HOURS_1_TO_6" in out["buckets"]
        assert "HOURS_6_TO_24" in out["buckets"]
        assert "STALE_GT_24H" in out["buckets"]
        assert NO_NEWS_BUCKET in out["buckets"]


class TestVerdictMatrix:
    def test_fresh_news_better_fires_when_gap_clears_threshold(self):
        # 5 fresh @+5%, 5 stale @+0.5% → gap +4.5pp ≥ 2.0
        fresh = [_s(10, 5.0, tid=i) for i in range(5)]
        stale = [_s(2000, 0.5, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert out["verdict"] == FRESH_NEWS_BETTER
        assert out["buckets"]["FRESH_LT_60M"]["mean_pct"] == 5.0
        assert out["buckets"]["STALE_GT_24H"]["mean_pct"] == 0.5

    def test_stale_news_better_fires_inverse(self):
        fresh = [_s(10, -3.0, tid=i) for i in range(5)]
        stale = [_s(2000, 2.0, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert out["verdict"] == STALE_NEWS_BETTER

    def test_no_pattern_when_gap_within_tolerance(self):
        # gap 1.0pp < 2.0pp threshold
        fresh = [_s(10, 2.0, tid=i) for i in range(5)]
        stale = [_s(2000, 1.0, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert out["verdict"] == NO_PATTERN

    def test_insufficient_when_one_bucket_short(self):
        # Only 2 fresh samples, default min_per_bucket=3
        fresh = [_s(10, 10.0, tid=i) for i in range(2)]
        stale = [_s(2000, 0.0, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert out["verdict"] == INSUFFICIENT_DATA


class TestAggregateMath:
    def test_mean_median_win_rate_exact(self):
        # 5 fresh: returns -2, -1, 0, 1, 2  → mean 0.0, median 0.0, win_rate 40%
        rows = [
            _s(10, r, tid=i) for i, r in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0])
        ]
        # pad stale to fire a verdict
        rows += [_s(2000, 5.0, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(rows, now=_now())
        fresh = out["buckets"]["FRESH_LT_60M"]
        assert fresh["n"] == 5
        assert fresh["mean_pct"] == 0.0
        assert fresh["median_pct"] == 0.0
        # 2 strictly > 0 out of 5 → 40.0
        assert fresh["win_rate"] == 40.0

    def test_median_even_n(self):
        # 4 samples: 1, 2, 3, 4 → median (2+3)/2 = 2.5
        rows = [_s(10, r, tid=i) for i, r in enumerate([1.0, 2.0, 3.0, 4.0])]
        rows += [_s(2000, 0.0, tid=i + 100) for i in range(5)]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["buckets"]["FRESH_LT_60M"]["median_pct"] == 2.5


class TestSampleNormalisation:
    def test_no_news_bucket_assignment(self):
        rows = [_s(None, 1.0, tid=i) for i in range(3)]
        rows += [_s(10, 1.0, tid=i + 100) for i in range(3)]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["n_no_news"] == 3
        assert out["n_with_news"] == 3
        assert out["buckets"][NO_NEWS_BUCKET]["n"] == 3

    def test_hold_decisions_dropped(self):
        rows = [
            _s(10, 1.0, action="HOLD"),
            _s(10, 1.0, action="NO_DECISION"),
            _s(10, 1.0, action="BLOCKED"),
            _s(10, 1.0, action="BUY"),
        ]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["n_samples"] == 1

    def test_options_verbs_accepted(self):
        rows = [
            _s(10, 1.0, action="BUY_CALL"),
            _s(10, 1.0, action="SELL_PUT"),
        ]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["n_samples"] == 2

    def test_malformed_silent_drop(self):
        rows = [
            "not a dict",
            None,
            {"realized_pct": None, "action": "BUY"},  # missing realized
            {"realized_pct": 1.0},                     # missing action
            {"realized_pct": float("nan"), "action": "BUY"},  # NaN return
            {"realized_pct": 1.0, "action": 123},      # wrong type
            _s(10, 1.0),                                # valid
        ]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["n_samples"] == 1


class TestThresholdOverrides:
    def test_min_per_bucket_override(self):
        fresh = [_s(10, 5.0, tid=i) for i in range(2)]
        stale = [_s(2000, 0.5, tid=i + 100) for i in range(2)]
        # default min=3 → INSUFFICIENT
        default = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert default["verdict"] == INSUFFICIENT_DATA
        # override to 2 → verdict fires
        over = build_news_age_at_decision_skill(
            fresh + stale, now=_now(), min_per_bucket=2,
        )
        assert over["verdict"] == FRESH_NEWS_BETTER

    def test_verdict_gap_override(self):
        fresh = [_s(10, 1.5, tid=i) for i in range(5)]
        stale = [_s(2000, 0.0, tid=i + 100) for i in range(5)]
        # default gap=2.0 → 1.5 not enough → NO_PATTERN
        default = build_news_age_at_decision_skill(fresh + stale, now=_now())
        assert default["verdict"] == NO_PATTERN
        # tighten gap to 1.0 → fires
        over = build_news_age_at_decision_skill(
            fresh + stale, now=_now(), verdict_gap_pct=1.0,
        )
        assert over["verdict"] == FRESH_NEWS_BETTER


class TestSamplesCard:
    def test_samples_capped_at_50(self):
        rows = [_s(10, 1.0, tid=i) for i in range(120)]
        out = build_news_age_at_decision_skill(rows, now=_now())
        assert out["n_samples"] == 120
        assert len(out["samples"]) == 50

    def test_closed_samples_emitted_first(self):
        rows = [
            _s(10, 1.0, tid=1, closed=False),
            _s(10, 1.0, tid=2, closed=True),
            _s(10, 1.0, tid=3, closed=False),
            _s(10, 1.0, tid=4, closed=True),
        ]
        out = build_news_age_at_decision_skill(rows, now=_now())
        # First two cards must be closed
        assert out["samples"][0]["closed"] is True
        assert out["samples"][1]["closed"] is True
