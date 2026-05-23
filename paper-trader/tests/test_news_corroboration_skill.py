"""Test suite for news_corroboration_skill.

Pinned exact-value tests. Mirror the news_age_at_decision_skill /
conviction_language_skill discipline.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_trader.analytics.news_corroboration_skill import (
    build_news_corroboration_skill,
    _bucket_for,
    NO_NEWS,
    SINGLE,
    SMALL_CHORUS,
    CHORUS,
    FLOOD,
    CORROBORATION_HELPS,
    SINGLE_HELPS,
    NO_PATTERN,
    INSUFFICIENT_DATA,
)

NOW = datetime(2026, 5, 21, 7, 30, tzinfo=timezone.utc)


def _sample(
    *,
    trade_id: int = 1,
    ticker: str = "NVDA",
    action: str = "BUY",
    article_count: int | None = 1,
    realized_pct: float = 1.0,
    closed: bool = True,
    trade_ts: str | None = None,
) -> dict:
    return {
        "trade_id": trade_id,
        "trade_ts": trade_ts or "2026-05-20T01:00:00+00:00",
        "ticker": ticker,
        "action": action,
        "article_count": article_count,
        "realized_pct": realized_pct,
        "closed": closed,
    }


class TestBucketBoundaries:
    """Bucket edges are load-bearing — lo inclusive, hi exclusive.

    A 1-count → SINGLE; 2 → SMALL_CHORUS; 4 → CHORUS; 10 → FLOOD.
    """

    def test_zero_articles_routes_to_no_news(self):
        assert _bucket_for(0) == NO_NEWS

    def test_single_article(self):
        assert _bucket_for(1) == SINGLE

    def test_two_articles_small_chorus(self):
        assert _bucket_for(2) == SMALL_CHORUS

    def test_three_articles_small_chorus(self):
        assert _bucket_for(3) == SMALL_CHORUS

    def test_four_articles_chorus(self):
        assert _bucket_for(4) == CHORUS

    def test_nine_articles_chorus(self):
        assert _bucket_for(9) == CHORUS

    def test_ten_articles_flood(self):
        assert _bucket_for(10) == FLOOD

    def test_huge_count_flood(self):
        assert _bucket_for(10_000) == FLOOD

    def test_negative_routes_to_no_news(self):
        # A negative count is a join bug; degrade to NO_NEWS not crash.
        assert _bucket_for(-5) == NO_NEWS

    def test_none_routes_to_no_news(self):
        assert _bucket_for(None) == NO_NEWS

    def test_string_routes_to_no_news(self):
        # Defensive: a string from a malformed sample never raises.
        assert _bucket_for("3") == NO_NEWS

    def test_boolean_rejected_as_count(self):
        # True is technically int(1) but operationally is a flag,
        # not a count — reject it so a truthy join column doesn't
        # become a SINGLE bucket.
        assert _bucket_for(True) == NO_NEWS
        assert _bucket_for(False) == NO_NEWS

    def test_nan_routes_to_no_news(self):
        assert _bucket_for(float("nan")) == NO_NEWS


class TestEnvelopeShape:
    """Output envelope keys must be stable across every verdict path
    so the UI binding never sees a missing field."""

    _REQUIRED_KEYS = {
        "as_of", "verdict", "headline", "n_samples",
        "buckets", "chorus_plus", "thresholds", "samples",
    }

    def test_empty_input_returns_insufficient_data(self):
        out = build_news_corroboration_skill([], now=NOW)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_samples"] == 0
        assert set(out.keys()) >= self._REQUIRED_KEYS

    def test_none_input_returns_insufficient_data(self):
        out = build_news_corroboration_skill(None, now=NOW)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_samples"] == 0

    def test_all_five_bucket_keys_present(self):
        out = build_news_corroboration_skill([_sample()], now=NOW)
        assert set(out["buckets"].keys()) == {
            NO_NEWS, SINGLE, SMALL_CHORUS, CHORUS, FLOOD,
        }

    def test_empty_bucket_has_none_metrics(self):
        out = build_news_corroboration_skill([_sample()], now=NOW)
        # FLOOD has no samples in this fixture
        assert out["buckets"][FLOOD] == {
            "n": 0, "mean_pct": None, "median_pct": None, "win_rate": None,
        }

    def test_as_of_uses_supplied_now(self):
        out = build_news_corroboration_skill([_sample()], now=NOW)
        assert out["as_of"].startswith("2026-05-21T07:30:00")

    def test_naive_now_promoted_to_utc(self):
        naive = datetime(2026, 5, 21, 7, 30)
        out = build_news_corroboration_skill([_sample()], now=naive)
        assert "+00:00" in out["as_of"]

    def test_thresholds_echoed(self):
        out = build_news_corroboration_skill(
            [_sample()], now=NOW,
            min_per_bucket=5, verdict_gap_pct=4.0,
        )
        assert out["thresholds"] == {
            "min_per_bucket": 5, "verdict_gap_pct": 4.0,
        }


class TestRobustness:
    """Builder must never raise on garbage samples."""

    def test_dropped_when_realized_missing(self):
        out = build_news_corroboration_skill([
            _sample(article_count=1, realized_pct=None),
        ], now=NOW)
        assert out["n_samples"] == 0

    def test_dropped_when_realized_nan(self):
        out = build_news_corroboration_skill([
            _sample(article_count=1, realized_pct=float("nan")),
        ], now=NOW)
        assert out["n_samples"] == 0

    def test_non_dict_sample_silently_dropped(self):
        out = build_news_corroboration_skill(
            ["not a dict", None, 42, _sample()],  # type: ignore[list-item]
            now=NOW,
        )
        assert out["n_samples"] == 1

    def test_missing_article_count_routes_to_no_news(self):
        out = build_news_corroboration_skill([
            {"trade_id": 1, "realized_pct": 1.0},
        ], now=NOW)
        assert out["n_samples"] == 1
        assert out["buckets"][NO_NEWS]["n"] == 1
        # surfaced count normalised to 0, not None
        assert out["samples"][0]["article_count"] == 0

    def test_negative_count_normalised_to_zero(self):
        out = build_news_corroboration_skill([
            _sample(article_count=-3, realized_pct=1.0),
        ], now=NOW)
        assert out["samples"][0]["article_count"] == 0
        assert out["samples"][0]["bucket"] == NO_NEWS


class TestVerdictMatrix:
    """Verdict gate: ≥ min_per_bucket in BOTH SINGLE and CHORUS+ (the
    union of CHORUS and FLOOD)."""

    def test_three_chorus_three_single_below_gap_no_pattern(self):
        # SINGLE mean = 1.0, CHORUS+ mean = 2.0 → gap = 1.0pp (< 2.0pp)
        rows = (
            [_sample(trade_id=i, article_count=1, realized_pct=1.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 4)]
            + [_sample(trade_id=10 + i, article_count=5, realized_pct=2.0,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == NO_PATTERN
        assert "NO_PATTERN" in out["headline"]

    def test_chorus_beats_single_by_gap_corroboration_helps(self):
        # SINGLE mean = 0.0, CHORUS+ mean = 5.0 → gap = +5.0pp
        rows = (
            [_sample(trade_id=i, article_count=1, realized_pct=0.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 4)]
            + [_sample(trade_id=10 + i, article_count=5, realized_pct=5.0,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == CORROBORATION_HELPS
        assert "CORROBORATION_HELPS" in out["headline"]
        # CHORUS+ mean appears in headline
        assert "+5.00%" in out["headline"]

    def test_single_beats_chorus_by_gap_single_helps(self):
        # SINGLE mean = 5.0, CHORUS+ mean = 0.0 → gap = -5.0pp
        rows = (
            [_sample(trade_id=i, article_count=1, realized_pct=5.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 4)]
            + [_sample(trade_id=10 + i, article_count=5, realized_pct=0.0,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == SINGLE_HELPS

    def test_below_min_per_bucket_single_withholds_verdict(self):
        rows = (
            [_sample(trade_id=1, article_count=1, realized_pct=5.0)]
            + [_sample(trade_id=10 + i, article_count=5, realized_pct=5.0,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == INSUFFICIENT_DATA
        # Per-bucket cards still ship even when verdict is withheld
        assert out["buckets"][SINGLE]["n"] == 1
        assert out["buckets"][CHORUS]["n"] == 3

    def test_below_min_per_bucket_chorus_withholds_verdict(self):
        rows = (
            [_sample(trade_id=i, article_count=1, realized_pct=5.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 4)]
            + [_sample(trade_id=10, article_count=5, realized_pct=5.0)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_only_no_news_and_single_withhold_verdict(self):
        # 5 NO_NEWS + 3 SINGLE — chorus_plus n = 0 — verdict withheld
        rows = (
            [_sample(trade_id=i, article_count=0, realized_pct=1.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 6)]
            + [_sample(trade_id=10 + i, article_count=1, realized_pct=1.0,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["buckets"][NO_NEWS]["n"] == 5

    def test_no_news_does_not_move_verdict(self):
        # 50 NO_NEWS samples cannot tip the verdict — only SINGLE vs
        # CHORUS+ counts.
        rows = (
            [_sample(trade_id=i, article_count=0, realized_pct=-50.0,
                     trade_ts=f"2026-05-18T{(i % 23):02d}:00:00+00:00")
             for i in range(50)]
            + [_sample(trade_id=100 + i, article_count=1, realized_pct=1.0,
                       trade_ts=f"2026-05-19T0{i}:00:00+00:00")
               for i in range(1, 4)]
            + [_sample(trade_id=200 + i, article_count=5, realized_pct=1.5,
                       trade_ts=f"2026-05-20T0{i}:00:00+00:00")
               for i in range(1, 4)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        # gap = +0.5pp, within tolerance — NO_PATTERN
        assert out["verdict"] == NO_PATTERN

    def test_chorus_and_flood_combine_for_verdict(self):
        # 2 CHORUS + 2 FLOOD → 4 chorus_plus, meets min_per_bucket=3
        rows = (
            [_sample(trade_id=i, article_count=1, realized_pct=0.0,
                     trade_ts=f"2026-05-19T0{i}:00:00+00:00")
             for i in range(1, 4)]
            + [_sample(trade_id=10, article_count=5, realized_pct=5.0)]
            + [_sample(trade_id=11, article_count=7, realized_pct=5.0)]
            + [_sample(trade_id=12, article_count=15, realized_pct=5.0)]
            + [_sample(trade_id=13, article_count=20, realized_pct=5.0)]
        )
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["verdict"] == CORROBORATION_HELPS
        assert out["chorus_plus"]["n"] == 4
        assert out["buckets"][CHORUS]["n"] == 2
        assert out["buckets"][FLOOD]["n"] == 2


class TestBucketMath:
    """Aggregate stats are arithmetic-pinned so future regression
    catches accidental sign-flips or sort-order drift."""

    def test_single_sample_mean_median_match(self):
        out = build_news_corroboration_skill([
            _sample(article_count=1, realized_pct=3.7),
        ], now=NOW)
        bucket = out["buckets"][SINGLE]
        assert bucket["n"] == 1
        assert bucket["mean_pct"] == 3.7
        assert bucket["median_pct"] == 3.7
        assert bucket["win_rate"] == 100.0

    def test_two_samples_median_is_midpoint(self):
        out = build_news_corroboration_skill([
            _sample(trade_id=1, article_count=1, realized_pct=2.0,
                    trade_ts="2026-05-19T01:00:00+00:00"),
            _sample(trade_id=2, article_count=1, realized_pct=4.0,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ], now=NOW)
        bucket = out["buckets"][SINGLE]
        assert bucket["mean_pct"] == 3.0
        assert bucket["median_pct"] == 3.0

    def test_win_rate_excludes_zero(self):
        # Zero is neither a win nor a loss in the existing skill family
        # (round_trips treats a flat close as a non-win) — match that
        # convention so the win_rate semantics stay consistent.
        out = build_news_corroboration_skill([
            _sample(trade_id=1, article_count=1, realized_pct=0.0,
                    trade_ts="2026-05-19T01:00:00+00:00"),
            _sample(trade_id=2, article_count=1, realized_pct=1.0,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ], now=NOW)
        bucket = out["buckets"][SINGLE]
        assert bucket["n"] == 2
        assert bucket["win_rate"] == 50.0

    def test_negative_only_returns_zero_win_rate(self):
        out = build_news_corroboration_skill([
            _sample(trade_id=1, article_count=1, realized_pct=-5.0,
                    trade_ts="2026-05-19T01:00:00+00:00"),
            _sample(trade_id=2, article_count=1, realized_pct=-3.0,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ], now=NOW)
        bucket = out["buckets"][SINGLE]
        assert bucket["win_rate"] == 0.0
        # median of two values is midpoint
        assert bucket["median_pct"] == -4.0


class TestSampleSerialisation:
    """The samples list is the operator's audit trail. Closed > open
    sort order; cap at 50; bucket label re-surfaced."""

    def test_closed_sorted_before_open(self):
        rows = [
            _sample(trade_id=1, closed=False,
                    trade_ts="2026-05-19T01:00:00+00:00"),
            _sample(trade_id=2, closed=True,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ]
        out = build_news_corroboration_skill(rows, now=NOW)
        # closed (id=2) appears before open (id=1)
        assert out["samples"][0]["trade_id"] == 2
        assert out["samples"][1]["trade_id"] == 1

    def test_samples_capped_at_50(self):
        rows = [
            _sample(trade_id=i, article_count=1, realized_pct=1.0,
                    trade_ts=f"2026-05-19T{(i % 23):02d}:00:00+00:00")
            for i in range(60)
        ]
        out = build_news_corroboration_skill(rows, now=NOW)
        assert out["n_samples"] == 60
        assert len(out["samples"]) == 50

    def test_sample_carries_bucket_label(self):
        out = build_news_corroboration_skill([
            _sample(article_count=5, realized_pct=1.0),
        ], now=NOW)
        assert out["samples"][0]["bucket"] == CHORUS

    def test_realized_pct_rounded_in_sample(self):
        out = build_news_corroboration_skill([
            _sample(article_count=1, realized_pct=1.23456),
        ], now=NOW)
        assert out["samples"][0]["realized_pct"] == 1.23


class TestChorusPlusEnvelope:
    """The chorus_plus aggregate is a top-level convenience for the UI
    — it must match the SINGLE_HELPS / CORROBORATION_HELPS comparand."""

    def test_chorus_plus_combines_chorus_and_flood(self):
        out = build_news_corroboration_skill([
            _sample(trade_id=1, article_count=5, realized_pct=2.0,
                    trade_ts="2026-05-19T01:00:00+00:00"),
            _sample(trade_id=2, article_count=20, realized_pct=4.0,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ], now=NOW)
        cp = out["chorus_plus"]
        assert cp["n"] == 2
        assert cp["mean_pct"] == 3.0

    def test_chorus_plus_empty_when_only_no_news_and_single(self):
        out = build_news_corroboration_skill([
            _sample(article_count=0, realized_pct=1.0),
            _sample(trade_id=2, article_count=1, realized_pct=1.0,
                    trade_ts="2026-05-19T02:00:00+00:00"),
        ], now=NOW)
        cp = out["chorus_plus"]
        assert cp["n"] == 0
        assert cp["mean_pct"] is None
