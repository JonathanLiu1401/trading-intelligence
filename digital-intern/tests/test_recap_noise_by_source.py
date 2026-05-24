"""Tests for analytics.recap_noise_by_source.build_report.

Pure-builder tests: feed (source, title, ai_score, ml_score) tuples directly,
assert specific numeric outputs and verdicts. No SQLite — the SQL is a thin
shim around the live DB; the analytics logic is the unit under test.

The build_report function uses the SSOT ``_RECAP_TEMPLATE_PATTERNS`` from
``watchers.alert_agent``, so titles that match any of those fingerprints
(``why_pct_after``, ``subject_pct_after``, ``quick_glance_metrics``, …) are
counted toward the source's ``recap_count``.
"""
from __future__ import annotations

import pytest

from analytics.recap_noise_by_source import (
    AI_FLOOR,
    MIN_PER_SOURCE,
    ML_FLOOR,
    NOISE_THRESHOLD,
    _matches_recap,
    build_report,
)


def _recap_title(i: int) -> str:
    """A title guaranteed to match the new ``subject_pct_after`` fingerprint."""
    return f"Lumentum (NASDAQ:LITE{i}) Shares Down 8.{i}% After Insider Selling"


def _clean_title(i: int) -> str:
    """A title that does NOT match any recap fingerprint (real-news shape)."""
    return f"Fed cuts rates by {i}bp citing labor weakness"


class TestPureRecapSource:
    """A source whose entire sample matches recap fingerprints is flagged."""

    def test_pure_recap_source_flagged_as_noise_factory(self):
        rows = [
            ("marketbeat", _recap_title(i), 0.0, 9.5)
            for i in range(MIN_PER_SOURCE + 5)
        ]
        report = build_report(rows)
        assert len(report["sources"]) == 1
        s = report["sources"][0]
        assert s["source"] == "marketbeat"
        assert s["high_relevance_count"] == MIN_PER_SOURCE + 5
        assert s["recap_count"] == MIN_PER_SOURCE + 5
        assert s["recap_rate"] == 1.0
        assert s["is_noise_factory"] is True
        assert report["noise_source_count"] == 1
        assert "marketbeat" in report["noise_sources"]
        assert report["evaluated_sources"] == 1


class TestCleanSource:
    """A real-wire source with no recap-shaped titles is never flagged."""

    def test_clean_source_recap_rate_zero(self):
        rows = [
            ("reuters", _clean_title(i), 8.0, None)
            for i in range(MIN_PER_SOURCE + 5)
        ]
        report = build_report(rows)
        assert len(report["sources"]) == 1
        s = report["sources"][0]
        assert s["recap_rate"] == 0.0
        assert s["recap_count"] == 0
        assert s["is_noise_factory"] is False
        assert report["noise_source_count"] == 0
        assert report["noise_sources"] == []


class TestSampleSizeFloor:
    """Sources below the per-source minimum are excluded from the report —
    a 1/2 recap rate is meaningless and should not flag."""

    def test_below_min_per_source_excluded(self):
        rows = [
            ("tiny_source", _recap_title(i), 0.0, 9.0)
            for i in range(MIN_PER_SOURCE - 1)
        ]
        report = build_report(rows)
        assert report["evaluated_sources"] == 0
        assert report["sources"] == []
        assert report["noise_source_count"] == 0

    def test_at_min_per_source_included(self):
        """Sample = exactly MIN_PER_SOURCE is the inclusion boundary."""
        rows = [
            ("borderline_sample", _recap_title(i), 0.0, 9.0)
            for i in range(MIN_PER_SOURCE)
        ]
        report = build_report(rows)
        assert report["evaluated_sources"] == 1
        assert report["sources"][0]["high_relevance_count"] == MIN_PER_SOURCE


class TestThresholdBoundary:
    """Verdict ladder math — recap_rate >= NOISE_THRESHOLD is flagged."""

    def test_at_threshold_flagged(self):
        recap_n = int(round(NOISE_THRESHOLD * MIN_PER_SOURCE))
        clean_n = MIN_PER_SOURCE - recap_n
        rows = [
            ("borderline", _recap_title(i), 0.0, 9.0)
            for i in range(recap_n)
        ] + [
            ("borderline", _clean_title(i), 8.0, None)
            for i in range(clean_n)
        ]
        report = build_report(rows)
        s = report["sources"][0]
        assert s["recap_count"] == recap_n
        assert s["high_relevance_count"] == MIN_PER_SOURCE
        assert s["recap_rate"] == round(recap_n / MIN_PER_SOURCE, 4)
        # >= threshold → flagged.
        assert s["is_noise_factory"] is True

    def test_below_threshold_not_flagged(self):
        # One fewer recap row than the threshold → just under the bar.
        recap_n = int(round(NOISE_THRESHOLD * MIN_PER_SOURCE)) - 1
        if recap_n < 0:
            pytest.skip("NOISE_THRESHOLD too low for this boundary test")
        clean_n = MIN_PER_SOURCE - recap_n
        rows = [
            ("clean_ish", _recap_title(i), 0.0, 9.0)
            for i in range(recap_n)
        ] + [
            ("clean_ish", _clean_title(i), 8.0, None)
            for i in range(clean_n)
        ]
        report = build_report(rows)
        s = report["sources"][0]
        assert s["recap_count"] == recap_n
        assert s["is_noise_factory"] is False
        assert report["noise_source_count"] == 0


class TestFingerprintBreakdown:
    """The per-source ``top_fingerprints`` field surfaces which gates the
    source primarily trips — the operator's actionable hint."""

    def test_top_fingerprints_returned_sorted_by_count(self):
        rows = (
            # 15 subject_pct_after matches.
            [("mixed", _recap_title(i), 0.0, 9.0) for i in range(15)]
            # 10 why_stock_is_after matches.
            + [
                ("mixed", "Why Nvidia Stock Is Down 3% After Q1 Earnings", 0.0, 9.0)
                for _ in range(10)
            ]
            # 5 quick_glance_metrics matches.
            + [
                ("mixed", "NVIDIA Earnings: A Quick Glance at Key Metrics", 0.0, 9.0)
                for _ in range(5)
            ]
        )
        report = build_report(rows)
        s = report["sources"][0]
        # Top 3 (the cap); order is by descending count.
        fps = s["top_fingerprints"]
        assert len(fps) == 3
        assert fps[0] == ("subject_pct_after", 15)
        assert fps[1] == ("why_stock_is_after", 10)
        assert fps[2] == ("quick_glance_metrics", 5)
        # All 30 rows counted as recap.
        assert s["recap_count"] == 30
        assert s["high_relevance_count"] == 30


class TestSourceOrdering:
    """Sources are ordered by descending recap_rate so the worst offenders are
    surfaced first — operator-readable, deterministic across runs."""

    def test_worst_offender_first(self):
        rows = (
            # source_a — 50% recap.
            [("source_a", _recap_title(i), 0.0, 9.0) for i in range(10)]
            + [("source_a", _clean_title(i), 8.0, None) for i in range(10)]
            # source_b — 100% recap.
            + [("source_b", _recap_title(i), 0.0, 9.0) for i in range(MIN_PER_SOURCE)]
            # source_c — 5% recap.
            + [("source_c", _recap_title(0), 0.0, 9.0)]
            + [("source_c", _clean_title(i), 8.0, None) for i in range(MIN_PER_SOURCE - 1)]
        )
        report = build_report(rows)
        names = [s["source"] for s in report["sources"]]
        assert names == ["source_b", "source_a", "source_c"], (
            f"sources should be ordered worst-first: got {names}"
        )

    def test_unknown_source_normalised(self):
        """``None``/``""`` source becomes ``(unknown)`` — never crashes
        on a missing source field."""
        rows = [(None, _recap_title(i), 0.0, 9.0) for i in range(MIN_PER_SOURCE)]
        report = build_report(rows)
        assert report["sources"][0]["source"] == "(unknown)"


class TestMatchesRecapHelper:
    """The ``_matches_recap`` helper is the unit driving the rate — its
    behaviour against representative titles must be exact, not approximate."""

    def test_known_recap_titles_match(self):
        # subject_pct_after — the new gate this module is meant to track.
        hit, name = _matches_recap(
            "D-Wave Quantum (QBTS) Is Up 44.5% After $100M Federal Equity Investment"
        )
        assert hit is True
        assert name == "subject_pct_after"

    def test_real_news_does_not_match(self):
        hit, name = _matches_recap(
            "Nvidia Q3 revenue rises 22% to $35.1 billion, beats estimates"
        )
        assert hit is False
        assert name == ""

    def test_empty_or_none_title_does_not_match(self):
        assert _matches_recap("") == (False, "")
        assert _matches_recap(None) == (False, "")


class TestReportMetadata:
    """The report carries the configuration constants used at audit time so
    a future tuning shift is visible on the disk snapshot."""

    def test_report_metadata_keys_present(self):
        rows = [("any", _clean_title(0), 8.0, None) for _ in range(MIN_PER_SOURCE)]
        report = build_report(rows)
        assert report["lookback_hours"] > 0
        assert report["min_per_source"] == MIN_PER_SOURCE
        assert report["noise_threshold"] == NOISE_THRESHOLD
        assert report["ai_floor"] == AI_FLOOR
        assert report["ml_floor"] == ML_FLOOR
