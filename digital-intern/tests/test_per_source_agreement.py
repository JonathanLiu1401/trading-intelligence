"""Pins ml/per_source_agreement — the per-source ML-vs-LLM drift breakdown.

Aggregate stats come from ml/score_agreement (re-exported, already pinned by
test_score_agreement.py); these tests pin only the grouping/threshold contract
that this module adds on top.
"""
from __future__ import annotations

from ml.per_source_agreement import (
    MIN_SAMPLES_FOR_CORR,
    compute_per_source,
)


def _row(ml, ai, source="rss"):
    return {"ml_score": ml, "ai_score": ai, "source": source}


class TestComputePerSource:
    def test_empty_is_safe(self):
        out = compute_per_source([])
        assert out["total_overlap_rows"] == 0
        assert out["source_count"] == 0
        assert out["sources"] == []

    def test_ai_zero_excluded_like_aggregate(self):
        rows = [
            _row(8.0, 0.0, "rss"),  # LLM never graded → dropped
            _row(5.0, 5.0, "rss"),
            _row(6.0, 6.0, "rss"),
        ]
        out = compute_per_source(rows)
        assert out["total_overlap_rows"] == 2
        assert out["sources"][0]["n"] == 2

    def test_rows_with_no_source_dropped(self):
        # source is mandatory for bucketing — otherwise the row poisons the
        # "unknown" bucket and inflates the global overlap count.
        rows = [
            {"ml_score": 7.0, "ai_score": 7.0, "source": None},
            _row(5.0, 5.0, "rss"),
        ]
        out = compute_per_source(rows)
        assert out["total_overlap_rows"] == 1
        assert out["source_count"] == 1
        assert out["sources"][0]["source"] == "rss"

    def test_grouping_by_source(self):
        rows = [
            _row(8.0, 8.0, "finnhub"),
            _row(7.0, 7.0, "finnhub"),
            _row(5.0, 9.0, "substack"),
        ]
        out = compute_per_source(rows)
        assert out["source_count"] == 2
        by_src = {s["source"]: s for s in out["sources"]}
        assert by_src["finnhub"]["n"] == 2
        assert by_src["substack"]["n"] == 1

    def test_correlation_below_threshold_is_none_not_zero(self):
        # Distinguishing "insufficient data" from "uncorrelated" matters for
        # the dashboard — None is unambiguous; 0.0 would conflate the two.
        rows = [_row(5.0, 5.0, "rss") for _ in range(MIN_SAMPLES_FOR_CORR - 1)]
        out = compute_per_source(rows)
        s = out["sources"][0]
        assert s["n"] == MIN_SAMPLES_FOR_CORR - 1
        assert s["pearson"] is None
        assert s["spearman"] is None
        # But the divergence stats remain meaningful at small n.
        assert s["mean_abs_divergence"] == 0.0
        assert s["bias_ml_minus_ai"] == 0.0

    def test_correlation_reported_at_threshold(self):
        # Exactly at the cutoff → reported as a float.
        rows = [_row(float(i), float(i), "rss") for i in range(1, MIN_SAMPLES_FOR_CORR + 1)]
        out = compute_per_source(rows)
        s = out["sources"][0]
        assert s["n"] == MIN_SAMPLES_FOR_CORR
        assert s["pearson"] == 1.0
        assert s["spearman"] == 1.0

    def test_bias_sign_per_source(self):
        # The cheap model runs hot on one source, cold on another — the
        # per-source breakdown surfaces the directional disagreement that a
        # global mean would average away to ~0.
        rows = [
            _row(9.0, 5.0, "hot"),
            _row(8.0, 4.0, "hot"),
            _row(2.0, 6.0, "cold"),
            _row(3.0, 7.0, "cold"),
        ]
        out = compute_per_source(rows)
        by_src = {s["source"]: s for s in out["sources"]}
        assert by_src["hot"]["bias_ml_minus_ai"] == 4.0
        assert by_src["cold"]["bias_ml_minus_ai"] == -4.0

    def test_sorted_by_descending_n(self):
        rows = (
            [_row(5.0, 5.0, "small")] * 2
            + [_row(5.0, 5.0, "big")] * 7
            + [_row(5.0, 5.0, "mid")] * 4
        )
        out = compute_per_source(rows)
        order = [s["source"] for s in out["sources"]]
        assert order == ["big", "mid", "small"]
