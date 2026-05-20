"""Tests for paper_trader.ml.news_volume_skill — per-news-volume scorer skill.

Mirrors test discipline of tests/test_action_skill.py — every test asserts
a specific expected verdict or numeric output, not just "no crash". Offline
by construction (scorer stubs + synthetic outcome records).
"""
from __future__ import annotations

import pytest

from paper_trader.ml import news_volume_skill as nvs


class _ScorerStub:
    """Minimal scorer stub. Yields predictions per call from a list.

    Mirrors DecisionScorer's predict signature shape — the test doesn't
    care about feature values; only the per-call output sequence drives
    the rank correlation we want to lock.
    """

    is_trained = True

    def __init__(self, predictions: list[float]):
        self._preds = list(predictions)
        self._i = 0

    def predict(self, **_kw) -> float:
        if self._i >= len(self._preds):
            raise IndexError("test scorer ran out of predictions")
        p = self._preds[self._i]
        self._i += 1
        return p


class _UntrainedScorer:
    is_trained = False

    def predict(self, **_kw):
        return 0.0


def _rec(news_count, fr: float, action: str = "BUY",
         ml_score: float = 1.0) -> dict:
    """Compact synthetic outcome row in the decision_outcomes.jsonl shape.

    `news_count` can be None (no-news sentinel), 0, or any positive count.
    """
    return {
        "action": action,
        "ticker": "NVDA",
        "sim_date": "2025-06-15",
        "ml_score": ml_score,
        "rsi": 50, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
        "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0 if news_count else None,
        "news_article_count": news_count,
        "forward_return_5d": fr,
    }


# ─────────────────────── _bucket_for ───────────────────────


class TestBucketFor:
    def test_none_maps_to_no_news(self):
        assert nvs._bucket_for(None) == "no_news"

    def test_zero_maps_to_no_news(self):
        # The 0-count case (`_compute_decision_outcomes` normalises this to
        # None upstream, but the bucket must still place a bare 0 correctly).
        assert nvs._bucket_for(0) == "no_news"
        assert nvs._bucket_for(0.0) == "no_news"

    def test_one_maps_to_sparse(self):
        assert nvs._bucket_for(1) == "sparse"

    def test_two_maps_to_sparse(self):
        assert nvs._bucket_for(2) == "sparse"

    def test_three_maps_to_moderate(self):
        assert nvs._bucket_for(3) == "moderate"

    def test_nine_maps_to_moderate(self):
        assert nvs._bucket_for(9) == "moderate"

    def test_ten_maps_to_dense(self):
        assert nvs._bucket_for(10) == "dense"

    def test_large_maps_to_dense(self):
        assert nvs._bucket_for(500) == "dense"

    def test_nan_returns_none(self):
        assert nvs._bucket_for(float("nan")) is None

    def test_negative_returns_none(self):
        assert nvs._bucket_for(-1) is None

    def test_unparseable_returns_none(self):
        assert nvs._bucket_for("not_a_number") is None
        assert nvs._bucket_for([]) is None


# ─────────────────────── _verdict_for ───────────────────────


class TestVerdictFor:
    def test_insufficient_below_n(self):
        # n below the per-bucket minimum is INSUFFICIENT regardless of ic
        n = nvs.MIN_OUTCOMES_PER_BUCKET - 1
        assert nvs._verdict_for(0.5, n) == "INSUFFICIENT"

    def test_none_ic_is_insufficient(self):
        assert nvs._verdict_for(None, 100) == "INSUFFICIENT"

    def test_nan_ic_is_insufficient(self):
        assert nvs._verdict_for(float("nan"), 100) == "INSUFFICIENT"

    def test_edge_at_threshold(self):
        # ic >= IC_GOOD is EDGE
        assert nvs._verdict_for(nvs.IC_GOOD, 100) == "EDGE"

    def test_inverted_at_threshold(self):
        assert nvs._verdict_for(-nvs.IC_GOOD, 100) == "INVERTED"

    def test_weak_edge(self):
        # Halfway between IC_MIN and IC_GOOD
        assert nvs._verdict_for((nvs.IC_MIN + nvs.IC_GOOD) / 2, 100) == "WEAK_EDGE"

    def test_no_edge_zero(self):
        assert nvs._verdict_for(0.0, 100) == "NO_EDGE"


# ─────────────────────── news_volume_skill ───────────────────────


class TestNewsVolumeSkill:
    def test_untrained_scorer_yields_untrained_status(self):
        rep = nvs.news_volume_skill(_UntrainedScorer(), [_rec(0, 5.0)])
        assert rep["status"] == "untrained"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_empty_records_yields_insufficient(self):
        rep = nvs.news_volume_skill(_ScorerStub([]), [])
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0

    def test_below_min_records_yields_insufficient(self):
        # 5 records < MIN_RECORDS (30) → insufficient_data
        recs = [_rec(0, 1.0) for _ in range(5)]
        scorer = _ScorerStub([1.0] * 5)
        rep = nvs.news_volume_skill(scorer, recs)
        assert rep["status"] == "insufficient_data"

    def test_monotonic_positive_verdict_synthetic(self):
        """Build a corpus where rank-IC strictly improves with news count.

        For each bucket we emit `MIN_OUTCOMES_PER_BUCKET` records. Across a
        bucket, scorer predictions are correlated with realized returns by
        a factor that GROWS with bucket index — no_news = pure noise (no
        correlation), sparse = small alignment, moderate = larger, dense =
        strong alignment. Result: rank_ic[no_news] < rank_ic[sparse] <
        rank_ic[moderate] < rank_ic[dense], a monotone-positive verdict."""
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        recs, preds = [], []
        # Bucket 0 (no_news): random preds, fixed realized (rank-IC ≈ 0).
        for i in range(N):
            recs.append(_rec(None, (i % 3) - 1))
            preds.append((i * 7) % 5 - 2)  # bouncy preds, weakly correlated
        # Bucket 1 (sparse, count=1): mild correlation (pred matches fr in sign).
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(1, fr))
            preds.append(fr * 0.3)
        # Bucket 2 (moderate, count=5): stronger correlation.
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(5, fr))
            preds.append(fr * 0.7)
        # Bucket 3 (dense, count=15): strongest correlation (pred == fr exactly).
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(15, fr))
            preds.append(fr)
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # Each bucket must have sufficient n
        for b in nvs._BUCKET_NAMES:
            cell = rep["by_bucket"][b]
            assert cell["n"] == N, f"{b}: {cell}"
        # rank-IC must be strictly monotone non-decreasing
        ics = [rep["by_bucket"][b]["rank_ic"] for b in nvs._BUCKET_NAMES]
        for i in range(len(ics) - 1):
            assert ics[i] <= ics[i + 1] + 1e-9, (
                f"non-monotone rank_ic at bucket index {i}: {ics}")
        # And the spread must be large enough to clear BUCKET_SPREAD_TOL.
        assert max(ics) - min(ics) >= nvs.BUCKET_SPREAD_TOL
        assert rep["verdict"] == "NEWS_VALUE_MONOTONIC_POSITIVE"

    def test_invariant_verdict_when_no_skill_differential(self):
        """All buckets carry the SAME small weak rank correlation — overall
        spread < BUCKET_SPREAD_TOL → NEWS_VALUE_INVARIANT.

        The cleanest way to guarantee a stable per-bucket IC is identical
        (pred, fr) pairs across all buckets. Each bucket then has IDENTICAL
        rank-IC (spread = 0), so the verdict must be INVARIANT.
        """
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        # Build one canonical (fr, pred) sequence with mild correlation.
        frs = [float(i - N / 2) for i in range(N)]
        # Pred shuffled deterministically so rank-IC is weakly positive but
        # well below IC_GOOD; tied across buckets so spread is exactly 0.
        canonical_preds = [frs[(i * 5) % N] * 0.05 for i in range(N)]
        recs, preds = [], []
        for bucket_count in (None, 1, 5, 15):
            for i in range(N):
                recs.append(_rec(bucket_count, frs[i]))
                preds.append(canonical_preds[i])
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # Every sufficient bucket must have the SAME rank-IC value.
        ics = [rep["by_bucket"][b]["rank_ic"] for b in nvs._BUCKET_NAMES
               if rep["by_bucket"][b]["rank_ic"] is not None]
        assert len(ics) == 4, f"expected 4 sufficient buckets, got {len(ics)}"
        assert max(ics) - min(ics) < 1e-6, (
            f"identical (pred, fr) sequences must yield identical ICs: {ics}")
        # No bucket can be INVERTED (rank_ic is well above -IC_GOOD).
        for b in nvs._BUCKET_NAMES:
            assert rep["by_bucket"][b]["verdict"] != "INVERTED", (
                f"bucket {b}: {rep['by_bucket'][b]}")
        assert rep["verdict"] == "NEWS_VALUE_INVARIANT"

    def test_inverted_bucket_surfaces_red_flag(self):
        """Build a corpus where one bucket has rank-IC < -IC_GOOD. The
        overall verdict must be HAS_INVERTED_BUCKET regardless of other
        buckets — operator-actionable surfacing test."""
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        recs, preds = [], []
        # no_news, sparse, moderate: small positive correlation
        for bucket_count in (None, 1, 5):
            for i in range(N):
                fr = float(i - N / 2)
                recs.append(_rec(bucket_count, fr))
                preds.append(fr * 0.5)
        # dense bucket: predictions ANTI-correlate (pred = -fr exactly).
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(15, fr))
            preds.append(-fr)
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # The dense bucket's rank-IC must be deeply negative
        assert rep["by_bucket"]["dense"]["rank_ic"] is not None
        assert rep["by_bucket"]["dense"]["rank_ic"] <= -nvs.IC_GOOD
        assert rep["by_bucket"]["dense"]["verdict"] == "INVERTED"
        # And the overall verdict must surface this as the red flag.
        assert rep["verdict"] == "HAS_INVERTED_BUCKET"

    def test_sell_action_flips_realized_sign(self):
        """A SELL record with fr=+5 → action-aligned target = -5 (flipped).
        Pair it with pred=+5 → opposite signs → rank-IC should be NEGATIVE
        when the SELL slice dominates. Mirrors the universal SELL-flip
        convention every other diagnostic uses."""
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        recs, preds = [], []
        # All records are SELL in the dense bucket with pred == +fr.
        # SELL-flip → realized = -fr. So Spearman(pred=+fr, realized=-fr) = -1.
        # Need MIN_RECORDS overall, so pad with no_news SELL records too.
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(15, fr, action="SELL"))
            preds.append(fr)
        for i in range(N):
            fr = float(i - N / 2)
            recs.append(_rec(None, fr, action="SELL"))
            preds.append(fr)
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # Both buckets should land at rank_ic ≈ -1 (perfectly inverse)
        for b in ("no_news", "dense"):
            ic = rep["by_bucket"][b]["rank_ic"]
            assert ic is not None
            assert ic <= -0.99, f"{b}: {ic}"
        assert rep["verdict"] == "HAS_INVERTED_BUCKET"

    def test_only_one_sufficient_bucket_yields_insufficient(self):
        """When ≥MIN_RECORDS overall but only ONE bucket has ≥
        MIN_OUTCOMES_PER_BUCKET, the verdict must be INSUFFICIENT_DATA
        (need ≥2 sufficient buckets to compare)."""
        N = nvs.MIN_RECORDS + 5
        recs = [_rec(15, float(i - N // 2)) for i in range(N)]
        preds = [float(i - N // 2) for i in range(N)]
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # Only dense bucket has enough records
        assert rep["by_bucket"]["dense"]["verdict"] != "INSUFFICIENT"
        assert rep["by_bucket"]["sparse"]["verdict"] == "INSUFFICIENT"
        assert rep["by_bucket"]["moderate"]["verdict"] == "INSUFFICIENT"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_non_finite_forward_return_dropped(self):
        """Records with non-finite forward_return_5d must be dropped, not
        crash or pollute Spearman."""
        # Build a clean monotonic-positive corpus, then sprinkle in NaN rows.
        # The clean corpus must be enough to still meet MIN_RECORDS.
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        recs, preds = [], []
        for bucket_count in (None, 1, 5, 15):
            for i in range(N):
                fr = float(i - N / 2)
                recs.append(_rec(bucket_count, fr))
                preds.append(fr)
        # Add a NaN record in the moderate bucket — must be silently dropped.
        bad = _rec(5, float("nan"))
        recs.append(bad)
        # No matching prediction needed since the record will be dropped
        # BEFORE reaching scorer.predict (the `t != t` NaN check is in
        # _aligned_pred before the predict call).
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        assert rep["status"] == "ok"
        # All four buckets should have exactly N records — the NaN was dropped.
        for b in nvs._BUCKET_NAMES:
            assert rep["by_bucket"][b]["n"] == N

    def test_records_without_news_article_count_drop(self):
        """A record whose news_article_count is unparseable (e.g. a string)
        is dropped — `_bucket_for` returns None, so the record never reaches
        a bucket. Total aligned outcomes excludes them."""
        N = nvs.MIN_OUTCOMES_PER_BUCKET
        recs, preds = [], []
        # Build a clean monotonic-positive corpus.
        for bucket_count in (None, 1, 5, 15):
            for i in range(N):
                fr = float(i - N / 2)
                recs.append(_rec(bucket_count, fr))
                preds.append(fr)
        # Inject one record with unparseable count — must be dropped.
        bad = _rec(None, 1.0)
        bad["news_article_count"] = "totally_invalid"
        recs.append(bad)
        rep = nvs.news_volume_skill(_ScorerStub(preds), recs)
        # n_records counts only successfully bucketed records
        assert rep["n_records"] == 4 * N


# ─────────────────────── analyze (CLI entrypoint) ───────────────────────


class TestAnalyze:
    def test_missing_outcomes_file_yields_insufficient(self, tmp_path):
        """A missing outcomes file degrades to INSUFFICIENT, not a crash."""
        missing = tmp_path / "no_such_file.jsonl"
        rep = nvs.analyze(missing, oos_only=False)
        # status is either error (scorer load failed) or insufficient_data
        # (records empty). Both are honest degraded responses.
        assert rep["status"] in ("error", "insufficient_data", "untrained")
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_corrupt_outcomes_line_skipped(self, tmp_path):
        """An unparseable JSONL line is skipped — never raises."""
        p = tmp_path / "outcomes.jsonl"
        p.write_text("this is not json\n{\"bad json\n")
        rep = nvs.analyze(p, oos_only=False)
        # All lines skipped → no records → insufficient
        assert rep["verdict"] == "INSUFFICIENT_DATA"
