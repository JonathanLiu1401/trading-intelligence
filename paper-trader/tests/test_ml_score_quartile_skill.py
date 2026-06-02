"""Tests for paper_trader.ml.ml_score_quartile_skill.

These tests pin the verdict ladder, the bucket schema contract, and the
defensive load/error paths. Synthetic scorers control the prediction-vs-
realized relationship deterministically so the rank-IC math is exactly
known per bucket — no model-training noise.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml.ml_score_quartile_skill import (
    MIN_PAIRS_PER_BUCKET,
    SPEARMAN_FLAT,
    SPEARMAN_GOOD,
    N_BUCKETS,
    analyze,
    build_quartile_skill_report,
    load_outcomes,
)


# ─────────────────────── synthetic scorer stubs ───────────────────────

class _PerfectScorer:
    """A scorer whose prediction == realized return. rank_ic should be +1
    on any sufficiently-large bucket."""
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None,
                news_urgency=None, news_article_count=None, **_extra_kwargs):
        # Carry the "future" realized return as a side-channel via the
        # `mom5` field — the test records below pack the realized value
        # into that slot. This isolates the rank-IC math from training.
        return float(mom5 or 0.0)


class _InverseScorer:
    """A scorer that predicts the OPPOSITE of realized — rank-IC ≈ -1."""
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None,
                news_urgency=None, news_article_count=None, **_extra_kwargs):
        return -float(mom5 or 0.0)


class _ConstantScorer:
    """A scorer that returns the SAME prediction for every row — rank_ic=0
    (tie-aware Spearman returns 0.0 on a constant predictor; the legacy
    argsort(argsort) would have fabricated rank skill here)."""
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None,
                news_urgency=None, news_article_count=None, **_extra_kwargs):
        return 1.0


class _TopHeavyScorer:
    """A scorer whose prediction matches realized ONLY for rows with
    ml_score in the top quartile. Lower quartiles get a flat 0
    prediction so rank-IC is 0 there. Produces the CONCENTRATED_HIGH
    verdict shape."""
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None,
                news_urgency=None, news_article_count=None, **_extra_kwargs):
        if (ml_score or 0.0) >= 3.0:   # top quartile of the 0..4 synthetic range
            return float(mom5 or 0.0)
        return 0.0


class _OnlyBottomBucketScorer:
    """Verifies the analyzer faithfully reports per-bucket rank-IC.

    Returns realized in the bottom bucket (perfect order) and the
    SCRAMBLED realized value in higher buckets (creates within-bucket
    noise so per-bucket rank-IC is ≈ 0 there).

    The aggregate Spearman with this construction tends to come out
    INVERTED in practice because constant-or-shuffled predictions in
    the top buckets create rank inversions globally — that is itself
    an honest, useful verdict (the documented "the gate is sizing on
    anti-predictive signal" alarm). The test below only pins the
    PER-BUCKET behavior (bot quartile carries skill, others don't),
    not the global verdict — both INVERTED and CONCENTRATED_LOW are
    legitimate readings of the shape depending on synthesized data.
    """
    is_trained = True

    def predict(self, ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=None, bb_pos=None,
                news_urgency=None, news_article_count=None, **_extra_kwargs):
        if (ml_score or 0.0) <= 1.0:
            return float(mom5 or 0.0)
        # Within-bucket noise: alternate predictions so rank-IC ≈ 0.
        v = float(mom5 or 0.0)
        return v + (1.0 if int(v * 100) % 2 == 0 else -1.0)


def _outcome(ml_score, realized, action="BUY", sim_date="2025-01-01",
             ticker="NVDA"):
    """Build a synthetic outcome row. Pack realized into both
    forward_return_5d (the analyzer's target) AND mom5 (the synthetic
    scorer's side-channel) so the scorer can rank perfectly without
    needing a trained model."""
    return {
        "ticker": ticker, "sim_date": sim_date, "action": action,
        "ml_score": float(ml_score), "rsi": 50.0, "macd": 0.0,
        "mom5": float(realized), "mom20": 0.0, "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "forward_return_5d": float(realized),
    }


def _records(scorer_type: str, n_per_bucket: int = 20) -> list[dict]:
    """Generate `4 * n_per_bucket` rows spread across 4 ml_score quartiles
    (0..4) with realized = ml_score * 2 + small jitter. Always returns
    distinct sim_dates so analyzer-side dedup never collapses rows."""
    out = []
    idx = 0
    for q in range(4):
        for i in range(n_per_bucket):
            ml = q + i / max(n_per_bucket - 1, 1)  # spread within bucket
            realized = ml * 2.0  # perfect monotone signal
            day = (idx % 28) + 1
            month = (idx // 28) % 12 + 1
            year = 2025 + (idx // (28 * 12))
            out.append(_outcome(
                ml_score=ml, realized=realized,
                sim_date=f"{year:04d}-{month:02d}-{day:02d}"))
            idx += 1
    return out


# ─────────────────────── analyzer behaviour ───────────────────────

class TestEmptyAndUntrained:
    def test_untrained_scorer_returns_insufficient(self):
        class _Un:
            is_trained = False

        rep = build_quartile_skill_report(_Un(), _records("perfect"))
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "not trained" in rep["hint"]

    def test_empty_records_returns_insufficient(self):
        rep = build_quartile_skill_report(_PerfectScorer(), [])
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_too_few_records_for_quartile_cut(self):
        # 4 buckets × MIN_PAIRS_PER_BUCKET = 40 required; supply 10.
        recs = _records("perfect", n_per_bucket=3)  # 12 rows
        rep = build_quartile_skill_report(_PerfectScorer(), recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 12

    def test_corrupt_rows_increment_drop_counters(self):
        # Mix in non-dict / missing-field rows; analyzer must tolerate
        # and tally honestly.
        good = _records("perfect", n_per_bucket=11)  # 44 rows
        recs = good + [
            "not a dict",
            None,
            123,
            {"ml_score": None},                          # no_ml_score
            {"ml_score": 1.0, "forward_return_5d": None}, # no_return
        ]
        rep = build_quartile_skill_report(_PerfectScorer(), recs)
        assert rep["status"] == "ok"
        assert rep["n"] == 44   # only good rows counted
        # The string/None/int rows fall to no_ml_score (the first filter).
        assert rep["n_dropped_no_ml_score"] >= 4
        # The one row with explicit no return.
        assert rep["n_dropped_no_return"] >= 1


class TestVerdictLadder:
    def test_perfect_signal_yields_concentrated_high_or_uniform(self):
        """A perfectly-aligned signal across all buckets — every quartile
        has rank-IC ≈ 1.0. The verdict should be UNIFORM (spread ≈ 0)
        — strictly correct under the uniform-skill clause."""
        rep = build_quartile_skill_report(
            _PerfectScorer(), _records("perfect", n_per_bucket=20))
        assert rep["status"] == "ok"
        # Every bucket has rank-IC > SPEARMAN_GOOD
        for b in rep["buckets"]:
            assert b["rank_ic"] is not None
            assert b["rank_ic"] >= SPEARMAN_GOOD, (
                f"perfect scorer bucket {b['idx']} should have high rank-IC; "
                f"got {b['rank_ic']}")
        # Aggregate >> 0
        assert rep["aggregate_rank_ic"] >= SPEARMAN_GOOD
        # The verdict for a fully aligned signal is UNIFORM (top-bot
        # spread is near zero since every bucket is +1.0).
        assert rep["verdict"] == "UNIFORM"

    def test_inverse_scorer_yields_inverted(self):
        rep = build_quartile_skill_report(
            _InverseScorer(), _records("inverse", n_per_bucket=20))
        assert rep["verdict"] == "INVERTED"
        assert rep["aggregate_rank_ic"] is not None
        assert rep["aggregate_rank_ic"] <= -SPEARMAN_FLAT

    def test_constant_scorer_yields_no_skill(self):
        # A constant predictor has Spearman==0 by the tie-aware contract.
        rep = build_quartile_skill_report(
            _ConstantScorer(), _records("constant", n_per_bucket=20))
        # Every bucket has |rank_ic| < SPEARMAN_FLAT
        for b in rep["buckets"]:
            assert b["rank_ic"] is not None
            assert abs(b["rank_ic"]) < SPEARMAN_FLAT
        assert rep["verdict"] == "NO_SKILL"

    def test_top_heavy_scorer_yields_concentrated_high(self):
        rep = build_quartile_skill_report(
            _TopHeavyScorer(), _records("top", n_per_bucket=20))
        # Top quartile has real skill; lower quartiles have rank-IC ≈ 0
        top = rep["buckets"][-1]
        bot = rep["buckets"][0]
        assert top["rank_ic"] >= SPEARMAN_GOOD
        assert abs(bot["rank_ic"]) < SPEARMAN_FLAT
        assert rep["verdict"] == "CONCENTRATED_HIGH"

    def test_per_bucket_skill_reported_faithfully(self):
        """The analyzer must honestly report per-bucket rank-IC even when
        the aggregate verdict swallows the nuance. Locks the per-bucket
        contract — buckets[].rank_ic is the source of truth a quant
        researcher reads to localise the scorer's edge."""
        rep = build_quartile_skill_report(
            _OnlyBottomBucketScorer(),
            _records("only_bot", n_per_bucket=20))
        bot = rep["buckets"][0]
        top = rep["buckets"][-1]
        # Bottom carries strong rank-IC (perfect prediction).
        assert bot["rank_ic"] is not None
        assert bot["rank_ic"] >= SPEARMAN_GOOD, (
            f"bottom bucket should carry skill; rank_ic={bot['rank_ic']}")
        # Top must have STRICTLY lower rank skill than bottom (the
        # scoring asymmetry is the point of this test). Doesn't require
        # the top rank-IC to be near zero — the alternation-based noise
        # we inject still has some residual order. The localised-skill
        # contract is captured by "bot >> top".
        assert top["rank_ic"] is not None
        assert top["rank_ic"] < bot["rank_ic"] - 0.20, (
            f"top should have materially less rank skill than bottom; "
            f"top={top['rank_ic']:+.3f} vs bot={bot['rank_ic']:+.3f}")


class TestBucketSchema:
    def test_bucket_count_matches_n_buckets(self):
        rep = build_quartile_skill_report(
            _PerfectScorer(), _records("perfect", n_per_bucket=15))
        assert len(rep["buckets"]) == N_BUCKETS

    def test_ml_score_lo_le_hi_per_bucket(self):
        rep = build_quartile_skill_report(
            _PerfectScorer(), _records("perfect", n_per_bucket=15))
        for b in rep["buckets"]:
            assert b["ml_score_lo"] <= b["ml_score_hi"], (
                f"bucket {b['idx']} has ml_score_lo > hi: {b}")

    def test_buckets_are_monotone_by_ml_score(self):
        # Bucket 1 max ≤ Bucket 2 min: quartile cut must produce sorted
        # ranges (a future refactor must not silently lose this).
        rep = build_quartile_skill_report(
            _PerfectScorer(), _records("perfect", n_per_bucket=15))
        for a, b in zip(rep["buckets"][:-1], rep["buckets"][1:]):
            assert a["ml_score_hi"] <= b["ml_score_lo"] + 1e-9, (
                f"buckets not sorted: {a} then {b}")

    def test_bucket_n_sums_to_total(self):
        rep = build_quartile_skill_report(
            _PerfectScorer(), _records("perfect", n_per_bucket=15))
        total = sum(b["n"] for b in rep["buckets"])
        assert total == rep["n"]


class TestSellSignFlip:
    def test_sell_rows_have_realized_negated(self):
        """A SELL whose realized return is -5% should be treated as a
        +5% "good" sign-flipped outcome. Mirror the conviction of
        evaluate_scorer_oos / _oos_rank_metrics."""
        # Mix BUY + SELL with sign-flipped realized. The PerfectScorer's
        # mom5 trick still works because SELL → -realized, so the
        # predicted side-channel (mom5) needs the SAME flipped value
        # for the scorer to "predict" the post-flip target.
        recs = []
        for i in range(40):
            ml = i / 10.0  # 0.0 to 3.9
            realized = ml * 2.0
            recs.append(_outcome(ml_score=ml, realized=realized,
                                 sim_date=f"2025-01-{(i % 28) + 1:02d}"))
        # Now 40 SELLs where realized = -ml*2.0 but the scorer sees
        # mom5 = +ml*2.0 (post-flip). Verifies the analyzer applies the
        # sign-flip BEFORE comparing prediction to realized.
        for i in range(40):
            ml = 0.5 + i / 10.0
            recs.append({
                "ticker": "NVDA",
                "sim_date": f"2025-02-{(i % 28) + 1:02d}",
                "action": "SELL",
                "ml_score": float(ml),
                "rsi": 50.0, "macd": 0.0,
                "mom5": float(ml * 2.0),  # what the scorer predicts (flipped good)
                "mom20": 0.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": float(-ml * 2.0),  # raw realized
            })
        rep = build_quartile_skill_report(_PerfectScorer(), recs)
        # Strongly positive aggregate (both BUY and post-flip SELL agree).
        assert rep["status"] == "ok"
        assert rep["aggregate_rank_ic"] >= SPEARMAN_GOOD


# ─────────────────────── loader + CLI ───────────────────────

class TestLoadOutcomes:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_outcomes(tmp_path / "nope.jsonl") == []

    def test_jsonl_loads_records(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text("\n".join(
            json.dumps({"ml_score": float(i), "forward_return_5d": float(i)})
            for i in range(10)) + "\n")
        recs = load_outcomes(p)
        assert len(recs) == 10

    def test_corrupt_lines_skipped(self, tmp_path):
        p = tmp_path / "corrupt.jsonl"
        p.write_text(
            json.dumps({"ml_score": 1.0, "forward_return_5d": 5.0}) + "\n"
            "not json\n"
            + json.dumps({"ml_score": 2.0, "forward_return_5d": 6.0}) + "\n"
        )
        recs = load_outcomes(p)
        # The corrupt line is silently dropped.
        assert len(recs) == 2


class TestAnalyzeIntegration:
    def test_analyze_missing_file_yields_insufficient(self, tmp_path):
        rep = analyze(tmp_path / "nope.jsonl")
        # Untrained scorer (no live pickle) OR no records: either lands
        # in the INSUFFICIENT_DATA bucket.
        assert rep["status"] in ("insufficient_data", "error")
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_analyze_with_synthetic_outcomes(self, tmp_path, monkeypatch):
        # Wire a fresh outcomes file + an untrained scorer; expect the
        # graceful "not trained" insufficient-data path.
        p = tmp_path / "out.jsonl"
        p.write_text("\n".join(
            json.dumps(_outcome(ml_score=float(i % 4),
                                realized=float(i % 4) * 2,
                                sim_date=f"2025-0{(i % 9) + 1}-01"))
            for i in range(40)))
        rep = analyze(p, oos_only=False)
        assert isinstance(rep, dict)
        assert "verdict" in rep
        # Whatever the live scorer says, the call must never raise.
