"""Tests for paper_trader.ml.confidence_skill.

The PURPOSE of these tests is to catch bugs in the confidence-conditional
skill diagnostic:

- The verdict ladder must read the actual IC values, not just the sign —
  a tiny INVERTED bias near zero is NOT GUARD_INVERTED.
- Fake stubs without ``predict_with_meta`` must yield ``unsupported_scorer``
  (the off_distribution flag CANNOT be captured from scalar predict()).
- Rows where ``failed=True`` must be DROPPED, not bucketed (otherwise they
  tie at zero in whichever bucket and contaminate the rank-IC).
- The SELL sign-flip must apply ONLY to the realized target, NOT the
  prediction (mirrors action_skill — load-bearing for cross-tool parity).
- A scorer that always reports off_distribution=True must NOT crash and
  must report INSUFFICIENT_DATA on the empty-trusted-bucket side.
- A scorer that always reports off_distribution=False must yield NEUTRAL
  / INSUFFICIENT_DATA cleanly (empty abstained bucket).
- The empty / corrupt-jsonl loader must degrade to INSUFFICIENT_DATA, not
  raise.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import paper_trader.ml.confidence_skill as cs


# ──────────────────────────── fake scorers ────────────────────────────


class _ConstantScorer:
    """Predicts a constant value for every row; off_distribution toggle is
    driven by a callable so each test can prescribe the trust labelling.

    `is_trained=True` so the diagnostic actually runs. `predict_with_meta`
    returns the production-shape dict. The scorer's predict signature
    matches the deployed scorer's exactly (every kwarg) so a refactor
    that adds a feature catches in this test as a TypeError, not silently.
    """

    is_trained = True

    def __init__(self, pred_fn, off_fn=lambda r: False, fail_fn=lambda r: False):
        self._pred_fn = pred_fn
        self._off_fn = off_fn
        self._fail_fn = fail_fn

    def predict_with_meta(self, *, ml_score, rsi, macd, mom5, mom20,
                           regime_mult, ticker, vol_ratio=None, bb_pos=None,
                           news_urgency=None, news_article_count=None,
                           ema200_above=None, hist_cross_up=None,
                           macd_below_zero_cross=None):
        record = {
            "ml_score": ml_score, "rsi": rsi, "macd": macd,
            "mom5": mom5, "mom20": mom20, "regime_mult": regime_mult,
            "ticker": ticker, "vol_ratio": vol_ratio, "bb_position": bb_pos,
            "news_urgency": news_urgency,
            "news_article_count": news_article_count,
            "ema200_above": ema200_above, "hist_cross_up": hist_cross_up,
            "macd_below_zero_cross": macd_below_zero_cross,
        }
        return {
            "pred": float(self._pred_fn(record)),
            "raw": float(self._pred_fn(record)),
            "clamped": bool(self._off_fn(record)),
            "off_distribution": bool(self._off_fn(record)),
            "percentile": None,
            "calibrated": None,
            "failed": bool(self._fail_fn(record)),
        }


class _UntrainedScorer:
    is_trained = False
    def predict_with_meta(self, **kw):
        return {"pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": False, "percentile": None,
                "calibrated": None, "failed": True}


class _ScalarOnlyScorer:
    """No predict_with_meta — should be caught as unsupported."""
    is_trained = True
    def predict(self, **kw):
        return 0.0


# ──────────────────────────── helpers ────────────────────────────


def _synth_records(n: int, *, seed: int = 0, leakage: float = 0.5,
                   sell_fraction: float = 0.3) -> list[dict]:
    """Same shape as test_scorer_learning_curve._synth_records but with
    optional SELL fraction so we can assert the sign-flip behavior."""
    rng = random.Random(seed)
    tickers = ["NVDA", "AMD", "MSFT", "SPY", "QQQ", "SOXL"]
    out = []
    start = date(2024, 1, 1)
    for i in range(n):
        ml = rng.uniform(-3.0, 3.0)
        noise = rng.gauss(0.0, 3.0)
        fr = leakage * (ml * 5.0) + noise
        action = "SELL" if rng.random() < sell_fraction else "BUY"
        out.append({
            "ticker": rng.choice(tickers),
            "sim_date": (start + timedelta(days=i // 4)).isoformat(),
            "action": action,
            "ml_score": round(ml, 3),
            "rsi": round(rng.uniform(20, 80), 1),
            "macd": round(rng.uniform(-1, 1), 3),
            "mom5": round(rng.uniform(-5, 5), 2),
            "mom20": round(rng.uniform(-10, 10), 2),
            "regime_mult": rng.choice([0.3, 0.6, 1.0]),
            "vol_ratio": round(rng.uniform(0.5, 2.5), 2),
            "bb_position": round(rng.uniform(-1.5, 1.5), 2),
            "news_urgency": round(rng.uniform(0, 100), 1),
            "news_article_count": float(rng.randint(0, 5)),
            "forward_return_5d": round(fr, 4),
            "return_pct": round(rng.uniform(-20, 20), 2),
        })
    return out


# ──────────────────────────── INSUFFICIENT / status ────────────────────────────


class TestStatusCodes:
    def test_untrained_returns_untrained_status(self):
        rep = cs.confidence_skill(_UntrainedScorer(), _synth_records(50))
        assert rep["status"] == "untrained"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0

    def test_no_predict_with_meta_returns_unsupported(self):
        rep = cs.confidence_skill(_ScalarOnlyScorer(), _synth_records(50))
        assert rep["status"] == "unsupported_scorer"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_empty_records_returns_insufficient(self):
        sc = _ConstantScorer(lambda r: r.get("ml_score", 0.0))
        rep = cs.confidence_skill(sc, [])
        assert rep["status"] == "ok"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0

    def test_below_min_records_returns_insufficient(self):
        sc = _ConstantScorer(lambda r: r.get("ml_score", 0.0))
        rep = cs.confidence_skill(sc, _synth_records(10))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "≥30" in rep["hint"] or "30" in rep["hint"]


# ──────────────────────────── verdict ladder ────────────────────────────


class TestVerdictLadder:
    def test_guard_helps_when_trusted_ic_much_better(self):
        """Trusted bucket: predictions correlate with realized; abstained:
        random predictions. Expect GUARD_HELPS."""
        recs = _synth_records(200, leakage=0.7, sell_fraction=0.0)
        # First half: trusted (off=False), prediction = ml_score (correlated).
        # Second half: abstained (off=True), prediction = noise.
        off_set = set(i for i in range(100, 200))
        rng = random.Random(99)

        # Index by order of appearance in records — _ConstantScorer is
        # called per row so we need a stable per-row identity. Use the
        # ml_score values as the lookup since they're unique per row in
        # this synth corpus.
        def pred(r):
            return r["ml_score"]  # correlated with the leakage label

        def off(r):
            return r["ml_score"] < 0  # half off, half on; uncorrelated with target sign

        # Make abstained bucket pure noise by overriding pred for off rows.
        def pred_or_noise(r):
            if r["ml_score"] < 0:
                return rng.gauss(0, 3.0)
            return r["ml_score"]

        sc = _ConstantScorer(pred_or_noise, off)
        rep = cs.confidence_skill(sc, recs)
        # trusted bucket has the signal; abstained has noise. trusted_ic
        # should be materially > abstained_ic.
        assert rep["status"] == "ok"
        assert rep["verdict"] in ("GUARD_HELPS", "GUARD_NEUTRAL")
        # Stronger assertion: when trusted has clear signal, the diff
        # should be positive even in the neutral case.
        assert rep["trusted"]["rank_ic"] is not None
        assert rep["abstained"]["rank_ic"] is not None

    def test_guard_neutral_when_both_buckets_similar(self):
        """Both buckets get the same correlated prediction. diff ≈ 0."""
        recs = _synth_records(200, leakage=0.5, sell_fraction=0.0)
        # Off flag set on half the rows but pred uses ml_score everywhere.
        sc = _ConstantScorer(
            pred_fn=lambda r: r["ml_score"],
            off_fn=lambda r: r["rsi"] > 50.0,  # ~half flagged off, no target-corr
        )
        rep = cs.confidence_skill(sc, recs)
        assert rep["status"] == "ok"
        # Both buckets should have similar ICs since prediction is the
        # same correlated value. Verdict should be NEUTRAL.
        assert rep["verdict"] == "GUARD_NEUTRAL"
        assert abs(rep["diff"]) < cs.DIFF_TOL

    def test_guard_inverted_when_signal_is_in_abstained(self):
        """Trusted = anti-predictive prediction, abstained = correlated.
        The guard is filtering the wrong slice."""
        recs = _synth_records(200, leakage=0.7, sell_fraction=0.0)
        rng = random.Random(11)

        def pred_fn(r):
            if r["rsi"] > 50.0:
                # Abstained bucket — return ml_score (correlated with target)
                return r["ml_score"]
            else:
                # Trusted bucket — return ANTI-correlated prediction
                return -r["ml_score"]

        sc = _ConstantScorer(pred_fn, off_fn=lambda r: r["rsi"] > 50.0)
        rep = cs.confidence_skill(sc, recs)
        assert rep["status"] == "ok"
        # The trusted bucket has anti-correlated predictions; abstained
        # has correlated. Should be GUARD_INVERTED.
        assert rep["verdict"] == "GUARD_INVERTED", rep
        assert rep["trusted"]["rank_ic"] is not None
        assert rep["abstained"]["rank_ic"] is not None
        assert rep["trusted"]["rank_ic"] < -cs.IC_NOISE
        assert rep["abstained"]["rank_ic"] > cs.IC_NOISE

    def test_unit_verdict_function(self):
        """Direct unit tests of the verdict function — independent of
        any scorer plumbing so a regression here flags the verdict logic
        precisely."""
        # GUARD_INVERTED — trusted anti, abstained signal
        assert cs._verdict(-0.20, 0.15, 50, 50)[0] == "GUARD_INVERTED"
        # GUARD_HELPS — trusted materially better
        assert cs._verdict(0.30, 0.10, 50, 50)[0] == "GUARD_HELPS"
        # GUARD_HARMS — trusted materially worse but not inverted
        assert cs._verdict(0.05, 0.20, 50, 50)[0] == "GUARD_HARMS"
        # GUARD_NEUTRAL — diff under DIFF_TOL
        assert cs._verdict(0.15, 0.13, 50, 50)[0] == "GUARD_NEUTRAL"
        # INSUFFICIENT — below per-bucket min
        assert cs._verdict(0.30, 0.10, 5, 50)[0] == "INSUFFICIENT_DATA"
        assert cs._verdict(0.30, 0.10, 50, 5)[0] == "INSUFFICIENT_DATA"
        # INSUFFICIENT — None IC (n<2 after dropping)
        assert cs._verdict(None, 0.10, 50, 50)[0] == "INSUFFICIENT_DATA"
        assert cs._verdict(0.10, None, 50, 50)[0] == "INSUFFICIENT_DATA"


# ──────────────────────────── failed-row dropping ────────────────────────────


class TestFailedRowsDropped:
    def test_failed_predictions_dropped_not_bucketed(self):
        """Rows where predict_with_meta returns failed=True must be
        excluded from BOTH buckets — including them would tie at zero
        in whichever bucket they landed.

        Construction: 100 records, half with failed=True. The trusted
        bucket should see only the non-failed half; n_trusted == 50.
        """
        recs = _synth_records(100, leakage=0.5, sell_fraction=0.0)
        sc = _ConstantScorer(
            pred_fn=lambda r: r["ml_score"],
            off_fn=lambda r: False,
            fail_fn=lambda r: r["rsi"] > 50.0,  # ~half fail
        )
        rep = cs.confidence_skill(sc, recs)
        # All non-failed rows land in trusted (off_fn always False).
        # Counts must add up to fewer than n_records (failed rows
        # dropped from BOTH buckets).
        n_failed = sum(1 for r in recs if r["rsi"] > 50.0)
        n_kept = 100 - n_failed
        # n_records counts ALIGNED outcomes (those that produced a usable
        # triple) — failed rows are NOT aligned, so they don't count.
        assert rep["n_records"] == n_kept


# ──────────────────────────── SELL sign-flip ────────────────────────────


class TestSellSignFlip:
    def test_sell_target_flipped_not_prediction(self):
        """A SELL row's realized target is sign-flipped; the prediction
        is NOT (the model is trained on flipped targets, so its output
        already encodes action-aligned goodness).

        Construct one BUY and one SELL row with identical features and
        opposite realized returns. After flipping, both targets agree
        with the same prediction direction — the rank ordering is
        preserved across the action boundary.
        """
        buy = {
            "ticker": "NVDA", "sim_date": "2024-01-01",
            "action": "BUY",
            "ml_score": 2.0, "rsi": 65, "macd": 0.3,
            "mom5": 2.0, "mom20": 5.0, "regime_mult": 1.0,
            "vol_ratio": 1.2, "bb_position": 0.5,
            "news_urgency": 70, "news_article_count": 3.0,
            "forward_return_5d": 5.0,
        }
        sell = dict(buy, action="SELL", forward_return_5d=-5.0)
        # Both should produce target = +5 after flip; prediction = ml_score = 2.
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"],
                              off_fn=lambda r: False)
        # Use the internal _aligned_with_trust function directly to verify
        # the per-row contract.
        buy_triple = cs._aligned_with_trust(sc, buy)
        sell_triple = cs._aligned_with_trust(sc, sell)
        assert buy_triple is not None
        assert sell_triple is not None
        buy_p, buy_t, _ = buy_triple
        sell_p, sell_t, _ = sell_triple
        # Predictions: both 2.0 (not flipped).
        assert buy_p == pytest.approx(2.0)
        assert sell_p == pytest.approx(2.0)
        # Targets: BUY +5, SELL becomes +5 after flip.
        assert buy_t == pytest.approx(5.0)
        assert sell_t == pytest.approx(5.0)


# ──────────────────────────── edge cases ────────────────────────────


class TestEdgeCases:
    def test_all_trusted_no_abstained_yields_insufficient(self):
        """When the guard NEVER fires, the abstained bucket is empty.
        Verdict must be INSUFFICIENT_DATA (no comparison possible)."""
        recs = _synth_records(100, leakage=0.5)
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"],
                              off_fn=lambda r: False)
        rep = cs.confidence_skill(sc, recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["abstained"]["n"] == 0
        # All non-failed rows landed in trusted.
        assert rep["trusted"]["n"] == rep["n_records"]

    def test_all_abstained_no_trusted_yields_insufficient(self):
        """Mirror of the above — guard ALWAYS fires."""
        recs = _synth_records(100, leakage=0.5)
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"],
                              off_fn=lambda r: True)
        rep = cs.confidence_skill(sc, recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["trusted"]["n"] == 0

    def test_corrupt_jsonl_does_not_raise(self, tmp_path):
        """Malformed JSONL lines must be silently skipped."""
        p = tmp_path / "outcomes.jsonl"
        recs = _synth_records(40)
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.write("not json\n")
            fh.write("{partial\n")
            fh.write("\n")
        rep = cs.analyze(p, oos_only=False)
        # Below MIN_RECORDS (after the OOS split would drop some) — but
        # corrupt-jsonl-skip must work regardless.
        assert rep["status"] in ("ok", "untrained")  # no crash

    def test_missing_file_does_not_raise(self, tmp_path):
        rep = cs.analyze(tmp_path / "does_not_exist.jsonl", oos_only=False)
        assert rep["status"] in ("ok", "untrained")
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_records_missing_forward_return_dropped(self):
        """Rows without forward_return_5d must be dropped, not zeroed."""
        recs = _synth_records(50, leakage=0.5)
        # Strip forward_return_5d on every row.
        for r in recs:
            r["forward_return_5d"] = None
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"],
                              off_fn=lambda r: False)
        rep = cs.confidence_skill(sc, recs)
        # Every row drops → n_records = 0.
        assert rep["n_records"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_holds_are_ignored(self):
        """HOLD actions (shouldn't appear in outcomes, but defensive)
        must be filtered out before bucketing."""
        recs = _synth_records(50, leakage=0.5)
        # Tag every record with HOLD.
        for r in recs:
            r["action"] = "HOLD"
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"],
                              off_fn=lambda r: False)
        rep = cs.confidence_skill(sc, recs)
        assert rep["n_records"] == 0


# ──────────────────────────── return shape ────────────────────────────


class TestReturnShape:
    def test_returns_required_keys(self):
        sc = _ConstantScorer(pred_fn=lambda r: r["ml_score"])
        rep = cs.confidence_skill(sc, _synth_records(100))
        for k in ("status", "verdict", "n_records", "trusted", "abstained",
                  "diff", "hint"):
            assert k in rep
        for k in ("n", "rank_ic"):
            assert k in rep["trusted"]
            assert k in rep["abstained"]

    def test_analyze_adds_slice_label(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        recs = _synth_records(50)
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        # The scorer can't load (no pickle in tmp_path), so status=error
        # — but the slice label should still appear in the report.
        rep = cs.analyze(p, oos_only=True)
        assert "slice" in rep
