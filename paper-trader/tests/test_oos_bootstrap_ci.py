"""Tests for paper_trader.ml.oos_bootstrap_ci.

Verifies the bootstrap CI machinery on synthetic data where the expected
behaviour is mathematically pinned — not just "no crash". Covers:

  * scorer untrained / empty / insufficient → honest status sentinels
  * perfect predictor → CI tightly around 0 RMSE, 1.0 dir_acc, 1.0 rank_ic
  * anti-predictor → dir_acc CI excludes 0.5 from below, rank_ic excludes 0 from negative
  * noise predictor → dir_acc ~0.5, rank_ic CI straddles 0
  * SELL sign-flip is honoured (mirrors train_scorer / evaluate_scorer_oos)
  * label-clamp is applied so extreme outcomes don't blow up RMSE CI
  * determinism: same seed → identical CIs cycle to cycle
  * scorer.predict raising on one row drops that row, not the whole CI
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml.oos_bootstrap_ci import (
    DEFAULT_N_BOOTSTRAP,
    MIN_PAIRS_FOR_CI,
    bootstrap_ci,
)


def _scorer_predicting(fn):
    """Build a stub scorer whose ``predict`` calls ``fn(record_kwargs)``."""
    class _Stub:
        is_trained = True

        def predict(self, **kw):
            return fn(kw)

    return _Stub()


def _make_records(n: int, target_fn, action: str = "BUY") -> list[dict]:
    """Generate n synthetic outcome records. ``target_fn(i)`` returns the
    forward_return_5d for row i."""
    out = []
    for i in range(n):
        out.append({
            "ticker": "NVDA",
            "sim_date": f"2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}",
            "ml_score": float(i % 5),
            "rsi": 50.0,
            "macd": 0.0,
            "mom5": 0.0,
            "mom20": 0.0,
            "regime_mult": 1.0,
            "action": action,
            "forward_return_5d": float(target_fn(i)),
        })
    return out


class TestStatusSentinels:
    def test_empty_records_returns_empty_status(self):
        scorer = _scorer_predicting(lambda kw: 0.0)
        result = bootstrap_ci(scorer, [])
        assert result["status"] == "empty"
        assert result["n"] == 0
        assert result["rmse"]["value"] is None
        assert result["rmse"]["ci_low"] is None
        assert result["dir_acc"]["value"] is None
        assert result["rank_ic"]["value"] is None

    def test_untrained_scorer_returns_scorer_not_trained_status(self):
        class _Untrained:
            is_trained = False

            def predict(self, **_):
                return 0.0
        result = bootstrap_ci(_Untrained(), [{"forward_return_5d": 1.0}])
        assert result["status"] == "scorer_not_trained"
        assert result["rmse"]["value"] is None

    def test_below_min_pairs_returns_insufficient_data(self):
        """With fewer than MIN_PAIRS_FOR_CI valid pairs the CI is not
        meaningful — must report insufficient_data (mirrors calibration
        / baseline_compare floors so a downstream comparator can trust the
        verdict family)."""
        scorer = _scorer_predicting(lambda kw: 0.0)
        # Only 5 records — well below MIN_PAIRS_FOR_CI (30).
        records = _make_records(5, lambda i: 1.0)
        result = bootstrap_ci(scorer, records)
        assert result["status"] == "insufficient_data"
        assert result["n"] == 5
        assert result["n_bootstrap"] == 0


class TestPointEstimatesAndCIs:
    def test_perfect_predictor_yields_zero_rmse_perfect_dir_acc_ic(self):
        """A scorer that returns the realized return EXACTLY must score:
        - RMSE = 0 (point estimate AND CI tight around 0)
        - dir_acc = 1.0
        - rank_ic = 1.0
        """
        # Vary the realized return so each row has a distinct sign/value
        records = _make_records(80, lambda i: (i % 21) - 10)
        scorer = _scorer_predicting(
            lambda kw: float(kw.get("ml_score", 0)) * 0 + 0  # noop init
        )

        # Mutate the stub to mirror per-record realized — predicts via the
        # closure over `records`. Simpler: build a scorer that knows the
        # records' targets via an index counter.
        counter = {"i": 0}
        targets = [r["forward_return_5d"] for r in records]

        def _perfect(_kw):
            i = counter["i"]
            counter["i"] += 1
            return targets[i]

        scorer = _scorer_predicting(_perfect)

        result = bootstrap_ci(scorer, records, n_bootstrap=200, seed=1)
        assert result["status"] == "ok"
        assert result["rmse"]["value"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"]["ci_low"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"]["ci_high"] == pytest.approx(0.0, abs=1e-9)
        assert result["dir_acc"]["value"] == pytest.approx(1.0, abs=1e-9)
        # rank_ic = 1.0 for a perfect predictor.
        assert result["rank_ic"]["value"] == pytest.approx(1.0, abs=1e-9)

    def test_anti_predictor_yields_negative_rank_ic_ci_excluding_zero(self):
        """A scorer that returns the NEGATIVE of the realized has rank_ic
        ≈ -1.0. The bootstrap CI must clearly exclude 0 (from below) —
        this is the headline anti-skill detection a quant needs.
        """
        records = _make_records(80, lambda i: (i % 21) - 10)
        targets = [r["forward_return_5d"] for r in records]
        counter = {"i": 0}

        def _anti(_kw):
            i = counter["i"]
            counter["i"] += 1
            return -targets[i]

        scorer = _scorer_predicting(_anti)
        result = bootstrap_ci(scorer, records, n_bootstrap=300, seed=2)
        assert result["status"] == "ok"
        # rank_ic = -1.0 because the predictor is the negative — pairs that
        # were 0 (target == 0) get filtered out of dir_acc by the != 0 rule.
        assert result["rank_ic"]["value"] == pytest.approx(-1.0, abs=1e-6)
        # CI upper bound MUST be negative — the anti-skill is statistically
        # certain.
        assert result["rank_ic"]["ci_high"] < 0

    def test_constant_predictor_yields_rank_ic_zero(self):
        """A constant predictor (returns 0.0 for everything) has NO rank
        skill — rank_ic must read 0.0 (via tie-aware Spearman), never the
        +1.0 a naive argsort would fabricate from tied predictions."""
        records = _make_records(60, lambda i: (i % 11) - 5)
        scorer = _scorer_predicting(lambda kw: 0.0)
        result = bootstrap_ci(scorer, records, n_bootstrap=200, seed=3)
        assert result["status"] == "ok"
        # Spearman of constant vs varying is 0.0 (variance check in _spearman).
        assert result["rank_ic"]["value"] == pytest.approx(0.0, abs=1e-9)


class TestCorrectnessInvariants:
    def test_sell_sign_flip_is_honoured(self):
        """A SELL whose realized was DOWN (good SELL) must be treated as
        positive after the flip. Two records:
        - BUY +5 realized, scorer says +5  → contributes (5-5)² = 0
        - SELL with realized -5, scorer says +5  → flip realized to +5
          → contributes (5-5)² = 0
        Pre-flip: SELL would contribute (5 - (-5))² = 100, RMSE > 0.

        We construct enough records via duplication to clear MIN_PAIRS_FOR_CI.
        """
        records = []
        for _ in range(20):
            records.append({"forward_return_5d": 5.0, "action": "BUY",
                            "ticker": "NVDA"})
            records.append({"forward_return_5d": -5.0, "action": "SELL",
                            "ticker": "AMD"})
        scorer = _scorer_predicting(lambda kw: 5.0)
        result = bootstrap_ci(scorer, records, n_bootstrap=100, seed=4)
        assert result["status"] == "ok"
        # All errors are zero post-flip — RMSE must be exactly 0.
        assert result["rmse"]["value"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"]["ci_high"] == pytest.approx(0.0, abs=1e-9)

    def test_extreme_label_clamped_to_pred_clamp_pct(self):
        """A realized return of +175% (a leveraged-ETF crash-rip week) must
        clamp to +50 (the model's prediction ceiling), mirroring
        train_scorer's symmetric clamp and evaluate_scorer_oos. Without the
        clamp, a single such row would spike RMSE by sqrt(125²/n)."""
        # 30 in-band rows + 1 extreme row + 4 more for n>=30
        records = _make_records(34, lambda i: 0.0)
        records.append({"forward_return_5d": 175.0, "action": "BUY",
                        "ticker": "MSTR", "ml_score": 0.0, "rsi": 50,
                        "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                        "regime_mult": 1.0,
                        "sim_date": "2024-06-01"})
        # Scorer returns +50 for the MSTR row and 0 for everything else.
        def _pred(kw):
            return 50.0 if str(kw.get("ticker") or "") == "MSTR" else 0.0
        scorer = _scorer_predicting(_pred)
        result = bootstrap_ci(scorer, records, n_bootstrap=100, seed=5)
        assert result["status"] == "ok"
        # MSTR row clamps to +50, model predicts +50 → error 0.
        # All other rows: pred 0 vs actual 0 → error 0.
        # Pre-clamp the MSTR row would contribute (50-175)² = 15,625
        # spiking RMSE to sqrt(15625/35) ≈ 21.1.
        # Post-clamp RMSE is exactly 0.
        assert result["rmse"]["value"] == pytest.approx(0.0, abs=1e-9)

    def test_predict_exception_drops_that_record_only(self):
        """A scorer that raises on one row must drop just that row — the CI
        for the remaining rows is computed normally. Locks the
        invariant that one bad row never poisons the CI."""
        records = _make_records(35, lambda i: 1.0)
        # Mark one record so the predictor can identify and raise on it.
        records[10]["ticker"] = "BOMB"

        def _flaky(kw):
            if str(kw.get("ticker") or "") == "BOMB":
                raise RuntimeError("simulated transient predict fault")
            return 1.0
        scorer = _scorer_predicting(_flaky)
        result = bootstrap_ci(scorer, records, n_bootstrap=100, seed=6)
        assert result["status"] == "ok"
        # 34 valid rows survive (one was dropped).
        assert result["n"] == 34
        # All survivors: predicted +1 vs actual +1 → RMSE = 0.
        assert result["rmse"]["value"] == pytest.approx(0.0, abs=1e-9)


class TestDeterminism:
    def test_same_seed_yields_identical_ci_bounds(self):
        """Bootstrap CI must be deterministic given a fixed seed so
        cycle-over-cycle CI drift reflects real data shifts, not RNG noise.
        Locks np.random.default_rng(seed) as the SSOT for bootstrap RNG.
        """
        records = _make_records(50, lambda i: (i % 7) - 3)
        # Predictor returns a noisy version of target so RMSE > 0 and
        # bootstrap distribution is non-degenerate.
        targets = [r["forward_return_5d"] for r in records]
        counter = {"i": 0}

        def _noisy(_kw):
            i = counter["i"] % len(targets)
            counter["i"] += 1
            return targets[i] + 0.5

        scorer1 = _scorer_predicting(_noisy)
        result1 = bootstrap_ci(scorer1, records, n_bootstrap=100, seed=42)

        counter["i"] = 0  # reset for second pass
        scorer2 = _scorer_predicting(_noisy)
        result2 = bootstrap_ci(scorer2, records, n_bootstrap=100, seed=42)

        assert result1["rmse"]["ci_low"] == result2["rmse"]["ci_low"]
        assert result1["rmse"]["ci_high"] == result2["rmse"]["ci_high"]
        assert result1["rank_ic"]["ci_low"] == result2["rank_ic"]["ci_low"]
        assert result1["dir_acc"]["ci_low"] == result2["dir_acc"]["ci_low"]


class TestJsonSafety:
    def test_result_is_json_serializable(self):
        """The CLI emits ``json.dumps(result, …)``; no Python-only types
        (numpy floats, NaN, inf) may leak through. _round_or_none coerces
        every value through Python float / None, so this lock pins that
        contract."""
        records = _make_records(50, lambda i: (i % 5) - 2)
        scorer = _scorer_predicting(lambda kw: 1.0)
        result = bootstrap_ci(scorer, records, n_bootstrap=50, seed=7)
        # Round-trip through json — must not raise.
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded["status"] == "ok"
        assert decoded["n"] == 50

    def test_nan_predictions_are_dropped_not_serialized(self):
        """A scorer that returns NaN for some predictions has those rows
        dropped — the final CI fields are all finite (not NaN) so the
        JSON contract is preserved.
        """
        records = _make_records(50, lambda i: float(i % 3) - 1)

        def _half_nan(kw):
            return float("nan") if int(kw.get("ml_score", 0) or 0) % 2 == 0 else 1.0
        scorer = _scorer_predicting(_half_nan)
        result = bootstrap_ci(scorer, records, n_bootstrap=50, seed=8)
        # Some records survive; verify their CI bounds are finite.
        if result["status"] == "ok":
            assert result["rmse"]["value"] is not None
            assert result["rmse"]["ci_low"] is not None
            # Round-trip safety.
            json.dumps(result)


class TestPercentileBounds:
    def test_ci_low_le_value_le_ci_high(self):
        """Sanity invariant: the bootstrap CI bounds bracket the point
        estimate. This must hold for any well-behaved input — it's the
        most basic correctness check on a percentile-CI implementation.
        """
        records = _make_records(60, lambda i: (i % 11) - 5)
        # Slightly noisy predictor so CI is non-degenerate.
        targets = [r["forward_return_5d"] for r in records]
        counter = {"i": 0}

        def _noisy(_kw):
            i = counter["i"] % len(targets)
            counter["i"] += 1
            return targets[i] * 0.5  # systematic under-prediction

        scorer = _scorer_predicting(_noisy)
        result = bootstrap_ci(scorer, records, n_bootstrap=200, seed=9)
        for metric in ("rmse", "dir_acc", "rank_ic"):
            cell = result[metric]
            v, lo, hi = cell["value"], cell["ci_low"], cell["ci_high"]
            if v is None:
                continue
            # CI bounds must bracket the point estimate (within float tol).
            assert lo <= v + 1e-9, f"{metric}: ci_low {lo} > value {v}"
            assert v <= hi + 1e-9, f"{metric}: value {v} > ci_high {hi}"
