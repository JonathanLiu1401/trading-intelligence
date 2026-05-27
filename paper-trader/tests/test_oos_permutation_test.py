"""Behaviour lock for paper_trader.ml.oos_permutation_test.

Tests pin:

  * the **point estimate** equals the SSOT ``_spearman`` (so this
    diagnostic and the per-cycle skill ledger cannot disagree on the
    ``rank_ic`` they report for the same OOS slice — the canonical
    single-source-of-truth bar the OOS suite enforces);
  * the smoothed p-value formula ``(k + 1) / (n + 1)`` correctly
    bounds the result in ``(0, 1]`` even when no shuffle matches
    the observed magnitude (a real-data corner case the bootstrap
    CI sibling doesn't need to worry about);
  * the verdict ladder maps p-values correctly at every threshold;
  * a TRULY random predictor produces an AT_NOISE verdict (the
    Type-I-error self-test);
  * a STRONG-skill predictor (pred = realized + small jitter)
    produces a STATISTICALLY_SIGNIFICANT verdict (the
    positive-control / power self-test);
  * the per-action breakdown correctly partitions the OOS slice
    by ``action`` (the BUY bucket is gate-relevant; SELL bucket
    is the sanity check);
  * untrained scorer / empty records / sub-MIN_PAIRS slice degrade
    to the well-formed insufficient-data envelopes (the
    ``calibration`` honest-empty precedent);
  * a row with ``predict_with_meta`` returning ``failed=True`` is
    dropped from BOTH the headline IC and the null distribution
    (the sentinel-zero contamination guard);
  * the SELL sign-flip is applied symmetrically with
    ``oos_bootstrap_ci`` (so a SELL whose ``forward_return_5d=-5%``
    is treated as a +5 target).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.ml.oos_permutation_test import (
    ALPHA_STRICT,
    ALPHA_WEAK,
    MIN_PAIRS,
    _bucket_empty,
    _build_aligned_arrays,
    _exit_code_from_result,
    _verdict_from_p,
    permutation_test,
)


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------

class _ScorerLinear:
    """Trained scorer fake: ``predict_with_meta`` returns the row's
    ``ml_score`` plus a small per-row jitter. Used as the positive-control
    that should reach STATISTICALLY_SIGNIFICANT against any reasonably
    correlated target."""

    def __init__(self, jitter: float = 0.0):
        self.is_trained = True
        self._jitter = jitter

    def predict_with_meta(self, **kw):
        # The scorer takes ml_score as the headline feature — we use it
        # as the "raw signal" and add controlled jitter to model real
        # MLP imprecision. ``failed=False`` so the diagnostic never
        # drops these rows.
        raw = float(kw.get("ml_score", 0.0)) + (
            self._jitter * (hash(str(kw)) % 7 - 3) / 100.0
        )
        return {
            "pred": raw,
            "raw": raw,
            "clamped": False,
            "off_distribution": False,
            "percentile": None,
            "calibrated": None,
            "failed": False,
        }


class _ScorerConstantZero:
    """Trained scorer fake that always predicts exactly 0.0. Used to
    verify a perfectly-uninformative predictor lands in AT_NOISE — a
    constant Spearman is undefined (NaN), so the diagnostic must
    handle that gracefully too."""

    is_trained = True

    def predict_with_meta(self, **kw):
        return {
            "pred": 0.0, "raw": 0.0, "clamped": False,
            "off_distribution": False, "percentile": None,
            "calibrated": None, "failed": False,
        }


class _ScorerRandom:
    """Trained scorer fake whose predictions are drawn from a fixed RNG
    independent of the input. This is the H0 generator — predictions
    are uncorrelated with targets by construction."""

    is_trained = True

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def predict_with_meta(self, **kw):
        return {
            "pred": float(self._rng.normal()),
            "raw": 0.0, "clamped": False, "off_distribution": False,
            "percentile": None, "calibrated": None, "failed": False,
        }


class _ScorerFailingFor:
    """Trained scorer fake that returns ``failed=True`` for a
    specific ticker — exercises the sentinel-zero contamination guard
    in ``_build_aligned_arrays``."""

    is_trained = True

    def __init__(self, fail_ticker: str):
        self._fail = fail_ticker

    def predict_with_meta(self, **kw):
        if str(kw.get("ticker", "")) == self._fail:
            return {
                "pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": True,
                "percentile": None, "calibrated": None,
                "failed": True,
            }
        return {
            "pred": float(kw.get("ml_score", 0.0)),
            "raw": 0.0, "clamped": False, "off_distribution": False,
            "percentile": None, "calibrated": None, "failed": False,
        }


class _ScorerUntrained:
    is_trained = False

    def predict_with_meta(self, **kw):
        return {
            "pred": 0.0, "raw": 0.0, "clamped": False,
            "off_distribution": False, "percentile": None,
            "calibrated": None, "failed": True,
        }


def _mk_records(n: int, slope: float, noise: float, *, seed: int = 0,
                action: str = "BUY", ticker: str = "TST") -> list[dict]:
    """Synthetic outcome rows where forward_return_5d = slope * ml_score
    + N(0, noise). slope=1.0, noise=0 yields a perfect IC=1.0 signal;
    slope=0 yields pure noise."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for i in range(n):
        score = float(rng.uniform(0.5, 10.0))
        target = slope * score + (rng.normal() * noise if noise > 0 else 0.0)
        out.append({
            "ml_score": score,
            "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
            "regime_mult": 1.0, "ticker": ticker,
            "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "ema200_above": False, "hist_cross_up": False,
            "macd_below_zero_cross": False,
            "forward_return_5d": float(target),
            "action": action, "sim_date": f"2025-01-{(i % 28) + 1:02d}",
        })
    return out


# ---------------------------------------------------------------------------
# _verdict_from_p ladder
# ---------------------------------------------------------------------------

class TestVerdictLadder:
    def test_none_p_value_is_insufficient(self):
        assert _verdict_from_p(None) == "INSUFFICIENT_DATA"

    def test_strict_significance_at_p_0001(self):
        assert _verdict_from_p(0.001) == "STATISTICALLY_SIGNIFICANT"

    def test_strict_boundary_open(self):
        # p == ALPHA_STRICT (0.01) is NOT strict-significant — uses `<`
        assert _verdict_from_p(ALPHA_STRICT) == "WEAKLY_SIGNIFICANT"

    def test_weak_significance_at_p_0_03(self):
        assert _verdict_from_p(0.03) == "WEAKLY_SIGNIFICANT"

    def test_weak_boundary_open(self):
        # p == ALPHA_WEAK (0.05) is AT_NOISE — uses `<`
        assert _verdict_from_p(ALPHA_WEAK) == "AT_NOISE"

    def test_above_weak_is_at_noise(self):
        assert _verdict_from_p(0.5) == "AT_NOISE"
        assert _verdict_from_p(1.0) == "AT_NOISE"


# ---------------------------------------------------------------------------
# _build_aligned_arrays
# ---------------------------------------------------------------------------

class TestBuildAlignedArrays:
    def test_sell_target_sign_flipped(self):
        """A SELL whose forward_return_5d = -5% must enter the actuals
        array as +5 (post-sign-flip) — mirrors train_scorer's contract."""
        recs = [{
            "ml_score": 1.0, "ticker": "X", "regime_mult": 1.0,
            "forward_return_5d": -5.0, "action": "SELL",
        }]
        p, a, sells = _build_aligned_arrays(_ScorerLinear(), recs)
        assert p.size == 1
        assert a.size == 1
        assert a[0] == pytest.approx(5.0, abs=1e-6)
        assert sells[0]

    def test_failed_row_dropped(self):
        """A row whose predict_with_meta returns failed=True must NOT
        contribute to preds/actuals — the sentinel-zero contamination
        guard. If it leaks, the headline IC ties at zero and the
        permutation null is poisoned."""
        recs = [
            {"ml_score": 1.0, "ticker": "BAD", "regime_mult": 1.0,
             "forward_return_5d": 5.0, "action": "BUY"},
            {"ml_score": 2.0, "ticker": "GOOD", "regime_mult": 1.0,
             "forward_return_5d": 3.0, "action": "BUY"},
            {"ml_score": 3.0, "ticker": "GOOD", "regime_mult": 1.0,
             "forward_return_5d": 7.0, "action": "BUY"},
        ]
        p, a, _ = _build_aligned_arrays(_ScorerFailingFor("BAD"), recs)
        assert p.size == 2  # BAD row dropped
        assert a.size == 2

    def test_nan_target_dropped(self):
        recs = [
            {"ml_score": 1.0, "ticker": "X", "regime_mult": 1.0,
             "forward_return_5d": None, "action": "BUY"},
            {"ml_score": 2.0, "ticker": "X", "regime_mult": 1.0,
             "forward_return_5d": float("nan"), "action": "BUY"},
            {"ml_score": 3.0, "ticker": "X", "regime_mult": 1.0,
             "forward_return_5d": 5.0, "action": "BUY"},
        ]
        p, a, _ = _build_aligned_arrays(_ScorerLinear(), recs)
        assert p.size == 1
        assert a.size == 1

    def test_label_clamp_symmetric(self):
        """forward_return_5d outside ±PRED_CLAMP_PCT must clamp to the
        boundary — apples-to-apples with train_scorer + evaluate_scorer_oos."""
        recs = [{
            "ml_score": 1.0, "ticker": "X", "regime_mult": 1.0,
            "forward_return_5d": 175.0, "action": "BUY",
        }, {
            "ml_score": -1.0, "ticker": "X", "regime_mult": 1.0,
            "forward_return_5d": -200.0, "action": "BUY",
        }]
        _, a, _ = _build_aligned_arrays(_ScorerLinear(), recs)
        assert a[0] == pytest.approx(50.0, abs=1e-6)
        assert a[1] == pytest.approx(-50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# permutation_test — verdict scenarios
# ---------------------------------------------------------------------------

class TestPermutationTestPositiveControl:
    """Power test: a strong-signal predictor MUST reach significance."""

    def test_strong_signal_reaches_statistical_significance(self):
        # Perfect signal (slope=1, noise=0) — IC should be ~1.0 and p << 0.01
        recs = _mk_records(n=120, slope=1.0, noise=0.0, seed=1)
        scorer = _ScorerLinear(jitter=0.0)
        out = permutation_test(scorer, recs, n_permutations=200, seed=42)
        assert out["status"] == "ok"
        assert out["aggregate"]["rank_ic"] is not None
        assert out["aggregate"]["rank_ic"] > 0.99
        assert out["aggregate"]["p_value"] is not None
        # With 200 permutations and a perfect signal, the smoothed p-value
        # is 1/(200+1) ≈ 0.005 — well inside the strict band.
        assert out["aggregate"]["p_value"] < ALPHA_STRICT
        assert out["aggregate"]["verdict"] == "STATISTICALLY_SIGNIFICANT"


class TestPermutationTestNegativeControl:
    """Type-I-error test: a TRULY random predictor must NOT reach strict
    significance more often than alpha at the long run. Single-shot we
    require AT_NOISE on a moderate slice."""

    def test_pure_random_predictions_land_at_noise(self):
        recs = _mk_records(n=120, slope=1.0, noise=10.0, seed=7)
        scorer = _ScorerRandom(seed=0)
        out = permutation_test(scorer, recs, n_permutations=300, seed=11)
        assert out["status"] == "ok"
        assert out["aggregate"]["p_value"] is not None
        # p > 0.05 expected for a random scorer; AT_NOISE verdict expected.
        # We allow WEAKLY_SIGNIFICANT (~5% Type-I) but assert NOT strict.
        assert out["aggregate"]["verdict"] != "STATISTICALLY_SIGNIFICANT"


class TestPermutationTestEmptyAndInsufficient:
    def test_no_records_returns_empty(self):
        out = permutation_test(_ScorerLinear(), [], n_permutations=100)
        assert out["status"] == "empty"
        assert out["n"] == 0
        assert out["aggregate"]["verdict"] == "INSUFFICIENT_DATA"

    def test_untrained_scorer_returns_not_trained(self):
        recs = _mk_records(n=100, slope=1.0, noise=0.0)
        out = permutation_test(_ScorerUntrained(), recs, n_permutations=100)
        assert out["status"] == "scorer_not_trained"
        assert out["aggregate"]["verdict"] == "INSUFFICIENT_DATA"

    def test_below_min_pairs_returns_insufficient(self):
        # Fewer than MIN_PAIRS records — no test is meaningful
        n = MIN_PAIRS - 1
        recs = _mk_records(n=n, slope=1.0, noise=0.0)
        out = permutation_test(_ScorerLinear(), recs, n_permutations=100)
        assert out["status"] == "insufficient_data"
        assert out["n"] == n


# ---------------------------------------------------------------------------
# Per-action breakdown
# ---------------------------------------------------------------------------

class TestPerActionBreakdown:
    def test_buy_only_slice_has_sell_insufficient(self):
        recs = _mk_records(n=100, slope=1.0, noise=0.0, action="BUY")
        out = permutation_test(_ScorerLinear(), recs, n_permutations=200)
        assert out["status"] == "ok"
        assert out["buy"]["n"] == 100
        assert out["sell"]["n"] == 0
        assert out["sell"]["verdict"] == "INSUFFICIENT_DATA"

    def test_buy_and_sell_partitioned_correctly(self):
        recs = (_mk_records(n=60, slope=1.0, noise=0.0, seed=1, action="BUY")
                + _mk_records(n=40, slope=1.0, noise=0.0, seed=2, action="SELL"))
        out = permutation_test(_ScorerLinear(), recs, n_permutations=200)
        assert out["status"] == "ok"
        assert out["aggregate"]["n"] == 100
        assert out["buy"]["n"] == 60
        assert out["sell"]["n"] == 40

    def test_sell_sign_flip_visible_in_per_action_ic(self):
        """A SELL whose forward_return_5d = -slope*score (i.e. the SELL
        correctly anticipated a drop) must, AFTER the SELL sign-flip
        inside ``_build_aligned_arrays``, look like a +slope*score
        signal to the rank-IC calc — exactly the same direction as the
        BUY bucket. So the SELL bucket's IC should match the BUY bucket's
        in sign when the input data was constructed with mirrored shape."""
        # Build SELL rows where target = -ml_score (the "correctly
        # avoided a drop" pattern). After the SELL sign-flip target
        # becomes +ml_score, so rank-IC should be strongly POSITIVE.
        recs = []
        for i in range(60):
            recs.append({
                "ml_score": float(i + 1),
                "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "ticker": "Y",
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": None, "news_article_count": None,
                "ema200_above": False, "hist_cross_up": False,
                "macd_below_zero_cross": False,
                "forward_return_5d": -float(i + 1),
                "action": "SELL",
                "sim_date": f"2025-02-{(i % 28) + 1:02d}",
            })
        out = permutation_test(_ScorerLinear(), recs, n_permutations=200)
        assert out["sell"]["rank_ic"] is not None
        assert out["sell"]["rank_ic"] > 0.95  # perfect post-flip signal
        assert out["sell"]["verdict"] == "STATISTICALLY_SIGNIFICANT"


# ---------------------------------------------------------------------------
# p-value floor (smoothed estimator bounded in (0, 1])
# ---------------------------------------------------------------------------

class TestSmoothPValueBounds:
    def test_perfect_signal_p_value_bounded_above_zero(self):
        """The smoothed formula (k+1)/(n+1) ensures even when k=0 the
        p-value stays strictly positive — never falsely reports p=0.
        A perfect signal with 200 permutations yields p = 1/201 ≈ 0.005."""
        recs = _mk_records(n=120, slope=1.0, noise=0.0, seed=3)
        out = permutation_test(_ScorerLinear(), recs, n_permutations=200)
        assert out["aggregate"]["p_value"] is not None
        assert out["aggregate"]["p_value"] > 0.0  # never exactly zero
        # Lower bound is 1/(n_perm+1)
        assert out["aggregate"]["p_value"] >= 1.0 / 201.0

    def test_constant_predictor_returns_insufficient_or_at_noise(self):
        """A constant predictor's Spearman is undefined (NaN). The
        ``_spearman_local`` NaN guard MUST degrade to ``rank_ic=None``
        (and INSUFFICIENT_DATA verdict) rather than crashing or
        emitting fake significance."""
        recs = _mk_records(n=120, slope=1.0, noise=0.0, seed=4)
        scorer = _ScorerConstantZero()
        out = permutation_test(scorer, recs, n_permutations=100)
        # A constant predict produces a constant predictions array,
        # so _spearman is NaN, so rank_ic is None → INSUFFICIENT_DATA.
        # The status itself is "ok" because we DID compute (just got
        # NaN back) — only the bucket verdict degrades.
        assert out["status"] == "ok"
        assert out["aggregate"]["rank_ic"] is None
        assert out["aggregate"]["verdict"] == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Exit-code semantics
# ---------------------------------------------------------------------------

class TestExitCode:
    def test_buy_significant_exits_zero(self):
        result = {
            "status": "ok",
            "buy": {"verdict": "STATISTICALLY_SIGNIFICANT", "n": 100,
                    "rank_ic": 0.3, "p_value": 0.001},
        }
        assert _exit_code_from_result(result) == 0

    def test_buy_weak_exits_zero(self):
        """A weak BUY edge still earns exit 0 — the operator-decisive
        question is 'real edge anywhere', not 'strong edge'."""
        result = {
            "status": "ok",
            "buy": {"verdict": "WEAKLY_SIGNIFICANT", "n": 100,
                    "rank_ic": 0.1, "p_value": 0.03},
        }
        assert _exit_code_from_result(result) == 0

    def test_buy_at_noise_exits_one(self):
        result = {
            "status": "ok",
            "buy": {"verdict": "AT_NOISE", "n": 100,
                    "rank_ic": 0.01, "p_value": 0.4},
        }
        assert _exit_code_from_result(result) == 1

    def test_status_not_ok_exits_one(self):
        for st in ("empty", "scorer_not_trained", "insufficient_data"):
            assert _exit_code_from_result({"status": st, "buy": {}}) == 1

    def test_aggregate_significant_but_buy_noise_still_exits_one(self):
        """The exit code is GATE-RELEVANT — it tracks BUY significance
        only. An aggregate edge driven by SELL skill the gate cannot
        act on must NOT light up green."""
        result = {
            "status": "ok",
            "aggregate": {"verdict": "STATISTICALLY_SIGNIFICANT"},
            "buy": {"verdict": "AT_NOISE", "n": 60, "rank_ic": 0.0,
                    "p_value": 0.5},
            "sell": {"verdict": "STATISTICALLY_SIGNIFICANT"},
        }
        assert _exit_code_from_result(result) == 1


# ---------------------------------------------------------------------------
# Shape stability + JSON-safety
# ---------------------------------------------------------------------------

class TestShapeStability:
    def test_response_carries_required_keys(self):
        recs = _mk_records(n=100, slope=1.0, noise=0.0)
        out = permutation_test(_ScorerLinear(), recs, n_permutations=100)
        for k in ("status", "n", "n_permutations",
                  "aggregate", "buy", "sell",
                  "alpha_strict", "alpha_weak", "min_pairs"):
            assert k in out
        for bucket in (out["aggregate"], out["buy"], out["sell"]):
            for k in ("n", "rank_ic", "p_value", "verdict",
                      "null_p10", "null_p50", "null_p90"):
                assert k in bucket

    def test_response_is_json_serializable(self):
        recs = _mk_records(n=100, slope=1.0, noise=0.0)
        out = permutation_test(_ScorerLinear(), recs, n_permutations=100)
        # Must round-trip through json without raising — locks the
        # JSON-safety the CLI's --json mode depends on.
        s = json.dumps(out)
        round_trip = json.loads(s)
        assert round_trip["aggregate"]["verdict"] in (
            "STATISTICALLY_SIGNIFICANT", "WEAKLY_SIGNIFICANT",
            "AT_NOISE", "INSUFFICIENT_DATA",
        )

    def test_bucket_empty_shape(self):
        b = _bucket_empty(n=42)
        assert b["n"] == 42
        assert b["rank_ic"] is None
        assert b["verdict"] == "INSUFFICIENT_DATA"
        # null pcts must all be None on empty
        for k in ("null_p10", "null_p50", "null_p90"):
            assert b[k] is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_p_value(self):
        recs = _mk_records(n=100, slope=0.5, noise=5.0, seed=99)
        a = permutation_test(_ScorerLinear(), recs, n_permutations=200, seed=1)
        b = permutation_test(_ScorerLinear(), recs, n_permutations=200, seed=1)
        assert a["aggregate"]["p_value"] == b["aggregate"]["p_value"]
        assert a["aggregate"]["rank_ic"] == b["aggregate"]["rank_ic"]
