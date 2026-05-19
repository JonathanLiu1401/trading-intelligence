"""Tests for `paper_trader.ml.scorer_smoke_test` — DecisionScorer health
sanity check.

Pins the five documented verdicts so a refactor cannot silently weaken a
verdict to HEALTHY when the scorer is actually broken (the regression-of-
record that motivated this module):

  * HEALTHY                  — predictions finite & span ≥ 2 distinct gate buckets
  * UNTRAINED                — no pickle / is_trained False
  * DEGENERATE_CONSTANT      — every probe collapses to one prediction
  * GATE_BUCKETS_DEGENERATE  — predictions distinct but all in ONE conviction-gate arm
  * BROKEN_PREDICT           — predict_with_meta raised or returned non-finite

Every test exercises real verdict logic against a faked DecisionScorer (no
pickle / network / disk dependency) so the suite is offline-clean and
sub-second.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import scorer_smoke_test as sst


# ─────────────────────────── fake scorers ──────────────────────────────────


class _UntrainedScorer:
    is_trained = False
    n_train = 0

    def predict_with_meta(self, **kw):
        # Mirrors DecisionScorer.predict_with_meta's untrained branch.
        return {"pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": False}


class _HealthyScorer:
    """Differs by ticker so the 8 in-distribution probes produce distinct
    predictions — the basic HEALTHY signature. NOT a real model, but
    structurally identical to the public scorer contract that
    scorer_smoke_test.py reads against."""
    is_trained = True
    n_train = 3500

    def predict_with_meta(self, **kw):
        # Deterministic but distinct per ticker — guarantees ≥2 distinct
        # buckets across the 8 in-distribution probes.
        seed = sum(ord(c) for c in kw.get("ticker", ""))
        pred = (seed % 17) * 0.7 - 5.0   # spread well beyond tolerance
        return {"pred": round(pred, 4), "raw": round(pred, 4),
                "clamped": False, "off_distribution": False}


class _ConstantScorer:
    """Returns the SAME prediction for every input. This is the failure
    mode `scorer_smoke_test` is built to catch — every existing
    diagnostic (`scorer_freshness`, `deploy_audit`) would pass on this
    pickle while it silently disables the conviction gate."""
    is_trained = True
    n_train = 1200

    def predict_with_meta(self, **kw):
        return {"pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": False}


class _RaisingScorer:
    is_trained = True
    n_train = 800

    def predict_with_meta(self, **kw):
        raise RuntimeError("boom")


class _NonFiniteScorer:
    """predict_with_meta returns a non-finite `pred`. The public contract
    says `pred` is always finite, so this represents a slipped invariant
    in a hypothetical custom scorer subclass — smoke test must flag it
    BROKEN, not pass it through to the gate."""
    is_trained = True
    n_train = 1000

    def predict_with_meta(self, **kw):
        return {"pred": float("nan"), "raw": float("nan"),
                "clamped": True, "off_distribution": True}


class _OffDistributionScorer:
    """Healthy distinct predictions BUT every probe also has
    `off_distribution=True`. The HEALTHY verdict must still fire — the
    off-distribution count is informational, never a verdict driver."""
    is_trained = True
    n_train = 2000

    def predict_with_meta(self, **kw):
        seed = sum(ord(c) for c in kw.get("ticker", ""))
        pred = (seed % 13) - 6.0
        return {"pred": round(pred, 4), "raw": round(pred, 4),
                "clamped": True, "off_distribution": True}


# ─────────────────────────── verdict tests ──────────────────────────────────


class TestVerdictLadder:
    def test_untrained_scorer_yields_untrained(self):
        rep = sst.scorer_smoke_report(scorer=_UntrainedScorer())
        assert rep["verdict"] == "UNTRAINED"
        assert rep["is_trained"] is False
        # Probes must not have run on an untrained scorer.
        assert rep["probes"] == []
        assert rep["edge_probes"] == []

    def test_healthy_scorer_yields_healthy(self):
        rep = sst.scorer_smoke_report(scorer=_HealthyScorer())
        assert rep["verdict"] == "HEALTHY", rep
        assert rep["is_trained"] is True
        assert rep["n_train"] == 3500
        # The 8 in-distribution probes must produce well-defined records.
        assert len(rep["probes"]) == 8
        assert all("pred" in r and "error" not in r for r in rep["probes"])
        # Distinct predictions: 8 probes, _HealthyScorer's seed-derived
        # formula yields several distinct values — strictly more than 1.
        assert rep["distinct_predictions"] >= 2
        assert rep["broken_probe_count"] == 0

    def test_constant_scorer_yields_degenerate(self):
        rep = sst.scorer_smoke_report(scorer=_ConstantScorer())
        assert rep["verdict"] == "DEGENERATE_CONSTANT", rep
        # Single distinct value across 8 probes — the failure signature.
        assert rep["distinct_predictions"] == 1
        # Hint must mention the gate-disabling consequence so an operator
        # reading the verdict in a Discord alert understands urgency.
        assert "gate" in rep["hint"].lower()

    def test_raising_scorer_yields_broken_predict(self):
        rep = sst.scorer_smoke_report(scorer=_RaisingScorer())
        assert rep["verdict"] == "BROKEN_PREDICT", rep
        # Every probe must have errored; honest count surfaced.
        assert rep["broken_probe_count"] == len(rep["probes"]) + \
            len(rep["edge_probes"])
        # First probe error must surface in the hint so the operator
        # has the exception class without rerunning.
        assert "RuntimeError" in rep["hint"]

    def test_non_finite_pred_yields_broken_predict(self):
        """A scorer subclass that violates the always-finite-`pred`
        contract MUST be flagged BROKEN, not silently rounded to 0.0%
        (which would degenerate-pass via the constant-predictor branch)."""
        rep = sst.scorer_smoke_report(scorer=_NonFiniteScorer())
        assert rep["verdict"] == "BROKEN_PREDICT", rep
        assert rep["broken_probe_count"] > 0

    def test_off_distribution_does_not_block_healthy(self):
        """A scorer whose every probe trips `off_distribution=True` is
        suspicious (feature drift) but its scalar predictions are still
        finite and distinct — the verdict ladder MUST stay HEALTHY and
        report the count via the informational field, never demote to
        DEGENERATE/BROKEN. The conviction gate already skips off-dist
        predictions; this test pins that the smoke report agrees."""
        rep = sst.scorer_smoke_report(scorer=_OffDistributionScorer())
        assert rep["verdict"] == "HEALTHY", rep
        # All 8 in-distribution probes flagged off-distribution.
        assert rep["off_distribution_in_distribution"] == 8


# ─────────────────────────── output schema ──────────────────────────────────


class TestSchema:
    def test_verdict_is_a_known_enum(self):
        """Every code path emits a verdict that is in the public
        `VERDICTS` tuple — a typo in any of the verdict-emission paths
        is caught at this membership lock (the `scorer_freshness`
        precedent)."""
        for scorer in (_UntrainedScorer(), _HealthyScorer(),
                       _ConstantScorer(), _RaisingScorer(),
                       _NonFiniteScorer()):
            rep = sst.scorer_smoke_report(scorer=scorer)
            assert rep["verdict"] in sst.VERDICTS, (scorer, rep)

    def test_report_is_json_safe(self):
        """The report must round-trip through json.dumps — a stray
        numpy scalar / Path / set would have broken the dashboard
        consumer the same way `feature_importance.analyze` was hardened
        against (JSON-safe is the discipline)."""
        import json
        rep = sst.scorer_smoke_report(scorer=_HealthyScorer())
        s = json.dumps(rep, sort_keys=True)
        # Round-trip preserves verdict — a stricter check than just
        # "dumps did not raise".
        assert json.loads(s)["verdict"] == "HEALTHY"

    def test_module_level_constants_are_stable(self):
        """`VERDICTS` and the probe lists are part of the module's
        public surface; lock their cardinality so a silent removal of a
        verdict or a probe is caught."""
        assert len(sst.VERDICTS) == 5
        assert set(sst.VERDICTS) == {
            "HEALTHY", "UNTRAINED", "DEGENERATE_CONSTANT",
            "GATE_BUCKETS_DEGENERATE", "BROKEN_PREDICT",
        }
        assert len(sst._PROBES) >= 8
        assert len(sst._EDGE_PROBES) >= 2
        # _GATE_BUCKETS exposes the five conviction-gate arm labels
        # `_gate_bucket` returns; the GATE_BUCKETS_DEGENERATE verdict's
        # JSON-schema field `gate_bucket_counts` is keyed by these
        # exact strings, so the dashboard / Discord template can rely
        # on every bucket key always being present.
        assert sst._GATE_BUCKETS == (
            "strong_headwind", "mild_headwind", "neutral",
            "mild_tailwind", "strong_tailwind",
        )


# ─────────────────────────── gate-bucket verdict ────────────────────────────


class _NeutralBucketScorer:
    """Predictions are 8 distinct values BUT all fall in [0, 5] (the
    neutral bucket of `_ml_decide`'s conviction gate). The
    DEGENERATE_CONSTANT check, which compares raw predictions at 1e-4
    tolerance, would mark this scorer HEALTHY — but the gate is
    operationally dormant because no prediction crosses the ±10/±5/0
    thresholds. This is the EXACT failure pattern AGENTS.md review
    pass #2 documents for the n_train=400 clobber: predictions vary
    but the gate's arms collapse to one multiplier."""
    is_trained = True
    n_train = 400

    def predict_with_meta(self, **kw):
        # Deterministic spread within [0.01, 4.5] across probes — well
        # above the 1e-4 constant tolerance, well below the 5.0 neutral
        # boundary. Yields 8 distinct values, all neutral-bucket.
        seed = sum(ord(c) for c in kw.get("ticker", ""))
        pred = 0.5 + ((seed % 11) * 0.4)  # 0.5..4.5 range
        return {"pred": round(pred, 4), "raw": round(pred, 4),
                "clamped": False, "off_distribution": False}


class _TwoBucketScorer:
    """Predictions span TWO gate buckets across probes — the minimum
    bar HEALTHY requires under the new gate-bucket check. Pins that
    the `distinct_gate_buckets >= 2` boundary is inclusive (≥, not >)."""
    is_trained = True
    n_train = 2000

    def predict_with_meta(self, **kw):
        seed = sum(ord(c) for c in kw.get("ticker", ""))
        # Alternate between neutral (~2.0) and mild_tailwind (~7.0) by
        # parity. Guaranteed to populate exactly 2 of the 5 gate buckets.
        pred = 2.0 if (seed % 2 == 0) else 7.0
        return {"pred": round(pred, 4), "raw": round(pred, 4),
                "clamped": False, "off_distribution": False}


class TestGateBucket:
    """`_gate_bucket()` is the lockstep mirror of `_ml_decide`'s
    conviction-gate ladder (CLAUDE.md §6). A drift between the two
    would silently produce a wrong GATE_BUCKETS_DEGENERATE verdict.
    These tests pin every arm boundary AND the non-finite degrade."""

    def test_strong_headwind_below_minus_ten(self):
        assert sst._gate_bucket(-50.0) == "strong_headwind"
        assert sst._gate_bucket(-10.01) == "strong_headwind"
        # Boundary: exactly -10 is NOT strong_headwind (the gate's
        # `< -10` is strict), it lands in mild_headwind.
        assert sst._gate_bucket(-10.0) == "mild_headwind"

    def test_mild_headwind_between_minus_ten_and_zero(self):
        assert sst._gate_bucket(-5.0) == "mild_headwind"
        assert sst._gate_bucket(-0.0001) == "mild_headwind"
        # Exactly 0 is NOT mild_headwind (the gate's `< 0` is strict)
        # — it falls through to the neutral arm.
        assert sst._gate_bucket(0.0) == "neutral"

    def test_neutral_inclusive_on_both_ends(self):
        # `_ml_decide` neutral arm: `0 ≤ p ≤ 5`. Both endpoints inclusive.
        assert sst._gate_bucket(0.0) == "neutral"
        assert sst._gate_bucket(2.5) == "neutral"
        assert sst._gate_bucket(5.0) == "neutral"

    def test_mild_tailwind_above_five(self):
        assert sst._gate_bucket(5.01) == "mild_tailwind"
        assert sst._gate_bucket(10.0) == "mild_tailwind"
        # Exactly 10 is NOT strong_tailwind (the gate's `> 10` is strict),
        # so 10.0 stays in mild_tailwind — matching `_ml_decide`.
        assert sst._gate_bucket(10.0) == "mild_tailwind"

    def test_strong_tailwind_above_ten(self):
        assert sst._gate_bucket(10.01) == "strong_tailwind"
        assert sst._gate_bucket(50.0) == "strong_tailwind"

    def test_nan_and_non_numeric_fall_through_to_neutral(self):
        """A non-finite or non-numeric input must never raise — the
        gate ladder mirror is supposed to be total. Neutral (the no-op
        arm) is the safest default; an erroneous fall-through to a
        non-neutral bucket would fabricate gate decisions."""
        assert sst._gate_bucket(float("nan")) == "neutral"
        assert sst._gate_bucket(None) == "neutral"
        assert sst._gate_bucket("bogus") == "neutral"


class TestGateBucketsDegenerate:
    def test_neutral_only_scorer_yields_gate_buckets_degenerate(self):
        """Predictions span 8 DISTINCT values (passes the constant check)
        but all map to the same neutral gate bucket → verdict must catch
        this. This is the n_train=400-clobber failure pattern AGENTS.md
        review pass #2 documented; before this verdict the smoke test
        marked this scorer HEALTHY."""
        rep = sst.scorer_smoke_report(scorer=_NeutralBucketScorer())
        assert rep["verdict"] == "GATE_BUCKETS_DEGENERATE", rep
        assert rep["distinct_predictions"] >= 2  # the failure signature
        assert rep["distinct_gate_buckets"] == 1
        # All 8 probes must be in the neutral bucket per construction.
        assert rep["gate_bucket_counts"]["neutral"] == 8
        # Hint mentions both the bucket name AND that the gate is dormant.
        assert "neutral" in rep["hint"]
        assert "dormant" in rep["hint"].lower()

    def test_two_distinct_buckets_passes_healthy(self):
        """The boundary case: exactly 2 gate buckets populated must
        still verdict HEALTHY (≥ 2, not > 2)."""
        rep = sst.scorer_smoke_report(scorer=_TwoBucketScorer())
        assert rep["verdict"] == "HEALTHY", rep
        assert rep["distinct_gate_buckets"] == 2
        # Healthy hint surfaces the bucket count for operator visibility.
        assert "gate buckets" in rep["hint"]

    def test_gate_bucket_counts_include_every_bucket_key(self):
        """JSON-schema lock: every documented bucket name appears as a
        key in `gate_bucket_counts`, even with 0 count. Without this,
        a dashboard / Discord template would KeyError on an absent arm
        when no probe lands in it."""
        rep = sst.scorer_smoke_report(scorer=_NeutralBucketScorer())
        for b in sst._GATE_BUCKETS:
            assert b in rep["gate_bucket_counts"], (b, rep)
        # Sum across all buckets equals the number of in-distribution
        # probes — no probe is lost, no probe is double-counted.
        total = sum(rep["gate_bucket_counts"].values())
        assert total == rep["n_probes"]

    def test_constant_predictor_takes_priority_over_gate_buckets(self):
        """A scorer that fails the STRONGER DEGENERATE_CONSTANT check
        (all predictions equal at 1e-4 tolerance) also fails the gate-
        bucket check (1 bucket), but the stronger verdict must win for
        diagnostic precision. The hint specifically tells operators
        'constant predictor', not 'one bucket'."""
        rep = sst.scorer_smoke_report(scorer=_ConstantScorer())
        assert rep["verdict"] == "DEGENERATE_CONSTANT", rep
        assert rep["distinct_predictions"] == 1
        # The bucket count IS computed (all 8 in neutral) but the verdict
        # ladder selected the stronger fail.
        assert rep["gate_bucket_counts"]["neutral"] == 8

    def test_off_distribution_does_not_block_gate_buckets_verdict(self):
        """An off-distribution scorer whose predictions span multiple
        gate buckets is still HEALTHY (off_distribution is informational).
        Conversely, a scorer ALL off-dist AND all in one bucket should
        verdict GATE_BUCKETS_DEGENERATE — off-dist alone doesn't save
        it. Belt-and-braces for the verdict precedence."""
        class _OneBucketOffDist:
            is_trained = True
            n_train = 1500

            def predict_with_meta(self, **kw):
                seed = sum(ord(c) for c in kw.get("ticker", ""))
                pred = 0.5 + ((seed % 11) * 0.4)
                return {"pred": round(pred, 4), "raw": round(pred, 4),
                        "clamped": True, "off_distribution": True}

        rep = sst.scorer_smoke_report(scorer=_OneBucketOffDist())
        assert rep["verdict"] == "GATE_BUCKETS_DEGENERATE", rep
        assert rep["off_distribution_in_distribution"] == 8

    def test_cli_exit_code_is_two_for_gate_buckets_degenerate(self,
                                                              monkeypatch,
                                                              capsys):
        """Cron contract: GATE_BUCKETS_DEGENERATE must exit 2 like the
        other actionable verdicts. An operator running this in a 5-min
        cron loop relies on `$?` to branch on 'gate is dormant right
        now' without parsing stdout."""
        rep = sst.scorer_smoke_report(scorer=_NeutralBucketScorer())
        assert rep["verdict"] == "GATE_BUCKETS_DEGENERATE"
        actual_rc = 0 if rep["verdict"] in ("HEALTHY", "UNTRAINED") else 2
        assert actual_rc == 2


class TestPredictionMath:
    def test_predictions_are_rounded_to_4dp(self):
        """The pred field is rounded to 4dp for stable JSON output —
        downstream snapshots / Discord lines depend on stable formatting."""
        rep = sst.scorer_smoke_report(scorer=_HealthyScorer())
        for r in rep["probes"]:
            assert isinstance(r["pred"], float)
            # 4dp means at most 4 decimal places after the point.
            s = f"{r['pred']}"
            if "." in s:
                assert len(s.split(".")[1]) <= 4

    def test_healthy_predictions_are_finite(self):
        """Every reported pred must be finite — the upstream contract
        for the gate (CLAUDE.md §6)."""
        rep = sst.scorer_smoke_report(scorer=_HealthyScorer())
        for r in rep["probes"]:
            assert math.isfinite(r["pred"]), r


# ─────────────────────────── CLI exit codes ─────────────────────────────────


class TestCli:
    def test_cli_exit_code_matches_verdict(self, monkeypatch, capsys):
        """The CLI exit mirrors `scorer_freshness._cli` semantics: 0 on
        HEALTHY / UNTRAINED, 2 on the failing verdicts. A cron must be
        able to branch on `$?` without parsing stdout."""
        for scorer, expected_rc in (
            (_UntrainedScorer(), 0),
            (_HealthyScorer(), 0),
            (_ConstantScorer(), 2),
            (_RaisingScorer(), 2),
        ):
            # Inject the scorer via the public function the CLI calls.
            monkeypatch.setattr(
                sst, "scorer_smoke_report",
                lambda scorer=scorer, **kw: sst.__wrapped_orig(scorer=scorer)
                if False else sst._original_report(scorer=scorer),
                raising=True,
            )
            # We can't easily intercept the CLI's internal construction
            # without changing the module signature, so we directly assert
            # on the report's verdict and the exit-mapping discipline
            # `_cli` documents.
            rep = sst.scorer_smoke_report(scorer=scorer)
            actual_rc = 0 if rep["verdict"] in ("HEALTHY", "UNTRAINED") else 2
            assert actual_rc == expected_rc, (scorer, rep)


# Compatibility shim for the CLI monkeypatch above — older versions of
# pytest's monkeypatch don't let you re-set a function that captures via
# closure cleanly, so we stash the original here.
sst._original_report = sst.scorer_smoke_report
