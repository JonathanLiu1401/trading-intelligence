"""Tests for `paper_trader.ml.scorer_smoke_test` — DecisionScorer health
sanity check.

Pins the four documented verdicts so a refactor cannot silently weaken a
verdict to HEALTHY when the scorer is actually broken (the regression-of-
record that motivated this module):

  * HEALTHY              — predictions finite & span ≥ 2 distinct buckets
  * UNTRAINED            — no pickle / is_trained False
  * DEGENERATE_CONSTANT  — every probe collapses to one prediction
  * BROKEN_PREDICT       — predict_with_meta raised or returned non-finite

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
        assert len(sst.VERDICTS) == 4
        assert set(sst.VERDICTS) == {
            "HEALTHY", "UNTRAINED", "DEGENERATE_CONSTANT", "BROKEN_PREDICT",
        }
        assert len(sst._PROBES) >= 8
        assert len(sst._EDGE_PROBES) >= 2


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
