"""Tests for paper_trader.ml.gate_threshold_sweep.

The sweep evaluates candidate ±strong-band gate thresholds against the
captured (gate_scorer_pred, forward_return_5d) pairs and reports whether
any non-deployed boundary delivers a statistically distinguishable
top-vs-bottom spread. These tests pin the business logic:

  * usable-pair filtering (BUY only, gate_scorer_pred present, finite
    realized 5d, off-distribution abstentions excluded)
  * the four verdicts (INSUFFICIENT_DATA / NO_THRESHOLD_HELPS /
    DEPLOYED_IS_BEST / ALTERNATIVE_THRESHOLD_BEATS_DEPLOYED) each fire
    on a constructed dataset with a known answer
  * the spread is signed (a negative spread is HARMFUL — top arm worse
    than bottom — not virtuous)
  * the CLI's exit code matches the host_guard discipline (0 on a real
    verdict, 1 on INSUFFICIENT_DATA)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import gate_threshold_sweep as gts


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows as JSONL to a fresh outcomes file."""
    out = tmp_path / "outcomes.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out


def _row(action="BUY", pred=0.0, fr=0.0, off_dist=False):
    return {"action": action, "gate_scorer_pred": pred,
            "gate_off_dist": off_dist, "forward_return_5d": fr}


# ─────────────────────── _usable_pairs ───────────────────────

class TestUsablePairs:
    def test_buy_with_pred_and_return_is_kept(self):
        pairs = gts._usable_pairs([_row(pred=5.0, fr=1.5)])
        assert pairs == [(5.0, 1.5)]

    def test_sell_is_dropped(self):
        # Gate is BUY-only (#5); SELL rows never carry a meaningful
        # gate_scorer_pred at decision time.
        pairs = gts._usable_pairs([_row(action="SELL", pred=5.0, fr=1.5)])
        assert pairs == []

    def test_missing_pred_is_dropped(self):
        # An untrained-cycle BUY emits forward_return_5d but no
        # gate_scorer_pred — must NOT contribute to the sweep.
        pairs = gts._usable_pairs([{
            "action": "BUY", "gate_scorer_pred": None,
            "forward_return_5d": 1.5,
        }])
        assert pairs == []

    def test_off_distribution_abstention_is_dropped(self):
        # When the live gate abstained the conviction was left
        # UNTOUCHED — including those rows would mis-attribute the
        # untouched-conviction realized return to a threshold arm the
        # gate never applied. Mirrors gate_realized's exclusion.
        pairs = gts._usable_pairs([_row(pred=50.0, fr=2.0, off_dist=True)])
        assert pairs == []

    def test_non_finite_return_is_dropped(self):
        pairs = gts._usable_pairs([
            _row(pred=5.0, fr=float("nan")),
            _row(pred=5.0, fr=float("inf")),
            _row(pred=5.0, fr=2.0),   # only this one survives
        ])
        assert pairs == [(5.0, 2.0)]


# ─────────────────────── _evaluate_threshold ───────────────────────

class TestEvaluateThreshold:
    def test_known_spread_arithmetic(self):
        # Top arm (pred > +5): 5 rows realizing 3, 5, 4, 6, 7 → mean 5.
        # Bottom arm (pred < -5): 5 rows realizing -1, -3, -2, -4, 0 → mean -2.
        # Spread = +5 - (-2) = +7. (n=5 each is the MIN_ARM_N gate for a
        # computable spread — see test_below_min_arm_n_yields_none_spread.)
        pairs = [(10.0, 3.0), (8.0, 5.0), (12.0, 4.0), (9.0, 6.0), (11.0, 7.0),
                 (-10.0, -1.0), (-8.0, -3.0), (-12.0, -2.0), (-9.0, -4.0),
                 (-11.0, 0.0)]
        out = gts._evaluate_threshold(pairs, bound=5.0)
        assert out["n_top"] == 5
        assert out["n_bot"] == 5
        assert out["mean_top"] == pytest.approx(5.0)
        assert out["mean_bot"] == pytest.approx(-2.0)
        # Bootstrap spread mean is close to (but not exactly) the
        # arithmetic difference — assert within a tolerance that captures
        # the sampling jitter at n=5 per arm.
        assert out["spread"] == pytest.approx(7.0, abs=0.4)
        # The CI must enclose the true spread (7.0) at 95% confidence.
        assert out["spread_ci_low"] < 7.0 < out["spread_ci_high"]

    def test_middle_rows_excluded_from_spread(self):
        # |pred| <= bound rows are middle/no-strong-call — they must NOT
        # contribute to top or bottom arm.
        pairs = [(10.0, 5.0), (-10.0, -5.0),
                 (0.0, 100.0), (3.0, 100.0), (-3.0, 100.0)]
        out = gts._evaluate_threshold(pairs, bound=5.0)
        assert out["n_top"] == 1
        assert out["n_bot"] == 1
        assert out["n_middle"] == 3
        # Middle rows contribute to neither mean.
        assert out["mean_top"] == pytest.approx(5.0)
        assert out["mean_bot"] == pytest.approx(-5.0)

    def test_below_min_arm_n_yields_none_spread(self):
        # 4 rows in each arm — both below MIN_ARM_N=5 ⇒ CI undefined.
        pairs = [(10.0, 1.0)] * (gts.MIN_ARM_N - 1) + \
                [(-10.0, -1.0)] * (gts.MIN_ARM_N - 1)
        out = gts._evaluate_threshold(pairs, bound=5.0)
        # Means are still reported (they're computable on n≥1)…
        assert out["mean_top"] is not None
        assert out["mean_bot"] is not None
        # …but the spread CI requires both arms ≥ MIN_ARM_N.
        assert out["spread"] is None
        assert out["spread_ci_low"] is None
        assert out["spread_ci_high"] is None
        assert out["spread_significant"] is None


# ─────────────────────── analyze() verdicts ───────────────────────

class TestAnalyzeVerdicts:
    def test_no_outcomes_file_yields_insufficient_data(self, tmp_path: Path):
        # Missing file is a fresh-deployment / no-data state — return
        # INSUFFICIENT_DATA, not an error.
        nonexistent = tmp_path / "nope.jsonl"
        rep = gts.analyze(outcomes_path=nonexistent)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_pairs"] == 0

    def test_below_min_records_is_insufficient_data(self, tmp_path: Path):
        # MIN_RECORDS=60 — write 30 usable rows and verify the gate exits.
        rows = [_row(pred=10.0, fr=1.0) for _ in range(30)]
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_pairs"] == 30
        assert "hint" in rep

    def test_no_threshold_helps_when_realized_is_pure_noise(self,
                                                            tmp_path: Path):
        # Construct 1000 BUY rows where realized return is INDEPENDENT
        # of prediction (Gaussian-ish noise around zero). Every
        # candidate's CI should overlap zero → NO_THRESHOLD_HELPS.
        import random as _rnd
        rng = _rnd.Random(42)
        rows = []
        for i in range(1000):
            # Predictions span [-30, +30]; realizations are pure noise
            # uncorrelated with prediction.
            pred = rng.uniform(-30.0, 30.0)
            fr = rng.gauss(0.0, 5.0)  # σ=5pp noise, mean 0
            rows.append(_row(pred=pred, fr=fr))
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        assert rep["n_pairs"] == 1000
        # With independent noise no candidate can carry signal.
        assert rep["verdict"] == "NO_THRESHOLD_HELPS"

    def test_deployed_is_best_when_signal_concentrates_at_pm10(
            self, tmp_path: Path):
        # Construct a dataset where the realized return IS a step
        # function of the prediction at the ±10 boundary — far above
        # +10 → +5pp, far below -10 → -5pp, middle → 0. The deployed
        # boundary is the best by construction. Larger bounds (±15, ±20)
        # have ROUGHLY the same per-row mean inside the top/bottom arm
        # (still +5/-5 by construction) but smaller n, so their spreads
        # are statistically indistinguishable from ±10 — the verdict
        # should land on DEPLOYED_IS_BEST regardless of which bound
        # exactly wins the argmax (the EDGE_TOL_PP guard handles that).
        import random as _rnd
        rng = _rnd.Random(7)
        rows = []
        for _ in range(400):
            pred = rng.uniform(-30.0, 30.0)
            if pred > 10.0:
                fr = rng.gauss(5.0, 0.5)
            elif pred < -10.0:
                fr = rng.gauss(-5.0, 0.5)
            else:
                fr = rng.gauss(0.0, 0.5)
            rows.append(_row(pred=pred, fr=fr))
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        # Verdict must be DEPLOYED_IS_BEST — the step at ±10 means no
        # alternative bound has a MATERIALLY (EDGE_TOL_PP) larger spread.
        assert rep["verdict"] == "DEPLOYED_IS_BEST"
        # The realized spread at ±10 is ~10pp (mean(+5) - mean(-5)).
        assert rep["deployed_spread"] > 5.0

    def test_alternative_threshold_when_signal_concentrates_at_pm15(
            self, tmp_path: Path):
        # Construct a dataset where the realized step IS at ±15. A
        # well-implemented sweep must report a different bound as best
        # AND flip the verdict to ALTERNATIVE_THRESHOLD_BEATS_DEPLOYED.
        import random as _rnd
        rng = _rnd.Random(13)
        rows = []
        for _ in range(600):
            pred = rng.uniform(-30.0, 30.0)
            if pred > 15.0:
                fr = rng.gauss(8.0, 0.5)
            elif pred < -15.0:
                fr = rng.gauss(-8.0, 0.5)
            else:
                # Inside the step, REALIZED is uniformly small noise so
                # ±10 cannot rescue the spread.
                fr = rng.gauss(0.0, 0.5)
            rows.append(_row(pred=pred, fr=fr))
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        # Best should be at or above 15.0 — the real step.
        assert rep["best_bound"] >= 15.0
        # The deployed ±10 spread should be measurably smaller than the
        # best (the realized step is at 15 so ±10 includes a lot of
        # noise in both arms, washing the signal).
        assert rep["best_spread"] > rep["deployed_spread"]
        assert rep["verdict"] == "ALTERNATIVE_THRESHOLD_BEATS_DEPLOYED"

    def test_negative_spread_at_deployed_is_no_threshold_helps(
            self, tmp_path: Path):
        # Realized step is INVERTED at the deployed bound (top arm
        # actually loses, bottom arm wins) — but the inversion shape
        # makes every other candidate also non-significant. Verdict
        # must be NO_THRESHOLD_HELPS, never DEPLOYED_IS_BEST (the gate
        # is actively harmful — calling it "best" would be a lie).
        import random as _rnd
        rng = _rnd.Random(99)
        rows = []
        for _ in range(400):
            pred = rng.uniform(-15.0, 15.0)
            # Tiny anti-correlation: top arm realized slightly negative.
            fr = -0.3 * pred + rng.gauss(0.0, 8.0)
            rows.append(_row(pred=pred, fr=fr))
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        # With this noise level no candidate's spread CI excludes zero.
        # Verdict MUST be NO_THRESHOLD_HELPS (not DEPLOYED_IS_BEST).
        assert rep["verdict"] == "NO_THRESHOLD_HELPS"

    def test_off_distribution_abstentions_are_filtered_before_n_pairs(
            self, tmp_path: Path):
        # 100 usable rows + 50 off-dist abstentions. n_pairs MUST count
        # only the 100 — off-dist rows can't describe gate threshold
        # behaviour because the gate didn't apply a multiplier there.
        rows = [_row(pred=10.0, fr=1.0) for _ in range(100)] + \
               [_row(pred=50.0, fr=99.0, off_dist=True) for _ in range(50)]
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        assert rep["n_pairs"] == 100


# ─────────────────────── CLI ───────────────────────

class TestCLI:
    def test_cli_json_emits_full_report(self, tmp_path: Path, capsys):
        # Empty outcomes ⇒ INSUFFICIENT_DATA verdict ⇒ rc=1.
        nonexistent = tmp_path / "x.jsonl"
        rc = gts._cli(["--outcomes", str(nonexistent), "--json"])
        assert rc == 1
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["verdict"] == "INSUFFICIENT_DATA"

    def test_cli_table_format_does_not_crash_on_real_data(
            self, tmp_path: Path, capsys):
        # 200 known rows with a step at ±10 — verify the table prints
        # without raising and that the verdict line lands.
        import random as _rnd
        rng = _rnd.Random(0)
        rows = []
        for _ in range(200):
            pred = rng.uniform(-30.0, 30.0)
            fr = (5.0 if pred > 10.0
                  else (-5.0 if pred < -10.0 else 0.0)) + \
                rng.gauss(0.0, 0.5)
            rows.append(_row(pred=pred, fr=fr))
        p = _write(tmp_path, rows)
        rc = gts._cli(["--outcomes", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "gate_threshold_sweep" in out
        assert "DEPLOYED_IS_BEST" in out


# ─────────────────────── analyze() never raises ───────────────────────

class TestNeverRaises:
    def test_analyze_swallows_unexpected_exceptions(self, monkeypatch,
                                                    tmp_path: Path):
        # If a downstream helper raised, analyze() must return a
        # status='error' verdict, NEVER propagate (the ledger discipline).
        def _boom(*_a, **_kw):
            raise RuntimeError("simulated downstream crash")
        monkeypatch.setattr(gts, "_evaluate_threshold", _boom)
        # Need enough rows to get past the INSUFFICIENT_DATA guard.
        rows = [_row(pred=10.0, fr=1.0) for _ in range(100)]
        p = _write(tmp_path, rows)
        rep = gts.analyze(outcomes_path=p)
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "hint" in rep
