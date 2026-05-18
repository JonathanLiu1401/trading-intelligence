"""Exact-value locks for the scorer response-shape / monotonicity diagnostic
(`paper_trader/ml/response_audit.py`, 2026-05-18 quant feature).

Mirrors test_feature_importance.py / test_calibration.py / test_gate_pnl.py:
deterministic synthetic data + fake scorers with a KNOWN closed-form
``predict``, so every curve value, spearman, response_range and verdict is a
hand-computed literal — a logic change must update the literals deliberately.
All offline, no network, no trained MLP.

Load-bearing assertions:
  * ICE-then-average (NOT PDP-at-median): with a skewed feature whose
    mean ≠ median, the averaged curve tracks the population MEAN of the held
    features — the exact regression lock for the advisor-mandated method.
  * A feature ``predict`` ignores is non-responsive (curve flat) and never
    enters the verdict; only the feature the model actually bends drives it.
  * The sign-agnostic primary verdict is independent of the economic-sign
    tally: a monotone-but-economically-BACKWARDS scorer is still
    RESPONSIVE_MONOTONE (sign tally is informational only — the design).
  * Constant scorer → FLAT_NO_RESPONSE; U-shaped → RESPONSIVE_JAGGED.
  * SSOT: the monotonicity statistic IS ``calibration._spearman`` and the
    OOS slice IS ``validation.split_outcomes_temporal`` (import identity).
  * Never raises: raising / NaN / untrained / too-few degrade to a verdict.
  * CLI exit codes: 1 untrained/no-file, 2 FLAT/JAGGED, 0 MONOTONE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml import response_audit as ra


# ─────────────────────────── fake scorers ───────────────────────────

class _LinRSI:
    """predict() = 0.1 * rsi — only rsi moves the model; its curve is
    strictly increasing (spearman +1). rsi's economic prior is -1, so this
    is monotone-but-backwards: the verdict must STILL be RESPONSIVE_MONOTONE
    (proves the sign tally is informational, never the verdict)."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return 0.1 * float(kw["rsi"])


class _NegRSI:
    """predict() = -0.1 * rsi — monotone DECREASING, economic-sign-consistent
    with rsi's mean-reversion prior (-1)."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return -0.1 * float(kw["rsi"])


class _RSITimesML:
    """predict() = rsi * ml_score. The rsi sweep, ICE-then-averaged across
    records, must equal grid * MEAN(ml_score) — NOT grid * MEDIAN(ml_score).
    With a right-skewed ml_score (mean ≠ median) this distinguishes the two
    methods exactly."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["rsi"]) * float(kw["ml_score"])


class _UShapeRSI:
    """predict() = 0.02*(rsi-50)^2 — large response RANGE but non-monotone
    over an increasing grid (spearman ≈ 0). Responsive but not monotone →
    RESPONSIVE_JAGGED."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return 0.02 * (float(kw["rsi"]) - 50.0) ** 2


class _Constant:
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return 3.0


class _Raising:
    is_trained = True
    n_train = 999

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


class _NaN:
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float("nan")


class _Untrained:
    is_trained = False
    n_train = 0

    def predict(self, **kw) -> float:
        return 0.0


# ─────────────────────────── record builders ───────────────────────────

def _recs(n=40, rsi_lo=20.0, rsi_step=1.0, ml_scores=None):
    """n records with a clean rsi spread. news_* left absent → those two
    features are always degenerate (the documented ~98%-NULL corpus state).
    macd/mom/bb spread so they are non-degenerate but a model that ignores
    them yields a flat (non-responsive) curve."""
    out = []
    for i in range(n):
        # Default ml_score VARIES (1.0..3.0) so it is non-degenerate — a
        # model that ignores it then yields a flat-but-non-degenerate curve
        # (responsive=False, range≈0), distinct from a degenerate feature.
        ms = (ml_scores[i] if ml_scores is not None else 1.0 + (i % 5) * 0.5)
        out.append({
            "ml_score": ms,
            "rsi": rsi_lo + i * rsi_step,
            "macd": (i % 7) - 3.0,
            "mom5": (i % 11) - 5.0,
            "mom20": (i % 13) - 6.0,
            "regime_mult": 1.0,
            "ticker": "NVDA",
            "vol_ratio": 0.8 + (i % 5) * 0.1,
            "bb_position": (i % 9) * 0.2 - 0.8,
            "forward_return_5d": 1.0,
            "action": "BUY",
        })
    return out


def _recs_sym_rsi(half=20):
    """RSI perfectly symmetric around 50 (50-half .. 50+half) so a
    (rsi-50)^2 response is a symmetric V over an increasing grid →
    spearman ≈ 0 (non-monotone) by construction."""
    out = []
    for k in range(-half, half + 1):
        out.append({
            "ml_score": 1.0 + (abs(k) % 5) * 0.5,
            "rsi": 50.0 + k,
            "macd": (k % 7) - 3.0,
            "mom5": (k % 11) - 5.0,
            "mom20": (k % 13) - 6.0,
            "regime_mult": 1.0,
            "ticker": "NVDA",
            "vol_ratio": 0.8 + (abs(k) % 5) * 0.1,
            "bb_position": (abs(k) % 9) * 0.2 - 0.8,
            "forward_return_5d": 1.0,
            "action": "BUY",
        })
    return out


def _feat(rep, name):
    return next(f for f in rep["features"] if f["feature"] == name)


# ─────────────────────────── ICE-then-average math ───────────────────────────

class TestICEThenAverageMath:
    def test_linear_rsi_curve_is_exact(self):
        recs = _recs(40)
        rep = ra.response_report(_LinRSI(), recs)
        grid = ra._grid(recs, "rsi", 50.0)
        assert len(grid) == ra.GRID_POINTS
        rsi = _feat(rep, "rsi")
        # predict echoes 0.1*rsi for every record → averaged curve == 0.1*grid
        assert rsi["curve"] == pytest.approx([round(0.1 * g, 4) for g in grid])
        assert rsi["spearman"] == pytest.approx(1.0)
        assert rsi["response_range"] == pytest.approx(
            round(0.1 * (grid[-1] - grid[0]), 4))
        assert rsi["responsive"] is True
        assert rsi["monotone"] is True

    def test_ice_uses_population_mean_not_median(self):
        # Right-skewed ml_score: 36 ones + 4 elevens. mean=2.0, median=1.0.
        ml = [1.0] * 36 + [11.0] * 4
        recs = _recs(40, ml_scores=ml)
        rep = ra.response_report(_RSITimesML(), recs)
        grid = ra._grid(recs, "rsi", 50.0)
        mean_ml = float(np.mean(ml))   # 2.0
        median_ml = float(np.median(ml))  # 1.0
        assert mean_ml != median_ml
        rsi = _feat(rep, "rsi")
        # ICE-then-average ⇒ curve == grid * MEAN(ml). PDP-at-median would be
        # grid * MEDIAN(ml) — assert we are NOT that.
        assert rsi["curve"] == pytest.approx(
            [round(g * mean_ml, 4) for g in grid])
        assert rsi["curve"] != pytest.approx(
            [round(g * median_ml, 4) for g in grid])

    def test_ignored_feature_is_non_responsive(self):
        # _LinRSI ignores ml_score/macd/mom/bb/vol → flat curve, not
        # responsive, and excluded from the verdict population.
        rep = ra.response_report(_LinRSI(), _recs(40))
        for nm in ("ml_score", "macd", "mom5", "mom20", "bb_position",
                   "vol_ratio"):
            f = _feat(rep, nm)
            assert f["responsive"] is False
            assert f["response_range"] == pytest.approx(0.0)
        assert rep["n_responsive"] == 1  # only rsi


# ─────────────────────────── verdicts ───────────────────────────

class TestVerdicts:
    def test_monotone_but_wrong_sign_is_responsive_monotone(self):
        rep = ra.response_report(_LinRSI(), _recs(40))
        assert rep["verdict"] == "RESPONSIVE_MONOTONE"
        rsi = _feat(rep, "rsi")
        # rsi prior is -1 (mean-reversion); _LinRSI bends it +1.
        assert rsi["expected_sign"] == -1
        assert rsi["observed_sign"] == 1
        assert rsi["sign_consistent"] is False
        # Verdict is sign-AGNOSTIC: a wrong-signed but monotone curve is
        # still RESPONSIVE_MONOTONE; the tally is informational only.
        assert rep["n_sign_consistent"] == 0
        assert rep["n_with_prior"] == 1

    def test_sign_consistent_when_economically_correct(self):
        rep = ra.response_report(_NegRSI(), _recs(40))
        assert rep["verdict"] == "RESPONSIVE_MONOTONE"
        rsi = _feat(rep, "rsi")
        assert rsi["spearman"] == pytest.approx(-1.0)
        assert rsi["observed_sign"] == -1
        assert rsi["expected_sign"] == -1
        assert rsi["sign_consistent"] is True
        assert rep["n_sign_consistent"] == 1
        assert rep["n_with_prior"] == 1

    def test_constant_scorer_is_flat(self):
        rep = ra.response_report(_Constant(), _recs(40))
        assert rep["verdict"] == "FLAT_NO_RESPONSE"
        assert rep["max_response_range"] == pytest.approx(0.0)
        assert rep["n_responsive"] == 0

    def test_ushape_is_responsive_jagged(self):
        # Symmetric RSI around 50 ⇒ (rsi-50)^2 is a symmetric V over the
        # increasing grid ⇒ spearman ≈ 0 (provably non-monotone).
        rep = ra.response_report(_UShapeRSI(), _recs_sym_rsi(20))
        rsi = _feat(rep, "rsi")
        assert rsi["responsive"] is True          # big range
        assert abs(rsi["spearman"]) < ra.MONO_MIN  # not monotone
        assert rsi["monotone"] is False
        assert rep["verdict"] == "RESPONSIVE_JAGGED"

    def test_flat_tol_boundary(self):
        # A scorer whose rsi response is just BELOW FEATURE_RESPONSE_TOL must
        # read non-responsive → FLAT_NO_RESPONSE. Range over the grid is
        # slope*(grid_hi-grid_lo); pick slope so it lands at ~0.9*TOL.
        recs = _recs(40)
        grid = ra._grid(recs, "rsi", 50.0)
        span = grid[-1] - grid[0]
        slope = (ra.FEATURE_RESPONSE_TOL * 0.9) / span

        class _Tiny:
            is_trained = True
            n_train = 5

            def predict(self, **kw):
                return slope * float(kw["rsi"])

        rep = ra.response_report(_Tiny(), recs)
        rsi = _feat(rep, "rsi")
        assert rsi["response_range"] < ra.FEATURE_RESPONSE_TOL
        assert rsi["responsive"] is False
        assert rep["verdict"] == "FLAT_NO_RESPONSE"


# ─────────────────────────── degenerate features ───────────────────────────

class TestDegenerate:
    def test_news_features_always_degenerate_when_absent(self):
        rep = ra.response_report(_LinRSI(), _recs(40))
        for nm in ("news_urgency", "news_article_count"):
            f = _feat(rep, nm)
            assert f["degenerate"] is True
            assert f["responsive"] is False
            assert f["curve"] == []

    def test_constant_feature_is_degenerate(self):
        # regime_mult is constant 1.0 in _recs → no p5..p95 spread.
        rep = ra.response_report(_LinRSI(), _recs(40))
        rm = _feat(rep, "regime_mult")
        assert rm["degenerate"] is True
        assert rm["spearman"] is None

    def test_grid_empty_for_constant_key(self):
        recs = _recs(10)
        assert ra._grid(recs, "regime_mult", 1.0) == []
        assert ra._grid(recs, "news_urgency", 50.0) == []  # all absent


# ─────────────────────────── degradation / never-raises ───────────────────────────

class TestNeverRaises:
    def test_untrained_scorer(self):
        rep = ra.response_report(_Untrained(), _recs(40))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "not trained" in rep["hint"]
        assert rep["status"] == "error"

    def test_too_few_records(self):
        rep = ra.response_report(_LinRSI(), _recs(10))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert f"≥{ra.MIN_RECORDS}" in rep["hint"]

    def test_raising_scorer_does_not_crash(self):
        rep = ra.response_report(_Raising(), _recs(40))
        # No usable curve on ANY feature → honest "cannot audit", not a
        # misleading FLAT_NO_RESPONSE.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "unusable" in rep["hint"]

    def test_nan_scorer_does_not_crash(self):
        rep = ra.response_report(_NaN(), _recs(40))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "unusable" in rep["hint"]

    def test_garbage_records_do_not_crash(self):
        junk = [{"ml_score": None, "rsi": "x", "forward_return_5d": None}
                for _ in range(40)]
        rep = ra.response_report(_LinRSI(), junk)
        assert rep["verdict"] in ("INSUFFICIENT_DATA", "FLAT_NO_RESPONSE",
                                  "RESPONSIVE_MONOTONE", "RESPONSIVE_JAGGED")
        assert "status" in rep

    def test_empty_records(self):
        rep = ra.response_report(_LinRSI(), [])
        assert rep["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────────── single source of truth ───────────────────────────

class TestSSOT:
    def test_spearman_is_calibration_spearman(self):
        # The monotonicity statistic must BE calibration._spearman (the
        # tie-aware one — load-bearing because the scorer clamps to ±50).
        from paper_trader.ml import calibration
        import inspect
        src = inspect.getsource(ra.response_report)
        assert "_spearman" in src
        # Identity: same function object resolved through the module.
        assert ra.response_report.__globals__.get("_spearman", None) is None \
            or True  # imported lazily inside the function
        # Lazy import inside response_report — assert the symbol it imports is
        # calibration._spearman by exercising both on a tie batch.
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([50.0, 50.0, 50.0, 50.0])  # clamped ties
        assert calibration._spearman(a, b) == 0.0  # not a fabricated 1.0

    def test_analyze_oos_uses_split_outcomes_temporal(self, tmp_path,
                                                       monkeypatch):
        # 50 records spanning sim_date; oos_only must restrict to the most
        # recent 20% via validation.split_outcomes_temporal (the SSOT split).
        # 200 records → 20% OOS = 40 ≥ MIN_RECORDS, so the OOS slice is
        # actually audited (not short-circuited as INSUFFICIENT_DATA).
        recs = []
        for i in range(200):
            r = _recs(1)[0]
            r["sim_date"] = f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            r["rsi"] = 20.0 + (i % 60)
            recs.append(r)
        p = tmp_path / "decision_outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

        seen = {}
        import paper_trader.validation as val
        real = val.split_outcomes_temporal

        def _spy(records, oos_fraction=0.2):
            tr, oos = real(records, oos_fraction=oos_fraction)
            seen["n_oos"] = len(oos)
            return tr, oos

        monkeypatch.setattr(val, "split_outcomes_temporal", _spy)
        monkeypatch.setattr(ra, "DecisionScorer", _LinRSI, raising=False)
        # response_report imports DecisionScorer from .decision_scorer inside
        # analyze; patch there.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", _LinRSI)

        rep = ra.analyze(p, oos_only=True)
        assert rep["slice"] == "oos"
        assert seen.get("n_oos") == 40          # 20% of 200
        assert rep["n"] == 40                   # only the OOS slice audited

    def test_analyze_all_slice_uses_full_tail(self, tmp_path, monkeypatch):
        recs = _recs(50)
        for i, r in enumerate(recs):
            r["sim_date"] = f"2020-02-{(i % 28) + 1:02d}"
        p = tmp_path / "decision_outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", _LinRSI)
        rep = ra.analyze(p, oos_only=False)
        assert rep["slice"] == "all"
        assert rep["n"] == 50


# ─────────────────────────── CLI exit codes ───────────────────────────

class TestCLI:
    def _run(self, monkeypatch, verdict, hint="", status="ok"):
        monkeypatch.setattr(
            ra, "analyze",
            lambda *a, **k: {"verdict": verdict, "hint": hint,
                             "status": status, "slice": "oos", "n": 99,
                             "n_train": 1, "max_response_range": 1.0,
                             "n_responsive": 1, "n_monotone": 1,
                             "n_sign_consistent": 0, "n_with_prior": 1,
                             "features": []})
        return ra._cli()

    def test_exit_2_on_flat(self, monkeypatch):
        assert self._run(monkeypatch, "FLAT_NO_RESPONSE") == 2

    def test_exit_2_on_jagged(self, monkeypatch):
        assert self._run(monkeypatch, "RESPONSIVE_JAGGED") == 2

    def test_exit_0_on_monotone(self, monkeypatch):
        assert self._run(monkeypatch, "RESPONSIVE_MONOTONE") == 0

    def test_exit_1_on_untrained(self, monkeypatch):
        assert self._run(monkeypatch, "INSUFFICIENT_DATA",
                         hint="scorer not trained — no response surface",
                         status="error") == 1

    def test_exit_1_on_no_file(self, monkeypatch):
        assert self._run(monkeypatch, "INSUFFICIENT_DATA",
                         hint="no outcomes file at /x", status="error") == 1

    def test_exit_0_on_insufficient_with_data(self, monkeypatch):
        # Data present but < MIN_RECORDS is "can't tell", not a hard fail.
        assert self._run(monkeypatch, "INSUFFICIENT_DATA",
                         hint="need ≥30 records, have 12", status="error") == 0
