"""Exact-value + known-answer locks for the linear-probe diagnostic
(`paper_trader/ml/linear_probe.py`, 2026-05-18 ML/backtest quant feature).

Mirrors test_baseline_compare.py: deterministic synthetic data, EXACT
verdicts (not ranges), all offline, no network, no trained MLP pickle.

The load-bearing assertions:
  * `_fit_ridge` RECOVERS a known noiseless linear combination (rank-IC ≈ 1)
    and a single feature alone cannot (the whole point of the probe).
  * the four-way verdict ladder on (probe vs MLP vs best one-liner), with
    Spearman fixed to +1.0 / 0.0 by construction (monotone vs symmetric tent).
  * **no-leakage discipline**: a signal present ONLY in the temporal-OOS
    slice (train target is noise) must NOT yield RECOVERS_SIGNAL — proves the
    probe is fit on train only and cannot see the OOS it is judged on.
  * end-to-end known answer: a combinable 2-feature signal a noise-MLP
    cannot model → LINEAR_PROBE_RECOVERS_SIGNAL; pure noise →
    NO_COMBINABLE_SIGNAL.
  * the MLP rank-IC equals `baseline_compare`'s on the identical scorer +
    records (single-source-of-truth via shared `_skill` — a no-drift check).
  * RIDGE_ALPHA robustness: same verdict across alpha 0.1 → 10.
  * the SELL `-forward_return_5d` flip is learned by the fitted probe.
  * never raises — empty / garbage / raising scorer / non-list degrade.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from paper_trader.ml import linear_probe as lp
from paper_trader.ml import baseline_compare as bc


# ─────────────────────────── fake scorers ───────────────────────────

class _NoiseScorer:
    """Deterministic, per-row-varying, but rank-UNCORRELATED 'prediction' —
    a model with no skill. Hashing the (mom20, mom5) pair destroys the
    feature ordering, so the output varies row-to-row (non-degenerate) yet
    its rank carries ≈0 IC with the target — exactly a no-skill MLP."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        key = (round(float(kw.get("mom20") or 0.0), 3),
               round(float(kw.get("mom5") or 0.0), 3))
        return float(hash(key) % 1999 - 999) / 100.0


class _EchoMom20:
    """predict() echoes mom20 so the MLP column is fully controllable."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        v = kw.get("mom20")
        return float(v) if v is not None else 0.0


class _RaisingScorer:
    is_trained = True
    n_train = 999

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


# ─────────────────────────── data builders ───────────────────────────

def _date(i: int) -> str:
    return (date(2020, 1, 1) + timedelta(days=i)).isoformat()


def _rec(i, *, forward, mom20, mom5, action="BUY", ticker="NVDA"):
    """Outcome row with ONLY mom20/mom5 varying — every other numeric
    feature constant so its one-liner is degenerate and the only
    non-degenerate baselines are `mom20` / `mom5` (controlled)."""
    return {
        "run_id": 1,
        "sim_date": _date(i),
        "ticker": ticker,
        "action": action,
        "ml_score": 5.0,
        "rsi": 50.0,
        "macd": 0.0,
        "mom5": mom5,
        "mom20": mom20,
        "regime_mult": 1.0,
        "vol_ratio": 1.0,
        "bb_position": 0.0,
        "news_urgency": None,
        "news_article_count": None,
        "forward_return_5d": forward,
    }


def _combinable_dataset(n=240, seed=7, noise=0.0):
    """forward = 0.3*mom20 + 0.6*mom5 — each feature contributes equal
    variance, so NEITHER alone explains it but their linear combination
    does almost perfectly. sim_date strictly increasing for a deterministic
    temporal split."""
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        m20 = float(rng.randn() * 10.0)
        m5 = float(rng.randn() * 5.0)
        fwd = 0.3 * m20 + 0.6 * m5 + (float(rng.randn()) * noise)
        recs.append(_rec(i, forward=round(fwd, 4), mom20=round(m20, 4),
                         mom5=round(m5, 4)))
    return recs


# ─────────────────────────── _fit_ridge ───────────────────────────

def test_fit_ridge_recovers_known_linear_combo():
    """Noiseless y = 2*x0 - 3*x4: the fitted probe's predictions must
    rank-correlate ≈ 1.0 with y (the core capability the diagnostic rests
    on). A single column alone must NOT (proves combination is required)."""
    rng = np.random.RandomState(1)
    X = rng.randn(150, lp._N_NUMERIC)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 4]
    probe = lp._fit_ridge(X, y, alpha=0.01)
    assert probe is not None
    pred = probe.predict(X)
    ic = bc._spearman(np.asarray(pred, float), np.asarray(y, float))
    assert ic > 0.99, f"ridge failed to recover linear combo: ic={ic}"
    # x0 alone is materially worse than the fitted combination.
    ic_x0 = bc._spearman(X[:, 0].astype(float), y.astype(float))
    assert ic_x0 < ic - 0.10


def test_fit_ridge_degrades_safely():
    assert lp._fit_ridge(np.zeros((1, 10)), np.zeros(1)) is None          # <2 rows
    assert lp._fit_ridge(np.zeros((0, 10)), np.zeros(0)) is None          # empty
    bad = np.full((5, 10), np.inf)
    assert lp._fit_ridge(bad, np.zeros(5)) is None                        # non-finite
    # Zero-variance column must NOT divide-by-zero — fit still succeeds.
    X = np.random.RandomState(2).randn(40, 10)
    X[:, 3] = 7.0  # constant column
    y = X[:, 0]
    probe = lp._fit_ridge(X, y)
    assert probe is not None
    assert np.all(np.isfinite(probe.predict(X)))


def test_fit_ridge_is_deterministic():
    rng = np.random.RandomState(3)
    X, y = rng.randn(60, 10), rng.randn(60)
    p1, p2 = lp._fit_ridge(X, y), lp._fit_ridge(X, y)
    assert np.allclose(p1.predict(X), p2.predict(X))


# ─────────────────────── _numeric_features ───────────────────────

def test_numeric_features_length_and_defaults():
    """Exactly 10 numeric features, in build_features order, with the SAME
    defaults the MLP sees (missing rsi → 50.0 at slot 1, missing mom5 → 0.0
    at slot 3)."""
    feats = lp._numeric_features({"ml_score": 4.0, "mom20": 2.5})
    assert len(feats) == lp._N_NUMERIC == 10
    assert feats[0] == 4.0      # ml_score
    assert feats[1] == 50.0     # rsi default
    assert feats[3] == 0.0      # mom5 default
    assert feats[4] == 2.5      # mom20


# ─────────────────────── verdict ladder (pure fn) ───────────────────────

def _perfect(n):  # spearman +1.0 vs increasing targets
    return [float(i) for i in range(n)]


def _tent(n):     # symmetric V → spearman exactly 0.0 vs increasing targets
    c = (n - 1) / 2.0
    return [abs(i - c) for i in range(n)]


def test_verdict_recovers_signal():
    n = 40
    tgt = _perfect(n)
    rep = lp.linear_probe_report(
        probe_preds=_perfect(n),            # ic +1.0
        mlp_preds=_tent(n),                 # ic  0.0
        baseline_preds={"mom20": _tent(n),  # ic  0.0 (non-degenerate)
                        "ml_score": [1.0] * n},  # constant → degenerate
        targets=tgt,
    )
    assert rep["verdict"] == "LINEAR_PROBE_RECOVERS_SIGNAL"
    assert rep["probe"]["rank_ic"] == 1.0
    assert rep["mlp"]["rank_ic"] == 0.0
    assert rep["best_baseline"] == "mom20"
    assert rep["probe_minus_mlp"] == 1.0
    assert rep["probe_minus_best_baseline"] == 1.0


def test_verdict_no_combinable_signal():
    n = 40
    rep = lp.linear_probe_report(
        probe_preds=_tent(n),               # ic 0.0
        mlp_preds=_tent(n),                 # ic 0.0
        baseline_preds={"mom20": _perfect(n)},  # ic +1.0 — one feature wins
        targets=_perfect(n),
    )
    assert rep["verdict"] == "NO_COMBINABLE_SIGNAL"
    assert rep["probe"]["rank_ic"] == 0.0
    assert rep["best_baseline_ic"] == 1.0


def test_verdict_linear_matches_mlp():
    n = 40
    rep = lp.linear_probe_report(
        probe_preds=_perfect(n),            # ic +1.0
        mlp_preds=_perfect(n),              # ic +1.0 — probe does NOT beat MLP
        baseline_preds={"mom20": _tent(n)}, # ic  0.0
        targets=_perfect(n),
    )
    assert rep["verdict"] == "LINEAR_MATCHES_MLP"
    assert rep["probe_minus_mlp"] == 0.0


def test_verdict_insufficient_data_below_min_pairs():
    n = lp.MIN_PAIRS - 1
    rep = lp.linear_probe_report(_perfect(n), _perfect(n),
                                 {"mom20": _perfect(n)}, _perfect(n))
    assert rep["verdict"] == "INSUFFICIENT_DATA"
    assert rep["status"] == "ok"


def test_report_never_raises_on_garbage():
    for args in [
        (None, None, None, None),
        ([1, 2], [1], {}, [1, 2, 3]),
        (_perfect(40), _perfect(40), {"x": _perfect(39)}, _perfect(40)),
    ]:
        rep = lp.linear_probe_report(*args)
        assert rep["verdict"] in (
            "INSUFFICIENT_DATA", "LINEAR_PROBE_RECOVERS_SIGNAL",
            "NO_COMBINABLE_SIGNAL", "LINEAR_MATCHES_MLP")


# ─────────────────────── scorer_linear_probe end-to-end ───────────────────────

def test_end_to_end_recovers_signal_vs_noise_mlp():
    """KNOWN ANSWER: forward = 0.3*mom20 + 0.6*mom5 (a combinable signal no
    single feature explains), MLP = pure noise. The probe, fit on the
    temporal-train slice only, must beat both the noise MLP and the best
    single one-liner on the held-out OOS slice → RECOVERS_SIGNAL."""
    recs = _combinable_dataset()
    rep = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    assert rep["slice"] == "oos"
    assert rep["verdict"] == "LINEAR_PROBE_RECOVERS_SIGNAL", rep["hint"]
    assert rep["probe"]["rank_ic"] > 0.10
    assert rep["probe"]["rank_ic"] > rep["mlp"]["rank_ic"] + bc.IC_MARGIN
    assert (rep["probe"]["rank_ic"]
            > rep["best_baseline_ic"] + bc.IC_MARGIN)
    # best single one-liner is real but materially below the combination.
    assert 0.4 < rep["best_baseline_ic"] < 0.9


def test_end_to_end_no_combinable_signal_on_pure_noise():
    """KNOWN ANSWER: forward independent of every feature → no linear
    combination beats the (also-zero) one-liners → NO_COMBINABLE_SIGNAL.

    Uses n=1600 (n_oos≈320) — the realistic data scale (live
    decision_outcomes.jsonl is ≈1507 OOS). At a tiny n_oos a fitted
    10-feature ridge can show a spurious OOS IC; the `MLP_IC_MIN` skill
    floor is exactly the buffer that prevents that small-sample artifact
    from ever firing a false RECOVERS — locked here at scale."""
    rng = np.random.RandomState(11)
    recs = [_rec(i, forward=round(float(rng.randn()), 4),
                 mom20=round(float(rng.randn() * 10), 4),
                 mom5=round(float(rng.randn() * 5), 4)) for i in range(1600)]
    rep = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    assert rep["verdict"] == "NO_COMBINABLE_SIGNAL", rep["hint"]
    # The decisive invariant: pure noise never fabricates RECOVERS, and the
    # probe stays below the real-skill floor.
    assert rep["verdict"] != "LINEAR_PROBE_RECOVERS_SIGNAL"
    assert abs(rep["probe"]["rank_ic"]) < bc.MLP_IC_MIN


def test_no_leakage_signal_only_in_oos_is_not_recovered():
    """The decisive no-leakage lock. Train rows (older sim_date, the first
    80%) carry a pure-NOISE target; OOS rows (recent 20%) carry the strong
    0.3*mom20+0.6*mom5 signal. A probe that (correctly) fits on train only
    learns nothing and CANNOT score the OOS it never saw → must NOT be
    RECOVERS_SIGNAL. A leak would flip this to RECOVERS."""
    rng = np.random.RandomState(13)
    recs = []
    n = 240
    n_oos = max(1, int(n * 0.2))
    for i in range(n):
        m20 = float(rng.randn() * 10.0)
        m5 = float(rng.randn() * 5.0)
        if i < n - n_oos:                      # train: noise target
            fwd = float(rng.randn())
        else:                                  # oos: strong signal
            fwd = 0.3 * m20 + 0.6 * m5
        recs.append(_rec(i, forward=round(fwd, 4), mom20=round(m20, 4),
                         mom5=round(m5, 4)))
    rep = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    assert rep["verdict"] != "LINEAR_PROBE_RECOVERS_SIGNAL", (
        f"LEAK: probe scored an OOS signal it was not trained on "
        f"(verdict={rep['verdict']}, probe_ic={rep['probe']['rank_ic']})")
    # And the probe is demonstrably worse than the (now-strong) one-liner
    # (None when the train-noise probe collapsed to a constant — also a
    # valid no-leak outcome; only assert the gap sign when it is computed).
    pmb = rep["probe_minus_best_baseline"]
    if pmb is not None:
        assert pmb < 0.0


def test_mlp_rank_ic_matches_baseline_compare_no_drift():
    """Single-source-of-truth lock: linear_probe and baseline_compare both
    reuse `_skill` on the SAME temporal-OOS slice + SAME alignment, so the
    MLP rank-IC they report for one scorer + dataset must be identical
    (a feature reorder or alignment change in either side breaks this)."""
    recs = _combinable_dataset(seed=21)
    scorer = _EchoMom20()
    a = lp.scorer_linear_probe(scorer, recs, oos_only=True)
    b = bc.scorer_baseline_compare(scorer, recs, oos_only=True)
    assert a["mlp"]["rank_ic"] == b["mlp"]["rank_ic"]
    assert a["mlp"]["n"] == b["mlp"]["n"]


def test_determinism_same_verdict_and_ic_across_calls():
    recs = _combinable_dataset(seed=33)
    r1 = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    r2 = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    assert r1["verdict"] == r2["verdict"]
    assert r1["probe"]["rank_ic"] == r2["probe"]["rank_ic"]


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_ridge_alpha_robustness_same_verdict(alpha, monkeypatch):
    """The module comment claims the verdict is robust across alpha 0.1–10.
    Lock it: the combinable-signal dataset reads RECOVERS at every alpha."""
    monkeypatch.setattr(lp, "RIDGE_ALPHA", alpha)
    rep = lp.scorer_linear_probe(_NoiseScorer(), _combinable_dataset(seed=5),
                                 oos_only=True)
    assert rep["verdict"] == "LINEAR_PROBE_RECOVERS_SIGNAL", (
        f"alpha={alpha}: {rep['hint']}")


def test_sell_flip_is_learned_by_probe():
    """All-SELL slice: a SELL whose price FELL was the right call, so the
    aligned target is -forward. forward = -(0.3*mom20+0.6*mom5) means the
    *aligned* target = +(0.3*mom20+0.6*mom5) — the probe (fit on the aligned
    target, exactly like the training-aligned MLP) must still recover it.
    Removing the flip would invert the correlation and break RECOVERS."""
    rng = np.random.RandomState(41)
    recs = []
    for i in range(240):
        m20 = float(rng.randn() * 10.0)
        m5 = float(rng.randn() * 5.0)
        fwd = -(0.3 * m20 + 0.6 * m5)        # SELL: drop ⇒ correct
        recs.append(_rec(i, forward=round(fwd, 4), mom20=round(m20, 4),
                         mom5=round(m5, 4), action="SELL"))
    rep = lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=True)
    assert rep["verdict"] == "LINEAR_PROBE_RECOVERS_SIGNAL", rep["hint"]
    assert rep["probe"]["rank_ic"] > 0.10


def test_analyze_and_scorer_probe_never_raise():
    # raising scorer → degrades, never propagates
    rep = lp.scorer_linear_probe(_RaisingScorer(), _combinable_dataset(),
                                 oos_only=True)
    assert rep["verdict"] in ("INSUFFICIENT_DATA", "NO_COMBINABLE_SIGNAL",
                              "LINEAR_MATCHES_MLP",
                              "LINEAR_PROBE_RECOVERS_SIGNAL")
    # empty / non-list / missing-file analyze
    assert lp.scorer_linear_probe(_NoiseScorer(), [],
                                  oos_only=True)["verdict"] == "INSUFFICIENT_DATA"
    assert lp.scorer_linear_probe(_NoiseScorer(), None,
                                  oos_only=True)["verdict"] == "INSUFFICIENT_DATA"
    out = lp.analyze("/no/such/outcomes.jsonl")
    assert out["verdict"] == "INSUFFICIENT_DATA"


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    """Exit 2 on the two actionable verdicts, 0 otherwise. Drive analyze()
    with a synthetic outcomes file + a trained-stub scorer."""
    import json

    recs = _combinable_dataset(seed=9)

    def _fake_analyze(_path, oos_only=True):
        return lp.scorer_linear_probe(_NoiseScorer(), recs, oos_only=oos_only)

    monkeypatch.setattr(lp, "analyze", _fake_analyze)
    rc = lp._cli([])
    out = capsys.readouterr().out
    assert "VERDICT:" in out
    assert "LINEAR_PROBE_RECOVERS_SIGNAL" in out
    # RECOVERS_SIGNAL is actionable → exit 2.
    assert rc == 2
