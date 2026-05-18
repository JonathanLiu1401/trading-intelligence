"""Offline tests for paper_trader.ml.gate_stability.

These assert *business logic*: that a strong deterministic signal reads
GATE_ARM_STABLE, that off-distribution extrapolation reads unstable, that
the diagnostic is reproducible, never raises, and — the load-bearing
safety invariant — NEVER writes the deployed decision_scorer.pkl.

All offline: synthetic outcome JSONL, no network, no backtest.db, no
yfinance. sklearn is required for the MLP path; tests that need it skip
cleanly if it is unavailable.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from paper_trader.ml import gate_stability as gs
from paper_trader.ml.gate_audit import gate_arm as ga_gate_arm

sk = pytest.importorskip("sklearn")


def _write_outcomes(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _strong_signal_rows(n: int) -> list[dict]:
    """forward_return_5d is a strong, deterministic linear function of
    ml_score that sweeps cleanly across every gate arm. Bootstrap models
    should all recover it ⇒ the arm decision must be stable."""
    rows = []
    base = date(2025, 1, 1)
    for i in range(n):
        ms = (i % 20) * 0.5            # 0.0 .. 9.5, dense & repeated
        fr = ms * 4.0 - 18.0           # spans roughly -18 .. +20 → all arms
        rows.append({
            "ticker": ["NVDA", "AMD", "SPY", "MSFT"][i % 4],
            # Unique sim_date per row so train_scorer's
            # (ticker, sim_date, action) dedup never collapses the slice.
            "sim_date": (base + timedelta(days=i)).isoformat(),
            "action": "BUY",
            "ml_score": ms,
            "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
            "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": fr,
            "return_pct": 10.0,
        })
    return rows


def _noisy_signal_rows(n: int) -> list[dict]:
    """forward_return_5d carries only a weak signal swamped by large label
    noise (sd ≈ 25pp on a slope-2 signal). Each bootstrap resample fits a
    materially different surface ⇒ the SAME eval row lands in different gate
    arms across bootstraps. This is the documented near-zero-OOS-skill
    regime — the gate sizing capital on resample luck — that the live scorer
    is in (`oos_ic ≈ 0`, `gate_active=true`)."""
    import random
    rng = random.Random(7)
    rows = []
    base = date(2024, 1, 1)
    for i in range(n):
        ms = 0.5 + (i % 30) * 0.1                 # 0.5 .. 3.4
        fr = ms * 2.0 + rng.gauss(0.0, 25.0)      # signal ≪ noise
        rows.append({
            "ticker": ["NVDA", "AMD", "SPY", "MU", "GS"][i % 5],
            "sim_date": (base + timedelta(days=i)).isoformat(),
            "action": "BUY",
            "ml_score": ms,
            "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
            "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": fr,
            "return_pct": 5.0,
        })
    # Newest rows (the OOS tail the audit evaluates) sit right at a gate
    # boundary (predicted ≈ 0) so noisy bootstrap dispersion straddles it.
    eval_base = base + timedelta(days=n + 10)
    for j in range(max(40, n // 4)):
        rows.append({
            "ticker": ["LITE", "SOXL", "TQQQ"][j % 3],
            "sim_date": (eval_base + timedelta(days=j)).isoformat(),
            "action": "BUY",
            "ml_score": 2.4 + (j % 3) * 0.1,
            "rsi": 55.0, "macd": 0.1, "mom5": 1.0, "mom20": 2.0,
            "regime_mult": 1.0, "vol_ratio": 1.2, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": 0.0,
            "return_pct": 5.0,
        })
    return rows


def _consistent_offdist_rows(n_train: int, n_eval: int) -> list[dict]:
    """Strong monotone in-band signal; the OOS tail is far off-distribution
    in magnitude but ALL bootstraps extrapolate it the SAME direction
    (consistently clamped). This documents the correct, non-obvious
    behaviour: off-distribution magnitude alone is NOT arm-instability —
    only genuine cross-bootstrap *disagreement* is."""
    rows = []
    base = date(2024, 1, 1)
    for i in range(n_train):
        ms = 1.0 + (i % 5) * 0.1
        rows.append({
            "ticker": ["NVDA", "AMD", "SPY"][i % 3],
            "sim_date": (base + timedelta(days=i)).isoformat(),
            "action": "BUY", "ml_score": ms,
            "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
            "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": ms * 2.0, "return_pct": 5.0,
        })
    eval_base = base + timedelta(days=n_train + 10)
    for j in range(n_eval):
        rows.append({
            "ticker": ["LITE", "SOXL", "TQQQ"][j % 3],
            "sim_date": (eval_base + timedelta(days=j)).isoformat(),
            "action": "BUY", "ml_score": 40.0 + (j % 7) * 5.0,
            "rsi": 95.0, "macd": 5.0, "mom5": 40.0, "mom20": 60.0,
            "regime_mult": 0.3, "vol_ratio": 4.9, "bb_position": 1.9,
            "news_urgency": None, "news_article_count": None,
            "forward_return_5d": 0.0, "return_pct": -50.0,
        })
    return rows


# ───────────────────────── core verdicts ─────────────────────────

def test_strong_signal_is_arm_stable(tmp_path):
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(400))
    rep = gs.analyze(p, oos_only=True, n_bootstrap=8, seed=42)
    assert rep["status"] == "ok", rep
    assert rep["verdict"] == "GATE_ARM_STABLE", rep
    assert rep["gate_arm_flip_rate"] is not None
    assert rep["gate_arm_flip_rate"] <= gs.STABLE_TOL, rep
    assert rep["n_bootstrap"] == 8
    assert rep["slice"] == "oos"
    # A strong recovered signal ⇒ the bootstraps largely agree per row.
    assert rep["mean_modal_agreement"] >= 0.75, rep


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_noisy_signal_is_arm_unstable(tmp_path):
    """A weak signal swamped by label noise ⇒ the gate's bucket decision is
    resample luck. This is the live scorer's documented regime; the tool
    MUST flag it GATE_ARM_UNSTABLE, not silently read it as fine."""
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _noisy_signal_rows(360))
    rep = gs.analyze(p, oos_only=True, n_bootstrap=8, seed=42)
    assert rep["status"] == "ok", rep
    assert rep["verdict"] == "GATE_ARM_UNSTABLE", rep
    assert rep["gate_arm_flip_rate"] >= gs.UNSTABLE_TOL, rep
    # Noisy bootstrap dispersion ⇒ a materially large cross-bootstrap σ
    # (the documented LITE volatility, here ≈ 3-4pp).
    assert rep["mean_pred_std"] is not None and rep["mean_pred_std"] > 1.0
    # And the per-row votes are genuinely split, not unanimous.
    assert rep["mean_modal_agreement"] < 1.0, rep


def test_consistent_offdist_extrapolation_is_not_flagged(tmp_path):
    """Non-obvious correctness lock: off-distribution *magnitude* alone is
    NOT instability — when every bootstrap extrapolates the SAME way
    (consistently clamped to one extreme arm), flip_rate is 0 and the
    verdict is STABLE. Guards against a future change that naively equates
    'clamped/off-distribution' with 'unstable'."""
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _consistent_offdist_rows(n_train=320, n_eval=80))
    rep = gs.analyze(p, oos_only=True, n_bootstrap=8, seed=42)
    assert rep["status"] == "ok", rep
    assert rep["verdict"] == "GATE_ARM_STABLE", rep
    assert rep["gate_arm_flip_rate"] == 0.0, rep
    # All eval rows extrapolate into a single (clamped) arm.
    assert sum(rep["arm_distribution"].values()) == rep["n_eval"]
    assert len(rep["arm_distribution"]) == 1, rep


# ───────────────────────── data gating ─────────────────────────

def test_insufficient_data_below_threshold(tmp_path):
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(gs.MIN_TRAIN + gs.MIN_EVAL - 5))
    rep = gs.analyze(p, n_bootstrap=8)
    assert rep["verdict"] == "INSUFFICIENT_DATA"
    assert rep["status"] == "error"
    assert "need ≥" in rep["hint"]


def test_too_few_bootstraps_is_insufficient(tmp_path):
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(400))
    rep = gs.analyze(p, n_bootstrap=gs.MIN_BOOTSTRAP - 1)
    assert rep["verdict"] == "INSUFFICIENT_DATA"
    assert "n_bootstrap" in rep["hint"]


def test_missing_file_is_insufficient_not_crash(tmp_path):
    rep = gs.analyze(tmp_path / "does_not_exist.jsonl", n_bootstrap=8)
    assert rep["verdict"] == "INSUFFICIENT_DATA"
    assert rep["status"] == "error"


# ───────────────────────── reproducibility ─────────────────────────

def test_determinism_same_seed_same_metrics(tmp_path):
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(400))
    a = gs.analyze(p, oos_only=True, n_bootstrap=6, seed=42)
    b = gs.analyze(p, oos_only=True, n_bootstrap=6, seed=42)
    for k in ("verdict", "gate_arm_flip_rate", "mean_pred_std",
              "median_pred_std", "mean_modal_agreement", "n_eval",
              "n_train", "n_bootstrap", "arm_distribution"):
        assert a[k] == b[k], (k, a[k], b[k])


# ───────────────────────── safety invariants ─────────────────────────

def test_does_not_write_scorer_pkl(tmp_path, monkeypatch):
    """The load-bearing invariant: a stability AUDIT must never clobber the
    deployed pickle the live gate consumes (it must NOT call train_scorer)."""
    import paper_trader.ml.decision_scorer as ds
    fake_pkl = tmp_path / "decision_scorer.pkl"
    monkeypatch.setattr(ds, "SCORER_PATH", fake_pkl)
    assert not fake_pkl.exists()
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(400))
    rep = gs.analyze(p, n_bootstrap=6)
    assert rep["status"] == "ok"
    # analyze() fit 6 throwaway models — none may have been persisted.
    assert not fake_pkl.exists(), "gate_stability wrote the deployed pickle!"


def test_gate_arm_is_single_source_of_truth():
    """gate_stability must use gate_audit.gate_arm verbatim so the two
    diagnostics can never disagree about the live gate's boundaries."""
    assert gs.gate_arm is ga_gate_arm
    # Spot-check the exact boundary semantics it relies on.
    assert gs.gate_arm(-10.01)[0] == "strong_headwind"
    assert gs.gate_arm(-10.0)[0] == "mild_headwind"   # p == -10 → mild
    assert gs.gate_arm(0.0)[0] == "neutral"           # p == 0 → neutral
    assert gs.gate_arm(5.0)[0] == "neutral"           # p == 5 → neutral
    assert gs.gate_arm(10.0)[0] == "mild_tailwind"    # p == 10 → mild
    assert gs.gate_arm(10.01)[0] == "strong_tailwind"
    assert gs.gate_arm(float("nan"))[0] == "neutral"


def test_never_raises_on_garbage_records(tmp_path):
    p = tmp_path / "oc.jsonl"
    rows = _strong_signal_rows(380)
    # Inject malformed rows: null/missing/non-finite fields.
    rows += [
        {"ticker": None, "sim_date": None, "action": None,
         "ml_score": None, "forward_return_5d": None, "return_pct": None},
        {"ticker": "X", "sim_date": "2025-06-01", "action": "BUY",
         "ml_score": "garbage", "forward_return_5d": float("inf"),
         "return_pct": float("nan"), "rsi": "bad"},
        {},
    ]
    _write_outcomes(p, rows)
    rep = gs.analyze(p, n_bootstrap=6, seed=42)   # must not raise
    assert isinstance(rep, dict)
    assert rep["status"] in ("ok", "error")
    assert rep["verdict"] in (
        "GATE_ARM_STABLE", "GATE_ARM_BORDERLINE",
        "GATE_ARM_UNSTABLE", "INSUFFICIENT_DATA",
    )


def test_all_slice_uses_full_set(tmp_path):
    p = tmp_path / "oc.jsonl"
    _write_outcomes(p, _strong_signal_rows(400))
    rep = gs.analyze(p, oos_only=False, n_bootstrap=6, seed=42)
    assert rep["slice"] == "all"
    assert rep["status"] == "ok"
    # --all evaluates on the full record set, not just the 20% OOS tail.
    assert rep["n_eval"] == 400


def test_main_cli_formats_and_exit_codes(monkeypatch, capsys):
    """main() must print the report and map ONLY GATE_ARM_UNSTABLE → exit 2
    (the cron-branchable contract), every other verdict → 0."""
    canned = {
        "status": "ok", "verdict": "GATE_ARM_STABLE", "slice": "oos",
        "n_eval": 50, "n_train": 200, "n_bootstrap": 10,
        "gate_arm_flip_rate": 0.04, "mean_pred_std": 0.9,
        "median_pred_std": 0.7, "mean_modal_agreement": 0.97,
        "arm_distribution": {"neutral": 50}, "hint": "stable",
    }
    monkeypatch.setattr(gs, "analyze", lambda *a, **k: dict(canned))
    rc = gs.main(["--bootstraps", "6"])
    out = capsys.readouterr().out
    assert "conviction-gate ARM stability" in out
    assert "GATE_ARM_STABLE" in out
    assert rc == 0

    canned["verdict"] = "GATE_ARM_UNSTABLE"
    monkeypatch.setattr(gs, "analyze", lambda *a, **k: dict(canned))
    assert gs.main([]) == 2          # the bad verdict → exit 2

    canned["verdict"] = "GATE_ARM_BORDERLINE"
    monkeypatch.setattr(gs, "analyze", lambda *a, **k: dict(canned))
    assert gs.main([]) == 0          # borderline is NOT the exit-2 trigger


def test_main_all_flag_selects_full_slice(monkeypatch):
    """`--all` must flip oos_only=False (the documented in-sample slice)."""
    seen = {}

    def _spy(path, oos_only=True, n_bootstrap=gs.DEFAULT_BOOTSTRAP, **k):
        seen["oos_only"] = oos_only
        seen["n_bootstrap"] = n_bootstrap
        return {"status": "ok", "verdict": "GATE_ARM_STABLE", "slice": "all",
                "n_eval": 1, "n_train": 1, "n_bootstrap": n_bootstrap,
                "gate_arm_flip_rate": 0.0, "mean_pred_std": 0.0,
                "median_pred_std": 0.0, "mean_modal_agreement": 1.0,
                "arm_distribution": {}, "hint": ""}

    monkeypatch.setattr(gs, "analyze", _spy)
    gs.main(["--all", "--bootstraps", "16"])
    assert seen["oos_only"] is False
    assert seen["n_bootstrap"] == 16
