"""Tests for paper_trader.ml.scorer_learning_curve.

The PURPOSE of these tests is to catch bugs in the analyzer:

- ``analyze`` must NEVER touch the deployed ``decision_scorer.pkl`` (the
  ``train_scorer(path=...)`` redirect is load-bearing — if a future
  refactor removes the kwarg, the deployed pickle silently gets
  overwritten by every test/CLI run).
- The verdict ladder must read shape, not endpoints alone — a U-shaped
  curve must NOT classify as DEGRADING.
- The multi-seed mean must be the published value (a noisy
  single-seed inversion at n=250 must not flip the verdict).
- The holdout must be CHRONOLOGICAL (sim_date-sorted), not append order
  — otherwise rank-IC would be evaluated on rows the model has already
  seen via the trailing-tail slice.
- The CLI exit code must distinguish "useful verdict" (0) from
  "INSUFFICIENT_DATA" (1) so a shell consumer can gate on $?.
"""
from __future__ import annotations

import json
import os
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import paper_trader.ml.decision_scorer as ds_mod
import paper_trader.ml.scorer_learning_curve as slc


# ──────────────────────────── helpers ────────────────────────────


def _synth_records(n: int, *, seed: int = 0,
                   start_date: date = date(2024, 1, 1),
                   leakage: float = 0.0) -> list[dict]:
    """Generate ``n`` synthetic outcome records with a controllable
    signal-to-noise ratio.

    Each record's ``forward_return_5d`` is ``leakage * ml_score + noise``,
    so:
      * ``leakage=0.0`` → pure noise corpus (model should NOT learn)
      * ``leakage=0.5`` → mild signal (rank-IC ~ 0.3+ at n>=500)
      * ``leakage=1.0`` → strong signal (rank-IC ~ 0.7+ at n>=500)

    Records are emitted in chronological order with a small RNG-driven
    gap so ``split_outcomes_temporal`` produces a clean holdout. Quant
    fields are filled with plausible random values so train_scorer does
    NOT drop any row for missing/non-finite labels.
    """
    rng = random.Random(seed)
    tickers = ["NVDA", "AMD", "MSFT", "AAPL", "META", "GOOGL", "TSLA",
               "SPY", "QQQ", "SOXL", "TQQQ", "XLF", "XLE", "GLD", "MU"]
    out: list[dict] = []
    for i in range(n):
        ml = rng.uniform(-3.0, 3.0)
        noise = rng.gauss(0.0, 3.0)
        fr = leakage * (ml * 5.0) + noise
        out.append({
            "ticker": rng.choice(tickers),
            "sim_date": (start_date + timedelta(days=i // 4)).isoformat(),
            "action": "BUY",
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


@pytest.fixture(autouse=True)
def _protect_deployed_pickle(monkeypatch, tmp_path):
    """Hard barrier: every test in this file gets a SCORER_PATH that
    points into tmp_path, so even a *bug* in analyze that bypasses the
    redirect cannot trample the deployed scorer.pkl.

    The analyzer's own discipline (train into temp dir, restore
    SCORER_PATH in finally) is tested EXPLICITLY in test_does_not_touch_deployed_pickle
    — this fixture is belt-and-braces, not a substitute.
    """
    sentinel = tmp_path / "sentinel_deployed.pkl"
    monkeypatch.setattr(ds_mod, "SCORER_PATH", sentinel)
    return sentinel


# ──────────────────────────── INSUFFICIENT_DATA ────────────────────────────


class TestInsufficientData:
    def test_empty_records(self):
        rep = slc.analyze([], seeds=1)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["curve"] == []
        assert rep["n_total"] == 0

    def test_below_min_total(self):
        recs = _synth_records(50, leakage=0.5)
        rep = slc.analyze(recs, seeds=1)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "need >=" in (rep.get("hint") or "")

    def test_missing_jsonl_path(self, tmp_path):
        rep = slc.analyze(tmp_path / "does_not_exist.jsonl", seeds=1)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 0

    def test_corrupt_jsonl_lines_skip_not_crash(self, tmp_path):
        # File with a few good rows + corrupt ones — corrupt should be
        # silently skipped, good should populate the corpus. Below MIN_TOTAL.
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in _synth_records(40, leakage=0.5):
                fh.write(json.dumps(r) + "\n")
            fh.write("not json\n")
            fh.write("{invalid}\n")
            fh.write("\n")
        rep = slc.analyze(p, seeds=1)
        # 40 < MIN_TOTAL ⇒ INSUFFICIENT, but loader survived corruption.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == 40


# ──────────────────────────── verdict ladder ────────────────────────────


class TestVerdictLadder:
    def test_no_skill_on_pure_noise(self):
        """With zero leakage every n should yield near-zero rank-IC →
        NO_SKILL. This is the most important sanity check: the analyzer
        must NOT report MONOTONE_LEARNING on a noise corpus.
        """
        recs = _synth_records(1500, leakage=0.0, seed=7)
        rep = slc.analyze(recs, seeds=2,
                          ladder=(200, 400, 800))
        assert rep["verdict"] == "NO_SKILL"
        # Every mean rank-IC should sit below MIN_SKILL_IC.
        for row in rep["curve"]:
            if row.get("mean_rank_ic") is not None:
                assert abs(row["mean_rank_ic"]) < slc.MIN_SKILL_IC

    def test_no_skill_verdict_ladder_pure_synthetic(self):
        """Unit-level: build a synthetic curve with all near-zero ICs
        and confirm _verdict returns NO_SKILL — independent of training,
        so a future training-pipeline regression cannot mask the
        verdict-logic correctness."""
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": 0.01,
             "std_rank_ic": 0.02, "mean_dir_acc": 0.5},
            {"train_n_requested": 500, "mean_rank_ic": -0.02,
             "std_rank_ic": 0.03, "mean_dir_acc": 0.5},
            {"train_n_requested": 1000, "mean_rank_ic": 0.01,
             "std_rank_ic": 0.02, "mean_dir_acc": 0.5},
        ]
        assert slc._verdict(curve) == "NO_SKILL"

    def test_monotone_learning_verdict(self):
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": 0.10},
            {"train_n_requested": 500, "mean_rank_ic": 0.20},
            {"train_n_requested": 1000, "mean_rank_ic": 0.30},
        ]
        assert slc._verdict(curve) == "MONOTONE_LEARNING"

    def test_degrading_verdict(self):
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": 0.25},
            {"train_n_requested": 500, "mean_rank_ic": 0.20},
            {"train_n_requested": 1000, "mean_rank_ic": 0.15},
        ]
        assert slc._verdict(curve) == "DEGRADING"

    def test_u_shaped_not_misread_as_degrading(self):
        """A curve that peaks in the middle (250 → 500 up, 500 → 1000
        down) must be classified U_SHAPED — NOT DEGRADING. This is the
        verdict-shape distinction the analyzer adds over a naive
        end-vs-start rule.
        """
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": 0.10},
            {"train_n_requested": 500, "mean_rank_ic": 0.35},
            {"train_n_requested": 1000, "mean_rank_ic": 0.18},
        ]
        assert slc._verdict(curve) == "U_SHAPED"

    def test_saturated_verdict(self):
        """Curve that is positive and roughly flat — no learning gain
        from more data but the model has SOME skill. SATURATED, not
        NO_SKILL, not MONOTONE_LEARNING."""
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": 0.20},
            {"train_n_requested": 500, "mean_rank_ic": 0.21},
            {"train_n_requested": 1000, "mean_rank_ic": 0.20},
        ]
        assert slc._verdict(curve) == "SATURATED"

    def test_one_point_curve_is_insufficient(self):
        curve = [{"train_n_requested": 500, "mean_rank_ic": 0.25}]
        assert slc._verdict(curve) == "INSUFFICIENT_DATA"

    def test_none_rank_ic_excluded(self):
        """A curve where some rungs have None rank-IC (training failed)
        — the verdict reads only the rungs with valid IC."""
        curve = [
            {"train_n_requested": 250, "mean_rank_ic": None},
            {"train_n_requested": 500, "mean_rank_ic": 0.20},
            {"train_n_requested": 1000, "mean_rank_ic": 0.25},
        ]
        # 2 valid points, last > first by 0.05 = LEARNING_DELTA — borderline.
        # Either MONOTONE_LEARNING (delta exactly meets bar) or SATURATED.
        # The contract: NOT INSUFFICIENT_DATA — 2+ valid points exist.
        assert slc._verdict(curve) != "INSUFFICIENT_DATA"


# ──────────────────────────── deployed-pickle isolation ────────────────────────────


class TestDoesNotTouchDeployedPickle:
    def test_analyze_leaves_deployed_path_unwritten(
            self, _protect_deployed_pickle, tmp_path):
        """The deployed SCORER_PATH (sentinel under tmp_path) MUST NOT
        be created or modified by an analyze() call. If a future
        refactor removes the train_scorer(path=...) override or breaks
        the SCORER_PATH save/restore in finally, this test fails loudly.
        """
        sentinel = _protect_deployed_pickle
        # Ensure deployed path does not pre-exist.
        assert not sentinel.exists()
        recs = _synth_records(1000, leakage=0.5, seed=11)
        slc.analyze(recs, seeds=1, ladder=(250, 500))
        # After a complete training sweep, the SENTINEL must still be missing.
        assert not sentinel.exists(), (
            "scorer_learning_curve trained into the deployed SCORER_PATH — "
            "the isolation layer is broken")

    def test_scorer_path_restored_even_on_exception(
            self, _protect_deployed_pickle, monkeypatch):
        """If train_scorer raises mid-rung, the SCORER_PATH module
        global MUST be restored to the original value (the
        try/finally contract). A leaked-temp path would cause every
        subsequent DecisionScorer() call to load garbage / nothing.
        """
        original = ds_mod.SCORER_PATH

        def _boom(*a, **kw):
            raise RuntimeError("simulated training failure")

        monkeypatch.setattr(slc, "train_scorer", _boom)
        # Call the lowest-level training+score wrapper directly so the
        # exception path is exercised.
        ic, da, n, n_train = slc._train_and_score_once(
            _synth_records(100), _synth_records(50), seed=42)
        assert ic is None
        # The deployed SCORER_PATH must be the original sentinel.
        assert ds_mod.SCORER_PATH is original

    def test_mlp_config_random_state_restored(
            self, _protect_deployed_pickle):
        """Each seed's training MUTATES MLP_CONFIG['random_state']; the
        finally must restore the original so unrelated callers (the
        continuous loop's _train_decision_scorer) keep their deterministic
        seed=42 behavior."""
        original_seed = ds_mod.MLP_CONFIG["random_state"]
        recs_train = _synth_records(200, leakage=0.5)
        recs_oos = _synth_records(50, leakage=0.5, seed=99)
        slc._train_and_score_once(recs_train, recs_oos, seed=123)
        assert ds_mod.MLP_CONFIG["random_state"] == original_seed


# ──────────────────────────── chronological holdout ────────────────────────────


class TestChronologicalHoldout:
    def test_holdout_is_sim_date_sorted_not_append_order(self):
        """Build records where the APPEND order is reverse-chronological
        (simulating multiple cycles whose oldest sim_date arrives last).
        The holdout must be the LATEST sim_dates, not the LAST APPENDED."""
        recs = _synth_records(200, leakage=0.5)
        # Reverse so the LATEST sim_dates are FIRST in the list.
        recs_reversed = list(reversed(recs))
        in_sample, holdout = slc._holdout_split(recs_reversed, oos_fraction=0.2)
        # Holdout's sim_dates must all be >= max(in_sample sim_dates).
        in_sample_max = max(r["sim_date"] for r in in_sample)
        for r in holdout:
            assert r["sim_date"] >= in_sample_max

    def test_holdout_empty_when_records_below_5(self):
        """split_outcomes_temporal degrades empty holdout when <5 rows —
        analyzer treats this as INSUFFICIENT_DATA upstream, but the
        primitive's contract is documented here."""
        in_sample, holdout = slc._holdout_split([{"sim_date": "2024-01-01"}])
        assert holdout == []


# ──────────────────────────── full integration ────────────────────────────


class TestFullIntegration:
    def test_strong_signal_yields_curve_with_positive_ic(self):
        """With a strong leaked signal the rank-IC at the largest rung
        should be clearly positive (> 0.15). This is the END-TO-END
        smoke that the pipeline (train → load tmp pickle → predict on
        holdout → compute Spearman) wires together correctly.
        """
        recs = _synth_records(1500, leakage=0.7, seed=2026)
        rep = slc.analyze(recs, seeds=2, ladder=(300, 700))
        assert rep["verdict"] != "INSUFFICIENT_DATA"
        # At least one rung must produce a positive mean rank-IC > 0.10.
        ics = [row.get("mean_rank_ic") for row in rep["curve"]
               if row.get("mean_rank_ic") is not None]
        assert ics, "no rung produced a rank-IC"
        assert max(ics) > 0.10, (
            f"strong-signal corpus produced max rank-IC {max(ics):.3f} "
            f"(expected > 0.10) — pipeline likely mis-wired")

    def test_curve_train_n_actual_below_or_equal_requested(self):
        """train_scorer dedups records before fit, so train_n_actual
        must be <= train_n_requested. A curve where actual > requested
        signals a dedup bypass (a real bug — accumulated history would
        be leaked into the small-n point)."""
        recs = _synth_records(1200, leakage=0.5)
        rep = slc.analyze(recs, seeds=1, ladder=(300, 600))
        for row in rep["curve"]:
            assert row["train_n_actual"] <= row["train_n_requested"], (
                f"train_n_actual {row['train_n_actual']} > requested "
                f"{row['train_n_requested']} — dedup may be bypassed")

    def test_returns_structured_dict_keys(self):
        """The dict shape is part of the contract every consumer
        (CLI, future ledger wiring) relies on; pin it."""
        recs = _synth_records(900, leakage=0.5)
        rep = slc.analyze(recs, seeds=1, ladder=(300, 600))
        assert set(rep.keys()) >= {
            "verdict", "curve", "n_total", "n_holdout", "ladder",
            "seeds", "holdout_fraction"}
        for row in rep["curve"]:
            assert set(row.keys()) >= {
                "train_n_requested", "train_n_actual", "mean_rank_ic",
                "std_rank_ic", "mean_dir_acc", "n_oos_pairs",
                "n_successful_seeds"}


# ──────────────────────────── CLI exit code ────────────────────────────


class TestCLIExitCode:
    def test_exit_1_on_insufficient(self, tmp_path, capsys):
        empty = tmp_path / "empty.jsonl"
        empty.touch()
        rc = slc.main(["--outcomes", str(empty), "--seeds", "1"])
        assert rc == 1

    def test_exit_0_on_useful_verdict(self, tmp_path, capsys):
        # Write the synthetic corpus to a JSONL file the CLI consumes.
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in _synth_records(1200, leakage=0.0, seed=99):
                fh.write(json.dumps(r) + "\n")
        rc = slc.main(["--outcomes", str(p), "--seeds", "1",
                       "--holdout-fraction", "0.2"])
        # NO_SKILL is a useful (decisive) verdict — exit 0.
        assert rc == 0
        out = capsys.readouterr().out
        assert "verdict" in out


# ──────────────────────────── train_scorer path kwarg ────────────────────────────


class TestTrainScorerPathKwarg:
    """The refactor that makes the analyzer possible: train_scorer must
    accept a ``path`` kwarg that overrides the deployed SCORER_PATH. If a
    refactor breaks this contract, the analyzer trains into the live
    pickle and corrupts the deployed gate.
    """

    def test_path_kwarg_writes_to_override(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer

        custom = tmp_path / "custom.pkl"
        recs = _synth_records(100, leakage=0.5)
        result = train_scorer(recs, path=custom)
        assert result["status"] == "ok"
        # The override must hold the new pickle...
        assert custom.exists()
        # ...and the deployed SCORER_PATH must NOT be touched (the
        # _protect_deployed_pickle fixture set it to tmp_path/sentinel).
        assert not ds_mod.SCORER_PATH.exists()

    def test_path_default_falls_back_to_module_scorer_path(self, tmp_path):
        """Without ``path``, train_scorer writes to the module-level
        SCORER_PATH — preserving the existing contract for the
        continuous loop and every legacy caller."""
        from paper_trader.ml.decision_scorer import train_scorer

        recs = _synth_records(100, leakage=0.5)
        result = train_scorer(recs)
        assert result["status"] == "ok"
        # The deployed sentinel (patched by the fixture) IS the path used.
        assert ds_mod.SCORER_PATH.exists()
