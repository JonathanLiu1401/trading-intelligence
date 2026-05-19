"""Tests for paper_trader.ml.scorer_freshness — the DecisionScorer training-
pipeline liveness/staleness monitor. Until now the module had zero direct
test coverage; these lock its verdict ladder so a refactor cannot silently
mute a real failure.

Focused on the documented verdicts the live unattended loop relies on:
  * FRESH               — heartbeat alive, pkl reflects the last logged retrain
  * INSUFFICIENT_DATA   — no skill-log / no pkl yet
  * STALE_PKL           — pkl mtime predates the heartbeat by > grace
  * PKL_REGRESSED       — pkl n_train ≪ last logged train_n (NEW: catches the
                          observed-live "production pkl carries n_train=400
                          while the loop logged train_n=3959" clobber)
  * LOOP_STALLED        — heartbeat older than STALE_HEARTBEAT_H
  * LOOP_DEAD           — heartbeat older than DEAD_HEARTBEAT_H
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.ml import scorer_freshness as sf


# ─────────────────────────── helpers ───────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_skill_log(path, rows: list[dict]) -> None:
    """Write one JSONL row per dict; mirrors the loop's append pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class _FakeModel:
    """Pickleable stand-in for sklearn's MLPRegressor used in the freshness
    `_pkl_n_train` reader. The reader only checks `DecisionScorer.n_train`
    so we don't need a real fit."""

    def predict(self, X):
        return [0.0] * len(X)


def _write_pkl(path, n_train: int, mtime_offset_h: float = 0.0) -> None:
    """Write a minimal valid scorer pickle with `n_train` set, optionally
    backdated by `mtime_offset_h` hours from now (positive → older)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": _FakeModel(), "scaler": None,
                     "n_train": int(n_train)}, f)
    if mtime_offset_h:
        target = time.time() - mtime_offset_h * 3600.0
        import os
        os.utime(path, (target, target))


@pytest.fixture
def _patched_paths(tmp_path, monkeypatch):
    """Redirect the three on-disk artifacts the freshness report reads."""
    skill = tmp_path / "data" / "scorer_skill_log.jsonl"
    outcomes = tmp_path / "data" / "decision_outcomes.jsonl"
    pkl = tmp_path / "data" / "ml" / "decision_scorer.pkl"
    monkeypatch.setattr(sf, "SKILL_LOG", skill)
    monkeypatch.setattr(sf, "OUTCOMES", outcomes)
    monkeypatch.setattr(sf, "PKL", pkl)
    # Also redirect the decision_scorer module path so DecisionScorer reads
    # the same pkl the freshness module sees.
    import paper_trader.ml.decision_scorer as ds
    monkeypatch.setattr(ds, "SCORER_PATH", pkl)
    if hasattr(ds, "_LOAD_CACHE"):
        ds._LOAD_CACHE.clear()
    return skill, outcomes, pkl


# ─────────────────────────── INSUFFICIENT_DATA / FRESH ──────────────────


class TestInsufficientData:
    def test_no_skill_log_no_pkl(self, _patched_paths):
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["skill_log_present"] is False
        assert rep["pkl_present"] is False

    def test_pkl_only_no_skill_log(self, _patched_paths):
        _, _, pkl = _patched_paths
        _write_pkl(pkl, n_train=1000)
        rep = sf.scorer_freshness_report()
        # pkl present but heartbeat unverifiable.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["pkl_present"] is True
        assert rep["pkl_n_train"] == 1000


class TestFresh:
    def test_recent_heartbeat_with_matching_pkl_n_train(self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Heartbeat 1 hour ago — well within grace.
        _write_skill_log(skill, [{
            "cycle": 42,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=4000)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "FRESH", rep
        assert rep["last_cycle"] == 42
        assert rep["pkl_n_train"] == 4000
        assert rep["heartbeat_age_h"] is not None
        assert 0 < rep["heartbeat_age_h"] < 2


# ─────────────────────────── LOOP_STALLED / LOOP_DEAD ────────────────────


class TestLoopHeartbeat:
    def test_loop_stalled_when_heartbeat_between_stale_and_dead(
            self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Heartbeat 10 hours ago: > STALE_HEARTBEAT_H=6h, < DEAD_HEARTBEAT_H=24h.
        _write_skill_log(skill, [{
            "cycle": 100,
            "timestamp": (_now() - timedelta(hours=10)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=4000)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "LOOP_STALLED", rep
        # The hint should mention the gate is ACTIVE — escalates the alarm.
        assert "gate is ACTIVE" in rep["hint"]

    def test_loop_dead_when_heartbeat_older_than_dead_threshold(
            self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Heartbeat 30 hours ago: > DEAD_HEARTBEAT_H=24h.
        _write_skill_log(skill, [{
            "cycle": 7,
            "timestamp": (_now() - timedelta(hours=30)).isoformat(),
            "train_n": 4000,
            "gate_active": False,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=4000)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "LOOP_DEAD", rep
        # Hint should NOT escalate gate-active warning when gate is inactive.
        assert "gate is ACTIVE" not in rep["hint"]


# ─────────────────────────── STALE_PKL ───────────────────────────────────


class TestStalePkl:
    def test_pkl_older_than_heartbeat_by_more_than_grace(self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Heartbeat 1 hour ago, pkl 10 hours ago: lag = 9h > PKL_LAG_GRACE_H=4h.
        _write_skill_log(skill, [{
            "cycle": 10,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=4000, mtime_offset_h=10.0)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "STALE_PKL", rep

    def test_pkl_lag_within_grace_does_not_trigger(self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Heartbeat 1 hour ago, pkl 2 hours ago: lag = 1h ≤ PKL_LAG_GRACE_H=4h.
        _write_skill_log(skill, [{
            "cycle": 11,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=4000, mtime_offset_h=2.0)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "FRESH", rep


# ─────────────────────────── PKL_REGRESSED (new) ─────────────────────────


class TestPklRegressed:
    """Locks the new verdict catching the observed-live failure mode:
    deployed pkl was clobbered with a tiny-corpus fit (n_train=400) while
    the loop's own ledger logged the most recent retrain at train_n=3959.
    STALE_PKL does NOT fire because pkl mtime is newer than the heartbeat
    — the gap is detected purely by comparing the two `n_train` values."""

    def test_pkl_regression_at_observed_ratio(self, _patched_paths):
        skill, _, pkl = _patched_paths
        # Exact production state: last logged train_n=3959, pkl n_train=400,
        # heartbeat fresh, pkl mtime newer than heartbeat.
        _write_skill_log(skill, [{
            "cycle": 4,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 3959,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=400)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "PKL_REGRESSED", rep
        # The hint should report both numbers so an operator can act on it.
        assert "n_train=400" in rep["hint"]
        assert "train_n=3959" in rep["hint"]

    def test_normal_train_n_wander_does_not_trigger(self, _patched_paths):
        """Across cycles the loop's `train_n` legitimately swings ±10% as the
        rolling tail of outcomes deduplicates differently — that is NOT a
        regression and must NOT fire. last_train_n=4000, pkl=3500 ⇒ ratio
        0.875, above the 0.5 tolerance — FRESH."""
        skill, _, pkl = _patched_paths
        _write_skill_log(skill, [{
            "cycle": 50,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=3500)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "FRESH", rep

    def test_below_min_train_n_floor_does_not_trigger(self, _patched_paths):
        """A tiny logged train_n (below `PKL_REGRESSION_MIN_TRAIN_N=500`)
        is the early-cycle regime where a swing carries no real signal —
        the regression check must be muted there. last_train_n=400 (matches
        an early cycle), pkl=10 ⇒ would technically be a 25× regression but
        below the absolute floor, so FRESH."""
        skill, _, pkl = _patched_paths
        _write_skill_log(skill, [{
            "cycle": 1,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 400,
            "gate_active": False,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=10)
        rep = sf.scorer_freshness_report()
        # Must not fire PKL_REGRESSED because 400 < PKL_REGRESSION_MIN_TRAIN_N.
        assert rep["verdict"] == "FRESH", rep

    def test_loop_dead_takes_precedence_over_pkl_regressed(
            self, _patched_paths):
        """Verdict priority: a dead loop is the more urgent signal, so
        LOOP_DEAD must fire even if pkl_n_train also regressed (the same
        precedence STALE_PKL has — the heartbeat ladder is checked first)."""
        skill, _, pkl = _patched_paths
        _write_skill_log(skill, [{
            "cycle": 9,
            "timestamp": (_now() - timedelta(hours=48)).isoformat(),
            "train_n": 4000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=200)
        rep = sf.scorer_freshness_report()
        assert rep["verdict"] == "LOOP_DEAD", rep


# ─────────────────────────── verdict ladder invariants ───────────────────


class TestVerdictLadder:
    def test_every_verdict_is_listed_in_module_public_constant(self):
        """`VERDICTS` is a public list of every possible verdict; the test
        catches a typo / missing entry in any of the emission paths."""
        # The set of strings the module is known to emit — derived from the
        # docstring ladder.
        expected = {
            "FRESH", "INSUFFICIENT_DATA",
            "STALE_PKL", "PKL_REGRESSED",
            "LOOP_STALLED", "LOOP_DEAD",
        }
        assert set(sf.VERDICTS) == expected

    def test_cli_exit_code_matches_verdict_severity(
            self, _patched_paths, capsys):
        """The CLI's exit code is the operator-facing contract: 0 only for
        FRESH / INSUFFICIENT_DATA, 2 for every actionable verdict."""
        skill, _, pkl = _patched_paths
        # Provoke PKL_REGRESSED.
        _write_skill_log(skill, [{
            "cycle": 5,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 3000,
            "gate_active": True,
            "status": "ok",
        }])
        _write_pkl(pkl, n_train=100)
        rc = sf._cli()
        assert rc == 2  # PKL_REGRESSED → non-zero exit


# ─────────────────────────── robustness ──────────────────────────────────


class TestRobustness:
    def test_unparseable_skill_log_lines_are_skipped(self, _patched_paths):
        """A torn JSONL line (a process killed mid-write would emit a
        partial line) must NOT crash the report — `_last_skill_row` skips
        unparseable lines and takes the newest valid one."""
        skill, _, pkl = _patched_paths
        skill.parent.mkdir(parents=True, exist_ok=True)
        with skill.open("w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({
                "cycle": 1,
                "timestamp": (_now() - timedelta(hours=1)).isoformat(),
                "train_n": 4000,
                "gate_active": False,
                "status": "ok",
            }) + "\n")
            f.write("{also broken\n")
        _write_pkl(pkl, n_train=4000)
        rep = sf.scorer_freshness_report()
        # Should pick up the middle valid row → FRESH.
        assert rep["verdict"] == "FRESH"
        assert rep["last_cycle"] == 1

    def test_corrupt_pkl_does_not_crash(self, _patched_paths):
        """A torn pkl (process killed mid-pickle.dump) must NOT raise from
        the freshness report — DecisionScorer's `_load` swallows the
        unpickle error and the n_train accessor falls back to its `__init__`
        default of 0. The verdict must still be a well-defined enum member."""
        skill, _, pkl = _patched_paths
        _write_skill_log(skill, [{
            "cycle": 1,
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "train_n": 4000,
            "gate_active": False,
            "status": "ok",
        }])
        pkl.parent.mkdir(parents=True, exist_ok=True)
        pkl.write_bytes(b"not a pickle file")
        rep = sf.scorer_freshness_report()
        # No crash — pkl_n_train degrades to 0 (DecisionScorer's
        # _load() exception path leaves _n_train at its 0 default).
        assert rep["pkl_n_train"] in (0, None)
        # Verdict should still be something sensible — not crash.
        assert rep["verdict"] in sf.VERDICTS
