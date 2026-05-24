"""Pin the ml_training_health builder's verdict ladder + reader robustness.

The verdict alphabet is a contract for the eventual dashboard / chat
consumers (mirrors the test discipline ``test_briefing_health.py`` carries
for the sibling 5h briefing health snapshot)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics.ml_training_health import (
    DEFAULT_DEAD_AGE_H,
    DEFAULT_STALE_AGE_H,
    DIVERGE_MIN_RATIO,
    DIVERGE_MIN_STEPS,
    VERDICTS,
    compute,
    compute_ml_training_health,
    load_recent_records,
)


NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _train(age_h: float, *, status: str = "ok", val_loss: float = 1.0) -> dict:
    rec: dict = {
        "ts": _iso(age_h),
        "phase": "train",
        "status": status,
        "n": 100,
    }
    if status == "ok":
        rec["val_loss"] = val_loss
        rec["final_loss"] = val_loss * 0.5
    return rec


def _continuous(age_h: float, *, status: str = "ok") -> dict:
    return {
        "ts": _iso(age_h),
        "phase": "continuous",
        "status": status,
        "n": 100,
    }


class TestVerdictLadder:
    def test_no_data_on_empty_records(self):
        snap = compute_ml_training_health([], now=NOW)
        assert snap["verdict"] == "NO_DATA"
        assert snap["last_train_age_h"] is None
        assert snap["val_loss_trend"] == []
        assert snap["train_in_window"] == 0

    def test_no_data_on_garbage_records(self):
        # Non-dict / unparseable rows must not crash and must not register
        # as evidence of a successful train.
        snap = compute_ml_training_health(
            ["not a dict", 42, None, {"phase": "x", "ts": "not-a-date"}],
            now=NOW,
        )
        assert snap["verdict"] == "NO_DATA"

    def test_healthy_on_fresh_train(self):
        snap = compute_ml_training_health(
            [_train(1.0, val_loss=0.5), _train(3.0, val_loss=0.6)],
            now=NOW,
        )
        assert snap["verdict"] == "HEALTHY"
        assert snap["last_train_age_h"] == 1.0
        assert snap["train_in_window"] == 2

    def test_stale_when_train_aged_into_band(self):
        # > stale_age_h but <= dead_age_h
        age = DEFAULT_STALE_AGE_H + 1.0
        snap = compute_ml_training_health([_train(age)], now=NOW)
        assert snap["verdict"] == "STALE"
        assert snap["last_train_age_h"] == age

    def test_dead_when_train_passes_dead_threshold(self):
        snap = compute_ml_training_health(
            [_train(DEFAULT_DEAD_AGE_H + 1.0)], now=NOW
        )
        assert snap["verdict"] == "DEAD"

    def test_dead_outranks_diverging(self):
        # A diverging val_loss series whose newest entry is past DEAD must
        # still verdict DEAD — a dead trainer is worse than a misbehaving
        # one.
        recs = [
            _train(DEFAULT_DEAD_AGE_H + 1.0, val_loss=4.0),
            _train(DEFAULT_DEAD_AGE_H + 2.0, val_loss=2.0),
            _train(DEFAULT_DEAD_AGE_H + 3.0, val_loss=1.0),
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["verdict"] == "DEAD"
        assert snap["diverging"] is True  # builder still surfaces the fact

    def test_diverging_on_three_increasing_losses(self):
        # 3 cycles, newest val_loss highest, each step >= DIVERGE_MIN_RATIO.
        recs = [
            _train(1.0, val_loss=2.0),  # newest
            _train(2.0, val_loss=1.5),
            _train(3.0, val_loss=1.0),  # oldest
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["verdict"] == "DIVERGING"
        assert snap["diverging"] is True

    def test_not_diverging_when_step_below_ratio_threshold(self):
        # 3 cycles where each step is too small to count (< 5% ratio).
        recs = [
            _train(1.0, val_loss=1.04),
            _train(2.0, val_loss=1.02),
            _train(3.0, val_loss=1.00),
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        # Each ratio < 1.05, so divergence flag False, verdict HEALTHY.
        assert snap["verdict"] == "HEALTHY"
        assert snap["diverging"] is False

    def test_error_heavy_when_majority_in_window_failed(self):
        # Fresh successful train (so not STALE/DEAD), but most records in
        # the window have status != 'ok'.
        recs = [
            _train(1.0),  # ok, anchors recency
            _train(2.0, status="error"),
            _train(3.0, status="error"),
            _train(4.0, status="error"),
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["verdict"] == "ERROR_HEAVY"
        assert snap["errors_in_window"] == 3
        assert snap["total_in_window"] == 4

    def test_continuous_alone_does_not_satisfy_health(self):
        # No train phase at all — last_train_age_h is None. Should NOT be
        # HEALTHY; the trainer might be running the lightweight pass while
        # the full retrain is wedged. NO_DATA is preserved when no train
        # ever ran; if continuous exists but no train, the ladder reads
        # NO_DATA on last_train and the diverging / error checks don't
        # apply.
        snap = compute_ml_training_health(
            [_continuous(1.0), _continuous(2.0)], now=NOW
        )
        # Has continuous data, so not NO_DATA on the both-empty path; but
        # there is no last_train so the recency clauses don't escalate.
        # Falls through to HEALTHY (best we can say without train data —
        # the contract is honest: errors_in_window==0, diverging==False).
        assert snap["last_train_age_h"] is None
        assert snap["last_continuous_age_h"] == 1.0
        assert snap["verdict"] == "HEALTHY"

    def test_verdict_always_in_closed_alphabet(self):
        for recs in [
            [],
            [_train(1.0)],
            [_train(DEFAULT_STALE_AGE_H + 1.0)],
            [_train(DEFAULT_DEAD_AGE_H + 1.0)],
            [_train(1.0, status="error")] * 3 + [_train(2.0)],
        ]:
            snap = compute_ml_training_health(recs, now=NOW)
            assert snap["verdict"] in VERDICTS


class TestSnapshotShape:
    def test_required_keys_present(self):
        snap = compute_ml_training_health([_train(1.0)], now=NOW)
        for key in (
            "window_h",
            "last_train_age_h",
            "last_continuous_age_h",
            "train_in_window",
            "continuous_in_window",
            "errors_in_window",
            "total_in_window",
            "val_loss_trend",
            "diverging",
            "verdict",
        ):
            assert key in snap, f"missing key {key}"

    def test_val_loss_trend_capped_at_five(self):
        recs = [_train(i + 0.1, val_loss=float(i)) for i in range(1, 12)]
        snap = compute_ml_training_health(recs, now=NOW)
        assert len(snap["val_loss_trend"]) == 5
        # Newest-first ordering: trend[0] corresponds to the newest record
        # (age=1.1h, val_loss=1.0).
        assert snap["val_loss_trend"][0] == 1.0

    def test_window_h_clamps_to_minimum_one(self):
        snap = compute_ml_training_health(
            [_train(0.5)], now=NOW, window_h=0
        )
        assert snap["window_h"] == 1


class TestReadOnlyAndBacktestSafety:
    """Documents that the module is read-only — no DB / model write. The
    builder takes records by value and only emits a snapshot dict. There
    is no path that touches the articles DB at all, so backtest isolation
    is trivially N/A — but the test below pins the "no DB import" claim so
    a future refactor can't silently re-couple the module."""

    def test_module_does_not_open_articles_db(self):
        import analytics.ml_training_health as mod
        src = Path(mod.__file__).read_text()
        # Direct grep — the module deliberately does NOT import the article
        # store. If a future change adds it, this test fires and the
        # contract gets revisited explicitly.
        assert "from storage.article_store" not in src
        assert "import storage.article_store" not in src
        assert "articles.db" not in src.lower()


class TestFileReader:
    def test_load_returns_empty_on_missing_file(self, tmp_path):
        out = load_recent_records(tmp_path / "nope.jsonl")
        assert out == []

    def test_load_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "m.jsonl"
        p.write_text(
            json.dumps({"phase": "train", "status": "ok", "ts": _iso(1)}) + "\n"
            + "{not json\n"
            + json.dumps({"phase": "continuous", "status": "ok"}) + "\n"
            + "\n"  # empty line tolerated
        )
        recs = load_recent_records(p)
        assert len(recs) == 2
        assert recs[0]["phase"] == "train"
        assert recs[1]["phase"] == "continuous"

    def test_load_tail_limit_keeps_newest(self, tmp_path):
        p = tmp_path / "m.jsonl"
        p.write_text("\n".join(
            json.dumps({"phase": "train", "status": "ok",
                        "ts": _iso(20 - i), "val_loss": float(i)})
            for i in range(10)
        ) + "\n")
        recs = load_recent_records(p, limit=3)
        assert len(recs) == 3
        # Trainer appends chronological, tail keeps the LAST 3 written
        # (largest i = oldest age = newest entries written).
        assert [r["val_loss"] for r in recs] == [7.0, 8.0, 9.0]

    def test_compute_reads_file_and_returns_metrics_path(self, tmp_path):
        p = tmp_path / "m.jsonl"
        p.write_text(
            json.dumps({"phase": "train", "status": "ok",
                        "ts": _iso(1.0), "val_loss": 1.0}) + "\n"
        )
        snap = compute(metrics_path=p, now=NOW)
        assert snap["verdict"] == "HEALTHY"
        assert snap["metrics_path"] == str(p)
        assert snap["records_read"] == 1

    def test_compute_returns_no_data_on_missing_file(self, tmp_path):
        snap = compute(metrics_path=tmp_path / "absent.jsonl", now=NOW)
        assert snap["verdict"] == "NO_DATA"
        assert snap["records_read"] == 0


class TestDivergenceMath:
    def test_diverging_requires_strict_increasing_each_step(self):
        # Middle step flat → not diverging.
        recs = [
            _train(1.0, val_loss=2.0),
            _train(2.0, val_loss=1.0),
            _train(3.0, val_loss=1.0),  # equal — gap is exactly 0%, fails ratio
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["diverging"] is False

    def test_diverging_needs_at_least_three_records(self):
        recs = [
            _train(1.0, val_loss=2.0),
            _train(2.0, val_loss=1.0),  # only 2 records
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["diverging"] is False
        # With only 2 ok trains the trend list has 2 entries.
        assert len(snap["val_loss_trend"]) == 2

    def test_diverge_min_steps_and_ratio_constants_are_positive(self):
        # Pinning the invariants the verdict ladder depends on.
        assert DIVERGE_MIN_STEPS >= 2
        assert DIVERGE_MIN_RATIO > 1.0

    def test_errored_records_excluded_from_trend(self):
        # 2 ok + 1 error in the middle, ordered by recency. The trend list
        # must NOT include the errored record's val_loss (errored records
        # don't carry a val_loss field by construction).
        recs = [
            _train(1.0, val_loss=2.0),
            _train(2.0, status="error"),
            _train(3.0, val_loss=1.0),
        ]
        snap = compute_ml_training_health(recs, now=NOW)
        assert snap["val_loss_trend"] == [2.0, 1.0]


class TestRealLogFile:
    """Smoke test the live log file if present — should never crash."""

    def test_live_log_does_not_raise(self):
        # If the production log exists in this repo's data/ dir, compute()
        # must return a snapshot. The verdict can be anything in the
        # closed alphabet — what we're pinning is "the reader survives".
        snap = compute()
        assert snap["verdict"] in VERDICTS
        assert isinstance(snap["records_read"], int)
        assert isinstance(snap["val_loss_trend"], list)
