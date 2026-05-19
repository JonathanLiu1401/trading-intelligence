"""Tests for paper_trader.ml.outcome_data_quality.

Read-only data-quality auditor for ``decision_outcomes.jsonl``. The tests
must assert specific verdicts on synthetic JSONL fixtures — no generic
"runs without error" coverage. Mirrors the discipline used by
``test_skill_uncertainty`` / ``test_calibration`` / ``test_corpus_audit``
(threshold-driven, deterministic, network-free).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import outcome_data_quality as odq
from paper_trader.ml.outcome_data_quality import (
    CONFLICT_COUNT_MAX,
    CONFLICT_TOL_PCT,
    EXTREME_PCT,
    MIN_ROWS,
    NONFINITE_COUNT_MAX,
    NUMERIC_FEATURE_KEYS,
    ZERO_TARGET_RATE_MAX,
    _is_finite_or_none,
    _percentile,
    _safe_float,
    audit_outcomes,
)


# ─────────────────────── helpers ───────────────────────

def _write_outcomes(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "outcomes.jsonl"
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def _base_row(**overrides) -> dict:
    """A schema-complete outcome row with sensible non-pathological values.

    Default ``forward_return_5d`` is a real non-zero number (+1.5%), so each
    test only needs to override the FIELD UNDER TEST and not worry about
    accidentally landing the row in another bucket (e.g., a default 0.0
    would silently bump the exact-zero count).
    """
    row = {
        "run_id": 100,
        "sim_date": "2025-06-15",
        "ticker": "NVDA",
        "action": "BUY",
        "ml_score": 2.5,
        "rsi": 60.0,
        "macd": 0.1,
        "mom5": 2.0,
        "mom20": 5.0,
        "regime_mult": 1.0,
        "vol_ratio": 1.0,
        "bb_position": 0.3,
        "news_urgency": None,
        "news_article_count": None,
        "forward_return_5d": 1.5,
        "forward_return_10d": 3.0,
        "forward_return_20d": 6.0,
        "return_pct": 12.0,
    }
    row.update(overrides)
    return row


# ─────────────────────── _is_finite_or_none / _safe_float ───────────────

class TestIsFiniteOrNone:
    def test_none_allowed(self):
        assert _is_finite_or_none(None) is True

    def test_finite_int(self):
        assert _is_finite_or_none(0) is True
        assert _is_finite_or_none(-5) is True

    def test_finite_float(self):
        assert _is_finite_or_none(3.14) is True

    def test_nan_rejected(self):
        assert _is_finite_or_none(float("nan")) is False

    def test_inf_rejected(self):
        assert _is_finite_or_none(float("inf")) is False
        assert _is_finite_or_none(float("-inf")) is False

    def test_bool_rejected(self):
        # bool subclasses int but a True/False in a numeric slot is a
        # type error, not a number — must NOT be counted as finite.
        assert _is_finite_or_none(True) is False
        assert _is_finite_or_none(False) is False

    def test_string_rejected(self):
        assert _is_finite_or_none("3.14") is False
        assert _is_finite_or_none("") is False


class TestSafeFloat:
    def test_finite_passthrough(self):
        assert _safe_float(3.14) == 3.14

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_bool_returns_none(self):
        # Same rule as _is_finite_or_none — bool must not become 1.0/0.0.
        assert _safe_float(True) is None
        assert _safe_float(False) is None


# ─────────────────────── _percentile ───────────────────────

class TestPercentile:
    def test_empty_returns_none(self):
        assert _percentile([], 50.0) is None

    def test_single_value(self):
        assert _percentile([7.0], 50.0) == 7.0

    def test_median_of_sorted(self):
        # Linear interpolation of sorted values — p50 of [1..9] = 5.0
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                           50.0) == pytest.approx(5.0)

    def test_p0_is_min(self):
        assert _percentile([10.0, 1.0, 5.0], 0.0) == 1.0

    def test_p100_is_max(self):
        assert _percentile([10.0, 1.0, 5.0], 100.0) == 10.0


# ─────────────────────── audit_outcomes happy path ───────────────────────

class TestAuditCleanCorpus:
    def test_clean_corpus_passes(self, tmp_path):
        # 60 healthy rows — non-zero, no NaN/inf, no conflicts.
        rows = [_base_row(run_id=i, forward_return_5d=0.5 + 0.1 * (i % 5))
                for i in range(60)]
        p = _write_outcomes(tmp_path, rows)
        rep = audit_outcomes(p)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "CLEAN"
        assert rep["n_parsed"] == 60
        assert rep["n_exact_zero_5d"] == 0
        assert rep["n_conflict_dup_5d"] == 0
        assert rep["n_nonfinite_feature"] == 0
        # Distribution stats are populated.
        assert rep["target_stats"] is not None
        assert rep["target_stats"]["n"] == 60
        assert rep["target_stats"]["mean"] > 0  # all positive rows
        assert rep["target_stats"]["min"] >= 0
        assert rep["target_stats"]["max"] <= 1.0


# ─────────────────────── ZERO_LABEL_CONTAMINATION ───────────────────────

class TestZeroLabelContamination:
    def test_threshold_trigger(self, tmp_path):
        """50 rows total + 3 exact-zero rows = 6% rate, well above the 0.5%
        threshold — verdict must be ZERO_LABEL_CONTAMINATION."""
        rows = [_base_row(run_id=i,
                          forward_return_5d=0.0 if i < 3 else 1.0)
                for i in range(60)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "ZERO_LABEL_CONTAMINATION"
        assert rep["n_exact_zero_5d"] == 3

    def test_below_threshold_stays_clean(self, tmp_path):
        """Under 0.5% rate — verdict stays CLEAN even with a few zeros.
        Concretely: 1 zero in 1000 rows = 0.1%."""
        rows = [_base_row(run_id=i,
                          forward_return_5d=0.0 if i == 0 else 1.0)
                for i in range(1000)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "CLEAN"
        assert rep["n_exact_zero_5d"] == 1

    def test_lots_of_zeros_unambiguous(self, tmp_path):
        """A heavily-contaminated corpus — every other row is a fabricated
        zero. The verdict must flag this loudly."""
        rows = [_base_row(run_id=i,
                          forward_return_5d=0.0 if i % 2 == 0 else 2.0)
                for i in range(80)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "ZERO_LABEL_CONTAMINATION"
        assert rep["n_exact_zero_5d"] == 40


# ─────────────────────── CONFLICTING_DUPLICATES ───────────────────────

class TestConflictingDuplicates:
    def test_conflict_detected(self, tmp_path):
        """Same (run_id, sim_date, ticker, action) appearing twice with
        DIFFERENT forward_return_5d > tolerance is a conflict."""
        base = [_base_row(run_id=i, forward_return_5d=1.0) for i in range(70)]
        # Add 6 conflicts (one over CONFLICT_COUNT_MAX).
        for i in range(6):
            base.append(_base_row(run_id=i, sim_date="2025-06-15",
                                  ticker="NVDA", forward_return_5d=1.0))
            base.append(_base_row(run_id=i, sim_date="2025-06-15",
                                  ticker="NVDA", forward_return_5d=5.0))
        rep = audit_outcomes(_write_outcomes(tmp_path, base))
        assert rep["n_conflict_dup_5d"] == 6
        # 6 > CONFLICT_COUNT_MAX (5) → CONFLICTING_DUPLICATES
        assert rep["verdict"] == "CONFLICTING_DUPLICATES"

    def test_below_tolerance_not_a_conflict(self, tmp_path):
        """4-decimal rounding noise (e.g., 1.2345 vs 1.2349) is NOT a
        conflict — the tolerance correctly accepts these."""
        rows = [_base_row(run_id=i, forward_return_5d=1.0) for i in range(60)]
        # Two rows differ by 0.005 — under CONFLICT_TOL_PCT=0.01
        rows.append(_base_row(run_id=999, sim_date="2025-06-16",
                              forward_return_5d=2.5005))
        rows.append(_base_row(run_id=999, sim_date="2025-06-16",
                              forward_return_5d=2.5050))
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["n_conflict_dup_5d"] == 0
        assert rep["verdict"] == "CLEAN"


# ─────────────────────── NONFINITE_FEATURES ───────────────────────

class TestNonFiniteFeatures:
    def test_nan_in_feature_flagged(self, tmp_path):
        """A NaN in any scorer-input feature is a corpus-quality bug —
        `train_scorer` would crash on it without the `_to_float` hardening
        in build_features."""
        rows = [_base_row(run_id=i) for i in range(60)]
        # Add 6 rows (over NONFINITE_COUNT_MAX=5) with NaN RSI.
        for i in range(6):
            rows.append(_base_row(run_id=200 + i,
                                  rsi=float("nan")))
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["n_nonfinite_feature"] == 6
        assert rep["feature_nonfinite_by_field"]["rsi"] == 6
        assert rep["verdict"] == "NONFINITE_FEATURES"

    def test_bool_in_numeric_slot_flagged(self, tmp_path):
        """A True/False in a numeric slot is a type bug — must NOT be
        silently coerced to 1.0/0.0. The `_safe_float`/`_is_finite_or_none`
        rules reject bools the same way `_to_float` in decision_scorer does.
        """
        rows = [_base_row(run_id=i) for i in range(60)]
        for i in range(6):
            rows.append(_base_row(run_id=300 + i, ml_score=True))
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["n_nonfinite_feature"] == 6
        assert rep["feature_nonfinite_by_field"]["ml_score"] == 6
        assert rep["verdict"] == "NONFINITE_FEATURES"

    def test_optional_none_is_legal(self, tmp_path):
        """forward_return_10d=None is legitimate (window past cache horizon)
        — must NOT be counted as non-finite."""
        rows = [_base_row(run_id=i, forward_return_10d=None,
                          forward_return_20d=None) for i in range(60)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["n_nonfinite_feature"] == 0
        assert rep["verdict"] == "CLEAN"


# ─────────────────────── INSUFFICIENT_DATA ───────────────────────

class TestInsufficientData:
    def test_empty_file(self, tmp_path):
        p = _write_outcomes(tmp_path, [])
        rep = audit_outcomes(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_parsed"] == 0

    def test_below_min_rows(self, tmp_path):
        rows = [_base_row(run_id=i) for i in range(MIN_ROWS - 5)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_parsed"] == MIN_ROWS - 5

    def test_missing_file(self, tmp_path):
        rep = audit_outcomes(tmp_path / "does_not_exist.jsonl")
        assert rep["status"] == "error"
        assert "not found" in rep["error"]


# ─────────────────────── corrupt input ───────────────────────

class TestCorruptInput:
    def test_corrupt_line_counted_not_fatal(self, tmp_path):
        """A single bad JSON line must not abort the audit — count it as
        corrupt and continue. This is the streaming-discipline
        the other auditors follow."""
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write("not a json line\n")
            for r in [_base_row(run_id=i) for i in range(60)]:
                fh.write(json.dumps(r) + "\n")
            fh.write("{partial json\n")
        rep = audit_outcomes(p)
        assert rep["n_corrupt"] == 2
        assert rep["n_parsed"] == 60
        assert rep["verdict"] == "CLEAN"

    def test_non_dict_payload_counted_as_corrupt(self, tmp_path):
        """A JSONL line containing a valid JSON value that ISN'T an object
        (e.g. a stray array or string) is malformed for our schema."""
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write("[1, 2, 3]\n")
            fh.write('"a string"\n')
            for r in [_base_row(run_id=i) for i in range(60)]:
                fh.write(json.dumps(r) + "\n")
        rep = audit_outcomes(p)
        assert rep["n_corrupt"] == 2
        assert rep["n_parsed"] == 60


# ─────────────────────── extreme returns ───────────────────────

class TestExtremeReturns:
    def test_extreme_counted_but_not_blocking_verdict(self, tmp_path):
        """|fr_5d| > 50% rows are surfaced for visibility, but extreme
        returns alone do NOT trip a verdict — leveraged ETF crash/rip
        weeks are real."""
        rows = [_base_row(run_id=i, forward_return_5d=1.0)
                for i in range(100)]
        rows.append(_base_row(run_id=200, forward_return_5d=85.0))
        rows.append(_base_row(run_id=201, forward_return_5d=-90.0))
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        assert rep["n_extreme_5d"] == 2
        assert rep["verdict"] == "CLEAN"


# ─────────────────────── target stats sanity ───────────────────────

class TestTargetStats:
    def test_known_distribution_recovered(self, tmp_path):
        """Synthetic values whose mean / min / max we already know — the
        audit must report exactly those (within float rounding)."""
        rows = [_base_row(run_id=i,
                          forward_return_5d=float(i - 49))  # -49..50
                for i in range(100)]
        rep = audit_outcomes(_write_outcomes(tmp_path, rows))
        ts = rep["target_stats"]
        assert ts["n"] == 100
        assert ts["min"] == -49.0
        assert ts["max"] == 50.0
        # mean of integers -49..50 = 0.5
        assert ts["mean"] == pytest.approx(0.5, abs=1e-4)


# ─────────────────────── CLI ───────────────────────

class TestCLI:
    def test_exit_code_clean(self, tmp_path, capsys):
        rows = [_base_row(run_id=i, forward_return_5d=1.0)
                for i in range(60)]
        p = _write_outcomes(tmp_path, rows)
        rc = odq.main(["--path", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CLEAN" in out
        assert "parsed=60" in out

    def test_exit_code_dirty(self, tmp_path, capsys):
        rows = [_base_row(run_id=i,
                          forward_return_5d=0.0 if i < 10 else 1.0)
                for i in range(80)]
        rc = odq.main(["--path", str(_write_outcomes(tmp_path, rows))])
        assert rc == 1
        out = capsys.readouterr().out
        assert "ZERO_LABEL_CONTAMINATION" in out

    def test_json_output(self, tmp_path, capsys):
        rows = [_base_row(run_id=i) for i in range(60)]
        rc = odq.main(["--path", str(_write_outcomes(tmp_path, rows)),
                       "--json"])
        out = capsys.readouterr().out
        # Output must be a parseable JSON object.
        obj = json.loads(out)
        assert obj["verdict"] == "CLEAN"
        assert obj["n_parsed"] == 60
        assert rc == 0
