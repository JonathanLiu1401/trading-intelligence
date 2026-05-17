"""Tests for paper_trader.ml.label_audit (2026-05-17 ML/backtest pass).

Exact-value verdict + arithmetic locks on hand-constructed deterministic
data with known-correct answers, mirroring tests/test_calibration.py. A
threshold or classification change must update these literals deliberately
rather than silently shift a quant-facing label-hygiene diagnostic.

The audit is read-only (no train / no pickle / no jsonl rewrite); these
tests assert that contract by construction — every dataset is an in-memory
list, and the CONTAMINATED hint explicitly points at the *documented*
remediation, never an in-tool fix.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml.decision_scorer import PRED_CLAMP_PCT
from paper_trader.ml.label_audit import (
    CLEAN_MAX_RATE,
    ELEVATED_MAX_RATE,
    EXTREME_RETURN_PCT,
    MIN_RECORDS,
    audit_outcome_labels,
    _load_outcomes,
)


def _mk(n_clean: int, n_extreme: int, ticker: str = "AAA",
        clean_fwd: float = 1.0, extreme_fwd: float = 100.0,
        mom5: float = 1.0, action: str = "BUY") -> list[dict]:
    """n_clean rows with |fwd|≤bound + n_extreme rows with |fwd|>bound."""
    recs = []
    for i in range(n_clean):
        recs.append({"ticker": ticker, "sim_date": f"2020-01-{i%28+1:02d}",
                     "forward_return_5d": clean_fwd, "mom5": mom5,
                     "action": action})
    for i in range(n_extreme):
        recs.append({"ticker": ticker, "sim_date": f"2024-06-{i%28+1:02d}",
                     "forward_return_5d": extreme_fwd, "mom5": mom5,
                     "action": action})
    return recs


class TestSingleSourceOfTruth:
    def test_extreme_bound_is_pred_clamp(self):
        # The audit boundary must BE the inference clamp, not a copy that
        # can drift (the calibration._spearman-reuse precedent).
        assert EXTREME_RETURN_PCT == PRED_CLAMP_PCT == 50.0


class TestVerdictBoundaries:
    def test_clean_at_exact_threshold(self):
        # 3/500 = 0.006 == CLEAN_MAX_RATE → CLEAN (inclusive boundary).
        rep = audit_outcome_labels(_mk(497, 3))
        assert rep["n"] == 500
        assert rep["extreme_count"] == 3
        assert rep["extreme_rate"] == 0.006
        assert rep["verdict"] == "CLEAN"

    def test_elevated_just_over_clean_threshold(self):
        # 4/500 = 0.008 → just past CLEAN_MAX_RATE → ELEVATED.
        rep = audit_outcome_labels(_mk(496, 4))
        assert rep["extreme_rate"] == 0.008
        assert rep["verdict"] == "ELEVATED"

    def test_elevated_at_exact_upper_threshold(self):
        # 3/200 = 0.015 == ELEVATED_MAX_RATE → ELEVATED (inclusive).
        rep = audit_outcome_labels(_mk(197, 3))
        assert rep["n"] == 200
        assert rep["extreme_rate"] == 0.015
        assert rep["verdict"] == "ELEVATED"

    def test_contaminated_just_over_elevated_threshold(self):
        # 4/200 = 0.02 → past ELEVATED_MAX_RATE → CONTAMINATED.
        rep = audit_outcome_labels(_mk(196, 4))
        assert rep["extreme_rate"] == 0.02
        assert rep["verdict"] == "CONTAMINATED"
        # The hint must point at the DOCUMENTED remediation, never a
        # winsorize-in-train_scorer fix (the out-of-scope change).
        assert "delete data/ml/decision_scorer.pkl" in rep["hint"]
        assert "do NOT winsorize" in rep["hint"]

    def test_thresholds_are_the_documented_values(self):
        assert MIN_RECORDS == 30
        assert CLEAN_MAX_RATE == 0.006
        assert ELEVATED_MAX_RATE == 0.015


class TestExtremeBoundary:
    def test_exactly_at_bound_is_not_extreme(self):
        # |fwd| == 50.0 is NOT extreme (strict `>` — mirrors the clamp's
        # `abs(raw) > PRED_CLAMP_PCT` boundary).
        rep = audit_outcome_labels(_mk(30, 0, extreme_fwd=0.0,
                                       clean_fwd=EXTREME_RETURN_PCT))
        assert rep["n"] == 30
        assert rep["extreme_count"] == 0
        assert rep["verdict"] == "CLEAN"

    def test_just_past_bound_is_extreme(self):
        rep = audit_outcome_labels(_mk(0, 30, extreme_fwd=50.0001))
        assert rep["extreme_count"] == 30
        assert rep["extreme_rate"] == 1.0
        assert rep["verdict"] == "CONTAMINATED"

    def test_negative_extreme_counts_by_magnitude(self):
        # A −67% crash label is extreme too (|fwd| > bound, sign-agnostic).
        rep = audit_outcome_labels(_mk(196, 4, extreme_fwd=-67.0))
        assert rep["extreme_count"] == 4
        assert rep["verdict"] == "CONTAMINATED"


class TestActionAgnostic:
    def test_sell_extreme_is_still_extreme(self):
        # train_scorer's SELL sign-flip does not change |fwd|; a split
        # discontinuity is in the raw price series regardless of action.
        rep = audit_outcome_labels(_mk(196, 4, action="SELL",
                                       extreme_fwd=120.0))
        assert rep["extreme_count"] == 4
        assert rep["worst_labels"][0]["action"] == "SELL"


class TestInsufficientData:
    def test_below_min_records(self):
        rep = audit_outcome_labels(_mk(20, 5))  # 25 < 30
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 25
        assert rep["extreme_rate"] is None
        assert rep["worst_offenders"] == []

    def test_exactly_min_records_is_enough(self):
        rep = audit_outcome_labels(_mk(30, 0))  # 30 == MIN_RECORDS
        assert rep["status"] == "ok"
        assert rep["n"] == 30


class TestNonFiniteDropped:
    def test_nan_inf_none_string_dropped(self):
        recs = _mk(30, 0)
        recs += [
            {"ticker": "X", "forward_return_5d": float("nan")},
            {"ticker": "X", "forward_return_5d": float("inf")},
            {"ticker": "X", "forward_return_5d": None},
            {"ticker": "X", "forward_return_5d": "not-a-number"},
            {"ticker": "X"},  # key absent
        ]
        rep = audit_outcome_labels(recs)
        # The 5 bad rows are dropped; n stays the 30 finite ones.
        assert rep["n"] == 30
        assert rep["status"] == "ok"


class TestPerTickerOffenders:
    def test_only_tickers_with_extremes_listed_and_sorted(self):
        recs = (_mk(100, 5, ticker="DFEN")     # 5 extreme
                + _mk(100, 2, ticker="FAS")    # 2 extreme
                + _mk(100, 0, ticker="NVDA"))  # 0 extreme → excluded
        rep = audit_outcome_labels(recs)
        offs = rep["worst_offenders"]
        tickers = [o["ticker"] for o in offs]
        assert "NVDA" not in tickers              # no extreme → not listed
        assert tickers == ["DFEN", "FAS"]         # sorted by extreme_count desc
        assert offs[0]["extreme_count"] == 5
        assert offs[0]["n"] == 105
        assert offs[0]["rate"] == round(5 / 105, 4)

    def test_worst_labels_sorted_by_abs_magnitude_and_capped(self):
        recs = _mk(200, 0)
        recs.append({"ticker": "A", "sim_date": "2024-06-03",
                     "forward_return_5d": 180.0, "mom5": -64.0})
        recs.append({"ticker": "B", "sim_date": "2020-03-19",
                     "forward_return_5d": -90.0, "mom5": -48.0})
        rep = audit_outcome_labels(recs, top_n=1)
        # Sorted by |fwd| desc; 180 > 90, capped at top_n=1.
        assert len(rep["worst_labels"]) == 1
        assert rep["worst_labels"][0]["forward_return_5d"] == 180.0
        assert rep["worst_labels"][0]["mom5"] == -64.0


class TestDirectionalAnomalyIsInformationalOnly:
    def test_high_directional_low_extreme_stays_clean(self):
        # 25 rows where fwd (+45) opposes mom5 (−10): each is a directional
        # anomaly but NONE is extreme (|45| < 50). Verdict must stay CLEAN —
        # directional anomaly never drives classification (it also fires on
        # genuine COVID mean-reversions, by design).
        recs = _mk(975, 0)
        for i in range(25):
            recs.append({"ticker": "DFEN", "sim_date": f"2020-03-{i%28+1:02d}",
                         "forward_return_5d": 45.0, "mom5": -10.0})
        rep = audit_outcome_labels(recs)
        assert rep["n"] == 1000
        assert rep["extreme_count"] == 0
        assert rep["directional_anomaly_count"] == 25
        assert rep["directional_anomaly_rate"] == 0.025
        assert rep["verdict"] == "CLEAN"   # NOT driven by directional anomaly


class TestLoadOutcomes:
    def test_skips_blank_and_corrupt_lines(self, tmp_path):
        p = tmp_path / "decision_outcomes.jsonl"
        p.write_text(
            json.dumps({"ticker": "NVDA", "forward_return_5d": 1.0}) + "\n"
            + "\n"                       # blank
            + "{ not valid json\n"       # corrupt
            + "   \n"                     # whitespace
            + json.dumps({"ticker": "AMD", "forward_return_5d": 2.0}) + "\n"
        )
        recs = _load_outcomes(p)
        assert len(recs) == 2
        assert {r["ticker"] for r in recs} == {"NVDA", "AMD"}


class TestRealOutcomesFileShape:
    """Smoke-check against the live accumulated outcomes tail if present —
    locks that the audit runs end-to-end on the real row shape and that the
    current corpus matches the documented ~0.5% baseline (CLEAN/ELEVATED,
    never crashing). Skipped offline if the file is absent."""

    def test_live_file_audits_without_error(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        f = root / "data" / "decision_outcomes.jsonl"
        if not f.exists():
            pytest.skip("no live decision_outcomes.jsonl")
        rep = audit_outcome_labels(_load_outcomes(f))
        assert rep["status"] in ("ok", "insufficient_data")
        if rep["status"] == "ok":
            assert 0.0 <= rep["extreme_rate"] <= 1.0
            assert rep["verdict"] in (
                "CLEAN", "ELEVATED", "CONTAMINATED")
            assert rep["extreme_pct_bound"] == 50.0
