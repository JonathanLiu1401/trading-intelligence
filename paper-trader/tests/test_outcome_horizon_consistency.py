"""Tests for paper_trader.ml.outcome_horizon_consistency.

Synthetic outcomes with known sign patterns drive every verdict path:
STRONG_PERSISTENCE, MIXED_PERSISTENCE, HIGH_REVERSAL, INSUFFICIENT_DATA.
Tests assert SPECIFIC EXPECTED VALUES (agreement rates, reversal rates,
verdict strings) — not just "didn't crash".
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml.outcome_horizon_consistency import (
    ANCHOR_HORIZON,
    LONGER_HORIZONS,
    MIN_PAIRS,
    STRONG_AGREE,
    WEAK_AGREE,
    _aligned_sign,
    _agreement_rate,
    _reversal_rate,
    analyze,
    consistency_report,
)


def _outcome(fr5: float | None, fr10: float | None, fr20: float | None,
             action: str = "BUY", ticker: str = "NVDA",
             sim_date: str = "2025-01-15") -> dict:
    return {
        "ticker": ticker,
        "action": action,
        "sim_date": sim_date,
        "forward_return_5d": fr5,
        "forward_return_10d": fr10,
        "forward_return_20d": fr20,
    }


# ─────────────────────── _aligned_sign ───────────────────────────

class TestAlignedSign:
    def test_buy_positive_is_positive(self):
        assert _aligned_sign(5.0, is_sell=False) == 1

    def test_buy_negative_is_negative(self):
        assert _aligned_sign(-3.0, is_sell=False) == -1

    def test_sell_flips_positive_to_negative(self):
        # A SELL whose price went UP was the WRONG call → aligned sign = -1
        assert _aligned_sign(5.0, is_sell=True) == -1

    def test_sell_flips_negative_to_positive(self):
        # A SELL whose price went DOWN was the RIGHT call → aligned sign = +1
        assert _aligned_sign(-5.0, is_sell=True) == 1

    def test_zero_returns_zero(self):
        # Zero carries no directional truth — excluded from agreement counts.
        assert _aligned_sign(0.0, is_sell=False) == 0
        assert _aligned_sign(0.0, is_sell=True) == 0


# ─────────────────────── _agreement_rate ───────────────────────────

class TestAgreementRate:
    def test_perfect_agreement(self):
        a = [1, 1, -1, -1, 1]
        b = [1, 1, -1, -1, 1]
        assert _agreement_rate(a, b) == 1.0

    def test_perfect_disagreement(self):
        a = [1, 1, -1, -1]
        b = [-1, -1, 1, 1]
        assert _agreement_rate(a, b) == 0.0

    def test_mixed_agreement(self):
        # 3/4 pairs agree → 0.75
        a = [1, 1, 1, -1]
        b = [1, 1, -1, -1]
        assert _agreement_rate(a, b) == 0.75

    def test_zero_excludes_pair_from_denominator(self):
        # Two zeros on the b side; only the 2 non-zero pairs count.
        # Of those, both agree → 1.0.
        a = [1, 1, 1, -1]
        b = [1, 0, 0, -1]
        assert _agreement_rate(a, b) == 1.0

    def test_all_zero_returns_none(self):
        # No non-zero pairs → no honest denominator → None.
        a = [0, 0, 0]
        b = [0, 0, 0]
        assert _agreement_rate(a, b) is None

    def test_mismatched_lengths_returns_none(self):
        assert _agreement_rate([1, 1], [1, 1, 1]) is None


# ─────────────────────── _reversal_rate ───────────────────────────

class TestReversalRate:
    def test_no_reversals(self):
        # All same-sign pairs → reversal rate 0.
        a = [1, 1, -1]
        b = [1, 1, -1]
        assert _reversal_rate(a, b) == 0.0

    def test_all_reversals(self):
        # All strictly-opposite pairs → 1.0.
        a = [1, -1, 1]
        b = [-1, 1, -1]
        assert _reversal_rate(a, b) == 1.0

    def test_zero_excludes_pair(self):
        # A 0 on either side is "no directional truth", not a reversal.
        # 2 non-zero pairs, 1 reverses → 0.5.
        a = [1, 1, 0]
        b = [-1, 1, -1]
        assert _reversal_rate(a, b) == 0.5

    def test_empty_returns_none(self):
        assert _reversal_rate([0], [0]) is None


# ─────────────────────── consistency_report (verdict ladder) ───────────────

class TestConsistencyReportVerdicts:
    def test_insufficient_data_below_min_pairs(self):
        """< MIN_PAIRS complete rows → INSUFFICIENT_DATA verdict."""
        recs = [_outcome(1.0, 2.0, 3.0) for _ in range(MIN_PAIRS - 1)]
        rep = consistency_report(recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_complete"] == MIN_PAIRS - 1
        # Hint must mention the required floor so an operator knows why.
        assert str(MIN_PAIRS) in rep["hint"]

    def test_strong_persistence_when_signs_agree(self):
        """100% sign agreement → STRONG_PERSISTENCE."""
        recs = []
        # 40 rows, all BUY, all positive across every horizon.
        for i in range(40):
            recs.append(_outcome(1.0 + i * 0.1, 2.0 + i * 0.1,
                                 3.0 + i * 0.1,
                                 sim_date=f"2025-01-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        assert rep["verdict"] == "STRONG_PERSISTENCE"
        assert rep["n_complete"] == 40
        # Per-cell rates: anchor 5d → 10d, anchor 5d → 20d.
        rates = {c["longer"]: c["agreement_rate"] for c in rep["cells"]}
        for h in LONGER_HORIZONS:
            assert rates[h] == pytest.approx(1.0)
        assert rep["min_agreement"] == pytest.approx(1.0)

    def test_high_reversal_when_signs_flip(self):
        """< WEAK_AGREE sign agreement → HIGH_REVERSAL."""
        recs = []
        # 40 rows: 5d=+1, 10d=-1, 20d=-1 → 100% reversal on both pairs.
        for i in range(40):
            recs.append(_outcome(1.0, -2.0, -3.0,
                                 sim_date=f"2025-02-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        assert rep["verdict"] == "HIGH_REVERSAL"
        rates = {c["longer"]: c["agreement_rate"] for c in rep["cells"]}
        for h in LONGER_HORIZONS:
            assert rates[h] == pytest.approx(0.0)
            # Reversal rate is the dual — every non-zero pair reverses.
            rr = next(c["reversal_rate"] for c in rep["cells"]
                      if c["longer"] == h)
            assert rr == pytest.approx(1.0)

    def test_mixed_persistence_between_thresholds(self):
        """In [WEAK_AGREE, STRONG_AGREE) → MIXED_PERSISTENCE."""
        # Construct 40 rows where 60% agree on 5d↔10d AND 5d↔20d.
        # 24/40 same-sign for both longer horizons; 16/40 flip.
        recs = []
        for i in range(40):
            if i < 24:
                recs.append(_outcome(1.0, 2.0, 3.0,
                                     sim_date=f"2025-03-{(i % 27) + 1:02d}"))
            else:
                # Flip 10d AND 20d but keep 5d positive.
                recs.append(_outcome(1.0, -2.0, -3.0,
                                     sim_date=f"2025-04-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        assert rep["verdict"] == "MIXED_PERSISTENCE"
        # 24/40 = 0.60 — between 0.50 and 0.75.
        for c in rep["cells"]:
            assert c["agreement_rate"] == pytest.approx(0.60)
        assert WEAK_AGREE <= rep["min_agreement"] < STRONG_AGREE

    def test_partial_horizon_rows_excluded_from_complete_count(self):
        """Rows missing ANY horizon must not contribute (denominator
        consistency across pairs is load-bearing for honest verdicts).
        """
        recs = []
        # 25 fully-resolved rows.
        for i in range(25):
            recs.append(_outcome(1.0, 2.0, 3.0,
                                 sim_date=f"2025-05-{(i % 27) + 1:02d}"))
        # 25 rows missing 20d — must be DROPPED from n_complete.
        for i in range(25):
            recs.append(_outcome(1.0, 2.0, None,
                                 sim_date=f"2025-06-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        # Only the 25 complete rows count; < MIN_PAIRS=30 → INSUFFICIENT.
        assert rep["n_complete"] == 25
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_sell_sign_flip_applied(self):
        """A SELL whose 5d=+5 / 20d=-5 has aligned 5d=-1 and aligned 20d=+1.
        That is a REVERSAL of action-aligned signs — even though the raw
        prices both moved in opposite directions, the trader's
        intent-relative goodness is what counts."""
        recs = []
        # 35 SELL rows: raw 5d=+5, 10d=-5, 20d=-5 → aligned: 5d=-1,
        # 10d=+1, 20d=+1. So aligned 5d↔10d reverses, 5d↔20d reverses.
        for i in range(35):
            recs.append(_outcome(5.0, -5.0, -5.0, action="SELL",
                                 sim_date=f"2025-07-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        # Aligned signs all flip on both pairs → HIGH_REVERSAL.
        assert rep["verdict"] == "HIGH_REVERSAL"
        for c in rep["cells"]:
            assert c["agreement_rate"] == pytest.approx(0.0)


# ─────────────────────── magnitude summaries ───────────────────────────

class TestMagnitudeSummaries:
    def test_mean_abs_5d_when_agree_vs_reverse(self):
        """The per-cell |5d| magnitude summary lets a quant ask 'are the
        persistent winners BIGGER than the reversers?'. Hand-built case:
        agree-rows have |5d|=10, reverse-rows have |5d|=2."""
        recs = []
        # 25 agree-rows with |5d|=10: 5d=+10, 10d=+5, 20d=+5.
        for i in range(25):
            recs.append(_outcome(10.0, 5.0, 5.0,
                                 sim_date=f"2025-08-{(i % 27) + 1:02d}"))
        # 25 reverse-rows with |5d|=2: 5d=+2, 10d=-3, 20d=-3.
        for i in range(25):
            recs.append(_outcome(2.0, -3.0, -3.0,
                                 sim_date=f"2025-09-{(i % 27) + 1:02d}"))
        rep = consistency_report(recs)
        # 25/50 agree on each pair → MIXED_PERSISTENCE (0.5 == WEAK_AGREE).
        assert rep["n_complete"] == 50
        for c in rep["cells"]:
            assert c["mean_abs_5d_when_agree"] == pytest.approx(10.0)
            assert c["mean_abs_5d_when_reverse"] == pytest.approx(2.0)


# ─────────────────────── analyze (file-driven, oos slice) ──────────────

class TestAnalyze:
    def test_missing_file_returns_insufficient(self, tmp_path):
        rep = analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_strong_persistence_round_trip(self, tmp_path):
        """End-to-end with --all slice: write JSONL → read → consistency
        report → verdict matches in-memory consistency_report."""
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as f:
            for i in range(40):
                rec = _outcome(1.0 + i * 0.05, 2.0 + i * 0.05,
                               3.0 + i * 0.05,
                               sim_date=f"2025-10-{(i % 27) + 1:02d}")
                f.write(json.dumps(rec) + "\n")
        rep = analyze(path, oos_only=False)
        assert rep["verdict"] == "STRONG_PERSISTENCE"
        assert rep["slice"] == "all"
        assert rep["n_records_total"] == 40

    def test_oos_slice_when_validation_available(self, tmp_path):
        """When oos_only=True and split_outcomes_temporal is importable,
        only the OOS 20% drives the verdict. With 200 rows, OOS = 40 rows
        → still above MIN_PAIRS=30."""
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as f:
            # All STRONG_PERSISTENCE-shaped rows.
            for i in range(200):
                rec = _outcome(1.0, 2.0, 3.0,
                               sim_date=f"2025-{(i % 12) + 1:02d}-"
                                        f"{(i % 27) + 1:02d}")
                f.write(json.dumps(rec) + "\n")
        rep = analyze(path, oos_only=True)
        # Whether the validation module is present or not the verdict is
        # the same (STRONG) — but `slice` tells us which path ran.
        assert rep["verdict"] == "STRONG_PERSISTENCE"
        assert rep["slice"] in ("oos", "all")
        assert rep["n_records_total"] == 200


# ─────────────────────── safety (never raise on bad input) ────────────

class TestNeverRaises:
    def test_empty_records(self):
        rep = consistency_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_garbage_records_silently_dropped(self):
        # Mix of valid + non-finite + missing-field rows. The pipeline must
        # NEVER raise — invalid rows drop and only the survivors drive the
        # verdict. This mirrors the discipline every sibling diagnostic
        # follows (`horizon_audit`, `outcome_drift`, `calibration`).
        recs: list[dict] = []
        # 35 valid rows.
        for i in range(35):
            recs.append(_outcome(1.0, 2.0, 3.0,
                                 sim_date=f"2025-11-{(i % 27) + 1:02d}"))
        # 5 garbage rows.
        recs.append(_outcome(float("nan"), 1.0, 1.0))   # NaN 5d
        recs.append(_outcome(1.0, float("inf"), 1.0))    # inf 10d
        recs.append(_outcome(1.0, 1.0, "not a number"))  # string
        recs.append({})                                  # missing every key
        recs.append(_outcome(None, None, None))          # explicit nulls
        rep = consistency_report(recs)
        # The 35 valid rows drive the verdict; garbage is dropped.
        assert rep["n_complete"] == 35
        assert rep["verdict"] == "STRONG_PERSISTENCE"


# ─────────────────────── constants pinned ───────────────────────────

class TestConstants:
    def test_thresholds_ordered(self):
        """WEAK_AGREE < STRONG_AGREE — a tuning regression that crossed
        them would break the verdict ladder silently. The ladder is the
        whole API surface, so pin the order explicitly."""
        assert 0.0 < WEAK_AGREE < STRONG_AGREE < 1.0

    def test_anchor_and_longer_horizons_distinct(self):
        """The anchor must not appear in LONGER_HORIZONS (would compare
        the horizon to itself — always 100% agreement, useless cell)."""
        assert ANCHOR_HORIZON not in LONGER_HORIZONS
