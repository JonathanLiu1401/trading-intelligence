"""Exact-value locks for the per-persona decision-signal-skill diagnostic
(`paper_trader/ml/persona_skill.py`, 2026-05-17 ML/backtest hybrid pass).

Mirrors tests/test_calibration.py / test_skill_trend.py /
test_persona_leaderboard_20260517.py: deterministic synthetic data with
hand-computable answers, exact metrics + exact verdicts (not ranges) so a
threshold or classification change must update the literals deliberately
rather than silently shift a quant-facing diagnostic.

`persona_for` is imported by the module (single source of truth); these
tests pin run_id→persona by *using that same function*, so a PERSONAS
reorder updates both sides together and can never silently desync the
historical aggregates. Every dataset is an in-memory list and the
`_load_outcomes` test uses a throwaway temp file — the read-only contract
is asserted by construction. All offline.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.backtest import persona_for
from paper_trader.ml.persona_skill import (
    IC_GOOD,
    IC_MIN,
    MIN_OUTCOMES_PER_PERSONA,
    MIN_RECORDS,
    _aligned,
    _load_outcomes,
    persona_skill,
)


def _rows_for(persona_idx: int, pairs, action="BUY"):
    """Build outcome dicts whose run_ids all map to `persona_idx` (1..10).

    run_id = persona_idx + 10*k ⇒ ((run_id-1) % 10) + 1 == persona_idx,
    exactly what backtest.persona_for computes. `pairs` is a list of
    (ml_score, forward_return_5d).
    """
    return [
        {"run_id": persona_idx + 10 * k, "action": action,
         "ml_score": ms, "forward_return_5d": fr}
        for k, (ms, fr) in enumerate(pairs)
    ]


# A clean, MIN_RECORDS-clearing perfectly-rank-correlated 30-row block.
_MONO = [(float(i), float(i)) for i in range(1, 31)]  # 30 rows, ml==fr==i


# ───────────────────────── _aligned helper ─────────────────────────

class TestAlignedHelper:
    def test_buy_passthrough(self):
        assert _aligned({"action": "BUY", "ml_score": 3.0,
                         "forward_return_5d": 4.0}) == (3.0, 4.0)

    def test_sell_double_flips_signal_and_target(self):
        # A strong (very negative) sell conviction that correctly preceded a
        # drop must read as (high signal, high goodness).
        assert _aligned({"action": "SELL", "ml_score": -2.0,
                         "forward_return_5d": -5.0}) == (2.0, 5.0)
        assert _aligned({"action": "SELL", "ml_score": 1.5,
                         "forward_return_5d": 3.0}) == (-1.5, -3.0)

    def test_genuine_zero_is_kept_not_dropped(self):
        # 0.0 is finite — must be a real (0,0) pair, distinct from a drop.
        assert _aligned({"action": "BUY", "ml_score": 0.0,
                         "forward_return_5d": 0.0}) == (0.0, 0.0)

    def test_missing_or_nonfinite_dropped(self):
        assert _aligned({"action": "BUY", "ml_score": 1.0}) is None
        assert _aligned({"action": "BUY", "forward_return_5d": 1.0}) is None
        assert _aligned({"action": "BUY", "ml_score": None,
                         "forward_return_5d": 1.0}) is None
        assert _aligned({"action": "BUY", "ml_score": float("inf"),
                         "forward_return_5d": 1.0}) is None
        assert _aligned({"action": "BUY", "ml_score": 1.0,
                         "forward_return_5d": float("nan")}) is None


# ───────────────────────── single source of truth ─────────────────────────

class TestSingleSourceOfTruth:
    def test_aggregates_attributed_to_correct_persona(self):
        rep = persona_skill(_rows_for(5, _MONO))
        assert rep["personas"][0]["persona"] == persona_for(5)["name"]
        assert rep["personas"][0]["persona"] == \
            "Growth at a Reasonable Price (GARP)"


# ───────────────────────── verdict boundaries ─────────────────────────

class TestPersonaVerdicts:
    def test_perfect_rank_correlation_is_signal_edge(self):
        # 30 BUY rows, ml_score == fr == i. Tie-aware Spearman of two
        # strictly-increasing identical series is exactly 1.0 (rounds to 1.0).
        rep = persona_skill(_rows_for(1, _MONO))
        e = rep["personas"][0]
        assert e["n"] == 30
        assert e["score_ic"] == 1.0
        assert e["verdict"] == "SIGNAL_EDGE"
        assert rep["verdict"] == "HEALTHY"
        assert rep["inverted_personas"] == []

    def test_perfect_anti_correlation_is_inverted(self):
        # ml_score increasing, fr strictly decreasing (still all > 0) ⇒
        # Spearman == -1.0 ⇒ INVERTED_SIGNAL ⇒ HAS_INVERTED_PERSONA overall.
        rows = _rows_for(2, [(float(i), float(31 - i)) for i in range(1, 31)])
        rep = persona_skill(rows)
        e = rep["personas"][0]
        assert e["score_ic"] == -1.0
        assert e["verdict"] == "INVERTED_SIGNAL"
        assert rep["verdict"] == "HAS_INVERTED_PERSONA"
        assert rep["inverted_personas"] == [persona_for(2)["name"]]

    def test_constant_signal_is_no_edge(self):
        # Constant ml_score (zero signal variance) ⇒ _spearman returns 0.0
        # (never NaN, never a tie-ordering artifact) ⇒ NO_SIGNAL_EDGE.
        rows = _rows_for(3, [(7.0, float(i)) for i in range(1, 31)])
        rep = persona_skill(rows)
        e = rep["personas"][0]
        assert e["score_ic"] == 0.0
        assert e["mean_signal"] == 7.0
        assert e["verdict"] == "NO_SIGNAL_EDGE"
        assert rep["verdict"] == "NO_PERSONA_EDGE"

    def test_below_min_per_persona_is_insufficient(self):
        # 19 rows < MIN_OUTCOMES_PER_PERSONA (20) ⇒ INSUFFICIENT even though
        # the overall set clears MIN_RECORDS via a second healthy persona.
        thin = _rows_for(4, [(float(i), float(i))
                             for i in range(1, MIN_OUTCOMES_PER_PERSONA)])
        assert len(thin) == 19
        rep = persona_skill(thin + _rows_for(1, _MONO))
        by = {p["persona"]: p for p in rep["personas"]}
        assert by[persona_for(4)["name"]]["verdict"] == "INSUFFICIENT"
        assert by[persona_for(4)["name"]]["n"] == 19
        # INSUFFICIENT persona sorts last regardless of any (unstable) ic.
        assert rep["personas"][-1]["verdict"] == "INSUFFICIENT"

    def test_overall_insufficient_data_below_min_records(self):
        rep = persona_skill(_rows_for(1, [(float(i), float(i))
                                          for i in range(1, MIN_RECORDS)]))
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == MIN_RECORDS - 1
        assert rep["personas"] == []


# ───────────────────────── exact arithmetic ─────────────────────────

class TestExactArithmetic:
    def test_known_means_winrate_std(self):
        # 30 BUY rows: 20×(fr=+5.0) then 10×(fr=-10.0); ml_score = i (1..30,
        # distinct). Hand-computable:
        #   mean_aligned_return = (20*5 + 10*-10)/30 = 0.0
        #   win_rate            = 20/30 = 0.6667
        #   population variance = (20*5^2 + 10*10^2)/30 = 1500/30 = 50
        #   std_return          = sqrt(50) ≈ 7.0711
        #   mean_signal         = mean(1..30) = 15.5
        frs = [5.0] * 20 + [-10.0] * 10
        rows = _rows_for(1, [(float(i + 1), frs[i]) for i in range(30)])
        e = persona_skill(rows)["personas"][0]
        assert e["mean_aligned_return"] == 0.0
        assert e["win_rate"] == 0.6667
        assert e["std_return"] == round(math.sqrt(50.0), 4) == 7.0711
        assert e["mean_signal"] == 15.5

    def test_sell_alignment_flows_through_aggregate(self):
        # 30 SELL rows: ml_score = -i (stronger conviction as i grows),
        # fr = -i (price dropped → correct sell). _aligned double-flips ⇒
        # signal = +i, target = +i ⇒ perfect rank corr, all-positive
        # aligned return ⇒ SIGNAL_EDGE, mean 15.5, win_rate 1.0.
        rows = _rows_for(1, [(-float(i), -float(i)) for i in range(1, 31)],
                         action="SELL")
        e = persona_skill(rows)["personas"][0]
        assert e["score_ic"] == 1.0
        assert e["mean_aligned_return"] == 15.5
        assert e["win_rate"] == 1.0
        assert e["verdict"] == "SIGNAL_EDGE"


# ───────────────────────── hardening / read-only ─────────────────────────

class TestHardening:
    def test_unmappable_run_id_and_bad_rows_dropped_never_raises(self):
        good = _rows_for(1, _MONO)
        junk = [
            {"action": "BUY", "ml_score": 1.0, "forward_return_5d": 1.0},  # no run_id
            {"run_id": "x", "ml_score": 1.0, "forward_return_5d": 1.0},     # unmappable
            {"run_id": 1, "ml_score": None, "forward_return_5d": 1.0},      # non-finite
            {"run_id": 1, "ml_score": 1.0, "forward_return_5d": "nope"},    # unparseable
        ]
        rep = persona_skill(good + junk)
        # Only the 30 good rows survive into the single persona bucket.
        assert rep["n_records"] == 30
        assert rep["personas"][0]["n"] == 30

    def test_empty_input_is_insufficient(self):
        rep = persona_skill([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0
        assert rep["n_personas"] == 0

    def test_load_outcomes_missing_file_and_corrupt_lines(self, tmp_path):
        assert _load_outcomes(tmp_path / "nope.jsonl") == []
        p = tmp_path / "o.jsonl"
        p.write_text('{"run_id": 1, "ml_score": 2.0, "forward_return_5d": 3.0}\n'
                      "not json\n"
                      "[1,2,3]\n"
                      "\n"
                      '{"run_id": 2, "ml_score": 4.0, "forward_return_5d": 5.0}\n')
        rows = _load_outcomes(p)
        assert len(rows) == 2
        assert rows[0]["run_id"] == 1 and rows[1]["run_id"] == 2


class TestThresholdsAreModuleConstants:
    def test_constants_have_expected_relationship(self):
        # Locks the verdict-band ordering the classification depends on.
        assert 0.0 < IC_MIN < IC_GOOD
        assert MIN_OUTCOMES_PER_PERSONA >= 2
        assert MIN_RECORDS >= MIN_OUTCOMES_PER_PERSONA
