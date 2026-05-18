"""Exact-value locks for the training-corpus / OOS-construction audit
(`paper_trader/ml/corpus_audit.py`, 2026-05-18 quant feature).

Mirrors test_skill_trend.py / test_regime_audit.py: deterministic synthetic
data, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline; reuses the REAL
`validation.split_outcomes_temporal` (the split the continuous loop uses) so
the test pins the audit against the exact production holdout construction.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from paper_trader.ml import corpus_audit as ca


def _rec(run_id, day_offset, regime_mult=1.0, action="BUY"):
    return {
        "run_id": run_id,
        "sim_date": (date(2020, 1, 1) + timedelta(days=day_offset)).isoformat(),
        "regime_mult": regime_mult,
        "action": action,
        "forward_return_5d": 1.0,
    }


class TestLoadOutcomes:
    def test_missing_file_yields_empty(self, tmp_path):
        assert ca.load_outcomes(tmp_path / "nope.jsonl") == []

    def test_corrupt_and_nondict_lines_skipped(self, tmp_path):
        p = tmp_path / "o.jsonl"
        p.write_text('{"run_id": 1, "sim_date": "2020-01-01"}\n'
                      "not json\n"
                      "[1,2,3]\n"
                      "\n"
                      '{"run_id": 2, "sim_date": "2020-01-02"}\n')
        rows = ca.load_outcomes(p)
        assert len(rows) == 2
        assert rows[0]["run_id"] == 1 and rows[1]["run_id"] == 2


class TestRegimeLabel:
    def test_known_multipliers(self):
        assert ca._regime_label(0.3) == "bear"
        assert ca._regime_label(0.6) == "sideways"
        assert ca._regime_label(1.0) == "bull_or_unknown"

    def test_float_noise_rounds(self):
        assert ca._regime_label(0.6000000001) == "sideways"

    def test_missing_or_bad_is_unmapped(self):
        assert ca._regime_label(None) == "unmapped"
        assert ca._regime_label("x") == "unmapped"
        assert ca._regime_label(float("nan")) == "unmapped"
        assert ca._regime_label(0.42) == "unmapped"


class TestInsufficientData:
    def test_empty_corpus(self):
        rep = ca.corpus_audit_report([])
        assert rep["status"] == "ok"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_below_min_records(self):
        recs = [_rec(1, i) for i in range(ca.MIN_RECORDS - 1)]
        rep = ca.corpus_audit_report(recs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == ca.MIN_RECORDS - 1
        # Composition descriptors are still populated below the verdict gate.
        assert rep["n_distinct_run_ids"] == 1
        assert rep["corpus_breadth"] == "SINGLE_DRAW"

    def test_no_oos_slice_when_fraction_is_one(self):
        # split_outcomes_temporal: n_oos = max(1, int(n*1.0)) == n >= n
        # ⇒ returns (all, []) ⇒ no holdout to assess.
        recs = [_rec(1, i) for i in range(40)]
        rep = ca.corpus_audit_report(recs, oos_fraction=1.0)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["oos_n"] == 0
        assert "no OOS slice" in rep["hint"]


class TestComposition:
    def test_regime_mix_and_dominant_fraction_exact(self):
        recs = []
        recs += [_rec(1, i, regime_mult=1.0) for i in range(15)]
        recs += [_rec(1, 15 + i, regime_mult=0.6) for i in range(9)]
        recs += [_rec(1, 24 + i, regime_mult=0.3) for i in range(5)]
        recs += [_rec(1, 29, regime_mult=None)]          # → unmapped
        rep = ca.corpus_audit_report(recs)
        assert rep["n"] == 30
        assert rep["regime_mix"] == {
            "bear": 5, "bull_or_unknown": 15, "sideways": 9, "unmapped": 1,
        }
        assert rep["dominant_regime_fraction"] == pytest.approx(0.5)

    def test_run_id_zero_not_coerced_and_none_sorts_last(self):
        recs = [_rec(0, i) for i in range(20)]
        recs += [_rec(None, 20 + i) for i in range(20)]
        rep = ca.corpus_audit_report(recs)
        assert rep["n_distinct_run_ids"] == 2
        # 0 kept as a real id; None placed last by the sort key.
        assert rep["run_id_counts"] == [[0, 20], [None, 20]]

    def test_sim_date_span(self):
        recs = [_rec(1, i) for i in range(30)]
        rep = ca.corpus_audit_report(recs)
        assert rep["sim_date_min"] == "2020-01-01"
        assert rep["sim_date_max"] == (date(2020, 1, 1)
                                       + timedelta(days=29)).isoformat()
        assert rep["n_distinct_sim_dates"] == 30


class TestVerdicts:
    def test_oos_not_held_out_single_draw(self):
        # 60 rows, 3 runs interleaved across the whole date range so every run
        # contributes to BOTH the early-80% train and the late-20% OOS.
        recs = [_rec(101 + (i % 3), i) for i in range(60)]
        rep = ca.corpus_audit_report(recs)
        assert rep["verdict"] == "OOS_NOT_HELD_OUT"
        assert rep["n_distinct_run_ids"] == 3
        assert rep["corpus_breadth"] == "SINGLE_DRAW"
        assert rep["likely_single_cycle"] is True
        assert rep["oos_shares_all_runs_with_train"] is True
        assert rep["oos_run_ids_not_in_train"] == 0
        assert rep["train_n"] == 48 and rep["oos_n"] == 12
        assert rep["train_sim_date_max"] < rep["oos_sim_date_min"]

    def test_oos_overlaps_train_diverse_corpus(self):
        # 150 rows, 15 runs interleaved → every run in train AND oos, but the
        # corpus spans > NARROW_MAX_RUNS distinct runs ⇒ milder verdict.
        recs = [_rec(101 + (i % 15), i) for i in range(150)]
        rep = ca.corpus_audit_report(recs)
        assert rep["verdict"] == "OOS_OVERLAPS_TRAIN"
        assert rep["n_distinct_run_ids"] == 15
        assert rep["corpus_breadth"] == "DIVERSE"
        assert rep["oos_shares_all_runs_with_train"] is True
        assert rep["oos_run_ids_not_in_train"] == 0

    def test_oos_held_out_when_runs_disjoint(self):
        # Runs 201/202/203 on the early dates; run 999 ONLY on the latest
        # dates ⇒ the OOS slice contains a run train never saw.
        recs = [_rec(201 + (i % 3), i) for i in range(40)]
        recs += [_rec(999, 40 + i) for i in range(10)]
        rep = ca.corpus_audit_report(recs)
        assert rep["verdict"] == "OOS_HELD_OUT"
        assert rep["oos_run_ids_not_in_train"] == 1
        assert rep["oos_shares_all_runs_with_train"] is False
        assert rep["n_oos_run_ids"] == 1

    def test_narrow_boundary_is_overlap_not_alarm(self):
        # Exactly NARROW_MAX_RUNS distinct runs, all in both folds → still the
        # alarm (n_runs <= NARROW_MAX_RUNS is inclusive).
        nruns = ca.NARROW_MAX_RUNS
        recs = [_rec(300 + (i % nruns), i) for i in range(nruns * 12)]
        rep = ca.corpus_audit_report(recs)
        assert rep["n_distinct_run_ids"] == nruns
        assert rep["verdict"] == "OOS_NOT_HELD_OUT"
        # One more distinct run flips it to the milder verdict.
        recs2 = [_rec(300 + (i % (nruns + 1)), i)
                 for i in range((nruns + 1) * 12)]
        rep2 = ca.corpus_audit_report(recs2)
        assert rep2["n_distinct_run_ids"] == nruns + 1
        assert rep2["verdict"] == "OOS_OVERLAPS_TRAIN"


class TestNeverRaises:
    def test_split_failure_degrades_gracefully(self, monkeypatch):
        import paper_trader.validation as v

        def _boom(*a, **k):
            raise RuntimeError("split exploded")

        monkeypatch.setattr(v, "split_outcomes_temporal", _boom)
        recs = [_rec(1, i) for i in range(40)]
        rep = ca.corpus_audit_report(recs)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "split unavailable" in rep["hint"]
        # Composition still computed before the split attempt.
        assert rep["n_distinct_run_ids"] == 1

    def test_analyze_missing_file(self, tmp_path):
        rep = ca.analyze(tmp_path / "absent.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0


class TestCliExitCode:
    def test_exit_2_on_oos_not_held_out(self, monkeypatch):
        monkeypatch.setattr(ca, "analyze",
                            lambda *_a, **_k: ca.corpus_audit_report(
                                [_rec(101 + (i % 3), i) for i in range(60)]))
        assert ca._cli() == 2

    def test_exit_0_on_held_out(self, monkeypatch):
        recs = [_rec(201 + (i % 3), i) for i in range(40)]
        recs += [_rec(999, 40 + i) for i in range(10)]
        monkeypatch.setattr(ca, "analyze",
                            lambda *_a, **_k: ca.corpus_audit_report(recs))
        assert ca._cli() == 0
