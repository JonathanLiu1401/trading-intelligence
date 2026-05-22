"""Tests for paper_trader.ml.corpus_diversity — the gate-eligibility audit.

Verifies the module faithfully reproduces train_scorer's dedup + label
validation so its `trainable_n` is the number that decides whether the
`_ml_decide` conviction gate (n_train >= 500) wakes up.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import corpus_diversity as cd


def _rec(ticker="NVDA", sim_date="2025-01-02", action="BUY",
         fwd=3.0, return_pct=10.0, run_id=1):
    return {
        "ticker": ticker, "sim_date": sim_date, "action": action,
        "forward_return_5d": fwd, "return_pct": return_pct, "run_id": run_id,
    }


def _distinct_recs(n, fwd=3.0):
    """n records with distinct (ticker,sim_date) keys — no collisions."""
    out = []
    for i in range(n):
        out.append(_rec(ticker=f"T{i % 50}",
                         sim_date=f"2025-{(i // 28) % 12 + 1:02d}-"
                                  f"{i % 28 + 1:02d}",
                         action="BUY" if i % 4 else "SELL",
                         fwd=fwd, run_id=i % 7))
    return out


# ─────────────────────────── dedup correctness ───────────────────────────

def test_distinct_count_no_collisions():
    recs = [_rec(ticker=f"X{i}", sim_date="2025-03-03") for i in range(40)]
    rep = cd.corpus_diversity_report(recs)
    assert rep["raw_n"] == 40
    assert rep["distinct_n"] == 40
    assert rep["dedup_ratio"] == 1.0


def test_dedup_collapses_same_key():
    # 5 raw rows, all (NVDA, 2025-01-02, BUY) — collapse to 1 distinct sample.
    recs = [_rec(return_pct=float(i)) for i in range(5)]
    rep = cd.corpus_diversity_report(recs)
    assert rep["raw_n"] == 5
    assert rep["distinct_n"] == 1
    assert rep["dedup_ratio"] == 5.0


def test_dedup_key_is_run_id_free():
    # Same (ticker,sim_date,action) across two run_ids must still collapse —
    # train_scorer's key omits run_id (unlike outcome_data_quality's key).
    recs = [_rec(run_id=1), _rec(run_id=2), _rec(run_id=3)]
    rep = cd.corpus_diversity_report(recs)
    assert rep["distinct_n"] == 1


def test_action_distinguishes_dedup_key():
    # A BUY and a SELL of the same ticker/date are distinct samples.
    recs = [_rec(action="BUY"), _rec(action="SELL")]
    rep = cd.corpus_diversity_report(recs)
    assert rep["distinct_n"] == 2
    assert rep["action_mix"] == {"BUY": 1, "SELL": 1}


# ─────────────────────────── label validation ───────────────────────────

@pytest.mark.parametrize("bad", [None, True, False, float("nan"),
                                 float("inf"), "not-a-number"])
def test_bad_label_excluded_from_trainable(bad):
    # 35 valid distinct + 1 distinct row with an unusable forward_return_5d.
    recs = [_rec(ticker=f"V{i}") for i in range(35)]
    recs.append(_rec(ticker="BADLBL"))
    recs[-1]["forward_return_5d"] = bad
    rep = cd.corpus_diversity_report(recs)
    assert rep["distinct_n"] == 36
    assert rep["trainable_n"] == 35
    assert rep["n_dropped_bad_label"] == 1


def test_valid_finite_label_counts():
    recs = [_rec(ticker=f"V{i}", fwd=-12.5) for i in range(30)]
    rep = cd.corpus_diversity_report(recs)
    assert rep["trainable_n"] == 30
    assert rep["n_dropped_bad_label"] == 0


# ─────────────────────────── verdicts ───────────────────────────

def test_insufficient_data_below_min_records():
    rep = cd.corpus_diversity_report([_rec(ticker=f"S{i}") for i in range(29)])
    assert rep["verdict"] == "INSUFFICIENT_DATA"


def test_gate_starved_below_floor():
    # 100 distinct trainable samples — above MIN_RECORDS, below GATE_MIN_N.
    rep = cd.corpus_diversity_report(_distinct_recs(100))
    assert rep["verdict"] == "GATE_STARVED"
    assert rep["gate_eligible"] is False
    assert rep["gate_shortfall"] == cd.GATE_MIN_N - 100


def test_gate_eligible_at_floor():
    rep = cd.corpus_diversity_report(_distinct_recs(cd.GATE_MIN_N))
    assert rep["trainable_n"] == cd.GATE_MIN_N
    assert rep["verdict"] == "GATE_ELIGIBLE"
    assert rep["gate_eligible"] is True
    assert rep["gate_shortfall"] == 0


def test_gate_starved_one_below_floor():
    rep = cd.corpus_diversity_report(_distinct_recs(cd.GATE_MIN_N - 1))
    assert rep["verdict"] == "GATE_STARVED"
    assert rep["gate_shortfall"] == 1


def test_collisions_keep_gate_starved_despite_many_raw_rows():
    # 5000 raw rows but only 50 distinct keys — the dedup-collapse scenario.
    recs = []
    for i in range(5000):
        recs.append(_rec(ticker=f"K{i % 50}", sim_date="2025-06-01",
                          return_pct=float(i)))
    rep = cd.corpus_diversity_report(recs)
    assert rep["raw_n"] == 5000
    assert rep["distinct_n"] == 50
    assert rep["trainable_n"] == 50
    assert rep["verdict"] == "GATE_STARVED"
    assert rep["dedup_ratio"] == 100.0


# ─────────────────────────── diversity descriptors ───────────────────────────

def test_distinct_descriptors():
    recs = [
        _rec(ticker="AAA", sim_date="2025-01-01", run_id=1),
        _rec(ticker="BBB", sim_date="2025-01-01", run_id=1),
        _rec(ticker="AAA", sim_date="2025-01-02", run_id=2),
    ]
    rep = cd.corpus_diversity_report(recs)
    assert rep["distinct_tickers"] == 2
    assert rep["distinct_sim_dates"] == 2
    assert rep["distinct_run_ids"] == 2


def test_top_collisions_surfaces_worst_key():
    recs = [_rec(ticker=f"U{i}") for i in range(30)]  # distinct filler
    recs += [_rec(ticker="HOT", sim_date="2025-09-09", return_pct=float(i))
             for i in range(7)]  # 7-way collision
    rep = cd.corpus_diversity_report(recs)
    assert rep["top_collisions"][0] == {
        "ticker": "HOT", "sim_date": "2025-09-09",
        "action": "BUY", "count": 7,
    }


def test_dedup_keeps_highest_return_pct_copy():
    # The surviving record should carry return_pct=99 (highest), so its
    # forward_return_5d (the winner's) is what counts toward trainable.
    recs = [
        _rec(return_pct=1.0, fwd=None),    # bad label, low return
        _rec(return_pct=99.0, fwd=4.4),    # good label, high return — kept
        _rec(return_pct=50.0, fwd=None),   # bad label, mid return
    ]
    rep = cd.corpus_diversity_report(recs)
    assert rep["distinct_n"] == 1
    # The kept copy has a valid label, so the sample is trainable.
    assert rep["trainable_n"] == 1
    assert rep["n_dropped_bad_label"] == 0


# ─────────────────────────── load + analyze ───────────────────────────

def test_load_outcomes_missing_file(tmp_path):
    assert cd.load_outcomes(tmp_path / "nope.jsonl") == []


def test_load_outcomes_skips_corrupt_lines(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text('{"ticker": "NVDA"}\nnot json\n[1,2,3]\n{"ticker": "AMD"}\n')
    rows = cd.load_outcomes(p)
    assert len(rows) == 2  # two valid dicts; non-dict + garbage skipped


def test_empty_records_never_raises():
    rep = cd.corpus_diversity_report([])
    assert rep["verdict"] == "INSUFFICIENT_DATA"
    assert rep["raw_n"] == 0
    assert rep["dedup_ratio"] is None


def test_analyze_applies_tail_cap_and_split(tmp_path):
    p = tmp_path / "decision_outcomes.jsonl"
    # 600 distinct rows; cap the tail to 200 → split 80/20 → 160 train slice.
    recs = _distinct_recs(600)
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    rep = cd.analyze(p, max_outcomes=200, oos_fraction=0.2)
    assert rep["file_raw_n"] == 600
    assert rep["tail_n"] == 200
    assert rep["train_slice_n"] == 160
    assert rep["raw_n"] == 160


def test_analyze_trainable_matches_train_scorer(tmp_path):
    """analyze()'s trainable_n must equal the n_train train_scorer pickles
    from the same file — the no-drift cross-check the module promises."""
    from paper_trader.ml.decision_scorer import train_scorer, SCORER_PATH
    import pickle

    recs = _distinct_recs(700)
    p = tmp_path / "decision_outcomes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    rep = cd.analyze(p, max_outcomes=5000, oos_fraction=0.2)

    # Reproduce the loop: tail → temporal split → train_scorer(train slice).
    from paper_trader.validation import split_outcomes_temporal
    train, _ = split_outcomes_temporal(recs, oos_fraction=0.2)
    result = train_scorer(train)
    # train_scorer writes the pickle as a side effect; read n_train back.
    with SCORER_PATH.open("rb") as fh:
        n_pickled = pickle.load(fh)["n_train"]

    assert result["status"] == "ok"
    assert rep["trainable_n"] == n_pickled == result["n"]


# ─────────────────────────── CLI ───────────────────────────

def test_cli_runs_against_live_corpus(capsys):
    rc = cd._cli()
    out = capsys.readouterr().out
    assert "VERDICT:" in out
    assert rc in (0, 2)
