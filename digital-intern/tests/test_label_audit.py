"""ml.label_audit — training-pool integrity audit.

Asserts exact bucket counts, the mutual-exclusion reconciliation invariant,
and the load-bearing rule that score_source='ml' rows can NEVER be counted in
the strong-label pool (mirrors tests/test_trainer.py's exclusion contract via
the shared ml.trainer.STRONG_LABEL_WHERE predicate).
"""
from __future__ import annotations

import json

from ml import label_audit


def _insert(store, *, id, title, ai_score, score_source, kw_score=0.0,
            url=None, source="rss"):
    if url is None:
        url = f"https://x.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, 0,
             "2026-05-15T00:00:00+00:00", 0, None, score_source),
        )
        store.conn.commit()


def _seed_mixed(store):
    """A controlled population spanning every bucket the audit distinguishes.

    Strong pool (9): 3 llm + 1 briefing_boost + 3 synthetic + 2 heuristic.
    Excluded: a NULL non-integer non-synthetic row, 2 ml@8.5, 1 ml@0.
    """
    for i in range(3):
        _insert(store, id=f"llm{i}", title=f"llm {i}",
                ai_score=7.0, score_source="llm")
    _insert(store, id="brief0", title="briefing nudge",
            ai_score=6.0, score_source="briefing_boost")
    # Pre-migration heuristic rows: NULL score_source, whole-number ai_score.
    for i in range(2):
        _insert(store, id=f"leg{i}", title=f"legacy {i}",
                ai_score=4.0, score_source=None)
    # NULL, non-integer, non-synthetic → fails the integer heuristic and is
    # not synthetic → must be in NO bucket and NOT in the strong pool.
    _insert(store, id="nullfrac", title="null fractional",
            ai_score=3.5, score_source=None)
    # Synthetic backtest winners (integer ai_score, but synthetic, so they
    # belong to the synthetic bucket, never the heuristic one).
    _insert(store, id="bt1", title="backtest winner one",
            ai_score=5.0, score_source=None,
            url="backtest://run_1/d/BUY/MU", source="backtest_run_1_winner")
    _insert(store, id="bt2", title="backtest winner two",
            ai_score=5.0, score_source=None,
            url="backtest://run_1/d/BUY/NV", source="backtest_run_1_winner")
    # Opus annotation (non-integer ai_score) → synthetic via source prefix.
    _insert(store, id="op1", title="opus neutral",
            ai_score=2.5, score_source=None,
            url="https://x.com/op1", source="opus_annotation_cycle_1")
    # Model self-predictions in the label column: a hygiene bug, and they
    # must stay out of the strong pool entirely.
    for i in range(2):
        _insert(store, id=f"ml{i}", title=f"ml pred {i}",
                ai_score=8.5, score_source="ml")
    _insert(store, id="mlzero", title="ml zeroed",
            ai_score=0.0, score_source="ml")


def test_bucket_counts_and_reconciliation(store):
    _seed_mixed(store)
    r = label_audit.audit(store)
    sp = r["strong_pool"]

    assert sp["llm"] == 3
    assert sp["briefing_boost"] == 1
    assert sp["synthetic_backtest_opus"] == 3      # bt1, bt2, op1
    assert sp["heuristic_null_integer"] == 2       # leg0, leg1
    assert sp["total"] == 9
    assert sp["reconciles"] is True                # 3+1+3+2 == 9

    assert r["heuristic_trust_gap"] == 2
    assert r["heuristic_fraction_of_strong"] == round(2 / 9, 4)
    assert r["column_hygiene_violations"] == 2     # the two ml@8.5
    assert r["ml_predictions_total"] == 3          # ml0, ml1, mlzero
    # Hygiene violation present ⇒ overall verdict must fail.
    assert r["ok"] is False


def test_ml_rows_never_enter_strong_pool(store):
    """The core invariant: even a high-ai_score score_source='ml' row is
    invisible to the strong-label pool. Only the single llm row counts."""
    _insert(store, id="real", title="real claude label",
            ai_score=9.0, score_source="llm")
    for i in range(20):
        _insert(store, id=f"selfpred{i}", title=f"self pred {i}",
                ai_score=9.5, score_source="ml")

    r = label_audit.audit(store)
    assert r["strong_pool"]["total"] == 1
    assert r["strong_pool"]["llm"] == 1
    assert r["strong_pool"]["reconciles"] is True
    assert r["column_hygiene_violations"] == 20
    assert r["ml_predictions_total"] == 20
    assert r["ok"] is False


def test_clean_store_is_ok(store):
    """No ml-into-ai_score contamination ⇒ ok True and buckets reconcile."""
    for i in range(3):
        _insert(store, id=f"llm{i}", title=f"llm {i}",
                ai_score=7.0, score_source="llm")
    _insert(store, id="bt", title="bt winner",
            ai_score=5.0, score_source=None,
            url="backtest://run_2/d/BUY/AA", source="backtest_run_2_winner")
    _insert(store, id="leg", title="legacy int", ai_score=4.0,
            score_source=None)

    r = label_audit.audit(store)
    assert r["strong_pool"]["total"] == 5          # 3 llm + 1 synth + 1 heur
    assert r["strong_pool"]["reconciles"] is True
    assert r["column_hygiene_violations"] == 0
    assert r["ok"] is True


def test_empty_store_does_not_divide_by_zero(store):
    r = label_audit.audit(store)
    assert r["strong_pool"]["total"] == 0
    assert r["heuristic_fraction_of_strong"] == 0.0
    assert r["strong_pool"]["reconciles"] is True  # 0 == 0
    assert r["ok"] is True


def test_format_report_emits_valid_json(store):
    _seed_mixed(store)
    parsed = json.loads(label_audit.format_report(label_audit.audit(store)))
    assert parsed["strong_pool"]["total"] == 9
    assert parsed["ok"] is False
