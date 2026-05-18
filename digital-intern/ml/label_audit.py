"""Training-label integrity audit (read-only).

digital-intern's single most load-bearing invariant (CLAUDE.md §5/§8) is the
separation between *labels* and *predictions*:

  * ``ai_score`` holds ground-truth labels — Sonnet/Opus ("llm"), the briefing
    curation nudge ("briefing_boost"), or synthetic backtest/opus rows
    (``score_source`` NULL). These are what ``ArticleNet`` learns from.
  * ``ml_score`` holds the model's OWN prediction; such rows carry
    ``score_source = 'ml'``. They must NEVER enter the trainer's strong-label
    pool — if they do, the model trains on its own output and the loss
    collapses to self-agreement instead of the Claude signal.

That invariant is currently only enforced by a one-time migration in
``storage/article_store.py`` and by the trainer's WHERE clause. Nothing
*continuously* answers "is the pool still clean, and how much of it is
provenance-tagged vs heuristically inferred?". This module does, by counting
against the exact ``ml.trainer.STRONG_LABEL_WHERE`` predicate the model trains
on (imported verbatim — it can never drift from training reality).

Two distinct findings, deliberately reported separately:

  * ``column_hygiene_violations`` — rows with ``score_source='ml'`` that also
    carry ``ai_score > 0``. This is a *code bug* (a writer put a model
    prediction in the label column). It is NOT, by itself, a training leak:
    ``STRONG_LABEL_WHERE`` already excludes ``score_source='ml'``. But it
    means the two columns are no longer cleanly separated and a future
    predicate change could expose it.
  * ``heuristic_trust_gap`` — rows that enter the strong pool *only* via the
    "score_source NULL + whole-number ai_score" pre-migration heuristic. Their
    provenance is inferred, not tagged: a model prediction that ever landed in
    ``ai_score`` with an integer value would be indistinguishable here. Not a
    confirmed leak — it is the size of the pool's trust gap.

Run standalone::

    python3 -m ml.label_audit          # JSON report; exit 0 ok, 1 if dirty
"""
from __future__ import annotations

import json
from typing import Optional

from ml.trainer import STRONG_LABEL_WHERE

# A model prediction sitting in the label column. Trainer excludes
# score_source='ml', so this does not (today) reach training — it is a
# column-separation bug, reported on its own so it is never conflated with an
# actual training leak.
_HYGIENE_VIOLATION_WHERE = "score_source = 'ml' AND ai_score > 0"

# The strong-pool rows whose "this is a Claude/synthetic label" status is
# *inferred* from an integer ai_score rather than an explicit score_source tag,
# and which are not synthetic backtest/opus rows. This is the audit's trust
# gap: the larger it is relative to the explicitly-tagged 'llm' count, the less
# of the training signal is verifiably a ground-truth label.
_HEURISTIC_TRUST_WHERE = (
    "score_source IS NULL AND ai_score > 0 "
    "AND ai_score = CAST(ai_score AS INTEGER) "
    "AND url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_SYNTHETIC_WHERE = (
    "score_source IS NULL AND (url LIKE 'backtest://%' "
    "OR source LIKE 'backtest_%' OR source LIKE 'opus_annotation%')"
)


def _count(conn, where: str) -> int:
    """Read-only COUNT(*). Safe against the live DB — WAL readers never block
    the daemon's writers and add no lock contention."""
    return int(
        conn.execute(f"SELECT COUNT(*) FROM articles WHERE {where}").fetchone()[0]
    )


def audit(store) -> dict:
    """Return an integrity report for the trainer's strong-label pool.

    The four strong-pool buckets are mutually exclusive and exhaustive by
    construction of ``STRONG_LABEL_WHERE`` (branch ``score_source IN
    ('llm','briefing_boost')`` is disjoint from the two ``score_source IS
    NULL`` branches; within the NULL branches, synthetic and
    non-synthetic-integer rows partition cleanly), so ``reconciles`` must
    always be True on an uncorrupted store — a False value is itself a finding.
    """
    conn = store.conn

    strong_total = _count(conn, STRONG_LABEL_WHERE)
    strong_llm = _count(conn, f"({STRONG_LABEL_WHERE}) AND score_source = 'llm'")
    strong_briefing = _count(
        conn, f"({STRONG_LABEL_WHERE}) AND score_source = 'briefing_boost'"
    )
    strong_synthetic = _count(
        conn, f"({STRONG_LABEL_WHERE}) AND {_SYNTHETIC_WHERE}"
    )
    heuristic_gap = _count(conn, _HEURISTIC_TRUST_WHERE)

    hygiene_violations = _count(conn, _HYGIENE_VIOLATION_WHERE)
    ml_predictions = _count(conn, "score_source = 'ml'")

    accounted = strong_llm + strong_briefing + strong_synthetic + heuristic_gap
    reconciles = accounted == strong_total
    heuristic_fraction = (
        round(heuristic_gap / strong_total, 4) if strong_total else 0.0
    )

    return {
        "strong_pool": {
            "total": strong_total,
            "llm": strong_llm,
            "briefing_boost": strong_briefing,
            "synthetic_backtest_opus": strong_synthetic,
            "heuristic_null_integer": heuristic_gap,
            "reconciles": reconciles,
        },
        # CODE-BUG signal, not a training leak (trainer excludes
        # score_source='ml'). Nonzero ⇒ some writer is putting model output
        # into the ai_score label column.
        "column_hygiene_violations": hygiene_violations,
        # Share of the strong pool whose ground-truth status is inferred from
        # an integer ai_score rather than an explicit score_source tag.
        "heuristic_trust_gap": heuristic_gap,
        "heuristic_fraction_of_strong": heuristic_fraction,
        "ml_predictions_total": ml_predictions,
        # Overall verdict: the pool is clean AND the buckets reconcile.
        "ok": hygiene_violations == 0 and reconciles,
    }


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


class _RoStore:
    """Minimal ``.conn``-bearing shim over a read-only SQLite connection.

    The standalone CLI must NOT instantiate a writable ``ArticleStore``: that
    opens the live production DB read/write and runs the score_source
    migration ``UPDATE``s, which block on ``busy_timeout`` (60s) under the
    daemon's writer-contention storm. ``audit`` only ever issues ``COUNT(*)``,
    so a ``mode=ro`` connection is correct, fast, and adds zero lock
    contention — exactly what this module's docstring promises.
    """

    def __init__(self, db_path) -> None:
        import sqlite3

        self.conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=15
        )

    def close(self) -> None:
        self.conn.close()


def main(argv: Optional[list] = None) -> int:
    from storage.article_store import _get_db_path

    store = _RoStore(_get_db_path())
    try:
        report = audit(store)
    finally:
        store.close()
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
