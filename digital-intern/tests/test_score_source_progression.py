"""End-to-end score_source progression invariants — the ladder
NULL → 'ml' → 'llm' (and the stable terminal at 'llm').

Existing test_article_store.py pins the individual batch-method invariants
(``update_ml_scores_batch`` writes 'ml', ``update_ai_scores_batch`` writes
'llm', neither clobbers an existing 'llm', briefing_boost writes
'briefing_boost'). What is NOT pinned anywhere is the *sequential*
progression a real article actually walks through: inserted (NULL) →
model-scored ('ml') → Sonnet-relabelled ('llm'), with each transition
preserving the load-bearing separation between ``ai_score`` (LLM ground
truth) and ``ml_score`` (model self-prediction).

A regression that flips the COALESCE direction in ``update_ml_scores_batch``
would still pass every per-method test (the 'ml' → 'llm' single-step test
would not catch it) but would break this multi-step ladder. So this file
is the cross-method drift guard the per-method tests cannot be.

Touches NO real DB — uses the conftest ``store`` fixture that redirects
``_get_db_path`` to a per-test tmp file.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_unscored(store, aid: str = "a", url: str = "https://reuters.com/a",
                     title: str = "Real news article", source: str = "rss"):
    """Insert an unscored row directly so we control the starting state."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, url, title, source, "", 1.0, 0.0, 0,
             _recent_iso(), 0, None, None),
        )
        store.conn.commit()


def _state(store, aid: str = "a") -> tuple:
    """Return ``(ai_score, ml_score, score_source, urgency)`` for ``aid``."""
    return store.conn.execute(
        "SELECT ai_score, ml_score, score_source, urgency "
        "FROM articles WHERE id=?",
        (aid,),
    ).fetchone()


class TestScoreSourceLadder:
    def test_full_progression_null_then_ml_then_llm(self, store):
        """A real article's lifetime: NULL → model says urgent → Sonnet
        ratifies → stable. Each transition must preserve the ai_score vs
        ml_score split."""
        _insert_unscored(store)
        # Starting state: every score column is empty / unset.
        assert _state(store) == (0.0, None, None, 0)

        # Step 1 — model scores it urgent.
        store.update_ml_scores_batch([("a", 9.0, 1)])
        ai, ml, src, urg = _state(store)
        assert ai == 0.0, "model output must NEVER pollute ai_score"
        assert ml == 9.0
        assert src == "ml"
        assert urg == 1

        # Step 2 — Sonnet relabels it. score_source becomes 'llm', and the
        # model's ml_score is preserved (so a downstream chart could show
        # "model said 9.0; Sonnet ratified at 8.5" without losing either).
        store.update_ai_scores_batch([("a", 8.5, 1)])
        ai, ml, src, urg = _state(store)
        assert ai == 8.5
        assert ml == 9.0, "ml_score must survive the LLM relabel"
        assert src == "llm"
        assert urg == 1

        # Step 3 — terminal: Sonnet re-scores the same row (recursive_labeler
        # path). score_source stays 'llm', ai_score updates.
        store.update_ai_scores_batch([("a", 7.0, 0)])
        ai, ml, src, urg = _state(store)
        assert ai == 7.0
        assert ml == 9.0, "ml_score still preserved"
        assert src == "llm"
        # urgency=MAX(urgency, 0) → previous urgency=1 is preserved
        # (documented MAX semantics; the stale-urgent reaper handles aging).
        assert urg == 1

    def test_ml_after_llm_is_a_no_op_on_score_source(self, store):
        """A late model write on an already-LLM-labelled row must NOT
        downgrade score_source. COALESCE(score_source, 'ml') is the guard;
        flipping it to plain 'ml' would corrupt the training pool by
        re-tagging real Sonnet labels as model self-predictions."""
        _insert_unscored(store)
        store.update_ai_scores_batch([("a", 8.0, 1)])
        # Late model write — happens if the daemon re-scored a row that
        # someone backdoor-cleared via a manual ai_score reset.
        store.update_ml_scores_batch([("a", 4.0, 0)])
        ai, ml, src, urg = _state(store)
        assert ai == 8.0, "LLM ai_score must stand"
        assert ml == 4.0, "model write to ml_score is independent"
        assert src == "llm", (
            "score_source must stay 'llm' — flipping back to 'ml' would "
            "make ArticleNet train on its own output as ground truth"
        )

    def test_briefing_boost_promoted_to_llm_when_sonnet_speaks(self, store):
        """An Opus-curated briefing_boost row that later gets Sonnet-labelled
        must terminate at 'llm', not stay at 'briefing_boost' — Sonnet's
        score is the stronger ground truth and the trainer weighs them
        identically anyway."""
        _insert_unscored(store)
        # Simulate the briefing path: update_scores_from_labels marks
        # in_briefing rows briefing_boost with ai_score=4.5.
        store.update_scores_from_labels([
            {"url": "https://reuters.com/a", "in_briefing": True},
        ])
        ai, ml, src, urg = _state(store)
        assert ai == 4.5
        assert src == "briefing_boost"

        # Sonnet then explicitly labels the same row.
        store.update_ai_scores_batch([("a", 9.0, 1)])
        ai, ml, src, urg = _state(store)
        assert ai == 9.0
        assert src == "llm"

    def test_synthetic_backtest_progression_isolated_from_live_reads(self, store):
        """A synthetic backtest row walks the SAME progression in storage
        (so the trainer's STRONG_LABEL_WHERE keeps catching it) but every
        live read must keep ignoring it. Pins the cross-method intersection
        of the score_source ladder and the backtest-isolation invariant."""
        _insert_unscored(
            store, aid="bt",
            url="backtest://run_42/2026-05-29/BUY/MU",
            title="Synthetic backtest training row",
            source="backtest_run_42_winner",
        )
        # The model "scores" the synthetic row — in practice get_unscored
        # filters it so this never happens, but storage must still tolerate
        # the call shape without corrupting state.
        store.update_ml_scores_batch([("bt", 8.0, 1)])
        ai, ml, src, urg = _state(store, "bt")
        assert (ai, ml, src, urg) == (0.0, 8.0, "ml", 1)

        # Live reads still skip it — the row is urgency=1 but synthetic.
        urgent = store.get_unalerted_urgent()
        assert urgent == [], (
            "synthetic backtest row leaked into urgent queue — the score "
            "ladder is not allowed to override backtest isolation"
        )
        top = store.get_top_for_briefing(hours=24, limit=10)
        assert top == [], (
            "synthetic backtest row reached briefing top — the score "
            "ladder is not allowed to override backtest isolation"
        )
