"""Briefing-boost labeling invariant — ``ArticleStore.update_scores_from_labels``.

This path runs on every 5h Opus heartbeat (daemon ``heartbeat_worker`` →
``store.update_scores_from_labels``): articles Opus curated into the briefing
get promoted into the training pool as mid-tier positive labels.

It is the only writer of ``score_source='briefing_boost'`` and is gated by two
documented invariants that a prior revision broke (see the method docstring):

  1. The formula is ``MAX(ai_score, 4.5)`` — it must NEVER downgrade a row that
     already carries a stronger LLM label (the old ``MIN(5.0, ai_score+0.3)``
     turned an 8.0 into 5.0), and an unscored briefing mention must enter the
     pool at a full 4.5 (the old formula made it 0.3 → "3% relevance" noise).
  2. ``score_source`` becomes ``'briefing_boost'`` for a non-LLM row but is
     preserved as ``'llm'`` when the row was already LLM ground truth — so the
     trainer's strongest-signal accounting stays correct.

These were directly exercised by nothing in the suite before this file: the
existing coverage only checked the trainer *consuming* briefing_boost rows
(``test_integration_pipeline.TestTrainerDataIntegrity``), never the method that
*produces* them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _insert(store, *, id, url, ai_score=0.0, score_source=None):
    """Insert a row bypassing the public API so a precise pre-state can be set."""
    first_seen = datetime.now(timezone.utc).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, f"title {id}", "rss", "", 1.0, ai_score, 0,
             first_seen, 0, None, score_source),
        )
        store.conn.commit()

    def _row():
        return store.conn.execute(
            "SELECT ai_score, score_source FROM articles WHERE id=?", (id,)
        ).fetchone()

    return _row


class TestBriefingBoost:
    def test_unscored_mention_becomes_4_5_briefing_boost(self, store):
        """An article Opus put in the briefing but Sonnet never scored must
        enter training at a full 4.5 tagged 'briefing_boost' — not the 0.3
        the buggy ``ai_score + 0.3`` formula produced."""
        row = _insert(store, id="u", url="https://x.com/u",
                      ai_score=0.0, score_source=None)
        n = store.update_scores_from_labels(
            [{"url": "https://x.com/u", "in_briefing": True}]
        )
        assert n == 1
        ai_score, src = row()
        assert ai_score == pytest.approx(4.5)
        assert src == "briefing_boost"

    def test_high_llm_label_not_downgraded(self, store):
        """The documented regression: ``MAX(ai_score, 4.5)`` must keep an 8.0
        LLM score at 8.0 (the old ``MIN(5.0, ...)`` clamped it to 5.0) and the
        row must stay tagged 'llm', not be relabeled 'briefing_boost'."""
        row = _insert(store, id="h", url="https://x.com/h",
                      ai_score=8.0, score_source="llm")
        n = store.update_scores_from_labels(
            [{"url": "https://x.com/h", "in_briefing": True}]
        )
        assert n == 1
        ai_score, src = row()
        assert ai_score == pytest.approx(8.0), "high LLM label was downgraded"
        assert src == "llm", "LLM ground-truth tag was clobbered"

    def test_low_llm_label_floored_to_4_5_but_stays_llm(self, store):
        """A weak LLM label below the 4.5 briefing floor is raised to 4.5
        (briefing inclusion is a strong positive signal) yet keeps its 'llm'
        tag — the score moves, the provenance does not."""
        row = _insert(store, id="lo", url="https://x.com/lo",
                      ai_score=2.0, score_source="llm")
        store.update_scores_from_labels(
            [{"url": "https://x.com/lo", "in_briefing": True}]
        )
        ai_score, src = row()
        assert ai_score == pytest.approx(4.5)
        assert src == "llm"

    def test_not_in_briefing_is_ignored(self, store):
        """Labels with in_briefing=False must not touch the row at all —
        otherwise every scanned article would be force-labeled 4.5."""
        row = _insert(store, id="n", url="https://x.com/n",
                      ai_score=0.0, score_source=None)
        n = store.update_scores_from_labels(
            [{"url": "https://x.com/n", "in_briefing": False}]
        )
        assert n == 0
        ai_score, src = row()
        assert ai_score == pytest.approx(0.0)
        assert src is None

    def test_missing_or_empty_url_ignored(self, store):
        """Defensive: a label dict with no usable url must be a no-op, not a
        crash (synthetic snapshot rows like the P&L block carry no url)."""
        assert store.update_scores_from_labels(
            [{"in_briefing": True}, {"url": "", "in_briefing": True},
             {"url": None, "in_briefing": True}]
        ) == 0

    def test_empty_label_list_returns_zero(self, store):
        assert store.update_scores_from_labels([]) == 0
