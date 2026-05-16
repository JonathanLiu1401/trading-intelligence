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


def _insert(store, *, id, url, ai_score=0.0, score_source=None, source="rss",
            ml_score=None):
    """Insert a row bypassing the public API so a precise pre-state can be set."""
    first_seen = datetime.now(timezone.utc).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, f"title {id}", source, "", 1.0, ai_score, 0,
             first_seen, 0, ml_score, score_source),
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

    def test_model_scored_row_promoted_off_ml_into_training_pool(self, store):
        """A row the local model scored (``score_source='ml'``, ``ai_score=0``,
        ``ml_score`` set) that Opus then curates into the briefing MUST be
        promoted to ``ai_score=4.5`` / ``score_source='briefing_boost'`` — it
        must NOT stay tagged ``'ml'``.

        Why this is load-bearing and not covered above: the trainer's strong
        pool is ``score_source IN ('llm','briefing_boost')`` — it deliberately
        ignores ``'ml'`` to keep the label-feedback loop closed. The CASE in
        ``update_scores_from_labels`` is ``WHEN score_source='llm' THEN 'llm'
        ELSE 'briefing_boost'`` precisely so an Opus-curated model row crosses
        into the trained-on pool. Every existing TestBriefingBoost case uses
        ``score_source`` of ``None`` or ``'llm'``; none exercises the ``'ml'``
        branch. If the CASE ever regressed to preserving any non-NULL source
        (``WHEN score_source IS NOT NULL THEN score_source ...``), an 'ml' row
        would stay 'ml' and Opus's strongest curation signal would be silently
        dropped from training with no test failing. ``ml_score`` is left
        untouched (the row keeps its model prediction)."""
        row = _insert(store, id="ml1", url="https://x.com/ml1",
                      ai_score=0.0, ml_score=7.2, score_source="ml")
        n = store.update_scores_from_labels(
            [{"url": "https://x.com/ml1", "in_briefing": True}]
        )
        assert n == 1
        ai_score, src = row()
        assert ai_score == pytest.approx(4.5), "model row not promoted to 4.5"
        assert src == "briefing_boost", (
            "model-scored row stayed 'ml' — Opus curation lost to the trainer"
        )
        ml_score = store.conn.execute(
            "SELECT ml_score FROM articles WHERE id=?", ("ml1",)
        ).fetchone()[0]
        assert ml_score == pytest.approx(7.2), "model prediction clobbered"

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


class TestBriefingBoostBacktestIsolation:
    """update_scores_from_labels is the only writer of 'briefing_boost', a tag
    the trainer reads as strong ground truth (same weight as 'llm'). The label
    list is derived from get_top_for_briefing() (already live-only) so a
    synthetic URL cannot reach here on the production path — but the write must
    still refuse to rewrite a synthetic row's outcome label if it ever does,
    exactly like every other live path. A regression here silently poisons the
    training pool: a backtest SELL-loser's 0.5 outcome label would be promoted
    to 4.5 'briefing_boost' and the model would learn losing trades as decent.
    """

    def test_backtest_url_row_is_not_boosted(self, store):
        row = _insert(store, id="bt", url="backtest://run_7/2026-01-01/SELL/MU",
                      ai_score=0.5, score_source=None,
                      source="backtest_run_7_loser")
        n = store.update_scores_from_labels(
            [{"url": "backtest://run_7/2026-01-01/SELL/MU", "in_briefing": True}]
        )
        assert n == 0, "backtest row was boosted into the briefing_boost pool"
        ai_score, src = row()
        assert ai_score == pytest.approx(0.5), "synthetic outcome label rewritten"
        assert src is None, "synthetic row tag flipped to briefing_boost"

    def test_opus_annotation_source_row_is_not_boosted(self, store):
        row = _insert(store, id="opa", url="https://x.com/opa",
                      ai_score=2.5, score_source=None,
                      source="opus_annotation_cycle_4")
        n = store.update_scores_from_labels(
            [{"url": "https://x.com/opa", "in_briefing": True}]
        )
        assert n == 0
        ai_score, src = row()
        assert ai_score == pytest.approx(2.5)
        assert src is None

    def test_live_row_alongside_synthetic_still_boosts(self, store):
        """The clause must not be over-broad: a genuine live row in the same
        label batch as a synthetic one is still boosted normally."""
        live = _insert(store, id="lv", url="https://reuters.com/x",
                       ai_score=0.0, score_source=None, source="rss")
        synth = _insert(store, id="bt2", url="backtest://run_9/d/BUY/MU",
                        ai_score=5.0, score_source=None,
                        source="backtest_run_9_winner")
        n = store.update_scores_from_labels([
            {"url": "https://reuters.com/x", "in_briefing": True},
            {"url": "backtest://run_9/d/BUY/MU", "in_briefing": True},
        ])
        assert n == 1, "exactly the live row should be updated"
        lv_ai, lv_src = live()
        assert lv_ai == pytest.approx(4.5)
        assert lv_src == "briefing_boost"
        bt_ai, bt_src = synth()
        assert bt_ai == pytest.approx(5.0), "synthetic BUY-winner label rewritten"
        assert bt_src is None
