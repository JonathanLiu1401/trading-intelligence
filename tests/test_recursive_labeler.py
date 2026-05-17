"""Recursive-labeler invariants nothing else in the suite pins.

Scope is deliberately narrow — only behaviour unique to ml/recursive_labeler:

  * The urgency-parse regression: a single malformed ``urgency`` value from
    Claude (it returns "1", "1.0", "yes", true — not a bare int) must NOT
    abort the run or discard the batch's already-collected good labels. This
    is the test that fails pre-fix and passes post-fix; the bug unwound
    _apply_labels → _run_round → run_recursive_labeling with no inner handler.
  * The 0..5 → 0..10 relevance rescale (a load-bearing magic ``* 2.0``).
  * Writes go through update_ai_scores_batch → score_source='llm' (Sonnet/Opus
    are ground-truth labelers; this is the ml-vs-ai invariant surface).
  * _fetch_round1_candidates backtest/opus exclusion — a separate WHERE filter
    than storage._LIVE_ONLY_CLAUSE, so it needs its own regression guard.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ml import recursive_labeler


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source="rss", ai_score=0.0,
                urgency=0, kw_score=1.0, first_seen=None):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, None, None),
        )
        store.conn.commit()


def _row(store, aid):
    return store.conn.execute(
        "SELECT ai_score, urgency, score_source FROM articles WHERE id=?",
        (aid,),
    ).fetchone()


class TestApplyLabels:
    def test_relevance_rescaled_and_tagged_llm(self, store):
        """relevance is 0..5 from the prompt; the store keeps 0..10. The
        labeler is an LLM path, so writes must tag score_source='llm'."""
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/a", "relevance": 4.0, "urgency": 1}]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        ai_score, urgency, src = _row(store, "a")
        assert ai_score == pytest.approx(8.0), "0..5 → 0..10 rescale (×2) broken"
        assert urgency == 1
        assert src == "llm", "recursive-labeler writes are ground-truth labels"

    def test_poison_urgency_does_not_abort_or_lose_siblings(self, store):
        """THE regression test. Middle label carries a non-int urgency Claude
        commonly emits. Pre-fix: int('yes') raised ValueError that escaped
        _apply_labels (no try/except round the loop), so the in-flight
        ``updates`` list — including the perfectly good siblings a and c —
        was discarded and the rest of the 4h cycle aborted. Post-fix: the bad
        urgency degrades to 0 and every good label is still persisted."""
        for aid in ("a", "b", "c"):
            _insert_raw(store, id=aid, url=f"https://x.com/{aid}", title=aid)
        articles = [
            {"_id": "a", "url": "https://x.com/a", "title": "a"},
            {"_id": "b", "url": "https://x.com/b", "title": "b"},
            {"_id": "c", "url": "https://x.com/c", "title": "c"},
        ]
        labels = [
            {"url": "https://x.com/a", "relevance": 4.0, "urgency": 1},
            {"url": "https://x.com/b", "relevance": 3.5, "urgency": "yes"},  # poison
            {"url": "https://x.com/c", "relevance": 2.0, "urgency": 0},
        ]

        n = recursive_labeler._apply_labels(store, articles, labels)

        # All three persisted — pre-fix this was 0 (whole batch discarded).
        assert n == 3
        a = _row(store, "a")
        b = _row(store, "b")
        c = _row(store, "c")
        assert a == (pytest.approx(8.0), 1, "llm")
        # Poison sibling keeps its relevance label; urgency degrades to 0.
        assert b == (pytest.approx(7.0), 0, "llm")
        assert c == (pytest.approx(4.0), 0, "llm")

    @pytest.mark.parametrize(
        "raw_urg, expect",
        [(1, 1), ("1", 1), ("1.0", 1), (1.0, 1), (True, 1),
         (0, 0), ("0", 0), (None, 0), ("yes", 0), ("high", 0), ([], 0)],
    )
    def test_urgency_coercion_matrix(self, store, raw_urg, expect):
        """Every value below must coerce, never raise. The string/bool forms
        are exactly what Claude returns in practice."""
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/a", "relevance": 5.0, "urgency": raw_urg}]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        ai_score, urgency, _ = _row(store, "a")
        assert ai_score == pytest.approx(10.0)
        assert urgency == expect

    def test_bad_relevance_skips_only_that_label(self, store):
        """A non-numeric relevance still skips its own label (the existing
        guard) without taking down siblings."""
        for aid in ("a", "b"):
            _insert_raw(store, id=aid, url=f"https://x.com/{aid}", title=aid)
        articles = [
            {"_id": "a", "url": "https://x.com/a", "title": "a"},
            {"_id": "b", "url": "https://x.com/b", "title": "b"},
        ]
        labels = [
            {"url": "https://x.com/a", "relevance": "garbage", "urgency": 1},
            {"url": "https://x.com/b", "relevance": 3.0, "urgency": 0},
        ]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        assert _row(store, "a") == (pytest.approx(0.0), 0, None)  # untouched
        assert _row(store, "b") == (pytest.approx(6.0), 0, "llm")

    def test_unknown_url_label_ignored(self, store):
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/NOT-IN-BATCH", "relevance": 5.0,
                   "urgency": 1}]
        assert recursive_labeler._apply_labels(store, articles, labels) == 0
        assert _row(store, "a") == (pytest.approx(0.0), 0, None)


class TestFetchRound1Candidates:
    def test_excludes_backtest_and_opus_rows(self, store):
        """Round-1 candidate selection has its OWN backtest/opus WHERE filter
        (not storage._LIVE_ONLY_CLAUSE) — re-scoring a synthetic training row
        with the live Sonnet labeler would corrupt the backtest signal."""
        _insert_raw(store, id="live", url="https://reuters.com/x",
                    title="Live unlabeled story", source="rss", ai_score=0.0)
        _insert_raw(store, id="bt_url", url="backtest://run_1/d/BUY/MU",
                    title="Synthetic by url", source="rss", ai_score=0.5)
        _insert_raw(store, id="bt_src", url="https://x.com/y",
                    title="Synthetic by source", source="backtest_run_1_winner",
                    ai_score=0.5)
        _insert_raw(store, id="opus", url="https://x.com/z",
                    title="Opus annotation row", source="opus_annotation_cycle_2",
                    ai_score=1.0)

        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        assert ids == {"live"}

    def test_excludes_already_strongly_labeled(self, store):
        """ai_score >= 2.0 is 'labeled enough' — only weak/zero rows recurse."""
        _insert_raw(store, id="weak", url="https://x.com/weak",
                    title="weakly labeled", ai_score=1.5)
        _insert_raw(store, id="strong", url="https://x.com/strong",
                    title="strongly labeled", ai_score=7.0)
        _insert_raw(store, id="zero", url="https://x.com/zero",
                    title="unlabeled", ai_score=0.0)

        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        assert ids == {"weak", "zero"}
        assert "strong" not in ids
