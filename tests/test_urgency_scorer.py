"""Urgency scorer behavior: Sonnet response → store state."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import urgency_scorer


def _insert(store, *, id, url="https://x.com/1", title="t", source="rss",
            kw_score=1.0, urgency=0, first_seen=None):
    # Keep first_seen inside the 24h freshness window so TestPreservesAlerted
    # (which calls get_unalerted_urgent) doesn't go stale on a later rerun.
    if first_seen is None:
        first_seen = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, urgency,
             first_seen, 0),
        )
        store.conn.commit()


def _patched_claude(response: list):
    """Return a fake claude_call that responds with the given JSON array."""
    body = json.dumps(response)
    return patch.object(urgency_scorer, "claude_call", return_value=body)


class TestScoreClassification:
    def test_high_score_marked_urgent(self, store):
        _insert(store, id="a", title="MU earnings beat")
        articles = [{"_id": "a", "title": "MU earnings beat", "summary": ""}]
        with _patched_claude([{"index": 0, "score": 9.5, "reason": "earnings"}]):
            n = urgency_scorer.score_batch(articles, store)
        assert n == 1
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(9.5)
        assert row[1] == 1
        assert row[2] == "llm"

    def test_low_score_not_urgent(self, store):
        _insert(store, id="a", title="generic equity story")
        articles = [{"_id": "a", "title": "x", "summary": ""}]
        with _patched_claude([{"index": 0, "score": 3.0, "reason": "noise"}]):
            n = urgency_scorer.score_batch(articles, store)
        assert n == 0
        row = store.conn.execute(
            "SELECT ai_score, urgency FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(3.0)
        assert row[1] == 0

    def test_threshold_boundary(self, store):
        """URGENT_THRESHOLD is 8.0 — exactly 8.0 must classify as urgent."""
        _insert(store, id="a", title="x")
        articles = [{"_id": "a", "title": "x", "summary": ""}]
        with _patched_claude([{"index": 0, "score": 8.0, "reason": "edge"}]):
            n = urgency_scorer.score_batch(articles, store)
        assert n == 1


class TestPreservesAlerted:
    def test_rescore_does_not_unalert(self, store):
        """An alerted article (urgency=2) that gets a fresh score must not be
        regressed back to urgent (1). The store uses MAX(urgency, new) — this
        catches a regression where the alerter would re-fire on the same item."""
        _insert(store, id="a", title="MU shock", urgency=2)
        articles = [{"_id": "a", "title": "MU shock", "summary": ""}]
        with _patched_claude([{"index": 0, "score": 9.5, "reason": "still urgent"}]):
            urgency_scorer.score_batch(articles, store)
        row = store.conn.execute(
            "SELECT urgency FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == 2, (
            "alerted article was regressed; alerter would re-fire next cycle"
        )
        assert store.get_unalerted_urgent() == []


class TestMalformedResponses:
    def test_unscored_items_get_noise_floor(self, store):
        """When Sonnet returns a partial response (e.g. skips an article), the
        unscored items must be floored to 0.01 so they exit the unscored
        backlog — otherwise the LLM keeps re-receiving them forever."""
        _insert(store, id="a")
        _insert(store, id="b")
        articles = [
            {"_id": "a", "title": "x", "summary": ""},
            {"_id": "b", "title": "y", "summary": ""},
        ]
        # Sonnet responds to only one index
        with _patched_claude([{"index": 0, "score": 7.0, "reason": "ok"}]):
            urgency_scorer.score_batch(articles, store)
        rows = dict(store.conn.execute(
            "SELECT id, ai_score FROM articles"
        ).fetchall())
        assert rows["a"] == pytest.approx(7.0)
        assert rows["b"] == pytest.approx(0.01), (
            "anti-loop floor missing — article will retry against Sonnet forever"
        )

    def test_truncated_tail_requeued_not_floored(self, store):
        """When Sonnet's output is truncated at the token limit, json_extract
        salvages only a leading prefix of indices. Flooring the missing tail to
        0.01 would silently bury genuine articles as noise forever. The scorer
        must detect the clean-prefix + long-tail truncation fingerprint and
        leave the tail at ai_score=0 so it re-queues next pass."""
        n = 30
        for i in range(n):
            _insert(store, id=str(i), url=f"https://x.com/{i}")
        articles = [{"_id": str(i), "title": "x", "summary": ""} for i in range(n)]
        # Sonnet only got through the first 10 before truncation.
        partial = [{"index": i, "score": 3.0, "reason": "ok"} for i in range(10)]
        with _patched_claude(partial):
            urgency_scorer.score_batch(articles, store)
        rows = dict(store.conn.execute(
            "SELECT id, ai_score FROM articles"
        ).fetchall())
        for i in range(10):
            assert rows[str(i)] == pytest.approx(3.0)
        for i in range(10, n):
            assert rows[str(i)] == 0, (
                f"article {i} was floored to noise on a truncated response; "
                "it must be re-queued (ai_score=0) instead"
            )

    def test_scattered_omission_still_floored(self, store):
        """A non-prefix gap (Sonnet skipped a middle index but covered the
        rest) is deliberate omission, not truncation — the anti-loop floor
        must still apply so the skipped item exits the backlog."""
        for i in range(4):
            _insert(store, id=str(i), url=f"https://x.com/{i}")
        articles = [{"_id": str(i), "title": "x", "summary": ""} for i in range(4)]
        # Indices 0,1,3 returned; 2 skipped (internal gap, not a clean prefix).
        with _patched_claude([
            {"index": 0, "score": 3.0, "reason": "ok"},
            {"index": 1, "score": 3.0, "reason": "ok"},
            {"index": 3, "score": 3.0, "reason": "ok"},
        ]):
            urgency_scorer.score_batch(articles, store)
        row = store.conn.execute(
            "SELECT ai_score FROM articles WHERE id='2'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01), (
            "scattered (non-prefix) omission must still be floored by anti-loop"
        )

    def test_empty_array_skips_floor(self, store):
        """A completely empty Sonnet response is ambiguous — refusal vs true
        zero. The scorer must NOT mass-label 100 articles as noise on an empty
        response; it should retry instead."""
        _insert(store, id="a")
        articles = [{"_id": "a", "title": "x", "summary": ""}]
        with _patched_claude([]):
            urgency_scorer.score_batch(articles, store)
        row = store.conn.execute(
            "SELECT ai_score FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == 0, "empty response must NOT poison the queue"
