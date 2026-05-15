"""Urgency scorer behavior: Sonnet response → store state."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from watchers import urgency_scorer


def _insert(store, *, id, url="https://x.com/1", title="t", source="rss",
            kw_score=1.0, urgency=0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, urgency,
             "2026-05-15T00:00:00+00:00", 0),
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
