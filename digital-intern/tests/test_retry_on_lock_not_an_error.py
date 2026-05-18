"""storage/article_store.py::_retry_on_lock — the 'not an error'
shared-connection cursor-collision variant.

Live evidence (2026-05-18 daemon.log): ``[recursive_labeler] error: not an
error`` at 12:09:20Z landed exactly at the onset of a ``database is locked``
writer-contention storm (``insert_batch``/``update_ml_scores_batch``
exhausting at 12:09:24-32Z). It surfaced from the ``@_retry_on_lock``-decorated
``update_ai_scores_batch.executemany`` inside the recursive-labeler's round 1
(the ``round=1 candidates=500`` log line preceded it; ``round=1 labeled=…``
was never logged — so the fetch had succeeded and the collision hit the
*writer*, not the round-1 reader). ``pysqlite`` returns "not an error" as the
``SQLITE_OK`` (errno 0) default message when a concurrent writer on the SAME
shared ``self.conn`` resets the statement state mid-call — the SAME corruption
class as the ``no more rows available`` / ``another row available`` variants,
just a different surfaced string.

``_retry_on_lock`` retries only when the message contains a substring in
``_RETRYABLE_DB_ERRORS``. That tuple had ``database is locked`` /
``another row available`` / ``another row pending`` / ``no more rows
available`` but NOT ``not an error`` — so the decorator declined to retry the
*idempotent* ``UPDATE … SET ai_score=?, urgency=MAX(urgency,?),
score_source='llm' WHERE id=?``, it bubbled to the recursive_labeler worker's
broad ``except``, and the ENTIRE 4h Sonnet/Opus gold-label cycle was aborted
(every remaining batch's labels discarded — ZERO successful recursive_labeler
runs since the 07:29Z daemon start).

These pin the fix with specific behaviour, not "no crash":
  * the new variant IS retried and the idempotent call eventually succeeds;
  * a genuine ``IntegrityError`` ("UNIQUE constraint failed") still PROPAGATES
    (the substring discriminator must stay tight — never swallow real bugs);
  * an unrecoverable storm exhausts exactly ``_LOCK_RETRY_ATTEMPTS`` and
    re-raises (bumping ``_lock_failures``), never an infinite loop;
  * the established siblings remain in the tuple (a refactor that drops any of
    them reopens this exact batch-drop class — same anti-drift discipline as
    ``tests/test_retry_on_lock_no_more_rows.py`` and the backtest-isolation
    parity suites).
"""
from __future__ import annotations

import sqlite3

import pytest

from storage import article_store


@pytest.fixture(autouse=True)
def _instant_retry(monkeypatch):
    """Neutralise the exponential backoff sleep so the retry budget is
    exercised in microseconds, not ~4s. Patches the module's ``time`` ref
    (the decorator calls ``time.sleep``)."""
    monkeypatch.setattr(article_store.time, "sleep", lambda *_a, **_k: None)


def test_not_an_error_is_retried_then_succeeds():
    """The decorated, idempotent writer that collided once recovers on retry —
    the recursive_labeler round-1 ``update_ai_scores_batch`` failure mode."""
    calls = {"n": 0}

    @article_store._retry_on_lock
    def flaky_writer():
        calls["n"] += 1
        if calls["n"] == 1:
            # pysqlite raises OperationalError (a DatabaseError subclass) with
            # the SQLITE_OK default message under shared-conn corruption.
            raise sqlite3.OperationalError("not an error")
        return "write-applied"

    assert flaky_writer() == "write-applied"
    assert calls["n"] == 2, "must have retried exactly once after the collision"


def test_not_an_error_inside_a_larger_message_still_retries():
    """Substring match (consistent with the other variants) — a wrapped
    sqlite3 message that embeds the errno-0 string is still the collision."""
    calls = {"n": 0}

    @article_store._retry_on_lock
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.DatabaseError("sqlite3 error: not an error")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_integrity_error_still_propagates_unretried():
    calls = {"n": 0}

    @article_store._retry_on_lock
    def writer():
        calls["n"] += 1
        raise sqlite3.IntegrityError("UNIQUE constraint failed: articles.id")

    with pytest.raises(sqlite3.IntegrityError):
        writer()
    assert calls["n"] == 1, "a real IntegrityError must NOT be retried/swallowed"


def test_persistent_not_an_error_exhausts_budget_and_raises():
    attempts = article_store._LOCK_RETRY_ATTEMPTS
    calls = {"n": 0}

    @article_store._retry_on_lock
    def always_collides():
        calls["n"] += 1
        raise sqlite3.OperationalError("not an error")

    before = article_store.lock_metrics()["lock_failures"]
    with pytest.raises(sqlite3.OperationalError, match="not an error"):
        always_collides()
    # Tried the full budget, then re-raised — bounded, never an infinite loop.
    assert calls["n"] == attempts
    after = article_store.lock_metrics()["lock_failures"]
    assert after == before + 1, "an exhausted retry budget must bump lock_failures"


def test_retryable_set_contains_the_new_variant_and_keeps_siblings():
    s = article_store._RETRYABLE_DB_ERRORS
    assert "not an error" in s, (
        "regression guard: dropping this reopens the recursive_labeler "
        "gold-label-cycle abort class (live 2026-05-18 12:09:20Z)"
    )
    # The established collision strings must not be lost in any refactor.
    for sibling in ("database is locked", "another row available",
                    "another row pending", "no more rows available"):
        assert sibling in s
