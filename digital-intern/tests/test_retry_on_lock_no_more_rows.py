"""storage/article_store.py::_retry_on_lock — the 'no more rows available'
shared-connection cursor-collision variant.

Live evidence (2026-05-18 daemon.log): ``[scorer_worker] error: no more rows
available`` recurred ~hourly (06:05, 08:43) plus ``[recursive_labeler]`` 08:01.
``_retry_on_lock`` catches ``sqlite3.DatabaseError`` and retries only when the
message contains a substring in ``_RETRYABLE_DB_ERRORS``. That tuple had
``database is locked`` / ``another row available`` / ``another row pending``
but NOT ``no more rows available`` — the SAME shared-``self.conn`` cursor-state
corruption a concurrent writer ``executemany`` causes mid-fetch, just a
different surfaced string. So a colliding ``get_unscored`` raised it, the
decorator declined to retry, it bubbled to the worker's broad ``except`` and
that cycle's scored batch was silently dropped → urgent items went un-scored →
delayed BREAKING alerts (the documented (2) failure mode, on the scoring path).

These pin the fix with specific behaviour, not "no crash":
  * the new variant IS retried and the idempotent call eventually succeeds;
  * a genuine ``IntegrityError`` ("UNIQUE constraint failed") still PROPAGATES
    (the substring discriminator must stay tight — never swallow real bugs);
  * an unrecoverable storm exhausts exactly ``_LOCK_RETRY_ATTEMPTS`` and
    re-raises (bumping ``_lock_failures``), never an infinite loop;
  * the established siblings remain in the tuple (a refactor that drops any of
    them reopens this exact batch-drop class — same anti-drift discipline as
    the backtest-isolation parity suites).
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


def test_no_more_rows_available_is_retried_then_succeeds():
    calls = {"n": 0}

    @article_store._retry_on_lock
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.DatabaseError("no more rows available")
        return "rows-here"

    assert flaky() == "rows-here"
    assert calls["n"] == 2, "must have retried exactly once after the collision"


def test_integrity_error_still_propagates_unretried():
    calls = {"n": 0}

    @article_store._retry_on_lock
    def writer():
        calls["n"] += 1
        raise sqlite3.IntegrityError("UNIQUE constraint failed: articles.id")

    with pytest.raises(sqlite3.IntegrityError):
        writer()
    assert calls["n"] == 1, "a real IntegrityError must NOT be retried/swallowed"


def test_persistent_no_more_rows_exhausts_budget_and_raises():
    attempts = article_store._LOCK_RETRY_ATTEMPTS
    calls = {"n": 0}

    @article_store._retry_on_lock
    def always_collides():
        calls["n"] += 1
        raise sqlite3.DatabaseError("no more rows available")

    before = article_store.lock_metrics()["lock_failures"]
    with pytest.raises(sqlite3.DatabaseError, match="no more rows available"):
        always_collides()
    # Tried the full budget, then re-raised — bounded, never an infinite loop.
    assert calls["n"] == attempts
    after = article_store.lock_metrics()["lock_failures"]
    assert after == before + 1, "an exhausted retry budget must bump lock_failures"


def test_retryable_set_contains_the_new_variant_and_keeps_siblings():
    s = article_store._RETRYABLE_DB_ERRORS
    assert "no more rows available" in s, (
        "regression guard: dropping this reopens the scorer/recursive_labeler "
        "batch-drop → delayed-alert class (live 2026-05-18)"
    )
    # The established collision strings must not be lost in any refactor.
    for sibling in ("database is locked", "another row available",
                    "another row pending"):
        assert sibling in s
