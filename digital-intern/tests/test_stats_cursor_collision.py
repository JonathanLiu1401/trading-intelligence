"""storage/article_store.py::_expect_row — the ``fetchone() -> None``
shared-connection cursor-collision variant.

Live evidence (2026-05-18 daemon.log): ``[stats_worker] error: 'NoneType'
object is not subscriptable`` recurred 12+×/h, exactly correlated with the
concurrent ``database is locked`` writer-contention storm. ``_retry_on_lock``
already retries the ``another row available`` / ``no more rows available``
``DatabaseError`` flavour of the documented shared-``self.conn`` cursor
collision — but the SAME collision can instead corrupt the fetch so
``cur.fetchone()`` returns ``None``. The aggregate readers then did
``.fetchone()[0]`` → ``TypeError``, which is NOT a ``sqlite3.DatabaseError``,
so the decorator declined it and it bubbled to ``stats_worker``'s broad
``except`` every contended cycle (``stats``/``count_unscored``/``stats_since``
all silently failing under load — the scorer-backlog gauge and /api/stats
went blind).

These pin the fix with specific behaviour, not "no crash":
  * a ``None`` aggregate fetch is converted to the SAME retryable signal the
    decorator already handles, and the idempotent reader then succeeds;
  * the raised message stays within ``_RETRYABLE_DB_ERRORS`` (anti-drift: a
    refactor of that tuple that drops the reused substring reopens this exact
    silent-failure class);
  * a real empty result is impossible at these call sites (``MAX``/``COUNT``
    always yield one row) so the guard can never mask a legitimate 0/None;
  * ``stats`` / ``count_unscored`` / ``stats_since`` no longer raise
    ``TypeError`` on the collision — they retry past the writer and return.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage import article_store


@pytest.fixture(autouse=True)
def _instant_retry(monkeypatch):
    """Neutralise the exponential backoff sleep so the retry budget is
    exercised in microseconds, not ~4s (mirrors test_retry_on_lock_no_more_rows)."""
    monkeypatch.setattr(article_store.time, "sleep", lambda *_a, **_k: None)


# ── unit: _expect_row ────────────────────────────────────────────────────────
class _Cur:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


def test_expect_row_returns_row_when_present():
    assert article_store._expect_row(_Cur((42,))) == (42,)
    # A legitimate aggregate result of (None,) — e.g. MAX(rowid) on an empty
    # table — is a real row and must pass straight through untouched.
    assert article_store._expect_row(_Cur((None,))) == (None,)


def test_expect_row_raises_retryable_on_none():
    with pytest.raises(sqlite3.DatabaseError) as ei:
        article_store._expect_row(_Cur(None))
    msg = str(ei.value).lower()
    assert any(s in msg for s in article_store._RETRYABLE_DB_ERRORS), (
        "the synthesised collision error must stay within _RETRYABLE_DB_ERRORS "
        "or _retry_on_lock will not retry it (anti-drift pin)"
    )


def test_expect_row_none_is_retried_by_decorator_then_succeeds():
    """The decorator + helper compose: a one-shot None collision is retried
    and the idempotent reader returns on the next attempt."""
    calls = {"n": 0}

    @article_store._retry_on_lock
    def flaky_reader():
        calls["n"] += 1
        row = None if calls["n"] == 1 else (7,)
        return article_store._expect_row(_Cur(row))[0]

    assert flaky_reader() == 7
    assert calls["n"] == 2, "must retry exactly once after the None collision"


def test_expect_row_raises_retryable_on_empty_tuple():
    """Empty-tuple cursor-collision class — same root cause as the None
    variant, surfacing as ``fetchone() -> ()`` instead. Live evidence
    (2026-05-19/20 daemon.log): ``[stats_worker] error: tuple index out of
    range`` recurred under writer contention. ``_expect_row`` must convert
    this into the SAME retryable signal so the @_retry_on_lock decorator
    succeeds on the next attempt instead of letting ``IndexError`` bubble
    to ``stats_worker``'s broad except."""
    with pytest.raises(sqlite3.DatabaseError) as ei:
        article_store._expect_row(_Cur(()))
    msg = str(ei.value).lower()
    assert any(s in msg for s in article_store._RETRYABLE_DB_ERRORS), (
        "the empty-tuple collision must stay within _RETRYABLE_DB_ERRORS "
        "or _retry_on_lock will not retry it (anti-drift pin)"
    )


def test_expect_row_empty_tuple_is_retried_by_decorator_then_succeeds():
    """Parallel to ``test_expect_row_none_is_retried_by_decorator_then_succeeds``
    for the empty-tuple variant of the same cursor collision."""
    calls = {"n": 0}

    @article_store._retry_on_lock
    def flaky_reader():
        calls["n"] += 1
        row = () if calls["n"] == 1 else (11,)
        return article_store._expect_row(_Cur(row))[0]

    assert flaky_reader() == 11
    assert calls["n"] == 2, "must retry exactly once after the () collision"


# ── integration: stats() / count_unscored() / stats_since() recover ──────────
class _FlakyCursor:
    """Delegates to a real cursor but returns ``None`` from the FIRST
    ``fetchone()`` across the whole store (one simulated collision), then
    behaves normally — so the @_retry_on_lock reader fails attempt 1 and
    succeeds attempt 2."""

    def __init__(self, real, state):
        self._real = real
        self._state = state

    def fetchone(self):
        if not self._state["fired"]:
            self._state["fired"] = True
            return None
        return self._real.fetchone()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FlakyConn:
    def __init__(self, real, state):
        self._real = real
        self._state = state

    def execute(self, *a, **kw):
        return _FlakyCursor(self._real.execute(*a, **kw), self._state)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _seed(store):
    """Insert one fixture row whose ``first_seen`` is INSIDE the 24h window
    every reader filters against. A hardcoded absolute date silently broke
    ``test_stats_since_recovers_from_collision`` the moment wall-clock passed
    it by 24h — a date-bound test failure on a real invariant that was
    actually intact (identical pattern to ``conftest._recent_iso``)."""
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles (id,url,title,source,published,kw_score,"
            "ai_score,urgency,first_seen,cycle) VALUES "
            "('a','http://x/a','Held MU earnings beat','rss','',2.0,0,0,"
            "?,0)",
            (fresh,),
        )
        store.conn.commit()


def test_stats_recovers_from_fetchone_none_collision(store):
    _seed(store)
    before = article_store.lock_metrics()["lock_retries"]
    state = {"fired": False}
    store.conn = _FlakyConn(store.conn, state)

    s = store.stats()  # must NOT raise TypeError — collision is retried

    assert isinstance(s, dict) and isinstance(s["total"], int)
    assert s["total"] >= 1
    assert state["fired"], "the simulated collision must actually have fired"
    assert article_store.lock_metrics()["lock_retries"] > before, (
        "the None collision must have gone through the retry path"
    )


def test_count_unscored_recovers_from_collision(store):
    _seed(store)
    state = {"fired": False}
    store.conn = _FlakyConn(store.conn, state)
    n = store.count_unscored(min_kw=0.0)
    assert n == 1 and state["fired"]


def test_stats_since_recovers_from_collision(store):
    _seed(store)
    state = {"fired": False}
    store.conn = _FlakyConn(store.conn, state)
    out = store.stats_since(hours=24)
    assert out["total"] == 1 and out["urgent"] == 0 and state["fired"]
