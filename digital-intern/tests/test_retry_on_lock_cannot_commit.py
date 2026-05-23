"""``_retry_on_lock`` must absorb the ``cannot commit transaction - SQL
statements in progress`` flavour of the shared-connection cursor-collision.

Same class as the ``another row available`` / ``no more rows available`` /
``not an error`` variants the decorator already retries — the only difference
is *where* in the call it surfaces (at ``commit()`` after a successful
``executemany``, instead of mid-fetch on a colliding ``execute``). Live
evidence (2026-05-23 daemon.log): the error fired 4 times in one day from
scorer_worker / hackernews_worker / alert_worker's ``mark_alerted_batch`` and
each occurrence bubbled to the worker's broad ``except`` — losing the whole
cycle's writes (urgent rows then re-fetched every 20s until aged out). If
this regex falls off the retryable list again, this test pins the regression.
"""
from __future__ import annotations

import sqlite3

import pytest


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                kw_score=1.0):
    from datetime import datetime, timezone
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             datetime.now(timezone.utc).isoformat(), 0),
        )
        store.conn.commit()


class _CannotCommitConn:
    """Delegates to a real sqlite connection but raises the live cursor-
    collision ``OperationalError`` on the first ``commit()`` ONLY — the
    ``executemany`` itself succeeds. Models the exact 2026-05-23 daemon.log
    interleave: a sibling lockless reader cursor on the shared ``self.conn``
    holds in-flight prepared statements, SQLite refuses to tear them down at
    the transaction boundary, the writer's commit raises."""

    def __init__(self, real):
        self._real = real
        self.commit_calls = 0
        self.em_calls = 0

    def executemany(self, sql, seq):
        # Apply the write so the retry can observe the desired state.
        self.em_calls += 1
        return self._real.executemany(sql, seq)

    def execute(self, sql, params=()):
        return self._real.execute(sql, params)

    def commit(self):
        self.commit_calls += 1
        if self.commit_calls == 1:
            raise sqlite3.OperationalError(
                "cannot commit transaction - SQL statements in progress"
            )
        return self._real.commit()

    def __getattr__(self, name):  # delegate cursor/rollback/…
        return getattr(self._real, name)


class TestCannotCommitTransactionRetry:
    def test_mark_alerted_batch_retries_on_cannot_commit(
        self, store, monkeypatch
    ):
        """The live failure mode: ``mark_alerted_batch.commit()`` raises
        ``OperationalError("cannot commit transaction - SQL statements in
        progress")``. Pre-fix the decorator re-raised and the alert worker's
        broad ``except`` logged ``[alert] failed to mark stale rows alerted``,
        so the urgent rows stayed at urgency=1 and got re-fetched every 20s
        cycle. With the cursor-collision string added to ``_RETRYABLE_DB_
        ERRORS`` the decorator transparently retries — first commit raises,
        second commit succeeds, the rows reach urgency=2 as intended."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        for i in range(3):
            _insert_raw(store, id=f"u{i}", url=f"https://x.com/{i}",
                        title=f"urgent {i}", source="rss", urgency=1,
                        ai_score=9.0)

        flaky = _CannotCommitConn(store.conn)
        store.conn = flaky
        before = A.lock_metrics()["lock_retries"]

        # Must NOT raise.
        n = store.mark_alerted_batch(["u0", "u1", "u2"])
        after = A.lock_metrics()["lock_retries"]

        assert flaky.commit_calls == 2, (
            "commit should be retried exactly once after the cursor-collision"
        )
        assert after == before + 1
        assert n == 3
        # The write actually landed (urgency now 2).
        rows = store.conn.execute(
            "SELECT id, urgency FROM articles WHERE id IN ('u0','u1','u2')"
        ).fetchall()
        assert {r[0]: r[1] for r in rows} == {"u0": 2, "u1": 2, "u2": 2}

    def test_update_ai_scores_batch_retries_on_cannot_commit(
        self, store, monkeypatch
    ):
        """Same shape against a different decorated writer — proves the
        retry-list change is universal, not specific to mark_alerted_batch."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="a", url="https://reuters.com/x",
                    title="title", source="rss", ai_score=0.0, kw_score=2.0)

        flaky = _CannotCommitConn(store.conn)
        store.conn = flaky
        before = A.lock_metrics()["lock_retries"]

        store.update_ai_scores_batch([("a", 7.5, 1)])
        after = A.lock_metrics()["lock_retries"]

        assert flaky.commit_calls == 2
        assert after == before + 1
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(7.5)
        assert row[1] == 1
        assert row[2] == "llm"

    def test_systemerror_error_return_without_exception_is_retried(
        self, store, monkeypatch
    ):
        """The CPython sqlite3 C bindings sometimes raise the cursor-collision
        failure as ``SystemError: error return without exception set`` instead
        of a proper ``DatabaseError`` (sibling lockless reader still holds
        prepared statements on the shared ``self.conn`` when ``commit()`` is
        attempted; the C code returns an error code but fails to set the
        Python-level exception). Live evidence (2026-05-23 daemon.log): 3
        such SystemErrors during the same writer-contention storm that
        produced the ``cannot commit transaction`` variant. Pre-fix the
        decorator only caught ``sqlite3.DatabaseError`` and the SystemError
        bubbled to the worker's broad ``except`` (``[rss_worker] error: error
        return without exception set; backing off 5s``)."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="a", url="https://x.com/sys",
                    title="t", source="rss", ai_score=0.0, kw_score=2.0)

        class _FlakySystemErrorConn:
            def __init__(self, real):
                self._real = real
                self.commit_calls = 0
            def executemany(self, sql, seq):
                return self._real.executemany(sql, seq)
            def execute(self, sql, params=()):
                return self._real.execute(sql, params)
            def commit(self):
                self.commit_calls += 1
                if self.commit_calls == 1:
                    raise SystemError("error return without exception set")
                return self._real.commit()
            def __getattr__(self, n):
                return getattr(self._real, n)

        flaky = _FlakySystemErrorConn(store.conn)
        store.conn = flaky
        before = A.lock_metrics()["lock_retries"]

        store.update_ai_scores_batch([("a", 6.5, 0)])
        after = A.lock_metrics()["lock_retries"]

        assert flaky.commit_calls == 2
        assert after == before + 1
        row = store.conn.execute(
            "SELECT ai_score, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(6.5)
        assert row[1] == "llm"

    def test_unrelated_systemerror_propagates(self, store, monkeypatch):
        """Tight discrimination: a SystemError with any OTHER message is a
        genuine Python internal bug and MUST propagate. Broadly catching
        SystemError would hide serious failures behind 5 silent retries."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="b", url="https://x.com/oops",
                    title="t", source="rss", ai_score=0.0, kw_score=2.0)

        class _BadSysConn:
            def __init__(self, real):
                self._real = real
            def executemany(self, sql, seq):
                return self._real.executemany(sql, seq)
            def execute(self, sql, params=()):
                return self._real.execute(sql, params)
            def commit(self):
                raise SystemError("something completely different")
            def __getattr__(self, n):
                return getattr(self._real, n)

        store.conn = _BadSysConn(store.conn)
        with pytest.raises(SystemError,
                           match="something completely different"):
            store.update_ai_scores_batch([("b", 5.0, 0)])

    def test_unrelated_commit_failure_still_propagates(self, store, monkeypatch):
        """Tight discrimination: a non-allowlisted OperationalError at commit
        (e.g. disk-full) must NOT be silently retried/swallowed — that would
        mask a real, non-transient failure as a slow recovery."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="a", url="https://x.com/y",
                    title="t", source="rss", ai_score=0.0, kw_score=2.0)

        class _DiskFullCommitConn:
            def __init__(self, real):
                self._real = real
            def executemany(self, sql, seq):
                return self._real.executemany(sql, seq)
            def execute(self, sql, params=()):
                return self._real.execute(sql, params)
            def commit(self):
                raise sqlite3.OperationalError("disk I/O error")
            def __getattr__(self, n):
                return getattr(self._real, n)

        store.conn = _DiskFullCommitConn(store.conn)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            store.update_ai_scores_batch([("a", 5.0, 0)])
