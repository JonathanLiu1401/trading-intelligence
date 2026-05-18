"""Storage layer invariants — the critical ones gate the live alert pipeline."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    """A first_seen value inside the 24h freshness window enforced by
    get_unalerted_urgent / get_top_for_briefing. Hardcoding an absolute date
    (the prior fixture) silently broke every backtest-isolation test the
    moment wall-clock passed it by 24h — a Saturday-morning rerun would fail
    on a real production invariant that was actually intact."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                ml_score=None, score_source=None, kw_score=1.0,
                first_seen=None):
    """Insert a row bypassing the public API so tests can build any state."""
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


# ── invariant #1: backtest rows must never surface to the alerter ────────────
class TestBacktestIsolation:
    def test_get_unalerted_urgent_excludes_backtest_urls(self, store):
        """Most critical invariant in the entire system: a backtest:// row that
        somehow lands with urgency=1 must NOT reach get_unalerted_urgent — that
        would post training data to Discord as breaking news."""
        _insert_raw(store, id="live1", url="https://reuters.com/x",
                    title="Real news", source="rss", urgency=1, ai_score=9.0)
        _insert_raw(store, id="bt1", url="backtest://run_1/2026-01-01/BUY/MU",
                    title="Synthetic", source="backtest_run_1", urgency=1,
                    ai_score=9.0)
        _insert_raw(store, id="bt2", url="https://example.com/x",
                    title="Opus annotation", source="opus_annotation_cycle_3",
                    urgency=1, ai_score=9.0)
        _insert_raw(store, id="bt3", url="https://example.com/y",
                    title="Backtest source", source="backtest_run_42_winner",
                    urgency=1, ai_score=9.0)

        urgent = store.get_unalerted_urgent()
        ids = {a["_id"] for a in urgent}

        assert ids == {"live1"}, (
            f"backtest rows leaked into urgent queue: {ids - {'live1'}}"
        )
        for a in urgent:
            assert not a["link"].startswith("backtest://")
            assert not a["source"].startswith("backtest_")
            assert not a["source"].startswith("opus_annotation")

    def test_get_top_for_briefing_excludes_backtest(self, store):
        _insert_raw(store, id="live1", url="https://reuters.com/x",
                    title="Real news headline goes here", source="rss",
                    ai_score=8.0)
        _insert_raw(store, id="bt1", url="backtest://run_1/2026-01-01/BUY/MU",
                    title="Synthetic backtest entry here", source="backtest_run_1",
                    ai_score=9.5)
        top = store.get_top_for_briefing(hours=24, limit=10)
        urls = [a["link"] for a in top]
        assert "https://reuters.com/x" in urls
        assert not any(u.startswith("backtest://") for u in urls)

    def test_get_unscored_excludes_backtest(self, store):
        _insert_raw(store, id="live1", url="https://reuters.com/x",
                    title="Live", source="rss", ai_score=0, kw_score=1.0)
        _insert_raw(store, id="bt1", url="backtest://x/y/z",
                    title="BT", source="backtest_run_1", ai_score=0,
                    kw_score=1.0)
        unscored = store.get_unscored(min_kw=0.0)
        ids = {a["_id"] for a in unscored}
        assert ids == {"live1"}


# ── invariant #2: mark_alerted prevents re-firing ─────────────────────────────
class TestAlertedMarking:
    def test_mark_alerted_removes_from_unalerted(self, store):
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="urgent thing", source="rss", urgency=1, ai_score=9.0)
        assert {a["_id"] for a in store.get_unalerted_urgent()} == {"a"}
        store.mark_alerted("a")
        assert store.get_unalerted_urgent() == []

    def test_mark_alerted_batch_removes_all(self, store):
        for i in range(3):
            _insert_raw(store, id=f"id{i}", url=f"https://x.com/{i}",
                        title=f"urgent {i}", source="rss", urgency=1,
                        ai_score=9.0)
        n = store.mark_alerted_batch(["id0", "id1", "id2"])
        assert n == 3
        assert store.get_unalerted_urgent() == []

    def test_subsequent_llm_rescore_does_not_un_alert(self, store):
        """A Sonnet rescore that hits an already-alerted article must NOT drop
        urgency back to 1 — the alerter would re-fire on the next cycle. This
        is enforced via SQL ``MAX(urgency, ?)``."""
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="x", source="rss", urgency=1, ai_score=9.0)
        store.mark_alerted("a")  # urgency now 2
        store.update_ai_scores_batch([("a", 9.5, 1)])  # rescore says urgent
        row = store.conn.execute(
            "SELECT urgency FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == 2, "alerted state was clobbered by rescore"
        assert store.get_unalerted_urgent() == []


# ── invariant #3: score_source separation (ml vs llm) ─────────────────────────
class TestScoreSourceSeparation:
    def test_update_ml_scores_batch_sets_ml(self, store):
        _insert_raw(store, id="a", url="https://x.com/1", title="x",
                    source="rss", ai_score=0, kw_score=2.0)
        store.update_ml_scores_batch([("a", 7.5, 0)])
        row = store.conn.execute(
            "SELECT ai_score, ml_score, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == 0, "model output must NOT pollute ai_score"
        assert row[1] == pytest.approx(7.5)
        assert row[2] == "ml"

    def test_update_ai_scores_batch_sets_llm(self, store):
        _insert_raw(store, id="a", url="https://x.com/1", title="x",
                    source="rss", ai_score=0, kw_score=2.0)
        store.update_ai_scores_batch([("a", 6.0, 0)])
        row = store.conn.execute(
            "SELECT ai_score, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(6.0)
        assert row[1] == "llm"

    def test_ml_does_not_overwrite_llm(self, store):
        """COALESCE guard: once an LLM has labeled a row, a subsequent model
        prediction must not relabel it as 'ml' — the trainer would see fewer
        ground-truth rows and the recursive labeling investment would be wasted."""
        _insert_raw(store, id="a", url="https://x.com/1", title="x",
                    source="rss", ai_score=0, kw_score=2.0)
        store.update_ai_scores_batch([("a", 6.0, 0)])
        store.update_ml_scores_batch([("a", 7.5, 0)])
        row = store.conn.execute(
            "SELECT ai_score, ml_score, score_source FROM articles WHERE id='a'"
        ).fetchone()
        assert row[0] == pytest.approx(6.0)
        assert row[1] == pytest.approx(7.5)
        assert row[2] == "llm", "ml score must not overwrite llm tag"


# ── invariant #4: insert + de-dup ─────────────────────────────────────────────
class TestInsertCrud:
    def test_insert_batch_returns_count(self, store):
        n = store.insert_batch([
            {"title": "A", "link": "https://x.com/1", "source": "rss",
             "published": "", "summary": "", "_relevance_score": 1.0},
            {"title": "B", "link": "https://x.com/2", "source": "rss",
             "published": "", "summary": "", "_relevance_score": 1.0},
        ])
        assert n == 2

    def test_insert_batch_dedupes(self, store):
        art = {"title": "A", "link": "https://x.com/1", "source": "rss",
               "published": "", "summary": "", "_relevance_score": 1.0}
        store.insert_batch([art])
        n = store.insert_batch([art])
        assert n == 0

    def test_stats_reports_counts(self, store):
        store.insert_batch([
            {"title": "A", "link": "https://x.com/1", "source": "rss",
             "published": "", "summary": "", "_relevance_score": 3.0},
        ])
        s = store.stats()
        assert s["total"] == 1
        assert "urgent" in s and "unscored" in s


# ── shared-connection cursor-collision retry ─────────────────────────────────
class _FlakyConn:
    """Delegates to a real sqlite connection but raises the shared-connection
    cursor-collision (``sqlite3.DatabaseError: another row available``) on the
    first ``executemany``, then behaves normally — the exact transient
    interleave observed 48x in one production ``daemon.log`` window when a
    lockless reader cursor on the shared ``self.conn`` raced a writer's
    ``executemany`` (``check_same_thread=False``, ~30 daemon threads)."""

    def __init__(self, real):
        self._real = real
        self.em_calls = 0

    def executemany(self, sql, seq):
        self.em_calls += 1
        if self.em_calls == 1:
            raise sqlite3.DatabaseError("another row available")
        return self._real.executemany(sql, seq)

    def __getattr__(self, name):  # delegate execute/commit/cursor/…
        return getattr(self._real, name)


class TestCursorCollisionRetry:
    """``_retry_on_lock`` must also absorb the transient cursor-collision
    DatabaseError (idempotent UPDATE/INSERT-OR-IGNORE ops), not only
    'database is locked' — otherwise a whole Sonnet-labeled batch is dropped
    and the articles re-enter the LLM queue forever (wasted quota + genuinely
    urgent items never get urgency=1 → missed alerts)."""

    def test_retry_on_another_row_available_then_succeeds(self, store,
                                                          monkeypatch):
        from storage import article_store as A
        # No real backoff sleeps — keep the test instant + deterministic.
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        store.insert_batch([
            {"title": "MU earnings beat Q3", "link": "https://reuters.com/mu",
             "source": "rss", "published": "", "summary": "",
             "_relevance_score": 2.0},
        ])
        aid = A.article_id("https://reuters.com/mu", "MU earnings beat Q3")

        flaky = _FlakyConn(store.conn)
        store.conn = flaky

        before = A.lock_metrics()["lock_retries"]
        # Must NOT raise — the decorator retries the idempotent UPDATE.
        store.update_ai_scores_batch([(aid, 7.5, 1)])
        after = A.lock_metrics()["lock_retries"]

        assert flaky.em_calls == 2, "executemany should be retried exactly once"
        assert after == before + 1, "collision must increment the retry counter"
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id=?",
            (aid,),
        ).fetchone()
        assert row[0] == pytest.approx(7.5)
        assert row[1] == 1
        assert row[2] == "llm"

    def test_non_retryable_databaseerror_still_propagates(self, store,
                                                          monkeypatch):
        """Tight discrimination: an IntegrityError (also a DatabaseError
        subclass) must propagate, never be silently retried/swallowed."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        store.insert_batch([
            {"title": "Y title long enough", "link": "https://x.com/y",
             "source": "rss", "published": "", "summary": "",
             "_relevance_score": 1.0},
        ])
        aid = A.article_id("https://x.com/y", "Y title long enough")

        class _IntegrityConn:
            def __init__(self, real):
                self._real = real
            def executemany(self, *a, **k):
                raise sqlite3.IntegrityError(
                    "UNIQUE constraint failed: articles.id")
            def __getattr__(self, n):
                return getattr(self._real, n)

        store.conn = _IntegrityConn(store.conn)
        with pytest.raises(sqlite3.IntegrityError):
            store.update_ai_scores_batch([(aid, 5.0, 0)])


class _FlakyReadConn:
    """Raises the shared-connection cursor-collision DatabaseError on the
    FIRST ``execute`` call, then delegates normally. Models a lockless reader
    cursor on the shared ``self.conn`` colliding with a concurrent writer's
    ``executemany`` — the exact interleave the ``ef7fbe4`` decorator absorbs
    for writers, but which it was NOT applied to for the named reader victims
    (get_unalerted_urgent / stats / get_unscored / …)."""

    def __init__(self, real):
        self._real = real
        self.exec_calls = 0

    def execute(self, sql, params=()):
        self.exec_calls += 1
        if self.exec_calls == 1:
            raise sqlite3.DatabaseError("another row available")
        return self._real.execute(sql, params)

    def __getattr__(self, name):  # delegate commit/cursor/executemany/…
        return getattr(self._real, name)


class TestReadPathCursorCollisionRetry:
    """Regression pin: the pure-SELECT readers must ALSO retry the transient
    cursor-collision, not only the writers. ``ef7fbe4`` decorated the writers
    but left ``get_unalerted_urgent`` / ``stats`` / ``get_unscored`` /
    ``count_unscored`` / ``get_top_for_briefing`` / ``stats_since`` /
    ``get_briefings_for_training`` undecorated — so a reader colliding with a
    writer's ``executemany`` bubbled to ``alert_worker``'s broad except
    (urgent items unfetched that 20s cycle → DELAYED ALERTS) and 500'd the
    dashboard ``/api/stats``. If the decorator is ever removed from a reader
    this fails with an uncaught ``DatabaseError``."""

    def test_get_unalerted_urgent_retries_then_succeeds(self, store,
                                                        monkeypatch):
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="u1", url="https://reuters.com/mu-beat",
                    title="MU smashes Q3 DRAM guidance", source="rss",
                    urgency=1, ai_score=9.0)

        flaky = _FlakyReadConn(store.conn)
        store.conn = flaky
        before = A.lock_metrics()["lock_retries"]

        # Must NOT raise — the decorator retries the idempotent SELECT. An
        # undecorated reader would propagate DatabaseError here and the alert
        # worker would silently skip the urgent item this cycle.
        rows = store.get_unalerted_urgent()
        after = A.lock_metrics()["lock_retries"]

        assert flaky.exec_calls >= 2, "the SELECT must be retried after collision"
        assert after == before + 1, "collision must increment the retry counter"
        assert [r["_id"] for r in rows] == ["u1"]
        assert rows[0]["ai_score"] == pytest.approx(9.0)

    def test_stats_retries_then_succeeds(self, store, monkeypatch):
        """The observed-class bug: ``/api/stats`` 500'd because ``store.stats``
        raced the shared writer connection with no retry."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        _insert_raw(store, id="s1", url="https://x.com/a",
                    title="Title long enough here", source="rss",
                    urgency=1, ai_score=4.0)

        flaky = _FlakyReadConn(store.conn)
        store.conn = flaky
        before = A.lock_metrics()["lock_retries"]

        s = store.stats()  # must not raise
        after = A.lock_metrics()["lock_retries"]

        assert after == before + 1
        assert s["total"] == 1
        assert s["urgent"] == 1

    def test_read_collision_non_retryable_propagates(self, store, monkeypatch):
        """Tight discrimination on the read path too: a non-allowlisted
        DatabaseError (IntegrityError) must still propagate, never be
        silently retried/swallowed by the now-decorated readers."""
        from storage import article_store as A
        monkeypatch.setattr(A.time, "sleep", lambda *a, **k: None)

        class _IntegrityReadConn:
            def __init__(self, real):
                self._real = real
            def execute(self, *a, **k):
                raise sqlite3.IntegrityError("UNIQUE constraint failed")
            def __getattr__(self, n):
                return getattr(self._real, n)

        store.conn = _IntegrityReadConn(store.conn)
        with pytest.raises(sqlite3.IntegrityError):
            store.stats()
