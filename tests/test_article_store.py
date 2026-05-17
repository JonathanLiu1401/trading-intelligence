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
