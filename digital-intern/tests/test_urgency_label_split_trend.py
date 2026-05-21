"""Tests for ``ArticleStore.urgency_label_split_trend`` — the time-axis sibling
to ``urgency_label_split``.

What we verify (specific values, not no-crash):

  * Bucket boundaries align to ``now - hours``; ``hours=24, bucket_h=4`` yields
    exactly 6 buckets in oldest-first order.
  * Each bucket carries the same five-key score_source breakdown
    (``llm``/``ml``/``briefing_boost``/``null`` + ``total``) plus ``llm_fraction``.
  * Empty buckets are emitted with zero counts — the fixed-length series
    discipline that lets a dashboard iterate without conditional branches.
  * Synthetic backtest:// rows + ``backtest_*`` / ``opus_annotation*`` sources
    NEVER count (the load-bearing live-only invariant).
  * Aggregate ``llm_fraction`` over all buckets equals what
    ``urgency_label_split`` reports for the same window (anti-drift parity).
  * urgency=0 rows are not counted (sanity).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

import storage.article_store as article_store_mod
from storage.article_store import ArticleStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An ArticleStore backed by an isolated on-disk SQLite file."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    monkeypatch.setattr(article_store_mod, "LOCAL_PATH", db_dir)
    monkeypatch.setattr(article_store_mod, "USB_PATH", db_dir / "_no_usb")
    monkeypatch.setattr(article_store_mod, "_schema_ready_path", None)
    s = ArticleStore()
    yield s
    s.close()


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert(store, *, aid, url, source, score_source, ai_score, urgency,
            hours_ago):
    """Direct insert — bypass insert_batch so we can set urgency/score_source
    in one shot, exactly mirroring what the live scorer would leave."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, url, f"title {aid}", source, _iso(hours_ago), 1.0,
             ai_score, urgency, _iso(hours_ago), 0, None, score_source, None),
        )
        store.conn.commit()


class TestBucketShape:
    def test_24h_with_4h_buckets_produces_6_buckets(self, store):
        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        assert out["window_h"] == 24
        assert out["bucket_h"] == 4
        assert len(out["buckets"]) == 6
        # Oldest first — bucket_start[i] < bucket_start[i+1]
        starts = [b["bucket_start"] for b in out["buckets"]]
        assert starts == sorted(starts)

    def test_empty_buckets_emit_zero_counts(self, store):
        """No rows in the window — every bucket is zero, not missing."""
        out = store.urgency_label_split_trend(hours=24, bucket_h=6)
        assert len(out["buckets"]) == 4
        for b in out["buckets"]:
            assert b["total"] == 0
            assert b["llm"] == 0
            assert b["ml"] == 0
            assert b["briefing_boost"] == 0
            assert b["null"] == 0
            assert b["llm_fraction"] == 0.0
        assert out["total"] == 0
        assert out["llm_fraction"] == 0.0

    def test_hours_rounded_up_to_complete_bucket(self, store):
        """``hours=10, bucket_h=4`` rounds to 3 buckets (12h covered)."""
        out = store.urgency_label_split_trend(hours=10, bucket_h=4)
        assert out["window_h"] == 10
        assert out["bucket_h"] == 4
        # 3 buckets * 4h = 12h coverage so 10h is fully contained.
        assert len(out["buckets"]) == 3


class TestCounting:
    def test_per_source_counted_into_correct_bucket(self, store):
        """An urgent row 2h ago + one 8h ago + one 14h ago go into three
        different buckets (1st, 2nd, 4th bucket counting from the oldest)."""
        # 24h window, bucket=4h → 6 buckets covering 24h..0h ago, oldest first.
        # 2h ago → idx 5 (last); 8h ago → idx 3 (third-from-last);
        # 14h ago → idx 2 (third-from-first).
        _insert(store, aid="a", url="https://example.com/a", source="rss",
                score_source="llm", ai_score=9.0, urgency=2, hours_ago=2.0)
        _insert(store, aid="b", url="https://example.com/b", source="rss",
                score_source="ml", ai_score=0.0, urgency=1, hours_ago=8.0)
        _insert(store, aid="c", url="https://example.com/c", source="rss",
                score_source="briefing_boost", ai_score=4.5, urgency=1,
                hours_ago=14.0)

        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        bs = out["buckets"]
        # Recover indices by elapsed-hour position.
        assert bs[5]["llm"] == 1
        assert bs[5]["total"] == 1
        assert bs[5]["llm_fraction"] == 1.0
        assert bs[3]["ml"] == 1
        assert bs[3]["total"] == 1
        assert bs[3]["llm_fraction"] == 0.0
        assert bs[2]["briefing_boost"] == 1
        assert bs[2]["total"] == 1
        # briefing_boost counts as vetted, so llm_fraction == 1.0.
        assert bs[2]["llm_fraction"] == 1.0
        # Empty buckets (idx 0, 1, 4) stay at zero.
        for i in (0, 1, 4):
            assert bs[i]["total"] == 0
            assert bs[i]["llm_fraction"] == 0.0

    def test_grand_totals_match_urgency_label_split(self, store):
        """Aggregate llm_fraction MUST equal urgency_label_split's value over
        the same window — anti-drift parity, the same SSOT discipline the
        per-source/per-ticker siblings carry."""
        _insert(store, aid="a", url="https://example.com/a", source="rss",
                score_source="llm", ai_score=9.0, urgency=2, hours_ago=2.0)
        _insert(store, aid="b", url="https://example.com/b", source="rss",
                score_source="ml", ai_score=0.0, urgency=1, hours_ago=8.0)
        _insert(store, aid="c", url="https://example.com/c", source="rss",
                score_source="ml", ai_score=0.0, urgency=1, hours_ago=14.0)

        trend = store.urgency_label_split_trend(hours=24, bucket_h=4)
        agg = store.urgency_label_split(hours=24)
        assert trend["total"] == agg["total"]
        assert trend["llm_fraction"] == agg["llm_fraction"]

    def test_urgency_zero_rows_ignored(self, store):
        """A row with urgency=0 (never marked urgent) is not counted —
        mirrors urgency_label_split's WHERE urgency>=1."""
        _insert(store, aid="a", url="https://example.com/a", source="rss",
                score_source="llm", ai_score=9.0, urgency=0, hours_ago=2.0)
        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        assert out["total"] == 0


class TestBacktestIsolation:
    """LOAD-BEARING INVARIANT: backtest:// and backtest_*/opus_annotation*
    rows must NEVER count. The whole reason this primitive exists is to
    measure live-news calibration; a synthetic injection burst would
    otherwise either inflate or deflate the trend."""

    def test_backtest_url_row_never_counted(self, store):
        _insert(store, aid="real", url="https://example.com/real",
                source="rss", score_source="llm", ai_score=9.0,
                urgency=1, hours_ago=2.0)
        _insert(store, aid="syn", url="backtest://run_42/2026-05-21/BUY/MU",
                source="rss",  # source ok, only the URL marks it synthetic
                score_source="llm", ai_score=9.0, urgency=1, hours_ago=2.0)
        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        assert out["total"] == 1, "backtest:// row leaked into the trend"

    def test_backtest_source_row_never_counted(self, store):
        _insert(store, aid="real", url="https://example.com/real",
                source="rss", score_source="llm", ai_score=9.0,
                urgency=1, hours_ago=2.0)
        _insert(store, aid="syn", url="https://example.com/syn",
                source="backtest_run_42_winner", score_source="llm",
                ai_score=9.0, urgency=1, hours_ago=2.0)
        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        assert out["total"] == 1, "backtest_* source leaked into the trend"

    def test_opus_annotation_source_row_never_counted(self, store):
        _insert(store, aid="real", url="https://example.com/real",
                source="rss", score_source="llm", ai_score=9.0,
                urgency=1, hours_ago=2.0)
        _insert(store, aid="syn", url="https://example.com/syn",
                source="opus_annotation_cycle_5", score_source="llm",
                ai_score=9.0, urgency=1, hours_ago=2.0)
        out = store.urgency_label_split_trend(hours=24, bucket_h=4)
        assert out["total"] == 1, "opus_annotation* leaked into the trend"
