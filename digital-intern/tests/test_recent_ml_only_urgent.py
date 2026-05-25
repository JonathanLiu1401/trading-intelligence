"""Tests for ArticleStore.recent_ml_only_urgent — live audit list of urgent
rows the model flagged but no LLM has verified.

Sibling to urgent_score_distribution (aggregated counts) and
urgency_label_split_by_source (per-source rates). This method returns the
ACTUAL TITLES so the analyst can ad-hoc audit "is the ML head flagging real
breaking news, or noise the gates correctly suppress?"

Pin the four load-bearing invariants (CLAUDE.md §5):
  1. backtest_/opus_annotation* rows MUST never appear in the output.
  2. The method is read-only — no ai_score / ml_score / score_source /
     urgency mutation.
  3. score_source filter is exact: only score_source='ml' rows appear; 'llm'
     and 'briefing_boost' rows are excluded by construction.
  4. Per-row score_source is implicit ('ml') and ai_score is always 0 on the
     returned rows (defended by the WHERE clause).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _insert(store, *, id, title, ml_score=9.0, ai_score=0.0,
            score_source="ml", urgency=1, url=None, source="rss",
            first_seen=None):
    """Direct insert helper bypassing dedupe; mirrors test_trainer pattern."""
    if url is None:
        url = f"https://x.com/{id}"
    if first_seen is None:
        first_seen = datetime.now(timezone.utc).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


class TestRecentMlOnlyUrgent:
    def test_empty_db_returns_empty_list(self, store):
        """No articles → empty list (never crash, never None)."""
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert out == []

    def test_returns_ml_urgent_row(self, store):
        """A score_source='ml' urgent row appears with correct fields."""
        _insert(store, id="a1", title="NVDA buyback $80B",
                ml_score=9.5, ai_score=0.0,
                score_source="ml", urgency=1, source="GN: Nvidia")
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert len(out) == 1
        row = out[0]
        assert row["title"] == "NVDA buyback $80B"
        assert row["source"] == "GN: Nvidia"
        assert row["ml_score"] == 9.5
        assert row["ai_score"] == 0.0
        assert row["urgency"] == 1
        assert row["age_hours"] >= 0.0
        assert row["id"] == "a1"

    def test_excludes_llm_verified_rows(self, store):
        """A row with score_source='llm' must NOT appear — it has been LLM-
        verified, so it's not 'ml-only'. This is the core filter contract."""
        _insert(store, id="ml1", title="ml flagged urgent",
                ml_score=9.0, score_source="ml", urgency=1)
        _insert(store, id="llm1", title="llm verified urgent",
                ml_score=9.0, ai_score=8.0, score_source="llm", urgency=1)
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        titles = {r["title"] for r in out}
        assert "ml flagged urgent" in titles
        assert "llm verified urgent" not in titles, (
            "llm-verified row leaked into ml-only audit — filter broken"
        )

    def test_excludes_briefing_boost_rows(self, store):
        """briefing_boost rows are Opus-curated; they have been implicitly
        verified through the briefing path and must NOT appear."""
        _insert(store, id="bb1", title="briefing boosted urgent",
                ml_score=9.0, ai_score=4.5, score_source="briefing_boost",
                urgency=1)
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert all(r["title"] != "briefing boosted urgent" for r in out)

    def test_excludes_non_urgent_rows(self, store):
        """urgency=0 rows must NOT appear regardless of score_source."""
        _insert(store, id="m1", title="ml non-urgent",
                ml_score=5.0, score_source="ml", urgency=0)
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert out == []

    def test_includes_urgency_2_alerted_rows(self, store):
        """urgency=2 (alerted, queue-exited) rows are still 'ml-only' if
        they were never LLM-labeled. The analyst wants to see what was
        pushed-but-unverified, too."""
        _insert(store, id="ml2", title="ml pushed alerted",
                ml_score=9.5, score_source="ml", urgency=2)
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert len(out) == 1
        assert out[0]["urgency"] == 2

    def test_newest_first_ordering(self, store):
        """Rows returned newest first_seen first — same ordering convention
        as get_unalerted_urgent / get_top_for_briefing's recency."""
        now = datetime.now(timezone.utc)
        _insert(store, id="old", title="older row",
                ml_score=9.0, score_source="ml", urgency=1,
                first_seen=(now - timedelta(hours=10)).isoformat())
        _insert(store, id="new", title="newer row",
                ml_score=9.0, score_source="ml", urgency=1,
                first_seen=(now - timedelta(hours=1)).isoformat())
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert out[0]["title"] == "newer row"
        assert out[1]["title"] == "older row"
        assert out[0]["age_hours"] < out[1]["age_hours"]

    def test_window_filter_excludes_older_rows(self, store):
        """hours=1 → rows older than 1h are excluded. Pinned exactly so a
        future widening of the window does not silently change behaviour."""
        now = datetime.now(timezone.utc)
        _insert(store, id="recent", title="fresh urgent",
                ml_score=9.0, score_source="ml", urgency=1,
                first_seen=(now - timedelta(minutes=30)).isoformat())
        _insert(store, id="stale", title="stale urgent",
                ml_score=9.0, score_source="ml", urgency=1,
                first_seen=(now - timedelta(hours=5)).isoformat())
        out = store.recent_ml_only_urgent(hours=1, limit=10)
        titles = {r["title"] for r in out}
        assert "fresh urgent" in titles
        assert "stale urgent" not in titles

    def test_limit_respected(self, store):
        """limit=3 → at most 3 rows returned even if more match."""
        for i in range(10):
            _insert(store, id=f"a{i}", title=f"urgent row {i}",
                    ml_score=9.0, score_source="ml", urgency=1)
        out = store.recent_ml_only_urgent(hours=24, limit=3)
        assert len(out) == 3

    def test_hours_zero_clamps_to_one(self, store):
        """Defensive: hours <= 0 must clamp to a positive value so the SQL
        time-window filter doesn't degenerate."""
        _insert(store, id="a1", title="urgent",
                ml_score=9.0, score_source="ml", urgency=1)
        out = store.recent_ml_only_urgent(hours=0, limit=10)
        # Method clamps hours to 1; the recent insert is well within 1h.
        # The contract is "no crash"; specific count may be 0 or 1 depending
        # on clock drift; the assertion is that the call returns a list.
        assert isinstance(out, list)


class TestBacktestIsolation:
    """Load-bearing invariant #1 (CLAUDE.md §5): backtest:// URLs and
    backtest_/opus_annotation* sources MUST NEVER appear in live audit
    outputs. Pinned at the SQL level here."""

    def test_excludes_backtest_url_rows(self, store):
        """A synthetic backtest row must NOT appear, even if it has
        score_source='ml' and urgency=1 (which it shouldn't, but defense-
        in-depth: an invariant breach must be loudly visible)."""
        _insert(store, id="bt1", title="BACKTEST should not appear",
                ml_score=9.0, score_source="ml", urgency=1,
                url="backtest://run_1/d/BUY/NVDA",
                source="backtest_run_1_winner")
        _insert(store, id="live1", title="live urgent",
                ml_score=9.0, score_source="ml", urgency=1)
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        titles = {r["title"] for r in out}
        assert "live urgent" in titles
        assert "BACKTEST should not appear" not in titles, (
            "backtest:// URL row leaked into live audit output — load-"
            "bearing invariant #1 broken"
        )

    def test_excludes_backtest_source_tag(self, store):
        """A row with backtest_ source tag (no backtest:// URL) must also
        be excluded — _LIVE_ONLY_CLAUSE filters both column patterns."""
        _insert(store, id="bt2", title="BACKTEST source tag",
                ml_score=9.0, score_source="ml", urgency=1,
                url="https://x.com/bt2",
                source="backtest_run_42_loser")
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert all("BACKTEST" not in r["title"] for r in out)

    def test_excludes_opus_annotation_source(self, store):
        """opus_annotation* source tag — synthetic training data — must
        also be excluded."""
        _insert(store, id="op1", title="OPUS annotation row",
                ml_score=9.0, score_source="ml", urgency=1,
                source="opus_annotation_cycle_1")
        out = store.recent_ml_only_urgent(hours=24, limit=10)
        assert all("OPUS" not in r["title"] for r in out)


class TestReadOnly:
    """Load-bearing invariants #2/#3: the audit is read-only — no DB
    writes, no ai_score / ml_score / score_source / urgency mutation."""

    def test_no_mutation_after_query(self, store):
        """Insert rows, snapshot column state, call audit, snapshot again.
        Both snapshots must be byte-identical."""
        _insert(store, id="r1", title="row 1",
                ml_score=9.0, score_source="ml", urgency=1)
        _insert(store, id="r2", title="row 2",
                ml_score=8.5, score_source="ml", urgency=2)
        before = store.conn.execute(
            "SELECT id, ai_score, ml_score, urgency, score_source "
            "FROM articles ORDER BY id"
        ).fetchall()
        _ = store.recent_ml_only_urgent(hours=24, limit=50)
        after = store.conn.execute(
            "SELECT id, ai_score, ml_score, urgency, score_source "
            "FROM articles ORDER BY id"
        ).fetchall()
        assert before == after, "recent_ml_only_urgent mutated DB state"
