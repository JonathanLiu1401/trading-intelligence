"""Tests for ArticleStore.source_urgent_yield()."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                kw_score=1.0, first_seen=None):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0),
        )
        store.conn.commit()


def _seed_source(store, source: str, total: int, urgent: int, alerted: int,
                 url_prefix: str = "https://x.com"):
    """Insert ``total`` rows from ``source``; ``urgent`` of them get
    urgency=1, ``alerted`` of them urgency=2; the rest urgency=0.

    Caller's contract: ``urgent + alerted <= total``. ``urgent`` is rows
    that count toward urgency>=1 but did NOT fire (urgency=1); ``alerted``
    is rows that fired (urgency=2 — also urgency>=1)."""
    assert urgent + alerted <= total, "test seeding error"
    for i in range(total):
        if i < urgent:
            u = 1
        elif i < urgent + alerted:
            u = 2
        else:
            u = 0
        _insert_raw(
            store, id=f"{source}_{i}",
            url=f"{url_prefix}/{source}/{i}",
            title=f"{source} title {i}",
            source=source, urgency=u,
        )


class TestUrgentYieldBasics:
    def test_returns_empty_on_empty_db(self, store):
        result = store.source_urgent_yield()
        assert result["window_h"] == 24
        assert result["by_source"] == []
        assert result["total_sources_qualifying"] == 0

    def test_computes_yield_correctly(self, store):
        """rss: 100 total, 10 urgent (urgency>=1: 6 queued + 4 alerted) → 10% urgent_pct, 4% alerted_pct."""
        _seed_source(store, "rss", total=100, urgent=6, alerted=4)
        result = store.source_urgent_yield(min_total=20)
        assert len(result["by_source"]) == 1
        row = result["by_source"][0]
        assert row["source"] == "rss"
        assert row["total"] == 100
        assert row["urgent"] == 10  # 6 at urgency=1 + 4 at urgency=2
        assert row["alerted"] == 4
        assert row["urgent_pct"] == pytest.approx(10.0)
        assert row["alerted_pct"] == pytest.approx(4.0)

    def test_zero_urgent_source(self, store):
        """A source with 50 articles but zero urgent — 0% yield."""
        _seed_source(store, "weather", total=50, urgent=0, alerted=0)
        result = store.source_urgent_yield(min_total=20)
        assert len(result["by_source"]) == 1
        row = result["by_source"][0]
        assert row["urgent"] == 0
        assert row["urgent_pct"] == 0.0
        assert row["alerted_pct"] == 0.0


class TestRanking:
    def test_highest_yield_first(self, store):
        """High-yield (signal-rich) sources rank above low-yield (noise) ones."""
        _seed_source(store, "reuters", total=100, urgent=20, alerted=10)  # 30%
        _seed_source(store, "stocktwits", total=500, urgent=5, alerted=0)  # 1%
        _seed_source(store, "rss", total=200, urgent=20, alerted=10)  # 15%
        result = store.source_urgent_yield(min_total=20, top_n=10)
        sources = [r["source"] for r in result["by_source"]]
        assert sources == ["reuters", "rss", "stocktwits"]
        # Verify percentages
        pct = {r["source"]: r["urgent_pct"] for r in result["by_source"]}
        assert pct["reuters"] == pytest.approx(30.0)
        assert pct["rss"] == pytest.approx(15.0)
        assert pct["stocktwits"] == pytest.approx(1.0)

    def test_alphabetical_tiebreak(self, store):
        """Equal urgent_pct → alphabetical on source (deterministic)."""
        _seed_source(store, "zeta", total=100, urgent=10, alerted=0)
        _seed_source(store, "alpha", total=100, urgent=10, alerted=0)
        _seed_source(store, "mike", total=100, urgent=10, alerted=0)
        result = store.source_urgent_yield(min_total=20)
        sources = [r["source"] for r in result["by_source"]]
        assert sources == ["alpha", "mike", "zeta"], "alphabetical tiebreak broken"


class TestMinTotalFilter:
    def test_below_min_total_excluded(self, store):
        """Sources with fewer articles than ``min_total`` are dropped — small-N noise
        would otherwise dominate the ranking (a single urgent row from a 2-article
        feed reads as 50% yield)."""
        _seed_source(store, "rare_feed", total=5, urgent=2, alerted=0)  # 40% but tiny
        _seed_source(store, "busy_feed", total=100, urgent=20, alerted=0)  # 20% real
        result = store.source_urgent_yield(min_total=20)
        sources = [r["source"] for r in result["by_source"]]
        assert "rare_feed" not in sources
        assert "busy_feed" in sources
        assert result["total_sources_qualifying"] == 1

    def test_min_total_at_exact_boundary_included(self, store):
        """A source with exactly ``min_total`` articles IS included (>= floor)."""
        _seed_source(store, "boundary", total=20, urgent=2, alerted=0)
        result = store.source_urgent_yield(min_total=20)
        sources = [r["source"] for r in result["by_source"]]
        assert "boundary" in sources


class TestBacktestIsolation:
    def test_excludes_backtest_urls(self, store):
        """CRITICAL invariant: backtest:// rows must NEVER inflate either side
        of the ratio (numerator or denominator). Otherwise yield is biased by
        synthetic training rows the analyst never asked about."""
        # Live rows
        _seed_source(store, "rss", total=50, urgent=5, alerted=2)
        # Backtest rows — should be invisible
        for i in range(100):
            _insert_raw(
                store, id=f"bt_{i}",
                url=f"backtest://run_1/2026-05-01/BUY/MU/{i}",
                title=f"synthetic {i}",
                source="rss",
                urgency=1, ai_score=8.0,
            )
        result = store.source_urgent_yield(min_total=20)
        rss_row = next(r for r in result["by_source"] if r["source"] == "rss")
        assert rss_row["total"] == 50, (
            "backtest rows inflated total — invariant #1 violated"
        )
        assert rss_row["urgent"] == 7, (
            "backtest urgent rows inflated numerator"
        )

    def test_excludes_backtest_source_tag(self, store):
        """``source LIKE 'backtest_%'`` rows also excluded — same invariant."""
        _seed_source(store, "rss", total=30, urgent=3, alerted=0)
        for i in range(20):
            _insert_raw(
                store, id=f"bs_{i}",
                url=f"https://x.com/synth/{i}",
                title=f"synthetic {i}",
                source="backtest_run_42_winner",
                urgency=1, ai_score=8.0,
            )
        result = store.source_urgent_yield(min_total=10)
        sources = {r["source"] for r in result["by_source"]}
        assert "backtest_run_42_winner" not in sources

    def test_excludes_opus_annotation_source(self, store):
        """``source LIKE 'opus_annotation%'`` rows also excluded."""
        _seed_source(store, "rss", total=30, urgent=3, alerted=0)
        for i in range(20):
            _insert_raw(
                store, id=f"opus_{i}",
                url=f"https://x.com/opus/{i}",
                title=f"opus {i}",
                source="opus_annotation_cycle_5",
                urgency=2, ai_score=9.0,
            )
        result = store.source_urgent_yield(min_total=10)
        sources = {r["source"] for r in result["by_source"]}
        assert "opus_annotation_cycle_5" not in sources


class TestWindowing:
    def test_window_h_excludes_older_rows(self, store):
        """Articles outside the window are not counted on either side."""
        # In-window: 50 rows, 10 urgent → 20% yield
        _seed_source(store, "rss", total=50, urgent=10, alerted=0)
        # Out-of-window: 200 rows from same source, all urgent — must NOT count
        old_seen = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).isoformat()
        for i in range(200):
            _insert_raw(
                store, id=f"old_{i}",
                url=f"https://x.com/old/{i}",
                title=f"old {i}",
                source="rss",
                urgency=1, first_seen=old_seen,
            )
        result = store.source_urgent_yield(hours=24, min_total=20)
        rss_row = next(r for r in result["by_source"] if r["source"] == "rss")
        assert rss_row["total"] == 50
        assert rss_row["urgent_pct"] == pytest.approx(20.0)


class TestReadOnlyInvariant:
    def test_does_not_mutate_articles(self, store):
        """Pure read — must NOT touch ai_score / ml_score / score_source /
        urgency on any row. All four load-bearing invariants intact."""
        _insert_raw(
            store, id="a", url="https://x.com/1", title="t",
            source="rss", urgency=1, ai_score=8.0,
        )
        # Snapshot
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()
        store.source_urgent_yield(min_total=1)
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()
        assert before == after, (
            "source_urgent_yield mutated an articles row — load-bearing "
            "invariant violated"
        )


class TestTopNCap:
    def test_top_n_truncates_results(self, store):
        """``top_n`` caps the returned list but ``total_sources_qualifying``
        reports the full count (UI can render '15 of N')."""
        for i in range(5):
            _seed_source(
                store, source=f"src_{i}", total=30,
                urgent=2 + i,  # Make yields differ so order is stable
                alerted=0,
            )
        result = store.source_urgent_yield(top_n=2, min_total=20)
        assert len(result["by_source"]) == 2
        assert result["total_sources_qualifying"] == 5
