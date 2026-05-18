"""ArticleStore.source_freshness — turns "which collectors went dark?" into
one queryable call instead of eyeballing the daemon log.

Pins two invariants that matter operationally:
  * synthetic backtest/opus rows are excluded (a dark *collector* must not be
    masked by backtest injections sharing the table — CLAUDE.md §5);
  * ordering is most-stale-first so a gone-dark source surfaces at the top.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _insert(store, *, id, url, title, source, first_seen):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, 0.0, 0, first_seen, 0,
             None, None),
        )
        store.conn.commit()


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).isoformat()


def test_excludes_synthetic_rows_and_orders_most_stale_first(store):
    # Fresh, busy collector.
    _insert(store, id="r1", url="https://a.com/1", title="rss one",
            source="rss", first_seen=_iso(2))
    _insert(store, id="r2", url="https://a.com/2", title="rss two",
            source="rss", first_seen=_iso(1))
    # A collector that went dark ~6h ago.
    _insert(store, id="p1", url="https://p.com/1", title="polygon one",
            source="polygon", first_seen=_iso(360))
    # Synthetic rows that must NOT appear (would mask the dark collector).
    _insert(store, id="b1", url="backtest://run_9/2026/BUY/AAPL",
            title="bt", source="backtest_run_9_winner", first_seen=_iso(1))
    _insert(store, id="o1", url="https://x.com/o1", title="opus",
            source="opus_annotation_cycle_3", first_seen=_iso(1))

    rows = store.source_freshness()
    by_src = {r["source"]: r for r in rows}

    assert set(by_src) == {"rss", "polygon"}, "synthetic rows leaked in"
    assert by_src["rss"]["count"] == 2
    assert by_src["polygon"]["count"] == 1
    # polygon (~6h stale) must rank ahead of rss (~1min) — dark source first.
    assert rows[0]["source"] == "polygon"
    assert by_src["polygon"]["newest_age_s"] > by_src["rss"]["newest_age_s"] > 0


def test_unparseable_timestamp_sorts_last_with_none_age(store):
    _insert(store, id="g1", url="https://g.com/1", title="good",
            source="rss", first_seen=_iso(5))
    _insert(store, id="x1", url="https://x.com/1", title="bad ts",
            source="weird", first_seen="not-a-timestamp")

    rows = store.source_freshness()
    by_src = {r["source"]: r for r in rows}

    assert by_src["weird"]["newest_age_s"] is None
    assert by_src["rss"]["newest_age_s"] is not None
    # Unknown age must not jump the queue ahead of a real stale source.
    assert rows[-1]["source"] == "weird"


def test_empty_db_returns_empty_list(store):
    assert store.source_freshness() == []
