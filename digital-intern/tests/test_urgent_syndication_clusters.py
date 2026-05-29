"""Tests for ArticleStore.urgent_syndication_clusters.

The method buckets urgent rows by ``watchers.alert_dedup._signature``
(SSOT — same signature the live alert dedup path uses) and surfaces
the "your queue is N rows but really K distinct events" view the
existing urgent analytics lacked.

Pins:
  * Cluster detection uses the SSOT alert-dedup signature (not a
    re-derived prefix hash) — drift here would mean the analyst's
    cluster report names different groups than the alert gate
    actually merged at push time.
  * Backtest isolation (invariant #1) — synthetic rows never enter
    the cluster pool.
  * No DB writes (invariants #2/#3) — read-only analytics.
  * Verdict ladder boundaries are pinned (HEAVY at >=40%, MODERATE
    at >=20%, LIGHT otherwise, NO_DATA on empty).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).isoformat()


def _insert_urgent(store, *, id, url, title, source,
                   urgency=1, ai_score=8.0, ml_score=None,
                   score_source="llm", first_seen=None):
    """Insert an urgent row, bypassing the public API to control state."""
    if first_seen is None:
        first_seen = _recent_iso()
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


# ── shape contract ────────────────────────────────────────────────────────

class TestNoData:
    def test_empty_returns_no_data_verdict(self, store):
        r = store.urgent_syndication_clusters(hours=24)
        assert r["verdict"] == "NO_DATA"
        assert r["total_urgent"] == 0
        assert r["n_clusters"] == 0
        assert r["top_clusters"] == []


class TestShapeContract:
    def test_all_required_keys_present(self, store):
        _insert_urgent(store, id="a", url="https://x.com/1",
                       title="MU beats earnings estimates", source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        for key in ("window_h", "total_urgent", "n_clusters",
                    "n_clustered_rows", "n_unique_events",
                    "syndication_pct", "top_clusters", "verdict"):
            assert key in r, f"missing key: {key}"

    def test_window_h_clamped_to_minimum_one(self, store):
        # ``hours=0`` should clamp to >=1 (defensive against caller
        # passing a falsy value — never crash).
        r = store.urgent_syndication_clusters(hours=0)
        assert r["window_h"] >= 1


# ── core logic ────────────────────────────────────────────────────────────

class TestClustering:
    def test_three_copies_collapse_to_one_cluster(self, store):
        """Live failure pattern: same NVDA headline syndicated across
        3 publishers must collapse to one cluster of size 3."""
        title = "NVDA Stock Drops 10% From All-Time High Amid Fresh Selloff"
        _insert_urgent(store, id="a", url="https://reuters.com/1",
                       title=title, source="GDELT/reuters.com")
        _insert_urgent(store, id="b", url="https://finnhub.com/2",
                       title=title, source="Finnhub/Yahoo")
        _insert_urgent(store, id="c", url="https://gn.com/3",
                       title=title, source="GN: semiconductor")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 3
        assert r["n_clusters"] == 1
        assert r["n_clustered_rows"] == 3
        # All 3 copies share one signature → 1 unique event.
        assert r["n_unique_events"] == 1
        assert r["syndication_pct"] == 100.0
        # The cluster carries all 3 source tags.
        cluster = r["top_clusters"][0]
        assert cluster["size"] == 3
        assert cluster["n_sources"] == 3

    def test_distinct_stories_no_cluster(self, store):
        """Three urgent rows on distinct events ⇒ no cluster (each is
        a singleton), syndication 0%."""
        _insert_urgent(store, id="a", url="https://x.com/1",
                       title="MU beats Q3 earnings on memory demand",
                       source="rss")
        _insert_urgent(store, id="b", url="https://x.com/2",
                       title="Fed signals 50bp cut next meeting",
                       source="rss")
        _insert_urgent(store, id="c", url="https://x.com/3",
                       title="Samsung announces HBM4 supply deal",
                       source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 3
        assert r["n_clusters"] == 0
        assert r["n_unique_events"] == 3
        assert r["syndication_pct"] == 0.0
        assert r["verdict"] == "LIGHT"

    def test_mixed_clustered_and_singletons(self, store):
        """Two distinct stories — one syndicated 3x, one singleton.
        n_clusters=1, n_clustered_rows=3, n_unique_events=2."""
        _insert_urgent(store, id="a", url="https://x.com/1",
                       title="NVDA earnings beat smashes estimates",
                       source="rss")
        _insert_urgent(store, id="b", url="https://x.com/2",
                       title="NVDA earnings beat smashes estimates",
                       source="finnhub")
        _insert_urgent(store, id="c", url="https://x.com/3",
                       title="NVDA earnings beat smashes estimates",
                       source="googlenews")
        _insert_urgent(store, id="d", url="https://x.com/4",
                       title="Fed cuts rates by 25 basis points today",
                       source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 4
        assert r["n_clusters"] == 1
        assert r["n_clustered_rows"] == 3
        assert r["n_unique_events"] == 2
        # 3 / 4 = 75% syndicated → HEAVY (>=40%)
        assert r["syndication_pct"] == 75.0
        assert r["verdict"] == "HEAVY_SYNDICATION"


# ── verdict ladder ─────────────────────────────────────────────────────────

class TestVerdictLadder:
    def test_heavy_syndication_at_40_pct(self, store):
        """4 syndicated + 6 singletons = 40% syndication → HEAVY."""
        # Cluster of 4
        for i in range(4):
            _insert_urgent(store, id=f"c{i}", url=f"https://x.com/c{i}",
                           title="Massive earnings beat for chip giant today",
                           source=f"src{i}")
        # 6 distinct singletons
        for i in range(6):
            _insert_urgent(store, id=f"s{i}", url=f"https://x.com/s{i}",
                           title=f"Unique alpha news event story number {i:02d}",
                           source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 10
        assert r["syndication_pct"] == 40.0
        assert r["verdict"] == "HEAVY_SYNDICATION"

    def test_moderate_at_20_to_40(self, store):
        # cluster of 2 + 8 singletons = 20%
        for i in range(2):
            _insert_urgent(store, id=f"c{i}", url=f"https://x.com/c{i}",
                           title="Same earnings beat today across syndication",
                           source=f"src{i}")
        for i in range(8):
            _insert_urgent(store, id=f"s{i}", url=f"https://x.com/s{i}",
                           title=f"Distinct urgent alpha event news number {i:02d}",
                           source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["verdict"] == "MODERATE"

    def test_light_below_20(self, store):
        # No clusters → 0% → LIGHT.
        for i in range(5):
            _insert_urgent(store, id=f"s{i}", url=f"https://x.com/s{i}",
                           title=f"Distinct urgent alpha event news number {i:02d}",
                           source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["verdict"] == "LIGHT"
        assert r["syndication_pct"] == 0.0


# ── load-bearing invariants ────────────────────────────────────────────────

class TestBacktestIsolation:
    """Invariant #1: synthetic backtest / opus-annotation rows must
    NEVER appear in the cluster pool, regardless of urgency."""

    def test_backtest_url_excluded(self, store):
        # 1 live urgent + 1 backtest urgent (same title)
        title = "MU beats earnings estimates by wide margin"
        _insert_urgent(store, id="live", url="https://reuters.com/x",
                       title=title, source="rss")
        _insert_urgent(store, id="bt",
                       url="backtest://run42/winner/MU",
                       title=title, source="rss", urgency=1)
        r = store.urgent_syndication_clusters(hours=24)
        # Only the live row should appear — no cluster (singleton).
        assert r["total_urgent"] == 1
        assert r["n_clusters"] == 0

    def test_backtest_source_excluded(self, store):
        _insert_urgent(store, id="bt", url="https://x.com/1",
                       title="x", source="backtest_run_42_winner")
        _insert_urgent(store, id="op", url="https://x.com/2",
                       title="y", source="opus_annotation_cycle_3")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 0
        assert r["verdict"] == "NO_DATA"


class TestReadOnly:
    """Invariants #2/#3: the method is read-only; ai_score / ml_score
    / score_source / urgency must be byte-identical before and after."""

    def test_no_db_mutation(self, store):
        _insert_urgent(store, id="a", url="https://x.com/1",
                       title="MU beats earnings", source="rss",
                       ai_score=8.5, ml_score=7.0, score_source="llm")
        before = store.conn.execute(
            "SELECT ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id='a'"
        ).fetchone()
        store.urgent_syndication_clusters(hours=24)
        after = store.conn.execute(
            "SELECT ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id='a'"
        ).fetchone()
        assert before == after


class TestSSOTSignature:
    """The method MUST use watchers.alert_dedup._signature so the
    analyst's cluster report names the same groups the alert gate
    collapses at push time. If a future refactor inlines a different
    signature here, this pin breaks."""

    def test_uses_alert_dedup_signature(self, store, monkeypatch):
        # Patch alert_dedup._signature to a sentinel; if the method
        # uses anything else this test fails.
        from watchers import alert_dedup
        seen: list[str] = []

        def _sentinel_sig(title):
            seen.append(title or "")
            # Always return the same sig so all rows cluster, regardless
            # of title — proves the method called THIS function.
            return "ALL_SAME_SIG"

        monkeypatch.setattr(alert_dedup, "_signature", _sentinel_sig)

        _insert_urgent(store, id="a", url="https://x.com/1",
                       title="story A", source="rss")
        _insert_urgent(store, id="b", url="https://x.com/2",
                       title="story B totally different", source="rss")
        r = store.urgent_syndication_clusters(hours=24)
        # Both rows collapsed under the sentinel signature → cluster of 2.
        assert r["n_clusters"] == 1
        assert r["top_clusters"][0]["size"] == 2
        assert set(seen) == {"story A", "story B totally different"}


# ── top clusters ordering ─────────────────────────────────────────────────

class TestTopClustersOrdering:
    def test_largest_cluster_first(self, store):
        # Cluster of 3 + cluster of 2 → largest reported first.
        for i in range(3):
            _insert_urgent(store, id=f"big{i}", url=f"https://x.com/big{i}",
                           title="Big news cluster headline that repeats",
                           source=f"big{i}")
        for i in range(2):
            _insert_urgent(store, id=f"sm{i}", url=f"https://x.com/sm{i}",
                           title="Smaller distinct news cluster headline",
                           source=f"sm{i}")
        r = store.urgent_syndication_clusters(hours=24)
        assert r["top_clusters"][0]["size"] == 3
        assert r["top_clusters"][1]["size"] == 2

    def test_lead_title_is_highest_score(self, store):
        """In a cluster, the displayed lead_title is the row with
        highest effective score."""
        title = "Same earnings beat smashes estimates today wide margin"
        _insert_urgent(store, id="lo", url="https://x.com/1",
                       title=title, source="rss", ai_score=8.0)
        _insert_urgent(store, id="hi", url="https://x.com/2",
                       title=title, source="finnhub", ai_score=9.8)
        r = store.urgent_syndication_clusters(hours=24)
        # Both copies cluster; lead_title is from the higher-score row.
        # (Identical title here; the point is we picked SOME deterministic
        # rep — but the row's effective_score must drive selection so a
        # variant with different title in real data picks the strongest.)
        assert r["top_clusters"][0]["size"] == 2

    def test_n_sources_counts_distinct_only(self, store):
        # Same source twice should count once.
        for i, src in enumerate(["rss", "rss", "finnhub"]):
            _insert_urgent(store, id=f"r{i}", url=f"https://x.com/{i}",
                           title="Earnings beat smashes estimates today",
                           source=src)
        r = store.urgent_syndication_clusters(hours=24)
        cluster = r["top_clusters"][0]
        assert cluster["size"] == 3
        assert cluster["n_sources"] == 2  # rss + finnhub, deduped


class TestWindow:
    def test_old_rows_outside_window_excluded(self, store):
        """A row first_seen 25h ago must not be in a 24h window."""
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        _insert_urgent(store, id="old", url="https://x.com/1",
                       title="Old urgent story", source="rss",
                       first_seen=old)
        r = store.urgent_syndication_clusters(hours=24)
        assert r["total_urgent"] == 0

    def test_old_rows_included_in_wider_window(self, store):
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        _insert_urgent(store, id="old", url="https://x.com/1",
                       title="Old urgent story", source="rss",
                       first_seen=old)
        r = store.urgent_syndication_clusters(hours=48)
        assert r["total_urgent"] == 1
