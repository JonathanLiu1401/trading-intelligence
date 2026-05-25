"""``ArticleStore.label_production_rate`` — per-minute LLM throughput verdict.

Time-derivative sibling to ``urgency_label_split``: that method counts
urgent rows by ``score_source`` and is silent on whether the Sonnet path
is actively producing labels on the full live stream. This method
answers the analyst's "is Sonnet labelling RIGHT NOW?" question with a
verdict at a glance.

Pins the contract documented in the method docstring: returns the four
score_source buckets exhaustively (including NULL), an aggregate
rate-per-minute, and a verdict ladder
(NO_DATA / DARK / THROTTLED / HEALTHY) that the dashboard / heartbeat
briefing can render without conditional branches.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _iso(offset_min: float = 0.0) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=offset_min)
    ).isoformat()


def _insert(store, *, id, url, title, source="rss", score_source=None,
            ai_score=0.0, ml_score=None, urgency=0, first_seen=None):
    """Insert a row bypassing the public API so tests can build any state."""
    if first_seen is None:
        first_seen = _iso(0)
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


class TestEmptyAndNoData:
    def test_empty_db_returns_no_data(self, store):
        r = store.label_production_rate(window_min=60)
        assert r["total"] == 0
        assert r["by_source"] == {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0}
        assert r["rate_per_min"] == 0.0
        assert r["llm_rate_per_min"] == 0.0
        assert r["unscored_fraction"] == 0.0
        assert r["verdict"] == "NO_DATA"

    def test_window_min_clamped_to_one(self, store):
        r = store.label_production_rate(window_min=0)
        assert r["window_min"] == 1
        r2 = store.label_production_rate(window_min=-5)
        assert r2["window_min"] == 1


class TestVerdictLadder:
    def test_dark_when_only_ml_labels(self, store):
        # 10 ML-only rows in window — every alert/score this hour is
        # unverified. The dashboard reads this as 100% unverified.
        for i in range(10):
            _insert(store, id=f"m{i}", url=f"https://x.com/{i}",
                    title=f"row {i}", score_source="ml", ml_score=7.0)
        r = store.label_production_rate(window_min=60)
        assert r["total"] == 10
        assert r["by_source"]["ml"] == 10
        assert r["by_source"]["llm"] == 0
        assert r["llm_rate_per_min"] == 0.0
        assert r["verdict"] == "DARK"

    def test_throttled_when_rate_below_one_per_20min(self, store):
        # Window = 60 min, 1 LLM label → 0.0167 LLM/min < 0.05 threshold.
        # Add some ML traffic alongside so total > 0 too.
        _insert(store, id="l0", url="https://x.com/l0", title="L",
                score_source="llm", ai_score=6.0)
        for i in range(20):
            _insert(store, id=f"m{i}", url=f"https://x.com/m{i}",
                    title=f"r {i}", score_source="ml", ml_score=5.0)
        r = store.label_production_rate(window_min=60)
        assert r["by_source"]["llm"] == 1
        assert r["llm_rate_per_min"] == pytest.approx(round(1 / 60, 3))
        assert r["verdict"] == "THROTTLED"

    def test_healthy_when_llm_rate_above_threshold(self, store):
        # Plenty of LLM labels in window → rate >> 0.05/min.
        for i in range(10):
            _insert(store, id=f"l{i}", url=f"https://x.com/l{i}",
                    title=f"L {i}", score_source="llm", ai_score=6.0)
        r = store.label_production_rate(window_min=60)
        assert r["by_source"]["llm"] == 10
        assert r["llm_rate_per_min"] == pytest.approx(round(10 / 60, 3))
        assert r["verdict"] == "HEALTHY"

    def test_briefing_boost_counts_as_llm_vetted(self, store):
        """``briefing_boost`` rows are Opus-curated and count alongside 'llm'
        for the vetted-rate verdict (matches the existing
        urgency_label_split's llm_fraction convention)."""
        for i in range(6):
            _insert(store, id=f"b{i}", url=f"https://x.com/b{i}",
                    title=f"B {i}", score_source="briefing_boost",
                    ai_score=4.5)
        r = store.label_production_rate(window_min=60)
        assert r["by_source"]["briefing_boost"] == 6
        # 6 / 60 = 0.1/min > 0.05 threshold → HEALTHY
        assert r["verdict"] == "HEALTHY"
        assert r["llm_rate_per_min"] == pytest.approx(0.1)


class TestWindowBoundary:
    def test_rows_outside_window_excluded(self, store):
        """A row first_seen > window_min ago must not count toward this
        window's rate — exactly mirroring urgency_label_split's first_seen
        boundary semantics."""
        # In-window: 2 LLM rows from 30min ago.
        _insert(store, id="in1", url="https://x.com/in1", title="In 1",
                score_source="llm", ai_score=7.0, first_seen=_iso(30))
        _insert(store, id="in2", url="https://x.com/in2", title="In 2",
                score_source="llm", ai_score=8.0, first_seen=_iso(45))
        # Out-of-window: 5 LLM rows from 2h ago.
        for i in range(5):
            _insert(store, id=f"old{i}", url=f"https://x.com/old{i}",
                    title=f"old {i}", score_source="llm", ai_score=8.0,
                    first_seen=_iso(120))
        r = store.label_production_rate(window_min=60)
        assert r["by_source"]["llm"] == 2, (
            "rows older than the window must be excluded — the dashboard "
            "would otherwise report stale labels as fresh throughput"
        )
        assert r["total"] == 2


class TestLiveOnlyClause:
    def test_backtest_rows_excluded(self, store):
        """Critical invariant #1 (CLAUDE.md §5): synthetic backtest /
        opus-annotation rows must NEVER inflate the live label-production
        rate, otherwise a paper-trader backtest replay would show as
        Sonnet activity. ``_LIVE_ONLY_CLAUSE`` is what enforces this."""
        # Real live rows.
        _insert(store, id="live1", url="https://reuters.com/x", title="x",
                source="rss", score_source="llm", ai_score=7.0)
        # Synthetic rows must be invisible to this method.
        _insert(store, id="bt1", url="backtest://run_1/2026/BUY/MU",
                title="bt", source="backtest_run_1", score_source="llm",
                ai_score=9.0)
        _insert(store, id="bt2", url="https://example.com/x",
                title="op", source="opus_annotation_cycle_3",
                score_source="llm", ai_score=8.0)
        _insert(store, id="bt3", url="https://example.com/y",
                title="bt", source="backtest_run_42_winner",
                score_source="llm", ai_score=9.0)

        r = store.label_production_rate(window_min=60)
        assert r["total"] == 1, (
            f"backtest rows leaked into label_production_rate: total={r['total']}"
        )
        assert r["by_source"]["llm"] == 1


class TestUnscoredAccounting:
    def test_null_score_source_bucket_populated(self, store):
        """An article inserted but not yet labeled (score_source=NULL) must
        land in the 'null' bucket so the analyst can see the unscored
        backlog as a fraction of recent throughput. Matches the same four-
        bucket discipline as urgency_label_split."""
        for i in range(8):
            _insert(store, id=f"u{i}", url=f"https://x.com/u{i}",
                    title=f"u {i}", score_source=None)
        for i in range(2):
            _insert(store, id=f"l{i}", url=f"https://x.com/l{i}",
                    title=f"l {i}", score_source="llm", ai_score=6.0)
        r = store.label_production_rate(window_min=60)
        assert r["by_source"]["null"] == 8
        assert r["by_source"]["llm"] == 2
        assert r["total"] == 10
        assert r["unscored_fraction"] == pytest.approx(0.8)
