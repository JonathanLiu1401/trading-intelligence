"""``ArticleStore.urgency_label_split_by_source`` — per-source slice of the
calibration metric.

The aggregate ``urgency_label_split`` answers "is the alert path mostly
LLM-vetted?" — pinned in production at 29% for days. The analyst then needs
the next question answered: *which sources* generate the bulk of the
remaining ML-only urgent firings. This module is that slice, exposed as a
sibling to ``source_freshness`` / ``source_throughput`` / the per-source
recap-template / quote-widget audits.

Discriminating asserts:

  1. Per-source counts equal the aggregate (no rows lost / double-counted).
  2. The four canonical buckets (``llm`` / ``ml`` / ``briefing_boost`` /
     ``null``) exist on every row even when zero — dashboard-stable shape.
  3. Sort order is ML-DESC, alphabetical tiebreak — the worst-offender
     feeder is first so the analyst's prune question is answered at a glance.
  4. ``top_n`` caps the list but ``total_sources`` reports the full count
     (mirrors ``audit_by_source``'s discipline).
  5. Backtest-isolation: synthetic ``backtest://`` / ``backtest_*`` /
     ``opus_annotation*`` rows are NEVER counted — invariant #1.
  6. Non-urgent rows (urgency=0) are NEVER counted — same predicate as
     the aggregate metric.
  7. The ``hours`` window is respected — an old row outside the window is
     excluded even when urgent.
  8. ``llm_fraction`` per-row is ``(llm + briefing_boost) / total``,
     matching the aggregate's definition exactly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _recent(minutes_ago: int = 5) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


def _insert_raw(
    store, *, id, url, title, source, urgency=1, ai_score=0.0,
    ml_score=None, score_source=None, kw_score=1.0, first_seen=None,
):
    """Build any (urgency, score_source) state without touching the live API."""
    if first_seen is None:
        first_seen = _recent()
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


class TestShape:
    def test_empty_store_returns_empty_list(self, store):
        out = store.urgency_label_split_by_source(hours=24)
        assert out["window_h"] == 24
        assert out["by_source"] == []
        assert out["total_urgent"] == 0
        assert out["total_sources"] == 0

    def test_single_source_has_all_four_buckets(self, store):
        """Even a source contributing only ``llm`` rows must expose the four
        canonical buckets so the dashboard can render a stable column set."""
        _insert_raw(
            store, id="a", url="https://reuters.com/1",
            title="LLM-vetted urgent", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_source(hours=24)
        assert len(out["by_source"]) == 1
        row = out["by_source"][0]
        assert row["source"] == "rss"
        for key in ("llm", "ml", "briefing_boost", "null"):
            assert key in row, f"bucket {key} missing — dashboard shape unstable"
        assert row["llm"] == 1
        assert row["ml"] == 0
        assert row["llm_fraction"] == 1.0


class TestPerSourceCounts:
    def test_mixed_sources_partition_exactly(self, store):
        """Three sources, mixed score_source tags. Per-source counts must
        sum to the aggregate ``urgency_label_split`` figure exactly — no
        loss, no double-count."""
        # Source A: 3 LLM-vetted + 1 briefing_boost = 4 total (vetted)
        for i in range(3):
            _insert_raw(
                store, id=f"a_llm{i}", url=f"https://reuters.com/{i}",
                title=f"Reuters wire {i}", source="rss",
                urgency=1, ai_score=9.0, score_source="llm",
            )
        _insert_raw(
            store, id="a_bb", url="https://reuters.com/bb",
            title="Opus-curated", source="rss",
            urgency=1, ai_score=4.5, score_source="briefing_boost",
        )
        # Source B: 5 ML-only
        for i in range(5):
            _insert_raw(
                store, id=f"b_ml{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent {i}", source="GN: Nvidia",
                urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
            )
        # Source C: 2 ML-only + 1 legacy NULL tag
        for i in range(2):
            _insert_raw(
                store, id=f"c_ml{i}", url=f"https://yfin.com/m{i}",
                title=f"YF model urgent {i}", source="yfinance/Motley Fool",
                urgency=1, ai_score=0.0, ml_score=9.5, score_source="ml",
            )
        _insert_raw(
            store, id="c_null", url="https://yfin.com/legacy",
            title="Legacy untagged row", source="yfinance/Motley Fool",
            urgency=1, ai_score=7.0, score_source=None,
        )

        out = store.urgency_label_split_by_source(hours=24)
        assert out["total_urgent"] == 4 + 5 + 3  # 12
        assert out["total_sources"] == 3
        sources = {r["source"]: r for r in out["by_source"]}
        assert sources["rss"]["llm"] == 3
        assert sources["rss"]["briefing_boost"] == 1
        assert sources["rss"]["total"] == 4
        assert sources["rss"]["llm_fraction"] == 1.0
        assert sources["GN: Nvidia"]["ml"] == 5
        assert sources["GN: Nvidia"]["total"] == 5
        assert sources["GN: Nvidia"]["llm_fraction"] == 0.0
        assert sources["yfinance/Motley Fool"]["ml"] == 2
        assert sources["yfinance/Motley Fool"]["null"] == 1
        assert sources["yfinance/Motley Fool"]["total"] == 3
        # 0/3 vetted under this source
        assert sources["yfinance/Motley Fool"]["llm_fraction"] == 0.0

        # Cross-check vs aggregate: sums must match exactly.
        agg = store.urgency_label_split(hours=24)
        for key in ("llm", "ml", "briefing_boost", "null"):
            assert sum(r[key] for r in out["by_source"]) == agg["by_source"][key], (
                f"per-source {key} sum drifted from aggregate — metric "
                f"definitions are inconsistent"
            )


class TestSortOrder:
    def test_worst_ml_offender_first(self, store):
        """The analyst-facing prune question demands worst-first ordering:
        most ML-only count at the top. Alphabetical tiebreak on equal
        counts so the order is reproducible (no nondeterminism)."""
        # Source A: 2 ml
        for i in range(2):
            _insert_raw(
                store, id=f"a{i}", url=f"https://a.com/{i}",
                title=f"A wire {i}", source="src_alpha",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        # Source B: 5 ml (most — should win the sort)
        for i in range(5):
            _insert_raw(
                store, id=f"b{i}", url=f"https://b.com/{i}",
                title=f"B wire {i}", source="src_beta",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        # Source C: 2 ml (alphabetical tiebreak with A)
        for i in range(2):
            _insert_raw(
                store, id=f"c{i}", url=f"https://c.com/{i}",
                title=f"C wire {i}", source="src_gamma",
                urgency=1, ml_score=9.0, score_source="ml",
            )

        out = store.urgency_label_split_by_source(hours=24)
        sources = [r["source"] for r in out["by_source"]]
        assert sources == ["src_beta", "src_alpha", "src_gamma"], (
            f"unexpected sort: {sources}; expected ml-desc with alphabetical tiebreak"
        )

    def test_zero_ml_sources_sort_alphabetically(self, store):
        """A source whose urgent rows are 100% LLM-vetted has ml=0 — must
        still appear, just at the bottom of the list, alphabetically ordered."""
        _insert_raw(
            store, id="z1", url="https://z.com/1",
            title="zebra urgent", source="zebra_source",
            urgency=1, ai_score=8.5, score_source="llm",
        )
        _insert_raw(
            store, id="a1", url="https://a.com/1",
            title="aardvark urgent", source="aardvark_source",
            urgency=1, ai_score=8.5, score_source="llm",
        )
        out = store.urgency_label_split_by_source(hours=24)
        sources = [r["source"] for r in out["by_source"]]
        assert sources == ["aardvark_source", "zebra_source"]


class TestTopN:
    def test_top_n_caps_list_but_not_total_sources(self, store):
        """``by_source`` is capped at ``top_n``; ``total_sources`` reports the
        full count so a UI can render "showing N of M". Mirrors
        ``audit_by_source``'s shape."""
        for i in range(5):
            _insert_raw(
                store, id=f"s{i}", url=f"https://s.com/{i}",
                title=f"urgent row {i}", source=f"source_{i:02d}",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        out = store.urgency_label_split_by_source(hours=24, top_n=2)
        assert len(out["by_source"]) == 2
        assert out["total_sources"] == 5
        # total_urgent counts ALL rows, not just the top_n shown
        assert out["total_urgent"] == 5


class TestBacktestIsolation:
    def test_synthetic_rows_never_inflate_a_source(self, store):
        """A backtest:// URL or ``backtest_*`` / ``opus_annotation*`` source
        tag must NEVER inflate the per-source figure — the metric is consumed
        by a live dashboard; an injection burst must never fake a vetted /
        unvetted figure. Same invariant as the rest of the audit family."""
        # Live row from a real publisher.
        _insert_raw(
            store, id="live", url="https://reuters.com/1",
            title="real wire", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        # Three classes of synthetic poison — all carry urgency=1 and would
        # add 1 to either an existing source bucket or create a phantom one
        # if isolation broke.
        _insert_raw(
            store, id="bt_url", url="backtest://run_1/2026-01-01/BUY/MU",
            title="synthetic backtest URL row", source="rss",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        _insert_raw(
            store, id="bt_src", url="https://x.com/bt",
            title="synthetic backtest source row",
            source="backtest_run_42_winner",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        _insert_raw(
            store, id="opus_ann", url="https://x.com/opus",
            title="synthetic opus annotation row",
            source="opus_annotation_cycle_3",
            urgency=1, ai_score=8.0, score_source="llm",
        )

        out = store.urgency_label_split_by_source(hours=24)
        # Only the live row is counted — the three synthetics are excluded
        # by ``_LIVE_ONLY_CLAUSE``.
        assert out["total_urgent"] == 1, (
            "synthetic rows leaked into the per-source metric — "
            "backtest isolation broken"
        )
        assert out["total_sources"] == 1
        assert out["by_source"][0]["source"] == "rss"
        # Backtest source names never appear — proves both URL and source
        # branches of the live-only clause are firing.
        names = {r["source"] for r in out["by_source"]}
        assert "backtest_run_42_winner" not in names
        assert "opus_annotation_cycle_3" not in names


class TestUrgencyAndWindow:
    def test_non_urgent_rows_not_counted(self, store):
        """urgency=0 rows are NEVER counted — same predicate as the
        aggregate ``urgency_label_split``. A model-low-relevance call that
        the alert path never saw must not pollute the alert-side metric."""
        _insert_raw(
            store, id="urg", url="https://x.com/1", title="urgent",
            source="rss", urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="not_urg", url="https://x.com/2", title="not urgent",
            source="rss", urgency=0, ai_score=3.0, score_source="llm",
        )
        out = store.urgency_label_split_by_source(hours=24)
        assert out["total_urgent"] == 1

    def test_old_urgent_row_excluded_by_window(self, store):
        """An urgent row older than ``hours`` must not appear — same window
        semantics as the aggregate metric."""
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _insert_raw(
            store, id="old", url="https://x.com/old",
            title="urgent but stale", source="rss", urgency=1,
            ai_score=8.0, score_source="llm", first_seen=old,
        )
        # 6h window excludes the 48h-old row entirely
        out = store.urgency_label_split_by_source(hours=6)
        assert out["total_urgent"] == 0
        assert out["by_source"] == []
