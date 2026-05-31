"""``ArticleStore.urgent_source_breakdown`` — per-source LLM-vs-ML split.

Per-source decomposition of ``urgency_label_split``. Pins the contract that
matters to the news analyst persona: WHICH source tags are dragging the
alerted-set ``llm_fraction`` down — the question the aggregate metric is
silent on.

Live evidence (2026-05-31 24h pull, urgency=2 rows): the recently alerted
set was dominated by ml-only ``GN: SP500`` / ``GN: Nvidia`` / ``hackernews`` /
``stocktwits`` / ``GDELT/ibtimes.com.au`` / ``AlphaVantage/Seeking Alpha`` /
``scraped/www.chinadaily.com.cn`` / ``market_valuation``. The aggregate
``urgency_label_split`` shows the top-line llm_fraction; this surfaces the
per-source story so the operator can down-rate the noisy ones.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(
    store, *, id, url, title, source, urgency=2, ai_score=0.0,
    ml_score=None, score_source=None, kw_score=1.0, first_seen=None,
):
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


# ── invariant: empty / shape ─────────────────────────────────────────────────
class TestShapeAndDefaults:
    def test_empty_store_returns_zero_shape(self, store):
        out = store.urgent_source_breakdown(hours=24)
        assert out["window_h"] == 24
        assert out["total"] == 0
        assert out["llm_vetted_total"] == 0
        assert out["ml_only_total"] == 0
        assert out["llm_fraction"] == 0.0
        assert out["by_source"] == []
        assert out["worst_offender"] is None

    def test_window_h_min_clamped_to_one(self, store):
        """Defensive: a zero/negative hours param is clamped to 1, not blindly
        passed through (a 0h window would return everything-since-epoch under
        SQL semantics — surprising operator footgun)."""
        out = store.urgent_source_breakdown(hours=0)
        assert out["window_h"] == 1
        out2 = store.urgent_source_breakdown(hours=-50)
        assert out2["window_h"] == 1

    def test_top_n_min_clamped_to_one(self, store):
        """top_n=0 would yield an empty by_source list even with real data —
        confusing. Clamp to 1."""
        _insert_raw(store, id="a", url="https://x.com/a", title="t",
                    source="rss", score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24, top_n=0)
        assert len(out["by_source"]) == 1


# ── invariant: per-source split is correct ──────────────────────────────────
class TestPerSourceSplit:
    def test_llm_vetted_counted_correctly(self, store):
        """``llm`` and ``briefing_boost`` are both ground-truth tags (the
        trainer's STRONG_LABEL_WHERE pulls both). Both must contribute to
        ``llm_vetted`` for the source."""
        _insert_raw(store, id="a", url="https://x.com/a",
                    title="llm tagged", source="rss",
                    score_source="llm", ai_score=9.0)
        _insert_raw(store, id="b", url="https://x.com/b",
                    title="briefing tagged", source="rss",
                    score_source="briefing_boost", ai_score=4.5)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 2
        assert out["llm_vetted_total"] == 2
        assert out["ml_only_total"] == 0
        assert out["llm_fraction"] == 1.0
        assert len(out["by_source"]) == 1
        rss = out["by_source"][0]
        assert rss["source"] == "rss"
        assert rss["total"] == 2
        assert rss["llm_vetted"] == 2
        assert rss["ml_only"] == 0
        assert rss["llm_fraction"] == 1.0

    def test_ml_only_counted_correctly(self, store):
        """``score_source='ml'`` is the explicit ml tag — counts toward
        ``ml_only`` regardless of ai_score."""
        _insert_raw(store, id="a", url="https://x.com/a",
                    title="ml tagged", source="GN: Nvidia",
                    score_source="ml", ai_score=0.0, ml_score=9.5)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 1
        assert out["ml_only_total"] == 1
        assert out["llm_vetted_total"] == 0
        assert out["llm_fraction"] == 0.0
        gn = out["by_source"][0]
        assert gn["source"] == "GN: Nvidia"
        assert gn["ml_only"] == 1
        assert gn["llm_vetted"] == 0

    def test_legacy_null_with_ai_score_zero_is_ml_only(self, store):
        """The score_source migration was after-the-fact: some pre-migration
        rows have score_source=NULL. If ai_score=0 on such a row, no LLM ever
        labeled it — the urgency came from a model call (invariant #2: ml
        outputs go to ml_score and never ai_score). Must count as ml_only."""
        _insert_raw(store, id="a", url="https://x.com/a",
                    title="legacy ml row", source="GDELT/foo.com",
                    score_source=None, ai_score=0.0, ml_score=8.5)
        out = store.urgent_source_breakdown(hours=24)
        assert out["ml_only_total"] == 1, (
            "legacy NULL score_source with ai_score=0 must count as ml_only — "
            "invariant #2 says only ml writes to ml_score, never ai_score"
        )

    def test_legacy_null_with_positive_ai_score_not_double_counted(self, store):
        """Pre-migration LLM-labeled rows carry score_source=NULL but with
        positive ai_score (Sonnet returned an int). These count toward
        ``total`` but should NOT inflate either tier — they are uncategorised
        ground-truth that may or may not be LLM-vetted."""
        _insert_raw(store, id="a", url="https://x.com/a",
                    title="legacy integer-labeled", source="rss",
                    score_source=None, ai_score=7.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 1
        # Defensive: should NOT inflate ml_only (would falsely worsen the
        # source's reputation) NOR llm_vetted (we cannot prove that without
        # the explicit tag).
        assert out["ml_only_total"] == 0
        assert out["llm_vetted_total"] == 0
        assert out["llm_fraction"] == 0.0

    def test_multiple_sources_sorted_by_total(self, store):
        """``by_source`` is sorted by total DESC so the dashboard can show
        the loudest channels first."""
        for i in range(3):
            _insert_raw(store, id=f"big{i}", url=f"https://x.com/big{i}",
                        title=f"t{i}", source="loud_source",
                        score_source="ml", ai_score=0.0, ml_score=9.0)
        _insert_raw(store, id="small1", url="https://x.com/s1",
                    title="t", source="quiet_source",
                    score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert [r["source"] for r in out["by_source"]] == [
            "loud_source", "quiet_source"
        ]
        assert out["by_source"][0]["total"] == 3
        assert out["by_source"][1]["total"] == 1


# ── invariant: backtest isolation ───────────────────────────────────────────
class TestBacktestIsolation:
    def test_backtest_url_rows_excluded(self, store):
        """The most load-bearing invariant: backtest:// rows must NOT inflate
        the operator-facing metric. A heavy backtest injection burst could
        otherwise drive llm_fraction to a false floor/ceiling."""
        _insert_raw(store, id="live", url="https://reuters.com/x",
                    title="real", source="rss",
                    score_source="llm", ai_score=9.0)
        _insert_raw(store, id="bt1", url="backtest://run_1/d/BUY/MU",
                    title="synthetic", source="backtest_run_1_winner",
                    score_source=None, ai_score=5.0)
        _insert_raw(store, id="bt2", url="https://x.com/opus",
                    title="opus", source="opus_annotation_cycle_5",
                    score_source=None, ai_score=2.5)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 1, (
            "synthetic backtest/opus rows must NEVER count — invariant #1"
        )
        assert [r["source"] for r in out["by_source"]] == ["rss"]


# ── invariant: urgency=0 rows excluded ──────────────────────────────────────
class TestUrgencyFilter:
    def test_non_alerted_rows_excluded(self, store):
        """The point of THIS metric is the alerted set (urgency>=2 — actually
        pushed). urgency=0 noise and urgency=1 pending must not inflate."""
        # urgency=0 (noise) — must NOT count
        _insert_raw(store, id="n", url="https://x.com/n", title="n",
                    source="rss", urgency=0, score_source="llm", ai_score=3.0)
        # urgency=1 (pending, not yet alerted) — must NOT count
        _insert_raw(store, id="p", url="https://x.com/p", title="p",
                    source="rss", urgency=1, score_source="llm", ai_score=9.0)
        # urgency=2 (actually alerted) — counts
        _insert_raw(store, id="a", url="https://x.com/a", title="a",
                    source="rss", urgency=2, score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 1
        assert out["by_source"][0]["total"] == 1


# ── invariant: window respected ─────────────────────────────────────────────
class TestWindow:
    def test_old_rows_excluded(self, store):
        """A row outside the hours window must not be counted, even if alerted.
        Otherwise the metric drifts with the 90-day retention window."""
        old_seen = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _insert_raw(store, id="old", url="https://x.com/old", title="old",
                    source="rss", score_source="llm", ai_score=9.0,
                    first_seen=old_seen)
        _insert_raw(store, id="new", url="https://x.com/new", title="new",
                    source="rss", score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["total"] == 1


# ── invariant: worst_offender semantics ─────────────────────────────────────
class TestWorstOffender:
    def test_worst_offender_picks_highest_ml_only(self, store):
        """The source with the highest ml_only count is the analyst-actionable
        worst offender — the one to gate or down-rate first."""
        # quiet_source: 1 ml_only — total=1, below the >=2 threshold
        _insert_raw(store, id="q", url="https://x.com/q", title="q",
                    source="quiet_source", score_source="ml",
                    ai_score=0.0, ml_score=9.0)
        # mixed_source: 2 llm + 4 ml_only = 6 total, 4 ml_only
        for i in range(2):
            _insert_raw(store, id=f"ml{i}", url=f"https://x.com/ml{i}",
                        title="t", source="mixed_source",
                        score_source="llm", ai_score=9.0)
        for i in range(4):
            _insert_raw(store, id=f"mo{i}", url=f"https://x.com/mo{i}",
                        title="t", source="mixed_source",
                        score_source="ml", ai_score=0.0, ml_score=8.5)
        # vetted_source: 3 llm — should not be worst
        for i in range(3):
            _insert_raw(store, id=f"v{i}", url=f"https://x.com/v{i}",
                        title="t", source="vetted_source",
                        score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["worst_offender"] is not None
        assert out["worst_offender"]["source"] == "mixed_source"
        assert out["worst_offender"]["ml_only"] == 4

    def test_worst_offender_requires_total_min_two(self, store):
        """A lone ml_only=1 row from a one-off unknown publisher should NOT
        be flagged as worst — that's noise, not a pattern. The min-sample
        floor mirrors the briefing_health 60%-floor defensive discipline."""
        _insert_raw(store, id="lone", url="https://x.com/lone", title="t",
                    source="rare_publisher", score_source="ml",
                    ai_score=0.0, ml_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["worst_offender"] is None, (
            "single ml_only row from an unknown publisher must not surface "
            "as worst_offender — total>=2 floor required"
        )

    def test_worst_offender_none_when_all_vetted(self, store):
        """A fully-vetted alerted set has no offender to surface."""
        for i in range(5):
            _insert_raw(store, id=f"v{i}", url=f"https://x.com/v{i}",
                        title="t", source="rss",
                        score_source="llm", ai_score=9.0)
        out = store.urgent_source_breakdown(hours=24)
        assert out["worst_offender"] is None


# ── invariant: aggregate sums ───────────────────────────────────────────────
class TestAggregateSums:
    def test_per_source_sums_to_total(self, store):
        """``total`` MUST equal the sum of by_source totals — otherwise rows
        are silently lost or double-counted (a class of subtle DB-aggregation
        bug that's invisible until the operator notices the discrepancy)."""
        srcs = ["rss", "GN: Nvidia", "scraped/cnbc.com", "stocktwits"]
        for i, s in enumerate(srcs):
            for j in range(i + 1):
                _insert_raw(store, id=f"{i}_{j}", url=f"https://x.com/{i}_{j}",
                            title="t", source=s,
                            score_source="ml" if i % 2 else "llm",
                            ai_score=0.0 if i % 2 else 9.0,
                            ml_score=8.0 if i % 2 else None)
        out = store.urgent_source_breakdown(hours=24, top_n=20)
        total_from_breakdown = sum(r["total"] for r in out["by_source"])
        assert total_from_breakdown == out["total"]
        # And per-source tiers sum to per-source total (no row in two tiers).
        for r in out["by_source"]:
            assert r["llm_vetted"] + r["ml_only"] <= r["total"]
            # Equality holds when no legacy-null-with-pos-ai-score is present.
            assert r["llm_vetted"] + r["ml_only"] == r["total"]

    def test_top_n_caps_output_but_total_includes_all(self, store):
        """top_n only caps the by_source list; ``total`` and the *_total fields
        still cover every source. A long tail of one-offs must not corrupt
        the top-line metric."""
        # 5 sources, top_n=2 should still report total=5
        for i in range(5):
            _insert_raw(store, id=f"s{i}", url=f"https://x.com/s{i}", title="t",
                        source=f"src_{i}", score_source="ml",
                        ai_score=0.0, ml_score=8.0)
        out = store.urgent_source_breakdown(hours=24, top_n=2)
        assert len(out["by_source"]) == 2
        assert out["total"] == 5
        assert out["ml_only_total"] == 5
