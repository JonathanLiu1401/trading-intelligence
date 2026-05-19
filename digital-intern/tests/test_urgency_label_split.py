"""``ArticleStore.urgency_label_split`` — calibration metric tests.

The live 2026-05-19 evidence (every urgency>=1 row alerted in the last 6h had
``ai_score=0`` / ``score_source='ml'``) was the case the analyst persona was
blind to in aggregate. ``urgency_label_split`` exposes the per-source split
so a near-zero ``llm_fraction`` can be surfaced before the analyst notices it
manually. Pin the contract end-to-end:

  * Sums equal the input cardinality (no rows lost or double-counted).
  * The four canonical buckets always exist in the return shape.
  * ``llm_fraction`` = (llm + briefing_boost) / total, 0.0 on empty.
  * Backtest-isolation invariant: synthetic backtest:// / backtest_* /
    opus_annotation* rows are NEVER counted (this metric is consumed by a
    live dashboard; an injection burst must never fake a vetted/unvetted
    figure).
  * Non-urgent rows (urgency=0) are NEVER counted.
  * The ``hours`` window is respected (an old row outside the window is
    excluded even if urgent).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _iso(now: datetime) -> str:
    return now.isoformat()


def _recent(minutes_ago: int = 5) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


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


class TestShapeAndDefaults:
    def test_empty_store_returns_zero_buckets(self, store):
        out = store.urgency_label_split(hours=24)
        assert out["window_h"] == 24
        assert out["total"] == 0
        assert out["llm_fraction"] == 0.0
        assert out["by_source"] == {
            "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
        }

    def test_buckets_always_present_even_when_only_one_used(self, store):
        _insert_raw(
            store, id="a", url="https://x.com/1", title="title long enough",
            source="rss", urgency=1, ai_score=8.0, score_source="llm",
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 1
        assert out["by_source"]["llm"] == 1
        # absent buckets must still surface as 0 (dashboard-stable shape)
        assert out["by_source"]["ml"] == 0
        assert out["by_source"]["briefing_boost"] == 0
        assert out["by_source"]["null"] == 0


class TestPerSourceCounts:
    def test_mixed_sources_count_correctly(self, store):
        # 3 LLM-vetted (score_source='llm', ai_score > 0)
        for i in range(3):
            _insert_raw(
                store, id=f"l{i}", url=f"https://reuters.com/{i}",
                title=f"LLM-vetted urgent number {i}",
                source="rss", urgency=1, ai_score=9.0, score_source="llm",
            )
        # 5 ML-only (score_source='ml', ai_score=0, ml_score>0)
        for i in range(5):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        # 1 briefing_boost (Opus curation nudge)
        _insert_raw(
            store, id="bb1", url="https://x.com/bb1",
            title="Opus-curated briefing boost", source="rss",
            urgency=1, ai_score=4.5, score_source="briefing_boost",
        )
        # 2 legacy NULL score_source (pre-migration)
        for i in range(2):
            _insert_raw(
                store, id=f"n{i}", url=f"https://x.com/n{i}",
                title=f"Legacy null source {i}",
                source="rss", urgency=1, ai_score=8.0, score_source=None,
            )

        out = store.urgency_label_split(hours=24)
        assert out["total"] == 11
        assert out["by_source"] == {
            "llm": 3, "ml": 5, "briefing_boost": 1, "null": 2,
        }
        # llm_fraction is (llm + briefing_boost) / total — the analyst-facing
        # "fraction of urgent calls a real LLM head signed off on".
        assert out["llm_fraction"] == pytest.approx((3 + 1) / 11, abs=1e-4)

    def test_alerted_state_urgency_two_still_counted(self, store):
        """Both urgency=1 (queued) and urgency=2 (already alerted) count —
        the metric is "of urgent CALLS in the window", which includes ones
        that already fired. The alerter clears the queue by setting
        urgency=2; if we excluded those we'd silently undercount the
        delivered/seen calls."""
        _insert_raw(
            store, id="q1", url="https://x.com/q1",
            title="Queued urgent here", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="a1", url="https://x.com/a1",
            title="Already alerted urgent", source="rss",
            urgency=2, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 2
        assert out["by_source"]["llm"] == 2


class TestExclusions:
    def test_non_urgent_rows_excluded(self, store):
        _insert_raw(
            store, id="u1", url="https://x.com/u1",
            title="Urgent normal article", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="n1", url="https://x.com/n1",
            title="Non-urgent normal", source="rss",
            urgency=0, ai_score=4.0, score_source="llm",
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 1
        assert out["by_source"]["llm"] == 1

    def test_backtest_urls_never_count(self, store):
        """Critical invariant — the metric must never be inflated by
        backtest injection rows, which carry legitimate fractional ai_score
        labels and a NULL score_source that would otherwise add to the
        ``null`` bucket and silently mask the live calibration figure."""
        _insert_raw(
            store, id="live", url="https://reuters.com/x",
            title="Real urgent live news", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="bt1", url="backtest://run_1/2026-01-01/BUY/MU",
            title="Synthetic backtest row", source="backtest_run_1",
            urgency=1, ai_score=9.0, score_source=None,
        )
        _insert_raw(
            store, id="opus1", url="https://x.com/opus1",
            title="Opus annotation", source="opus_annotation_cycle_3",
            urgency=1, ai_score=5.0, score_source=None,
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 1, (
            f"backtest rows leaked into the metric: by_source={out['by_source']}"
        )
        assert out["by_source"]["llm"] == 1
        assert out["by_source"]["null"] == 0

    def test_hours_window_filters_old_rows(self, store):
        """A genuinely stale urgent row (>= ``hours`` ago) must NOT count —
        the live calibration is a recent-window measurement, not a lifetime
        total. A user asking "is the LLM head alive RIGHT NOW?" must not
        be answered with rows from days ago."""
        old_first_seen = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).isoformat()
        _insert_raw(
            store, id="stale1", url="https://x.com/old",
            title="48h-old urgent row", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
            first_seen=old_first_seen,
        )
        _insert_raw(
            store, id="fresh1", url="https://x.com/new",
            title="Fresh urgent row title", source="rss",
            urgency=1, ai_score=0.0, score_source="ml", ml_score=9.0,
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 1
        assert out["by_source"]["ml"] == 1
        assert out["by_source"]["llm"] == 0


class TestLLMFractionCalculation:
    def test_pure_ml_window_yields_zero_fraction(self, store):
        """The live evidence case: every urgent row in the window is ML-only.
        ``llm_fraction`` MUST be 0.0 (not None, not undefined) so a dashboard
        check (``if split['llm_fraction'] < 0.3: warn``) renders correctly."""
        for i in range(4):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"ML-only urgent {i} headline",
                source="rss", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 4
        assert out["llm_fraction"] == 0.0

    def test_pure_llm_window_yields_full_fraction(self, store):
        for i in range(3):
            _insert_raw(
                store, id=f"l{i}", url=f"https://x.com/l{i}",
                title=f"All-LLM urgent {i} headline",
                source="rss", urgency=1, ai_score=9.0, score_source="llm",
            )
        out = store.urgency_label_split(hours=24)
        assert out["llm_fraction"] == 1.0

    def test_briefing_boost_counts_as_vetted(self, store):
        """``briefing_boost`` is an Opus curation nudge — also a real LLM
        signal, so it MUST count toward ``llm_fraction`` alongside ``llm``.
        If a future refactor splits these apart this test fails first."""
        _insert_raw(
            store, id="bb1", url="https://x.com/bb1",
            title="Opus briefing-boosted item", source="rss",
            urgency=1, ai_score=4.5, score_source="briefing_boost",
        )
        _insert_raw(
            store, id="m1", url="https://x.com/m1",
            title="ML-only counterpart row", source="rss",
            urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
        )
        out = store.urgency_label_split(hours=24)
        assert out["total"] == 2
        # 1 vetted (briefing_boost) out of 2 → 0.5
        assert out["llm_fraction"] == pytest.approx(0.5, abs=1e-4)
