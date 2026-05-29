"""Tests for ``ArticleStore.urgent_count_per_briefing_window_trend`` — the
URGENT-flow trend sibling to ``briefing_article_count_trend``.

Each "window" is the gap between two consecutive briefings; the method
counts ``urgency>=1`` rows whose ``first_seen`` lands in that interval. The
verdict (STABLE / SURGING / QUIETING / NO_DATA) mirrors the article-count
trend's ladder but oriented around the urgent-flow analyst signal.

The method is pure-read with no DB mutation — all four load-bearing
invariants intact tautologically. The tests assert specific verdicts and
specific count values, not just "no crash".
"""
from __future__ import annotations

import zlib


def _save_briefing(store, ts: str, text: str = "x") -> None:
    """Direct INSERT bypassing ``save_briefing`` so tests can set explicit
    timestamps without going through the heartbeat path."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?,?,?)",
            (ts, text, 50),
        )
        store.conn.commit()


def _insert_urgent_row(
    store, aid: str, first_seen: str, urgency: int = 1,
    url: str = "", source: str = "rss",
) -> None:
    """Insert an article row at a chosen ``first_seen`` for the urgent-flow
    window test. The schema requires ``url`` / ``title`` / ``first_seen`` —
    everything else has defaults. Goes through the writer-lock so concurrent
    fixtures cannot race."""
    if not url:
        url = f"https://example.com/{aid}"
    title = f"Urgent headline {aid}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, first_seen, urgency, ai_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (aid, url, title, source, first_seen, urgency, 9.0, "llm"),
        )
        store.conn.commit()


class TestNoDataBranches:
    def test_empty_briefings_returns_no_data(self, store):
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "NO_DATA"
        assert rep["n_windows"] == 0
        assert rep["counts"] == []
        assert rep["windows"] == []
        assert rep["median_count"] is None
        assert rep["surge_ratio"] is None

    def test_three_windows_below_min_returns_no_data(self, store):
        """Need at least 4 windows (5 briefings) to split into older/newer
        halves of ≥2 each. Below that any verdict would be noise."""
        # 4 briefings → 3 windows → below the 4-window minimum.
        for i in range(4):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "NO_DATA"
        assert rep["n_windows"] == 3
        # NO_DATA branch returns no windows / no counts (matches the sibling
        # NO_DATA contract that abstains rather than half-emitting).
        assert rep["counts"] == []

    def test_older_median_zero_returns_no_data(self, store):
        """Older half all-zero (a quiet wire) → cannot compute a surge_ratio.
        Method returns NO_DATA but DOES populate counts / windows for the
        dashboard to see the underlying numbers."""
        # 5 briefings at hours 0..4 = 4 windows [0,1), [1,2), [2,3), [3,4).
        # Older 2 windows (0..1) have 0 urgent, newer 2 windows (2..3) have
        # 3 urgent each. The 3-3 newer median is real, but with
        # older_median=0 the ratio is undefined.
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Window 2 = [T02:00, T03:00): 3 urgent rows
        _insert_urgent_row(store, "n1", "2026-05-01T02:30:00+00:00")
        _insert_urgent_row(store, "n2", "2026-05-01T02:31:00+00:00")
        _insert_urgent_row(store, "n3", "2026-05-01T02:32:00+00:00")
        # Window 3 = [T03:00, T04:00): 3 urgent rows
        _insert_urgent_row(store, "n4", "2026-05-01T03:30:00+00:00")
        _insert_urgent_row(store, "n5", "2026-05-01T03:31:00+00:00")
        _insert_urgent_row(store, "n6", "2026-05-01T03:32:00+00:00")
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "NO_DATA"
        assert rep["n_windows"] == 4
        # older_median == 0 specifically — that's the discriminator.
        assert rep["older_median"] == 0
        assert rep["recent_median"] == 3
        assert rep["surge_ratio"] is None
        # Counts ARE populated in this branch (matches briefing_*_trend
        # divide-by-zero defense — partial data is still useful).
        assert rep["counts"] == [0, 0, 3, 3]


class TestVerdictLadder:
    def test_stable_counts_verdict_stable(self, store):
        """Uniform counts → surge_ratio = 1.0 → STABLE."""
        # 5 briefings = 4 windows. 2 urgent rows in each window.
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        for i, hour in enumerate([0, 1, 2, 3]):
            _insert_urgent_row(
                store, f"w{i}a", f"2026-05-01T{hour:02d}:15:00+00:00"
            )
            _insert_urgent_row(
                store, f"w{i}b", f"2026-05-01T{hour:02d}:30:00+00:00"
            )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "STABLE"
        assert rep["counts"] == [2, 2, 2, 2]
        assert rep["surge_ratio"] == 1.0
        assert rep["older_median"] == 2
        assert rep["recent_median"] == 2

    def test_surging_recent_double_baseline_verdict_surging(self, store):
        """Recent half 2× older half → surge_ratio = 2.0 → SURGING."""
        # 5 briefings = 4 windows. Older windows have 2 urgent each, newer
        # windows have 4 urgent each. recent_median/older_median = 2.0.
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Older 2 windows: 2 urgent each
        for w_idx, hour in enumerate([0, 1]):
            _insert_urgent_row(
                store, f"o{w_idx}a", f"2026-05-01T{hour:02d}:10:00+00:00"
            )
            _insert_urgent_row(
                store, f"o{w_idx}b", f"2026-05-01T{hour:02d}:20:00+00:00"
            )
        # Newer 2 windows: 4 urgent each
        for w_idx, hour in enumerate([2, 3]):
            for k in range(4):
                _insert_urgent_row(
                    store, f"n{w_idx}{k}",
                    f"2026-05-01T{hour:02d}:{10 + k * 5:02d}:00+00:00",
                )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "SURGING"
        assert rep["counts"] == [2, 2, 4, 4]
        assert rep["surge_ratio"] == 2.0
        assert rep["recent_median"] == 4
        assert rep["older_median"] == 2

    def test_quieting_recent_half_baseline_verdict_quieting(self, store):
        """Recent half 0.5× older half → surge_ratio = 0.5 → QUIETING."""
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Older 2 windows: 4 urgent each
        for w_idx, hour in enumerate([0, 1]):
            for k in range(4):
                _insert_urgent_row(
                    store, f"o{w_idx}{k}",
                    f"2026-05-01T{hour:02d}:{10 + k * 5:02d}:00+00:00",
                )
        # Newer 2 windows: 2 urgent each
        for w_idx, hour in enumerate([2, 3]):
            _insert_urgent_row(
                store, f"n{w_idx}a", f"2026-05-01T{hour:02d}:10:00+00:00"
            )
            _insert_urgent_row(
                store, f"n{w_idx}b", f"2026-05-01T{hour:02d}:20:00+00:00"
            )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["verdict"] == "QUIETING"
        assert rep["counts"] == [4, 4, 2, 2]
        assert rep["surge_ratio"] == 0.5

    def test_borderline_surge_below_threshold_stays_stable(self, store):
        """A 22% increase (recent=11/older=9) is below the 30% SURGING bar →
        STABLE. Pin the boundary so a future threshold change is intentional."""
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Older windows: 9 each. Newer windows: 11 each. ratio ≈ 1.222.
        for w_idx, hour in enumerate([0, 1]):
            for k in range(9):
                _insert_urgent_row(
                    store, f"o{w_idx}{k:02d}",
                    f"2026-05-01T{hour:02d}:{10 + k * 5:02d}:00+00:00",
                )
        for w_idx, hour in enumerate([2, 3]):
            for k in range(11):
                _insert_urgent_row(
                    store, f"n{w_idx}{k:02d}",
                    f"2026-05-01T{hour:02d}:{10 + k * 4:02d}:00+00:00",
                )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        # 11/9 ≈ 1.222 — below the 1.30 SURGING bar.
        assert rep["verdict"] == "STABLE"
        assert rep["recent_median"] == 11
        assert rep["older_median"] == 9


class TestBacktestIsolation:
    """Synthetic backtest/opus rows must never inflate the per-window urgent
    count — mirrors the canonical _LIVE_ONLY_CLAUSE invariant the existing
    trend siblings carry."""

    def test_backtest_url_excluded_from_count(self, store):
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Two real urgent rows in window 0
        _insert_urgent_row(store, "r1", "2026-05-01T00:30:00+00:00")
        _insert_urgent_row(store, "r2", "2026-05-01T00:31:00+00:00")
        # Synthetic backtest:// URL — must NOT count
        _insert_urgent_row(
            store, "bt1", "2026-05-01T00:32:00+00:00",
            url="backtest://run_42/2026-05-01/buy/NVDA",
        )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        # 4 windows, only first contains REAL urgents (2) — the backtest
        # row is correctly excluded.
        assert rep["windows"][0]["urgent_count"] == 2

    def test_backtest_source_tag_excluded(self, store):
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        _insert_urgent_row(store, "r1", "2026-05-01T00:30:00+00:00")
        # source LIKE 'backtest_%' — must NOT count
        _insert_urgent_row(
            store, "bt1", "2026-05-01T00:32:00+00:00",
            source="backtest_run_42_winner",
        )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["windows"][0]["urgent_count"] == 1

    def test_opus_annotation_source_excluded(self, store):
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        _insert_urgent_row(store, "r1", "2026-05-01T00:30:00+00:00")
        # source LIKE 'opus_annotation%' — must NOT count
        _insert_urgent_row(
            store, "op1", "2026-05-01T00:32:00+00:00",
            source="opus_annotation_cycle_3",
        )
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        assert rep["windows"][0]["urgent_count"] == 1


class TestWindowBoundaries:
    def test_first_seen_at_boundaries_half_open_interval(self, store):
        """Window is [older_ts, newer_ts) — first_seen == newer_ts must go to
        the NEXT window, not the current one."""
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        # Right at window-2's start (== window-1's end). Per half-open
        # interval semantics, this lands in window 2 (the newer one), NOT
        # window 1.
        _insert_urgent_row(store, "boundary", "2026-05-01T02:00:00+00:00")
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        # 4 windows. The boundary row at T02:00 lands in window index 2
        # (which is [T02:00, T03:00)).
        assert rep["counts"] == [0, 0, 1, 0]

    def test_window_duration_h_populated(self, store):
        """Each window dict carries the duration in hours (newer_ts - older_ts)
        — a stable analyst-visible value for the dashboard."""
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        rep = store.urgent_count_per_briefing_window_trend(last_n=10)
        # Each consecutive pair is exactly 1h apart in the fixture.
        for w in rep["windows"]:
            assert w["duration_h"] == 1.0


class TestLastNWindowCap:
    def test_last_n_bounds_returned_window_count(self, store):
        """last_n=4 with 10 briefings → exactly 4 windows returned (the
        last_n+1 = 5 newest briefings yield 4 inter-briefing windows)."""
        for i in range(10):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        rep = store.urgent_count_per_briefing_window_trend(last_n=4)
        assert rep["n_windows"] == 4
        assert len(rep["counts"]) == 4
        # The 4 returned windows are the most-recent — newest pair is
        # T08:00 → T09:00.
        assert rep["windows"][-1]["newer_ts"].startswith("2026-05-01T09:00")


class TestReadOnlyInvariant:
    """Pure read primitive — must not mutate ai_score / ml_score /
    score_source / urgency on any row. Pins the load-bearing invariant by
    construction."""

    def test_does_not_mutate_articles(self, store):
        for i in range(5):
            _save_briefing(store, f"2026-05-01T{i:02d}:00:00+00:00")
        _insert_urgent_row(store, "r1", "2026-05-01T00:30:00+00:00")
        # Snapshot the row BEFORE
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?",
            ("r1",),
        ).fetchone()
        _ = store.urgent_count_per_briefing_window_trend(last_n=10)
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?",
            ("r1",),
        ).fetchone()
        assert before == after
