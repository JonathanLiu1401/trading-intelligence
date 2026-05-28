"""Tests for ``ArticleStore.briefing_article_count_trend`` — the briefing
INPUT-pool-size trend method.

Distinct from ``briefing_length_trend`` (which measures OUTPUT text length):
this measures the candidate-article pool the briefing was built FROM
(``briefings.article_count``). A briefing pipeline that's quietly feeding
Opus fewer articles per cycle is a real failure mode that the existing
trend siblings (cadence / overlap / length) cannot catch.

The method is pure-read with no DB mutation — all four load-bearing
invariants intact tautologically.
"""
from __future__ import annotations


def _save_briefing(store, ts: str, text: str, count: int) -> None:
    """Direct INSERT bypassing ``save_briefing`` so tests can set arbitrary
    ``article_count`` values without going through the heartbeat path."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?,?,?)",
            (ts, text, count),
        )
        store.conn.commit()


class TestNoDataBranch:
    def test_empty_briefings_returns_no_data(self, store):
        rep = store.briefing_article_count_trend(last_n=10)
        assert rep["verdict"] == "NO_DATA"
        assert rep["n_briefings"] == 0
        assert rep["counts"] == []
        assert rep["recent_median"] is None
        assert rep["shrink_ratio"] is None

    def test_three_briefings_below_min_split_returns_no_data(self, store):
        """Below 4 briefings the older/newer-half split would be a 1-vs-2
        comparison — too noisy. Method returns NO_DATA."""
        for i in range(3):
            _save_briefing(
                store, f"2026-05-01T0{i}:00:00+00:00", f"briefing {i}", 40
            )
        rep = store.briefing_article_count_trend(last_n=10)
        assert rep["verdict"] == "NO_DATA"
        assert rep["n_briefings"] == 3
        # counts are still returned (helpful for the dashboard even at NO_DATA)
        assert rep["counts"] == [40, 40, 40]
        assert rep["min_count"] == 40
        assert rep["max_count"] == 40


class TestVerdictLadder:
    def test_stable_counts_verdict_stable(self, store):
        """All counts equal → shrink_ratio = 1.0 → STABLE."""
        for i in range(6):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"briefing {i}", 50,
            )
        rep = store.briefing_article_count_trend(last_n=10)
        assert rep["verdict"] == "STABLE"
        assert rep["n_briefings"] == 6
        assert rep["counts"] == [50, 50, 50, 50, 50, 50]
        assert rep["shrink_ratio"] == 1.0
        assert rep["recent_median"] == 50
        assert rep["older_median"] == 50

    def test_shrinking_pool_flagged(self, store):
        """Older 50 → newer 30 = 0.6 ratio (< 0.7 threshold) → SHRINKING.

        Insert older briefings first so id-order matches time-order; the
        method ORDER BY id DESC then reverses to chronological."""
        # Older half (3 rows, median 50)
        for i in range(3):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"older {i}", 50,
            )
        # Newer half (3 rows, median 30)
        for i in range(3):
            _save_briefing(
                store, f"2026-05-02T{i:02d}:00:00+00:00",
                f"newer {i}", 30,
            )
        rep = store.briefing_article_count_trend(last_n=10)
        assert rep["n_briefings"] == 6
        # Chronological: older first, newer last.
        assert rep["counts"] == [50, 50, 50, 30, 30, 30]
        assert rep["older_median"] == 50
        assert rep["recent_median"] == 30
        assert rep["shrink_ratio"] == 0.6
        assert rep["verdict"] == "SHRINKING"

    def test_growing_pool_flagged(self, store):
        """Older 20 → newer 40 = 2.0 ratio (>= 1.3 threshold) → GROWING.
        Surfaced for symmetry; not analyst-actionable on its own."""
        for i in range(3):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"older {i}", 20,
            )
        for i in range(3):
            _save_briefing(
                store, f"2026-05-02T{i:02d}:00:00+00:00",
                f"newer {i}", 40,
            )
        rep = store.briefing_article_count_trend(last_n=10)
        assert rep["verdict"] == "GROWING"
        assert rep["shrink_ratio"] == 2.0

    def test_borderline_22_pct_shrink_is_stable(self, store):
        """30→23.4 ≈ 0.78 ratio — above the 0.70 SHRINKING threshold, so
        STABLE. Pins the threshold against accidental tightening."""
        for i in range(3):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"older {i}", 30,
            )
        for n in (23, 24, 24):  # median 24
            _save_briefing(
                store, f"2026-05-02T0{n - 22}:00:00+00:00",
                f"newer {n}", n,
            )
        rep = store.briefing_article_count_trend(last_n=10)
        # 24 / 30 = 0.8 — above 0.70 SHRINKING cutoff
        assert rep["shrink_ratio"] == 0.8
        assert rep["verdict"] == "STABLE"


class TestEdgeCases:
    def test_older_median_zero_returns_no_data(self, store):
        """If the older half is all zero (the schema's default value for
        article_count) shrink_ratio cannot be computed — NO_DATA, not crash."""
        # Older half: all 0
        for i in range(3):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"older {i}", 0,
            )
        # Newer half: real values
        for i in range(3):
            _save_briefing(
                store, f"2026-05-02T{i:02d}:00:00+00:00",
                f"newer {i}", 40,
            )
        rep = store.briefing_article_count_trend(last_n=10)
        # Method gracefully returns NO_DATA on divide-by-zero
        assert rep["verdict"] == "NO_DATA"
        assert rep["shrink_ratio"] is None
        # But the raw stats are still populated for dashboard rendering
        assert rep["older_median"] == 0
        assert rep["recent_median"] == 40

    def test_last_n_caps_window(self, store):
        """Only the last_n most-recent rows feed the trend — older rows are
        ignored. Insert 8 rows but request last_n=4 → only newest 4 used."""
        for i in range(8):
            _save_briefing(
                store, f"2026-05-01T{i:02d}:00:00+00:00",
                f"row {i}", (i + 1) * 10,
            )
        # Newest 4 rows have article_count in {50, 60, 70, 80}
        rep = store.briefing_article_count_trend(last_n=4)
        assert rep["n_briefings"] == 4
        assert rep["counts"] == [50, 60, 70, 80]
        # older half = [50, 60] median 60; newer half = [70, 80] median 80
        assert rep["older_median"] == 60
        assert rep["recent_median"] == 80
        # 80/60 ≈ 1.333 — GROWING
        assert rep["verdict"] == "GROWING"

    def test_does_not_mutate_articles(self, store):
        """Read-only invariant — calling the method must not write to articles
        or change any score/score_source/urgency. Pinned because the file's
        load-bearing invariants discipline says every read-only must be
        provably non-mutating."""
        # Insert one article and one briefing
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                "urgency, first_seen, cycle, ml_score, score_source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("a", "https://x.com/1", "title", "rss", "", 1.0, 5.0,
                 1, "2026-05-15T00:00:00+00:00", 0, None, "llm"),
            )
            store.conn.commit()
        for i in range(5):
            _save_briefing(
                store, f"2026-05-01T0{i}:00:00+00:00", f"briefing {i}", 25,
            )

        # Snapshot the article row before the trend call.
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()

        store.briefing_article_count_trend(last_n=10)

        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()
        assert before == after, "trend method must not mutate articles"


class TestLiveScenario:
    def test_chronic_under_feeding_flagged(self, store):
        """Live-evidence shape: briefing cadence is HEALTHY and text length is
        STABLE, but the input pool dropped from ~50 articles/cycle (older
        half) to ~25 (newer half) — exactly the failure mode this trend is
        the only surface for. shrink_ratio 0.5 → SHRINKING."""
        # Older briefings: pool of ~50
        for c in (50, 52, 48, 50):
            _save_briefing(
                store, f"2026-05-01T{c % 24:02d}:00:00+00:00",
                "old briefing — content body", c,
            )
        # Newer briefings: pool of ~25
        for c in (25, 24, 26, 25):
            _save_briefing(
                store, f"2026-05-02T{c % 24:02d}:00:00+00:00",
                "new briefing — content body", c,
            )
        rep = store.briefing_article_count_trend(last_n=20)
        assert rep["verdict"] == "SHRINKING"
        assert rep["recent_median"] == 25
        assert rep["older_median"] == 50
        assert rep["shrink_ratio"] == 0.5
        # Sanity: the full window covers all 8 rows
        assert rep["n_briefings"] == 8
