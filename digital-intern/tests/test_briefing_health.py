"""Tests for ``ArticleStore.briefing_health`` — pipeline health snapshot
of the 5h Opus heartbeat-briefing path.

Why these tests exist (news-analyst lens): the briefing is the analyst's
primary synthesised intelligence product. A silent failure of the briefing
path (Opus quota exhausted, Claude CLI auth lapsed, heartbeat worker wedged)
leaves the dashboard happily serving a stale digest and the analyst doesn't
realise the digest is stale — exactly the "silent dark" failure mode the
analyst persona fears most. ``briefing_health`` is the standing primitive
that distinguishes HEALTHY / STALE / DEAD / NO_DATA states, and these tests
pin the verdict boundaries so a future tuning never quietly flips a STALE
window to HEALTHY (or vice-versa) without test breakage.

Coverage:
  * NO_DATA verdict on an empty briefings table — distinct from DEAD so
    a just-started daemon doesn't fire false outage signals.
  * HEALTHY verdict on a normal cadence (last < 6h ago, count meets floor).
  * STALE verdict on a single 6-12h-old briefing.
  * STALE verdict on a count-below-floor window (single briefing in 24h).
  * DEAD verdict on a briefing > 12h ago.
  * Read-only contract: the method does NOT touch ai_score / ml_score /
    score_source / urgency on any article row.
  * Backtest isolation: the briefings table is unrelated to the articles
    table, but the method must still ignore any synthetic articles
    sitting alongside (defense-in-depth — a future broken refactor that
    incorrectly cross-joins must surface here).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _save_briefing(store, hours_ago: float, text: str = "briefing body") -> None:
    """Insert a briefing row dated ``hours_ago`` hours before now.

    save_briefing() uses an INSERT, but we want explicit control of the ``ts``
    column to test age-bucketed verdicts deterministically, so we write the row
    directly through the store's connection (the same path save_briefing uses).
    """
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
            (_iso_ago(hours_ago), text, 10),
        )
        store.conn.commit()


class TestVerdictLadder:
    def test_no_data_when_briefings_table_empty(self, store):
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "NO_DATA"
        assert h["last_briefing_age_h"] is None
        assert h["count_in_window"] == 0
        # Expected = window_h / 5h cadence = 24/5 = 4.8
        assert h["expected_in_window"] == 4.8
        assert h["window_h"] == 24

    def test_healthy_on_normal_cadence(self, store):
        """4 briefings spread across the last 20h with the newest 2h ago —
        the canonical healthy production state."""
        for hours_ago in (2.0, 7.0, 12.0, 17.0):
            _save_briefing(store, hours_ago=hours_ago)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "HEALTHY"
        assert h["count_in_window"] == 4
        assert h["last_briefing_age_h"] == 2.0

    def test_stale_when_last_briefing_8h_ago(self, store):
        """A single 8h-old briefing — last > 6h triggers STALE even if the
        count would otherwise satisfy the floor."""
        _save_briefing(store, hours_ago=8.0)
        # Add more briefings far enough back to bump count without resetting age.
        for hours_ago in (10.0, 14.0, 18.0):
            _save_briefing(store, hours_ago=hours_ago)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "STALE"
        # Newest is 8h old.
        assert 7.9 <= h["last_briefing_age_h"] <= 8.1

    def test_stale_when_count_below_floor(self, store):
        """One recent briefing in 24h. Newest is fresh (2h ago) so age alone
        is HEALTHY, but the count (1) is below the 60%-of-expected floor
        (max(1, int(0.6 * 4.8)) = 2), so verdict drops to STALE."""
        _save_briefing(store, hours_ago=2.0)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "STALE"
        assert h["count_in_window"] == 1
        assert h["last_briefing_age_h"] == 2.0

    def test_dead_when_last_briefing_15h_ago(self, store):
        """A 15h-old briefing — last > 12h triggers DEAD verdict. The
        briefing is INSIDE the 24h window so count_in_window is 1; the
        DEAD verdict here is driven by age, not by an empty window."""
        _save_briefing(store, hours_ago=15.0)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "DEAD"
        assert h["count_in_window"] == 1
        assert 14.9 <= h["last_briefing_age_h"] <= 15.1

    def test_dead_when_last_briefing_far_outside_window(self, store):
        """Briefing 30h ago + 24h window — completely outside; verdict DEAD,
        count_in_window=0, last_briefing_age_h reflects the overall newest."""
        _save_briefing(store, hours_ago=30.0)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "DEAD"
        assert h["count_in_window"] == 0
        # Newest overall is 30h — well past the 12h DEAD threshold.
        assert h["last_briefing_age_h"] >= 29.5

    def test_dead_distinct_from_no_data(self, store):
        """A DEAD verdict (stale briefing exists) must NOT collapse to
        NO_DATA. The analyst-facing meaning differs materially:
        DEAD = was working, has stopped; NO_DATA = never started."""
        _save_briefing(store, hours_ago=20.0)
        h = store.briefing_health(window_h=24)
        assert h["verdict"] == "DEAD"
        assert h["last_briefing_age_h"] is not None


class TestWindowParameter:
    def test_window_h_floor_enforced(self, store):
        """window_h is clamped to >=1 so a 0 or negative caller can't crash."""
        _save_briefing(store, hours_ago=0.5)
        h = store.briefing_health(window_h=0)
        assert h["window_h"] == 1
        # Expected = 1/5 = 0.2 — floor of min_count_healthy is 1 by max() guard.
        assert h["expected_in_window"] == 0.2

    def test_short_window_excludes_old_briefings(self, store):
        """A 1h window only counts briefings within that hour."""
        _save_briefing(store, hours_ago=0.5)  # in window
        _save_briefing(store, hours_ago=3.0)  # outside 1h window
        _save_briefing(store, hours_ago=10.0)  # outside
        h = store.briefing_health(window_h=1)
        assert h["count_in_window"] == 1
        # Newest in window is 0.5h ago → HEALTHY (count >= floor of 1).
        assert h["verdict"] == "HEALTHY"


class TestReadOnlyContract:
    def test_briefing_health_does_not_mutate_articles(self, store):
        """Read-only invariant: the method must not touch ai_score / ml_score /
        score_source / urgency on any article row. Mirrors the storage-layer
        load-bearing constraint that every read primitive carries."""
        # Insert one urgent, llm-vetted article row first.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                " urgency, first_seen, cycle, ml_score, score_source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("aid1", "https://reuters.com/x", "MU earnings beat",
                 "rss", "", 1.0, 9.0, 1, _iso_ago(0.1), 0, None, "llm"),
            )
            store.conn.commit()
        _save_briefing(store, hours_ago=2.0)
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?", ("aid1",)
        ).fetchone()
        store.briefing_health(window_h=24)
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?", ("aid1",)
        ).fetchone()
        assert before == after

    def test_briefing_health_ignores_articles_table_entirely(self, store):
        """Defense-in-depth: even with a synthetic backtest article sitting in
        the articles table, the briefing_health view is computed solely from
        the briefings table — the synthetic row cannot inflate, mask, or
        otherwise perturb any field of the result."""
        # Insert a synthetic backtest article alongside.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                " urgency, first_seen, cycle, ml_score, score_source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("synth1", "backtest://run_42/2026-05-23/buy/NVDA",
                 "NVDA wins", "backtest_run_42_winner", "", 1.0, 5.0, 0,
                 _iso_ago(0.5), 0, None, None),
            )
            store.conn.commit()
        baseline = store.briefing_health(window_h=24)
        _save_briefing(store, hours_ago=2.0)
        with_briefing = store.briefing_health(window_h=24)
        # The synthetic article contributes ZERO to the briefing count;
        # adding one real briefing must take count_in_window from 0 -> 1.
        assert baseline["count_in_window"] == 0
        assert with_briefing["count_in_window"] == 1
        assert baseline["verdict"] == "NO_DATA"
        assert with_briefing["verdict"] in ("HEALTHY", "STALE")


class TestResultShape:
    def test_keys_are_stable_for_dashboard_consumers(self, store):
        """A dashboard / health-check consumer must be able to iterate the
        same set of keys regardless of state. Pin the exact key set so a
        future change cannot silently break downstream rendering code."""
        h = store.briefing_health(window_h=24)
        assert set(h.keys()) == {
            "window_h", "last_briefing_age_h", "count_in_window",
            "expected_in_window", "verdict",
        }

    def test_verdict_is_always_one_of_the_four_strings(self, store):
        """The verdict alphabet is closed — no other string may leak out
        (a buggy refactor that introduced a 5th value would break here)."""
        for setup in (
            lambda s: None,                               # empty
            lambda s: _save_briefing(s, hours_ago=2.0),    # fresh single
            lambda s: _save_briefing(s, hours_ago=8.0),    # stale-by-age
            lambda s: _save_briefing(s, hours_ago=15.0),   # dead-by-age in win
            lambda s: _save_briefing(s, hours_ago=30.0),   # outside window
        ):
            with store._write_lock:
                store.conn.execute("DELETE FROM briefings")
                store.conn.commit()
            setup(store)
            v = store.briefing_health(window_h=24)["verdict"]
            assert v in ("HEALTHY", "STALE", "DEAD", "NO_DATA")
