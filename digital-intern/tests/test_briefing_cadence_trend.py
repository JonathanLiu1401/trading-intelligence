"""Tests for ``ArticleStore.briefing_cadence_trend`` — the *trend* sibling to
``briefing_health``.

Why these tests exist (news-analyst lens): ``briefing_health`` is point-in-time
("is the MOST RECENT briefing fresh?"). It returns HEALTHY the instant a fresh
briefing fires, even if the prior 10 briefings averaged 9h gaps (Opus quota
throttling, Claude CLI auth lapsing intermittently, the heartbeat_worker
getting blocked on a slow DB read). The trend sibling answers "is the
cadence slipping?" — early warning that flips before STALE/DEAD fire.

Live evidence motivating the primitive (2026-05-24 pull, last 11 intervals
in the briefings table):

    5.21, 5.26, 6.26, 10.23, 7.08, 10.26, 5.09, 27.64, 5.21, 5.43, 8.61

mean ≈ 8.75h (75% slower than the 5h ``HEARTBEAT_INTERVAL``), max 27.64h.
``briefing_health`` on the same pull returned HEALTHY (most recent 5.15h
ago); the cadence trend exposes the otherwise-invisible degradation.

Coverage:
  * NO_DATA when fewer than 2 intervals (need 2 for a trend).
  * ON_CADENCE on a normal 5h-cadence stream.
  * SLIPPING when mean is 20-50% slow OR one gap >= 2 cadences.
  * DRIFTING when mean exceeds 1.5x expected.
  * Verdict ladder precedence (DRIFTING outranks SLIPPING).
  * Result shape: closed key set, closed verdict alphabet.
  * Read-only: no DB write, no article-row mutation.
  * Defensive parsing: unparseable timestamps don't crash; bad rows dropped.
  * ``last_n`` clamp and ``expected_cadence_h`` clamp.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _save_briefing(store, hours_ago: float, text: str = "briefing body") -> None:
    """Insert a briefing row dated ``hours_ago`` hours before now.

    Mirrors ``test_briefing_health._save_briefing`` exactly so the two
    test files stay in lockstep on test-data convention.
    """
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
            (_iso_ago(hours_ago), text, 10),
        )
        store.conn.commit()


def _save_briefings_chronological(store, hours_ago_list: list[float]) -> None:
    """Insert multiple briefings (oldest first in argument)."""
    for h in hours_ago_list:
        _save_briefing(store, hours_ago=h)


# ───────────────────────────────────────────────────────────────────────────
# Verdict ladder
# ───────────────────────────────────────────────────────────────────────────


class TestVerdictLadder:
    def test_no_data_when_zero_briefings(self, store):
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "NO_DATA"
        assert h["n_intervals"] == 0
        assert h["intervals_h"] == []
        assert h["mean_interval_h"] is None
        assert h["max_interval_h"] is None
        assert h["p50_interval_h"] is None
        assert h["drift_pct"] is None

    def test_no_data_when_only_one_briefing(self, store):
        """A single briefing yields zero intervals — NO_DATA, not HEALTHY.
        We need at least 2 intervals (3 briefings) to draw a trend."""
        _save_briefing(store, hours_ago=2.0)
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "NO_DATA"
        assert h["n_intervals"] == 0

    def test_no_data_when_only_two_briefings_one_interval(self, store):
        """Two briefings = one interval. We require >= 2 intervals — a
        single gap could be a transient and is already the STALE regime."""
        _save_briefing(store, hours_ago=2.0)
        _save_briefing(store, hours_ago=7.0)
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "NO_DATA"
        assert h["n_intervals"] == 1
        # The interval itself IS computed and exposed for diagnostics —
        # only the verdict is suppressed.
        assert len(h["intervals_h"]) == 1

    def test_on_cadence_when_intervals_near_expected(self, store):
        """Five briefings spaced ~5h apart — canonical ON_CADENCE state."""
        # Chronological: oldest first. Spacings ~5h.
        _save_briefings_chronological(store, [20.0, 15.1, 10.2, 5.0, 0.1])
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "ON_CADENCE"
        assert h["n_intervals"] == 4
        # All gaps near 5.0h.
        assert all(4.5 <= g <= 5.5 for g in h["intervals_h"]), h["intervals_h"]
        assert 4.5 <= h["mean_interval_h"] <= 5.5
        # drift_pct should be small in absolute terms.
        assert abs(h["drift_pct"]) <= 20.0

    def test_slipping_when_mean_20_to_50_pct_slow(self, store):
        """Mean ~7h (40% slower than 5h expected) — SLIPPING."""
        # 4 intervals of ~7h each.
        _save_briefings_chronological(store, [28.0, 21.0, 14.0, 7.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "SLIPPING"
        assert h["mean_interval_h"] >= 6.0
        assert h["mean_interval_h"] <= 7.5  # well below 1.5x = 7.5
        assert h["drift_pct"] >= 20.0

    def test_slipping_when_one_gap_exceeds_two_cadences(self, store):
        """Mean is ~5h ON CADENCE but ONE 11h gap (>= 2 * 5h) is SLIPPING.

        Early warning: even when average is fine, a single >= 2-cadence
        gap is the exact pattern that flipped DEAD once in the live log
        (the 27.64h gap in the motivating evidence). Catch it BEFORE the
        next gap turns the average bad too.
        """
        # Sequence: 5h, 5h, 11h, 5h. Mean = 6.5h (30% slow), max 11h (>= 10).
        # Mean drift 30% triggers SLIPPING under the mean rule too — to
        # isolate the "max > 2*cadence" rule we'd need mean < 1.2x AND
        # max > 2x. With 5h cadence: 5,5,5,11 → mean 6.5 → 30%. So the
        # max rule is reinforcing here, but the mean rule alone also
        # demands SLIPPING. A more isolating sequence:
        # 4.5, 4.5, 4.5, 11 → mean 6.125 → drift 22.5% → SLIPPING (mean).
        # We want a sequence where mean < 1.2x (drift < 20%) but max > 2x.
        # 4, 4, 4, 11 → mean 5.75 → drift 15% → max 11 > 10 → SLIPPING (max).
        _save_briefings_chronological(store, [23.0, 19.0, 15.0, 11.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "SLIPPING"
        assert h["max_interval_h"] >= 10.0
        # The MEAN rule alone wouldn't fire (drift < 20%); it's the MAX rule.
        assert h["drift_pct"] < 20.0, (
            f"this test asserts the max-rule path; mean drift was "
            f"{h['drift_pct']} (>= 20% means mean rule fired too)"
        )

    def test_drifting_when_mean_exceeds_1_5x(self, store):
        """Mean ~10h (100% slower than 5h expected) — DRIFTING."""
        _save_briefings_chronological(store, [40.0, 30.0, 20.0, 10.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "DRIFTING"
        assert h["mean_interval_h"] >= 7.5  # 1.5 * 5

    def test_drifting_outranks_slipping(self, store):
        """When BOTH conditions hold (mean drift > 50% AND max > 2x), the
        verdict must be DRIFTING — the more-severe label wins. Mirrors
        ``ml_training_health``'s DEAD-outranks-DIVERGING precedence."""
        # Mean ~12h (140% slow), max 25h (way over 10) — both fire.
        _save_briefings_chronological(store, [48.0, 36.0, 23.0, 12.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["verdict"] == "DRIFTING"
        assert h["mean_interval_h"] > 7.5  # DRIFTING threshold
        assert h["max_interval_h"] > 10.0  # SLIPPING max threshold

    def test_live_evidence_reproduces_drifting(self, store):
        """Reproduce the exact live evidence cited in the docstring.

        Last 11 intervals from the live log:
            5.21, 5.26, 6.26, 10.23, 7.08, 10.26, 5.09, 27.64, 5.21, 5.43, 8.61

        Mean ≈ 8.75 (75% slow) → DRIFTING.
        """
        # Reverse-engineer chronological hours_ago values from these
        # intervals (newest 0.1h ago).
        intervals = [5.21, 5.26, 6.26, 10.23, 7.08, 10.26, 5.09, 27.64,
                     5.21, 5.43, 8.61]
        # Build hours_ago list (oldest first): start with the cumulative
        # sum going backward.
        # Newest is 0.1h ago; each prior briefing is its successor's
        # hours_ago + the interval between them.
        ages = [0.1]
        for gap in reversed(intervals):
            ages.append(ages[-1] + gap)
        ages.reverse()  # oldest first
        _save_briefings_chronological(store, ages)
        h = store.briefing_cadence_trend(last_n=11)
        # Live mean was ~8.75 — should land DRIFTING.
        assert h["verdict"] == "DRIFTING"
        assert h["mean_interval_h"] >= 7.5
        # The 27.64h gap should be the max.
        assert h["max_interval_h"] >= 25.0

    def test_verdict_always_in_closed_alphabet(self, store):
        """The verdict alphabet is closed — no other string may leak out
        (a buggy refactor that introduced a 5th value breaks here)."""
        for setup in (
            lambda s: None,                                                  # empty
            lambda s: _save_briefing(s, hours_ago=2.0),                       # 1 briefing
            lambda s: _save_briefings_chronological(s, [20.0, 15.0, 10.0, 5.0, 0.0]),
            lambda s: _save_briefings_chronological(s, [40.0, 30.0, 20.0, 10.0, 0.0]),
        ):
            with store._write_lock:
                store.conn.execute("DELETE FROM briefings")
                store.conn.commit()
            setup(store)
            v = store.briefing_cadence_trend()["verdict"]
            assert v in ("ON_CADENCE", "SLIPPING", "DRIFTING", "NO_DATA")


# ───────────────────────────────────────────────────────────────────────────
# Interval computation correctness
# ───────────────────────────────────────────────────────────────────────────


class TestIntervalComputation:
    def test_intervals_are_chronological_newest_last(self, store):
        """Returned intervals_h must be in chronological order (oldest gap
        first, newest gap last) so a chart consumer can render
        left-to-right by time without re-sorting."""
        _save_briefings_chronological(store, [15.0, 10.0, 5.0, 0.0])
        h = store.briefing_cadence_trend(last_n=4)
        assert h["n_intervals"] == 3
        # All gaps are ~5h. To pin chronology, force ASYMMETRIC gaps:
        with store._write_lock:
            store.conn.execute("DELETE FROM briefings")
            store.conn.commit()
        # Gaps oldest-to-newest: 8h, 5h, 2h.
        _save_briefings_chronological(store, [15.0, 7.0, 2.0, 0.0])
        h = store.briefing_cadence_trend(last_n=4)
        assert h["n_intervals"] == 3
        # Gap-1 ≈ 8h (oldest), gap-2 ≈ 5h, gap-3 ≈ 2h (newest LAST).
        assert h["intervals_h"][0] >= 7.5
        assert 4.5 <= h["intervals_h"][1] <= 5.5
        assert h["intervals_h"][2] <= 2.5

    def test_last_n_caps_intervals_returned(self, store):
        """``last_n`` bounds the number of briefings pulled (last_n+1 rows),
        yielding ``last_n`` intervals at most."""
        # 8 briefings.
        _save_briefings_chronological(
            store, [35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]
        )
        h = store.briefing_cadence_trend(last_n=3)
        # 3 intervals from 4 briefings.
        assert h["n_intervals"] == 3
        # Should be the most RECENT 3 intervals.
        assert all(g <= 6.0 for g in h["intervals_h"])

    def test_mean_p50_max_are_consistent(self, store):
        """For a known input, mean / p50 / max must match the obvious math."""
        # Gaps: 3, 4, 5, 6, 7. Mean=5, median=5, max=7.
        _save_briefings_chronological(store, [25.0, 22.0, 18.0, 13.0, 7.0, 0.0])
        h = store.briefing_cadence_trend(last_n=5)
        assert h["n_intervals"] == 5
        # Mean ~5.0 (sum 25 / 5).
        assert 4.5 <= h["mean_interval_h"] <= 5.5
        # Median ~5.0.
        assert 4.5 <= h["p50_interval_h"] <= 5.5
        # Max ~7.0.
        assert 6.5 <= h["max_interval_h"] <= 7.5

    def test_drift_pct_sign_convention(self, store):
        """drift_pct = (mean - expected) / expected * 100.
        Positive = slipping; negative = ahead of cadence."""
        # Slow stream (10h mean, 100% slipping).
        _save_briefings_chronological(store, [40.0, 30.0, 20.0, 10.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["drift_pct"] is not None and h["drift_pct"] > 50.0
        # Fast stream (3h mean, ahead of cadence — uncommon but supported).
        with store._write_lock:
            store.conn.execute("DELETE FROM briefings")
            store.conn.commit()
        _save_briefings_chronological(store, [12.0, 9.0, 6.0, 3.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["drift_pct"] is not None and h["drift_pct"] < 0.0

    def test_zero_gap_handled_without_crash(self, store):
        """Two briefings with the same timestamp → 0h interval. Must not
        crash; the mean/drift math must remain finite."""
        same_ts = _iso_ago(2.0)
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
                (same_ts, "a", 1),
            )
            store.conn.execute(
                "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
                (same_ts, "b", 1),
            )
            store.conn.commit()
        # Add a third briefing 7h further back so we have 2 intervals total.
        _save_briefing(store, hours_ago=9.0)
        h = store.briefing_cadence_trend()
        assert h["n_intervals"] == 2
        assert 0.0 in h["intervals_h"]
        # Result must be a finite number — no NaN / inf leaking out.
        assert isinstance(h["mean_interval_h"], (int, float))


# ───────────────────────────────────────────────────────────────────────────
# Parameter clamps
# ───────────────────────────────────────────────────────────────────────────


class TestParameters:
    def test_last_n_floor_enforced(self, store):
        """``last_n`` is clamped to >= 1 so a 0 or negative caller can't
        crash. Returned last_n field reflects the clamped value."""
        _save_briefings_chronological(store, [10.0, 5.0, 0.0])
        h = store.briefing_cadence_trend(last_n=0)
        assert h["last_n"] == 1

    def test_expected_cadence_h_floor(self, store):
        """``expected_cadence_h`` is clamped above 0 so a divide-by-zero in
        drift_pct cannot occur on a misconfigured caller."""
        _save_briefings_chronological(store, [10.0, 5.0, 0.0])
        h = store.briefing_cadence_trend(expected_cadence_h=0)
        # The returned field reflects the clamp (small positive value).
        assert h["expected_cadence_h"] > 0
        # drift_pct must still be a finite number.
        assert h["drift_pct"] is None or isinstance(
            h["drift_pct"], (int, float)
        )

    def test_custom_expected_cadence(self, store):
        """A caller can pass a different expected cadence (e.g. for a path
        running at 2h)."""
        # Gaps ~5h. At expected=5 → ON_CADENCE; at expected=2 → DRIFTING.
        _save_briefings_chronological(store, [20.0, 15.0, 10.0, 5.0, 0.0])
        h_5 = store.briefing_cadence_trend(expected_cadence_h=5.0)
        h_2 = store.briefing_cadence_trend(expected_cadence_h=2.0)
        assert h_5["verdict"] == "ON_CADENCE"
        assert h_2["verdict"] == "DRIFTING"


# ───────────────────────────────────────────────────────────────────────────
# Defensive parsing
# ───────────────────────────────────────────────────────────────────────────


class TestDefensiveParsing:
    def test_unparseable_timestamp_rows_dropped_silently(self, store):
        """A briefing row with a corrupt ts must NOT crash the call —
        same discipline as ``briefing_health`` (defensive fromisoformat
        guard). The bad row is dropped; valid rows still computed."""
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
                ("not-a-timestamp", "junk", 1),
            )
            store.conn.commit()
        _save_briefings_chronological(store, [15.0, 10.0, 5.0, 0.0])
        h = store.briefing_cadence_trend()
        assert h["n_intervals"] == 3
        # Verdict is computed from the valid rows only.
        assert h["verdict"] in ("ON_CADENCE", "SLIPPING", "DRIFTING", "NO_DATA")


# ───────────────────────────────────────────────────────────────────────────
# Read-only contract
# ───────────────────────────────────────────────────────────────────────────


class TestReadOnlyContract:
    def test_does_not_mutate_articles(self, store):
        """Read-only invariant: the method must not touch ai_score / ml_score
        / score_source / urgency on any article row. Mirrors the
        storage-layer load-bearing constraint every read primitive carries."""
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
        _save_briefings_chronological(store, [15.0, 10.0, 5.0, 0.0])
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?", ("aid1",)
        ).fetchone()
        store.briefing_cadence_trend()
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id=?", ("aid1",)
        ).fetchone()
        assert before == after

    def test_ignores_articles_table_entirely(self, store):
        """Defense-in-depth: a synthetic backtest article sitting in the
        articles table cannot inflate, mask, or otherwise perturb any
        field of the cadence trend result — the primitive reads only
        the briefings table."""
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
        baseline = store.briefing_cadence_trend()
        _save_briefings_chronological(store, [15.0, 10.0, 5.0, 0.0])
        with_briefings = store.briefing_cadence_trend()
        assert baseline["verdict"] == "NO_DATA"
        assert with_briefings["n_intervals"] == 3


# ───────────────────────────────────────────────────────────────────────────
# Result shape
# ───────────────────────────────────────────────────────────────────────────


class TestResultShape:
    def test_keys_are_stable_for_dashboard_consumers(self, store):
        """A dashboard / health-check consumer must be able to iterate the
        same set of keys regardless of state. Pin the exact key set so a
        future change cannot silently break downstream rendering code."""
        h = store.briefing_cadence_trend()
        assert set(h.keys()) == {
            "expected_cadence_h", "last_n", "n_intervals", "intervals_h",
            "mean_interval_h", "max_interval_h", "p50_interval_h",
            "drift_pct", "verdict",
        }

    def test_keys_stable_across_no_data_and_populated(self, store):
        """The result must have the SAME key set whether NO_DATA or
        populated, so a consumer can dispatch on verdict alone without
        guarding key existence."""
        empty = store.briefing_cadence_trend()
        _save_briefings_chronological(store, [15.0, 10.0, 5.0, 0.0])
        populated = store.briefing_cadence_trend()
        assert set(empty.keys()) == set(populated.keys())
