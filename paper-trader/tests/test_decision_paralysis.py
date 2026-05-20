"""Tests for analytics/decision_paralysis.py — pure, deterministic.

The contract under test: detect contiguous runs of passive actions
(HOLD-only / NO_DECISION-only / mixed passive) on the leading (newest)
edge of the decisions table, and emit a verdict ladder distinct from
``decision_health`` (24h aggregate) and ``runner_heartbeat`` (NO_DECISION
only). HOLD_LOCK is the gap this module fills — every cycle decided, but
none of them moved the book.
"""
from datetime import datetime, timezone, timedelta

from paper_trader.analytics.decision_paralysis import (
    HOLD_LOCK_THRESHOLD,
    IDLE_STORM_THRESHOLD,
    PASSIVE_LOOP_THRESHOLD,
    _classify,
    _leading_run,
    _longest_run,
    build_decision_paralysis,
)


def _dec(ts: str, action: str) -> dict:
    return {"timestamp": ts, "action_taken": action}


def _newest_first(actions: list[str], now: datetime,
                  spacing_minutes: float = 30.0) -> list[dict]:
    """Build a newest-first decisions list with even spacing back from ``now``."""
    rows = []
    for i, a in enumerate(actions):
        ts = (now - timedelta(minutes=spacing_minutes * i)).isoformat(
            timespec="seconds")
        rows.append(_dec(ts, a))
    return rows


class TestClassify:
    def test_buckets(self):
        assert _classify("BUY NVDA → FILLED") == "FILLED"
        assert _classify("SELL MU → BLOCKED") == "BLOCKED"
        assert _classify("HOLD NVDA → HOLD") == "HOLD"
        assert _classify("NO_DECISION") == "NO_DECISION"
        assert _classify("") == "NO_DECISION"
        assert _classify(None) == "NO_DECISION"
        assert _classify("REBALANCE NVDA → SOMETHING") == "OTHER"

    def test_mirrors_decision_health_classify(self):
        # Drift-lock — keeps the predicate aligned with the canonical
        # _classify in decision_health.
        from paper_trader.analytics.decision_health import _classify as canon
        for action in (
            "BUY NVDA → FILLED",
            "SELL MU → BLOCKED",
            "HOLD NVDA → HOLD",
            "NO_DECISION",
            "",
            None,
            "REBALANCE XYZ → SOMETHING",
        ):
            assert _classify(action) == canon(action)[0]


class TestRunHelpers:
    def test_leading_run_counts_only_leading(self):
        cats = ["HOLD", "HOLD", "FILLED", "HOLD", "HOLD"]
        assert _leading_run(cats, lambda c: c == "HOLD") == 2

    def test_leading_run_breaks_at_first_nonmatch(self):
        cats = ["FILLED", "HOLD", "HOLD", "HOLD"]
        assert _leading_run(cats, lambda c: c == "HOLD") == 0

    def test_longest_run_finds_max_contiguous(self):
        cats = ["HOLD", "HOLD", "FILLED", "HOLD", "HOLD", "HOLD", "FILLED"]
        assert _longest_run(cats, lambda c: c == "HOLD") == 3


class TestBuildDecisionParalysis:
    def setup_method(self):
        self.now = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)

    def test_empty(self):
        out = build_decision_paralysis([], now=self.now)
        assert out["verdict"] == "NO_DATA"
        assert out["n_decisions_scanned"] == 0
        assert out["current_hold_streak"] == 0

    def test_active_on_recent_fill(self):
        # One FILLED leading the run breaks any passive streak.
        rows = _newest_first(["BUY NVDA → FILLED"] + ["HOLD NVDA → HOLD"] * 8,
                             now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "ACTIVE"
        assert out["current_hold_streak"] == 0
        assert out["current_passive_streak"] == 0
        assert out["last_active_action"] == "BUY NVDA → FILLED"

    def test_hold_lock_at_threshold(self):
        rows = _newest_first(["HOLD NVDA → HOLD"] * HOLD_LOCK_THRESHOLD,
                             now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "HOLD_LOCK"
        assert out["current_hold_streak"] == HOLD_LOCK_THRESHOLD
        assert out["current_no_decision_streak"] == 0
        # No FILLED/BLOCKED row exists.
        assert out["last_active_action"] is None
        assert "HOLD_LOCK" in out["headline"]

    def test_hold_lock_below_threshold_is_active(self):
        rows = _newest_first(["HOLD NVDA → HOLD"] * (HOLD_LOCK_THRESHOLD - 1),
                             now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "ACTIVE"
        assert out["current_hold_streak"] == HOLD_LOCK_THRESHOLD - 1

    def test_idle_storm_takes_precedence_over_hold_lock(self):
        # 10 NO_DECISION leading, then HOLDs behind — should fire IDLE_STORM,
        # not HOLD_LOCK, even though both passive types are present.
        actions = ["NO_DECISION"] * IDLE_STORM_THRESHOLD + \
                  ["HOLD NVDA → HOLD"] * 5
        rows = _newest_first(actions, now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "IDLE_STORM"
        assert out["current_no_decision_streak"] == IDLE_STORM_THRESHOLD
        # Hold streak is 0 because NO_DECISION is the leading category.
        assert out["current_hold_streak"] == 0

    def test_passive_loop_mixed_hold_and_no_decision(self):
        # Alternate HOLD and NO_DECISION so neither hits its own threshold
        # individually, but the combined passive streak does.
        assert PASSIVE_LOOP_THRESHOLD > HOLD_LOCK_THRESHOLD  # contract
        actions = []
        for i in range(PASSIVE_LOOP_THRESHOLD):
            actions.append("NO_DECISION" if i % 3 == 0 else "HOLD NVDA → HOLD")
        rows = _newest_first(actions, now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["current_passive_streak"] >= PASSIVE_LOOP_THRESHOLD
        # The most-specific bands shouldn't fire on this mix — verify and
        # then confirm PASSIVE_LOOP. The alternation guarantees neither
        # current_hold_streak nor current_no_decision_streak can exceed
        # their own thresholds.
        assert out["current_hold_streak"] < HOLD_LOCK_THRESHOLD
        assert out["current_no_decision_streak"] < IDLE_STORM_THRESHOLD
        assert out["verdict"] == "PASSIVE_LOOP"

    def test_hold_lock_fires_before_passive_loop(self):
        # 12 HOLDs in a row: HOLD_LOCK threshold (10) is hit, PASSIVE_LOOP
        # threshold (15) is not — verdict ladder order matters.
        rows = _newest_first(["HOLD NVDA → HOLD"] * 12, now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "HOLD_LOCK"

    def test_last_active_ts_recorded_when_present(self):
        old_fill_ts = (self.now - timedelta(hours=3)).isoformat(
            timespec="seconds")
        rows = (
            [_dec((self.now - timedelta(minutes=15 * i)).isoformat(
                timespec="seconds"), "HOLD NVDA → HOLD")
             for i in range(HOLD_LOCK_THRESHOLD)]
            + [_dec(old_fill_ts, "BUY NVDA → FILLED")]
        )
        out = build_decision_paralysis(rows, now=self.now)
        assert out["verdict"] == "HOLD_LOCK"
        assert out["last_active_action"] == "BUY NVDA → FILLED"
        assert out["last_active_ts"] == old_fill_ts
        # 3h ago FILLED — hours_since_last_active should be ~3.0
        assert abs(out["hours_since_last_active"] - 3.0) < 0.01

    def test_24h_window_bounds_longest_run(self):
        # 20 HOLDs in the last hour + 5 HOLDs outside the 24h window. The
        # 24h-bounded longest run should not include the older block.
        recent = _newest_first(["HOLD NVDA → HOLD"] * 20, now=self.now,
                               spacing_minutes=2.0)
        far_past_ts = (self.now - timedelta(hours=30)).isoformat(
            timespec="seconds")
        old = [_dec(far_past_ts, "HOLD NVDA → HOLD")] * 5
        out = build_decision_paralysis(recent + old, now=self.now)
        assert out["longest_hold_streak_24h"] == 20
        assert out["n_decisions_24h"] == 20

    def test_garbage_rows_do_not_raise(self):
        rows = [
            {"action_taken": None, "timestamp": "not-a-date"},
            {},  # no keys at all
            {"action_taken": "HOLD NVDA → HOLD", "timestamp": None},
        ]
        out = build_decision_paralysis(rows, now=self.now)
        # Two of three classify as NO_DECISION, one as HOLD — no verdict
        # band fires; verdict is ACTIVE by fallthrough.
        assert out["verdict"] == "ACTIVE"
        assert out["n_decisions_scanned"] == 3

    def test_thresholds_exposed_in_output(self):
        rows = _newest_first(["HOLD NVDA → HOLD"], now=self.now)
        out = build_decision_paralysis(rows, now=self.now)
        # Thresholds in output so a UI can render the gap to verdict
        # without hardcoding module constants.
        assert out["hold_lock_threshold"] == HOLD_LOCK_THRESHOLD
        assert out["idle_storm_threshold"] == IDLE_STORM_THRESHOLD
        assert out["passive_loop_threshold"] == PASSIVE_LOOP_THRESHOLD


class TestEndpointIntegration:
    def test_endpoint_returns_json(self, monkeypatch):
        # Verify the route is wired and returns a sane shape via Flask
        # test client — the analytics_verification convention.
        from paper_trader import dashboard as dash_mod

        # Stub out get_store to return a synthetic decisions list — avoids
        # touching the real paper_trader.db on disk.
        class _FakeStore:
            def recent_decisions(self, limit):
                return [{"action_taken": "HOLD NVDA → HOLD",
                         "timestamp": "2026-05-19T12:00:00+00:00"}
                        for _ in range(HOLD_LOCK_THRESHOLD)]

        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        # SWR cache may have stale data from a previous test — clear it
        # before exercising the endpoint.
        cache = getattr(dash_mod, "_SWR_CACHE", None)
        if cache is not None:
            cache.pop("decision-paralysis", None)

        client = dash_mod.app.test_client()
        resp = client.get("/api/decision-paralysis")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["verdict"] == "HOLD_LOCK"
        assert body["current_hold_streak"] == HOLD_LOCK_THRESHOLD
