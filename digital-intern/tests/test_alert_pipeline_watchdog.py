"""Alert-pipeline watchdog decision core — pure logic, no files/Discord.

Pins the behaviour that converts a silently-stalled urgent-alert pipeline
into a Discord page: critical-worker liveness, stale/missing snapshot, the
throttle (anchored to incident start, not last check), and recovery notices.
"""
from __future__ import annotations

import importlib

wd = importlib.import_module("scripts.alert_pipeline_watchdog")


def _snap(workers):
    return {"ts": "2026-05-18T01:30:00+00:00", "workers_ok": 0,
            "workers_dead": 0, "workers": workers}


def _w(name, *, alive, state="ok", age=5.0, exc=""):
    return {"name": name, "alive": alive, "state": state,
            "crashes_5m": 0, "last_ok_age_s": age, "last_exception": exc}


NOW = 1_000_000.0


class TestHealthy:
    def test_fresh_snapshot_all_critical_alive_is_silent(self):
        snap = _snap([_w("alert", alive=True), _w("scorer", alive=True),
                      _w("heartbeat", alive=True), _w("gdelt", alive=False)])
        msgs, state = wd.evaluate(snap, snapshot_age_s=30.0,
                                  now_epoch=NOW, throttle_state={})
        assert msgs == []
        assert state == {}

    def test_noncritical_dead_worker_is_ignored(self):
        """A dead gdelt/web worker is not an analyst-visible alert outage —
        only alert/scorer/heartbeat escalate."""
        snap = _snap([_w("alert", alive=True), _w("gdelt", alive=False),
                      _w("web", alive=False)])
        msgs, state = wd.evaluate(snap, 30.0, NOW, {})
        assert msgs == []
        assert state == {}


class TestCriticalWorkerHung:
    def test_hung_alert_worker_escalates(self):
        snap = _snap([_w("alert", alive=False, state="ok", age=908.0,
                          exc="")])
        msgs, state = wd.evaluate(snap, 30.0, NOW, {})
        assert len(msgs) == 1
        assert "alert" in msgs[0] and "NOT being delivered" in msgs[0]
        # 908s with no exception is the observed blocked-not-crashed case.
        assert "15.1 min" in msgs[0]
        assert state == {"worker:alert": NOW}

    def test_each_critical_worker_independent(self):
        snap = _snap([_w("alert", alive=False, age=600.0),
                      _w("scorer", alive=False, age=600.0),
                      _w("heartbeat", alive=True)])
        msgs, state = wd.evaluate(snap, 30.0, NOW, {})
        assert len(msgs) == 2
        assert set(state) == {"worker:alert", "worker:scorer"}


class TestSnapshotStaleOrMissing:
    def test_missing_snapshot_escalates_daemon_down(self):
        msgs, state = wd.evaluate(None, None, NOW, {})
        assert len(msgs) == 1
        assert "daemon is down" in msgs[0]
        assert "supervisor_state_missing" in state

    def test_stale_snapshot_escalates_supervisor_wedged(self):
        # 20 min old > STALE_SNAPSHOT_S (15 min): daemon crash-looping/wedged.
        msgs, state = wd.evaluate(_snap([_w("alert", alive=True)]),
                                  snapshot_age_s=20 * 60,
                                  now_epoch=NOW, throttle_state={})
        assert len(msgs) == 1
        assert "stale" in msgs[0] and "crash-looping" in msgs[0]
        assert "supervisor_state_stale" in state

    def test_stale_snapshot_short_circuits_worker_checks(self):
        """When the snapshot is stale its per-worker data is lying; only the
        single stale condition fires, not also a worker condition."""
        msgs, state = wd.evaluate(_snap([_w("alert", alive=False)]),
                                  snapshot_age_s=20 * 60, now_epoch=NOW,
                                  throttle_state={})
        assert list(state) == ["supervisor_state_stale"]


class TestThrottle:
    def test_repeat_within_window_is_suppressed_but_anchored(self):
        snap = _snap([_w("alert", alive=False, age=908.0)])
        # First page at NOW.
        _, s1 = wd.evaluate(snap, 30.0, NOW, {})
        assert s1 == {"worker:alert": NOW}
        # 10 min later, still down — no second page, fire-time still anchored
        # to the incident start (NOW), not the re-check time.
        msgs, s2 = wd.evaluate(snap, 30.0, NOW + 600, s1)
        assert msgs == []
        assert s2 == {"worker:alert": NOW}

    def test_repage_after_throttle_window(self):
        snap = _snap([_w("alert", alive=False, age=2000.0)])
        prev = {"worker:alert": NOW}
        later = NOW + wd.ESCALATION_THROTTLE_S + 1
        msgs, state = wd.evaluate(snap, 30.0, later, prev)
        assert len(msgs) == 1
        assert state == {"worker:alert": later}


class TestRecovery:
    def test_cleared_condition_emits_recovery_and_drops_key(self):
        # alert was paged about ~5 min ago, now alive again.
        prev = {"worker:alert": NOW - 300}
        snap = _snap([_w("alert", alive=True)])
        msgs, state = wd.evaluate(snap, 30.0, NOW, prev)
        assert len(msgs) == 1
        assert "recovered" in msgs[0] and "alert" in msgs[0]
        assert "worker:alert" not in state  # dropped after recovery

    def test_ancient_entry_pruned_without_recovery_spam(self):
        """A throttle entry far past its TTL must not emit a stale 'recovered'
        line and must not be carried forward (bounded state file)."""
        prev = {"worker:alert": NOW - wd._STATE_TTL_S - 10}
        snap = _snap([_w("alert", alive=True)])
        msgs, state = wd.evaluate(snap, 30.0, NOW, prev)
        assert msgs == []
        assert state == {}

    def test_recovery_only_for_genuinely_paged_condition(self):
        """A zero/falsy last-fire timestamp (was throttled, never actually
        paged) must not produce a phantom recovery message."""
        prev = {"worker:scorer": 0.0}
        snap = _snap([_w("scorer", alive=True)])
        msgs, state = wd.evaluate(snap, 30.0, NOW, prev)
        assert msgs == []
        assert state == {}
