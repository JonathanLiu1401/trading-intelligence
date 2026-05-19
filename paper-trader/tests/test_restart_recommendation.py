"""Tests for analytics/restart_recommendation.py.

The discriminating locks:

* **Precedence ladder** — URGENT > RECOMMENDED > MONITOR > OK, first match
  wins. A regression that reorders the ladder must fail.
* **False-HEALTHY closure** — the exact live scenario (empty=81%, held event
  in 11.5h, $445 exposure) MUST read RESTART_URGENT/restart_now=True.
  ``runner_heartbeat`` cadence-HEALTHY does not unstick this verdict.
* **IDLE_STORM intercept** — 5+ consecutive NO_DECISION cycles flips
  RESTART_RECOMMENDED even with empty_rate unknown, mirroring the
  ``runner_heartbeat`` decision-efficacy gate.
* **None-tolerance** — every scalar input MAY be None; the builder degrades
  to OK rather than raising. None means "unknown", not "zero".
* **No held event** = no RESTART_URGENT, no matter how bad empty_rate is —
  the operator's restart-now bit is anchored on exposure-to-event.
* **MONITOR_NO_DECISION_N (3) vs IDLE_STORM_N (5)** — 3..4 consecutive
  NO_DECISION reads MONITOR; 5+ reads RESTART_RECOMMENDED.
* **next_check_seconds** monotonically shortens as urgency climbs so a
  cron-driven caller polls fastest exactly when it matters most.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.restart_recommendation import (
    IDLE_STORM_N,
    MONITOR_EMPTY_RATE,
    MONITOR_NO_DECISION_N,
    RECOMMENDED_EMPTY_RATE,
    RECOMMENDED_HOURS,
    URGENT_EMPTY_RATE,
    URGENT_HOURS,
    build_restart_recommendation,
)

_NOW = datetime(2026, 5, 19, 12, 30, 0, tzinfo=timezone.utc)


def _build(**kw):
    kw.setdefault("now", _NOW)
    kw.setdefault("empty_rate_pct", None)
    kw.setdefault("host_saturated", None)
    kw.setdefault("held_imminent_exposure_usd", None)
    kw.setdefault("hours_to_nearest_held_event", None)
    kw.setdefault("consecutive_no_decision", None)
    return build_restart_recommendation(**kw)


# ───────────────────────── False-HEALTHY closure (the anchor) ─────────────────────────


class TestFalseHealthyClosure:
    """The exact pathology that motivated this surface."""

    def test_live_wedge_with_imminent_print_reads_urgent(self):
        r = _build(
            empty_rate_pct=81.4,
            host_saturated=True,
            held_imminent_exposure_usd=444.7,
            hours_to_nearest_held_event=11.5,
            consecutive_no_decision=16,
        )
        assert r["verdict"] == "RESTART_URGENT"
        assert r["restart_now"] is True
        assert r["urgency_score"] == 1.0
        # The headline must surface the wedge AND the print so a Discord
        # post is self-explanatory.
        assert "NVDA" not in r["headline"]  # we don't pass a ticker
        assert "11" in r["headline"]        # the event proximity
        assert "URGENT" in r["headline"]

    def test_inputs_block_round_trips_for_transparency(self):
        r = _build(
            empty_rate_pct=81.4,
            host_saturated=True,
            held_imminent_exposure_usd=444.7,
            hours_to_nearest_held_event=11.5,
            consecutive_no_decision=16,
        )
        assert r["inputs"]["empty_rate_pct"] == 81.4
        assert r["inputs"]["host_saturated"] is True
        assert r["inputs"]["held_imminent_exposure_usd"] == 444.7
        assert r["inputs"]["hours_to_nearest_held_event"] == 11.5
        assert r["inputs"]["consecutive_no_decision"] == 16


# ───────────────────────── Precedence ladder ─────────────────────────


class TestPrecedenceLadder:

    def test_urgent_dominates_recommended(self):
        # The same inputs that would otherwise be IDLE_STORM are still URGENT
        # because the held-imminent exposure + URGENT_EMPTY_RATE wedge wins.
        r = _build(
            empty_rate_pct=URGENT_EMPTY_RATE,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=URGENT_HOURS,
            consecutive_no_decision=IDLE_STORM_N,
        )
        assert r["verdict"] == "RESTART_URGENT"

    def test_idle_storm_recommended_without_event(self):
        r = _build(consecutive_no_decision=IDLE_STORM_N)
        assert r["verdict"] == "RESTART_RECOMMENDED"
        assert r["restart_now"] is True

    def test_idle_storm_with_imminent_event_bumps_urgency(self):
        r1 = _build(consecutive_no_decision=IDLE_STORM_N)
        r2 = _build(
            consecutive_no_decision=IDLE_STORM_N,
            held_imminent_exposure_usd=100.0,
            hours_to_nearest_held_event=10.0,
        )
        assert r2["urgency_score"] > r1["urgency_score"]

    def test_moderate_empty_with_imminent_event_recommends(self):
        r = _build(
            empty_rate_pct=RECOMMENDED_EMPTY_RATE,
            held_imminent_exposure_usd=100.0,
            hours_to_nearest_held_event=20.0,
        )
        assert r["verdict"] == "RESTART_RECOMMENDED"
        assert r["restart_now"] is True

    def test_moderate_empty_without_event_is_at_most_monitor(self):
        # Same empty rate, no imminent event — must NOT recommend a restart.
        r = _build(empty_rate_pct=RECOMMENDED_EMPTY_RATE)
        assert r["verdict"] in ("OK", "MONITOR")
        assert r["restart_now"] is False

    def test_low_empty_with_imminent_event_is_monitor(self):
        r = _build(
            empty_rate_pct=MONITOR_EMPTY_RATE,
            held_imminent_exposure_usd=100.0,
            hours_to_nearest_held_event=20.0,
        )
        assert r["verdict"] == "MONITOR"
        assert r["restart_now"] is False

    def test_host_saturated_alone_is_monitor(self):
        r = _build(host_saturated=True)
        assert r["verdict"] == "MONITOR"
        assert r["restart_now"] is False

    def test_monitor_no_decision_below_storm_is_monitor(self):
        r = _build(consecutive_no_decision=MONITOR_NO_DECISION_N)
        assert r["verdict"] == "MONITOR"
        assert r["restart_now"] is False

    def test_monitor_no_decision_minus_one_is_ok(self):
        r = _build(consecutive_no_decision=MONITOR_NO_DECISION_N - 1)
        assert r["verdict"] == "OK"
        assert r["restart_now"] is False

    def test_clean_state_is_ok(self):
        r = _build(
            empty_rate_pct=5.0,
            host_saturated=False,
            held_imminent_exposure_usd=0.0,
            hours_to_nearest_held_event=None,
            consecutive_no_decision=0,
        )
        assert r["verdict"] == "OK"
        assert r["restart_now"] is False
        assert r["urgency_score"] == 0.0


# ───────────────────────── Exposure / event proximity gating ─────────────────────────


class TestEventProximity:

    def test_urgent_requires_event_within_urgent_hours(self):
        # empty_rate hits URGENT but the event is 30h away — must not be URGENT.
        r = _build(
            empty_rate_pct=URGENT_EMPTY_RATE + 1.0,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=URGENT_HOURS + 6.0,
        )
        assert r["verdict"] != "RESTART_URGENT"

    def test_urgent_at_boundary(self):
        r = _build(
            empty_rate_pct=URGENT_EMPTY_RATE,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=URGENT_HOURS,
        )
        assert r["verdict"] == "RESTART_URGENT"

    def test_zero_exposure_blocks_urgent(self):
        r = _build(
            empty_rate_pct=URGENT_EMPTY_RATE + 10.0,
            held_imminent_exposure_usd=0.0,
            hours_to_nearest_held_event=5.0,
        )
        assert r["verdict"] != "RESTART_URGENT"

    def test_outside_recommended_window_is_at_most_monitor(self):
        r = _build(
            empty_rate_pct=RECOMMENDED_EMPTY_RATE,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=RECOMMENDED_HOURS + 1.0,
        )
        # Outside the held-imminent window even moderate empty_rate cannot
        # trigger a RESTART recommendation.
        assert r["verdict"] != "RESTART_RECOMMENDED"
        assert r["restart_now"] is False


# ───────────────────────── None-tolerance / robustness ─────────────────────────


class TestNoneTolerance:

    def test_all_none_inputs_are_ok(self):
        r = _build()
        assert r["verdict"] == "OK"
        assert r["restart_now"] is False
        assert r["urgency_score"] == 0.0

    def test_empty_rate_none_does_not_trigger_urgent(self):
        r = _build(
            empty_rate_pct=None,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=10.0,
        )
        assert r["verdict"] != "RESTART_URGENT"

    def test_garbage_inputs_do_not_raise(self):
        # The endpoint may pass through user-supplied / upstream-error
        # values. We coerce, never raise.
        r = build_restart_recommendation(
            empty_rate_pct="not-a-number",  # type: ignore[arg-type]
            host_saturated="yes",            # type: ignore[arg-type]
            held_imminent_exposure_usd="500",
            hours_to_nearest_held_event="ten",
            consecutive_no_decision="five",  # type: ignore[arg-type]
            now=_NOW,
        )
        # The unparseable empty_rate / hours_to_event collapse to None ⇒
        # downstream effect is just OK or a streak-only MONITOR.
        assert r["verdict"] in ("OK", "MONITOR", "RESTART_RECOMMENDED")

    def test_negative_consecutive_no_decision_does_not_trigger(self):
        r = _build(consecutive_no_decision=-3)
        assert r["verdict"] == "OK"


# ───────────────────────── next_check_seconds cadence ─────────────────────────


class TestNextCheckCadence:

    def test_cadence_monotonically_shortens_with_urgency(self):
        ok = _build()
        mon = _build(host_saturated=True)
        rec = _build(consecutive_no_decision=IDLE_STORM_N)
        urg = _build(
            empty_rate_pct=URGENT_EMPTY_RATE + 5.0,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=5.0,
        )
        assert (urg["next_check_seconds"]
                < rec["next_check_seconds"]
                < mon["next_check_seconds"]
                < ok["next_check_seconds"])

    def test_urgent_cadence_under_2min(self):
        urg = _build(
            empty_rate_pct=URGENT_EMPTY_RATE,
            held_imminent_exposure_usd=500.0,
            hours_to_nearest_held_event=URGENT_HOURS,
        )
        assert urg["next_check_seconds"] <= 60


# ───────────────────────── Output schema stability ─────────────────────────


class TestSchemaStability:

    def test_required_keys_always_present(self):
        for params in [
                {},
                {"empty_rate_pct": 81.4, "held_imminent_exposure_usd": 500.0,
                 "hours_to_nearest_held_event": 10.0},
                {"consecutive_no_decision": 6},
        ]:
            r = _build(**params)
            for k in ("as_of", "verdict", "restart_now", "urgency_score",
                      "headline", "reasons", "next_check_seconds",
                      "inputs", "thresholds"):
                assert k in r, f"missing {k} for {params}"
            for k in ("urgent_empty_rate_pct", "recommended_empty_rate_pct",
                      "monitor_empty_rate_pct", "urgent_hours",
                      "recommended_hours", "idle_storm_n",
                      "monitor_no_decision_n"):
                assert k in r["thresholds"], k

    def test_thresholds_match_module_constants(self):
        r = _build()
        assert r["thresholds"]["urgent_empty_rate_pct"] == URGENT_EMPTY_RATE
        assert r["thresholds"]["recommended_empty_rate_pct"] == RECOMMENDED_EMPTY_RATE
        assert r["thresholds"]["monitor_empty_rate_pct"] == MONITOR_EMPTY_RATE
        assert r["thresholds"]["urgent_hours"] == URGENT_HOURS
        assert r["thresholds"]["recommended_hours"] == RECOMMENDED_HOURS
        assert r["thresholds"]["idle_storm_n"] == IDLE_STORM_N
        assert r["thresholds"]["monitor_no_decision_n"] == MONITOR_NO_DECISION_N
