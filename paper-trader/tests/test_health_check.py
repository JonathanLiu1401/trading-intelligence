"""Tests for paper_trader.health_check — unified preflight + should_restart.

The router has three jobs:
  1. Map each child verdict into the unified state token (HEALTHY / NO_DATA /
     DEGRADED / OPS_ONLY / RESTART / DOWN).
  2. Pick the WORST state by precedence — DOWN > RESTART > {OPS_ONLY,
     DEGRADED} > HEALTHY > NO_DATA, first-reported wins ties so the
     operator sees the more specific narrative.
  3. Surface the right child's headline + action verbatim for the chosen
     state (single source of truth — the router never re-derives wording).
"""
from __future__ import annotations

from datetime import datetime, timezone

from paper_trader.health_check import (
    _EXIT,
    _RANK,
    _preflight_to_state,
    _should_restart_to_state,
    _worse,
    build_health_check,
)


# ── state mapping ─────────────────────────────────────────────────────────

class TestPreflightToState:
    def test_pass_through_known_states(self):
        for s in ("DOWN", "DEGRADED", "HEALTHY", "NO_DATA"):
            assert _preflight_to_state({"overall": s}) == s

    def test_unknown_state_becomes_no_data(self):
        assert _preflight_to_state({"overall": "WEIRD"}) == "NO_DATA"

    def test_none_input_is_no_data(self):
        assert _preflight_to_state(None) == "NO_DATA"

    def test_non_dict_is_no_data(self):
        assert _preflight_to_state("string") == "NO_DATA"
        assert _preflight_to_state([]) == "NO_DATA"

    def test_missing_overall_is_no_data(self):
        assert _preflight_to_state({}) == "NO_DATA"


class TestShouldRestartToState:
    def test_ok_becomes_healthy(self):
        assert _should_restart_to_state({"state": "OK"}) == "HEALTHY"

    def test_restart_passes_through(self):
        assert _should_restart_to_state({"state": "RESTART"}) == "RESTART"

    def test_ops_only_passes_through(self):
        assert _should_restart_to_state({"state": "OPS_ONLY"}) == "OPS_ONLY"

    def test_error_becomes_no_data(self):
        assert _should_restart_to_state({"state": "ERROR"}) == "NO_DATA"

    def test_unknown_state_is_no_data(self):
        assert _should_restart_to_state({"state": "WEIRD"}) == "NO_DATA"

    def test_none_input_is_no_data(self):
        assert _should_restart_to_state(None) == "NO_DATA"

    def test_non_dict_is_no_data(self):
        assert _should_restart_to_state(42) == "NO_DATA"


# ── precedence rank ───────────────────────────────────────────────────────

class TestWorse:
    def test_down_beats_everything(self):
        for other in ("RESTART", "OPS_ONLY", "DEGRADED", "HEALTHY", "NO_DATA"):
            assert _worse("DOWN", other) == "DOWN"
            assert _worse(other, "DOWN") == "DOWN"

    def test_restart_beats_ops_and_below(self):
        for other in ("OPS_ONLY", "DEGRADED", "HEALTHY", "NO_DATA"):
            assert _worse("RESTART", other) == "RESTART"
            assert _worse(other, "RESTART") == "RESTART"

    def test_ops_and_degraded_tie_first_reported_wins(self):
        # Both rank 2 — ``a`` (first) wins ties so first-reported drives the
        # narrative (the OPS_ONLY narrative has a concrete remediation; a
        # generic DEGRADED is informational, but EITHER is the worst).
        assert _worse("OPS_ONLY", "DEGRADED") == "OPS_ONLY"
        assert _worse("DEGRADED", "OPS_ONLY") == "DEGRADED"

    def test_healthy_beats_no_data(self):
        # Real healthy signal must override the sentinel.
        assert _worse("HEALTHY", "NO_DATA") == "HEALTHY"
        # but only if HEALTHY comes first (ranks 1 > 0).
        assert _worse("NO_DATA", "HEALTHY") == "HEALTHY"

    def test_rank_table_consistent_with_exit_codes(self):
        # DOWN must be the highest rank AND highest exit (3); HEALTHY/NO_DATA
        # are the only states mapping to exit 0.
        assert _RANK["DOWN"] == max(_RANK.values())
        assert _EXIT["HEALTHY"] == 0
        assert _EXIT["NO_DATA"] == 0


# ── build_health_check: end-to-end routing ────────────────────────────────

class TestBuildHealthCheck:
    NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)

    def _pf(self, overall, headline="", drivers=None, action=""):
        return {
            "overall": overall,
            "headline": headline,
            "drivers": drivers or [],
            "recommended_action": action,
        }

    def _sr(self, state, headline="", actions=None,
            restart_reasons=None, ops_reasons=None):
        return {
            "state": state,
            "headline": headline,
            "actions": actions or [],
            "restart_reasons": restart_reasons or [],
            "ops_reasons": ops_reasons or [],
        }

    def test_both_none_is_no_data(self):
        out = build_health_check(
            preflight_result=None, should_restart_result=None, now=self.NOW)
        assert out["state"] == "NO_DATA"
        assert out["exit_code"] == 0
        assert "NO_DATA" in out["headline"]

    def test_both_healthy_is_healthy(self):
        out = build_health_check(
            preflight_result=self._pf("HEALTHY", "all green"),
            should_restart_result=self._sr("OK", "nothing to do"),
            now=self.NOW,
        )
        assert out["state"] == "HEALTHY"
        assert out["exit_code"] == 0
        assert "HEALTHY" in out["headline"]

    def test_down_preflight_drives_overall(self):
        out = build_health_check(
            preflight_result=self._pf(
                "DOWN", "DOWN — the trading loop is not running",
                action="Restart paper-trader."),
            should_restart_result=self._sr("OK", "ok"),
            now=self.NOW,
        )
        assert out["state"] == "DOWN"
        assert out["exit_code"] == 3
        assert "loop is not running" in out["headline"]
        assert "Restart" in out["recommended_action"]

    def test_restart_recommended_drives_overall(self):
        out = build_health_check(
            preflight_result=self._pf("HEALTHY", "ok"),
            should_restart_result=self._sr(
                "RESTART", "RESTART RECOMMENDED — supervision STALE",
                actions=["systemctl --user restart paper-trader"],
                restart_reasons=["supervision STALE"]),
            now=self.NOW,
        )
        assert out["state"] == "RESTART"
        assert out["exit_code"] == 1
        assert "RESTART RECOMMENDED" in out["headline"]
        assert "systemctl" in out["recommended_action"]

    def test_ops_only_drives_overall_when_preflight_healthy(self):
        out = build_health_check(
            preflight_result=self._pf("HEALTHY", "ok"),
            should_restart_result=self._sr(
                "OPS_ONLY", "OPS — host saturation",
                actions=["reduce concurrent Opus load"],
                ops_reasons=["host saturated"]),
            now=self.NOW,
        )
        assert out["state"] == "OPS_ONLY"
        assert out["exit_code"] == 2
        assert "host saturation" in out["headline"]
        assert "Opus" in out["recommended_action"]

    def test_degraded_preflight_drives_when_should_restart_ok(self):
        out = build_health_check(
            preflight_result=self._pf(
                "DEGRADED", "DEGRADED — reliability CRITICAL",
                drivers=["reliability CRITICAL"],
                action="Review the drivers."),
            should_restart_result=self._sr("OK", "ok"),
            now=self.NOW,
        )
        assert out["state"] == "DEGRADED"
        assert out["exit_code"] == 2
        assert "reliability CRITICAL" in out["headline"]

    def test_restart_beats_ops_only(self):
        # Both should_restart and preflight could fire, but should_restart
        # itself returns one state at a time. Make should_restart return
        # RESTART and preflight DEGRADED — overall should be RESTART (rank 3
        # > 2).
        out = build_health_check(
            preflight_result=self._pf("DEGRADED", "deg"),
            should_restart_result=self._sr(
                "RESTART", "restart now",
                actions=["systemctl --user restart paper-trader"]),
            now=self.NOW,
        )
        assert out["state"] == "RESTART"
        assert out["exit_code"] == 1

    def test_down_beats_restart(self):
        # Preflight DOWN (no runner) and should_restart RESTART — DOWN wins.
        out = build_health_check(
            preflight_result=self._pf(
                "DOWN", "no runner",
                action="Restart paper-trader."),
            should_restart_result=self._sr(
                "RESTART", "restart",
                actions=["systemctl --user restart paper-trader"]),
            now=self.NOW,
        )
        assert out["state"] == "DOWN"
        assert out["exit_code"] == 3

    def test_should_restart_failure_does_not_take_down_preflight(self):
        # should_restart returned None (dashboard unreachable); preflight
        # said HEALTHY — overall must remain HEALTHY, not get knocked to
        # NO_DATA by the missing child.
        out = build_health_check(
            preflight_result=self._pf("HEALTHY", "ok"),
            should_restart_result=None,
            now=self.NOW,
        )
        assert out["state"] == "HEALTHY"

    def test_preflight_failure_does_not_take_down_should_restart(self):
        out = build_health_check(
            preflight_result=None,
            should_restart_result=self._sr("OK", "ok"),
            now=self.NOW,
        )
        assert out["state"] == "HEALTHY"

    def test_both_failures_is_no_data(self):
        # Both children unavailable — nothing actionable, but the command
        # must not raise.
        out = build_health_check(
            preflight_result=None, should_restart_result=None, now=self.NOW)
        assert out["state"] == "NO_DATA"
        assert out["exit_code"] == 0

    def test_payload_carries_children_verbatim(self):
        pf = self._pf("DEGRADED", "x")
        sr = self._sr("OPS_ONLY", "y")
        out = build_health_check(
            preflight_result=pf, should_restart_result=sr, now=self.NOW)
        # The router never copies/mutates the children — exact identity.
        assert out["preflight"] is pf
        assert out["should_restart"] is sr

    def test_as_of_is_iso_at_seconds(self):
        out = build_health_check(
            preflight_result=None, should_restart_result=None, now=self.NOW)
        # Seconds-resolution ISO, no microseconds in the string.
        assert out["as_of"].startswith("2026-05-24T12:00:00")

    def test_action_for_ops_only_joins_multiple(self):
        # When should_restart provides a list of actions, the router must
        # surface ALL of them (joined), not just the first.
        out = build_health_check(
            preflight_result=self._pf("HEALTHY", "ok"),
            should_restart_result=self._sr(
                "OPS_ONLY", "ops",
                actions=["kill agents", "wait 5 min"]),
            now=self.NOW,
        )
        assert "kill agents" in out["recommended_action"]
        assert "wait 5 min" in out["recommended_action"]

    def test_exit_code_for_each_state(self):
        cases = [
            ("DOWN",     3),
            ("RESTART",  1),
            ("OPS_ONLY", 2),
            ("DEGRADED", 2),
            ("HEALTHY",  0),
            ("NO_DATA",  0),
        ]
        for state, expected in cases:
            # Construct an input that uniquely produces this state.
            if state == "DOWN":
                args = {"preflight_result": self._pf("DOWN", ""),
                        "should_restart_result": None}
            elif state == "RESTART":
                args = {"preflight_result": self._pf("HEALTHY", ""),
                        "should_restart_result": self._sr("RESTART", "")}
            elif state == "OPS_ONLY":
                args = {"preflight_result": self._pf("HEALTHY", ""),
                        "should_restart_result": self._sr("OPS_ONLY", "")}
            elif state == "DEGRADED":
                args = {"preflight_result": self._pf("DEGRADED", ""),
                        "should_restart_result": self._sr("OK", "")}
            elif state == "HEALTHY":
                args = {"preflight_result": self._pf("HEALTHY", ""),
                        "should_restart_result": self._sr("OK", "")}
            else:  # NO_DATA
                args = {"preflight_result": None,
                        "should_restart_result": None}
            out = build_health_check(now=self.NOW, **args)
            assert out["state"] == state, (state, out)
            assert out["exit_code"] == expected, (state, out)
