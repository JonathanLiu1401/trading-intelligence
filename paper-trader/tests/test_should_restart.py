"""Locks for paper_trader.should_restart — the one-shot 'should I restart?'
verdict.

The builder is pure; these tests pin its discriminating logic with hand-
constructed inputs that mirror the live verdict shapes (`/api/supervision`,
`/api/runner-heartbeat`, `host_guard.pulse()`). The CLI itself only does
network + pretty-print — its `_render` is locked separately so we know the
operator-facing strings haven't drifted.

All assertions are exact: states, exit codes, restart_recommended booleans,
and substring presence of the human-facing reason strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import should_restart as sr


# ── builder: OK ─────────────────────────────────────────────────────────────

def _healthy_supervision() -> dict:
    return {
        "verdict": "HEALTHY",
        "recommendation": "Supervised and current — no action.",
        "actionable": False,
    }


def _healthy_heartbeat() -> dict:
    return {
        "verdict": "HEALTHY",
        "headline": "HEALTHY — last decision 12m ago.",
        "restart_recommended": False,
        "decision_efficacy": {"verdict": "PRODUCING", "headline": ""},
        "singleton_lock": {
            "status": "acquired", "have_lock": True, "degraded": False,
            "holder_pid": 12345,
            "headline": "OK — this runner holds the single-instance lock.",
        },
        "notify": {
            "verdict": "HEALTHY",
            "consecutive_failures": 0,
            "restart_recommended": False,
            "last_error": "",
        },
    }


def _clear_pulse() -> dict:
    return {"state": "CLEAR", "headline": "", "saturated": False}


class TestBuilderOK:
    def test_all_healthy_returns_ok(self):
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=_healthy_heartbeat(),
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "OK"
        assert v["exit_code"] == 0
        assert v["restart_recommended"] is False
        assert v["restart_reasons"] == []
        assert v["ops_reasons"] == []
        assert v["actions"] == []
        assert "OK" in v["headline"]


# ── builder: restart-recommended discriminators ────────────────────────────

class TestBuilderRestart:
    def test_stale_supervision_recommends_restart(self):
        sup = {
            "verdict": "STALE",
            "recommendation": "Supervised but running old code (boot a vs head b, behind 1).",
            "actionable": True,
        }
        v = sr.build_should_restart(
            supervision=sup,
            heartbeat=_healthy_heartbeat(),
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert v["exit_code"] == 1
        assert v["restart_recommended"] is True
        assert any("supervision STALE" in r for r in v["restart_reasons"])
        # No ops needed.
        assert v["ops_reasons"] == []
        assert v["actions"] == ["systemctl --user restart paper-trader"]

    def test_heartbeat_restart_flag_recommends_restart(self):
        hb = _healthy_heartbeat()
        hb["restart_recommended"] = True
        hb["headline"] = "STALLED — 18 cycles ago"
        hb["decision_efficacy"] = {
            "verdict": "IDLE_STORM",
            "headline": "95% NO_DECISION over last 20 cycles",
        }
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=hb,
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert v["restart_recommended"] is True
        assert any("STALLED — 18 cycles ago" in r for r in v["restart_reasons"])
        assert any("95% NO_DECISION" in r for r in v["restart_reasons"])

    def test_degraded_singleton_lock_recommends_restart(self):
        """A degraded singleton is a double-trade risk — actionable even if
        every other signal looks fine."""
        hb = _healthy_heartbeat()
        hb["singleton_lock"] = {
            "status": "degraded",
            "have_lock": False,
            "degraded": True,
            "holder_pid": None,
        }
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=hb,
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert v["restart_recommended"] is True
        assert any("singleton lock DEGRADED" in r
                   for r in v["restart_reasons"])

    def test_dark_discord_channel_recommends_restart(self):
        hb = _healthy_heartbeat()
        hb["notify"] = {
            "verdict": "DEGRADED",
            "consecutive_failures": 5,
            "restart_recommended": True,
            "last_error": "rc=127 env node not found",
        }
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=hb,
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert v["restart_recommended"] is True
        assert any("Discord channel DARK" in r for r in v["restart_reasons"])
        assert any("env node" in r for r in v["restart_reasons"])

    def test_dashboard_unreachable_recommends_restart(self):
        """The dashboard runs INSIDE the runner — if it's silent, the
        runner is the most likely cause."""
        v = sr.build_should_restart(
            supervision=None, heartbeat=None,
            host_pulse=_clear_pulse(),
            dashboard_reachable=False,
        )
        assert v["state"] == "RESTART"
        assert v["restart_recommended"] is True
        assert any("dashboard at :8090 unreachable" in r
                   for r in v["restart_reasons"])


# ── builder: ops-only and mixed ─────────────────────────────────────────────

class TestBuilderOps:
    def test_saturated_alone_is_ops_only(self):
        """A saturated host with an otherwise healthy trader gets OPS_ONLY:
        restart will not free the box, killing concurrent Opus will."""
        pulse = {
            "state": "SATURATED",
            "headline": "Opus is starved by the box — 5 concurrent Opus",
            "saturated": True,
        }
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=_healthy_heartbeat(),
            host_pulse=pulse,
            dashboard_reachable=True,
        )
        assert v["state"] == "OPS_ONLY"
        assert v["exit_code"] == 2
        assert v["restart_recommended"] is False
        assert any("5 concurrent Opus" in r for r in v["ops_reasons"])
        # No systemctl restart in the action list — that would not help.
        assert all("systemctl" not in a for a in v["actions"])
        assert any("reduce concurrent Opus" in a for a in v["actions"])

    def test_starved_alone_is_ops_only(self):
        pulse = {
            "state": "STARVED",
            "headline": "78% of the last 120 decisions never reached Opus",
            "saturated": False,
        }
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=_healthy_heartbeat(),
            host_pulse=pulse,
            dashboard_reachable=True,
        )
        assert v["state"] == "OPS_ONLY"
        assert v["restart_recommended"] is False
        assert any("78%" in r for r in v["ops_reasons"])

    def test_restart_wins_over_ops_when_both(self):
        """Stale code AND saturated host: restart wins state but ops action
        is listed FIRST in the remediation list (a freshly-booted runner
        re-starves immediately if you restart without clearing load)."""
        sup = {
            "verdict": "STALE",
            "recommendation": "Restart to apply commits",
            "actionable": True,
        }
        pulse = {
            "state": "SATURATED",
            "headline": "5 concurrent Opus",
            "saturated": True,
        }
        v = sr.build_should_restart(
            supervision=sup,
            heartbeat=_healthy_heartbeat(),
            host_pulse=pulse,
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert v["restart_recommended"] is True
        assert v["restart_reasons"] and v["ops_reasons"]
        # Action ordering matters — ops first.
        assert v["actions"][0].startswith("(ops)")
        assert "systemctl" in v["actions"][1]


# ── builder: error path ─────────────────────────────────────────────────────

class TestBuilderError:
    def test_all_inputs_none_is_error_state(self):
        v = sr.build_should_restart(
            supervision=None, heartbeat=None, host_pulse=None,
            dashboard_reachable=True,
        )
        assert v["state"] == "ERROR"
        assert v["exit_code"] == 3
        assert v["restart_recommended"] is False
        assert "ERROR" in v["headline"]

    def test_clear_pulse_only_still_returns_ok(self):
        """A live host_pulse=CLEAR with no other inputs is enough to say OK
        — we know at least one diagnostic. The ERROR state requires *every*
        input to be None."""
        v = sr.build_should_restart(
            supervision=None, heartbeat=None,
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "OK"
        assert v["exit_code"] == 0


# ── multiple reasons get counted in headline ────────────────────────────────

class TestHeadlineCount:
    def test_multiple_restart_reasons_appended_to_headline(self):
        sup = {"verdict": "UNSUPERVISED_STALE",
               "recommendation": "x", "actionable": True}
        hb = _healthy_heartbeat()
        hb["restart_recommended"] = True
        hb["headline"] = "STALLED"
        v = sr.build_should_restart(
            supervision=sup, heartbeat=hb,
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        assert v["state"] == "RESTART"
        assert len(v["restart_reasons"]) >= 2
        assert "(+ 1 more)" in v["headline"]


# ── CLI render is scannable ─────────────────────────────────────────────────

class TestRender:
    def test_ok_render_omits_empty_sections(self):
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=_healthy_heartbeat(),
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        out = sr._render(v)
        assert "[should-restart]" in out
        # OK case carries nothing else.
        assert "why restart" not in out
        assert "ops" not in out
        assert "remediation" not in out

    def test_restart_render_includes_reasons_and_actions(self):
        sup = {"verdict": "STALE", "recommendation": "Restart to deploy",
               "actionable": True}
        v = sr.build_should_restart(
            supervision=sup,
            heartbeat=_healthy_heartbeat(),
            host_pulse=_clear_pulse(),
            dashboard_reachable=True,
        )
        out = sr._render(v)
        assert "[should-restart] RESTART RECOMMENDED" in out
        assert "why restart" in out
        assert "supervision STALE" in out
        assert "1. systemctl --user restart paper-trader" in out

    def test_ops_render_distinguishes_from_restart(self):
        pulse = {"state": "SATURATED",
                 "headline": "5 concurrent Opus", "saturated": True}
        v = sr.build_should_restart(
            supervision=_healthy_supervision(),
            heartbeat=_healthy_heartbeat(),
            host_pulse=pulse,
            dashboard_reachable=True,
        )
        out = sr._render(v)
        assert "[should-restart] OPS" in out
        assert "restart will NOT fix" in out
        assert "ops" in out
        # No systemctl in remediation list — that's the whole point.
        assert "systemctl" not in out


# ── CLI fetch is degrade-safe ───────────────────────────────────────────────

class TestFetchDegradeSafe:
    def test_fetch_json_returns_none_on_connection_error(self, monkeypatch):
        """A connection refused / DNS / timeout must never raise — the CLI
        is supposed to run *when the dashboard is dead*."""
        import urllib.error

        def _boom(*a, **k):
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr("urllib.request.urlopen", _boom)
        assert sr._fetch_json("http://127.0.0.1:1/api/x") is None

    def test_fetch_json_returns_none_on_garbage_payload(self, monkeypatch):
        """Non-JSON bytes from a misconfigured proxy / wrong port must also
        degrade silently — no traceback to the operator."""
        class _Resp:
            status = 200
            def read(self): return b"<html>not json</html>"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda *a, **k: _Resp())
        assert sr._fetch_json("http://x/") is None


# ── CLI main wires inputs to builder ────────────────────────────────────────

class TestMain:
    def test_main_returns_zero_when_all_inputs_ok(self, monkeypatch, capsys):
        monkeypatch.setattr(sr, "gather", lambda *a, **k: {
            "supervision": _healthy_supervision(),
            "heartbeat": _healthy_heartbeat(),
            "host_pulse": _clear_pulse(),
            "dashboard_reachable": True,
        })
        rc = sr.main(argv=[])
        assert rc == 0
        captured = capsys.readouterr()
        assert "[should-restart]" in captured.out

    def test_main_returns_one_on_restart_recommended(self,
                                                      monkeypatch, capsys):
        monkeypatch.setattr(sr, "gather", lambda *a, **k: {
            "supervision": {"verdict": "STALE",
                            "recommendation": "Restart", "actionable": True},
            "heartbeat": _healthy_heartbeat(),
            "host_pulse": _clear_pulse(),
            "dashboard_reachable": True,
        })
        rc = sr.main(argv=[])
        assert rc == 1
        assert "RESTART RECOMMENDED" in capsys.readouterr().out

    def test_main_json_flag_emits_machine_payload(self,
                                                    monkeypatch, capsys):
        import json as _json
        monkeypatch.setattr(sr, "gather", lambda *a, **k: {
            "supervision": _healthy_supervision(),
            "heartbeat": _healthy_heartbeat(),
            "host_pulse": _clear_pulse(),
            "dashboard_reachable": True,
        })
        rc = sr.main(argv=["--json"])
        assert rc == 0
        payload = _json.loads(capsys.readouterr().out)
        assert payload["state"] == "OK"
        assert payload["restart_recommended"] is False
        assert "as_of" in payload
        assert "exit_code" in payload
