"""Tests for paper_trader.preflight — offline trader health check.

The motivating bug: ``_running_runner_pids`` historically used substring
matching on ``/proc/<pid>/cmdline``, so a Claude HYBRID review agent whose
prompt text contained the string ``paper_trader.runner`` was false-positively
counted as a runner process. Worst case: the real runner dies but review
agents are still running → preflight reports "runner alive" and the DOWN
verdict the operator needs never fires. The fix tokenises the cmdline and
requires argv[0] to be a python interpreter.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.preflight import (
    _cmdline_is_runner,
    build_preflight,
)


# ── _cmdline_is_runner: the substring-vs-token bug ────────────────────────

class TestCmdlineIsRunner:
    """The cmdline matcher must accept REAL runner invocations and reject
    look-alikes (the original bug)."""

    # --- ACCEPT real runner cmdlines ---

    def test_accepts_module_mode_launch(self):
        assert _cmdline_is_runner(
            "/usr/bin/python3 -m paper_trader.runner"
        ) is True

    def test_accepts_module_mode_with_extra_args(self):
        assert _cmdline_is_runner(
            "/usr/bin/python3 -m paper_trader.runner --verbose"
        ) is True

    def test_accepts_script_mode_paper_dash_trader(self):
        # The live systemd unit form.
        assert _cmdline_is_runner(
            "/usr/bin/python3 /home/zeph/trading-intelligence/"
            "paper-trader/runner.py"
        ) is True

    def test_accepts_script_mode_paper_underscore_trader(self):
        assert _cmdline_is_runner(
            "/usr/bin/python3 /home/zeph/paper_trader/runner.py"
        ) is True

    def test_accepts_python_version_suffixes(self):
        # python / python3 / python3.12 / python3.13 are all valid argv[0].
        for py in ("python", "python3", "python3.12", "python3.13"):
            assert _cmdline_is_runner(f"{py} -m paper_trader.runner") is True
        # Full path forms.
        assert _cmdline_is_runner(
            "/usr/local/bin/python3.13 -m paper_trader.runner"
        ) is True

    # --- REJECT the live false-positives (the canonical bug) ---

    def test_rejects_claude_agent_with_runner_substring_in_prompt(self):
        # The exact pathology: a Claude HYBRID review agent invoked with a
        # prompt that mentions ``paper_trader/runner.py`` as discussion text.
        # argv[0] is ``claude``, not python — must NOT count as a runner.
        prompt = (
            "claude --model claude-opus-4-7 --permission-mode "
            "bypassPermissions --print BEFORE STARTING: Read "
            "paper_trader/runner.py and paper_trader/strategy.py "
            "before touching anything"
        )
        assert _cmdline_is_runner(prompt) is False

    def test_rejects_claude_agent_with_dotted_module_in_prompt(self):
        # The dotted-name form, also in HYBRID agent prompts.
        prompt = (
            "claude --print 'from paper_trader.runner import main; main()'"
        )
        assert _cmdline_is_runner(prompt) is False

    def test_rejects_bare_runner_py_in_unrelated_path(self):
        # Some unrelated process with a script named runner.py — without the
        # ``paper`` path segment, it must not match.
        assert _cmdline_is_runner(
            "/usr/bin/python3 /tmp/runner.py"
        ) is False

    def test_rejects_module_mode_with_wrong_module(self):
        assert _cmdline_is_runner(
            "/usr/bin/python3 -m some_other.runner"
        ) is False

    def test_rejects_module_mode_with_substring_only_match(self):
        # paper_trader.runner_helpers is not paper_trader.runner — token
        # equality, not prefix, must be enforced.
        assert _cmdline_is_runner(
            "/usr/bin/python3 -m paper_trader.runner_helpers"
        ) is False

    def test_rejects_dash_m_without_immediately_following_module(self):
        # "-m" then a different token, then the module name — the order
        # matters: -m must be IMMEDIATELY followed by the module.
        assert _cmdline_is_runner(
            "/usr/bin/python3 -m some_pkg paper_trader.runner extra"
        ) is False

    def test_rejects_empty_cmdline(self):
        assert _cmdline_is_runner("") is False

    def test_rejects_whitespace_only_cmdline(self):
        assert _cmdline_is_runner("   \t  ") is False

    def test_rejects_shell_command_with_runner_substring(self):
        assert _cmdline_is_runner(
            "/bin/bash -c 'echo paper_trader.runner'"
        ) is False

    def test_rejects_grep_pgrep_of_runner(self):
        # Operator inspecting the runner — must not get counted as one.
        assert _cmdline_is_runner(
            "pgrep -f paper_trader.runner"
        ) is False
        assert _cmdline_is_runner(
            "grep -r paper_trader.runner ."
        ) is False


# ── build_preflight: the verdict router ────────────────────────────────────

class TestBuildPreflight:
    """The router is the single source of overall verdict + exit code."""

    NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)

    def test_no_data_when_everything_missing(self):
        out = build_preflight(
            runner_pids=None,
            heartbeat=None,
            reliability=None,
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "NO_DATA"
        assert out["exit_code"] == 0
        assert "process probe unavailable" in out["runner_process"]

    def test_down_when_no_runner_found(self):
        # runner_pids=[] (provably empty), not None (probe unavailable).
        out = build_preflight(
            runner_pids=[],
            heartbeat=None,
            reliability=None,
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "DOWN"
        assert out["exit_code"] == 3
        assert any("runner process not running" in d for d in out["drivers"])

    def test_down_beats_degraded(self):
        # Both DOWN driver (no runner) and DEGRADED driver (stale feed)
        # present → overall must be DOWN.
        out = build_preflight(
            runner_pids=[],
            heartbeat=None,
            reliability=None,
            feed={"split_brain": False, "stale": True,
                  "chosen_age_hours": 8.0},
            now=self.NOW,
        )
        assert out["overall"] == "DOWN"
        assert out["exit_code"] == 3

    def test_no_actionable_signal_when_all_green(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed={"split_brain": False, "stale": False,
                  "chosen_age_hours": 0.5},
            now=self.NOW,
        )
        assert out["overall"] == "HEALTHY"
        assert out["exit_code"] == 0
        assert "alive (pid 12345)" in out["runner_process"]
        # No drivers when nothing tripped (any DEGRADED/DOWN would have
        # appended one).
        assert out["drivers"] == []

    def test_insufficient_reliability_with_healthy_heartbeat_is_healthy(self):
        out = build_preflight(
            runner_pids=None,
            heartbeat={"verdict": "HEALTHY", "headline": "last decision 8s ago"},
            reliability={"state": "INSUFFICIENT", "headline": "sample 3 (<12)"},
            feed={"split_brain": False, "stale": False,
                  "chosen_age_hours": 0.5},
            now=self.NOW,
        )
        assert out["overall"] == "HEALTHY"
        assert out["exit_code"] == 0
        assert "no decisions recorded" not in out["headline"]

    def test_heartbeat_stalled_drives_down(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "STALLED", "headline": "no decisions in 4h"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "DOWN"
        assert out["exit_code"] == 3
        assert any("STALLED" in d for d in out["drivers"])

    def test_heartbeat_lagging_drives_degraded(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "LAGGING", "headline": "behind cadence"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"
        assert out["exit_code"] == 2

    def test_reliability_critical_drives_degraded(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "CRITICAL", "headline": "55% parse fail"},
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"
        assert out["exit_code"] == 2
        assert any("CRITICAL" in d for d in out["drivers"])

    def test_reliability_restart_recommended_drives_degraded(self):
        # restart_recommended True must drive DEGRADED even when state is
        # otherwise benign.
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok",
                         "restart_recommended": True},
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"

    def test_feed_split_brain_drives_degraded(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed={"split_brain": True, "stale": False,
                  "chosen_age_hours": 0.5},
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"
        assert any("split-brain" in d for d in out["drivers"])

    def test_feed_stale_drives_degraded_with_age_in_driver(self):
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed={"split_brain": False, "stale": True,
                  "chosen_age_hours": 12.5},
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"
        assert any("12.5h" in d for d in out["drivers"])

    def test_feed_stale_with_non_numeric_age_does_not_crash(self):
        # The ternary string-format branch must tolerate a None age.
        out = build_preflight(
            runner_pids=[12345],
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed={"split_brain": False, "stale": True,
                  "chosen_age_hours": None},
            now=self.NOW,
        )
        assert out["overall"] == "DEGRADED"
        # Driver still posted, with the generic fallback phrasing.
        assert any("freshest live article is old" in d for d in out["drivers"])

    def test_runner_pids_none_does_not_drive_down(self):
        # ``None`` means probe unavailable (no /proc); must NOT trip DOWN.
        out = build_preflight(
            runner_pids=None,
            heartbeat={"verdict": "HEALTHY", "headline": "ok"},
            reliability={"state": "HEALTHY", "headline": "ok"},
            feed=None,
            now=self.NOW,
        )
        assert out["overall"] == "HEALTHY"
        assert out["exit_code"] == 0
        assert "probe unavailable" in out["runner_process"]
        # The "no runner found" driver must NOT be present — None ≠ [].
        assert not any("runner process not running" in d
                       for d in out["drivers"])

    def test_exit_codes_match_rank(self):
        # HEALTHY = 0, DEGRADED = 2, DOWN = 3. NO_DATA also 0 (nothing
        # actionable yet).
        cases = [
            ({"verdict": "HEALTHY", "headline": ""}, 0),
            ({"verdict": "LAGGING", "headline": ""}, 2),
            ({"verdict": "STALLED", "headline": ""}, 3),
        ]
        for hb, expected_code in cases:
            out = build_preflight(
                runner_pids=[1], heartbeat=hb, reliability=None,
                feed=None, now=self.NOW,
            )
            assert out["exit_code"] == expected_code, (hb, out)

    def test_payload_carries_constituent_dicts_verbatim(self):
        hb = {"verdict": "HEALTHY", "headline": "X"}
        rl = {"state": "HEALTHY", "headline": "Y"}
        fd = {"split_brain": False, "stale": False, "chosen_age_hours": 1.0}
        out = build_preflight(
            runner_pids=[99], heartbeat=hb, reliability=rl, feed=fd,
            now=self.NOW,
        )
        assert out["heartbeat"] is hb
        assert out["reliability"] is rl
        assert out["feed"] is fd
        assert out["runner_pids"] == [99]

    def test_as_of_is_iso(self):
        out = build_preflight(
            runner_pids=[1], heartbeat=None, reliability=None,
            feed=None, now=self.NOW,
        )
        # ISO-formatted at seconds resolution.
        assert out["as_of"].startswith("2026-05-24T12:00:00")


# ── Integration smoke: the scan loop's I/O contract ─────────────────────

class TestScanProcLoop:
    """The proc-scan loop wraps the pure ``_cmdline_is_runner`` predicate
    with a small amount of glob+open+pid-parse I/O. Verify it returns
    [] / None / [pid] in the three exit paths without faking /proc — the
    pid filtering itself is covered by ``TestCmdlineIsRunner`` above (16
    cases including every documented false-positive)."""

    def test_no_proc_returns_none(self, monkeypatch):
        # Probe unavailable (macOS / CI without /proc) MUST return None,
        # not [], so the router distinguishes "could not tell" from
        # "provably no runner".
        from paper_trader import preflight as pf

        monkeypatch.setattr(pf.os.path, "isdir", lambda p: False)
        assert pf._running_runner_pids() is None

    def test_empty_glob_returns_empty_list(self, monkeypatch):
        # Scan ran but found no candidate cmdlines → [] (not None).
        from paper_trader import preflight as pf

        monkeypatch.setattr(pf.os.path, "isdir",
                            lambda p: True if p == "/proc" else False)
        monkeypatch.setattr(pf.glob, "glob", lambda pat: [])
        assert pf._running_runner_pids() == []

    def test_real_scan_filters_to_actual_runner_only(self):
        # On a live host with the running paper_trader, the scan must
        # accept the real runner and reject every concurrent Claude agent
        # whose prompt happens to embed ``paper_trader.runner`` as text.
        # Sanity check: at most one runner pid, and if any are returned
        # each must satisfy the pure ``_cmdline_is_runner`` predicate.
        from paper_trader import preflight as pf
        import os as _os

        pids = pf._running_runner_pids()
        if pids is None:
            pytest.skip("/proc unavailable")
        for pid in pids:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cmdline = fh.read().replace(b"\x00", b" ").decode(
                        "utf-8", "replace")
            except (OSError, IOError):
                continue  # process exited between scan and verify
            assert pf._cmdline_is_runner(cmdline), (
                f"Scan returned pid {pid} but its cmdline does not "
                f"pass the runner predicate: {cmdline!r}"
            )


class TestCli:
    def test_json_flag_emits_machine_readable_payload(self, monkeypatch, capsys):
        from paper_trader import preflight as pf

        payload = {
            "overall": "HEALTHY",
            "exit_code": 0,
            "headline": "HEALTHY — ok",
        }
        monkeypatch.setattr(pf, "run_preflight", lambda: payload)

        assert pf._cli(["--json"]) == 0
        out = capsys.readouterr().out
        assert json.loads(out) == payload
        assert "=== paper-trader preflight ===" not in out

    def test_human_mode_still_prints_report(self, monkeypatch, capsys):
        from paper_trader import preflight as pf

        payload = {
            "as_of": "2026-05-24T12:00:00+00:00",
            "overall": "NO_DATA",
            "exit_code": 0,
            "headline": "NO_DATA — no decisions recorded yet.",
            "recommended_action": "Re-run later.",
            "runner_process": "process probe unavailable",
            "heartbeat": None,
            "reliability": None,
            "feed": None,
            "drivers": [],
        }
        monkeypatch.setattr(pf, "run_preflight", lambda: payload)

        assert pf._cli([]) == 0
        out = capsys.readouterr().out
        assert "=== paper-trader preflight ===" in out
        assert "overall  : NO_DATA" in out
