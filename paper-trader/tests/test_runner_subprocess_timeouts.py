"""Regression: every subprocess.run/check_output call in runner.py that
talks to git or pkill MUST pass a finite timeout.

Background: without a timeout, a wedged git (work tree mid-rebase with a
crashed editor still holding index.lock, or the data dir's underlying
filesystem stalled under load) would block ``subprocess.check_output``
*forever*, freezing the git-watcher thread and silently defeating the
deferred-restart-on-new-commits feature — a committed fix would never deploy
because the watcher never observes HEAD changing, and the RESTART_GRACE_S
deadman path is only reached AFTER a successful HEAD-change observation.

Same hazard with the breaker's ``pkill``: a hung pkill (a zombie subprocess
wedging /proc) would stall ``_cycle`` for the whole run, defeating the
circuit-breaker's whole point. These tests pin both contracts so a future
edit that drops the timeout keyword is caught in CI.
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from paper_trader import runner


def test_kill_stale_claude_passes_timeout_kwarg():
    """The pkill subprocess.run call must include a finite ``timeout=`` kwarg
    so a wedged pkill cannot stall the cycle."""
    captured: list[dict] = []

    def fake_run(cmd, **kwargs):
        captured.append({"cmd": cmd, "kwargs": kwargs})
        class FakeR:
            returncode = 1
        return FakeR()

    with patch.object(subprocess, "run", side_effect=fake_run):
        runner._kill_stale_claude()

    # Two pkill patterns (opus + sonnet) → two calls; both must carry timeout.
    assert len(captured) == 2, f"expected 2 pkill calls, got {len(captured)}"
    for c in captured:
        timeout = c["kwargs"].get("timeout")
        assert timeout is not None, (
            f"pkill call missing timeout kwarg: {c['cmd']}"
        )
        assert isinstance(timeout, (int, float)) and timeout > 0, (
            f"pkill timeout must be a positive number, got {timeout!r}"
        )
        # Sanity: the constant should be the module-level _PKILL_TIMEOUT_S.
        assert timeout == runner._PKILL_TIMEOUT_S


def test_kill_stale_claude_survives_pkill_timeout():
    """A TimeoutExpired from one pkill must not prevent the second pkill
    from running, and must not raise from ``_kill_stale_claude``."""
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        # Find the pattern arg — pkill ... -f <pattern>
        idx = cmd.index("-f") if "-f" in cmd else -1
        pattern = cmd[idx + 1] if idx >= 0 else ""
        calls.append(pattern)
        if "opus" in pattern:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))
        class FakeR:
            returncode = 1
        return FakeR()

    with patch.object(subprocess, "run", side_effect=fake_run):
        runner._kill_stale_claude()  # must NOT raise

    # Opus path raised TimeoutExpired; sonnet path must still have been tried.
    opus_seen = any("opus" in p for p in calls)
    sonnet_seen = any("sonnet" in p for p in calls)
    assert opus_seen and sonnet_seen, (
        f"both patterns must be attempted even after timeout: {calls}"
    )


def test_git_probe_timeout_constant_is_finite_and_positive():
    """The module-level _GIT_PROBE_TIMEOUT_S must be a finite positive
    number — the entire point of the constant is to bound a wedged git."""
    t = runner._GIT_PROBE_TIMEOUT_S
    assert isinstance(t, (int, float))
    assert t > 0
    # Sanity: should be < a minute (a git rev-parse on a healthy tree is
    # sub-second; longer than 60s is no longer a "timeout" in any useful sense).
    assert t < 60.0


def test_pkill_timeout_constant_is_finite_and_positive():
    t = runner._PKILL_TIMEOUT_S
    assert isinstance(t, (int, float))
    assert t > 0
    # pkill is normally microseconds; >30s is unreasonable.
    assert t < 30.0


def test_git_watcher_source_passes_timeout_to_check_output():
    """The git-watcher's two ``subprocess.check_output`` call sites in
    runner.py both pass ``timeout=_GIT_PROBE_TIMEOUT_S``.

    Static check rather than dynamic — running _git_watcher() spawns a
    long-lived daemon thread that sleeps 120s before the second probe,
    so a runtime mock would either hang or race the sleep. The behaviour
    we want to pin is *source-level*: every check_output(['git', ...])
    site must carry the timeout kwarg.
    """
    import inspect
    src = inspect.getsource(runner._git_watcher)
    # Two check_output sites — baseline + per-iteration.
    assert src.count("subprocess.check_output") == 2, (
        "test stale: _git_watcher used to have exactly 2 check_output sites; "
        "if a third was added, extend this test rather than weakening it."
    )
    # Every one must have an explicit timeout kwarg.
    assert src.count("timeout=_GIT_PROBE_TIMEOUT_S") >= 2, (
        "git check_output sites missing timeout=_GIT_PROBE_TIMEOUT_S — a "
        "wedged git would hang the watcher forever (the entire restart "
        "feature would silently die under load)"
    )
