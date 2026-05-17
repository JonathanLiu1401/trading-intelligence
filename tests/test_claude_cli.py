"""Quota circuit-breaker invariants for core.claude_cli.

The breaker exists to stop doomed subprocess churn once the CLI reports an
org usage / rate / quota limit. These tests pin the behaviour that matters:

  * A quota-flagged failure trips the breaker and the *next* call returns None
    WITHOUT spawning a subprocess (the whole point — no wasted process).
  * A non-quota failure (e.g. JSON/parse, generic rc=1) does NOT trip it:
    transient errors must still retry next cycle.
  * The breaker self-heals once the cooldown elapses.

subprocess.run is monkeypatched throughout — no real `claude` is invoked.
"""
from __future__ import annotations

import subprocess
import types

import pytest

from core import claude_cli


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch):
    """Pretend `claude` is on PATH and start every test with the breaker shut."""
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _: "/usr/bin/claude")
    claude_cli.reset_quota_breaker()
    yield
    claude_cli.reset_quota_breaker()


def _fake_run(returncode=0, stdout="", stderr=""):
    """Build a subprocess.run replacement that records its call count."""
    calls = {"n": 0}

    def runner(*_a, **_k):
        calls["n"] += 1
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    runner.calls = calls
    return runner


def test_quota_error_trips_breaker_and_skips_next_subprocess(monkeypatch):
    runner = _fake_run(returncode=1,
                        stderr="You've hit your org's monthly usage limit")
    monkeypatch.setattr(subprocess, "run", runner)

    assert claude_cli.claude_call("p") is None          # 1st call: spawns, fails
    assert claude_cli.quota_blocked() is True
    assert claude_cli.claude_call("p") is None           # 2nd call: short-circuit
    assert claude_cli.claude_call("p") is None           # 3rd: still short-circuit

    # Only the first call ever reached subprocess.run.
    assert runner.calls["n"] == 1


def test_non_quota_failure_does_not_trip_breaker(monkeypatch):
    runner = _fake_run(returncode=1, stderr="some transient parse error")
    monkeypatch.setattr(subprocess, "run", runner)

    assert claude_cli.claude_call("p") is None
    assert claude_cli.quota_blocked() is False
    assert claude_cli.claude_call("p") is None
    # Both calls spawned — generic errors must keep retrying.
    assert runner.calls["n"] == 2


def test_rate_limit_string_also_trips(monkeypatch):
    runner = _fake_run(returncode=1, stdout="Error: rate limit exceeded")
    monkeypatch.setattr(subprocess, "run", runner)

    assert claude_cli.claude_call("p") is None
    assert claude_cli.quota_blocked() is True


def test_breaker_self_heals_after_cooldown(monkeypatch):
    runner = _fake_run(returncode=1, stderr="monthly usage limit reached")
    monkeypatch.setattr(subprocess, "run", runner)
    assert claude_cli.claude_call("p") is None
    assert claude_cli.quota_blocked() is True

    # Fast-forward past the cooldown without sleeping.
    monkeypatch.setattr(
        claude_cli.time, "time",
        lambda: claude_cli._quota_blocked_until + 1,
    )
    assert claude_cli.quota_blocked() is False

    ok = _fake_run(returncode=0, stdout="result body")
    monkeypatch.setattr(subprocess, "run", ok)
    assert claude_cli.claude_call("p") == "result body"
    assert ok.calls["n"] == 1


def test_success_passes_through_unchanged(monkeypatch):
    runner = _fake_run(returncode=0, stdout="  hello  ")
    monkeypatch.setattr(subprocess, "run", runner)
    assert claude_cli.claude_call("p") == "hello"
    assert claude_cli.quota_blocked() is False
