"""Tests for paper_trader.runner — the hourly + daily-close gating logic.

The runner is hard to unit-test as a whole because it runs an infinite loop,
but the gating helpers _maybe_hourly() and _maybe_daily_close() are pure
functions over module state + the wall clock. We patch the clock and the
reporter so each test deterministically reaches a single decision branch.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import runner

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _ny(year, month, day, hour, minute):
    """Build a UTC datetime corresponding to a given NY wall-clock time."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


@pytest.fixture(autouse=True)
def _reset_runner_state(monkeypatch, tmp_path):
    """Each test starts with a fresh module state — no prior hourly/daily fired.

    Also redirect the restart-durability sidecar into tmp so a test that
    drives `_maybe_hourly`/`_maybe_daily_close` to success can never write the
    real ``data/runner_state.json`` (offline / side-effect-free invariant).
    """
    monkeypatch.setattr(runner, "_daily_close_sent_for", None)
    monkeypatch.setattr(runner, "_last_hourly", None)
    monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "runner_state.json")


def _patch_now(monkeypatch, when):
    """Patch datetime.now inside runner so the gating sees a fixed time."""
    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return when.astimezone(tz) if tz else when

        # The module also calls datetime.fromisoformat etc. — preserve passthroughs.
        @staticmethod
        def fromisoformat(s):
            return datetime.fromisoformat(s)

    monkeypatch.setattr(runner, "datetime", _FakeDT)


class TestMaybeDailyClose:
    def test_does_not_fire_on_saturday(self, monkeypatch):
        # 2026-05-16 is a Saturday, 17:00 ET — past the trigger time but weekend.
        _patch_now(monkeypatch, _ny(2026, 5, 16, 17, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []
        # Sent-for flag must NOT advance on weekends.
        assert runner._daily_close_sent_for is None

    def test_does_not_fire_on_sunday(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 17, 17, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_does_not_fire_on_nyse_holiday(self, monkeypatch):
        # 2026-05-25 is Memorial Day — a Monday (weekday) full-market close.
        # 16:10 ET is past the trigger time, so only the holiday guard can
        # stop the spurious "DAILY CLOSE" post.
        _patch_now(monkeypatch, _ny(2026, 5, 25, 16, 10))
        assert _ny(2026, 5, 25, 16, 10).astimezone(NY).weekday() < 5  # is a weekday
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []
        # Holiday guard must not advance the sent-for flag.
        assert runner._daily_close_sent_for is None

    def test_does_not_fire_before_1605_ET(self, monkeypatch):
        # Thursday 16:04 NY — too early.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 4))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_does_not_fire_at_1500_ET(self, monkeypatch):
        # 3 PM — well before close.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 15, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_fires_at_1605_ET(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 5))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == [1]
        assert runner._daily_close_sent_for == "2026-05-14"

    def test_only_fires_once_per_day(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        runner._maybe_daily_close()
        runner._maybe_daily_close()
        assert calls == [1]

    def test_fires_again_next_day(self, monkeypatch):
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        # Fire on Thursday.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        runner._maybe_daily_close()
        # Fire on Friday — different date → fires again.
        _patch_now(monkeypatch, _ny(2026, 5, 15, 16, 10))
        runner._maybe_daily_close()
        assert len(calls) == 2

    def test_send_failure_does_not_advance_sent_for(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        # Simulate openclaw failure: send_daily_close returns False.
        monkeypatch.setattr(runner.reporter, "send_daily_close", lambda: False)
        runner._maybe_daily_close()
        # Must NOT mark today as sent, so we retry next cycle.
        assert runner._daily_close_sent_for is None


class TestMaybeHourly:
    def test_fires_when_last_hourly_none(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._last_hourly = None
        runner._maybe_hourly()
        assert calls == [1]

    def test_does_not_fire_within_3600s(self, monkeypatch):
        first_t = _ny(2026, 5, 14, 10, 0)
        runner._last_hourly = first_t
        # 30 minutes later — should NOT fire.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 30))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._maybe_hourly()
        assert calls == []

    def test_fires_after_3600s(self, monkeypatch):
        first_t = _ny(2026, 5, 14, 10, 0)
        runner._last_hourly = first_t
        # 65 minutes later — should fire.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 11, 5))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._maybe_hourly()
        assert calls == [1]

    def test_send_failure_does_not_advance_last_hourly(self, monkeypatch):
        runner._last_hourly = None
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        monkeypatch.setattr(runner.reporter, "send_hourly_summary", lambda: False)
        runner._maybe_hourly()
        # If send failed, we want to retry on the next cycle, not skip an hour.
        assert runner._last_hourly is None

    def test_send_exception_swallowed(self, monkeypatch):
        runner._last_hourly = None
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))

        def boom():
            raise RuntimeError("openclaw exploded")

        monkeypatch.setattr(runner.reporter, "send_hourly_summary", boom)
        # Must not raise; the runner is a daemon loop.
        runner._maybe_hourly()
        assert runner._last_hourly is None


class TestKillStaleClaude:
    """The auto-recovery circuit breaker reaps wedged `claude` subprocesses.

    `strategy._claude_call` always invokes the CLI as
    ``claude --model <model> --print …`` — the `--model <model>` argument
    sits *between* `claude` and `--print`. So a `pkill -f` pattern of bare
    ``claude --print`` is never a contiguous substring of the real command
    line and silently matches nothing. The breaker must therefore anchor on
    ``claude --model <family>`` for *both* the live Opus model and the
    Sonnet fallback — otherwise a wedged Sonnet-fallback child survives the
    breaker and keeps starving the decision loop in exactly the
    Opus-timeout → Sonnet-fallback path the breaker exists to recover.

    The sweep is ALSO scoped to the runner's own child processes
    (``pkill -P os.getpid()``). A host-wide ``pkill -f "claude --model
    claude-opus"`` is catastrophic collateral damage on this box: it would
    SIGTERM the hourly self-review agents, sibling review agents, and any
    operator interactive `claude` session — all of which match the same
    pattern. The decision subprocess is always a *direct* child of the
    runner, so ``-P <our pid>`` restricts the kill to exactly what the
    breaker is meant to reap. (The old argv ``["pkill", "-f", pattern]``
    that the prior assertion codified WAS the host-wide-broadcast bug — the
    AGENTS.md invariant-#16 "a test that literally codified the bug" /
    correction-not-weakening precedent.)
    """

    @staticmethod
    def _real_cmdline(model: str) -> str:
        # Mirrors the argv strategy._claude_call builds, joined the way
        # `pkill -f` matches (against the space-joined command line).
        return " ".join(
            ["claude", "--model", model, "--print",
             "--permission-mode", "bypassPermissions"]
        )

    def _captured_calls(self, monkeypatch) -> list[list[str]]:
        """Capture the full pkill argv list of every breaker invocation."""
        seen: list[list[str]] = []

        def _fake_run(argv, *a, **k):
            assert argv[0] == "pkill"
            seen.append(list(argv))

            class _R:
                returncode = 1  # "nothing matched" — harmless

            return _R()

        monkeypatch.setattr(runner.subprocess, "run", _fake_run)
        runner._kill_stale_claude()
        return seen

    def _captured_patterns(self, monkeypatch) -> list[str]:
        # The `-f` pattern is the argv token immediately after "-f".
        pats: list[str] = []
        for argv in self._captured_calls(monkeypatch):
            assert "-f" in argv, f"breaker pkill argv has no -f pattern: {argv}"
            pats.append(argv[argv.index("-f") + 1])
        return pats

    def test_patterns_match_both_opus_and_sonnet_cmdlines(self, monkeypatch):
        from paper_trader import strategy

        patterns = self._captured_patterns(monkeypatch)
        assert patterns, "circuit breaker issued no pkill patterns"

        for model in (strategy.MODEL, strategy.FALLBACK_MODEL):
            cmdline = self._real_cmdline(model)
            # `pkill -f` treats the pattern as an ERE matched against the
            # full command line; our literal patterns contain no regex
            # metacharacters, so re.search faithfully models it.
            assert any(re.search(p, cmdline) for p in patterns), (
                f"no breaker pattern matches the real command line for "
                f"{model!r}: {cmdline!r} (patterns={patterns})"
            )

    def test_regression_bare_claude_print_would_miss_sonnet(self, monkeypatch):
        # Locks the actual bug: the OLD second pattern `claude --print`
        # never matches the Sonnet-fallback command line, so a regression
        # back to it leaves the fallback zombie un-reaped.
        from paper_trader import strategy

        sonnet_cmdline = self._real_cmdline(strategy.FALLBACK_MODEL)
        assert re.search("claude --print", sonnet_cmdline) is None
        # The shipped breaker MUST match it.
        patterns = self._captured_patterns(monkeypatch)
        assert any(re.search(p, sonnet_cmdline) for p in patterns)

    def test_kill_is_scoped_to_own_child_processes(self, monkeypatch):
        """Regression lock for the host-wide-broadcast bug.

        Every breaker pkill MUST be scoped with ``-P <our pid>`` so it can
        only reap the runner's own direct claude children — never the
        hourly-review agents, sibling review agents, or an operator's
        interactive `claude` session that match the same `-f` pattern.
        A regression back to a host-wide ``["pkill", "-f", pattern]`` (no
        ``-P``) fails here.
        """
        import os as _os

        calls = self._captured_calls(monkeypatch)
        assert calls, "circuit breaker issued no pkill calls"
        own_pid = str(_os.getpid())
        for argv in calls:
            assert "-P" in argv, (
                f"breaker pkill is NOT scoped to its own children "
                f"(host-wide broadcast bug regressed): {argv}"
            )
            assert argv[argv.index("-P") + 1] == own_pid, (
                f"breaker pkill -P target is not this process: {argv}"
            )
            # `-f` (full-cmdline match) must still be present alongside `-P`.
            assert "-f" in argv, f"breaker lost its -f pattern match: {argv}"

    def test_breaker_swallows_pkill_failure(self, monkeypatch):
        # The breaker runs inside the daemon loop — a pkill OSError must
        # never propagate and kill the runner.
        def _boom(argv, *a, **k):
            raise OSError("pkill not found")

        monkeypatch.setattr(runner.subprocess, "run", _boom)
        runner._kill_stale_claude()  # must not raise


class TestRunnerStatePersistence:
    """Restart-durable report markers (2026-05-17 feature).

    `_daily_close_sent_for` / `_last_hourly` were module globals lost on
    every process restart, so a frequently-bounced runner either never sent
    an hourly summary (boot kept resetting the 1h clock) or double-posted
    the DAILY CLOSE after a post-16:05 restart. These lock the JSON-sidecar
    persistence + rehydrate that fixes both, asserting the exact restart
    behaviour — not merely "no crash".
    """

    # ── sidecar IO contract: never raises, round-trips ──
    def test_load_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "nope.json")
        assert runner._load_runner_state() == {}

    def test_load_corrupt_json_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "runner_state.json"
        p.write_text("{ this is not json")
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        assert runner._load_runner_state() == {}  # never raises

    def test_load_non_dict_json_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "runner_state.json"
        p.write_text("[1, 2, 3]")  # valid JSON, wrong shape
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        assert runner._load_runner_state() == {}

    def test_save_then_load_round_trips(self, monkeypatch, tmp_path):
        p = tmp_path / "runner_state.json"
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        when = _ny(2026, 5, 14, 11, 0)
        monkeypatch.setattr(runner, "_daily_close_sent_for", "2026-05-14")
        monkeypatch.setattr(runner, "_last_hourly", when)
        runner._save_runner_state()
        loaded = runner._load_runner_state()
        assert loaded["daily_close_sent_for"] == "2026-05-14"
        assert loaded["last_hourly_iso"] == when.isoformat()
        # No leftover temp file from the atomic write.
        assert list(tmp_path.glob("*.tmp")) == []

    def test_save_swallows_io_error(self, monkeypatch, tmp_path):
        # _STATE_PATH under a *file* (not dir) → mkdir/replace raise; the
        # daemon loop must survive a read-only data dir.
        bad_parent = tmp_path / "afile"
        bad_parent.write_text("x")
        monkeypatch.setattr(runner, "_STATE_PATH", bad_parent / "runner_state.json")
        monkeypatch.setattr(runner, "_daily_close_sent_for", "2026-05-14")
        runner._save_runner_state()  # must not raise

    # ── rehydrate contract ──
    def test_restore_no_sidecar_leaves_globals(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "nope.json")
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)
        sentinel = _ny(2026, 5, 14, 9, 30)
        monkeypatch.setattr(runner, "_last_hourly", sentinel)
        runner._restore_runner_state()
        assert runner._daily_close_sent_for is None
        assert runner._last_hourly is sentinel  # untouched

    def test_restore_rehydrates_both_markers(self, monkeypatch, tmp_path):
        p = tmp_path / "runner_state.json"
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        when = _ny(2026, 5, 14, 11, 0)
        monkeypatch.setattr(runner, "_daily_close_sent_for", "2026-05-14")
        monkeypatch.setattr(runner, "_last_hourly", when)
        runner._save_runner_state()
        # Simulate a fresh process: globals back to boot defaults.
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)
        monkeypatch.setattr(runner, "_last_hourly", None)
        runner._restore_runner_state()
        assert runner._daily_close_sent_for == "2026-05-14"
        assert runner._last_hourly == when

    def test_restore_corrupt_last_hourly_left_unchanged(self, monkeypatch, tmp_path):
        p = tmp_path / "runner_state.json"
        p.write_text('{"daily_close_sent_for": "2026-05-14", '
                      '"last_hourly_iso": "not-a-timestamp"}')
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)
        monkeypatch.setattr(runner, "_last_hourly", None)
        runner._restore_runner_state()  # must not raise
        # The good field still rehydrates; the bad one is just skipped.
        assert runner._daily_close_sent_for == "2026-05-14"
        assert runner._last_hourly is None

    # ── the two trader-visible bugs this fixes ──
    def test_restart_after_close_does_not_double_post(self, monkeypatch, tmp_path):
        """A bounce after 16:05 NY on a day the close already fired must NOT
        re-post the DAILY CLOSE (the duplication bug)."""
        monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "runner_state.json")
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        # Process A: fires the close at 16:10 NY and persists.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        runner._maybe_daily_close()
        assert calls == [1]
        # Process B (restart 50 min later, still the same NY day):
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)  # fresh proc
        runner._restore_runner_state()
        _patch_now(monkeypatch, _ny(2026, 5, 14, 17, 0))
        runner._maybe_daily_close()
        assert calls == [1], "DAILY CLOSE double-posted after a restart"

    def test_restart_does_not_starve_overdue_hourly(self, monkeypatch, tmp_path):
        """An overdue hourly (>1h since the last real send) must fire on the
        first post-restart cycle — a restart must not reset the clock and
        swallow another hour."""
        monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "runner_state.json")
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        # Process A sends an hourly at 10:00 NY, persists.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        runner._last_hourly = None
        runner._maybe_hourly()
        assert calls == [1]
        # Process B restarts 2h later. main() would boot-anchor _last_hourly
        # to *now* (which alone would starve the hourly forever under frequent
        # restarts); _restore_runner_state must override it with the real
        # 10:00 marker so the overdue summary fires this cycle.
        monkeypatch.setattr(runner, "_last_hourly", _ny(2026, 5, 14, 12, 0))  # boot anchor
        runner._restore_runner_state()
        _patch_now(monkeypatch, _ny(2026, 5, 14, 12, 0))
        runner._maybe_hourly()
        assert calls == [1, 1], "overdue hourly starved by the restart"

    def test_restart_within_hour_does_not_early_fire(self, monkeypatch, tmp_path):
        """The mirror of the above: a restart <1h after the last real hourly
        must NOT fire an early summary on boot."""
        monkeypatch.setattr(runner, "_STATE_PATH", tmp_path / "runner_state.json")
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        runner._last_hourly = None
        runner._maybe_hourly()
        assert calls == [1]
        # Restart only 20 min later.
        monkeypatch.setattr(runner, "_last_hourly", _ny(2026, 5, 14, 10, 20))
        runner._restore_runner_state()
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 20))
        runner._maybe_hourly()
        assert calls == [1], "fired an early hourly <1h after the last one"
