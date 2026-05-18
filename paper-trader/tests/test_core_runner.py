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


class TestDeferredRestartOverdue:
    """The git-watcher deadman predicate. The watcher requests a graceful
    deferred restart, then force-exits if the main loop is wedged and never
    honors it (observed live: a 3-day-uptime runner still on stale code with
    a committed fix never deployed). `_deferred_restart_overdue` is the pure
    decision over monotonic clocks."""

    def test_not_overdue_before_request(self):
        # No restart requested yet → never overdue, whatever the clock says.
        assert runner._deferred_restart_overdue(None, 1e9) is False

    def test_not_overdue_within_grace(self):
        # Requested at t=100; only 599s elapsed (< 600 grace) → wait, the
        # main loop may still honor it gracefully.
        assert runner._deferred_restart_overdue(100.0, 100.0 + 599.0,
                                                grace_s=600.0) is False

    def test_overdue_at_exactly_grace(self):
        # Boundary is inclusive: exactly grace_s elapsed → force-exit (the
        # loop has had its full healthy-cycle budget and then some).
        assert runner._deferred_restart_overdue(100.0, 100.0 + 600.0,
                                                grace_s=600.0) is True

    def test_overdue_well_past_grace(self):
        assert runner._deferred_restart_overdue(100.0, 100.0 + 5000.0,
                                                grace_s=600.0) is True

    def test_default_grace_is_module_constant(self):
        # Just under the real default → not overdue; at it → overdue. Guards
        # against the constant being silently dropped/renamed.
        g = runner.RESTART_GRACE_S
        assert runner._deferred_restart_overdue(0.0, g - 0.01) is False
        assert runner._deferred_restart_overdue(0.0, g) is True

    def test_grace_exceeds_worst_case_healthy_cycle(self):
        """The grace must be safely above the longest *healthy* cycle so a
        slow-but-live loop is never force-killed. Worst case ≈ the strategy
        claude budgets (180 + 45 + 60) + the 180s watcher poll cadence."""
        from paper_trader import strategy
        worst_healthy = (strategy.DECISION_TIMEOUT_S
                         + strategy.RETRY_TIMEOUT_S
                         + strategy.FALLBACK_TIMEOUT_S
                         + 180)
        assert runner.RESTART_GRACE_S > worst_healthy


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

    # ── future-marker hardening (clock stepped backward after a save) ──
    def test_restore_clamps_future_last_hourly_so_hourly_is_not_muted(
        self, monkeypatch, tmp_path
    ):
        """A clock step BACKWARD after a `_save_runner_state` leaves
        `last_hourly_iso` in the future. Restoring it verbatim makes
        `(now - _last_hourly) < 3600` true for up to (skew + 1h), so
        `_maybe_hourly` MUTES the operator's primary monitoring surface with
        zero signal. The restore must clamp a future marker back to now so the
        normal 1h cadence resumes."""
        p = tmp_path / "runner_state.json"
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        # Persist a marker 2h in the future relative to the (patched) boot now.
        future = _ny(2026, 5, 14, 14, 0)
        p.write_text(f'{{"last_hourly_iso": "{future.isoformat()}"}}')
        monkeypatch.setattr(runner, "_last_hourly", None)

        boot_now = _ny(2026, 5, 14, 12, 0)
        _patch_now(monkeypatch, boot_now)
        runner._restore_runner_state()
        # Clamped to now — NOT left 2h in the future.
        assert runner._last_hourly == boot_now

        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        # 30 min after boot: still inside the hour — must NOT fire.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 12, 30))
        runner._maybe_hourly()
        assert calls == []
        # 1h01m after boot: WITHOUT the clamp _last_hourly would be 14:00 and
        # this would still be muted (12:01 < 14:00). With the clamp it fires.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 13, 1))
        runner._maybe_hourly()
        assert calls == [1], "hourly stayed muted — future marker not clamped"

    def test_restore_drops_future_daily_close_sent_for(
        self, monkeypatch, tmp_path
    ):
        """A `daily_close_sent_for` strictly after today (NY) is non-physical
        (you cannot have sent a close for a future calendar day). Restoring it
        would suppress that day's real close once the clock reaches it — drop
        it (treat as not-sent)."""
        p = tmp_path / "runner_state.json"
        # Boot "today" (NY) is 2026-05-14 (Thu); sidecar claims a close for
        # 05-15 (Fri — a real trading day, so the "fires when it arrives"
        # leg isn't masked by the weekend guard).
        p.write_text('{"daily_close_sent_for": "2026-05-15"}')
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)
        _patch_now(monkeypatch, _ny(2026, 5, 14, 12, 0))
        runner._restore_runner_state()
        assert runner._daily_close_sent_for is None, \
            "future daily_close_sent_for restored — would suppress a real close"

        # And the close for that future day must actually fire when it arrives.
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        _patch_now(monkeypatch, _ny(2026, 5, 15, 16, 10))
        runner._maybe_daily_close()
        assert calls == [1]

    def test_restore_keeps_today_and_past_daily_close(
        self, monkeypatch, tmp_path
    ):
        """The dedup behaviour must NOT regress: a `daily_close_sent_for` that
        is today (NY) or in the past is a legitimate marker and is restored
        verbatim (today → still suppresses today's duplicate close)."""
        p = tmp_path / "runner_state.json"
        p.write_text('{"daily_close_sent_for": "2026-05-14"}')
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        monkeypatch.setattr(runner, "_daily_close_sent_for", None)
        _patch_now(monkeypatch, _ny(2026, 5, 14, 17, 0))
        runner._restore_runner_state()
        assert runner._daily_close_sent_for == "2026-05-14"
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == [], "today's close double-posted (dedup regressed)"

    def test_restore_past_last_hourly_unchanged(self, monkeypatch, tmp_path):
        """A non-future (overdue) `last_hourly_iso` is restored verbatim — the
        clamp must only touch genuinely future markers, never an old one."""
        p = tmp_path / "runner_state.json"
        past = _ny(2026, 5, 14, 9, 0)
        p.write_text(f'{{"last_hourly_iso": "{past.isoformat()}"}}')
        monkeypatch.setattr(runner, "_STATE_PATH", p)
        monkeypatch.setattr(runner, "_last_hourly", None)
        _patch_now(monkeypatch, _ny(2026, 5, 14, 12, 0))
        runner._restore_runner_state()
        assert runner._last_hourly == past

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


class TestSingletonLock:
    """`_acquire_singleton_lock` — the single-instance guard added 2026-05-17
    after observing TWO concurrent runner.py processes (an orphaned manual
    launch + the systemd unit) double-trading the same paper book.

    Exercises the real fcntl.flock primitive on a tmp lockfile (no fork: a
    second open()+flock from the same process gets a distinct open-file
    description, so LOCK_NB contends exactly as a second process would)."""

    def test_first_acquire_succeeds_and_writes_pid(self, tmp_path):
        lp = tmp_path / "pt.lock"
        lk = runner._acquire_singleton_lock(lp)
        try:
            assert lk.status == "acquired"
            assert lk.handle is not None
            import os
            assert lk.holder_pid == os.getpid()
            assert lp.read_text().strip() == str(os.getpid())
        finally:
            lk.handle.close()

    def test_second_acquire_is_busy_with_holder_pid(self, tmp_path):
        import os
        lp = tmp_path / "pt.lock"
        first = runner._acquire_singleton_lock(lp)
        try:
            assert first.status == "acquired"
            # A would-be second runner.
            second = runner._acquire_singleton_lock(lp)
            assert second.status == "busy"
            assert second.handle is None
            # It must surface the live holder's PID for an actionable log.
            assert second.holder_pid == os.getpid()
        finally:
            first.handle.close()

    def test_lock_is_released_on_holder_death_then_reacquirable(self, tmp_path):
        """Robust-across-restart: closing the fd (== the holder process dying)
        frees the kernel flock, so the next start re-acquires cleanly. This is
        exactly the stale-PID-file failure a naive guard would introduce."""
        lp = tmp_path / "pt.lock"
        first = runner._acquire_singleton_lock(lp)
        assert first.status == "acquired"
        # Simulate the prior process exiting.
        first.handle.close()
        second = runner._acquire_singleton_lock(lp)
        try:
            assert second.status == "acquired", "stale lock blocked a restart"
            assert second.handle is not None
        finally:
            second.handle.close()

    def test_unwritable_lock_dir_degrades_open_not_closed(self, tmp_path):
        """Fail-OPEN: if the lock plumbing is unusable the sole runner must
        still start. A path whose parent component is a regular file makes
        `.parent.mkdir` raise → status 'degraded', never 'busy'."""
        afile = tmp_path / "afile"
        afile.write_text("x")
        lp = afile / "sub" / "pt.lock"   # afile is a file, not a dir
        lk = runner._acquire_singleton_lock(lp)
        assert lk.status == "degraded"
        assert lk.handle is None

    def test_main_exits_before_store_when_busy(self, monkeypatch):
        """The wiring lock: a 'busy' result must `sys.exit(1)` BEFORE the
        store / dashboard / ONLINE ping are touched (a second runner must not
        even mark-to-market the shared book)."""
        called = {"store": False}

        def _no_store():
            called["store"] = True
            raise AssertionError("get_store() reached despite busy lock")

        monkeypatch.setattr(runner, "get_store", _no_store)
        monkeypatch.setattr(
            runner, "_acquire_singleton_lock",
            lambda *a, **k: runner.SingletonLock(None, "busy", 4242),
        )
        with pytest.raises(SystemExit) as ei:
            runner.main()
        assert ei.value.code == 1
        assert called["store"] is False

    def test_main_degraded_continues_past_lock(self, monkeypatch):
        """A 'degraded' lock must NOT exit — it proceeds (fail-open). We stop
        it right after the guard by making get_store raise a sentinel."""
        class _Sentinel(Exception):
            pass

        def _boom():
            raise _Sentinel()

        monkeypatch.setattr(runner, "get_store", _boom)
        monkeypatch.setattr(
            runner, "_acquire_singleton_lock",
            lambda *a, **k: runner.SingletonLock(None, "degraded", None),
        )
        # Reached get_store → it did NOT sys.exit on degraded. SystemExit
        # would mean the fail-open contract broke.
        with pytest.raises(_Sentinel):
            runner.main()


class TestRecheckSingletonLock:
    """`_recheck_singleton_lock` — the degraded-runner upgrade/exit retry
    added 2026-05-18 after observing a degraded runner (PID 1255030, lock
    plumbing unusable at boot) and a properly-locked runner (PID 1465599)
    BOTH cycling the same $1000 paper book for ~12h. The boot-time guard
    fails open by design (invariant #19); without a periodic re-check the
    two-runner window stays open forever.

    Exercises the real fcntl.flock primitive on a tmp lockfile (a second
    open()+flock from the same process contends exactly as a 2nd process)."""

    @pytest.fixture(autouse=True)
    def _reset_lock_globals(self, monkeypatch):
        monkeypatch.setattr(runner, "_lock_status", "degraded")
        monkeypatch.setattr(runner, "_lock_holder_pid", None)
        monkeypatch.setattr(runner, "_SINGLETON_LOCK_FH", None)
        monkeypatch.setattr(runner, "_degraded_recheck_warned", False)

    def test_noop_when_already_acquired(self, tmp_path):
        """The load-bearing guard: once we hold the lock, recheck must NOT
        re-open+re-flock the same file — a 2nd fd in the same process is
        denied by our OWN flock and would mis-read as `busy`, exiting the
        real holder. So `_lock_status == "acquired"` is an early no-op even
        when the path's flock is held."""
        lp = tmp_path / "pt.lock"
        held = runner._acquire_singleton_lock(lp)
        try:
            assert held.status == "acquired"
            runner._lock_status = "acquired"
            # Must return without raising SystemExit and without flipping state.
            runner._recheck_singleton_lock(lp)
            assert runner._lock_status == "acquired"
        finally:
            held.handle.close()

    def test_still_degraded_does_not_exit(self, tmp_path):
        """Plumbing STILL unusable (parent is a regular file → mkdir/open
        raise → degraded). Invariant #19: a degraded re-check must keep the
        sole trader running, never exit."""
        afile = tmp_path / "afile"
        afile.write_text("x")
        lp = afile / "sub" / "pt.lock"  # afile is a file, not a dir
        runner._lock_status = "degraded"
        runner._recheck_singleton_lock(lp)  # must not raise SystemExit
        assert runner._lock_status == "degraded"
        assert runner._SINGLETON_LOCK_FH is None

    def test_upgrades_when_lock_becomes_free(self, tmp_path):
        """Plumbing recovered and NO other trader holds it → upgrade in
        place: status flips to acquired and the handle is retained so the
        flock is held for life."""
        import os
        lp = tmp_path / "pt.lock"
        runner._lock_status = "degraded"
        runner._recheck_singleton_lock(lp)
        try:
            assert runner._lock_status == "acquired"
            assert runner._SINGLETON_LOCK_FH is not None
            assert runner._lock_holder_pid == os.getpid()
            assert lp.read_text().strip() == str(os.getpid())
        finally:
            if runner._SINGLETON_LOCK_FH is not None:
                runner._SINGLETON_LOCK_FH.close()

    def test_exits_when_another_trader_acquired_lock(self, tmp_path):
        """The bug this closes: plumbing recovered and ANOTHER live trader
        now holds the lock → the degraded runner is double-trading the
        shared book and must exit(1) so the locked instance is sole writer."""
        import os
        lp = tmp_path / "pt.lock"
        other = runner._acquire_singleton_lock(lp)  # the legit locked trader
        try:
            assert other.status == "acquired"
            runner._lock_status = "degraded"
            with pytest.raises(SystemExit) as ei:
                runner._recheck_singleton_lock(lp)
            assert ei.value.code == 1
            # We must NOT have stolen/altered the holder's state.
            assert runner._lock_status == "degraded"
            assert other.handle is not None
            assert lp.read_text().strip() == str(os.getpid())
        finally:
            other.handle.close()

    def test_singleton_lock_state_accessor(self, monkeypatch):
        monkeypatch.setattr(runner, "_lock_status", "degraded")
        monkeypatch.setattr(runner, "_lock_holder_pid", None)
        st = runner.singleton_lock_state()
        assert st == {"status": "degraded", "holder_pid": None,
                      "have_lock": False, "degraded": True}
        monkeypatch.setattr(runner, "_lock_status", "acquired")
        monkeypatch.setattr(runner, "_lock_holder_pid", 4242)
        st = runner.singleton_lock_state()
        assert st == {"status": "acquired", "holder_pid": 4242,
                      "have_lock": True, "degraded": False}
