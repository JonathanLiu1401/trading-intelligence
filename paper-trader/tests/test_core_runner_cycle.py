"""Regression locks for paper_trader.runner._cycle() — the post-decision
report-dispatch fan-out.

`_maybe_hourly` / `_maybe_daily_close` are exhaustively covered in
test_core_runner.py, but `_cycle()` itself — the branch that turns a
strategy.decide() summary into Discord traffic — had **zero** direct
coverage despite carrying real conditional logic that a refactor could
silently break:

  * FILLED gates BOTH the trade alert AND the decision log.
  * a non-FILLED status (HOLD / NO_DECISION / BLOCKED) must stay silent
    AND must not even query the store (the outer guard short-circuits).
  * `auto_exits` is an orthogonal channel: it fires its own `_send`
    line(s) and is independent of the FILLED trade-alert/decision-log
    gate. `strategy.decide()` currently hard-codes `auto_exits = []`
    (Opus has full autonomy — invariant #12), so this branch is *dead
    today on purpose*; the lock exists so re-enabling auto-exits later
    is a deliberate, tested change rather than an accidental one. Do
    not delete these cases as "unreachable".
  * the `if trades and status == FILLED` guard: a FILLED cycle whose
    `recent_trades(1)` came back empty must skip the trade alert but
    STILL emit the decision log.
  * every reporter fault is swallowed — `_cycle` runs inside the daemon
    `while True:` loop, so a raised exception there would kill the
    trader.

Everything is monkeypatched against the names bound *in the runner
module* (runner.strategy.decide / runner.get_store / runner.reporter.*),
never the originals — `_cycle` resolves them through those bindings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import runner


class _FakeStore:
    """Minimal store stand-in. Records whether recent_trades was queried so
    a test can prove the outer guard short-circuited before touching it."""

    def __init__(self, trades):
        self._trades = trades
        self.recent_trades_calls = 0

    def recent_trades(self, n):
        self.recent_trades_calls += 1
        return list(self._trades[:n])


@pytest.fixture
def spy(monkeypatch):
    """Patch decide()/get_store()/reporter.* and capture every dispatch."""
    calls = {
        "trade_alert": [],
        "decision_log": [],
        "send": [],
    }

    def _set_summary(summary, trades=None):
        store = _FakeStore(trades or [])
        monkeypatch.setattr(runner.strategy, "decide", lambda: summary)
        monkeypatch.setattr(runner, "get_store", lambda: store)
        # send_trade_alert now accepts ``snapshot=`` / ``store=`` kwargs (the
        # post-trade book-impact line); stub must accept-and-ignore them so
        # the call signature evolution does not break this regression lock.
        monkeypatch.setattr(runner.reporter, "send_trade_alert",
                            lambda t, **_: calls["trade_alert"].append(t) or True)
        monkeypatch.setattr(runner.reporter, "send_decision_log",
                            lambda s: calls["decision_log"].append(s) or True)
        monkeypatch.setattr(runner.reporter, "_send",
                            lambda m: calls["send"].append(m) or True)
        return store

    return calls, _set_summary


class TestCycleDispatch:
    def test_filled_sends_trade_alert_and_decision_log(self, spy):
        calls, setup = spy
        trade = {"id": 7, "ticker": "NVDA", "action": "BUY"}
        summary = {"status": "FILLED", "decision": {"action": "BUY"}}
        store = setup(summary, trades=[trade])

        runner._cycle()

        # The most-recent trade (recent_trades(1)[0]) is the one alerted.
        assert calls["trade_alert"] == [trade]
        assert calls["decision_log"] == [summary]
        assert store.recent_trades_calls == 1

    def test_hold_is_silent_and_never_touches_store(self, spy):
        calls, setup = spy
        summary = {"status": "HOLD", "decision": {"action": "HOLD"}}
        store = setup(summary, trades=[{"id": 1, "ticker": "MU"}])

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["decision_log"] == []
        assert calls["send"] == []
        # Outer guard is False → recent_trades must not be queried at all.
        assert store.recent_trades_calls == 0

    def test_no_decision_is_silent(self, spy):
        calls, setup = spy
        summary = {"status": "NO_DECISION", "decision": None}
        setup(summary, trades=[])

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["decision_log"] == []
        assert calls["send"] == []

    def test_blocked_is_silent(self, spy):
        calls, setup = spy
        summary = {"status": "BLOCKED", "decision": {"action": "SELL"}}
        setup(summary, trades=[])

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["decision_log"] == []

    def test_auto_exits_fire_independently_of_filled_gate(self, spy):
        """auto_exits opens the outer guard but does NOT imply FILLED:
        each exit emits its own alert, while the decision-log stays gated
        on status == FILLED.

        Free-text labels like "SL NVDA -8%" / "TP MU +12%" (no matching
        SELL trade with HARD_SL/HARD_TP reason) fall back to the bare
        `**AUTO RISK EXIT** TICKER` text. The rich-alert path is
        exercised separately in
        ``test_auto_exit_with_matching_hard_sl_trade_emits_rich_alert``."""
        calls, setup = spy
        summary = {"status": "HOLD", "auto_exits": ["SL NVDA -8%", "TP MU +12%"]}
        store = setup(summary, trades=[{"id": 1, "ticker": "NVDA"}])

        runner._cycle()

        assert calls["send"] == [
            "**AUTO RISK EXIT** `SL NVDA -8%`",
            "**AUTO RISK EXIT** `TP MU +12%`",
        ]
        # status != FILLED → neither the trade alert nor the decision log.
        assert calls["trade_alert"] == []
        assert calls["decision_log"] == []
        # Outer guard opened by auto_exits → store *is* consulted.
        assert store.recent_trades_calls == 1

    def test_auto_exit_with_matching_hard_sl_trade_emits_rich_alert(self, spy):
        """Production-shape auto_exits=["MU"] with the matching SELL trade
        in the store: the trader gets a FULL trade alert (qty/price/value/
        reason + post-trade book impact) instead of bare
        AUTO RISK EXIT MU. This is the new behavior — prior to this change
        a live HARD_TP that freed $1156.35 emitted zero detail to Discord."""
        calls, setup = spy
        hard_tp_trade = {
            "id": 15, "ticker": "MU", "action": "SELL", "qty": 1.3,
            "price": 889.5, "value": 1156.35,
            "reason": "HARD_TP: price 889.50 >= threshold 773.31",
        }
        summary = {"status": "HOLD", "auto_exits": ["MU"]}
        store = setup(summary, trades=[hard_tp_trade])

        runner._cycle()

        # The matching SELL/HARD_TP trade is alerted via the rich path.
        assert calls["trade_alert"] == [hard_tp_trade]
        # No bare-text fallback fired.
        assert calls["send"] == []
        assert calls["decision_log"] == []
        assert store.recent_trades_calls == 1

    def test_auto_exit_with_no_matching_trade_falls_back_to_bare_text(self, spy):
        """When the auto_exit ticker has no matching HARD_SL/HARD_TP SELL
        in the recent trades (an older trade evicted, or a discretionary
        SELL that happened to share the ticker), the bare-text fallback
        keeps the operator informed — byte-identical to the pre-change
        behavior."""
        calls, setup = spy
        # Discretionary SELL (reason is NOT a HARD_* marker) and a BUY —
        # neither qualifies as a hard exit even though the ticker matches.
        unrelated = [
            {"id": 9, "ticker": "MU", "action": "SELL", "qty": 1.0,
             "price": 700.0, "value": 700.0, "reason": "discretionary trim"},
            {"id": 8, "ticker": "MU", "action": "BUY", "qty": 1.3,
             "price": 750.0, "value": 975.0, "reason": "earnings setup"},
        ]
        summary = {"status": "HOLD", "auto_exits": ["MU"]}
        setup(summary, trades=unrelated)

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["send"] == ["**AUTO RISK EXIT** `MU`"]

    def test_auto_exit_plus_filled_emits_both_rich_alerts(self, spy):
        """A cycle with BOTH a FILLED decision-trade AND an auto-exit must
        emit TWO rich alerts: the FILLED's (trades[0], the newest) AND the
        auto-exit's matching HARD_SL/HARD_TP trade. The trader needs to
        see both fills, not just the FILLED one. Decision log still fires
        once (gated on status==FILLED)."""
        calls, setup = spy
        filled_trade = {
            "id": 20, "ticker": "NVDA", "action": "BUY", "qty": 0.5,
            "price": 900.0, "value": 450.0, "reason": "earnings setup",
        }
        hard_sl_trade = {
            "id": 19, "ticker": "SOXL", "action": "SELL", "qty": 5.0,
            "price": 30.0, "value": 150.0,
            "reason": "HARD_SL: price 30.00 <= threshold 30.20",
        }
        summary = {
            "status": "FILLED", "decision": {"action": "BUY"},
            "auto_exits": ["SOXL"],
            "snapshot": {"cash": 850.0, "total_value": 1300.0, "positions": []},
        }
        setup(summary, trades=[filled_trade, hard_sl_trade])

        runner._cycle()

        # Order: FILLED's trade first (trades[0]), then the auto-exit's.
        assert calls["trade_alert"] == [filled_trade, hard_sl_trade]
        assert calls["decision_log"] == [summary]
        assert calls["send"] == []

    def test_auto_exit_only_matches_hard_sl_or_hard_tp_reason(self, spy):
        """A SELL whose reason starts with neither HARD_SL nor HARD_TP must
        NOT be promoted to a rich alert. Guards against future discretionary
        sells that happen to share the auto_exit ticker silently masquerading
        as a hard-exit alert."""
        calls, setup = spy
        # SELL with the SAME ticker but reason is a custom string.
        rebalance = {
            "id": 12, "ticker": "TQQQ", "action": "SELL", "qty": 2.0,
            "price": 80.0, "value": 160.0, "reason": "rebalance into cash",
        }
        summary = {"status": "HOLD", "auto_exits": ["TQQQ"]}
        setup(summary, trades=[rebalance])

        runner._cycle()

        # No rich alert; falls back to bare text.
        assert calls["trade_alert"] == []
        assert calls["send"] == ["**AUTO RISK EXIT** `TQQQ`"]

    def test_filled_with_empty_trades_skips_alert_keeps_decision_log(self, spy):
        """The `if trades and status == FILLED` guard: an empty
        recent_trades(1) must suppress the trade alert but the decision
        log (gated only on FILLED) must still fire."""
        calls, setup = spy
        summary = {"status": "FILLED", "decision": {"action": "BUY"}}
        setup(summary, trades=[])

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["decision_log"] == [summary]

    def test_reporter_exception_is_swallowed(self, spy, monkeypatch):
        """A reporter fault must never escape _cycle — it runs inside the
        daemon `while True:` loop and an unhandled raise would kill the
        live trader."""
        calls, setup = spy
        summary = {"status": "FILLED", "decision": {"action": "BUY"}}
        setup(summary, trades=[{"id": 1, "ticker": "NVDA"}])

        def boom(_):
            raise RuntimeError("openclaw exploded")

        # send_decision_log raises *after* the trade alert already fired.
        # Re-patch via monkeypatch (not a raw attribute write) so `boom`
        # is reverted after this test and cannot leak into other modules'
        # reporter import.
        monkeypatch.setattr(runner.reporter, "send_decision_log", boom)

        # Must not raise.
        runner._cycle()
        assert calls["trade_alert"] == [{"id": 1, "ticker": "NVDA"}]

    def test_decide_returning_no_status_key_is_silent(self, spy):
        """summary.get('status') is None when the key is absent — must be
        treated as not-FILLED, not crash on a missing key."""
        calls, setup = spy
        setup({}, trades=[{"id": 1, "ticker": "NVDA"}])

        runner._cycle()

        assert calls["trade_alert"] == []
        assert calls["decision_log"] == []
        assert calls["send"] == []


class TestNoDecisionCause:
    """Pin the cause-suffix the breaker alert body carries — wrong cause
    text means the operator misdiagnoses (e.g. blames Anthropic for a
    host-saturation problem they can fix themselves)."""

    def test_quota_takes_priority(self):
        cause = runner._no_decision_cause(
            {"quota_exhausted": True, "host_saturated": True, "raw": "ignored"}
        )
        assert cause == "quota/usage limit"

    def test_host_saturated_over_raw(self):
        cause = runner._no_decision_cause(
            {"quota_exhausted": False, "host_saturated": True, "raw": "X"}
        )
        assert "host saturated" in cause

    def test_raw_first_line_returned_when_no_flags(self):
        cause = runner._no_decision_cause(
            {"quota_exhausted": False, "host_saturated": False,
             "raw": "parse_failed: <prose>\nmore lines\nhere"})
        assert cause.startswith("raw response (first line)")
        assert "parse_failed" in cause
        assert "more lines" not in cause          # only first line surfaces

    def test_empty_summary_yields_empty_string(self):
        assert runner._no_decision_cause({}) == ""

    def test_non_string_raw_yields_empty_string(self):
        assert runner._no_decision_cause({"raw": None}) == ""
        assert runner._no_decision_cause({"raw": 42}) == ""

    def test_raw_first_line_truncated_to_120_chars(self):
        long = "A" * 500
        cause = runner._no_decision_cause({"raw": long})
        # 120-char cap from the slice — body fits the Discord alert
        assert cause.endswith("A" * 120)
        assert len(cause) < 200

    def test_last_claude_fail_used_when_raw_is_none(self):
        """When raw is None (timeout/empty/CLI), the strategy's per-call
        ``last_claude_fail`` tag becomes the cause — previously this path
        returned ``""`` and the breaker alert went out with no diagnostic."""
        cause = runner._no_decision_cause(
            {"raw": None, "last_claude_fail": "timeout"})
        assert "timeout" in cause
        assert cause.startswith("claude no-response")

    def test_last_claude_fail_each_bucket_surfaces(self):
        """Cover the five distinct strategy cause codes."""
        for code in ("timeout", "nonzero_rc", "empty_stdout", "cli_missing",
                     "exception"):
            cause = runner._no_decision_cause(
                {"raw": None, "last_claude_fail": code})
            assert code in cause

    def test_whitespace_only_raw_falls_through_to_last_claude_fail(self):
        """A whitespace-only ``raw`` carries no diagnostic value (the CLI
        streamed an empty line then disconnected). Historically this took
        the raw-response branch and returned ``"raw response (first line): "``
        with an empty body — masking the per-call ``last_claude_fail`` tag the
        breaker alert exists to surface. The fix strips first and only emits
        the raw-response branch when something prose-y remains.
        """
        for ws in ("", "   ", "\n", "  \n  \t  ", "\n\n\n"):
            cause = runner._no_decision_cause(
                {"raw": ws, "last_claude_fail": "empty_stdout"})
            assert cause == "claude no-response (empty_stdout)", (
                f"whitespace raw={ws!r} did not fall through to "
                f"last_claude_fail; got {cause!r}"
            )

    def test_whitespace_only_raw_empty_when_no_fail_tag(self):
        """And when both ``raw`` is whitespace AND ``last_claude_fail`` is
        absent, the cause is the empty string — not a misleading
        ``raw response (first line): `` line with no payload."""
        assert runner._no_decision_cause({"raw": "   \n  "}) == ""
        assert runner._no_decision_cause(
            {"raw": "", "last_claude_fail": ""}) == ""


class TestCircuitBreakerAlert:
    """The breaker historically fired silently — stdout WARNING only.
    These tests pin the Discord-side dispatch + the dedupe latch (matches
    the quota alarm's `_quota_alert_active` discipline)."""

    @pytest.fixture
    def breaker_setup(self, monkeypatch):
        # Reset module globals between tests so state doesn't leak.
        monkeypatch.setattr(runner, "_consecutive_no_decisions", 0)
        monkeypatch.setattr(runner, "_breaker_alert_active", False)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        # Reset wedge-elapsed tracker so prior tests can't leak a stale ts.
        monkeypatch.setattr(runner, "_no_decision_first_ts", None)
        # Session outage counters — module globals tracked across the
        # session. Reset so per-test assertions on the COUNT (the new
        # "delivered-only" edge counter) are deterministic.
        monkeypatch.setattr(runner, "_breaker_outage_count", 0)
        monkeypatch.setattr(runner, "_quota_outage_count", 0)
        alerts = {"fired": [], "send": []}
        # Capture elapsed_s too — runner now passes it as a kwarg so the
        # Discord body carries the real wall-clock wedge duration.
        # ``store`` is also accepted (runner now passes the live store so
        # the alert body carries a one-line book-exposure snapshot) but
        # we don't assert on it from this fixture's call log — the
        # exposure plumbing is exercised directly in test_core_reporter.py.
        monkeypatch.setattr(
            runner.reporter, "send_breaker_fired_alert",
            lambda n, cause="", *, elapsed_s=None, store=None: (
                alerts["fired"].append((n, cause, elapsed_s)) or True
            ),
        )
        monkeypatch.setattr(runner.reporter, "_send",
                            lambda m: alerts["send"].append(m) or True)
        # No-op the actual subprocess kill — we only care about the alert.
        monkeypatch.setattr(runner, "_kill_stale_claude", lambda: None)
        monkeypatch.setattr(runner, "get_store", lambda: _FakeStore([]))
        return alerts

    def _run_no_decision_cycles(self, n, monkeypatch, summary_extra=None):
        extra = summary_extra or {}
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None, **extra})
        for _ in range(n):
            runner._cycle()

    def test_breaker_fires_alert_at_threshold(self, breaker_setup, monkeypatch):
        """5 consecutive NO_DECISION cycles fire ONE Discord alert."""
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert len(breaker_setup["fired"]) == 1
        n, cause, elapsed_s = breaker_setup["fired"][0]
        assert n == runner.CONSECUTIVE_NO_DECISION_LIMIT
        # elapsed_s is wall-clock seconds since the first NO_DECISION in this
        # run — small (cycles run back-to-back in the test) but must be int,
        # never None, when the breaker fires within the same process lifetime.
        assert isinstance(elapsed_s, int) and elapsed_s >= 0
        # Latch is set so the NEXT breaker fire stays silent.
        assert runner._breaker_alert_active is True
        # Counter resets after the breaker fires.
        assert runner._consecutive_no_decisions == 0

    def test_breaker_dedupes_within_outage(self, breaker_setup, monkeypatch):
        """A second breaker fire inside the same outage stays silent —
        the alert latch prevents flooding the channel on a long wedge."""
        # First breaker fire — alert
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        # Second breaker fire — latched, no NEW alert
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert len(breaker_setup["fired"]) == 1

    def test_real_decision_clears_latch_and_re_arms(self, breaker_setup,
                                                    monkeypatch):
        """A real decision clears `_breaker_alert_active` (sends recovery
        notice) so the NEXT wedge re-alerts the operator."""
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert runner._breaker_alert_active is True
        # One real (HOLD) decision — latch should clear and a recovery
        # notice should hit Discord.
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._breaker_alert_active is False
        # Recovery notice landed (contains "CLEARED").
        assert any("CLEARED" in m for m in breaker_setup["send"])
        # NEXT wedge re-alerts.
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert len(breaker_setup["fired"]) == 2

    def test_recovery_alert_receives_real_wedge_duration(
            self, breaker_setup, monkeypatch):
        """The recovery message must close the FIRED→CLEARED bracket with the
        actual wall-clock wedge duration. The runner must capture
        ``_no_decision_first_ts`` BEFORE resetting it (otherwise the cleared
        notice always degrades to the no-duration fallback). Mock the new
        ``send_breaker_cleared_alert`` so we see exactly what elapsed_s the
        runner passed."""
        # Trip the breaker first so the latch is armed.
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        # Force a deterministic wedge-start that's ~7 minutes in the past, so
        # the recovery message renders "after ~7m dark" instead of ~0s.
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        wedge_started = _dt.now(_tz.utc) - _td(seconds=420)
        monkeypatch.setattr(runner, "_no_decision_first_ts", wedge_started)
        # Capture the elapsed_s the runner passes to the cleared alert. The
        # real send_breaker_cleared_alert is exercised by the dedicated
        # reporter test class — here we want to assert the runner's kwarg
        # plumbing, not duplicate body formatting.
        cleared_calls: list = []

        def _fake_cleared(**kwargs):
            cleared_calls.append(kwargs)
            return True

        monkeypatch.setattr(
            runner.reporter, "send_breaker_cleared_alert", _fake_cleared
        )
        # One real decision drives recovery.
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._breaker_alert_active is False
        # The recovery alert must have been called EXACTLY once with a
        # non-None elapsed_s in the right ballpark (>=420s, < a bit more
        # because the recovery cycle itself takes a moment of wall clock).
        assert len(cleared_calls) == 1
        elapsed_s = cleared_calls[0].get("elapsed_s")
        assert isinstance(elapsed_s, int)
        assert 420 <= elapsed_s < 600
        # And the wedge-start ts is reset for the NEXT outage to start fresh.
        assert runner._no_decision_first_ts is None

    def test_failed_recovery_send_leaves_breaker_anchor_armed(
            self, breaker_setup, monkeypatch):
        """Symmetric with ``TestQuotaOutageRecovery.test_failed_recovery_send
        _leaves_latch_and_anchor_armed``: a Discord send failure on the
        breaker recovery notice must leave BOTH the latch AND
        ``_no_decision_first_ts`` armed so the next cycle retries with the
        real (slightly longer) elapsed-time figure intact.

        Without this, an openclaw blip at recovery time silently buries
        the FIRED→CLEARED bracket forever — the next cycle's retry would
        render the bare "responding again" body with no "after ~Xh dark"
        close, because `_no_decision_first_ts` had already been reset to
        None unconditionally. This was an asymmetry against the quota
        recovery path (which correctly only clears its anchor on
        confirmed send success — see ``_quota_first_ts`` handling)."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        wedge_started = _dt.now(_tz.utc) - _td(seconds=900)
        monkeypatch.setattr(runner, "_no_decision_first_ts", wedge_started)
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        # send_breaker_cleared_alert returns False → latch + anchor must
        # both stay so the next cycle can retry with elapsed_s intact.
        monkeypatch.setattr(
            runner.reporter, "send_breaker_cleared_alert",
            lambda *, elapsed_s=None: False,
        )
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._breaker_alert_active is True
        assert runner._no_decision_first_ts is wedge_started

    def test_failed_recovery_then_success_passes_full_elapsed(
            self, breaker_setup, monkeypatch):
        """End-to-end: first recovery cycle fails (latch + anchor stay
        armed), second cycle succeeds and passes the FULL elapsed_s from
        the original wedge start — not a truncated value reflecting only
        the gap since the failed first attempt. Pins that the anchor
        survives the retry gap."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        wedge_started = _dt.now(_tz.utc) - _td(seconds=600)
        monkeypatch.setattr(runner, "_no_decision_first_ts", wedge_started)
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        cleared_calls: list = []

        def _flaky_cleared(*, elapsed_s=None):
            cleared_calls.append(elapsed_s)
            # First call fails, all subsequent calls succeed.
            return len(cleared_calls) > 1

        monkeypatch.setattr(
            runner.reporter, "send_breaker_cleared_alert", _flaky_cleared
        )
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()                                  # first send fails
        assert runner._breaker_alert_active is True
        assert runner._no_decision_first_ts is wedge_started
        runner._cycle()                                  # retry succeeds
        assert runner._breaker_alert_active is False
        assert runner._no_decision_first_ts is None
        # Two send attempts; both elapsed_s carry the wedge-start ts (not
        # truncated to 0s). The second is >= the first because wall clock
        # advanced.
        assert len(cleared_calls) == 2
        assert all(isinstance(e, int) and e >= 600 for e in cleared_calls)
        assert cleared_calls[1] >= cleared_calls[0]

    def test_real_decision_with_no_latch_resets_anchor(
            self, breaker_setup, monkeypatch):
        """Sub-threshold path: a wedge ran for a couple of cycles (no
        breaker fire, no latch armed), then the engine recovers. The
        anchor must still reset unconditionally so the NEXT outage starts
        its own clock — the latch-on path's symmetric protection only
        kicks in once the operator alert went out."""
        self._run_no_decision_cycles(2, monkeypatch)
        assert runner._no_decision_first_ts is not None
        assert runner._breaker_alert_active is False
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._no_decision_first_ts is None

    def test_quota_path_does_not_trip_breaker_alert(self, breaker_setup,
                                                     monkeypatch):
        """Quota-exhausted cycles intentionally keep the counter at 0
        (the breaker pkill is futile against a quota outage). The
        breaker alert must NEVER fire on a pure quota outage — that has
        its own alarm (send_quota_alert)."""
        self._run_no_decision_cycles(
            10, monkeypatch, summary_extra={"quota_exhausted": True})
        assert breaker_setup["fired"] == []           # no breaker alert
        assert runner._breaker_alert_active is False  # latch never armed

    def test_reporter_failure_does_not_set_latch(self, breaker_setup,
                                                  monkeypatch):
        """A Discord-send failure must leave the latch False so the
        next cycle re-tries the alert (mirrors the quota-alert path's
        symmetric latch discipline)."""
        # Send returns False → latch stays False.
        monkeypatch.setattr(runner.reporter, "send_breaker_fired_alert",
                            lambda n, cause="", *, elapsed_s=None,
                            store=None: False)
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert runner._breaker_alert_active is False

    def test_failed_send_keeps_counter_at_threshold_for_immediate_retry(
            self, breaker_setup, monkeypatch):
        """A failed delivery must leave the counter AT threshold so the
        next NO_DECISION cycle re-fires the breaker path immediately.

        Bug-fix regression: the runner previously reset
        ``_consecutive_no_decisions = 0`` UNCONDITIONALLY before the
        send attempt. So when openclaw rc=13 dropped the alert (the
        live 2026-05-30 case — `notify_health DEGRADED`,
        `breaker_outage_count: 0` across 16k seconds of confirmed
        wedge), the counter went to 0 and the breaker had to climb
        5 MORE NO_DECISION cycles (~2.5h under dynamic_interval)
        before retrying delivery. Holding the counter at threshold
        on a failed send lets the very next NO_DECISION cycle re-trip
        the breaker path and retry delivery within one tick."""
        monkeypatch.setattr(runner.reporter, "send_breaker_fired_alert",
                            lambda n, cause="", *, elapsed_s=None,
                            store=None: False)
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch)
        assert runner._breaker_alert_active is False
        # Counter stayed AT (or above) threshold — the fix.
        assert (runner._consecutive_no_decisions
                >= runner.CONSECUTIVE_NO_DECISION_LIMIT)
        # Outage count was NOT incremented on a silent fire — the operator
        # was never told, the count must not lie about being told.
        assert runner._breaker_outage_count == 0

    def test_failed_send_then_success_delivers_alert_on_next_cycle(
            self, breaker_setup, monkeypatch):
        """End-to-end: send fails on the 5th NO_DECISION cycle (counter
        reaches threshold, latch stays False). The 6th NO_DECISION cycle
        re-trips the breaker condition immediately (because the counter
        was NOT reset on the failed send) and the second delivery attempt
        succeeds — the operator gets the alert one cycle late, not
        ``CONSECUTIVE_NO_DECISION_LIMIT * dynamic_interval`` late.

        Pins the contract that an openclaw blip costs the operator at most
        ONE extra cycle of silence, not ~2.5 hours."""
        delivered = {"calls": 0}

        def _flaky_send(n, cause="", *, elapsed_s=None, store=None):
            delivered["calls"] += 1
            # First attempt fails, second succeeds.
            ok = delivered["calls"] > 1
            if ok:
                # Append to the fired log so the existing assertion shape
                # still applies (1 successful delivery).
                breaker_setup["fired"].append((n, cause, elapsed_s))
            return ok

        monkeypatch.setattr(
            runner.reporter, "send_breaker_fired_alert", _flaky_send
        )
        # First 5 cycles → threshold hit, send fails, latch stays False,
        # counter stays at threshold.
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch
        )
        assert delivered["calls"] == 1
        assert runner._breaker_alert_active is False
        assert (runner._consecutive_no_decisions
                >= runner.CONSECUTIVE_NO_DECISION_LIMIT)
        assert len(breaker_setup["fired"]) == 0
        # 6th NO_DECISION cycle → counter still ≥ threshold, breaker re-trips,
        # delivery succeeds, latch arms, counter resets.
        self._run_no_decision_cycles(1, monkeypatch)
        assert delivered["calls"] == 2
        assert runner._breaker_alert_active is True
        assert runner._consecutive_no_decisions == 0
        # Outage count incremented exactly once (the delivered attempt).
        assert runner._breaker_outage_count == 1
        assert len(breaker_setup["fired"]) == 1

    def test_already_latched_breaker_resets_counter(self, breaker_setup,
                                                     monkeypatch):
        """When the latch is ALREADY active (a previous outage delivered),
        the counter still resets on breaker-trip — the dedupe path doesn't
        attempt a second send, and the counter must not bleed across the
        latched cycles. Pins that the new counter-reset discipline doesn't
        break the established dedupe contract."""
        # Arm the latch as if a prior outage already delivered.
        monkeypatch.setattr(runner, "_breaker_alert_active", True)
        monkeypatch.setattr(runner, "_breaker_outage_count", 1)
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch
        )
        # No NEW alert (dedupe path) and counter reset to 0.
        assert len(breaker_setup["fired"]) == 0
        assert runner._consecutive_no_decisions == 0
        assert runner._breaker_alert_active is True
        # Outage count NOT incremented — we didn't send a new alert.
        assert runner._breaker_outage_count == 1

    def test_alert_carries_cause_suffix(self, breaker_setup, monkeypatch):
        """The Discord alert body carries the cause code so the operator
        knows whether to wait, restart, escalate to Anthropic, etc."""
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch,
            summary_extra={"host_saturated": True})
        assert len(breaker_setup["fired"]) == 1
        _n, cause, _elapsed = breaker_setup["fired"][0]
        assert "host saturated" in cause

    def test_last_claude_fail_propagates_to_cause(self, breaker_setup,
                                                    monkeypatch):
        """When raw is None (no parse-able response) but the strategy set
        a per-call cause code (timeout / nonzero_rc / cli_missing / …),
        the breaker alert surfaces the code instead of a blank ``""``."""
        self._run_no_decision_cycles(
            runner.CONSECUTIVE_NO_DECISION_LIMIT, monkeypatch,
            summary_extra={"raw": None, "last_claude_fail": "cli_missing"})
        assert len(breaker_setup["fired"]) == 1
        _n, cause, _elapsed = breaker_setup["fired"][0]
        # The cause string is built by _no_decision_cause from summary fields.
        assert "cli_missing" in cause

    def test_real_decision_resets_wedge_elapsed_marker(self, breaker_setup,
                                                        monkeypatch):
        """Once the engine produces a real decision, ``_no_decision_first_ts``
        must reset to None so the NEXT wedge starts its elapsed-time clock
        from a fresh first-NO_DECISION (not from the stale prior outage)."""
        self._run_no_decision_cycles(2, monkeypatch)
        assert runner._no_decision_first_ts is not None
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._no_decision_first_ts is None


class TestQuotaOutageRecovery:
    """The quota EXHAUSTED → RECOVERED bracket. Mirror of the breaker
    FIRED → CLEARED tests one dimension over: quota is a distinct failure
    mode from a wedged CLI (the claude process exited fast, non-zero,
    pkill is futile), so it has its own latch (`_quota_alert_active`),
    its own elapsed-time anchor (`_quota_first_ts`), and its own
    recovery message (`send_quota_recovered_alert`)."""

    @pytest.fixture
    def quota_setup(self, monkeypatch):
        # Reset all module globals between tests so state doesn't leak.
        monkeypatch.setattr(runner, "_consecutive_no_decisions", 0)
        monkeypatch.setattr(runner, "_breaker_alert_active", False)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        monkeypatch.setattr(runner, "_no_decision_first_ts", None)
        monkeypatch.setattr(runner, "_quota_first_ts", None)
        alerts = {
            "fired_quota": [],
            "recovered_quota": [],
            "fired_breaker": [],
            "send": [],
        }
        monkeypatch.setattr(
            runner.reporter, "send_quota_alert",
            lambda detail="", *, store=None: (
                alerts["fired_quota"].append(detail) or True
            ),
        )
        monkeypatch.setattr(
            runner.reporter, "send_quota_recovered_alert",
            lambda *, elapsed_s=None: (
                alerts["recovered_quota"].append(elapsed_s) or True
            ),
        )
        # send_breaker_fired_alert must accept the same kwargs the runner
        # passes — the quota path must NEVER fire it, but the stub still
        # has to match the signature so a misrouted call records cleanly.
        monkeypatch.setattr(
            runner.reporter, "send_breaker_fired_alert",
            lambda n, cause="", *, elapsed_s=None, store=None: (
                alerts["fired_breaker"].append((n, cause, elapsed_s))
                or True
            ),
        )
        monkeypatch.setattr(runner.reporter, "_send",
                            lambda m: alerts["send"].append(m) or True)
        monkeypatch.setattr(runner, "_kill_stale_claude", lambda: None)
        monkeypatch.setattr(runner, "get_store", lambda: _FakeStore([]))
        return alerts

    def _run_quota_cycles(self, n, monkeypatch):
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": True})
        for _ in range(n):
            runner._cycle()

    def test_first_quota_cycle_anchors_first_ts(
            self, quota_setup, monkeypatch):
        """The first quota cycle of an outage MUST set ``_quota_first_ts``
        so the recovery message can later render the real wedge duration.
        Without this anchor, ``send_quota_recovered_alert`` always degrades
        to the no-duration fallback."""
        assert runner._quota_first_ts is None
        self._run_quota_cycles(1, monkeypatch)
        assert runner._quota_first_ts is not None

    def test_quota_first_ts_does_not_re_anchor_on_later_cycles(
            self, quota_setup, monkeypatch):
        """A multi-cycle quota outage must NOT keep re-anchoring on every
        cycle (that would silently collapse the elapsed duration to ~0 at
        recovery and defeat the bracket close — the whole point of the
        feature). The anchor is set once on the FIRST quota cycle and held
        until the outage clears."""
        self._run_quota_cycles(1, monkeypatch)
        first = runner._quota_first_ts
        assert first is not None
        # Drive 4 more quota cycles — the anchor must be the SAME instance.
        self._run_quota_cycles(4, monkeypatch)
        assert runner._quota_first_ts is first

    def test_recovery_passes_elapsed_s_and_clears_anchor(
            self, quota_setup, monkeypatch):
        """A real decision after a quota outage must (a) fire the recovery
        alert with a non-None ``elapsed_s`` reflecting the actual outage
        duration, and (b) reset both the latch AND the ``_quota_first_ts``
        anchor so the NEXT outage starts its own clock fresh.
        Mirrors the breaker test_recovery_alert_receives_real_wedge_duration
        contract one dimension over."""
        # Force a deterministic outage start ~7 minutes in the past.
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        outage_started = _dt.now(_tz.utc) - _td(seconds=420)
        monkeypatch.setattr(runner, "_quota_first_ts", outage_started)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        # Recovery alert fired exactly once with a real elapsed_s.
        assert len(quota_setup["recovered_quota"]) == 1
        elapsed_s = quota_setup["recovered_quota"][0]
        assert isinstance(elapsed_s, int)
        assert 420 <= elapsed_s < 600
        # Latch cleared AND anchor reset for the next outage.
        assert runner._quota_alert_active is False
        assert runner._quota_first_ts is None

    def test_failed_recovery_send_leaves_latch_and_anchor_armed(
            self, quota_setup, monkeypatch):
        """Symmetric with the breaker recovery path: a transient openclaw
        failure on the recovery message must leave BOTH the latch AND
        ``_quota_first_ts`` armed so the next cycle retries with the
        real (now slightly longer) elapsed-time figure intact. Without
        this, an openclaw blip at recovery time silently buries the
        EXHAUSTED→RECOVERED bracket forever."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        outage_started = _dt.now(_tz.utc) - _td(seconds=200)
        monkeypatch.setattr(runner, "_quota_first_ts", outage_started)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        # Send returns False → latch + anchor must stay.
        monkeypatch.setattr(
            runner.reporter, "send_quota_recovered_alert",
            lambda *, elapsed_s=None: False,
        )
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._quota_alert_active is True
        assert runner._quota_first_ts is outage_started

    def test_recovery_when_anchor_missing_still_alerts(
            self, quota_setup, monkeypatch):
        """If ``_quota_first_ts`` was somehow not captured (legacy DB-restart
        or test seam dropped it), the recovery must STILL fire — just with
        elapsed_s=None. The latch must still clear on a successful send so
        the operator gets the "we're back" message even without duration
        context. Mirrors the breaker recovery's None-elapsed fallback."""
        monkeypatch.setattr(runner, "_quota_first_ts", None)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert len(quota_setup["recovered_quota"]) == 1
        assert quota_setup["recovered_quota"][0] is None
        assert runner._quota_alert_active is False

    def test_quota_recovery_does_not_fire_on_no_decision(
            self, quota_setup, monkeypatch):
        """A NO_DECISION cycle that isn't a quota miss (timeout / parse fail)
        must NOT clear the quota latch — only a CONFIRMED claude response
        (HOLD / FILLED / BLOCKED) proves the quota is back. Pre-feature
        behavior was the same; pin it so the new helper can't accidentally
        be wired into the no-decision path."""
        from datetime import datetime as _dt, timezone as _tz
        anchor = _dt.now(_tz.utc)
        monkeypatch.setattr(runner, "_quota_first_ts", anchor)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": False})
        runner._cycle()
        # No recovery alert.
        assert quota_setup["recovered_quota"] == []
        # Latch + anchor preserved for the eventual real recovery cycle.
        assert runner._quota_alert_active is True
        assert runner._quota_first_ts is anchor

    def test_orphan_quota_anchor_cleared_when_latch_never_armed(
            self, quota_setup, monkeypatch):
        """An ``_quota_first_ts`` set on the first quota cycle but whose
        alert ``send_quota_alert`` REJECTED (returned False — transient
        openclaw / Discord blip) leaves ``_quota_alert_active`` False yet
        the anchor armed. On the next non-quota cycle the recovery branch
        skips (its ``_quota_alert_active`` predicate is False), so the
        anchor would otherwise stay stuck forever — and a FUTURE quota
        outage (which would re-anchor only if ``_quota_first_ts is None``)
        would inherit the stale timestamp, producing a wildly inflated
        ``elapsed_s`` on its recovery alert (the gap between BOTH outages,
        not the new one).

        Symmetric with the breaker anchor's "no latch → reset" cleanup on
        the non-quota else arm (runner.py:975). Pin the cleanup so this
        regression cannot return."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        # Simulate the "alert send failed on first quota cycle" residue:
        # anchor is set but the latch never armed.
        orphan_anchor = _dt.now(_tz.utc) - _td(seconds=400)
        monkeypatch.setattr(runner, "_quota_first_ts", orphan_anchor)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        # Recovery must NOT have fired (latch was False to begin with — there
        # was no outage in the operator's view to "recover" from).
        assert quota_setup["recovered_quota"] == []
        # Anchor must be cleared so the NEXT quota outage starts a fresh
        # clock with the correct first-quota-cycle timestamp.
        assert runner._quota_first_ts is None

    def test_orphan_quota_anchor_does_not_break_armed_recovery(
            self, quota_setup, monkeypatch):
        """The orphan-anchor cleanup must not interfere with a LEGITIMATE
        armed recovery on the same cycle. When ``_quota_alert_active`` is
        True (a real outage WAS alerted) and a real decision arrives, the
        recovery branch must fire normally AND clear both fields — exactly
        as before. The new cleanup only fires when the latch was never
        armed in the first place."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        outage_started = _dt.now(_tz.utc) - _td(seconds=500)
        monkeypatch.setattr(runner, "_quota_first_ts", outage_started)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        # Recovery fired with a real elapsed_s.
        assert len(quota_setup["recovered_quota"]) == 1
        assert quota_setup["recovered_quota"][0] is not None
        assert 500 <= quota_setup["recovered_quota"][0] < 700
        # Both fields cleared by the legitimate recovery — the new cleanup
        # branch does not re-fire on an already-cleared anchor.
        assert runner._quota_alert_active is False
        assert runner._quota_first_ts is None

    def test_orphan_quota_anchor_cleared_on_blocked_decision(
            self, quota_setup, monkeypatch):
        """Same orphan-cleanup must also fire when the engine returns a
        BLOCKED (real decision, just risk-rejected). The cleanup keys on the
        latch state, not the status string — any non-quota path through
        ``_cycle`` must drop a stale anchor when the latch was never armed.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        orphan_anchor = _dt.now(_tz.utc) - _td(seconds=300)
        monkeypatch.setattr(runner, "_quota_first_ts", orphan_anchor)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "BLOCKED",
                     "decision": {"action": "SELL", "ticker": "NVDA"}})
        runner._cycle()
        assert runner._quota_first_ts is None

    def test_orphan_quota_anchor_cleared_on_non_quota_no_decision(
            self, quota_setup, monkeypatch):
        """Symmetry gap: the orphan-cleanup must ALSO fire when the next
        cycle is a NON-quota NO_DECISION (timeout / parse fail / host-saturated
        — anything that didn't go through the quota arm). The original
        cleanup only ran in the recovery (``else`` / non-NO_DECISION)
        branch of ``_cycle``, so an orphaned ``_quota_first_ts`` would
        persist through every subsequent NO_DECISION wedge — surfacing a
        misleading ``quota_outage_s`` via ``alarm_latch_state()`` while
        ``quota_active=False``. A trader reading
        ``/api/alarm-latches`` during a non-quota host-saturation wedge
        would see a non-null quota outage age that does NOT correspond to
        a current quota state. The cleanup keys on latch state, not the
        status string — any non-quota path through ``_cycle`` must drop
        a stale anchor when the latch was never armed.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        # Simulate the "first quota cycle's send failed → orphan anchor
        # remains, latch never armed" state from the failed-delivery path.
        orphan_anchor = _dt.now(_tz.utc) - _td(seconds=500)
        monkeypatch.setattr(runner, "_quota_first_ts", orphan_anchor)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        # Next cycle is a non-quota NO_DECISION (e.g. CLI timeout, parse fail,
        # host saturated — anything that doesn't set quota_exhausted=True).
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": False})
        runner._cycle()
        # Anchor cleaned up so a FUTURE legitimate quota outage starts a
        # fresh clock from its own first cycle (not from this orphan).
        assert runner._quota_first_ts is None
        # Latch still False — nothing to recover from in the operator's view.
        assert runner._quota_alert_active is False
        # No recovery alert fired (no outage to recover from).
        assert quota_setup["recovered_quota"] == []

    def test_orphan_anchor_does_not_clear_active_quota_during_wedge(
            self, quota_setup, monkeypatch):
        """Counterpart to the cleanup test: when the quota latch IS armed
        (we already alerted the operator about an ongoing quota outage),
        a non-quota NO_DECISION cycle that happens between the alert and
        the recovery must NOT drop the anchor — the eventual recovery
        cycle still needs ``_quota_first_ts`` to compute the real elapsed
        duration for the operator-facing ``EXHAUSTED → RECOVERED`` bracket.
        The cleanup must key on latch state, exactly mirroring the
        recovery-branch precedent on runner.py:1051.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        ongoing_anchor = _dt.now(_tz.utc) - _td(seconds=900)
        monkeypatch.setattr(runner, "_quota_first_ts", ongoing_anchor)
        monkeypatch.setattr(runner, "_quota_alert_active", True)
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": False})
        runner._cycle()
        # Anchor preserved — eventual recovery still needs the duration.
        assert runner._quota_first_ts is ongoing_anchor
        assert runner._quota_alert_active is True


class TestOutageCounters:
    """Session-scoped distinct-outage counters. ``_quota_outage_count`` and
    ``_breaker_outage_count`` increment ONCE per outage on the edge that
    flips the latch True (i.e., the moment the operator alert actually
    delivers). They are NOT incremented when the alert send fails (the
    documented invariant: the count must reflect what the operator was
    actually told about, never silent failures)."""

    @pytest.fixture
    def counter_setup(self, monkeypatch):
        # Reset every alarm / latch field and both counters between tests.
        monkeypatch.setattr(runner, "_consecutive_no_decisions", 0)
        monkeypatch.setattr(runner, "_breaker_alert_active", False)
        monkeypatch.setattr(runner, "_quota_alert_active", False)
        monkeypatch.setattr(runner, "_no_decision_first_ts", None)
        monkeypatch.setattr(runner, "_quota_first_ts", None)
        monkeypatch.setattr(runner, "_quota_outage_count", 0)
        monkeypatch.setattr(runner, "_breaker_outage_count", 0)
        sends = {"quota": [], "breaker": [], "send": []}
        monkeypatch.setattr(
            runner.reporter, "send_quota_alert",
            lambda detail="", *, store=None:
                sends["quota"].append(detail) or True,
        )
        monkeypatch.setattr(
            runner.reporter, "send_breaker_fired_alert",
            lambda n, cause="", *, elapsed_s=None, store=None:
                sends["breaker"].append((n, cause)) or True,
        )
        monkeypatch.setattr(
            runner.reporter, "send_quota_recovered_alert",
            lambda *, elapsed_s=None: True,
        )
        monkeypatch.setattr(
            runner.reporter, "send_breaker_cleared_alert",
            lambda *, elapsed_s=None: True,
        )
        monkeypatch.setattr(runner.reporter, "_send",
                            lambda m: sends["send"].append(m) or True)
        monkeypatch.setattr(runner, "_kill_stale_claude", lambda: None)
        monkeypatch.setattr(runner, "get_store", lambda: _FakeStore([]))
        return sends

    def test_quota_outage_count_increments_on_first_alerted_cycle(
            self, counter_setup, monkeypatch):
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": True})
        assert runner._quota_outage_count == 0
        runner._cycle()
        assert runner._quota_outage_count == 1
        # Latch is held — further cycles in the same outage must not
        # double-count.
        runner._cycle()
        runner._cycle()
        assert runner._quota_outage_count == 1
        assert len(counter_setup["quota"]) == 1  # only one Discord alert

    def test_quota_outage_count_does_not_increment_on_failed_delivery(
            self, counter_setup, monkeypatch):
        """A ``send_quota_alert`` that returns False (transient openclaw /
        Discord blip) must NOT increment the counter. The invariant is that
        the count reflects outages the operator was told about; a silent
        failure that never reached Discord is the orphan-anchor case, NOT a
        legitimate counted outage. Confirms the counter never disagrees
        with what's actually visible on Discord."""
        monkeypatch.setattr(
            runner.reporter, "send_quota_alert",
            lambda detail="", *, store=None: False,  # delivery failed
        )
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": True})
        runner._cycle()
        assert runner._quota_outage_count == 0
        assert runner._quota_alert_active is False
        # And the orphan-anchor cleanup fires on the next non-quota cycle
        # (the Phase 1 fix this Phase 2 feature builds on).
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._quota_first_ts is None
        # Still zero — the silent quota outage never counted.
        assert runner._quota_outage_count == 0

    def test_quota_outage_count_increments_per_outage_not_per_cycle(
            self, counter_setup, monkeypatch):
        """A second DISTINCT outage after a recovery must increment the
        counter to 2. Pins the "edge-triggered" semantics: count fires on
        every False→True latch transition, not on every quota cycle."""
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": True})
        runner._cycle()
        assert runner._quota_outage_count == 1
        # Recovery — real decision lands.
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "HOLD", "decision": {"action": "HOLD"}})
        runner._cycle()
        assert runner._quota_alert_active is False
        # New, distinct outage.
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": True})
        runner._cycle()
        assert runner._quota_outage_count == 2

    def test_breaker_outage_count_increments_on_threshold_reached(
            self, counter_setup, monkeypatch):
        """The breaker counter mirrors the quota counter one dimension over.
        Increments once on the cycle that pushes the consecutive count over
        the threshold AND that cycle actually delivers the breaker alert.
        Subsequent NO_DECISION cycles in the same wedge do not re-count
        (the latch dedupes them)."""
        monkeypatch.setattr(
            runner.strategy, "decide",
            lambda: {"status": "NO_DECISION", "decision": None,
                     "quota_exhausted": False})
        # 5 NO_DECISION cycles trips the breaker (CONSECUTIVE_NO_DECISION_LIMIT).
        for _ in range(runner.CONSECUTIVE_NO_DECISION_LIMIT):
            runner._cycle()
        assert runner._breaker_outage_count == 1
        # Latch is held. More NO_DECISIONs must not re-count.
        for _ in range(3):
            runner._cycle()
        assert runner._breaker_outage_count == 1

    def test_counters_surface_in_alarm_latch_state(
            self, counter_setup, monkeypatch):
        """Both counters must be exposed by ``alarm_latch_state`` so any
        operator surface (the ``/api/alarm-latches`` endpoint, the Discord
        hourly summary, a chat helper) reads the same number. Pins the API
        shape so downstream consumers can rely on the keys' presence."""
        monkeypatch.setattr(runner, "_quota_outage_count", 2)
        monkeypatch.setattr(runner, "_breaker_outage_count", 1)
        st = runner.alarm_latch_state()
        assert st["quota_outage_count"] == 2
        assert st["breaker_outage_count"] == 1
        # Both counters present even when fresh-boot zero — never `None` /
        # `KeyError` (the dashboard panel must always render).
        monkeypatch.setattr(runner, "_quota_outage_count", 0)
        monkeypatch.setattr(runner, "_breaker_outage_count", 0)
        st = runner.alarm_latch_state()
        assert st["quota_outage_count"] == 0
        assert st["breaker_outage_count"] == 0
