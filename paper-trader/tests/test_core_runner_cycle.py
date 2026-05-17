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
        monkeypatch.setattr(runner.reporter, "send_trade_alert",
                            lambda t: calls["trade_alert"].append(t) or True)
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
        each exit emits its own AUTO RISK EXIT line, while the
        trade-alert/decision-log stay gated on status == FILLED."""
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
