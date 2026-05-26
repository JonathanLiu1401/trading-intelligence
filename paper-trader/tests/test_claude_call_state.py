"""Tests for ``strategy.claude_call_state()`` and its wiring into the
runner-heartbeat endpoint.

The motivation is a trader-visible gap: under ``DECISION_TIMEOUT_S = None``
a healthy Opus call can legitimately run for minutes. From the heartbeat
the operator only sees ``secs_since_last_decision`` rising — they cannot
distinguish "Opus is thinking" (good) from "engine wedged" (bad). The
deadman already uses the same in-flight Popen signal to defer force-exit;
this surfaces that signal to the dashboard too.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import strategy


class _StubProc:
    """Minimal Popen-shaped stub: ``poll`` returns None when alive."""

    def __init__(self, alive: bool = True, pid: int = 12345):
        self._alive = alive
        self.pid = pid

    def poll(self):
        return None if self._alive else 0


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch):
    """Each test gets a clean ``_active_claude_proc`` / ``_active_claude_started_at``."""
    monkeypatch.setattr(strategy, "_active_claude_proc", None,
                        raising=False)
    monkeypatch.setattr(strategy, "_active_claude_started_at", None,
                        raising=False)


class TestClaudeCallStateInactive:
    def test_no_active_proc_reports_inactive(self):
        st = strategy.claude_call_state()
        assert st["active"] is False
        assert st["elapsed_s"] is None
        assert st["pid"] is None

    def test_exited_proc_reports_inactive(self, monkeypatch):
        """A finished subprocess (``poll()`` returns rc) is NOT active."""
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=False), raising=False)
        monkeypatch.setattr(strategy, "_active_claude_started_at",
                            time.monotonic() - 10.0, raising=False)
        st = strategy.claude_call_state()
        assert st["active"] is False
        # elapsed_s suppressed when not active — knowing how long a dead
        # subprocess ran is not the heartbeat-reader's question.
        assert st["elapsed_s"] is None
        assert st["pid"] is None


class TestClaudeCallStateActive:
    def test_active_proc_reports_active_with_elapsed(self, monkeypatch):
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=True, pid=42), raising=False)
        monkeypatch.setattr(strategy, "_active_claude_started_at",
                            time.monotonic() - 6.0, raising=False)
        st = strategy.claude_call_state()
        assert st["active"] is True
        # elapsed_s is int seconds, floored from monotonic delta. Allow a 2s
        # tolerance to absorb whatever monotonic skew the test runner adds.
        assert 5 <= st["elapsed_s"] <= 8
        assert st["pid"] == 42

    def test_active_without_started_at_elapsed_none(self, monkeypatch):
        """If somehow the start ts was not captured (legacy or test path),
        elapsed_s degrades to None rather than crashing — the active flag
        and pid still ship so the heartbeat is partially useful."""
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=True, pid=7), raising=False)
        monkeypatch.setattr(strategy, "_active_claude_started_at",
                            None, raising=False)
        st = strategy.claude_call_state()
        assert st["active"] is True
        assert st["elapsed_s"] is None
        assert st["pid"] == 7

    def test_future_started_at_clamps_to_zero(self, monkeypatch):
        """A monotonic that somehow goes backward (CI VM oddity) should
        clamp to 0, never render negative — same hardening as
        ``alarm_latch_state``'s wall-clock clamp."""
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=True), raising=False)
        monkeypatch.setattr(strategy, "_active_claude_started_at",
                            time.monotonic() + 10.0, raising=False)
        st = strategy.claude_call_state()
        assert st["active"] is True
        assert st["elapsed_s"] == 0


class TestClaudeCallStateNeverRaises:
    def test_poll_exception_falls_back(self, monkeypatch):
        class _BadProc:
            pid = 1
            def poll(self):
                raise RuntimeError("simulated poll failure")
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _BadProc(), raising=False)
        # Must not raise — degrade to inactive.
        st = strategy.claude_call_state()
        assert st["active"] is False
        assert st["elapsed_s"] is None

    def test_returns_dict_shape(self):
        st = strategy.claude_call_state()
        assert set(st.keys()) == {"active", "elapsed_s", "pid"}


class TestIsClaudeCallActiveUnchanged:
    """The legacy boolean accessor is byte-compatible: the deadman uses it
    by name (``strategy.is_claude_call_active``) and must keep returning a
    bare bool with the same semantics."""

    def test_inactive_returns_false(self):
        assert strategy.is_claude_call_active() is False

    def test_active_returns_true(self, monkeypatch):
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=True), raising=False)
        assert strategy.is_claude_call_active() is True


# ─────────────────────── /api/runner-heartbeat integration ──────────────────
class TestHeartbeatExposesClaudeCallState:
    """The dashboard endpoint must include a ``claude_call`` block whose
    contents come from ``strategy.claude_call_state()`` — the trader's
    surface for "is Opus thinking right now"."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        # Isolated store so the test never reads the live paper_trader.db.
        from paper_trader import store as store_mod
        monkeypatch.setattr(store_mod, "DB_PATH",
                            tmp_path / "paper_trader.db")
        store_mod._singleton = None
        s = store_mod.Store()
        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as c:
            yield c, s
        s.close()
        store_mod._singleton = None

    def test_idle_heartbeat_shows_inactive_call(self, client, monkeypatch):
        c, s = client
        s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
        # No active proc — the default fixture state.
        monkeypatch.setattr(strategy, "_active_claude_proc", None,
                            raising=False)
        r = c.get("/api/runner-heartbeat")
        assert r.status_code == 200
        d = r.get_json()
        assert d["verdict"] == "HEALTHY"  # unchanged
        assert "claude_call" in d
        assert d["claude_call"]["active"] is False
        assert d["claude_call"]["elapsed_s"] is None

    def test_active_heartbeat_shows_thinking(self, client, monkeypatch):
        c, s = client
        s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
        monkeypatch.setattr(strategy, "_active_claude_proc",
                            _StubProc(alive=True, pid=9999), raising=False)
        monkeypatch.setattr(strategy, "_active_claude_started_at",
                            time.monotonic() - 12.0, raising=False)
        r = c.get("/api/runner-heartbeat")
        d = r.get_json()
        assert d["claude_call"]["active"] is True
        assert d["claude_call"]["pid"] == 9999
        # >=10 because we set started_at 12s ago; <=15 to absorb test jitter.
        assert 10 <= d["claude_call"]["elapsed_s"] <= 15

    def test_strategy_fault_does_not_break_heartbeat(self, client,
                                                       monkeypatch):
        """If ``claude_call_state`` itself raises (the bare minimum invariant
        — the additive contract), the heartbeat must still ship with the
        liveness verdict; only the ``claude_call`` block goes missing."""
        c, s = client
        s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
        monkeypatch.setattr(strategy, "claude_call_state",
                            lambda: (_ for _ in ()).throw(
                                RuntimeError("simulated")),
                            raising=False)
        r = c.get("/api/runner-heartbeat")
        assert r.status_code == 200
        d = r.get_json()
        assert d["verdict"] == "HEALTHY"
        assert "claude_call" not in d
