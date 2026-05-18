"""Quota-exhaustion guard — the feature that turns a SILENT frozen trader into
a loud one.

Live failure observed 2026-05-17 (runner.log): every claude attempt returned
rc=1 with stdout `You've hit your org's monthly usage limit`. Old behaviour:
this degraded to NO_DECISION forever, the circuit breaker spun uselessly
(nothing to pkill), and the operator got *zero* Discord notice — the worst
silent failure for a live trader ("I thought it was running; it hasn't traded
in hours and nobody told me"). Compounding it: `openclaw` is an npm-global
under the nvm node bin which the systemd unit's minimal PATH excludes, so
`shutil.which` returned None and EVERY Discord message was being dropped too.

This locks the whole chain:
  * strategy._is_quota_exhausted  — precise marker detection (no false alarms)
  * strategy._claude_call         — sets the module flag on a quota rc
  * strategy.decide()             — surfaces summary["quota_exhausted"],
                                    reset per cycle
  * reporter._resolve_openclaw    — robust binary resolution (env / PATH /
                                    well-known fallbacks)
  * reporter.send_quota_alert     — the alarm body
  * runner._cycle                 — dedupe latch, recovery re-arm, and the
                                    "do NOT run the futile breaker" rule
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.strategy as strategy
from paper_trader import reporter, runner


# ───────────────────────── strategy._is_quota_exhausted ─────────────────────
class TestIsQuotaExhausted:
    def test_observed_live_string_matches(self):
        assert strategy._is_quota_exhausted(
            "You've hit your org's monthly usage limit"
        ) is True

    def test_case_insensitive(self):
        assert strategy._is_quota_exhausted("ORG MONTHLY USAGE LIMIT reached") is True

    def test_quota_exceeded_phrasings(self):
        assert strategy._is_quota_exhausted("Error: quota exceeded for org") is True
        assert strategy._is_quota_exhausted("api quota exhausted") is True
        assert strategy._is_quota_exhausted("you are out of credit") is True

    def test_transient_errors_do_not_match(self):
        # These must NOT cry wolf — they are timeouts / parse misses / net blips.
        assert strategy._is_quota_exhausted("claude timeout after 180s") is False
        assert strategy._is_quota_exhausted("connection reset by peer") is False
        assert strategy._is_quota_exhausted('{"action": "HOLD"}') is False
        assert strategy._is_quota_exhausted("500 internal server error") is False

    def test_none_and_empty_safe(self):
        assert strategy._is_quota_exhausted(None) is False
        assert strategy._is_quota_exhausted("") is False


# ───────────────────────── strategy._claude_call wiring ─────────────────────
class _FakeProc:
    def __init__(self, rc, stdout, stderr):
        self.returncode = rc
        self._out = stdout
        self._err = stderr
        self.killed = False

    def communicate(self, input=None, timeout=None):
        return self._out, self._err

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def reset_quota_flag():
    strategy._quota_exhausted = False
    yield
    strategy._quota_exhausted = False


class TestClaudeCallSetsQuotaFlag:
    def _wire(self, monkeypatch, stdout, stderr="", rc=1):
        monkeypatch.setattr(strategy.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(strategy, "_active_claude_proc", None)
        proc = _FakeProc(rc, stdout, stderr)
        monkeypatch.setattr(strategy.subprocess, "Popen", lambda *a, **k: proc)
        return proc

    def test_quota_rc_sets_flag_and_returns_none(self, monkeypatch, reset_quota_flag):
        self._wire(monkeypatch, "You've hit your org's monthly usage limit")
        out = strategy._claude_call("prompt", timeout_s=5)
        assert out is None
        assert strategy._quota_exhausted is True

    def test_non_quota_rc_leaves_flag_false(self, monkeypatch, reset_quota_flag):
        self._wire(monkeypatch, "", stderr="segfault in subprocess")
        out = strategy._claude_call("prompt", timeout_s=5)
        assert out is None
        assert strategy._quota_exhausted is False

    def test_success_does_not_set_flag(self, monkeypatch, reset_quota_flag):
        self._wire(monkeypatch, '{"action":"HOLD"}', rc=0)
        out = strategy._claude_call("prompt", timeout_s=5)
        assert out == '{"action":"HOLD"}'
        assert strategy._quota_exhausted is False


# ───────────────────────── strategy.decide() surface ───────────────────────
@pytest.fixture
def stub_decide_inputs(monkeypatch):
    snap = {"total_value": 1000.0, "cash": 1000.0, "open_value": 0.0, "positions": []}
    monkeypatch.setattr(strategy, "_portfolio_snapshot", lambda store: snap)
    fake_store = mock.MagicMock()
    monkeypatch.setattr(strategy, "get_store", lambda: fake_store)
    monkeypatch.setattr(strategy.signals, "get_top_signals", lambda *a, **k: [])
    monkeypatch.setattr(strategy.signals, "get_urgent_articles", lambda *a, **k: [])
    monkeypatch.setattr(strategy.signals, "ticker_sentiments", lambda *a, **k: [])
    monkeypatch.setattr(strategy.market, "is_market_open", lambda: True)
    monkeypatch.setattr(strategy.market, "get_prices", lambda *a, **k: {})
    monkeypatch.setattr(strategy.market, "get_futures_price", lambda *a, **k: None)
    monkeypatch.setattr(strategy.market, "benchmark_sp500", lambda: 5000.0)
    monkeypatch.setattr(strategy, "get_quant_signals_live", lambda *a, **k: {})
    return fake_store


class TestDecideSurfacesQuota:
    def test_quota_attempt_surfaces_in_summary(self, stub_decide_inputs, monkeypatch,
                                               reset_quota_flag):
        def fake_claude(*a, **k):
            strategy._quota_exhausted = True   # mimic _claude_call's real effect
            return None
        monkeypatch.setattr(strategy, "_claude_call", fake_claude)
        summary = strategy.decide()
        assert summary["quota_exhausted"] is True
        assert summary["status"] == "NO_DECISION"
        # The recorded reasoning is the quota-specific string, not parse_failed.
        args = stub_decide_inputs.record_decision.call_args[0]
        assert "quota" in args[3].lower()

    def test_plain_timeout_is_not_quota(self, stub_decide_inputs, monkeypatch,
                                        reset_quota_flag):
        monkeypatch.setattr(strategy, "_claude_call", lambda *a, **k: None)
        summary = strategy.decide()
        assert summary["quota_exhausted"] is False
        assert summary["status"] == "NO_DECISION"

    def test_flag_is_reset_per_cycle(self, stub_decide_inputs, monkeypatch,
                                     reset_quota_flag):
        # A stale True from a prior cycle must NOT bleed into a clean cycle.
        strategy._quota_exhausted = True
        monkeypatch.setattr(strategy, "_claude_call",
                            lambda *a, **k: '{"action":"HOLD","ticker":"","reasoning":"q"}')
        summary = strategy.decide()
        assert summary["quota_exhausted"] is False
        assert summary["status"] == "HOLD"

    def test_host_saturation_skips_claude_call(self, stub_decide_inputs,
                                               monkeypatch, reset_quota_flag):
        """The pre-flight host-saturation guard must SKIP the claude call(s)
        entirely (no doomed 1.5GB subprocess into the storm) and record a
        distinct reason — kept separate from the 'claude returned no response'
        empty/timeout bucket so /api/empty-claude-rate stays accurate."""
        called = {"n": 0}

        def spy_claude(*a, **k):
            called["n"] += 1
            return '{"action":"BUY","ticker":"NVDA","reasoning":"x"}'

        # Override the conftest autouse neutralisation: simulate a saturated box.
        monkeypatch.setattr(strategy, "host_saturated",
                            lambda *a, **k: (True, "host saturated: 9 concurrent Opus (>4)"))
        monkeypatch.setattr(strategy, "_claude_call", spy_claude)
        summary = strategy.decide()

        assert called["n"] == 0, "claude must NOT be spawned when host saturated"
        assert summary["status"] == "NO_DECISION"
        assert summary["host_saturated"] is True
        assert summary["quota_exhausted"] is False
        reason = stub_decide_inputs.record_decision.call_args[0][3]
        assert reason.startswith("skipped claude call —")
        assert not reason.startswith("claude returned no response")


# ───────────────────────── reporter._resolve_openclaw ──────────────────────
class TestResolveOpenclaw:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        fake = tmp_path / "oc"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv("OPENCLAW_BIN", str(fake))
        monkeypatch.setattr(reporter.shutil, "which", lambda n: "/should/not/win")
        assert reporter._resolve_openclaw() == str(fake)

    def test_env_override_ignored_when_not_executable(self, tmp_path, monkeypatch):
        plain = tmp_path / "oc.txt"
        plain.write_text("not exec")
        plain.chmod(0o644)
        monkeypatch.setenv("OPENCLAW_BIN", str(plain))
        monkeypatch.setattr(reporter.shutil, "which", lambda n: "/usr/bin/openclaw")
        # Falls through to PATH since the override is not an executable file.
        assert reporter._resolve_openclaw() == "/usr/bin/openclaw"

    def test_path_used_when_no_override(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_BIN", raising=False)
        monkeypatch.setattr(reporter.shutil, "which", lambda n: "/opt/bin/openclaw")
        assert reporter._resolve_openclaw() == "/opt/bin/openclaw"

    def test_fallback_glob_when_not_on_path(self, tmp_path, monkeypatch):
        """The exact live bug: not on PATH but present in a well-known
        location (the nvm bin). Must still resolve."""
        nvm_bin = tmp_path / ".nvm" / "versions" / "node" / "v24.15.0" / "bin"
        nvm_bin.mkdir(parents=True)
        oc = nvm_bin / "openclaw"
        oc.write_text("#!/bin/sh\n")
        oc.chmod(0o755)
        monkeypatch.delenv("OPENCLAW_BIN", raising=False)
        monkeypatch.setattr(reporter.shutil, "which", lambda n: None)
        monkeypatch.setattr(reporter.os.path, "expanduser", lambda p: str(tmp_path))
        assert reporter._resolve_openclaw() == str(oc)

    def test_none_when_genuinely_absent(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_BIN", raising=False)
        monkeypatch.setattr(reporter.shutil, "which", lambda n: None)
        monkeypatch.setattr(reporter, "_openclaw_fallback_candidates", lambda: [])
        assert reporter._resolve_openclaw() is None


# ───────────────────────── reporter.send_quota_alert ───────────────────────
class TestSendQuotaAlert:
    def test_message_body_and_detail(self, monkeypatch):
        sent = []
        monkeypatch.setattr(reporter, "_send", lambda m: sent.append(m) or True)
        ok = reporter.send_quota_alert("Opus + Sonnet both rejected.")
        assert ok is True
        body = sent[0]
        assert "QUOTA EXHAUSTED" in body
        assert "FROZEN" in body
        assert "No new trades will execute" in body
        assert "Opus + Sonnet both rejected." in body

    def test_no_detail_still_sends(self, monkeypatch):
        sent = []
        monkeypatch.setattr(reporter, "_send", lambda m: sent.append(m) or True)
        assert reporter.send_quota_alert() is True
        assert "QUOTA EXHAUSTED" in sent[0]

    def test_returns_send_result(self, monkeypatch):
        monkeypatch.setattr(reporter, "_send", lambda m: False)
        assert reporter.send_quota_alert("x") is False


# ───────────────────────── runner._cycle dedupe / breaker ──────────────────
@pytest.fixture
def cycle_spy(monkeypatch):
    calls = {"quota_alert": 0, "send": [], "killed": 0}

    monkeypatch.setattr(runner.reporter, "send_quota_alert",
                        lambda detail="": calls.__setitem__("quota_alert",
                                                             calls["quota_alert"] + 1) or True)
    monkeypatch.setattr(runner.reporter, "_send",
                        lambda m: calls["send"].append(m) or True)
    monkeypatch.setattr(runner.reporter, "send_trade_alert", lambda t: True)
    monkeypatch.setattr(runner.reporter, "send_decision_log", lambda s: True)
    monkeypatch.setattr(runner, "_kill_stale_claude",
                        lambda: calls.__setitem__("killed", calls["killed"] + 1))
    monkeypatch.setattr(runner, "get_store", lambda: mock.MagicMock())
    # Clean global state for every test.
    monkeypatch.setattr(runner, "_quota_alert_active", False)
    monkeypatch.setattr(runner, "_consecutive_no_decisions", 0)

    def run(summary):
        monkeypatch.setattr(runner.strategy, "decide", lambda: summary)
        runner._cycle()

    return calls, run


class TestCycleQuotaAlarm:
    _Q = {"status": "NO_DECISION", "decision": None, "quota_exhausted": True}

    def test_first_quota_cycle_alarms_once(self, cycle_spy):
        calls, run = cycle_spy
        run(self._Q)
        assert calls["quota_alert"] == 1
        assert runner._quota_alert_active is True

    def test_subsequent_quota_cycles_do_not_respam(self, cycle_spy):
        calls, run = cycle_spy
        run(self._Q)
        run(self._Q)
        run(self._Q)
        assert calls["quota_alert"] == 1  # deduped

    def test_breaker_never_fires_under_quota(self, cycle_spy):
        calls, run = cycle_spy
        for _ in range(runner.CONSECUTIVE_NO_DECISION_LIMIT + 3):
            run(self._Q)
        # The futile pkill must NEVER run for a quota outage, no matter how
        # many consecutive quota cycles elapse.
        assert calls["killed"] == 0
        assert runner._consecutive_no_decisions == 0

    def test_recovery_sends_notice_and_rearms(self, cycle_spy):
        calls, run = cycle_spy
        run(self._Q)                       # outage → alarmed
        assert runner._quota_alert_active is True
        run({"status": "HOLD", "decision": {"action": "HOLD"},
             "quota_exhausted": False})    # real decision → recovered
        assert runner._quota_alert_active is False
        assert any("RECOVERED" in m for m in calls["send"])
        # Re-armed: a fresh outage alarms again.
        run(self._Q)
        assert calls["quota_alert"] == 2

    def test_non_quota_no_decision_holds_alarmed_state(self, cycle_spy):
        """A plain timeout after a quota outage is NOT proof the quota is
        back — stay alarmed (no premature 'recovered'), and the ordinary
        breaker still counts that timeout."""
        calls, run = cycle_spy
        run(self._Q)
        run({"status": "NO_DECISION", "decision": None, "quota_exhausted": False})
        assert runner._quota_alert_active is True              # not cleared
        assert not any("RECOVERED" in m for m in calls["send"])
        assert runner._consecutive_no_decisions == 1           # breaker counts it
