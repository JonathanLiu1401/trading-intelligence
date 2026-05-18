"""Tests for paper_trader.reporter — Discord message formatting and the
subprocess shim for openclaw.

We never actually invoke openclaw — instead we patch shutil.which to pretend
it's missing (returns False/print path) or patch subprocess.run to simulate
success/failure/timeout. Message-formatting tests assert that the text body
contains the right fields.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone

from paper_trader import reporter
from paper_trader import store as store_mod
from paper_trader.store import Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """A real Store backed by a temp DB (mirrors test_core_strategy.py)."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


class TestSend:
    def test_returns_false_when_openclaw_missing(self, monkeypatch, capsys):
        # "Genuinely unresolvable" now means all three resolver steps fail:
        # no OPENCLAW_BIN override, not on PATH, and no fallback candidate on
        # disk. (Before the robust-resolver feature this was just which→None;
        # the assertions below — False + logged would-send — are unchanged.)
        monkeypatch.delenv("OPENCLAW_BIN", raising=False)
        monkeypatch.setattr(reporter.shutil, "which", lambda name: None)
        monkeypatch.setattr(reporter, "_openclaw_fallback_candidates", lambda: [])
        ok = reporter._send("hello")
        assert ok is False
        # And it logs what it would have sent so we can debug offline.
        out = capsys.readouterr().out
        assert "would send" in out

    def test_returns_true_on_zero_exit_code(self, monkeypatch):
        monkeypatch.setattr(reporter.shutil, "which", lambda name: "/usr/bin/openclaw")
        fake = MagicMock()
        fake.returncode = 0
        fake.stderr = ""
        monkeypatch.setattr(reporter.subprocess, "run", lambda *a, **k: fake)
        assert reporter._send("hi") is True

    def test_returns_false_on_nonzero_exit_code(self, monkeypatch, capsys):
        monkeypatch.setattr(reporter.shutil, "which", lambda name: "/usr/bin/openclaw")
        fake = MagicMock()
        fake.returncode = 2
        fake.stderr = "boom"
        monkeypatch.setattr(reporter.subprocess, "run", lambda *a, **k: fake)
        assert reporter._send("hi") is False
        assert "openclaw failed" in capsys.readouterr().out

    def test_timeout_returns_false(self, monkeypatch, capsys):
        monkeypatch.setattr(reporter.shutil, "which", lambda name: "/usr/bin/openclaw")

        def _raise(*a, **k):
            raise subprocess.TimeoutExpired(cmd="openclaw", timeout=60)

        monkeypatch.setattr(reporter.subprocess, "run", _raise)
        assert reporter._send("hi") is False
        assert "timeout" in capsys.readouterr().out

    def test_generic_exception_returns_false(self, monkeypatch, capsys):
        monkeypatch.setattr(reporter.shutil, "which", lambda name: "/usr/bin/openclaw")

        def _raise(*a, **k):
            raise OSError("permission denied")

        monkeypatch.setattr(reporter.subprocess, "run", _raise)
        assert reporter._send("hi") is False
        assert "exception" in capsys.readouterr().out

    def test_subprocess_env_path_includes_openclaw_bin_dir(self, monkeypatch):
        """Regression lock for the 2026-05-17 silent-Discord outage: openclaw
        is a Node script (`#!/usr/bin/env node`); under systemd's minimal
        PATH `env node` fails and every message is dropped. `_send` must run
        the subprocess with PATH prefixed by the resolved binary's directory
        (where nvm colocates `node`)."""
        bin_path = "/home/zeph/.nvm/versions/node/v24.15.0/bin/openclaw"
        monkeypatch.setattr(reporter, "_resolve_openclaw", lambda: bin_path)
        # Simulate the systemd-minimal PATH that does NOT include the nvm bin.
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        captured = {}

        def _fake_run(*a, **k):
            captured["env"] = k.get("env")
            fake = MagicMock()
            fake.returncode = 0
            fake.stderr = ""
            return fake

        monkeypatch.setattr(reporter.subprocess, "run", _fake_run)
        assert reporter._send("hi") is True
        env = captured["env"]
        assert env is not None, "subprocess.run must be given an explicit env"
        path_entries = env["PATH"].split(os.pathsep)
        assert path_entries[0] == "/home/zeph/.nvm/versions/node/v24.15.0/bin"
        # The original entries are preserved after the prepended bin dir.
        assert "/usr/bin" in path_entries and "/bin" in path_entries

    def test_subprocess_env_path_handles_empty_inherited_path(self, monkeypatch):
        """A missing/empty inherited PATH must not crash or produce a stray
        leading separator-only entry."""
        monkeypatch.setattr(reporter, "_resolve_openclaw", lambda: "/opt/oc/bin/openclaw")
        monkeypatch.delenv("PATH", raising=False)
        captured = {}

        def _fake_run(*a, **k):
            captured["env"] = k.get("env")
            fake = MagicMock()
            fake.returncode = 0
            fake.stderr = ""
            return fake

        monkeypatch.setattr(reporter.subprocess, "run", _fake_run)
        assert reporter._send("hi") is True
        assert captured["env"]["PATH"].split(os.pathsep)[0] == "/opt/oc/bin"


@pytest.fixture
def fresh_notify_state(monkeypatch):
    """Isolate the module-global delivery-health tracker per test."""
    monkeypatch.setattr(reporter, "_notify_state", {
        "last_attempt_ts": None, "last_ok_ts": None, "last_result": None,
        "consecutive_failures": 0, "last_error": "",
    })
    return reporter


class TestNotifyHealth:
    """Discord delivery-health tracker — turns the silent-channel class of
    failure (the 2026-05-17 `env node` outage) into a visible verdict."""

    def test_unknown_before_any_send(self, fresh_notify_state):
        h = reporter.notify_health()
        assert h["verdict"] == "UNKNOWN"
        assert h["consecutive_failures"] == 0
        assert h["restart_recommended"] is False
        assert h["last_ok_ts"] is None

    def test_success_marks_healthy_and_resets_failures(self, fresh_notify_state):
        reporter._record_send_outcome(False, "boom")
        reporter._record_send_outcome(False, "boom")
        assert reporter.notify_health()["consecutive_failures"] == 2
        reporter._record_send_outcome(True)
        h = reporter.notify_health()
        assert h["verdict"] == "HEALTHY"
        assert h["consecutive_failures"] == 0
        assert h["last_error"] == ""
        assert h["last_ok_ts"] is not None
        assert h["restart_recommended"] is False

    def test_failure_marks_degraded_and_counts(self, fresh_notify_state):
        reporter._record_send_outcome(False, "/usr/bin/env: 'node': No such file")
        h = reporter.notify_health()
        assert h["verdict"] == "DEGRADED"
        assert h["consecutive_failures"] == 1
        assert "node" in h["last_error"]
        assert h["restart_recommended"] is False  # <3 failures
        assert "1 consecutive send failure," in h["headline"]

    def test_restart_recommended_after_three_consecutive_failures(self, fresh_notify_state):
        for _ in range(3):
            reporter._record_send_outcome(False, "rc=1")
        h = reporter.notify_health()
        assert h["verdict"] == "DEGRADED"
        assert h["consecutive_failures"] == 3
        assert h["restart_recommended"] is True
        assert "3 consecutive send failures," in h["headline"]

    def test_send_failure_path_updates_tracker(self, fresh_notify_state, monkeypatch):
        """Driving the real _send failure path (nonzero exit) must flip the
        tracker to DEGRADED with the CLI error captured."""
        monkeypatch.setattr(reporter, "_resolve_openclaw", lambda: "/usr/bin/openclaw")
        fake = MagicMock()
        fake.returncode = 1
        fake.stderr = "/usr/bin/env: 'node': No such file or directory"
        fake.stdout = ""
        monkeypatch.setattr(reporter.subprocess, "run", lambda *a, **k: fake)
        assert reporter._send("x") is False
        h = reporter.notify_health()
        assert h["verdict"] == "DEGRADED"
        assert "node" in h["last_error"]

    def test_send_success_path_updates_tracker(self, fresh_notify_state, monkeypatch):
        monkeypatch.setattr(reporter, "_resolve_openclaw", lambda: "/usr/bin/openclaw")
        fake = MagicMock()
        fake.returncode = 0
        fake.stderr = ""
        fake.stdout = ""
        monkeypatch.setattr(reporter.subprocess, "run", lambda *a, **k: fake)
        assert reporter._send("x") is True
        assert reporter.notify_health()["verdict"] == "HEALTHY"

    def test_unresolvable_binary_path_updates_tracker(self, fresh_notify_state, monkeypatch):
        monkeypatch.setattr(reporter, "_resolve_openclaw", lambda: None)
        assert reporter._send("x") is False
        h = reporter.notify_health()
        assert h["verdict"] == "DEGRADED"
        assert "not resolvable" in h["last_error"]


class TestSendTradeAlert:
    def test_stock_trade_message_format(self, monkeypatch):
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {
            "action": "BUY", "ticker": "NVDA", "qty": 3, "price": 500.0,
            "value": 1500.0, "reason": "earnings beat",
        }
        assert reporter.send_trade_alert(trade) is True
        body = captured[0]
        assert "BUY" in body
        assert "NVDA" in body
        # qty, price, value all appear (formatted).
        assert "3" in body
        assert "500.00" in body
        assert "1500.00" in body
        assert "earnings beat" in body

    def test_option_trade_includes_strike_and_expiry(self, monkeypatch):
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {
            "action": "BUY_CALL", "ticker": "NVDA", "qty": 1, "price": 5.0,
            "value": 500.0, "reason": "",
            "option_type": "call", "strike": 600.0, "expiry": "2026-12-19",
        }
        reporter.send_trade_alert(trade)
        body = captured[0]
        assert "600.0C" in body or "600C" in body
        assert "2026-12-19" in body


class TestSendDecisionLog:
    def test_includes_action_and_pl(self, monkeypatch):
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        summary = {
            "decision": {"action": "BUY", "ticker": "AMD",
                         "confidence": 0.8, "reasoning": "test"},
            "status": "FILLED",
            "detail": "BUY 5 AMD @ 100.00",
            "snapshot": {"cash": 800.0, "total_value": 1200.0},
        }
        reporter.send_decision_log(summary)
        body = captured[0]
        assert "BUY AMD" in body
        # P/L = 1200 - 1000 = +200; pct = +20%
        assert "+200" in body
        assert "20.00%" in body or "+20.00%" in body

    def test_missing_decision_does_not_crash(self, monkeypatch):
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        # The summary has no decision (NO_DECISION cycle). Reporter should
        # still produce a body and not raise.
        reporter.send_decision_log({"status": "NO_DECISION", "snapshot": {}})
        assert len(captured) == 1
        assert "NO_DECISION" in captured[0]


class TestSendDailyCloseBaseline:
    """The daily-close P/L baseline label must track reporter._INITIAL_EQUITY,
    not a hardcoded literal. reporter.py's own header comment makes
    _INITIAL_EQUITY the single source of truth ('A hardcoded copy silently
    desyncs every reported P/L'); the displayed 'vs $X start' string used to
    hardcode $1000 and would lie if INITIAL_CASH ever moved."""

    def _wire(self, monkeypatch, total_value, baseline):
        captured = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter, "_INITIAL_EQUITY", baseline)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)
        fake_store = MagicMock()
        fake_store.get_portfolio.return_value = {
            "total_value": total_value, "cash": total_value,
        }
        fake_store.open_positions.return_value = []
        fake_store.recent_trades.return_value = []
        monkeypatch.setattr(reporter, "get_store", lambda: fake_store)
        return captured

    def test_baseline_label_tracks_initial_equity(self, monkeypatch):
        # Baseline moved to $2000. P/L on $2200 equity must read +$200 / +10%
        # against a 'vs $2000 start' label — never the stale '$1000'.
        captured = self._wire(monkeypatch, total_value=2200.0, baseline=2000.0)
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "vs $2000 start" in body
        assert "vs $1000 start" not in body
        # And the numbers use the same baseline (pl = 2200-2000, pct = 10%).
        assert "+200.00" in body
        assert "+10.00%" in body

    def test_default_baseline_still_renders(self, monkeypatch):
        captured = self._wire(monkeypatch, total_value=1050.0, baseline=1000.0)
        reporter.send_daily_close()
        body = captured[0]
        assert "vs $1000 start" in body
        assert "+50.00" in body
        assert "+5.00%" in body


class TestSendDailyClosePnlReal:
    """`send_daily_close` reports a same-day realized P/L on a *cash-flow*
    basis: every SELL* adds its trade `value`, every other action (BUY*)
    subtracts it. The trade `value` itself is written by `store.record_trade`
    with the option ×100 contract multiplier. Both halves of that contract
    were unlocked — only the baseline-label was tested. A sign flip
    (`.startswith("SELL")` → `"BUY"`) or a dropped ×100 in `record_trade`
    would ship green without this. One exact-value assertion pins both:
    if ×100 were missing on options the total would be -449.50, not -400.00;
    if the sign were inverted it would be +400.00.
    """

    def _run(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Mixed same-day ledger. record_trade computes value = qty*price*1
        # for stock, qty*price*100 for options.
        fresh_store.record_trade("NVDA", "BUY", 10, 100.0)          # value 1000 → -1000
        fresh_store.record_trade("NVDA", "SELL", 5, 110.0)          # value  550 →  +550
        fresh_store.record_trade("NVDA", "BUY_CALL", 1, 2.50,
                                 expiry="2026-12-19", strike=600.0,
                                 option_type="call")               # value  250 →  -250
        fresh_store.record_trade("NVDA", "SELL_CALL", 1, 3.00,
                                 expiry="2026-12-19", strike=600.0,
                                 option_type="call")               # value  300 →  +300
        assert reporter.send_daily_close() is True
        return captured[0]

    def test_realized_pl_cash_flow_sign_and_option_multiplier(
            self, fresh_store, monkeypatch):
        body = self._run(fresh_store, monkeypatch)
        # -1000 + 550 - 250 + 300 = -400.00 exactly.
        assert "Realized P/L (today, cash flow basis)  $-400.00" in body
        # Both option legs are buy/sell-classified; all four count as "today".
        assert "Trades today   4" in body
        # Guard against the two regressions explicitly.
        assert "$+400.00" not in body      # sign not inverted
        assert "$-449.50" not in body      # option ×100 not dropped


class TestSendDailyCloseRealizedRoundTrips:
    """`send_daily_close` now also emits a *true* realized-P/L line driven by
    the `build_round_trips` single source of truth (AGENTS.md invariant #10):
    only round-trips that *closed today* count, paired BUY→SELL, so it answers
    "what did I lock in today?" — distinct from the existing cash-flow line.

    The asserted values pin the contract exactly:
      * NVDA 10@100 → 10@112  : closed today, +$120.00, a WIN
      * MU   5@80   → 5@70    : closed today, -$50.00,  a LOSS
      * AMD  4@50             : opened today, NOT closed — must NOT count
    Net realized = +120 - 50 = +$70.00 over 2 closed trips (1W/1L). A
    regression that counted open positions, dropped the win/loss split, or
    re-derived P&L instead of consuming build_round_trips fails here.
    """

    def _wire(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        return captured

    def test_realized_roundtrip_line_value_and_winloss(
            self, fresh_store, monkeypatch):
        captured = self._wire(fresh_store, monkeypatch)
        fresh_store.record_trade("NVDA", "BUY", 10, 100.0)
        fresh_store.record_trade("NVDA", "SELL", 10, 112.0)   # +120 win
        fresh_store.record_trade("MU", "BUY", 5, 80.0)
        fresh_store.record_trade("MU", "SELL", 5, 70.0)       # -50 loss
        fresh_store.record_trade("AMD", "BUY", 4, 50.0)       # still open
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert ("Realized P/L (today, 2 round-trips closed, 1W/1L)  "
                "$+70.00") in body
        # The cash-flow line is untouched and still present alongside it.
        assert "Realized P/L (today, cash flow basis)" in body

    def test_no_line_when_nothing_closed_today(
            self, fresh_store, monkeypatch):
        captured = self._wire(fresh_store, monkeypatch)
        # Only opens — nothing returns to flat, so no round-trip closes.
        fresh_store.record_trade("NVDA", "BUY", 3, 100.0)
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "round-trip" not in body
        # Cash-flow line still renders (it counts opens too).
        assert "Realized P/L (today, cash flow basis)" in body

    def test_singular_round_trip_grammar(self, fresh_store, monkeypatch):
        captured = self._wire(fresh_store, monkeypatch)
        fresh_store.record_trade("LITE", "BUY", 2, 50.0)
        fresh_store.record_trade("LITE", "SELL", 2, 55.0)     # +10 win
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert ("Realized P/L (today, 1 round-trip closed, 1W/0L)  "
                "$+10.00") in body


class TestBehaviouralBlock:
    """`reporter._behavioural_block()` composes `build_trader_scorecard`
    *verbatim* (single source of truth, AGENTS.md invariant #10 — it must
    never re-derive a verdict) into a compact Discord block, and follows the
    reporter's failure contract: a builder/store fault degrades to an empty
    string ('no behavioural block'), never an exception ('no Discord
    summary'). It mirrors the unified `_fetch_scorecard` chat-line contract:
    NO_DATA / ERROR / None state is suppressed; a mature verdict is shown."""

    _SCORECARD = {
        "state": "FLAGS_PRESENT",
        "headline": "4 of 5 behavioural checks flagging: "
                    "PAYOFF_TRAP, CHURNING, PINNED, SELECTION_DRAG.",
        "focus": {
            "name": "trade_asymmetry",
            "label": "PAYOFF_TRAP",
            "theme": "EXIT_DISCIPLINE",
            "headline": "Payoff 0.15:1 vs 0.83 breakeven — negative "
                        "expectancy; the desk is in a payoff trap.",
        },
        "concordance": [
            {"theme": "EXIT_DISCIPLINE", "count": 2,
             "labels": ["PAYOFF_TRAP", "CHURNING"],
             "checks": ["trade_asymmetry", "churn"]},
        ],
        "flags": [], "checks": [], "n_flags": 4, "n_ok": 1,
    }

    def _patch_builder(self, monkeypatch, ret=None, raises=False):
        import paper_trader.analytics.trader_scorecard as tsc

        def fake(*a, **k):
            if raises:
                raise RuntimeError("builder boom")
            return ret

        monkeypatch.setattr(tsc, "build_trader_scorecard", fake)
        fake_store = MagicMock()
        fake_store.get_portfolio.return_value = {"total_value": 970.0,
                                                 "cash": 6.0}
        fake_store.open_positions.return_value = []
        fake_store.recent_trades.return_value = []
        fake_store.recent_decisions.return_value = []
        fake_store.equity_curve.return_value = []
        monkeypatch.setattr(reporter, "get_store", lambda: fake_store)
        return fake_store

    def test_composes_state_headline_focus_concordance_verbatim(
            self, monkeypatch):
        self._patch_builder(monkeypatch, ret=self._SCORECARD)
        block = reporter._behavioural_block()
        # State surfaced.
        assert "FLAGS_PRESENT" in block
        # Headline forwarded verbatim — not re-summarised.
        assert ("4 of 5 behavioural checks flagging: "
                "PAYOFF_TRAP, CHURNING, PINNED, SELECTION_DRAG.") in block
        # Highest-precedence focus: builder name + its own headline verbatim.
        assert "trade_asymmetry" in block
        assert ("Payoff 0.15:1 vs 0.83 breakeven — negative expectancy; "
                "the desk is in a payoff trap.") in block
        # Concordance: the factual count + theme + the builders' own labels.
        assert "EXIT_DISCIPLINE" in block
        assert "PAYOFF_TRAP" in block and "CHURNING" in block

    def test_empty_string_when_state_no_data(self, monkeypatch):
        self._patch_builder(monkeypatch, ret={"state": "NO_DATA",
                                              "headline": "No mature "
                                              "behavioural history yet."})
        assert reporter._behavioural_block() == ""

    def test_empty_string_when_builder_raises(self, monkeypatch):
        # Never raises — the failure mode is 'no block', never 'no summary'.
        self._patch_builder(monkeypatch, raises=True)
        assert reporter._behavioural_block() == ""

    def test_aligned_healthy_is_shown(self, monkeypatch):
        # A clean desk is also worth telling the operator (mirrors the
        # unified _fetch_scorecard contract: only NO_DATA/ERROR suppressed).
        self._patch_builder(monkeypatch, ret={
            "state": "ALIGNED_HEALTHY",
            "headline": "All 3 mature behavioural checks healthy.",
            "focus": None, "concordance": [],
        })
        block = reporter._behavioural_block()
        assert "ALIGNED_HEALTHY" in block
        assert "All 3 mature behavioural checks healthy." in block


class TestSendHourlyBehavioural:
    """The behavioural block is appended to the hourly summary, but a builder
    fault must never suppress the summary itself (reporter failure contract)."""

    def _wire(self, monkeypatch, scorecard_ret=None, scorecard_raises=False):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)
        fake_store = MagicMock()
        fake_store.get_portfolio.return_value = {"total_value": 970.0,
                                                 "cash": 6.0}
        fake_store.open_positions.return_value = []
        fake_store.recent_trades.return_value = []
        fake_store.recent_decisions.return_value = []
        fake_store.equity_curve.return_value = []
        monkeypatch.setattr(reporter, "get_store", lambda: fake_store)
        import paper_trader.analytics.trader_scorecard as tsc

        def fake(*a, **k):
            if scorecard_raises:
                raise RuntimeError("builder boom")
            return scorecard_ret

        monkeypatch.setattr(tsc, "build_trader_scorecard", fake)
        return captured

    def test_hourly_includes_behavioural_block_when_flags_present(
            self, monkeypatch):
        captured = self._wire(monkeypatch, scorecard_ret={
            "state": "FLAGS_PRESENT",
            "headline": "2 of 5 behavioural checks flagging: CHURNING, PINNED.",
            "focus": {"name": "churn", "label": "CHURNING",
                      "theme": "EXIT_DISCIPLINE",
                      "headline": "0.26d median hold — churning."},
            "concordance": [],
        })
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        # Normal summary still intact.
        assert "Equity" in body and "$970.00" in body
        # Behavioural verdict surfaced verbatim.
        assert "2 of 5 behavioural checks flagging: CHURNING, PINNED." in body
        assert "0.26d median hold — churning." in body

    def test_hourly_still_sends_when_builder_raises(self, monkeypatch):
        captured = self._wire(monkeypatch, scorecard_raises=True)
        # Summary must still send — builder fault degrades to no block.
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "Equity" in body and "$970.00" in body

    def test_daily_close_includes_behavioural_block(self, monkeypatch):
        captured = self._wire(monkeypatch, scorecard_ret={
            "state": "FLAGS_PRESENT",
            "headline": "1 of 5 behavioural checks flagging: PAYOFF_TRAP.",
            "focus": None, "concordance": [],
        })
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "DAILY CLOSE" in body
        assert "1 of 5 behavioural checks flagging: PAYOFF_TRAP." in body


class TestPortfolioLines:
    def test_stock_line_format(self):
        positions = [{
            "ticker": "AMD", "type": "stock", "qty": 5,
            "avg_cost": 100.0, "current_price": 110.0, "unrealized_pl": 50.0,
        }]
        out = reporter._portfolio_lines(positions)
        assert len(out) == 1
        line = out[0]
        assert "AMD" in line
        assert "5" in line  # qty
        assert "100.00" in line  # avg
        assert "110.00" in line  # now
        assert "+50.00" in line  # P/L (must be signed)

    def test_option_line_includes_strike(self):
        positions = [{
            "ticker": "NVDA", "type": "call", "qty": 2,
            "avg_cost": 5.0, "strike": 600.0, "expiry": "2026-12-19",
            "current_price": 7.0, "unrealized_pl": 400.0,
        }]
        out = reporter._portfolio_lines(positions)
        assert "NVDA CALL600" in out[0] or "NVDA CALL 600" in out[0] or "600" in out[0]
        assert "2026-12-19" in out[0]
        assert "+400.00" in out[0]

    def test_stale_mark_annotated_when_flagged(self):
        # A position whose live price was unavailable (stale_mark True) must
        # be visibly flagged so the operator does not read the $0.00 P/L as a
        # genuine flat position (the MU live case).
        positions = [{
            "ticker": "MU", "type": "stock", "qty": 0.5,
            "avg_cost": 724.12, "current_price": 724.12, "unrealized_pl": 0.0,
            "stale_mark": True,
        }]
        line = reporter._portfolio_lines(positions)[0]
        assert "STALE" in line.upper()

    def test_no_stale_annotation_when_absent_or_false(self):
        # Backward-compat: open_positions() table rows carry no stale_mark
        # key — output must be byte-identical to before (no annotation).
        rows = [
            {"ticker": "AMD", "type": "stock", "qty": 5, "avg_cost": 100.0,
             "current_price": 110.0, "unrealized_pl": 50.0},  # key absent
            {"ticker": "LITE", "type": "stock", "qty": 1, "avg_cost": 90.0,
             "current_price": 95.0, "unrealized_pl": 5.0,
             "stale_mark": False},  # key present but False
        ]
        out = reporter._portfolio_lines(rows)
        assert all("STALE" not in ln.upper() for ln in out)


class TestClassifyDecisionOutcome:
    """`decisions.action_taken` is free text (AGENTS.md invariant #11).
    The bucket order is load-bearing: a FILLED/BLOCKED verb line also
    contains its own BUY/SELL verb and must NOT be misread as `hold`."""

    def test_filled_not_misread_as_hold(self):
        # 'BUY NVDA → FILLED' must be filled, never other/hold.
        assert reporter._classify_decision_outcome("BUY NVDA → FILLED") == "filled"
        assert reporter._classify_decision_outcome("SELL LITE → FILLED") == "filled"

    def test_hold(self):
        assert reporter._classify_decision_outcome("HOLD MU → HOLD") == "hold"
        # REBALANCE is treated as HOLD by _execute — still classified hold.
        assert reporter._classify_decision_outcome("REBALANCE X → HOLD") == "hold"

    def test_no_decision_has_no_arrow(self):
        assert reporter._classify_decision_outcome("NO_DECISION") == "no_decision"

    def test_blocked(self):
        assert reporter._classify_decision_outcome("SELL X → BLOCKED") == "blocked"

    def test_none_and_empty_and_unknown_are_other(self):
        assert reporter._classify_decision_outcome(None) == "other"
        assert reporter._classify_decision_outcome("") == "other"
        assert reporter._classify_decision_outcome("WAT") == "other"

    def test_case_insensitive(self):
        assert reporter._classify_decision_outcome("buy nvda → filled") == "filled"


class TestActivityCounts:
    """At-or-after a lexically-comparable ISO cutoff (the store's own
    UTC isoformat). A row whose ts == cutoff is INCLUDED (`< since` excludes
    only strictly-older rows); strictly-older rows are dropped."""

    _DECISIONS = [  # newest-first, as store.recent_decisions() returns
        {"timestamp": "2026-05-17T09:30:00+00:00", "action_taken": "BUY NVDA → FILLED"},
        {"timestamp": "2026-05-17T09:00:00+00:00", "action_taken": "HOLD MU → HOLD"},
        {"timestamp": "2026-05-17T08:30:00+00:00", "action_taken": "NO_DECISION"},
        {"timestamp": "2026-05-17T08:00:00+00:00", "action_taken": "SELL X → BLOCKED"},
        {"timestamp": "2026-05-17T06:00:00+00:00", "action_taken": "HOLD Y → HOLD"},
    ]

    def test_window_excludes_strictly_older(self):
        c = reporter._activity_counts(self._DECISIONS, "2026-05-17T08:15:00+00:00")
        # 09:30 filled, 09:00 hold, 08:30 no_decision in window;
        # 08:00 blocked and 06:00 hold are strictly older → excluded.
        assert c == {"filled": 1, "hold": 1, "no_decision": 1,
                     "blocked": 0, "other": 0}

    def test_cutoff_equal_timestamp_is_included(self):
        # since == an 08:00 row's ts: it is NOT < since, so it counts.
        c = reporter._activity_counts(self._DECISIONS, "2026-05-17T08:00:00+00:00")
        assert c == {"filled": 1, "hold": 1, "no_decision": 1,
                     "blocked": 1, "other": 0}

    def test_all_in_window(self):
        c = reporter._activity_counts(self._DECISIONS, "2026-05-17T00:00:00+00:00")
        assert c == {"filled": 1, "hold": 2, "no_decision": 1,
                     "blocked": 1, "other": 0}

    def test_empty(self):
        assert reporter._activity_counts([], "2026-05-17T00:00:00+00:00") == {
            "filled": 0, "hold": 0, "no_decision": 0, "blocked": 0, "other": 0}


class TestMovers:
    def test_best_and_worst_picked_by_pl(self):
        positions = [
            {"ticker": "NVDA", "unrealized_pl": -12.5},
            {"ticker": "LITE", "unrealized_pl": 30.0},
            {"ticker": "MU", "unrealized_pl": 4.0},
            {"ticker": "AMD", "unrealized_pl": None},  # non-numeric → filtered
        ]
        best, worst = reporter._movers(positions)
        assert best["ticker"] == "LITE" and best["unrealized_pl"] == 30.0
        assert worst["ticker"] == "NVDA" and worst["unrealized_pl"] == -12.5

    def test_single_position_best_is_worst_identity(self):
        positions = [{"ticker": "X", "unrealized_pl": 5.0}]
        best, worst = reporter._movers(positions)
        assert best is worst  # callers use identity to render one line

    def test_no_numeric_positions(self):
        assert reporter._movers([]) == (None, None)
        assert reporter._movers([{"ticker": "X", "unrealized_pl": None}]) == (None, None)


class TestWindowDelta:
    _CURVE = [
        {"timestamp": "2026-05-17T07:00:00+00:00", "total_value": 1000.0, "sp500_price": 5000.0},
        {"timestamp": "2026-05-17T08:00:00+00:00", "total_value": 1010.0, "sp500_price": 5050.0},
        {"timestamp": "2026-05-17T09:00:00+00:00", "total_value": 1030.0, "sp500_price": 5100.0},
    ]

    def test_exact_port_spy_alpha(self):
        d = reporter._window_delta(self._CURVE, "2026-05-17T08:00:00+00:00")
        # base = 08:00 (1010/5050), last = 09:00 (1030/5100)
        assert d["port_pct"] == pytest.approx((1030 / 1010 - 1) * 100)
        assert d["spy_pct"] == pytest.approx((5100 / 5050 - 1) * 100)
        assert d["alpha_pct"] == pytest.approx(d["port_pct"] - d["spy_pct"])

    def test_too_few_points(self):
        assert reporter._window_delta(self._CURVE[:1], "2026-05-17T00:00:00+00:00") is None

    def test_base_is_last_returns_none(self):
        # since after every point but the last → only in-window point is last.
        assert reporter._window_delta(self._CURVE, "2026-05-17T09:00:00+00:00") is None

    def test_missing_spy_degrades_to_port_only(self):
        curve = [
            {"timestamp": "2026-05-17T07:00:00+00:00", "total_value": 1000.0, "sp500_price": None},
            {"timestamp": "2026-05-17T09:00:00+00:00", "total_value": 1100.0, "sp500_price": None},
        ]
        d = reporter._window_delta(curve, "2026-05-17T00:00:00+00:00")
        assert d == {"port_pct": pytest.approx(10.0)}
        assert "spy_pct" not in d and "alpha_pct" not in d


class TestSessionBlock:
    """End-to-end on a real temp Store: the SESSION block reports the exact
    decision-activity mix, the correct best/worst mover, and the exact
    portfolio-vs-SPY window delta — and degrades to '' (never raises) on a
    store fault, the reporter failure contract."""

    def test_composed_block_exact(self, fresh_store):
        fresh_store.record_decision(True, 5, "BUY NVDA → FILLED", "x", 1000, 500)
        fresh_store.record_decision(True, 5, "HOLD MU → HOLD", "x", 1000, 500)
        fresh_store.record_decision(True, 5, "NO_DECISION", "x", 1000, 500)
        fresh_store.upsert_position("NVDA", "stock", 2, 100.0)
        fresh_store.upsert_position("LITE", "stock", 1, 50.0)
        ids = {p["ticker"]: p["id"] for p in fresh_store.open_positions()}
        fresh_store.update_position_marks({
            ids["NVDA"]: (110.0, 20.0),   # +20 → best
            ids["LITE"]: (40.0, -10.0),   # -10 → worst
        })
        fresh_store.record_equity_point(1000.0, 500.0, 5000.0)
        fresh_store.record_equity_point(1030.0, 470.0, 5100.0)

        block = reporter._session_block(fresh_store, 24.0, "24h")
        assert "**SESSION** ◈ last 24h" in block
        assert ("Decisions   3   filled 1  hold 1  no-dec 1  blocked 0"
                in block)
        assert "Best `NVDA` $+20.00  ·  Worst `LITE` $-10.00" in block
        # (1030/1000-1)*100 = +3.00, (5100/5000-1)*100 = +2.00, alpha +1.00
        assert "Δ port `+3.00%`  spy `+2.00%`  alpha `+1.00%`" in block

    def test_single_position_one_mover_line(self, fresh_store):
        fresh_store.upsert_position("NVDA", "stock", 1, 100.0)
        pid = fresh_store.open_positions()[0]["id"]
        fresh_store.update_position_marks({pid: (115.0, 15.0)})
        block = reporter._session_block(fresh_store, 1.0, "1h")
        assert "Only open mover `NVDA` $+15.00" in block
        assert "Worst" not in block

    def test_store_fault_degrades_to_empty(self):
        boom = MagicMock()
        boom.recent_decisions.side_effect = RuntimeError("db gone")
        assert reporter._session_block(boom, 1.0, "1h") == ""

    def test_hourly_summary_includes_session_block(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        fresh_store.record_decision(True, 5, "BUY NVDA → FILLED", "x", 1000, 500)
        fresh_store.record_decision(True, 5, "NO_DECISION", "x", 1000, 500)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**SESSION** ◈ last 1h" in body
        assert "filled 1" in body and "no-dec 1" in body
        # The pre-existing summary is still intact alongside the new block.
        assert "**HOURLY**" in body and "Equity" in body


class TestSingletonLockLine:
    """`_singleton_lock_line` + its hourly-summary wiring — the 2026-05-18
    feature that makes a guard-less (degraded) runner self-report. A
    degraded runner double-trading the shared book was previously invisible
    from every operator surface; the operator lives in Discord."""

    def test_empty_when_lock_acquired(self, monkeypatch):
        from paper_trader import runner
        monkeypatch.setattr(runner, "singleton_lock_state", lambda: {
            "status": "acquired", "holder_pid": 1, "have_lock": True,
            "degraded": False})
        assert reporter._singleton_lock_line() == ""

    def test_warns_when_degraded(self, monkeypatch):
        from paper_trader import runner
        monkeypatch.setattr(runner, "singleton_lock_state", lambda: {
            "status": "degraded", "holder_pid": None, "have_lock": False,
            "degraded": True})
        line = reporter._singleton_lock_line()
        assert "RUNNER DEGRADED" in line
        assert "double-trading" in line
        assert "Restart paper-trader" in line

    def test_degrades_to_empty_on_runner_fault(self, monkeypatch):
        from paper_trader import runner

        def _boom():
            raise RuntimeError("runner introspection blew up")

        monkeypatch.setattr(runner, "singleton_lock_state", _boom)
        # Additive failure contract: a fault drops THIS line, never raises.
        assert reporter._singleton_lock_line() == ""

    def test_hourly_summary_includes_degraded_warning(self, fresh_store,
                                                      monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        from paper_trader import runner
        monkeypatch.setattr(runner, "singleton_lock_state", lambda: {
            "status": "degraded", "holder_pid": None, "have_lock": False,
            "degraded": True})
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "RUNNER DEGRADED" in body
        # Pre-existing summary intact alongside the new warning.
        assert "**HOURLY**" in body and "Equity" in body

    def test_hourly_summary_no_warning_when_acquired(self, fresh_store,
                                                     monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        from paper_trader import runner
        monkeypatch.setattr(runner, "singleton_lock_state", lambda: {
            "status": "acquired", "holder_pid": 99, "have_lock": True,
            "degraded": False})
        assert reporter.send_hourly_summary() is True
        assert "RUNNER DEGRADED" not in captured[0]


class TestHeartbeatLine:
    """`_heartbeat_line` + its hourly/daily wiring — the 2026-05-18 feature
    that routes the runner-heartbeat verdict to Discord so a host-load
    NO_DECISION storm (IDLE_STORM, ``restart_recommended:true``) is no longer
    invisible to the operator who lives in Discord (pass #17 finding #1).

    Composes ``build_runner_heartbeat`` verbatim (single source of truth,
    invariant #10); surfaces only when actionable; a builder/store fault
    drops the line, never the summary (the reporter failure contract)."""

    def _patch_builder(self, monkeypatch, ret=None, raises=False):
        import paper_trader.analytics.runner_heartbeat as rhb

        def fake(*a, **k):
            if raises:
                raise RuntimeError("heartbeat builder boom")
            return ret

        monkeypatch.setattr(rhb, "build_runner_heartbeat", fake)
        fake_store = MagicMock()
        fake_store.recent_decisions.return_value = [
            {"timestamp": "2026-05-18T10:00:00+00:00",
             "action_taken": "NO_DECISION"}
        ]
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        return fake_store

    def test_idle_storm_surfaces_headline_verbatim_with_restart_prefix(
            self, monkeypatch):
        hb = {
            "verdict": "HEALTHY",
            "restart_recommended": True,
            "headline": ("HEALTHY — last decision 11m ago, within the 60m "
                         "market-closed cadence. ⚠ but the last 18 cycles "
                         "were ALL NO_DECISION — the engine is cycling, not "
                         "deciding; a restart may clear a wedged Claude CLI."),
            "decision_efficacy": {"verdict": "IDLE_STORM",
                                  "headline": "IDLE_STORM — the last 18 "
                                  "cycles were ALL NO_DECISION (90% of the "
                                  "last 20)."},
        }
        store = self._patch_builder(monkeypatch, ret=hb)
        line = reporter._heartbeat_line(store)
        assert "**RUNNER** ◈ HEALTHY" in line
        assert "⚠️ RESTART RECOMMENDED — " in line
        # Builder headline forwarded verbatim, not re-summarised.
        assert hb["headline"] in line
        # IDLE_STORM detail is already in the top-level headline → not
        # duplicated as a separate efficacy line.
        assert "efficacy —" not in line

    def test_healthy_producing_is_suppressed(self, monkeypatch):
        hb = {
            "verdict": "HEALTHY",
            "restart_recommended": False,
            "headline": "HEALTHY — last decision 2m ago, within cadence.",
            "decision_efficacy": {"verdict": "PRODUCING",
                                  "headline": "PRODUCING — 18/20 decided."},
        }
        store = self._patch_builder(monkeypatch, ret=hb)
        # Nothing actionable → no hourly noise (the lying-green-light guard).
        assert reporter._heartbeat_line(store) == ""

    def test_stalled_surfaces_with_restart_prefix(self, monkeypatch):
        hb = {
            "verdict": "STALLED",
            "restart_recommended": True,
            "headline": ("STALLED — no decision in 3h (>2x the 1h expected "
                         "market-open cadence); the trading loop appears "
                         "dead. Restart paper-trader."),
            "decision_efficacy": None,
        }
        store = self._patch_builder(monkeypatch, ret=hb)
        line = reporter._heartbeat_line(store)
        assert "**RUNNER** ◈ STALLED" in line
        assert "⚠️ RESTART RECOMMENDED — " in line
        assert hb["headline"] in line

    def test_lagging_surfaces_without_restart_prefix(self, monkeypatch):
        hb = {
            "verdict": "LAGGING",
            "restart_recommended": False,
            "headline": ("LAGGING — last decision 80m ago (>1.25x the 60m "
                         "market-closed cadence); the loop is slow."),
            "decision_efficacy": {"verdict": "PRODUCING", "headline": "x"},
        }
        store = self._patch_builder(monkeypatch, ret=hb)
        line = reporter._heartbeat_line(store)
        assert "**RUNNER** ◈ LAGGING" in line
        assert "RESTART RECOMMENDED" not in line  # LAGGING ≠ restart
        assert hb["headline"] in line

    def test_degraded_efficacy_surfaces_with_efficacy_subline(
            self, monkeypatch):
        hb = {
            "verdict": "HEALTHY",
            "restart_recommended": False,
            "headline": "HEALTHY — last decision 5m ago, within cadence.",
            "decision_efficacy": {
                "verdict": "DEGRADED",
                "headline": ("DEGRADED — 60% of the last 20 cycles were "
                             "NO_DECISION (latest still produced a "
                             "decision); throughput is impaired."),
            },
        }
        store = self._patch_builder(monkeypatch, ret=hb)
        line = reporter._heartbeat_line(store)
        assert "**RUNNER** ◈ HEALTHY" in line
        assert "RESTART RECOMMENDED" not in line
        # DEGRADED detail is NOT in the top-level headline → surfaced as its
        # own additive line.
        assert "efficacy — DEGRADED — 60% of the last 20 cycles" in line

    def test_degrades_to_empty_when_builder_raises(self, monkeypatch):
        store = self._patch_builder(monkeypatch, raises=True)
        # Never raises — failure mode is 'no line', never 'no summary'.
        assert reporter._heartbeat_line(store) == ""

    def test_no_drift_real_builder_idle_storm(self, fresh_store, monkeypatch):
        """End-to-end on the REAL builder + a real Store: 18 NO_DECISION rows
        with a fresh newest timestamp → liveness HEALTHY, efficacy
        IDLE_STORM, restart_recommended. The Discord headline must equal the
        builder's own headline verbatim (no re-derivation — invariant #10)."""
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        for _ in range(18):
            fresh_store.record_decision(False, 0, "NO_DECISION",
                                        "claude timeout", 1000.0, 1000.0)
        from paper_trader.analytics.runner_heartbeat import (
            build_runner_heartbeat)
        decs = fresh_store.recent_decisions(20)
        expected = build_runner_heartbeat(
            decs[0]["timestamp"], False,
            recent_actions=[d["action_taken"] for d in decs])
        assert expected["restart_recommended"] is True
        assert expected["decision_efficacy"]["verdict"] == "IDLE_STORM"

        line = reporter._heartbeat_line(fresh_store)
        assert expected["headline"] in line          # verbatim, no drift
        assert "⚠️ RESTART RECOMMENDED — " in line
        assert "**RUNNER** ◈ HEALTHY" in line

    def test_hourly_summary_includes_idle_storm(self, fresh_store,
                                                monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        for _ in range(18):
            fresh_store.record_decision(False, 0, "NO_DECISION",
                                        "claude timeout", 1000.0, 1000.0)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**RUNNER** ◈" in body
        assert "RESTART RECOMMENDED" in body
        # Pre-existing summary intact alongside the new block.
        assert "**HOURLY**" in body and "Equity" in body

    def test_daily_close_includes_idle_storm(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        for _ in range(18):
            fresh_store.record_decision(False, 0, "NO_DECISION",
                                        "claude timeout", 1000.0, 1000.0)
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "**RUNNER** ◈" in body and "RESTART RECOMMENDED" in body
        assert "**DAILY CLOSE**" in body

    def test_summary_still_sends_when_heartbeat_builder_faults(
            self, fresh_store, monkeypatch):
        """A heartbeat fault drops only its block — the hourly summary itself
        must still send (the reporter 'no block, never no summary' contract).
        """
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        import paper_trader.analytics.runner_heartbeat as rhb

        def _boom(*a, **k):
            raise RuntimeError("builder boom")

        monkeypatch.setattr(rhb, "build_runner_heartbeat", _boom)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body and "Equity" in body
        assert "**RUNNER** ◈" not in body


class TestEquityIntegrityLine:
    """`_equity_integrity_line` + its hourly/daily wiring — routes the
    equity-curve integrity verdict (CORRUPT/SUSPECT) to Discord so a silent
    P&L-history corruption is no longer invisible to the operator who lives
    in Discord (it was dashboard-only via /api/equity-integrity).

    Composes ``build_equity_integrity`` verbatim (single source of truth,
    invariant #10); surfaces only when actionable; a builder/store fault
    drops the line, never the summary (the reporter failure contract)."""

    def _patch_builder(self, monkeypatch, ret=None, raises=False):
        import paper_trader.analytics.equity_integrity as eim

        def fake(*a, **k):
            if raises:
                raise RuntimeError("equity-integrity builder boom")
            return ret

        monkeypatch.setattr(eim, "build_equity_integrity", fake)
        fake_store = MagicMock()
        fake_store.equity_curve.return_value = []
        fake_store.recent_trades.return_value = []
        return fake_store

    def test_corrupt_surfaces_headline_verbatim(self, monkeypatch):
        ei = {
            "verdict": "CORRUPT",
            "headline": ("Recorded equity is CORRUPT across 9 points: 1 "
                         "negative-cash point(s) (min $-5.0) — the book was "
                         "over-drawn; P&L history (drawdown, benchmark, "
                         "Sharpe, hourly P/L) is unreliable."),
        }
        store = self._patch_builder(monkeypatch, ret=ei)
        line = reporter._equity_integrity_line(store)
        assert "⚠️ **EQUITY INTEGRITY** ◈ CORRUPT" in line
        # Builder headline forwarded verbatim, never re-derived (invariant #10).
        assert ei["headline"] in line

    def test_suspect_surfaces_headline_verbatim(self, monkeypatch):
        ei = {
            "verdict": "SUSPECT",
            "headline": ("1 unexplained equity jump(s) >=8% with no trade in "
                         "the window across 5 points; largest +20.00% "
                         "($+200.00) — likely a mismark."),
        }
        store = self._patch_builder(monkeypatch, ret=ei)
        line = reporter._equity_integrity_line(store)
        assert "⚠️ **EQUITY INTEGRITY** ◈ SUSPECT" in line
        assert ei["headline"] in line

    def test_clean_is_suppressed(self, monkeypatch):
        ei = {"verdict": "CLEAN",
              "headline": "Equity curve consistent across 40 points."}
        store = self._patch_builder(monkeypatch, ret=ei)
        # A trustworthy curve adds no hourly noise (lying-green-light guard).
        assert reporter._equity_integrity_line(store) == ""

    def test_no_data_is_suppressed(self, monkeypatch):
        ei = {"verdict": "NO_DATA", "headline": "Only 1 usable point."}
        store = self._patch_builder(monkeypatch, ret=ei)
        assert reporter._equity_integrity_line(store) == ""

    def test_error_and_nondict_are_suppressed(self, monkeypatch):
        store = self._patch_builder(
            monkeypatch, ret={"verdict": "ERROR", "headline": "boom"})
        assert reporter._equity_integrity_line(store) == ""
        store2 = self._patch_builder(monkeypatch, ret=None)
        assert reporter._equity_integrity_line(store2) == ""

    def test_degrades_to_empty_when_builder_raises(self, monkeypatch):
        store = self._patch_builder(monkeypatch, raises=True)
        # Never raises — failure mode is 'no line', never 'no summary'.
        assert reporter._equity_integrity_line(store) == ""

    def test_no_drift_real_builder_corrupt(self, fresh_store):
        """End-to-end on the REAL builder + a real Store: a negative-cash
        equity point → CORRUPT, and the Discord headline must equal the
        builder's own headline verbatim (no re-derivation — invariant #10)."""
        fresh_store.record_equity_point(1000.0, 500.0, None)
        fresh_store.record_equity_point(1000.0, -5.0, None)   # over-drawn
        fresh_store.record_equity_point(1000.0, 500.0, None)
        from paper_trader.analytics.equity_integrity import (
            build_equity_integrity)
        expected = build_equity_integrity(
            fresh_store.equity_curve(limit=5000),
            fresh_store.recent_trades(5000))
        assert expected["verdict"] == "CORRUPT"
        line = reporter._equity_integrity_line(fresh_store)
        assert "⚠️ **EQUITY INTEGRITY** ◈ CORRUPT" in line
        assert expected["headline"] in line          # verbatim, no drift

    def test_no_drift_real_builder_suspect(self, fresh_store):
        """A no-trade +20% jump on a real Store → SUSPECT, headline verbatim.
        fresh_store has zero trades so the jump is genuinely unexplained."""
        fresh_store.record_equity_point(1000.0, 100.0, None)
        fresh_store.record_equity_point(1000.0, 100.0, None)  # flat
        fresh_store.record_equity_point(1200.0, 100.0, None)  # +20%, no trade
        from paper_trader.analytics.equity_integrity import (
            build_equity_integrity)
        expected = build_equity_integrity(
            fresh_store.equity_curve(limit=5000),
            fresh_store.recent_trades(5000))
        assert expected["verdict"] == "SUSPECT"
        line = reporter._equity_integrity_line(fresh_store)
        assert "⚠️ **EQUITY INTEGRITY** ◈ SUSPECT" in line
        assert expected["headline"] in line

    def test_hourly_summary_includes_corrupt_line(self, fresh_store,
                                                  monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        fresh_store.record_equity_point(1000.0, 500.0, None)
        fresh_store.record_equity_point(1000.0, -5.0, None)
        fresh_store.record_equity_point(1000.0, 500.0, None)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**EQUITY INTEGRITY** ◈ CORRUPT" in body
        # Pre-existing summary intact alongside the new block.
        assert "**HOURLY**" in body and "Equity" in body

    def test_daily_close_includes_corrupt_line(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: False)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        fresh_store.record_equity_point(1000.0, 500.0, None)
        fresh_store.record_equity_point(1000.0, -5.0, None)
        fresh_store.record_equity_point(1000.0, 500.0, None)
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "**EQUITY INTEGRITY** ◈ CORRUPT" in body
        assert "**DAILY CLOSE**" in body

    def test_summary_still_sends_when_integrity_builder_faults(
            self, fresh_store, monkeypatch):
        """An integrity-builder fault drops only its block — the hourly
        summary itself must still send (the reporter 'no block, never no
        summary' contract)."""
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        import paper_trader.analytics.equity_integrity as eim

        def _boom(*a, **k):
            raise RuntimeError("builder boom")

        monkeypatch.setattr(eim, "build_equity_integrity", _boom)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body and "Equity" in body
        assert "**EQUITY INTEGRITY**" not in body


class TestAgo:
    """`_ago` bucket boundaries — minutes < 1h, hours < 1d, days beyond."""

    def test_sub_minute_reads_zero_minutes(self):
        assert reporter._ago(0) == "0m"
        assert reporter._ago(59) == "0m"

    def test_minute_and_hour_boundaries(self):
        assert reporter._ago(60) == "1m"
        assert reporter._ago(3599) == "59m"
        assert reporter._ago(3600) == "1h"
        assert reporter._ago(86399) == "23h"

    def test_day_boundary(self):
        assert reporter._ago(86400) == "1d"
        assert reporter._ago(200000) == "2d"

    def test_negative_clamps_to_zero(self):
        assert reporter._ago(-500) == "0m"


class TestFmtTradeStamp:
    """The hourly recent-trade label: today stays bare HH:MM (unchanged),
    older trades gain the date + a relative age so a frozen-but-active-looking
    book is unmissable (the desk's #1 documented pathology)."""

    NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)

    def test_today_is_bare_hhmm(self):
        assert reporter._fmt_trade_stamp(
            "2026-05-18T09:38:08.435126+00:00", now=self.NOW) == "09:38"

    def test_yesterday_gets_date_and_day_age(self):
        # 2026-05-18T12:00 − 2026-05-17T09:38 ≈ 26h → "1d"
        assert reporter._fmt_trade_stamp(
            "2026-05-17T09:38:08+00:00", now=self.NOW) == "05-17 09:38 · 1d ago"

    def test_recent_but_not_today_gets_hour_age(self):
        now = datetime(2026, 5, 18, 1, 0, 0, tzinfo=timezone.utc)
        assert reporter._fmt_trade_stamp(
            "2026-05-17T23:30:00+00:00", now=now) == "05-17 23:30 · 1h ago"

    def test_naive_timestamp_treated_as_utc(self):
        assert reporter._fmt_trade_stamp(
            "2026-05-17T09:38:08", now=self.NOW) == "05-17 09:38 · 1d ago"

    def test_future_different_day_has_no_negative_age(self):
        # Clock skew: a future-dated trade must not render "· -1d ago".
        assert reporter._fmt_trade_stamp(
            "2026-05-19T13:00:00+00:00", now=self.NOW) == "05-19 13:00"

    def test_malformed_degrades_to_clean_sentinel_never_raises(self):
        # store always writes a valid ISO timestamp, so a parse failure is
        # genuinely corrupt data: a clean "??:??" sentinel beats the old
        # raw[11:16] slice (which rendered garbage like "tamp"), and must
        # never raise (the reporter additive-line contract).
        assert reporter._fmt_trade_stamp("not-a-timestamp",
                                         now=self.NOW) == "??:??"
        assert reporter._fmt_trade_stamp(None, now=self.NOW) == "??:??"

    def test_hourly_summary_dates_a_stale_trade(self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5000.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # A trade whose ISO timestamp is two days before "now". record_trade
        # stamps _now(), so inject the row shape send_hourly_summary consumes.
        stale_ts = "2026-05-16T09:38:08+00:00"
        monkeypatch.setattr(fresh_store, "recent_trades", lambda n=5: [
            {"timestamp": stale_ts, "action": "BUY", "qty": 0.5,
             "ticker": "MU", "price": 724.12},
        ])
        # Freeze "now" so the assertion is deterministic regardless of when
        # the suite runs.
        import paper_trader.reporter as rep
        real_dt = rep.datetime

        class _FrozenDT(real_dt):
            @classmethod
            def now(cls, tz=None):
                return real_dt(2026, 5, 18, 12, 0, 0, tzinfo=tz)

        monkeypatch.setattr(rep, "datetime", _FrozenDT)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        # The trade line now carries the date AND a "Nd ago" age, so a
        # 2-day-old fill can no longer be misread as today's.
        assert "05-16 09:38 · 2d ago" in body
        assert "BUY 0.5 MU @ $724.12" in body
