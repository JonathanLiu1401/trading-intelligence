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

    def test_no_impact_line_when_snapshot_missing(self, monkeypatch):
        # Backwards compat: a caller that passes only ``trade`` gets the
        # byte-compatible body (no trailing "post: …" line).
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {"action": "BUY", "ticker": "NVDA", "qty": 1, "price": 100.0,
                 "value": 100.0, "reason": ""}
        reporter.send_trade_alert(trade)
        assert "post:" not in captured[0]


class TestTradeAlertImpactLine:
    """The post-trade book-impact one-liner — the trader-useful payload."""

    def test_buy_shows_lot_weight_and_cash(self, monkeypatch):
        # Post-trade snapshot: NVDA at $100, 5 shares = $500, cash $500,
        # total $1000 → NVDA = 50% of book.
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {"action": "BUY", "ticker": "NVDA", "qty": 5, "price": 100.0,
                 "value": 500.0, "reason": "high conviction",
                 "timestamp": "2026-05-18T16:00:00+00:00"}
        snapshot = {
            "cash": 500.0, "total_value": 1000.0,
            "positions": [{
                "ticker": "NVDA", "type": "stock", "qty": 5, "avg_cost": 100.0,
                "current_price": 100.0, "market_value": 500.0,
            }],
        }
        reporter.send_trade_alert(trade, snapshot=snapshot, store=None)
        body = captured[0]
        assert "post:" in body
        assert "NVDA now 50.0% of book" in body
        assert "cash $500.00" in body

    def test_buy_call_shows_lot_label(self, monkeypatch):
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {
            "action": "BUY_CALL", "ticker": "NVDA", "qty": 1, "price": 5.0,
            "value": 500.0, "reason": "",
            "option_type": "call", "strike": 600.0, "expiry": "2026-12-19",
            "timestamp": "2026-05-18T16:00:00+00:00",
        }
        snapshot = {
            "cash": 500.0, "total_value": 1000.0,
            "positions": [{
                "ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
                "current_price": 5.0, "market_value": 500.0,
                "strike": 600.0, "expiry": "2026-12-19",
            }],
        }
        reporter.send_trade_alert(trade, snapshot=snapshot)
        body = captured[0]
        # Whole-number strike renders without ".0"
        assert "NVDA 600C 2026-12-19" in body
        assert "50.0% of book" in body

    def test_sell_shows_realized_pnl_and_hold(self, monkeypatch, tmp_path):
        # Seed a real store so build_round_trips operates on actual trades.
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            # BUY 5 @ 100 → SELL 5 @ 120 closes a round-trip with +$100, +20%.
            s.record_trade("NVDA", "BUY", 5, 100.0)
            s.record_trade("NVDA", "SELL", 5, 120.0)
            sell_trade = s.recent_trades(1)[0]
            captured = []
            monkeypatch.setattr(reporter, "_send",
                                lambda msg: captured.append(msg) or True)
            snapshot = {
                "cash": 1100.0, "total_value": 1100.0, "positions": [],
            }
            reporter.send_trade_alert(sell_trade, snapshot=snapshot, store=s)
            body = captured[0]
            assert "post:" in body
            # +20% on $500 cost = +$100 realized; cash now $1100.
            assert "realized $+100.00" in body
            assert "+20.0%" in body
            assert "cash $1100.00" in body
        finally:
            s.close()

    def test_sell_partial_close_does_not_invent_pnl(self, monkeypatch, tmp_path):
        # Partial close (still holding qty>0) → no round-trip closed yet,
        # so the alert must NOT manufacture a realized P/L figure. Instead
        # it falls back to "partial — NVDA still X% of book".
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            s.record_trade("NVDA", "BUY", 10, 100.0)
            s.record_trade("NVDA", "SELL", 4, 120.0)   # partial close
            sell_trade = s.recent_trades(1)[0]
            captured = []
            monkeypatch.setattr(reporter, "_send",
                                lambda msg: captured.append(msg) or True)
            snapshot = {
                "cash": 480.0, "total_value": 1080.0,
                "positions": [{
                    "ticker": "NVDA", "type": "stock", "qty": 6,
                    "avg_cost": 100.0, "current_price": 100.0,
                    "market_value": 600.0,
                }],
            }
            reporter.send_trade_alert(sell_trade, snapshot=snapshot, store=s)
            body = captured[0]
            # Must NOT fabricate a realized line for an open round-trip.
            assert "realized" not in body
            # Partial close path surfaces remaining exposure.
            assert "partial" in body.lower()
            assert "NVDA still" in body
        finally:
            s.close()

    def test_zero_total_value_suppresses_line(self, monkeypatch):
        # A book sitting at $0 (impossible in practice, but the guard
        # prevents a div-by-zero or "0.0% of book" misleading text).
        captured = []
        monkeypatch.setattr(reporter, "_send", lambda msg: captured.append(msg) or True)
        trade = {"action": "BUY", "ticker": "NVDA", "qty": 1, "price": 100.0,
                 "value": 100.0, "reason": ""}
        snapshot = {"cash": 0.0, "total_value": 0.0, "positions": []}
        reporter.send_trade_alert(trade, snapshot=snapshot)
        assert "post:" not in captured[0]

    def test_sell_full_close_without_store_does_not_duplicate_cash(
        self, monkeypatch
    ):
        # Full close + no store (so no round-trip lookup) used to emit
        # "closed — cash $X · cash $X" because the fallback branch baked
        # cash into its own token AND the unconditional cash-append fired.
        # Lock the single cash token now.
        captured = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        sell_trade = {"action": "SELL", "ticker": "NVDA", "qty": 5,
                       "price": 120.0, "value": 600.0, "reason": "exit",
                       "timestamp": "2026-05-18T16:00:00+00:00"}
        # Snapshot is post-trade with NVDA fully closed → no NVDA row.
        snapshot = {"cash": 1100.0, "total_value": 1100.0, "positions": []}
        reporter.send_trade_alert(sell_trade, snapshot=snapshot, store=None)
        body = captured[0]
        assert "post:" in body
        assert "closed" in body
        assert body.count("cash $1100.00") == 1, (
            f"expected single cash token, got: {body!r}")

    def test_sell_full_close_with_failing_store_falls_back_cleanly(
        self, monkeypatch
    ):
        # Same path as above, but exercised through a store whose
        # recent_trades raises so the round-trip lookup fails internally.
        # The "closed" branch must still emit exactly one cash token.
        class _FailingStore:
            def recent_trades(self, _n):
                raise RuntimeError("simulated store fault")
        captured = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        sell_trade = {"action": "SELL", "ticker": "AMD", "qty": 3,
                       "price": 200.0, "value": 600.0, "reason": "exit",
                       "timestamp": "2026-05-18T16:00:00+00:00"}
        snapshot = {"cash": 600.0, "total_value": 600.0, "positions": []}
        reporter.send_trade_alert(sell_trade, snapshot=snapshot,
                                  store=_FailingStore())
        body = captured[0]
        assert "post:" in body
        assert "closed" in body
        assert body.count("cash $600.00") == 1
        assert "realized" not in body  # never invent a P/L when the lookup failed

    def test_hold_str_from_days_buckets(self):
        # Locked: <1h → minutes, <1d → fractional hours, ≥1d → fractional days.
        assert reporter._hold_str_from_days(0.0007) == "1m"     # ~1 min
        assert reporter._hold_str_from_days(0.5) == "12.0h"
        assert reporter._hold_str_from_days(2.5) == "2.5d"
        assert reporter._hold_str_from_days(None) == ""
        assert reporter._hold_str_from_days(-1) == ""           # bad data → silent


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

    def test_pct_weight_appended_when_total_value_passed(self):
        # The live hourly/daily callers pass pf['total_value']; the position
        # line must then carry the position's own return % AND its book
        # weight, without dropping any of the prior raw fields.
        positions = [{
            "ticker": "AMD", "type": "stock", "qty": 5,
            "avg_cost": 100.0, "current_price": 110.0, "unrealized_pl": 50.0,
        }]
        line = reporter._portfolio_lines(positions, total_value=1000.0)[0]
        # raw fields still present (backward compat)
        assert "AMD" in line and "100.00" in line and "+50.00" in line
        # +10% return, 5*110/1000 = 55% of book
        assert "+10.0%" in line
        assert "55% bk" in line

    def test_pct_only_when_no_total_value(self):
        # Default (no total) → return % only, no weight token (keeps the
        # existing unit-test callers byte-compatible on the weight axis).
        positions = [{
            "ticker": "AMD", "type": "stock", "qty": 5,
            "avg_cost": 100.0, "current_price": 110.0, "unrealized_pl": 50.0,
        }]
        line = reporter._portfolio_lines(positions)[0]
        assert "+10.0%" in line
        assert "bk" not in line


class TestPosPctWeight:
    """`_pos_pct_weight` — pure per-position return% + book-weight token."""

    def test_canonical_format_locked(self):
        p = {"ticker": "AMD", "type": "stock", "qty": 5,
             "avg_cost": 100.0, "current_price": 110.0}
        assert reporter._pos_pct_weight(p, 1000.0) == "  (+10.0% · 55% bk)"

    def test_negative_return_live_lite_shape(self):
        # The live 2026-05-18 LITE position: 0.61 @ 980.90, mark 872.77,
        # book $900.84 → −11.0% and 59% of the entire book.
        p = {"ticker": "LITE", "type": "stock", "qty": 0.61,
             "avg_cost": 980.8971727591637, "current_price": 872.77001953125}
        tok = reporter._pos_pct_weight(p, 900.8433966064453)
        assert "-11.0%" in tok
        assert "59% bk" in tok

    def test_stale_mark_suppresses_pct_but_keeps_weight(self):
        # stale ⇒ mark == cost; a "+0.0%" next to the STALE flag would lie.
        p = {"ticker": "MU", "type": "stock", "qty": 5,
             "avg_cost": 100.0, "current_price": 100.0, "stale_mark": True}
        tok = reporter._pos_pct_weight(p, 1000.0)
        assert "%" in tok and "+0.0%" not in tok and "-0.0%" not in tok
        assert "50% bk" in tok  # 5*100/1000

    def test_zero_avg_cost_suppresses_pct(self):
        p = {"ticker": "X", "type": "stock", "qty": 1,
             "avg_cost": 0.0, "current_price": 10.0}
        assert reporter._pos_pct_weight(p, None) == ""

    def test_option_weight_uses_100x_multiplier(self):
        # 2 calls @ $5 → mark $7: +40% premium move, notional 7*2*100=1400.
        p = {"ticker": "NVDA", "type": "call", "qty": 2,
             "avg_cost": 5.0, "current_price": 7.0,
             "strike": 600.0, "expiry": "2026-12-19"}
        tok = reporter._pos_pct_weight(p, 2000.0)
        assert "+40.0%" in tok
        assert "70% bk" in tok  # 1400/2000

    def test_nonpositive_total_drops_weight(self):
        p = {"ticker": "AMD", "type": "stock", "qty": 5,
             "avg_cost": 100.0, "current_price": 110.0}
        assert reporter._pos_pct_weight(p, 0.0) == "  (+10.0%)"
        assert reporter._pos_pct_weight(p, -5.0) == "  (+10.0%)"

    def test_nan_current_price_degrades_to_empty(self):
        p = {"ticker": "AMD", "type": "stock", "qty": 5,
             "avg_cost": 100.0, "current_price": float("nan")}
        assert reporter._pos_pct_weight(p, 1000.0) == ""

    def test_sub_one_percent_weight_keeps_one_decimal(self):
        # A tiny tail position must not round to "0% bk" (invisible).
        p = {"ticker": "T", "type": "stock", "qty": 1,
             "avg_cost": 4.0, "current_price": 5.0}
        tok = reporter._pos_pct_weight(p, 1000.0)  # 5/1000 = 0.5%
        assert "0.5% bk" in tok

    def test_worthless_expired_option_shows_minus_100_pct(self):
        # An OTM-expired option settles via strategy._expired_intrinsic at a
        # real ``current_price`` of $0 with ``stale_mark`` False. The trader
        # must see -100% on the wiped contract — previously the strict
        # ``cur > 0`` guard silently suppressed it.
        p = {"ticker": "NVDA", "type": "call", "qty": 1,
             "avg_cost": 5.50, "current_price": 0.0,
             "stale_mark": False,
             "strike": 600.0, "expiry": "2026-05-16"}
        tok = reporter._pos_pct_weight(p, 1000.0)
        assert "-100.0%" in tok
        # Weight clause keeps its strict ``cur > 0`` gate: a 0-mark contract
        # has 0 market value, so emitting "0.0% bk" adds no information.

    def test_stale_mark_with_zero_price_still_suppresses(self):
        # A stale_mark position whose mark fell through to 0 (not a real
        # settlement) must still suppress P/L% — the stale flag is the
        # discriminator, not the price value.
        p = {"ticker": "MU", "type": "stock", "qty": 5,
             "avg_cost": 100.0, "current_price": 0.0,
             "stale_mark": True}
        tok = reporter._pos_pct_weight(p, 1000.0)
        assert "%" not in tok or "% bk" in tok  # only weight, never P/L%
        assert "-100.0%" not in tok


class TestPosHoldAgeToken:
    """`_pos_hold_age_token` — pure per-position hold-age annotation that
    mirrors the Opus prompt's `held=Xd` tag on the Discord surface."""

    def test_minute_bucket(self):
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-18T11:42:00+00:00"}  # 18 min ago
        assert reporter._pos_hold_age_token(p, now=now) == "  held 18m"

    def test_hour_bucket(self):
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-18T07:00:00+00:00"}  # 5h ago
        assert reporter._pos_hold_age_token(p, now=now) == "  held 5h"

    def test_day_bucket(self):
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-15T12:00:00+00:00"}  # 3d ago
        assert reporter._pos_hold_age_token(p, now=now) == "  held 3d"

    def test_sub_minute_is_silent_to_skip_just_filled_noise(self):
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-18T11:59:45+00:00"}  # 15s ago
        assert reporter._pos_hold_age_token(p, now=now) == ""

    def test_missing_opened_at_degrades_silent(self):
        # The existing unit-test position dicts have no opened_at — output
        # must stay byte-compatible (no token).
        assert reporter._pos_hold_age_token({"ticker": "X"}) == ""

    def test_unparseable_opened_at_degrades_silent(self):
        # store always writes datetime.now(utc).isoformat(); a garbage value
        # is genuinely corrupt — drop the token, never raise (reporter
        # additive contract).
        p = {"opened_at": "not-a-date"}
        assert reporter._pos_hold_age_token(p) == ""

    def test_future_opened_at_clamps_to_silent(self):
        # Wall clock stepped back (NTP / VM time-sync — documented hazard);
        # rendering "held -2h" would be misleading. Clamp to silent so the
        # operator gets no token rather than a wrong one.
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-18T14:00:00+00:00"}  # 2h in the future
        # secs is negative, so secs < 60 → "" via the sub-minute guard.
        assert reporter._pos_hold_age_token(p, now=now) == ""

    def test_naive_opened_at_treated_as_utc(self):
        # store records ISO with explicit +00:00, but a hand-rolled migration
        # could plausibly write a naive value. Treat as UTC rather than
        # raising on tz arithmetic.
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        p = {"opened_at": "2026-05-18T10:00:00"}  # naive, 2h before now
        assert reporter._pos_hold_age_token(p, now=now) == "  held 2h"

    def test_portfolio_lines_includes_age_when_opened_at_present(self):
        # Live caller (store.open_positions()) carries opened_at; the
        # position line must include the hold-age annotation.
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        positions = [{
            "ticker": "AMD", "type": "stock", "qty": 5,
            "avg_cost": 100.0, "current_price": 110.0,
            "unrealized_pl": 50.0,
            "opened_at": "2026-05-15T12:00:00+00:00",  # 3d ago
        }]
        # The helper is used unconditionally; we just need a sufficiently old
        # opened_at to land in the day bucket regardless of wall clock.
        ages = [_p["opened_at"] for _p in positions]
        line = reporter._portfolio_lines(positions, total_value=1000.0)[0]
        # The age token landed in the line (exact value depends on wall
        # clock — assert presence of the prefix instead).
        assert "held " in line, line
        # Existing assertions still hold (byte-compat on the other tokens).
        assert "+10.0%" in line and "55% bk" in line and "AMD" in line

    def test_portfolio_lines_unchanged_for_unit_test_positions(self):
        # Backward compat: positions without opened_at (the existing
        # test_stock_line_format / test_option_line_includes_strike shape)
        # must produce a line with no "held" token.
        positions = [{
            "ticker": "AMD", "type": "stock", "qty": 5,
            "avg_cost": 100.0, "current_price": 110.0, "unrealized_pl": 50.0,
        }]
        line = reporter._portfolio_lines(positions)[0]
        assert "held " not in line, line


class TestPosEarningsToken:
    """`_pos_earnings_token` — pure per-position earnings-imminent flag from
    the `build_event_calendar` events list. Same additive-only contract as
    `_pos_hold_age_token`: a missing / corrupt / out-of-scope input drops the
    token, never raises."""

    def test_held_imminent_renders_warning_token(self):
        # NVDA earnings tomorrow, held — the canonical "must-see" alert.
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                            "tier": "HELD_IMMINENT", "held": True}}
        p = {"ticker": "NVDA", "type": "stock"}
        tok = reporter._pos_earnings_token(p, ev_map)
        assert tok == "  ⚠ ER 0.7d"

    def test_held_soon_renders_compact_token(self):
        # 5d away — informational, no warning glyph (line stays compact).
        ev_map = {"AAPL": {"ticker": "AAPL", "days_away": 5.0,
                            "tier": "HELD_SOON", "held": True}}
        p = {"ticker": "AAPL", "type": "stock"}
        tok = reporter._pos_earnings_token(p, ev_map)
        assert tok == "  ER 5.0d"

    def test_same_day_post_bell_renders_after_close(self):
        # days_away < 0 means the event has just happened. Surface explicitly
        # rather than rendering a confusing "-0.1d".
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": -0.05,
                            "tier": "HELD_IMMINENT", "held": True}}
        p = {"ticker": "NVDA", "type": "stock"}
        assert reporter._pos_earnings_token(p, ev_map) == "  ⚠ ER after close"

    def test_held_soon_same_day_post_bell_no_warning(self):
        # HELD_SOON tier shouldn't carry the warning glyph even post-bell.
        ev_map = {"AAPL": {"ticker": "AAPL", "days_away": -0.2,
                            "tier": "HELD_SOON", "held": True}}
        p = {"ticker": "AAPL", "type": "stock"}
        assert reporter._pos_earnings_token(p, ev_map) == "  ER after close"

    def test_ticker_not_in_events_is_silent(self):
        # No event for this name → no token.
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                            "tier": "HELD_IMMINENT", "held": True}}
        p = {"ticker": "TQQQ", "type": "stock"}
        assert reporter._pos_earnings_token(p, ev_map) == ""

    def test_none_events_is_silent_byte_compat(self):
        # The pre-existing unit-test callers (and any backwards-compat path
        # without earnings data) get the no-token form.
        p = {"ticker": "NVDA", "type": "stock"}
        assert reporter._pos_earnings_token(p, None) == ""
        assert reporter._pos_earnings_token(p, {}) == ""

    def test_missing_days_away_is_silent(self):
        # A corrupt event row (no days_away or non-numeric) drops the token
        # rather than raising — the additive failure contract.
        ev_map = {"NVDA": {"ticker": "NVDA", "tier": "HELD_IMMINENT"}}
        p = {"ticker": "NVDA", "type": "stock"}
        assert reporter._pos_earnings_token(p, ev_map) == ""
        ev_map_bad = {"NVDA": {"ticker": "NVDA", "days_away": "soon",
                                "tier": "HELD_IMMINENT"}}
        assert reporter._pos_earnings_token(p, ev_map_bad) == ""

    def test_ticker_normalized_to_upper(self):
        # Position records carry uppercase; defend the lookup against any
        # mixed-case ticker by uppercasing.
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                            "tier": "HELD_IMMINENT", "held": True}}
        p = {"ticker": "nvda", "type": "stock"}
        assert reporter._pos_earnings_token(p, ev_map) == "  ⚠ ER 0.7d"

    def test_missing_ticker_is_silent(self):
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                            "tier": "HELD_IMMINENT", "held": True}}
        assert reporter._pos_earnings_token({}, ev_map) == ""
        assert reporter._pos_earnings_token({"ticker": ""}, ev_map) == ""

    def test_portfolio_lines_includes_earnings_token(self):
        # End-to-end: a position with an HELD_IMMINENT event in the map
        # gets the warning token in its rendered line.
        ev_map = {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                            "tier": "HELD_IMMINENT", "held": True}}
        positions = [{
            "ticker": "NVDA", "type": "stock", "qty": 2,
            "avg_cost": 222.0, "current_price": 222.0, "unrealized_pl": 0.0,
        }]
        line = reporter._portfolio_lines(
            positions, total_value=444.0, events_by_ticker=ev_map
        )[0]
        assert "⚠ ER 0.7d" in line
        # Existing tokens still present (byte-compat on the other axes).
        assert "NVDA" in line and "222.00" in line

    def test_portfolio_lines_byte_compat_when_no_events(self):
        # Backward compat: positions without an events map (the existing
        # test callers) must produce a line with no ER token.
        positions = [{
            "ticker": "NVDA", "type": "stock", "qty": 2,
            "avg_cost": 222.0, "current_price": 222.0, "unrealized_pl": 0.0,
        }]
        line = reporter._portfolio_lines(positions)[0]
        assert "ER " not in line and "⚠" not in line


class TestEarningsEventsByTicker:
    """`_earnings_events_by_ticker` — the reporter's pure-resolver layer
    between ``build_event_calendar`` and ``_portfolio_lines``."""

    def test_builder_unavailable_returns_none(self, monkeypatch):
        # If build_event_calendar raises, the whole resolver must degrade to
        # None — the calling hourly/daily report must still ship.
        def _raise(*a, **k):
            raise RuntimeError("disk gone")
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            _raise,
        )
        assert reporter._earnings_events_by_ticker() is None

    def test_source_not_ok_returns_none(self, monkeypatch):
        # Calendar JSON missing/corrupt → source_ok=False → None.
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            lambda *a, **k: {"source_ok": False, "events": []},
        )
        assert reporter._earnings_events_by_ticker() is None

    def test_returns_map_keyed_by_uppercase_ticker(self, monkeypatch):
        # The happy path: a source-ok report with events becomes a
        # ticker→event dict for the portfolio lines lookup.
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            lambda *a, **k: {
                "source_ok": True,
                "events": [
                    {"ticker": "NVDA", "days_away": 0.7,
                     "tier": "HELD_IMMINENT"},
                    {"ticker": "aapl", "days_away": 5.0, "tier": "HELD_SOON"},
                ],
            },
        )
        out = reporter._earnings_events_by_ticker()
        assert isinstance(out, dict)
        assert "NVDA" in out and "AAPL" in out
        assert out["NVDA"]["days_away"] == 0.7

    def test_empty_events_returns_empty_dict(self, monkeypatch):
        # Calendar OK but nothing scheduled → {} (not None) so the position
        # lines render with no ER token instead of treating it as a fault.
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            lambda *a, **k: {"source_ok": True, "events": []},
        )
        assert reporter._earnings_events_by_ticker() == {}

    def test_malformed_event_rows_silently_skipped(self, monkeypatch):
        # A None or missing-ticker event must not crash the resolver.
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            lambda *a, **k: {
                "source_ok": True,
                "events": [
                    None,
                    {"ticker": "", "days_away": 1.0, "tier": "HELD_IMMINENT"},
                    {"ticker": "NVDA", "days_away": 0.7,
                     "tier": "HELD_IMMINENT"},
                ],
            },
        )
        out = reporter._earnings_events_by_ticker()
        assert out == {"NVDA": {"ticker": "NVDA", "days_away": 0.7,
                                 "tier": "HELD_IMMINENT"}}

    def test_non_dict_report_returns_none(self, monkeypatch):
        # A future builder regression that returns the wrong shape (list /
        # str / None) degrades to None instead of TypeErroring.
        monkeypatch.setattr(
            "paper_trader.analytics.event_calendar.build_event_calendar",
            lambda *a, **k: [{"ticker": "NVDA"}],
        )
        assert reporter._earnings_events_by_ticker() is None


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


class TestRealizedPlWindow:
    """`_realized_pl_window` — the pure helper that powers the SESSION
    block's "Closed N trips realized $X" line. Mirrors `_realized_pl_today`
    but with a proper ISO comparison instead of a date-only startswith so
    arbitrary windows (1h / 4h / 24h / since-last-summary) compose."""

    @staticmethod
    def _trade(i, ticker, action, qty, price, ts):
        return {
            "id": i, "ticker": ticker, "action": action, "qty": qty,
            "price": price, "value": qty * price, "timestamp": ts,
            "option_type": None, "strike": None, "expiry": None,
        }

    def test_nothing_closed_returns_none(self):
        # An open BUY produces no round-trip → None (suppression case).
        trades_newest_first = [
            self._trade(1, "NVDA", "BUY", 1, 100.0,
                        "2026-05-19T10:00:00+00:00"),
        ]
        assert reporter._realized_pl_window(
            trades_newest_first, "2026-05-19T00:00:00+00:00") is None

    def test_one_winning_trip_in_window(self):
        # BUY at 100, SELL at 120 → +$20 PnL, 1 win, 1 closed.
        trades_newest_first = [
            self._trade(2, "NVDA", "SELL", 1, 120.0,
                        "2026-05-19T15:00:00+00:00"),
            self._trade(1, "NVDA", "BUY", 1, 100.0,
                        "2026-05-19T10:00:00+00:00"),
        ]
        result = reporter._realized_pl_window(
            trades_newest_first, "2026-05-19T00:00:00+00:00")
        assert result is not None
        pnl, n_closed, n_wins = result
        assert pnl == 20.0
        assert n_closed == 1
        assert n_wins == 1

    def test_mixed_winners_and_losers_in_window(self):
        # NVDA win +$10, MU loss -$5 → net +$5, 2 closed, 1 win.
        trades_newest_first = [
            self._trade(4, "MU", "SELL", 1, 55.0,
                        "2026-05-19T14:00:00+00:00"),
            self._trade(3, "MU", "BUY", 1, 60.0,
                        "2026-05-19T11:00:00+00:00"),
            self._trade(2, "NVDA", "SELL", 1, 110.0,
                        "2026-05-19T13:00:00+00:00"),
            self._trade(1, "NVDA", "BUY", 1, 100.0,
                        "2026-05-19T09:00:00+00:00"),
        ]
        result = reporter._realized_pl_window(
            trades_newest_first, "2026-05-19T00:00:00+00:00")
        assert result is not None
        pnl, n_closed, n_wins = result
        assert pnl == 5.0
        assert n_closed == 2
        assert n_wins == 1

    def test_trip_before_window_is_excluded(self):
        # Older trip closes BEFORE since; only the newer trip counts.
        trades_newest_first = [
            self._trade(4, "MU", "SELL", 1, 60.0,
                        "2026-05-19T15:00:00+00:00"),   # in window
            self._trade(3, "MU", "BUY", 1, 50.0,
                        "2026-05-19T13:00:00+00:00"),   # in window (open leg)
            self._trade(2, "NVDA", "SELL", 1, 110.0,
                        "2026-05-18T15:00:00+00:00"),   # closed BEFORE window
            self._trade(1, "NVDA", "BUY", 1, 100.0,
                        "2026-05-18T10:00:00+00:00"),
        ]
        result = reporter._realized_pl_window(
            trades_newest_first, "2026-05-19T00:00:00+00:00")
        assert result is not None
        pnl, n_closed, n_wins = result
        # Only the MU trip (+$10) — the NVDA trip closed before the window.
        assert pnl == 10.0
        assert n_closed == 1
        assert n_wins == 1

    def test_breakeven_trip_is_not_counted_as_a_win(self):
        # A trip with pnl_usd == 0.0 is closed but neither a win nor a loss.
        # The current contract sums wins as `pnl > 0`, so breakeven → 0 wins,
        # 1 closed, 0 losses (n_losses = n_closed - n_wins = 1) — losses
        # therefore include the breakeven case, which is the conservative
        # read for an hourly summary.
        trades_newest_first = [
            self._trade(2, "NVDA", "SELL", 1, 100.0,
                        "2026-05-19T15:00:00+00:00"),
            self._trade(1, "NVDA", "BUY", 1, 100.0,
                        "2026-05-19T10:00:00+00:00"),
        ]
        result = reporter._realized_pl_window(
            trades_newest_first, "2026-05-19T00:00:00+00:00")
        assert result is not None
        pnl, n_closed, n_wins = result
        assert pnl == 0.0
        assert n_closed == 1
        assert n_wins == 0

    def test_garbage_input_degrades_to_none_not_raise(self):
        # The additive failure contract: a builder/parser fault MUST NOT
        # take down the whole hourly summary — degrade to None silently.
        # build_round_trips ignores rows without a recognised action, so the
        # garbage rows below produce zero trips → returns None (the empty
        # contract, indistinguishable from "nothing closed").
        garbage = [{"not": "a trade"}, {"timestamp": None}, None]  # type: ignore
        # Should not raise; result is None (no round-trips parseable).
        assert reporter._realized_pl_window(
            garbage, "2026-05-19T00:00:00+00:00") is None


class TestSessionBlockRealizedPl:
    """End-to-end: the SESSION block surfaces the realized round-trip line
    after the window-delta line and degrades silently when nothing closed."""

    def test_session_block_includes_realized_line_on_closed_trip(
            self, fresh_store, monkeypatch):
        # Force a recent realized trip — BUY 100 then SELL 110, both inside
        # the 24h window relative to a frozen `now`.
        from datetime import datetime as _dt, timezone as _tz
        frozen_now = _dt(2026, 5, 19, 20, 0, 0, tzinfo=_tz.utc)

        class _FrozenDatetime(_dt):
            @classmethod
            def now(cls, tz=None):
                return frozen_now if tz is None else frozen_now.astimezone(tz)

        monkeypatch.setattr(reporter, "datetime", _FrozenDatetime)

        # A clean opening BUY and closing SELL — both inside the 24h window.
        with fresh_store._lock:
            fresh_store.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, "
                "value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("2026-05-19T10:00:00+00:00", "NVDA", "BUY", 1, 100.0, 100.0,
                 "x", None, None, None),
            )
            fresh_store.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, "
                "value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("2026-05-19T15:00:00+00:00", "NVDA", "SELL", 1, 110.0, 110.0,
                 "x", None, None, None),
            )
            fresh_store.conn.commit()

        block = reporter._session_block(fresh_store, 24.0, "24h")
        # +$10.00 realized, 1 trip, 1 win.
        assert "Closed 1 trip (1W/0L) realized `$+10.00`" in block

    def test_session_block_omits_realized_line_when_nothing_closed(
            self, fresh_store):
        # No closed round-trips → the realized line is suppressed; the rest
        # of the SESSION block still emits.
        fresh_store.record_decision(True, 5, "NO_DECISION", "x", 1000, 500)
        block = reporter._session_block(fresh_store, 1.0, "1h")
        assert "**SESSION**" in block
        assert "Closed " not in block
        assert "realized " not in block

    def test_session_block_plural_grammar_for_multiple_trips(
            self, fresh_store, monkeypatch):
        from datetime import datetime as _dt, timezone as _tz
        frozen_now = _dt(2026, 5, 19, 20, 0, 0, tzinfo=_tz.utc)

        class _FrozenDatetime(_dt):
            @classmethod
            def now(cls, tz=None):
                return frozen_now if tz is None else frozen_now.astimezone(tz)

        monkeypatch.setattr(reporter, "datetime", _FrozenDatetime)
        with fresh_store._lock:
            # Two complete trips: NVDA +$10, MU -$5 → net +$5, 1W/1L.
            for ts, ticker, action, price in [
                ("2026-05-19T09:00:00+00:00", "NVDA", "BUY", 100.0),
                ("2026-05-19T11:00:00+00:00", "NVDA", "SELL", 110.0),
                ("2026-05-19T12:00:00+00:00", "MU", "BUY", 60.0),
                ("2026-05-19T14:00:00+00:00", "MU", "SELL", 55.0),
            ]:
                fresh_store.conn.execute(
                    "INSERT INTO trades (timestamp, ticker, action, qty, "
                    "price, value, reason, expiry, strike, option_type) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ts, ticker, action, 1, price, price, "x", None, None,
                     None),
                )
            fresh_store.conn.commit()

        block = reporter._session_block(fresh_store, 24.0, "24h")
        assert "Closed 2 trips (1W/1L) realized `$+5.00`" in block

    def test_session_block_realized_line_drops_on_builder_fault(
            self, fresh_store, monkeypatch):
        # A round-trip builder fault must drop ONLY the realized line, never
        # take down the whole SESSION block (the additive failure contract).
        monkeypatch.setattr(
            reporter, "_realized_pl_window",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        fresh_store.record_decision(True, 5, "NO_DECISION", "x", 1000, 500)
        # The outer _session_block's try/except wraps everything so a fault
        # in _realized_pl_window degrades to "" if not protected internally;
        # but here we want the rest of the block to still render. Direct
        # call: assert no exception escapes.
        block = reporter._session_block(fresh_store, 1.0, "1h")
        # Either silent realized line (preferred) OR the block degraded
        # cleanly to "" — but NEVER an exception escaped.
        assert isinstance(block, str)


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


class TestEquityFreshnessLine:
    """`_equity_freshness_line` + its hourly/daily wiring — routes the
    live-portfolio-vs-latest-equity-point divergence (DIVERGED/STALE_CURVE)
    to Discord so a benchmark/P&L headline silently computed off a frozen
    curve under a NO_DECISION storm is no longer invisible to the operator
    who lives in Discord (it was dashboard-only via /api/equity-freshness).

    Composes ``build_equity_freshness`` verbatim (single source of truth,
    invariant #10); surfaces only when actionable; a builder/store fault
    drops the line, never the summary (the reporter failure contract)."""

    def _patch_builder(self, monkeypatch, ret=None, raises=False):
        import paper_trader.analytics.equity_freshness as efm

        def fake(*a, **k):
            if raises:
                raise RuntimeError("equity-freshness builder boom")
            return ret

        monkeypatch.setattr(efm, "build_equity_freshness", fake)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: True)
        fake_store = MagicMock()
        fake_store.get_portfolio.return_value = {"total_value": 924.13}
        fake_store.equity_curve.return_value = []
        return fake_store

    def test_diverged_surfaces_headline_verbatim(self, monkeypatch):
        efd = {"verdict": "DIVERGED",
               "headline": ("Recorded equity point is STALE *and* materially "
                            "off the live book — live $924.13 vs recorded "
                            "$928.92; misstates the true account by $4.79.")}
        store = self._patch_builder(monkeypatch, ret=efd)
        line = reporter._equity_freshness_line(store)
        assert "⚠️ **EQUITY FRESHNESS** ◈ DIVERGED" in line
        assert efd["headline"] in line          # verbatim, invariant #10

    def test_stale_curve_surfaces_headline_verbatim(self, monkeypatch):
        efd = {"verdict": "STALE_CURVE",
               "headline": "Recorded equity point is stale (90m old) ..."}
        store = self._patch_builder(monkeypatch, ret=efd)
        line = reporter._equity_freshness_line(store)
        assert "⚠️ **EQUITY FRESHNESS** ◈ STALE_CURVE" in line
        assert efd["headline"] in line

    def test_fresh_is_suppressed(self, monkeypatch):
        efd = {"verdict": "FRESH", "headline": "current and agrees."}
        store = self._patch_builder(monkeypatch, ret=efd)
        assert reporter._equity_freshness_line(store) == ""

    def test_no_data_is_suppressed(self, monkeypatch):
        efd = {"verdict": "NO_DATA", "headline": "nothing to reconcile."}
        store = self._patch_builder(monkeypatch, ret=efd)
        assert reporter._equity_freshness_line(store) == ""

    def test_error_and_nondict_are_suppressed(self, monkeypatch):
        store = self._patch_builder(
            monkeypatch, ret={"verdict": "ERROR", "headline": "boom"})
        assert reporter._equity_freshness_line(store) == ""
        store2 = self._patch_builder(monkeypatch, ret=None)
        assert reporter._equity_freshness_line(store2) == ""

    def test_degrades_to_empty_when_builder_raises(self, monkeypatch):
        store = self._patch_builder(monkeypatch, raises=True)
        assert reporter._equity_freshness_line(store) == ""

    def _backdate_equity_point(self, store, secs_ago, total, cash=18.49):
        """Insert an equity_curve row timestamped ``secs_ago`` in the past so
        the wall-clock-age-dependent builder sees a genuinely stale point
        (record_equity_point always stamps 'now')."""
        ts = datetime.now(timezone.utc).timestamp() - secs_ago
        iso = datetime.fromtimestamp(ts, timezone.utc).isoformat()
        store.conn.execute(
            "INSERT INTO equity_curve (timestamp, total_value, cash, "
            "sp500_price) VALUES (?,?,?,?)", (iso, total, cash, 7400.0))
        store.conn.commit()

    def test_no_drift_real_builder_diverged(self, fresh_store, monkeypatch):
        """End-to-end on the REAL builder + a real Store: a 2h-old recorded
        point materially off the live portfolio → DIVERGED, and the Discord
        headline must equal the builder's own headline verbatim."""
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: True)
        self._backdate_equity_point(fresh_store, 7200, 928.92)
        fresh_store.update_portfolio(18.49, 924.13, [])
        from paper_trader.analytics.equity_freshness import (
            build_equity_freshness)
        expected = build_equity_freshness(
            fresh_store.get_portfolio(),
            fresh_store.equity_curve(limit=5000),
            True)
        assert expected["verdict"] == "DIVERGED"
        line = reporter._equity_freshness_line(fresh_store)
        assert "⚠️ **EQUITY FRESHNESS** ◈ DIVERGED" in line
        assert expected["headline"] in line          # verbatim, no drift

    def test_no_drift_real_builder_fresh_suppressed(self, fresh_store,
                                                    monkeypatch):
        """A just-recorded equity point that agrees with the book → FRESH →
        suppressed (no hourly noise on a trustworthy curve)."""
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: True)
        fresh_store.record_equity_point(1000.0, 500.0, 7400.0)
        fresh_store.update_portfolio(500.0, 1000.0, [])
        assert reporter._equity_freshness_line(fresh_store) == ""

    def test_hourly_summary_includes_diverged_line(self, fresh_store,
                                                   monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: True)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._backdate_equity_point(fresh_store, 7200, 928.92)
        fresh_store.update_portfolio(18.49, 924.13, [])
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**EQUITY FRESHNESS** ◈ DIVERGED" in body
        assert "**HOURLY**" in body and "Equity" in body

    def test_daily_close_includes_diverged_line(self, fresh_store,
                                                monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter.market, "is_market_open", lambda: True)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._backdate_equity_point(fresh_store, 7200, 928.92)
        fresh_store.update_portfolio(18.49, 924.13, [])
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "**EQUITY FRESHNESS** ◈ DIVERGED" in body
        assert "**DAILY CLOSE**" in body

    def test_summary_still_sends_when_freshness_builder_faults(
            self, fresh_store, monkeypatch):
        """A freshness-builder fault drops only its block — the hourly
        summary itself must still send (the reporter contract)."""
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        import paper_trader.analytics.equity_freshness as efm

        def _boom(*a, **k):
            raise RuntimeError("builder boom")

        monkeypatch.setattr(efm, "build_equity_freshness", _boom)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body and "Equity" in body
        assert "**EQUITY FRESHNESS**" not in body


class TestDrawdownLine:
    """`_drawdown_line` + its hourly/daily wiring — routes drawdown-from-peak
    (depth / time-underwater / claw-back / top drag) to Discord so the
    operator who lives in Discord sees the risk number that
    P/L-vs-$1000-start silently hides (it was dashboard-only via
    /api/drawdown). Consumes compute_drawdown's OWN fields verbatim; surfaces
    only when off the high; a builder/store fault drops the line, never the
    summary (the reporter failure contract)."""

    def _seed(self, store, values):
        """Append equity points with the given total_values (chronological)."""
        for v in values:
            store.record_equity_point(float(v), 500.0, 5000.0)

    def test_at_high_water_is_suppressed(self, fresh_store):
        # Single point == peak == current → at_high_water True → silent.
        self._seed(fresh_store, [1000.0])
        assert reporter._drawdown_line(fresh_store) == ""

    def test_empty_curve_is_suppressed(self, fresh_store):
        # compute_drawdown on no history returns at_high_water True.
        assert reporter._drawdown_line(fresh_store) == ""

    def test_recovered_drawdown_exact_numbers_from_real_builder(
            self, fresh_store):
        # peak 1000 → trough 900 → back to 950: dd -5.00%/-$50.00,
        # trough -10.00%, recovered 50% — all the builder's own numbers.
        self._seed(fresh_store, [1000.0, 900.0, 950.0])
        from paper_trader.analytics.drawdown import compute_drawdown
        exp = compute_drawdown(fresh_store.equity_curve(limit=2000),
                               fresh_store.open_positions(),
                               starting_equity=reporter._INITIAL_EQUITY)
        assert exp["drawdown_pct"] == -5.0
        assert exp["trough_pct"] == -10.0
        assert exp["recovery_pct"] == 50.0
        line = reporter._drawdown_line(fresh_store)
        assert line.startswith("**DRAWDOWN** ◈ off the high-water mark\n> ")
        assert "`-5.00%` ($-50.00) from peak" in line
        assert "in DD" in line
        assert "trough `-10.00%` (recovered 50%)" in line

    def test_at_trough_omits_recovery_segment(self, fresh_store):
        # Still at the lows (current == trough): recovery is 0 and there is
        # no deeper trough to report — the trough/recovered segment is gated
        # off, but the headline draw is still surfaced.
        self._seed(fresh_store, [1000.0, 900.0])
        line = reporter._drawdown_line(fresh_store)
        assert "`-10.00%` ($-100.00) from peak" in line
        assert "trough" not in line
        assert "recovered" not in line

    def test_top_drag_position_surfaces_with_value(self, fresh_store):
        self._seed(fresh_store, [1000.0, 940.0])
        fresh_store.upsert_position("LITE", "stock", 10, 50.0)
        fresh_store.upsert_position("MU", "stock", 5, 80.0)
        ids = {p["ticker"]: p["id"] for p in fresh_store.open_positions()}
        # LITE marked −$58.40 (the worst), MU −$10.00.
        fresh_store.update_position_marks({
            ids["LITE"]: (44.16, -58.40),
            ids["MU"]: (78.0, -10.00),
        })
        line = reporter._drawdown_line(fresh_store)
        assert "top drag LITE $-58.40" in line
        assert "MU" not in line  # only the single worst name

    def test_no_drag_token_when_worst_open_name_is_green(self, fresh_store):
        # Drawdown from a realized loss; the only open name is profitable →
        # no open position is dragging → no "top drag" token.
        self._seed(fresh_store, [1000.0, 930.0])
        fresh_store.upsert_position("NVDA", "stock", 2, 100.0)
        ids = {p["ticker"]: p["id"] for p in fresh_store.open_positions()}
        fresh_store.update_position_marks({ids["NVDA"]: (120.0, 40.0)})
        line = reporter._drawdown_line(fresh_store)
        assert "from peak" in line
        assert "top drag" not in line

    def test_hours_underwater_formatted_through_ago(self, fresh_store):
        # Backdate the peak row ~2h05m into the past so the builder's
        # hours_in_dd → _ago → "2h in DD" (the established age format);
        # the extra 5m margin keeps the int-hour bucket stable against
        # sub-second test-execution drift around the 7200s boundary.
        past = (datetime.now(timezone.utc).timestamp() - 7500)
        iso = datetime.fromtimestamp(past, timezone.utc).isoformat()
        fresh_store.conn.execute(
            "INSERT INTO equity_curve (timestamp, total_value, cash, "
            "sp500_price) VALUES (?,?,?,?)", (iso, 1000.0, 500.0, 5000.0))
        fresh_store.conn.commit()
        fresh_store.record_equity_point(900.0, 500.0, 5000.0)
        line = reporter._drawdown_line(fresh_store)
        assert "2h in DD" in line

    def test_nondict_result_is_suppressed(self, fresh_store, monkeypatch):
        import paper_trader.analytics.drawdown as ddm
        monkeypatch.setattr(ddm, "compute_drawdown", lambda *a, **k: None)
        assert reporter._drawdown_line(fresh_store) == ""

    def test_degrades_to_empty_when_builder_raises(self, fresh_store,
                                                   monkeypatch):
        import paper_trader.analytics.drawdown as ddm

        def _boom(*a, **k):
            raise RuntimeError("drawdown builder boom")

        monkeypatch.setattr(ddm, "compute_drawdown", _boom)
        assert reporter._drawdown_line(fresh_store) == ""

    def test_hourly_summary_includes_drawdown_when_off_high(
            self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._seed(fresh_store, [1000.0, 880.0])
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body and "Equity" in body
        assert "**DRAWDOWN** ◈ off the high-water mark" in body
        assert "`-12.00%` ($-120.00) from peak" in body

    def test_daily_close_includes_drawdown_when_off_high(
            self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._seed(fresh_store, [1000.0, 880.0])
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "**DAILY CLOSE**" in body
        assert "**DRAWDOWN** ◈ off the high-water mark" in body

    def test_hourly_summary_omits_drawdown_at_high_water(
            self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._seed(fresh_store, [1000.0])  # at the high
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body
        assert "**DRAWDOWN**" not in body

    def test_summary_still_sends_when_drawdown_builder_faults(
            self, fresh_store, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        self._seed(fresh_store, [1000.0, 900.0])
        import paper_trader.analytics.drawdown as ddm

        def _boom(*a, **k):
            raise RuntimeError("builder boom")

        monkeypatch.setattr(ddm, "compute_drawdown", _boom)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOURLY**" in body and "Equity" in body
        assert "**DRAWDOWN**" not in body


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


class TestHostPulseLine:
    """`_host_pulse_line` + its hourly/daily wiring — the #1 live-pathology
    operator surface. A 27 h host-saturation NO_DECISION drought bleeding
    -5.87% alpha was invisible from Discord, and the capital-pulse line that
    DID reach Discord misframed it as a sell-to-unfreeze problem. The
    operator lives in Discord."""

    _SAT = {
        "state": "SATURATED",
        "headline": ("Opus is starved by the box — host saturated: 7 "
                     "concurrent Opus (>4). The desk is frozen by host "
                     "load, not the market or capital — ops — reduce "
                     "concurrent Opus (review/backtest agents); the bot "
                     "cannot resolve this by trading."),
    }
    _CLEAR = {"state": "CLEAR", "headline": ""}

    def test_empty_when_clear(self, monkeypatch):
        from paper_trader import host_guard
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: self._CLEAR)
        assert reporter._host_pulse_line() == ""

    def test_surfaces_saturated_verbatim(self, monkeypatch):
        from paper_trader import host_guard
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: self._SAT)
        line = reporter._host_pulse_line()
        assert line.startswith("**HOST** ◈ SATURATED")
        # Headline carried VERBATIM (single source of truth — no re-derive).
        assert self._SAT["headline"] in line
        # The OPS discriminator that stops the operator conflating this with
        # a capital-paralysis "sell something" fix.
        assert "the bot cannot resolve this by trading" in line

    def test_surfaces_starved(self, monkeypatch):
        from paper_trader import host_guard
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: {
            "state": "STARVED", "headline": "44% never reached Opus, ops fix"})
        line = reporter._host_pulse_line()
        assert line == "**HOST** ◈ STARVED\n> 44% never reached Opus, ops fix"

    def test_degrades_to_empty_on_pulse_fault(self, monkeypatch):
        from paper_trader import host_guard

        def _boom(*a, **k):
            raise RuntimeError("host_guard.pulse blew up")

        monkeypatch.setattr(host_guard, "pulse", _boom)
        # Additive failure contract: a fault drops THIS line, never raises.
        assert reporter._host_pulse_line() == ""

    def test_hourly_summary_host_before_capital(self, fresh_store,
                                                monkeypatch):
        """The load-bearing ORDER: a top-down Discord read must hit the
        non-trading-fixable HOST cause before the CAPITAL one (both can be
        independently true; neither suppresses the other)."""
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: self._SAT)
        # Force the capital line to also emit so the ordering is observable.
        monkeypatch.setattr(reporter, "_capital_pulse_line",
                            lambda store: "**CAPITAL** ◈ PINNED\n> ~98% deployed")
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOST** ◈ SATURATED" in body
        assert "**CAPITAL** ◈ PINNED" in body
        assert body.index("**HOST**") < body.index("**CAPITAL**")
        # Pre-existing summary intact alongside the new line.
        assert "**HOURLY**" in body and "Equity" in body

    def test_hourly_summary_silent_when_clear(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: self._CLEAR)
        assert reporter.send_hourly_summary() is True
        assert "**HOST**" not in captured[0]

    def test_daily_close_includes_host(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: self._SAT)
        assert reporter.send_daily_close() is True
        assert "**HOST** ◈ SATURATED" in captured[0]


class TestIdleOpportunityLine:
    """`_idle_opportunity_line` + its hourly/daily wiring — the missing
    *regret* surface for a PARALYSIS drought. Host-pulse names WHY the bot
    is dark; idle-opportunity names WHAT was missed while the cause held."""

    _DROUGHT_OK = {
        "state": "OK",
        "headline": ("Idle opportunity: drought 8.0h (33 NO_DECISION) — "
                     "2 watchlist signal(s) ≥6.0 arrived; loudest: NVDA "
                     "(HELD) @ ai_score 9.0."),
        "n_opportunities": 2,
        "drought": {"ongoing": True},
        "opportunities": [{"ticker": "NVDA", "top_score": 9.0, "held": True}],
    }
    _DROUGHT_QUIET = {
        "state": "OK", "headline": "no signals", "n_opportunities": 0,
        "drought": {"ongoing": True}, "opportunities": [],
    }
    _NO_DROUGHT = {
        "state": "NO_DROUGHT", "headline": "filling normally",
        "n_opportunities": 0, "drought": None, "opportunities": [],
    }

    def _stub(self, monkeypatch, payload):
        """Stub BOTH builders the helper composes: build_decision_drought
        provides the gate (must return an ongoing drought for the helper to
        proceed to the article scan); build_idle_opportunity returns the
        final verdict the reporter renders. This mirrors the helper's
        actual composition order so a regression in either path is
        observable."""
        from paper_trader.analytics import idle_opportunity as io_mod
        from paper_trader.analytics import decision_drought as dd_mod
        # If the test payload represents no drought / no opps, also stub
        # decision_drought to match — but if it's the OK-with-regret case
        # we need an ongoing drought block so the helper proceeds.
        ongoing = bool(payload.get("drought") and payload["drought"].get("ongoing"))
        dd_payload = {
            "current_drought": (
                {"ongoing": True, "start": "2026-05-19T03:15:42+00:00"}
                if ongoing else None
            )
        }
        monkeypatch.setattr(dd_mod, "build_decision_drought",
                            lambda *a, **k: dd_payload)
        monkeypatch.setattr(io_mod, "build_idle_opportunity",
                            lambda *a, **k: payload)

    def test_empty_when_no_drought(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, self._NO_DROUGHT)
        assert reporter._idle_opportunity_line(fresh_store) == ""

    def test_empty_when_drought_quiet(self, fresh_store, monkeypatch):
        """Silence-when-nothing-actionable — an empty opportunities list is
        informative ("nothing missed") and must not become Discord filler."""
        self._stub(monkeypatch, self._DROUGHT_QUIET)
        assert reporter._idle_opportunity_line(fresh_store) == ""

    def test_surfaces_regret_verbatim(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, self._DROUGHT_OK)
        line = reporter._idle_opportunity_line(fresh_store)
        assert line.startswith("**IDLE** ◈ regret")
        # Headline carried VERBATIM (single source of truth — no re-derive).
        assert self._DROUGHT_OK["headline"] in line

    def test_degrades_to_empty_on_builder_fault(self, fresh_store, monkeypatch):
        from paper_trader.analytics import idle_opportunity as io_mod

        def _boom(*a, **k):
            raise RuntimeError("builder blew up")

        monkeypatch.setattr(io_mod, "build_idle_opportunity", _boom)
        # Additive failure contract — drops this line, never raises.
        assert reporter._idle_opportunity_line(fresh_store) == ""

    def test_hourly_summary_idle_between_host_and_capital(
            self, fresh_store, monkeypatch):
        """Order: HOST (cause) → IDLE (regret) → CAPITAL (manual-fix
        suggestion). All three can be independently true; none suppresses
        the others — the same independence as HOST/CAPITAL."""
        self._stub(monkeypatch, self._DROUGHT_OK)
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Force HOST and CAPITAL lines to also emit so the ordering is
        # observable end-to-end.
        monkeypatch.setattr(reporter, "_host_pulse_line",
                            lambda: "**HOST** ◈ SATURATED\n> opus starved")
        monkeypatch.setattr(reporter, "_capital_pulse_line",
                            lambda store: "**CAPITAL** ◈ PINNED\n> ~98%")
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOST**" in body
        assert "**IDLE** ◈ regret" in body
        assert "**CAPITAL**" in body
        assert body.index("**HOST**") < body.index("**IDLE**")
        assert body.index("**IDLE**") < body.index("**CAPITAL**")

    def test_hourly_silent_when_quiet(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, self._DROUGHT_QUIET)
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        assert reporter.send_hourly_summary() is True
        assert "**IDLE**" not in captured[0]

    def test_daily_close_includes_idle(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, self._DROUGHT_OK)
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        assert reporter.send_daily_close() is True
        assert "**IDLE** ◈ regret" in captured[0]


class TestPositionAttentionLine:
    """`_position_attention_line` + its hourly/daily wiring — the missing
    per-held-position attention surface in Discord. ``/api/position-attention``
    answers "WHICH lots has Opus stopped examining?" on the dashboard;
    nothing surfaced it in Discord, where the operator lives. Under a
    NO_DECISION storm a held lot can sit unmonitored for many hours while
    the operator (reading only Discord) assumes Opus is still watching it."""

    def _stub(self, monkeypatch, payload):
        """Replace build_position_attention with a fixed return."""
        from paper_trader.analytics import position_attention as pa_mod
        monkeypatch.setattr(pa_mod, "build_position_attention",
                            lambda *a, **k: payload)

    def test_empty_when_ok(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {
            "verdict": "OK",
            "note": "All 2 held position(s) examined by Opus in the last 6h.",
            "positions": [],
        })
        assert reporter._position_attention_line(fresh_store) == ""

    def test_empty_when_insufficient_data(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {
            "verdict": "INSUFFICIENT_DATA",
            "note": "No open positions to evaluate.",
            "positions": [],
        })
        assert reporter._position_attention_line(fresh_store) == ""

    def test_surfaces_neglected_book(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {
            "verdict": "NEGLECTED_BOOK",
            "note": ("1 of 2 held position(s) have had no Opus look in "
                     ">24h — model attention has lapsed on them (likely "
                     "passive HOLD via NO_DECISION storms)."),
            "positions": [
                {"ticker": "LITE", "verdict": "NEGLECTED",
                 "hours_since_last_decision": 31.7},
                {"ticker": "MU", "verdict": "MONITORED",
                 "hours_since_last_decision": 4.2},
            ],
        })
        line = reporter._position_attention_line(fresh_store)
        assert line.startswith("⚠️ **ATTENTION** ◈ NEGLECTED_BOOK")
        # Note verbatim — single source of truth, never re-derived here.
        assert "have had no Opus look in >24h" in line
        # The actual neglected ticker is named (operator can triage directly).
        assert "LITE" in line
        assert "NEGLECTED" in line
        assert "31.7h" in line
        # MONITORED row filtered out (only NEGLECTED/STALE shown).
        assert "MU" not in line

    def test_surfaces_stale_book(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {
            "verdict": "STALE_BOOK",
            "note": "1 of 1 held position(s) last seen by Opus >6h ago — monitor for drift.",
            "positions": [
                {"ticker": "NVDA", "verdict": "STALE",
                 "hours_since_last_decision": 8.3},
            ],
        })
        line = reporter._position_attention_line(fresh_store)
        assert "⚠️ **ATTENTION** ◈ STALE_BOOK" in line
        assert "NVDA" in line
        assert "STALE" in line
        assert "8.3h" in line

    def test_neglected_with_no_last_look(self, fresh_store, monkeypatch):
        # A position with no recorded decision at all → hours_since=None;
        # the line must render a distinct, parseable token (not crash on
        # f-string formatting of None).
        self._stub(monkeypatch, {
            "verdict": "NEGLECTED_BOOK",
            "note": "1 of 1 held position(s) have had no Opus look in >24h",
            "positions": [
                {"ticker": "LITE", "verdict": "NEGLECTED",
                 "hours_since_last_decision": None},
            ],
        })
        line = reporter._position_attention_line(fresh_store)
        assert "LITE" in line
        assert "no Opus look on record" in line

    def test_caps_per_position_lines_at_three(self, fresh_store, monkeypatch):
        # Five neglected — only the top 3 should appear (builder-sorted
        # worst-first; the reporter must not flood the summary).
        self._stub(monkeypatch, {
            "verdict": "NEGLECTED_BOOK",
            "note": "5 of 5 neglected",
            "positions": [
                {"ticker": f"TK{i}", "verdict": "NEGLECTED",
                 "hours_since_last_decision": float(40 - i)}
                for i in range(5)
            ],
        })
        line = reporter._position_attention_line(fresh_store)
        assert "TK0" in line and "TK1" in line and "TK2" in line
        assert "TK3" not in line and "TK4" not in line

    def test_degrades_to_empty_on_builder_fault(self, fresh_store, monkeypatch):
        from paper_trader.analytics import position_attention as pa_mod

        def _boom(*a, **k):
            raise RuntimeError("position_attention blew up")

        monkeypatch.setattr(pa_mod, "build_position_attention", _boom)
        # Additive failure contract: a fault drops THIS line, never raises.
        assert reporter._position_attention_line(fresh_store) == ""

    def test_degrades_to_empty_on_non_dict(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, None)
        assert reporter._position_attention_line(fresh_store) == ""

    def test_degrades_to_empty_on_missing_note(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {"verdict": "NEGLECTED_BOOK",
                                 "note": "", "positions": []})
        assert reporter._position_attention_line(fresh_store) == ""

    def test_hourly_summary_surfaces_neglected(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "verdict": "NEGLECTED_BOOK",
            "note": "1 held lot unmonitored >24h",
            "positions": [
                {"ticker": "LITE", "verdict": "NEGLECTED",
                 "hours_since_last_decision": 26.0},
            ],
        })
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**ATTENTION** ◈ NEGLECTED_BOOK" in body
        # Pre-existing hourly intact alongside the new block.
        assert "**HOURLY**" in body and "Equity" in body

    def test_hourly_summary_silent_when_ok(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "verdict": "OK",
            "note": "All 2 held position(s) examined by Opus in the last 6h.",
            "positions": [],
        })
        assert reporter.send_hourly_summary() is True
        assert "**ATTENTION**" not in captured[0]

    def test_daily_close_surfaces_neglected(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "verdict": "STALE_BOOK",
            "note": "1 lot last seen >6h ago",
            "positions": [
                {"ticker": "MU", "verdict": "STALE",
                 "hours_since_last_decision": 7.5},
            ],
        })
        assert reporter.send_daily_close() is True
        assert "**ATTENTION** ◈ STALE_BOOK" in captured[0]

    def test_hourly_summary_still_sends_on_builder_fault(self, fresh_store,
                                                         monkeypatch):
        """The whole hourly must still ship if the attention builder faults —
        the additive failure contract (a bad block drops one line, never the
        report)."""
        from paper_trader import host_guard
        from paper_trader.analytics import position_attention as pa_mod
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        monkeypatch.setattr(pa_mod, "build_position_attention",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("builder boom")))
        assert reporter.send_hourly_summary() is True
        # No attention block but the summary itself shipped.
        assert "**ATTENTION**" not in captured[0]
        assert "**HOURLY**" in captured[0]


class TestConcentrationLine:
    """`_concentration_line` + its hourly/daily wiring — the missing
    SINGLE_NAME_RISK Discord surface. ``/api/correlation`` answers
    "is one name dominating the stock book?" on the dashboard and the
    ``risk_mirror`` block surfaces the same to Opus; nothing surfaced it
    to Discord, where the operator lives. The live 2026-05-19 book sat at
    NVDA 75% of stock book with no concentration verdict in any hourly /
    daily report."""

    def _stub(self, monkeypatch, payload):
        """Replace build_correlation with a fixed return so we can exercise
        every state/verdict branch deterministically."""
        from paper_trader.analytics import correlation as corr_mod
        monkeypatch.setattr(corr_mod, "build_correlation",
                            lambda *a, **k: payload)

    def test_silent_when_diversified(self, fresh_store, monkeypatch):
        # A balanced book → DIVERSIFIED verdict → no concentration line.
        # Asserts the suppression discipline (the summary must never become
        # its own lying green light).
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "DIVERSIFIED",
            "headline": "DIVERSIFIED — names move largely independently.",
            "top_weight_pct": 35.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 4,
        })
        assert reporter._concentration_line(fresh_store) == ""

    def test_silent_when_moderate(self, fresh_store, monkeypatch):
        # MODERATE (top weight 50-60%, not yet single-name-risk) → silent;
        # the per-position weights in `_portfolio_lines` already expose the
        # raw number. Surface ONLY actionable SINGLE_NAME_RISK.
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "MODERATE",
            "headline": "MODERATE — partial diversification.",
            "top_weight_pct": 55.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2,
        })
        assert reporter._concentration_line(fresh_store) == ""

    def test_surfaces_when_insufficient_but_top_weight_dominates(
            self, fresh_store, monkeypatch):
        # INSUFFICIENT state (no price_history supplied — the live no-network
        # Discord path) but top_weight_pct ≥ DOMINANT_WEIGHT (60%) → surface
        # using the weight-based fallback synthesis. This is the
        # discriminating path versus the OK-state verbatim-headline branch:
        # the same SINGLE_NAME_RISK condition must still reach Discord even
        # when the builder's own verdict is None (no correlation history yet).
        self._stub(monkeypatch, {
            "state": "INSUFFICIENT", "verdict": None,
            "headline": "correlation verdict withheld.",
            "top_weight_pct": 75.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2, "effective_positions_naive": 1.6,
        })
        line = reporter._concentration_line(fresh_store)
        assert line.startswith("⚠️ **CONCENTRATION** ◈ SINGLE_NAME_RISK")
        # Builder's buried "verdict withheld" line must NOT leak through —
        # we synthesise our own one-liner from the same structured fields
        # the OK-headline reads (the risk_mirror weight-based fallback
        # precedent).
        assert "verdict withheld" not in line
        assert "NVDA is 75% of a 2-name stock book" in line
        assert "1.6 effective name(s)" in line

    def test_silent_when_insufficient_and_top_weight_below_threshold(
            self, fresh_store, monkeypatch):
        # INSUFFICIENT (no price_history) AND top_weight_pct < 60% →
        # silent. The threshold is the same builder constant
        # ``DOMINANT_WEIGHT`` (0.60) so the no-history and OK-state paths
        # land on the same SINGLE_NAME_RISK gate.
        self._stub(monkeypatch, {
            "state": "INSUFFICIENT", "verdict": None,
            "headline": "correlation verdict withheld.",
            "top_weight_pct": 55.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2,
        })
        assert reporter._concentration_line(fresh_store) == ""

    def test_silent_when_top_weight_unparseable(self, fresh_store, monkeypatch):
        # Defensive: a non-numeric top_weight_pct (future builder bug)
        # degrades silently rather than crashing the comparison.
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "SINGLE_NAME_RISK",
            "headline": "single name risk", "top_weight_pct": "not a number",
            "top_weight_ticker": "NVDA", "n_stock_positions": 2,
        })
        assert reporter._concentration_line(fresh_store) == ""

    def test_silent_when_no_data(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, {
            "state": "NO_DATA", "verdict": None,
            "headline": "No stock positions — concentration risk undefined.",
            "top_weight_pct": None, "top_weight_ticker": None,
            "n_stock_positions": 0,
        })
        assert reporter._concentration_line(fresh_store) == ""

    def test_surfaces_single_name_risk(self, fresh_store, monkeypatch):
        # The live 2026-05-19 shape: NVDA 75% of stock book.
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "SINGLE_NAME_RISK",
            "headline": ("SINGLE_NAME_RISK — NVDA is 75% of the book; "
                         "1.18 effective independent bet(s)."),
            "top_weight_pct": 75.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2,
        })
        line = reporter._concentration_line(fresh_store)
        assert line.startswith("⚠️ **CONCENTRATION** ◈ SINGLE_NAME_RISK")
        # Headline verbatim — single source of truth, never re-derived.
        assert "NVDA is 75% of the book" in line
        assert "1.18 effective independent bet" in line

    def test_synthesises_block_when_headline_empty(
            self, fresh_store, monkeypatch):
        # OK verdict + SINGLE_NAME_RISK + missing headline (defensive against
        # a future builder bug): fall through to the same weight-based
        # synthesis path the INSUFFICIENT branch uses, so the operator still
        # sees the concentration alarm.
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "SINGLE_NAME_RISK",
            "headline": "",
            "top_weight_pct": 75.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2, "effective_positions_naive": 1.6,
        })
        line = reporter._concentration_line(fresh_store)
        assert line.startswith("⚠️ **CONCENTRATION** ◈ SINGLE_NAME_RISK")
        assert "NVDA is 75% of a 2-name stock book" in line

    def test_degrades_to_empty_on_non_dict(self, fresh_store, monkeypatch):
        self._stub(monkeypatch, None)
        assert reporter._concentration_line(fresh_store) == ""

    def test_degrades_to_empty_on_builder_fault(self, fresh_store, monkeypatch):
        from paper_trader.analytics import correlation as corr_mod

        def _boom(*a, **k):
            raise RuntimeError("correlation blew up")

        monkeypatch.setattr(corr_mod, "build_correlation", _boom)
        # Additive failure contract: a fault drops THIS line, never raises.
        assert reporter._concentration_line(fresh_store) == ""

    def test_market_value_uses_option_multiplier(self, fresh_store, monkeypatch):
        """An option position must contribute ×100 to market_value so a 1-contract
        NVDA call at $10 weighs ${10 \\times 100 = 1000}, not $10. Verify by
        capturing the positions actually passed to build_correlation."""
        from paper_trader.analytics import correlation as corr_mod
        captured: dict = {}

        def _capture(positions, *a, **k):
            captured["positions"] = list(positions)
            return {
                "state": "NO_DATA", "verdict": None, "headline": "",
                "top_weight_pct": None, "top_weight_ticker": None,
            }

        monkeypatch.setattr(corr_mod, "build_correlation", _capture)
        # Seed a stock + option position via the store's upsert primitives.
        fresh_store.upsert_position("NVDA", "stock", 2, 222.0)
        fresh_store.upsert_position("AAPL", "call", 1, 10.0,
                                     expiry="2026-06-19", strike=200.0)
        # Update marks so the line has cur prices to work with.
        marks = {p["id"]: (p["avg_cost"], 0.0)
                 for p in fresh_store.open_positions()}
        fresh_store.update_position_marks(marks)
        # Smoke: line returns "" because builder returned NO_DATA.
        assert reporter._concentration_line(fresh_store) == ""
        # The sized positions passed to build_correlation must reflect the
        # option ×100 multiplier — a regression that drops it would
        # cataclysmically underweight options in the concentration calc.
        sized = {p["ticker"]: p["market_value"] for p in captured["positions"]}
        assert sized["NVDA"] == 2 * 222.0          # stock: qty × price
        assert sized["AAPL"] == 1 * 10.0 * 100.0   # option: × 100

    def test_market_value_falls_back_to_avg_cost_when_no_mark(
            self, fresh_store, monkeypatch):
        """A stale-mark position (current_price falls back to avg_cost) must
        still contribute to the weight Herfindahl — otherwise a yfinance
        outage silently halves the book's apparent concentration."""
        from paper_trader.analytics import correlation as corr_mod
        captured: dict = {}

        def _capture(positions, *a, **k):
            captured["positions"] = list(positions)
            return {"state": "NO_DATA", "verdict": None, "headline": "",
                    "top_weight_pct": None, "top_weight_ticker": None}

        monkeypatch.setattr(corr_mod, "build_correlation", _capture)
        fresh_store.upsert_position("NVDA", "stock", 2, 100.0)
        # Don't update_position_marks → current_price stays at the default 0
        # (the stale-mark scenario the live trader documents — see
        # strategy._mark_to_market).
        reporter._concentration_line(fresh_store)
        nvda = next(p for p in captured["positions"]
                    if p["ticker"] == "NVDA")
        # Fell back to avg_cost × qty, not silently zeroed.
        assert nvda["market_value"] == 2 * 100.0

    def test_hourly_summary_surfaces_single_name_risk(
            self, fresh_store, monkeypatch):
        """End-to-end wiring: a SINGLE_NAME_RISK verdict must appear in the
        Discord-bound hourly body and the rest of the summary must still
        ship (the additive failure contract)."""
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "SINGLE_NAME_RISK",
            "headline": "SINGLE_NAME_RISK — NVDA is 75% of the book.",
            "top_weight_pct": 75.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 2,
        })
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**CONCENTRATION** ◈ SINGLE_NAME_RISK" in body
        assert "NVDA is 75% of the book" in body
        # Pre-existing hourly intact alongside the new block.
        assert "**HOURLY**" in body and "Equity" in body

    def test_hourly_summary_silent_when_diversified(
            self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "DIVERSIFIED",
            "headline": "DIVERSIFIED — names move largely independently.",
            "top_weight_pct": 35.0, "top_weight_ticker": "NVDA",
            "n_stock_positions": 4,
        })
        assert reporter.send_hourly_summary() is True
        assert "**CONCENTRATION**" not in captured[0]

    def test_daily_close_surfaces_single_name_risk(
            self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        self._stub(monkeypatch, {
            "state": "OK", "verdict": "SINGLE_NAME_RISK",
            "headline": "SINGLE_NAME_RISK — LITE is 80% of the book.",
            "top_weight_pct": 80.0, "top_weight_ticker": "LITE",
            "n_stock_positions": 1,
        })
        assert reporter.send_daily_close() is True
        assert "**CONCENTRATION** ◈ SINGLE_NAME_RISK" in captured[0]
        assert "LITE is 80% of the book" in captured[0]

    def test_hourly_still_sends_when_builder_raises(self, fresh_store,
                                                    monkeypatch):
        """The whole hourly must still ship if the correlation builder
        faults — the additive failure contract (a bad block drops one line,
        never the report)."""
        from paper_trader import host_guard
        from paper_trader.analytics import correlation as corr_mod
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "pulse",
                            lambda *a, **k: {"state": "CLEAR", "headline": ""})
        monkeypatch.setattr(corr_mod, "build_correlation",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("builder boom")))
        assert reporter.send_hourly_summary() is True
        # No concentration block but the summary itself shipped.
        assert "**CONCENTRATION**" not in captured[0]
        assert "**HOURLY**" in captured[0]

    def test_calls_real_build_correlation_with_correct_signature(
            self, fresh_store):
        """Regression lock: a previous version of this code called
        ``build_correlation(sized)`` with ONE positional arg, but the
        builder requires ``(positions, price_history)``. Every monkeypatch
        test above happened to mask the TypeError because their lambdas
        accept ``*a, **k`` — so the broken-signature regression was caught
        ONLY by live validation (the line silently returned "" every cycle).
        This test exercises the REAL builder end-to-end with a 75%-NVDA
        book — exactly the live 2026-05-19 shape — and asserts the line
        renders, so a future signature drift fails loudly in CI rather
        than silently dropping the Discord block."""
        # Seed a clearly-SINGLE_NAME_RISK book (NVDA 75% / TQQQ 25%).
        fresh_store.upsert_position("NVDA", "stock", 3, 250.0)   # $750
        fresh_store.upsert_position("TQQQ", "stock", 5, 50.0)    # $250
        # Update marks so build_correlation receives positive market_value.
        marks = {p["id"]: (p["avg_cost"], 0.0)
                 for p in fresh_store.open_positions()}
        fresh_store.update_position_marks(marks)
        # Real builder, no monkeypatch. price_history={} ⇒ INSUFFICIENT
        # state with verdict=None, but top_weight_pct=75 ≥ 60 trips the
        # weight-based fallback path.
        line = reporter._concentration_line(fresh_store)
        assert "**CONCENTRATION** ◈ SINGLE_NAME_RISK" in line
        assert "NVDA" in line
        assert "75% of a 2-name stock book" in line


class TestCountdown:
    """`_countdown` — compact "in Xh Ym" countdown label.

    Pure formatting; the value-add over the existing `_ago` is the explicit
    "in N..." prefix and the H:M composition (a 14h-3m gap should not lose
    the minute precision)."""

    def test_minute_bucket(self):
        assert reporter._countdown(30 * 60) == "in 30m"

    def test_sub_minute_clamps_to_zero(self):
        assert reporter._countdown(5) == "in 0m"

    def test_hour_with_minutes(self):
        # 2h 30m → "in 2h 30m"
        assert reporter._countdown(2 * 3600 + 30 * 60) == "in 2h 30m"

    def test_hour_exact(self):
        # Whole-hour gap drops the "0m" tail for a clean read.
        assert reporter._countdown(3 * 3600) == "in 3h"

    def test_day_with_hours(self):
        # 2d 4h gap.
        assert reporter._countdown(2 * 86400 + 4 * 3600) == "in 2d 4h"

    def test_day_exact(self):
        assert reporter._countdown(3 * 86400) == "in 3d"

    def test_negative_clamps_to_zero(self):
        # Clock skew must never render "-Xm"; clamp to "in 0m" instead.
        assert reporter._countdown(-300) == "in 0m"


class TestNextSessionLine:
    """`_next_session_line` — orientation cue for the operator who checks
    Discord on weekends/overnight. Pure: no I/O, uses market.next_session_open
    + market.is_market_open. Discord-side suppression: emit nothing when the
    market is currently open."""

    def _utc_from_ny(self, year, month, day, hour, minute):
        return datetime(year, month, day, hour, minute,
                        tzinfo=reporter.market.NY).astimezone(timezone.utc)

    def test_market_open_returns_empty(self):
        # 2026-05-14 Thu 10:00 ET — market open, no orientation line.
        now = self._utc_from_ny(2026, 5, 14, 10, 0)
        assert reporter._next_session_line(now) == ""

    def test_friday_after_close_points_to_monday(self):
        # 2026-05-15 Fri 17:00 ET → Mon 2026-05-18 09:30 ET (~64.5h).
        now = self._utc_from_ny(2026, 5, 15, 17, 0)
        line = reporter._next_session_line(now)
        # Header + day-of-week token from %a (Mon) + the date.
        assert line.startswith("**MARKET** ◈ closed — next session:")
        assert "Mon 05-18 09:30 ET" in line
        # 64.5h gap → "in 2d 16h" (countdown switches to day-bucket ≥24h).
        assert "in 2d 16h" in line

    def test_saturday_morning_points_to_monday(self):
        now = self._utc_from_ny(2026, 5, 16, 10, 0)  # Sat
        line = reporter._next_session_line(now)
        assert "Mon 05-18 09:30 ET" in line

    def test_premarket_today_today_open(self):
        # 09:00 ET Thursday — market closed, today's open is the next.
        now = self._utc_from_ny(2026, 5, 14, 9, 0)
        line = reporter._next_session_line(now)
        assert "Thu 05-14 09:30 ET" in line
        assert "in 30m" in line  # exactly 30 minutes until open

    def test_holiday_skips_to_next_open_day(self):
        # Wed 2026-11-25 17:00 ET. Thu 11-26 is Thanksgiving — skip. Fri
        # 11-27 (half-day) opens at 09:30 still.
        now = self._utc_from_ny(2026, 11, 25, 17, 0)
        line = reporter._next_session_line(now)
        assert "Fri 11-27 09:30 ET" in line

    def test_helper_never_raises_on_garbage_now(self, monkeypatch):
        # If market.next_session_open faults for some reason, the helper
        # must degrade to "" — never propagate ("no Discord summary").
        def _boom(*_a, **_k):
            raise RuntimeError("clock broken")
        monkeypatch.setattr(reporter.market, "next_session_open", _boom)
        # is_market_open returns False → we'd hit next_session_open.
        monkeypatch.setattr(reporter.market, "is_market_open", lambda *_a, **_k: False)
        # Returns "" rather than raising.
        assert reporter._next_session_line() == ""

    def test_wired_into_hourly_summary_when_closed(
        self, fresh_store, monkeypatch
    ):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Freeze "now" to a Saturday so the line is rendered.
        sat = self._utc_from_ny(2026, 5, 16, 10, 0)
        monkeypatch.setattr(reporter.market, "is_market_open",
                            lambda *_a, **_k: False)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5000.0)
        monkeypatch.setattr(
            reporter, "_next_session_line",
            lambda *_a, **_k: reporter._next_session_line.__wrapped__(sat)
            if hasattr(reporter._next_session_line, "__wrapped__")
            else "**MARKET** ◈ closed — next session: Mon 05-18 09:30 ET (in 47h 30m)",
        )
        assert reporter.send_hourly_summary() is True
        assert "**MARKET** ◈ closed — next session: Mon 05-18" in captured[0]

    def test_not_wired_into_hourly_summary_when_open(
        self, fresh_store, monkeypatch
    ):
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(reporter.market, "is_market_open",
                            lambda *_a, **_k: True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5000.0)
        assert reporter.send_hourly_summary() is True
        assert "**MARKET** ◈ closed" not in captured[0]


class TestSourceMixLine:
    """`_source_mix_line` + its hourly/daily wiring — the news-breadth
    false-signal surface. `news_velocity` measures rate (BUILDING/FADING);
    a SURGING z-score is identical whether five distinct outlets are
    reporting or one wire is mirrored across five feeds. This line fires
    ONLY when at least one held ticker reads ECHO (single-source ≥70%);
    every other state is silent."""

    def _stub_builder(self, monkeypatch, payload):
        from paper_trader.analytics import news_source_mix as nsm
        monkeypatch.setattr(nsm, "build_news_source_mix",
                            lambda *a, **k: payload)

    def _seed_held_nvda(self, store):
        """Open a real NVDA position so the reporter's open_positions read
        produces a held set the builder is invoked against."""
        store.upsert_position("NVDA", qty=2, avg_cost=200.0,
                              type_="stock")
        store.update_portfolio(cash=500.0, total_value=1000.0, positions=[])

    def _stub_db_present(self, monkeypatch, tmp_path):
        """``_source_mix_line`` reads articles.db via signals._db_path. We
        stub it to return a placeholder file so the helper proceeds past the
        no-DB gate (the SQL itself returns no rows on an empty DB, which is
        the documented degrade path)."""
        from paper_trader import signals as sigs
        # Create an empty articles.db so sqlite can open it.
        import sqlite3
        p = tmp_path / "articles.db"
        c = sqlite3.connect(str(p))
        c.execute(
            "CREATE TABLE articles ("
            "id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, "
            "first_seen TEXT NOT NULL, ai_score REAL, urgency INTEGER, "
            "full_text BLOB)"
        )
        c.commit()
        c.close()
        monkeypatch.setattr(sigs, "_db_path", lambda: p)
        return p

    def test_empty_when_no_positions(self, fresh_store, monkeypatch):
        """No held names → the builder is never reached; line empty."""
        # No held positions seeded.
        assert reporter._source_mix_line(fresh_store) == ""

    def test_empty_when_no_db_path(self, fresh_store, monkeypatch):
        self._seed_held_nvda(fresh_store)
        from paper_trader import signals as sigs
        monkeypatch.setattr(sigs, "_db_path", lambda: None)
        assert reporter._source_mix_line(fresh_store) == ""

    def test_empty_when_state_is_quiet(self, fresh_store, tmp_path,
                                       monkeypatch):
        """Silence-when-nothing-actionable: STRONG/MODERATE/QUIET states all
        suppress. Only ECHO surfaces (the false-signal warning)."""
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        self._stub_builder(monkeypatch, {
            "state": "OK", "any_echo": False,
            "headline": "Source mix: NVDA STRONG (...).",
            "per_ticker": [{"ticker": "NVDA", "state": "STRONG"}],
        })
        assert reporter._source_mix_line(fresh_store) == ""

    def test_empty_when_state_no_data(self, fresh_store, tmp_path,
                                      monkeypatch):
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        self._stub_builder(monkeypatch, {
            "state": "NO_DATA", "any_echo": False,
            "headline": "Source mix: 0 articles ...",
        })
        assert reporter._source_mix_line(fresh_store) == ""

    def test_surfaces_echo_verbatim(self, fresh_store, tmp_path, monkeypatch):
        """ECHO state → fire the warning line with the builder's headline
        carried VERBATIM (single source of truth — no re-derive)."""
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        headline = ("Source mix: NVDA ECHO (5 articles, 100% from yahoo) — "
                    "surge may be syndication, not breadth.")
        self._stub_builder(monkeypatch, {
            "state": "OK", "any_echo": True, "headline": headline,
            "per_ticker": [{"ticker": "NVDA", "state": "ECHO",
                            "n_articles": 5, "top_source": "yahoo"}],
        })
        line = reporter._source_mix_line(fresh_store)
        assert line.startswith("**NEWS BREADTH** ◈ syndication warning")
        # Builder's headline appears verbatim — no re-derive.
        assert headline in line

    def test_degrades_to_empty_on_builder_fault(self, fresh_store, tmp_path,
                                                 monkeypatch):
        """Additive failure contract — drops this line, never raises."""
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        from paper_trader.analytics import news_source_mix as nsm

        def _boom(*a, **k):
            raise RuntimeError("builder blew up")

        monkeypatch.setattr(nsm, "build_news_source_mix", _boom)
        assert reporter._source_mix_line(fresh_store) == ""

    def test_hourly_summary_includes_echo_after_idle(
            self, fresh_store, tmp_path, monkeypatch):
        """Wiring order: IDLE (regret) → NEWS BREADTH (false-signal) →
        CAPITAL (manual-fix). NEWS BREADTH sits adjacent to IDLE because
        both read articles.db on held names."""
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        self._stub_builder(monkeypatch, {
            "state": "OK", "any_echo": True,
            "headline": "Source mix: NVDA ECHO (...).",
            "per_ticker": [],
        })
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Force CAPITAL line so the ordering is end-to-end observable.
        monkeypatch.setattr(reporter, "_capital_pulse_line",
                            lambda store: "**CAPITAL** ◈ FREE\n> $500 cash")
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**NEWS BREADTH**" in body
        assert "**CAPITAL**" in body
        # Order: NEWS BREADTH appears before CAPITAL.
        assert body.index("**NEWS BREADTH**") < body.index("**CAPITAL**")

    def test_hourly_silent_when_no_echo(
            self, fresh_store, tmp_path, monkeypatch):
        """End-to-end silence guarantee: a STRONG / MODERATE / QUIET verdict
        must NEVER reach Discord — the summary must not become its own
        green-light line."""
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        self._stub_builder(monkeypatch, {
            "state": "OK", "any_echo": False,
            "headline": "Source mix: NVDA MODERATE.",
            "per_ticker": [],
        })
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        assert reporter.send_hourly_summary() is True
        assert "**NEWS BREADTH**" not in captured[0]

    def test_daily_close_includes_echo(self, fresh_store, tmp_path,
                                       monkeypatch):
        self._seed_held_nvda(fresh_store)
        self._stub_db_present(monkeypatch, tmp_path)
        self._stub_builder(monkeypatch, {
            "state": "OK", "any_echo": True,
            "headline": "Source mix: NVDA ECHO (...).",
            "per_ticker": [],
        })
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        assert reporter.send_daily_close() is True
        assert "**NEWS BREADTH**" in captured[0]
