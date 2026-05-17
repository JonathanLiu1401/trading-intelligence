"""Tests for paper_trader.reporter — Discord message formatting and the
subprocess shim for openclaw.

We never actually invoke openclaw — instead we patch shutil.which to pretend
it's missing (returns False/print path) or patch subprocess.run to simulate
success/failure/timeout. Message-formatting tests assert that the text body
contains the right fields.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
        monkeypatch.setattr(reporter.shutil, "which", lambda name: None)
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
