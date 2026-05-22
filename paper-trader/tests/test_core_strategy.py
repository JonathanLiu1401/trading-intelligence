"""Tests for paper_trader.strategy — JSON parsing, indicators, pre-trade enforcement,
and the BUY/SELL/SELL_CALL/BUY_CALL execution path against a real Store.

The live trader has NO hard limits by design — the system prompt grants Opus
full autonomy. So tests around "max position size" and "stop loss" instead
verify the limits that DO exist: cash must not go negative, sells must not
exceed held qty, and option closes must disambiguate when multiple contracts
match.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import strategy
from paper_trader import market
from paper_trader import store as store_mod
from paper_trader.store import Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


# ─────────────────────────── _parse_decision ───────────────────────────

class TestParseDecision:
    def test_plain_json_object(self):
        d = strategy._parse_decision('{"action": "BUY", "ticker": "NVDA", "qty": 1}')
        assert d == {"action": "BUY", "ticker": "NVDA", "qty": 1}

    def test_strips_json_fence(self):
        d = strategy._parse_decision('```json\n{"action": "HOLD"}\n```')
        assert d == {"action": "HOLD"}

    def test_strips_bare_fence(self):
        d = strategy._parse_decision('```\n{"action": "HOLD"}\n```')
        assert d == {"action": "HOLD"}

    def test_extracts_first_object_with_trailing_text(self):
        # The model may emit a JSON object followed by prose.
        raw = '{"action": "BUY", "ticker": "AMD", "qty": 1.0}\n\nNotes: this is fine'
        d = strategy._parse_decision(raw)
        assert d["action"] == "BUY"
        assert d["ticker"] == "AMD"

    def test_returns_none_for_garbage(self):
        assert strategy._parse_decision("definitely not json at all") is None

    def test_returns_none_for_empty(self):
        assert strategy._parse_decision("") is None
        assert strategy._parse_decision(None) is None

    def test_skips_prose_before_json(self):
        raw = 'Here is my decision: {"action":"SELL", "ticker":"NVDA", "qty":2}'
        d = strategy._parse_decision(raw)
        assert d["action"] == "SELL"
        assert d["ticker"] == "NVDA"


# ─────────────────────────── _claude_call failure-cause tracking ───────────────────────────

class TestLastClaudeFail:
    """``_claude_call`` now sets ``strategy._last_claude_fail`` on every
    failure path with a short cause code (or None on success). decide()
    surfaces it in the NO_DECISION reasoning so no_decision_reasons can
    bucket the empty-response case by *why* the CLI was empty (timeout vs
    rc!=0 vs empty stdout vs missing CLI). A regression here turns the
    whole feature back into a single ``model_empty`` line."""

    def _reset(self):
        strategy._last_claude_fail = None
        strategy._quota_exhausted = False
        strategy._active_claude_proc = None

    def test_cli_missing_sets_cli_missing(self, monkeypatch):
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: None)
        assert strategy._claude_call("p") is None
        assert strategy._last_claude_fail == "cli_missing"

    def test_timeout_sets_timeout(self, monkeypatch):
        import subprocess as sp
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        class _FakeProc:
            def __init__(self):
                self.returncode = None

            def communicate(self, input=None, timeout=None):  # noqa: A002
                raise sp.TimeoutExpired(cmd="claude", timeout=timeout)

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        monkeypatch.setattr(strategy.subprocess, "Popen",
                            lambda *a, **k: _FakeProc())
        assert strategy._claude_call("p", timeout_s=1) is None
        assert strategy._last_claude_fail == "timeout"

    def test_empty_stdout_sets_empty_stdout(self, monkeypatch):
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        class _FakeProc:
            def __init__(self):
                self.returncode = 0

            def communicate(self, input=None, timeout=None):  # noqa: A002
                return ("", "")

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return self.returncode

        monkeypatch.setattr(strategy.subprocess, "Popen",
                            lambda *a, **k: _FakeProc())
        assert strategy._claude_call("p", timeout_s=1) is None
        assert strategy._last_claude_fail == "empty_stdout"

    def test_nonzero_rc_sets_nonzero_rc(self, monkeypatch):
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        class _FakeProc:
            def __init__(self):
                self.returncode = 1

            def communicate(self, input=None, timeout=None):  # noqa: A002
                return ("transient upstream error", "")

            def kill(self):
                pass

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        monkeypatch.setattr(strategy.subprocess, "Popen",
                            lambda *a, **k: _FakeProc())
        assert strategy._claude_call("p", timeout_s=1) is None
        assert strategy._last_claude_fail == "nonzero_rc"
        # And the quota flag must NOT be set — the error text has no quota
        # marker. cli_nonzero_rc and quota_exhausted are deliberately
        # different paths with different operator actions.
        assert strategy._quota_exhausted is False

    def test_quota_marker_does_not_set_nonzero_rc(self, monkeypatch):
        # When the rc!=0 output looks like a quota rejection, the dedicated
        # quota flag wins and _last_claude_fail stays None — decide()'s
        # quota_exhausted branch owns the reason text in that case.
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        class _FakeProc:
            def __init__(self):
                self.returncode = 1

            def communicate(self, input=None, timeout=None):  # noqa: A002
                return ("you've hit your org's monthly usage limit", "")

            def kill(self):
                pass

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        monkeypatch.setattr(strategy.subprocess, "Popen",
                            lambda *a, **k: _FakeProc())
        assert strategy._claude_call("p", timeout_s=1) is None
        assert strategy._quota_exhausted is True
        assert strategy._last_claude_fail is None

    def test_exception_sets_exception(self, monkeypatch):
        self._reset()
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        def _boom(*a, **k):
            raise OSError("Popen failed")

        monkeypatch.setattr(strategy.subprocess, "Popen", _boom)
        assert strategy._claude_call("p", timeout_s=1) is None
        assert strategy._last_claude_fail == "exception"

    def test_success_resets_to_none(self, monkeypatch):
        # A previous failure must NOT leak into a successful call's slot;
        # otherwise a successful cycle following a failed one would record
        # a stale reason if decide() ever read the variable post-success.
        self._reset()
        strategy._last_claude_fail = "timeout"  # simulate leftover state
        monkeypatch.setattr(strategy.shutil, "which", lambda _: "/usr/bin/claude")

        class _FakeProc:
            def __init__(self):
                self.returncode = 0

            def communicate(self, input=None, timeout=None):  # noqa: A002
                return ('{"action": "HOLD"}', "")

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return self.returncode

        monkeypatch.setattr(strategy.subprocess, "Popen",
                            lambda *a, **k: _FakeProc())
        out = strategy._claude_call("p", timeout_s=1)
        assert out == '{"action": "HOLD"}'
        assert strategy._last_claude_fail is None


# ─────────────────────────── indicator helpers ───────────────────────────

class TestRSILive:
    def test_returns_none_for_short_input(self):
        # Need > period closes; period=14 means need ≥ 15.
        assert strategy._rsi_live([1.0] * 14) is None

    def test_returns_100_when_no_losses(self):
        closes = [float(i) for i in range(1, 30)]  # strictly increasing
        rsi = strategy._rsi_live(closes, period=14)
        assert rsi == 100.0

    def test_rsi_range(self):
        # Alternating up/down should give RSI somewhere in (0, 100).
        closes = [100.0 + ((-1) ** i) * 0.5 for i in range(30)]
        rsi = strategy._rsi_live(closes, period=14)
        assert rsi is not None
        assert 0.0 <= rsi <= 100.0


class TestEMALive:
    def test_returns_empty_for_short(self):
        assert strategy._ema_live([1.0, 2.0, 3.0], period=5) == []

    def test_length_is_n_minus_period_plus_1(self):
        out = strategy._ema_live([float(i) for i in range(20)], period=5)
        assert len(out) == 20 - 5 + 1

    def test_first_value_is_sma(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        out = strategy._ema_live(vals, period=5)
        # First EMA value is the SMA of the first 5 elements.
        assert out[0] == pytest.approx(30.0)


class TestMACDLive:
    def test_returns_none_for_too_few_closes(self):
        # MACD needs at least 35 closes (26 EMA + 9 signal smoothing).
        assert strategy._macd_live([float(i) for i in range(34)]) is None

    def test_accelerating_uptrend_is_bullish(self):
        # A *strictly linear* uptrend hits a MACD steady-state where the
        # signal line equals the MACD line; floating-point noise then decides
        # the comparison. An accelerating uptrend keeps MACD above signal.
        closes = [100.0 + i + 0.02 * i * i for i in range(60)]
        assert strategy._macd_live(closes) == "bullish"

    def test_accelerating_downtrend_is_bearish(self):
        closes = [100.0 - i - 0.02 * i * i for i in range(60)]
        assert strategy._macd_live(closes) == "bearish"


class TestStdevLive:
    """`_stdev_live` is a *population* stdev (÷n) and is the only input to
    `bb_position` in get_quant_signals_live. It had zero direct coverage, yet
    a `/n`→`/(n-1)` slip or a moved `n < 2` guard would silently shift every
    Bollinger reading Opus and the DecisionScorer see. Each test pins one
    branch with an exact value so such a regression fails loudly."""

    def test_short_input_returns_zero(self):
        # The caller guards on `if sd20 > 0:` before dividing — so the
        # degenerate path MUST return exactly 0.0, not raise / not NaN.
        assert strategy._stdev_live([]) == 0.0
        assert strategy._stdev_live([5.0]) == 0.0

    def test_two_element_is_population_not_sample(self):
        # n=2 is the smallest *non*-degenerate case: locks that `n < 2` is
        # exclusive (it computes here, doesn't short-circuit to 0.0) AND that
        # the divisor is n. [0,2] → mean 1, dev² {1,1}, /2 = 1 → sqrt = 1.0.
        # Sample stdev (÷ n-1) would be sqrt(2) ≈ 1.414 and fail this.
        assert strategy._stdev_live([0.0, 2.0]) == pytest.approx(1.0)

    def test_constant_series_is_zero(self):
        # Distinct code path from the short-input guard: it runs the full
        # variance sum and must still yield 0.0 so a flat 20-day window
        # leaves bb_position None instead of dividing by zero.
        assert strategy._stdev_live([3.0, 3.0, 3.0, 3.0]) == 0.0

    def test_known_series_exact_population_value(self):
        # Textbook set: mean 5.0, Σ dev² = 32, /8 = 4, sqrt = exactly 2.0.
        # Sample variance would be 32/7 ≈ 4.571 → 2.138, so this is the
        # hard lock against the population→sample regression.
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert strategy._stdev_live(vals) == pytest.approx(2.0)


class TestFormatQuantSignals:
    """`_format_quant_signals` builds the TECHNICAL SIGNALS block of the live
    Opus prompt. Zero prior coverage. Each test targets a branch a refactor
    could silently break — not the literal format string."""

    def test_empty_dict_returns_sentinel(self):
        # `if not sigs` — losing this guard would emit an empty block (no
        # rows, no notice) and Opus couldn't tell "no data" from a bug.
        assert strategy._format_quant_signals({}) == "  (no quant signals available)"

    def test_pct_vs_v_field_coercion(self):
        # momentum / 52w-proximity fields go through `_pct` (None → "?",
        # value → "{x}%"); rsi/macd/etc go through `_v` (None → "?", no %).
        # A `_pct`↔`_v` swap on a field would pass through the prompt
        # unnoticed, so pin both the None and the present-value rendering.
        line = strategy._format_quant_signals({
            "NVDA": {"rsi": None, "mom_5d": None, "mom_20d": 3.5,
                     "pct_from_52h": -1.2},
        })
        assert "rsi=?" in line          # _v None → "?", NOT "?%"
        assert "rsi=?%" not in line
        assert "mom_5d=?" in line       # _pct None → "?", NOT "?%"
        assert "mom_5d=?%" not in line
        assert "mom_20d=3.5%" in line   # _pct value → "{x}%"
        assert "52h=-1.2%" in line

    def test_rows_sorted_by_ticker(self):
        # `sorted(sigs.items())` — a regression to plain `.items()` would
        # reorder the prompt non-deterministically; lock alphabetical.
        out = strategy._format_quant_signals({
            "ZM": {"rsi": 50}, "AAPL": {"rsi": 60}, "MU": {"rsi": 40},
        })
        assert out.index("  AAPL:") < out.index("  MU:") < out.index("  ZM:")


class TestBollingerPositionCalibration:
    """`get_quant_signals_live` computes `bb_position = (last - sma20) /
    (2 * sd20)`, so a price sitting *on* the upper/lower Bollinger band (2
    standard deviations from the 20-day mean) lands at ≈ +1 / -1 — NOT ±2.

    This pins the calibration the live system prompt now states ("bb_position
    approaching +1 or -1 means price is at the upper/lower Bollinger band").
    The previous prompt told Opus to watch for "+2 or -2", a threshold the
    metric only reaches at ~4σ — i.e. effectively never — so a genuinely
    stretched name read as un-stretched to the decision engine. A regression
    that switched the denominator to `sd20` (band at ±2) would fail here."""

    def _fake_yf(self, monkeypatch, closes):
        import pandas as pd
        df = pd.DataFrame({
            "Close": closes,
            "Volume": [1_000_000.0] * len(closes),
        })

        class _FakeTicker:
            def __init__(self, _sym):
                pass

            def history(self, period="1y", auto_adjust=False):
                return df

        monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
        strategy._QUANT_CACHE.clear()

    def test_price_at_two_sigma_band_reads_near_plus_one(self, monkeypatch):
        # Last 20 closes: 18 alternating 95/105 + 100 + a stretched 111. The
        # 111 sits ~1.97σ above the 20-day mean — right at the upper band.
        window20 = [95.0, 105.0] * 9 + [100.0, 111.0]
        closes = [100.0] * 230 + window20
        self._fake_yf(monkeypatch, closes)

        sig = strategy.get_quant_signals_live(["BBTEST1"])["BBTEST1"]
        bb = sig["bb_position"]
        # A price at the 2σ band reads ≈ +1, NOT ≈ +2.
        assert 0.9 <= bb <= 1.1
        assert bb < 1.5

    def test_bb_position_matches_documented_formula(self, monkeypatch):
        # Independent recompute of (last - sma20) / (2 * sd20) over the exact
        # 20-element window — pins the denominator and the ±2 clamp.
        window20 = [98.0, 101.0, 99.0, 103.0, 97.0] * 3 + [
            100.0, 102.0, 96.0, 104.0, 108.0]
        assert len(window20) == 20
        closes = [100.0] * 230 + window20
        self._fake_yf(monkeypatch, closes)

        sig = strategy.get_quant_signals_live(["BBTEST2"])["BBTEST2"]
        last = window20[-1]
        sma20 = sum(window20) / 20
        sd20 = strategy._stdev_live(window20)
        expected = round(max(-2.0, min(2.0, (last - sma20) / (2 * sd20))), 2)
        assert sig["bb_position"] == expected

    def test_price_below_lower_band_is_negative(self, monkeypatch):
        # Mirror case: a sharply depressed last close yields a negative
        # bb_position (the oversold side of the band).
        window20 = [95.0, 105.0] * 9 + [100.0, 89.0]
        closes = [100.0] * 230 + window20
        self._fake_yf(monkeypatch, closes)

        sig = strategy.get_quant_signals_live(["BBTEST3"])["BBTEST3"]
        assert sig["bb_position"] < 0.0


# ─────────────────────────── _enforce_risk_pre_trade ───────────────────────────

class TestEnforceRiskPreTrade:
    def test_hold_always_allowed(self):
        snap = {"positions": []}
        ok, why = strategy._enforce_risk_pre_trade({"action": "HOLD"}, snap)
        assert ok is True
        assert why == ""

    def test_buy_with_zero_qty_blocked(self):
        snap = {"positions": []}
        ok, why = strategy._enforce_risk_pre_trade(
            {"action": "BUY", "ticker": "NVDA", "qty": 0}, snap)
        assert ok is False
        assert "qty" in why.lower()

    def test_buy_allowed_when_no_holdings(self):
        snap = {"positions": []}
        ok, _ = strategy._enforce_risk_pre_trade(
            {"action": "BUY", "ticker": "NVDA", "qty": 5}, snap)
        assert ok is True

    def test_sell_without_position_blocked(self):
        snap = {"positions": []}
        ok, why = strategy._enforce_risk_pre_trade(
            {"action": "SELL", "ticker": "NVDA", "qty": 1}, snap)
        assert ok is False
        assert "no open" in why.lower()

    def test_sell_exceeding_held_qty_blocked(self):
        snap = {"positions": [{"ticker": "NVDA", "type": "stock", "qty": 5}]}
        ok, why = strategy._enforce_risk_pre_trade(
            {"action": "SELL", "ticker": "NVDA", "qty": 10}, snap)
        assert ok is False
        assert "exceeds held" in why.lower()

    def test_sell_within_held_qty_allowed(self):
        snap = {"positions": [{"ticker": "NVDA", "type": "stock", "qty": 5}]}
        ok, _ = strategy._enforce_risk_pre_trade(
            {"action": "SELL", "ticker": "NVDA", "qty": 5}, snap)
        assert ok is True

    def test_non_numeric_qty_blocks_cleanly_not_crashes(self):
        # Regression: a Claude decision with qty="all" / "half" used to raise
        # ValueError inside _enforce_risk_pre_trade — the unguarded float()
        # call would propagate up and abort the whole decide() cycle (no
        # decision row, no equity point). The helper must now return
        # ok=False with an actionable detail instead.
        snap = {"positions": [{"ticker": "NVDA", "type": "stock", "qty": 5}]}
        ok, why = strategy._enforce_risk_pre_trade(
            {"action": "SELL", "ticker": "NVDA", "qty": "all"}, snap)
        assert ok is False
        assert "qty" in why.lower()
        assert "all" in why
        ok2, why2 = strategy._enforce_risk_pre_trade(
            {"action": "BUY", "ticker": "NVDA", "qty": None}, snap)
        # qty None coerces to 0, blocked as qty must be > 0 (separate path).
        # We just need to confirm there's no crash and ok is False.
        assert ok2 is False


# ─────────────────────────── _execute (BUY / SELL) ───────────────────────────

class TestExecuteBuy:
    def test_buy_decreases_cash_and_creates_position(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 100.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "AMD", "qty": 5, "reasoning": "test"}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        assert "BUY 5" in detail
        pf = fresh_store.get_portfolio()
        # 1000 - 5 * 100 = 500
        assert pf["cash"] == 500.0
        positions = fresh_store.open_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "AMD"
        assert positions[0]["qty"] == 5

    def test_buy_blocked_when_cash_insufficient(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 100.0)
        snap = {"cash": 50.0, "total_value": 50.0, "positions": []}
        decision = {"action": "BUY", "ticker": "AMD", "qty": 5, "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "insufficient cash" in detail

    def test_buy_blocked_when_no_price(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: None)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "AMD", "qty": 1, "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "no price" in detail

    def test_buy_blocked_on_non_numeric_qty(self, fresh_store):
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY", "ticker": "AMD", "qty": "lots", "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "qty" in detail.lower()


class TestExecuteSell:
    def test_sell_increases_cash_and_closes_position(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 120.0)
        # Seed position: 5 @ 100. Snapshot reflects the open position.
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        snap = {
            "cash": 500.0, "total_value": 1000.0,
            "positions": [{"ticker": "AMD", "type": "stock", "qty": 5, "avg_cost": 100.0}],
        }
        decision = {"action": "SELL", "ticker": "AMD", "qty": 5, "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        pf = fresh_store.get_portfolio()
        # 500 cash + 5*120 = 1100
        assert pf["cash"] == 1100.0
        # Position fully closed.
        assert fresh_store.open_positions() == []


class TestExecuteBuyCall:
    def test_buy_call_records_position_with_strike_and_expiry(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 5.0)
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": 600, "expiry": "2026-12-19", "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        positions = fresh_store.open_positions()
        assert len(positions) == 1
        assert positions[0]["type"] == "call"
        assert positions[0]["strike"] == 600.0
        # Cash: 1000 - 5 * 1 * 100 = 500
        assert fresh_store.get_portfolio()["cash"] == 500.0

    def test_buy_call_blocked_without_strike(self, fresh_store):
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY_CALL", "ticker": "NVDA", "qty": 1,
                    "expiry": "2026-12-19", "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "strike" in detail.lower()

    def test_buy_call_blocked_when_insufficient_cash(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 100.0)
        snap = {"cash": 50.0, "total_value": 50.0, "positions": []}
        decision = {"action": "BUY_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": 600, "expiry": "2026-12-19", "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "insufficient cash" in detail

    def test_buy_call_blocked_on_non_numeric_strike(self, fresh_store):
        # Regression: Claude can emit strike="ATM" (a description, not a
        # number). Before the fix this raised ValueError inside _execute and
        # crashed the whole decide() cycle (no decision row, no equity point).
        # Must now cleanly BLOCK with an actionable detail.
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        decision = {"action": "BUY_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": "ATM", "expiry": "2026-12-19", "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "strike" in detail.lower()
        assert "ATM" in detail
        # Same defensive check for puts.
        decision_p = {**decision, "action": "BUY_PUT"}
        status_p, detail_p = strategy._execute(decision_p, snap, fresh_store)
        assert status_p == "BLOCKED"
        assert "strike" in detail_p.lower()

    def test_sell_call_blocked_on_non_numeric_strike(self, fresh_store):
        # Same regression on the close side: a non-numeric strike must not
        # reach the list-comp ``float(strike)`` and crash the cycle.
        positions = [{"ticker": "NVDA", "type": "call", "qty": 1,
                      "avg_cost": 5.0, "strike": 600.0, "expiry": "2026-12-19"}]
        snap = {"cash": 1000.0, "total_value": 2000.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": "ITM", "expiry": "2026-12-19", "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "strike" in detail.lower()
        assert "ITM" in detail


class TestExecuteSellCallDisambiguation:
    """Regression: silently picking the first match when multiple option
    contracts share the same ticker+type is dangerous. The execute path now
    BLOCKS unless strike+expiry are specified."""

    def test_ambiguous_close_blocked(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 6.0)
        positions = [
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 600.0, "expiry": "2026-12-19"},
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 700.0, "expiry": "2026-12-19"},
        ]
        snap = {"cash": 1000.0, "total_value": 2000.0, "positions": positions}
        # No strike → ambiguous → must be BLOCKED.
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1, "reasoning": ""}
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "ambiguous" in detail.lower()

    def test_unambiguous_close_works(self, fresh_store, monkeypatch):
        # Only ONE open contract → strike not strictly required to disambiguate.
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 6.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        positions = [{"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
                      "strike": 600.0, "expiry": "2026-12-19"}]
        snap = {"cash": 500.0, "total_value": 600.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1, "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"

    def test_disambiguated_close_works(self, fresh_store, monkeypatch):
        # Two contracts but caller specifies strike + expiry → match resolves.
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 6.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=700.0)
        positions = [
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 600.0, "expiry": "2026-12-19"},
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 700.0, "expiry": "2026-12-19"},
        ]
        snap = {"cash": 1000.0, "total_value": 2000.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": 700, "expiry": "2026-12-19", "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"

    def test_disambiguated_close_caps_at_matched_contract_qty(
            self, fresh_store, monkeypatch):
        # Two contracts (qty 1 each) → _enforce_risk_pre_trade sums held=2
        # across strikes, so a qty=2 SELL_CALL passes the pre-trade gate. But
        # the caller disambiguates to the 700C, which only holds qty 1.
        # _execute must apply its own per-contract cap and BLOCK — otherwise
        # cash is over-credited for a contract that was never held. Pins the
        # SELL_CALL per-contract recheck seam in strategy._execute.
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 6.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=700.0)
        positions = [
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 600.0, "expiry": "2026-12-19"},
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
             "strike": 700.0, "expiry": "2026-12-19"},
        ]
        snap = {"cash": 1000.0, "total_value": 2000.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 2,
                    "strike": 700, "expiry": "2026-12-19", "reasoning": ""}
        # The pre-trade gate alone would pass (held summed across strikes = 2).
        ok, _ = strategy._enforce_risk_pre_trade(decision, snap)
        assert ok is True
        # But _execute caps at the matched contract's qty (1) and blocks.
        status, detail = strategy._execute(decision, snap, fresh_store)
        assert status == "BLOCKED"
        assert "exceeds held" in detail.lower()
        # No phantom SELL recorded, cash untouched.
        assert fresh_store.recent_trades(5) == []
        assert fresh_store.get_portfolio()["cash"] == 1000.0


# ─────────────────────────── HOLD / REBALANCE / unknown ───────────────────────────

class TestExecuteOtherActions:
    def test_hold_returns_hold_status(self, fresh_store):
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, _ = strategy._execute(
            {"action": "HOLD", "reasoning": "waiting"}, snap, fresh_store)
        assert status == "HOLD"

    def test_rebalance_returns_hold_for_now(self, fresh_store):
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "REBALANCE"}, snap, fresh_store)
        assert status == "HOLD"
        assert "not yet implemented" in detail.lower()

    def test_unknown_action_blocked(self, fresh_store):
        snap = {"cash": 1000.0, "total_value": 1000.0, "positions": []}
        status, detail = strategy._execute(
            {"action": "TELEPORT", "ticker": "NVDA", "qty": 1, "reasoning": ""}, snap, fresh_store)
        assert status == "BLOCKED"
        assert "unknown action" in detail.lower()


# ─────────────────────────── expired-option settlement ───────────────────────────
# Regression: yfinance has no option chain past expiry, so get_option_price
# returns None. The old `cur = cur or p["avg_cost"]` then marked an expired
# (often worthless) contract at full purchase premium *forever*, never closing
# it — silently inflating equity and every reported P/L. The system prompt
# explicitly tells Opus it "can hold options through expiry", so this is
# reachable by design, not an accident. Expired options must settle at
# intrinsic against the underlying (0.0 when OTM or the underlying is
# unavailable), never at avg_cost.

from datetime import date as _date  # noqa: E402


class TestOptionExpired:
    def test_past_date_is_expired(self):
        assert strategy._option_expired("2020-01-17", today=_date(2026, 5, 16)) is True

    def test_expiry_day_itself_is_not_expired(self):
        # An option is still live and tradeable *on* its expiry date.
        assert strategy._option_expired("2026-05-16", today=_date(2026, 5, 16)) is False

    def test_future_date_is_not_expired(self):
        assert strategy._option_expired("2026-12-19", today=_date(2026, 5, 16)) is False

    def test_none_expiry_is_not_expired(self):
        assert strategy._option_expired(None) is False

    def test_malformed_expiry_is_not_expired(self):
        # A garbage expiry must not crash the mark loop nor be treated as
        # expired (which would zero a live position).
        assert strategy._option_expired("not-a-date") is False

    def test_datetime_prefixed_expiry_parses(self):
        assert strategy._option_expired("2020-01-17T00:00:00", today=_date(2026, 5, 16)) is True


class TestOptionExpiredCloseGate:
    """The new NY-tz-aware path: expiry day flips at the NYSE close
    (16:00 ET regular / 13:00 ET half-day), not at UTC midnight.

    Pre-fix bug (AGENTS.md review pass #33): an expired option was marked at
    avg_cost with stale_mark=True for the ~3-4h window between the actual
    close and UTC midnight, every monthly expiry."""

    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _Zi
    NY = _Zi("America/New_York")
    UTC = _Zi("UTC")

    def test_expiry_day_before_close_still_live(self):
        # 15:00 ET on 2026-05-15 (regular session day) — pre-close.
        now = self._dt(2026, 5, 15, 15, 0, tzinfo=self.NY)
        assert strategy._option_expired("2026-05-15", now=now) is False

    def test_expiry_day_at_close_is_expired(self):
        # 16:00 ET exactly — the bell. Expired.
        now = self._dt(2026, 5, 15, 16, 0, tzinfo=self.NY)
        assert strategy._option_expired("2026-05-15", now=now) is True

    def test_expiry_day_after_close_is_expired(self):
        # 16:30 ET — the window the bug left mis-marked.
        now = self._dt(2026, 5, 15, 16, 30, tzinfo=self.NY)
        assert strategy._option_expired("2026-05-15", now=now) is True

    def test_utc_midnight_window_now_correct(self):
        # 20:00 ET = 00:00 UTC the next day. Old logic flipped here; new
        # logic flipped 4h earlier (16:00 ET) — the bug's resolution.
        now = self._dt(2026, 5, 15, 20, 0, tzinfo=self.NY)
        assert strategy._option_expired("2026-05-15", now=now) is True

    def test_half_day_early_close_flips_at_13_00_et(self):
        # 2026-11-27 (day after Thanksgiving) closes at 13:00 ET, not 16:00.
        # 12:30 ET — pre-early-close, still live.
        pre = self._dt(2026, 11, 27, 12, 30, tzinfo=self.NY)
        assert strategy._option_expired("2026-11-27", now=pre) is False
        # 13:00 ET — early-close bell, expired.
        at = self._dt(2026, 11, 27, 13, 0, tzinfo=self.NY)
        assert strategy._option_expired("2026-11-27", now=at) is True
        # 14:00 ET — formerly the buggy window (regular close = 16:00 would
        # have read False; new logic correctly reads True for the half day).
        after = self._dt(2026, 11, 27, 14, 0, tzinfo=self.NY)
        assert strategy._option_expired("2026-11-27", now=after) is True

    def test_future_expiry_still_not_expired_even_late_in_day(self):
        # Future date — never expired regardless of time of day.
        now = self._dt(2026, 5, 15, 23, 30, tzinfo=self.NY)
        assert strategy._option_expired("2026-12-19", now=now) is False

    def test_past_expiry_always_expired(self):
        # Past date — always expired, no close-time check needed.
        now = self._dt(2026, 5, 15, 9, 0, tzinfo=self.NY)
        assert strategy._option_expired("2020-01-17", now=now) is True

    def test_utc_input_is_normalized_to_ny(self):
        # 20:30 UTC on 2026-05-15 = 16:30 ET — should be expired (post-close).
        now_utc = self._dt(2026, 5, 15, 20, 30, tzinfo=self.UTC)
        assert strategy._option_expired("2026-05-15", now=now_utc) is True
        # 19:30 UTC on 2026-05-15 = 15:30 ET — pre-close, still live.
        now_utc_pre = self._dt(2026, 5, 15, 19, 30, tzinfo=self.UTC)
        assert strategy._option_expired("2026-05-15", now=now_utc_pre) is False

    def test_naive_datetime_treated_as_utc(self):
        # A naive datetime is interpreted as UTC (the function tolerates the
        # legacy callers that built datetime.now() without tzinfo).
        now_naive = self._dt(2026, 5, 15, 20, 30)  # naive == UTC == 16:30 ET
        assert strategy._option_expired("2026-05-15", now=now_naive) is True


class TestExpiredIntrinsic:
    def test_call_in_the_money(self, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 650.0)
        assert strategy._expired_intrinsic("NVDA", "call", 600.0) == 50.0

    def test_call_out_of_the_money_is_zero(self, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 550.0)
        assert strategy._expired_intrinsic("NVDA", "call", 600.0) == 0.0

    def test_put_in_the_money(self, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 80.0)
        assert strategy._expired_intrinsic("AMD", "put", 100.0) == 20.0

    def test_put_out_of_the_money_is_zero(self, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 120.0)
        assert strategy._expired_intrinsic("AMD", "put", 100.0) == 0.0

    def test_underlying_unavailable_is_zero_not_premium(self, monkeypatch):
        # The crux: no underlying price must NOT become avg_cost. 0.0.
        monkeypatch.setattr(market, "get_price", lambda t: None)
        assert strategy._expired_intrinsic("NVDA", "call", 600.0) == 0.0

    def test_nonpositive_underlying_is_zero(self, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 0.0)
        assert strategy._expired_intrinsic("NVDA", "call", 600.0) == 0.0


class TestPortfolioSnapshotSummation:
    """total_value must equal cash + Σ(position market value) across a mixed
    stock+option book. Existing tests only assert open_value for a single
    expired option; this pins the cash+positions identity the spec requires
    and would catch a multiplier/sign regression in the aggregation."""

    def test_total_value_is_cash_plus_all_position_market_values(
        self, fresh_store, monkeypatch
    ):
        # Stock: 5 AMD marked @ $120  → 5 * 120 * 1   = $600
        # Option: 2 NVDA 600C marked @ $7 → 2 * 7 * 100 = $1400
        # Cash starts at the store default ($1000), untouched by upserts.
        monkeypatch.setattr(market, "get_prices", lambda tks: {"AMD": 120.0})
        monkeypatch.setattr(market, "get_option_price",
                            lambda t, e, s, ot: 7.0)
        fresh_store.upsert_position("AMD", "stock", qty=5, avg_cost=100.0)
        fresh_store.upsert_position("NVDA", "call", qty=2, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)

        snap = strategy._portfolio_snapshot(fresh_store)

        assert snap["cash"] == pytest.approx(1000.0)
        assert snap["open_value"] == pytest.approx(600.0 + 1400.0)
        assert snap["total_value"] == pytest.approx(1000.0 + 2000.0)
        # The identity itself, derived from the per-position market_value the
        # snapshot reports, must hold exactly.
        summed = snap["cash"] + sum(p["market_value"] for p in snap["positions"])
        assert snap["total_value"] == pytest.approx(summed)
        # And it must be persisted, not just returned.
        assert fresh_store.get_portfolio()["total_value"] == pytest.approx(3000.0)

    def test_empty_book_total_equals_cash(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_prices", lambda tks: {})
        snap = strategy._portfolio_snapshot(fresh_store)
        assert snap["open_value"] == 0.0
        assert snap["total_value"] == pytest.approx(snap["cash"])


class TestPortfolioSnapshotExpiredOptions:
    def test_expired_otm_option_marked_to_zero_not_premium(self, fresh_store, monkeypatch):
        # Bought a call for $5.00 premium; it expired OTM. Must mark to 0,
        # realizing the full -$500 loss — NOT sit at avg_cost showing $0 P/L.
        monkeypatch.setattr(market, "get_price", lambda t: 550.0)  # OTM vs 600 strike
        monkeypatch.setattr(market, "get_option_price",
                            lambda *a, **k: pytest.fail("must not query a dead chain"))
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        assert len(snap["positions"]) == 1
        pos = snap["positions"][0]
        assert pos["current_price"] == 0.0
        assert pos["unrealized_pl"] == pytest.approx(-500.0)  # (0 - 5) * 1 * 100
        assert snap["open_value"] == 0.0

    def test_expired_itm_option_settles_at_intrinsic(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_price", lambda t: 650.0)  # ITM vs 600
        fresh_store.upsert_position("NVDA", "call", qty=2, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        pos = snap["positions"][0]
        assert pos["current_price"] == 50.0          # 650 - 600
        assert pos["unrealized_pl"] == pytest.approx((50.0 - 5.0) * 2 * 100)
        assert snap["open_value"] == pytest.approx(50.0 * 2 * 100)

    def test_expired_option_no_underlying_does_not_inflate_equity(self, fresh_store, monkeypatch):
        # The phantom-equity regression: underlying price unavailable AND
        # chain dead → still 0.0, never the $5 premium.
        monkeypatch.setattr(market, "get_price", lambda t: None)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        assert snap["positions"][0]["current_price"] == 0.0
        assert snap["open_value"] == 0.0

    def test_live_option_still_uses_chain_price(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: 7.5)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        assert snap["positions"][0]["current_price"] == 7.5

    def test_live_option_transient_none_still_falls_back_to_avg_cost(self, fresh_store, monkeypatch):
        # Behaviour preserved for *non-expired* options: a transient yfinance
        # miss (None) on a live contract still marks at avg_cost, not 0.
        monkeypatch.setattr(market, "get_option_price", lambda t, e, s, ot: None)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        assert snap["positions"][0]["current_price"] == 5.0


class TestExecuteCloseExpiredOption:
    def test_sell_call_on_expired_contract_settles_at_intrinsic(self, fresh_store, monkeypatch):
        # Closing an expired ITM call must credit cash the intrinsic value,
        # not the avg_cost breakeven the old `or match["avg_cost"]` produced.
        monkeypatch.setattr(market, "get_option_price", lambda *a, **k: None)
        monkeypatch.setattr(market, "get_price", lambda t: 650.0)  # ITM vs 600
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        positions = [{"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
                      "strike": 600.0, "expiry": "2020-01-17"}]
        snap = {"cash": 100.0, "total_value": 600.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": 600, "expiry": "2020-01-17", "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        # cash 100 + intrinsic 50 * 1 * 100 = 5100  (NOT 100 + 5*100 = 600)
        assert fresh_store.get_portfolio()["cash"] == pytest.approx(5100.0)

    def test_sell_call_on_expired_otm_contract_settles_at_zero(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_option_price", lambda *a, **k: None)
        monkeypatch.setattr(market, "get_price", lambda t: 500.0)  # OTM vs 600
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        positions = [{"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
                      "strike": 600.0, "expiry": "2020-01-17"}]
        snap = {"cash": 100.0, "total_value": 600.0, "positions": positions}
        decision = {"action": "SELL_CALL", "ticker": "NVDA", "qty": 1,
                    "strike": 600, "expiry": "2020-01-17", "reasoning": ""}
        status, _ = strategy._execute(decision, snap, fresh_store)
        assert status == "FILLED"
        # Worthless settlement → no cash credit (NOT the $500 avg_cost breakeven).
        assert fresh_store.get_portfolio()["cash"] == pytest.approx(100.0)


# ─────────────────────── stale-mark surfacing (Phase 2) ───────────────────────

class TestStaleMarkFlag:
    """A held position whose live price is unavailable is silently marked at
    avg_cost (current_price == avg_cost, unrealized_pl == 0.0). To a trader
    that reads as a *flat* position when it is really *unknown* — exactly the
    MU case seen live (avg_cost == current_price == 724.12, P/L $0.00). The
    `stale_mark` flag distinguishes "genuinely flat" from "price missing"."""

    def test_stock_no_price_is_flagged_stale(self, fresh_store, monkeypatch):
        # The live MU scenario: get_prices returns nothing for the ticker.
        monkeypatch.setattr(market, "get_prices", lambda tks: {})
        fresh_store.upsert_position("MU", "stock", qty=0.5, avg_cost=724.12)
        snap = strategy._portfolio_snapshot(fresh_store)
        pos = snap["positions"][0]
        assert pos["stale_mark"] is True
        # Behaviour preserved: still falls back to avg_cost, P/L still 0.0 —
        # the flag is the ONLY thing that changed.
        assert pos["current_price"] == pytest.approx(724.12)
        assert pos["unrealized_pl"] == pytest.approx(0.0)

    def test_stock_with_price_is_not_stale(self, fresh_store, monkeypatch):
        monkeypatch.setattr(market, "get_prices", lambda tks: {"AMD": 130.0})
        fresh_store.upsert_position("AMD", "stock", qty=2, avg_cost=100.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        pos = snap["positions"][0]
        assert pos["stale_mark"] is False
        assert pos["current_price"] == pytest.approx(130.0)
        assert pos["unrealized_pl"] == pytest.approx((130.0 - 100.0) * 2)

    def test_live_option_none_price_is_flagged_stale(self, fresh_store, monkeypatch):
        # A non-expired option whose chain price is momentarily unavailable
        # still falls back to avg_cost (existing behaviour) but is now flagged.
        monkeypatch.setattr(market, "get_option_price", lambda *a, **k: None)
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2026-12-19", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        pos = snap["positions"][0]
        assert pos["stale_mark"] is True
        assert pos["current_price"] == pytest.approx(5.0)  # behaviour preserved

    def test_expired_option_intrinsic_is_not_stale(self, fresh_store, monkeypatch):
        # An expired option settled at intrinsic is a DELIBERATE, real mark —
        # not a missing price. It must NOT be flagged stale even though no
        # live chain price exists.
        monkeypatch.setattr(market, "get_price", lambda t: 650.0)  # ITM vs 600
        fresh_store.upsert_position("NVDA", "call", qty=1, avg_cost=5.0,
                                    expiry="2020-01-17", strike=600.0)
        snap = strategy._portfolio_snapshot(fresh_store)
        pos = snap["positions"][0]
        assert pos["stale_mark"] is False
        assert pos["current_price"] == pytest.approx(50.0)  # 650 - 600

    def test_build_payload_annotates_stale_position(self):
        # The core value: Opus must SEE that a $0.00 P/L is unreliable, not a
        # genuine flat position, before sizing a trade against it.
        snap = {
            "cash": 100.0, "open_value": 362.0, "total_value": 462.0,
            "positions": [
                {"ticker": "MU", "type": "stock", "qty": 0.5,
                 "avg_cost": 724.12, "current_price": 724.12,
                 "unrealized_pl": 0.0, "pl_pct": 0.0, "market_value": 362.06,
                 "stale_mark": True},
                {"ticker": "LITE", "type": "stock", "qty": 0.61,
                 "avg_cost": 980.90, "current_price": 970.71,
                 "unrealized_pl": -6.21, "pl_pct": -1.04, "market_value": 592.13,
                 "stale_mark": False},
            ],
        }
        body = strategy._build_payload(
            snap, [], [], {}, {}, None, False, quant_signals={},
        )
        lines = [ln for ln in body.splitlines() if ln.strip().startswith(("MU", "LITE"))]
        mu_line = next(ln for ln in lines if ln.strip().startswith("MU"))
        lite_line = next(ln for ln in lines if ln.strip().startswith("LITE"))
        assert "STALE MARK" in mu_line
        assert "STALE MARK" not in lite_line
        # Regression guard for the hold-age feature: these handcrafted
        # snapshots carry NO opened_at, so NO held= token must be rendered
        # (degrade-safe — byte-identical to pre-feature for this shape).
        assert "held=" not in mu_line
        assert "held=" not in lite_line


class TestHoldAgeStr:
    """`_hold_age_str` — the pure hold-age primitive surfaced into the Opus
    prompt so the decision engine can self-check the disposition effect."""

    from datetime import datetime, timedelta, timezone
    _NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)

    def test_minutes_bucket(self):
        opened = (self._NOW - self.timedelta(minutes=42)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "42m"

    def test_sub_minute_is_zero_m(self):
        opened = (self._NOW - self.timedelta(seconds=30)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "0m"

    def test_hours_bucket_floors(self):
        # 5h 59m must read "5h" (floor), not round up to 6h.
        opened = (self._NOW - self.timedelta(hours=5, minutes=59)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "5h"

    def test_days_bucket_floors(self):
        # 3d 23h must read "3d" — matches dashboard._position_ages_from_trades
        # / /api/risk day flooring so the two surfaces never disagree.
        opened = (self._NOW - self.timedelta(days=3, hours=23)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "3d"

    def test_exactly_one_hour_is_1h(self):
        opened = (self._NOW - self.timedelta(hours=1)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "1h"

    def test_exactly_one_day_is_1d(self):
        opened = (self._NOW - self.timedelta(days=1)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "1d"

    def test_missing_returns_empty(self):
        assert strategy._hold_age_str(None, now=self._NOW) == ""
        assert strategy._hold_age_str("", now=self._NOW) == ""

    def test_unparseable_returns_empty(self):
        assert strategy._hold_age_str("not-a-date", now=self._NOW) == ""

    def test_future_clamps_to_zero(self):
        # Wall clock stepped back (documented skew hazard) — never render a
        # negative age; clamp to "0m".
        opened = (self._NOW + self.timedelta(hours=5)).isoformat()
        assert strategy._hold_age_str(opened, now=self._NOW) == "0m"

    def test_naive_timestamp_treated_as_utc(self):
        # store writes tz-aware ISO, but a naive value must not crash the
        # subtraction (offset-naive vs offset-aware TypeError); treat as UTC.
        naive = (self._NOW - self.timedelta(days=2)).replace(tzinfo=None).isoformat()
        assert strategy._hold_age_str(naive, now=self._NOW) == "2d"


class TestHoldAgeInPrompt:
    """`_build_payload` must surface `held=<age>` per position when opened_at
    is present, and stay byte-identical when it is absent."""

    from datetime import datetime, timedelta, timezone

    def _snap(self, positions):
        return {"cash": 100.0, "open_value": 0.0, "total_value": 100.0,
                "positions": positions}

    def test_stock_line_shows_hold_age(self):
        opened = (self.datetime.now(self.timezone.utc)
                  - self.timedelta(days=3, hours=2)).isoformat()
        snap = self._snap([
            {"ticker": "LITE", "type": "stock", "qty": 0.61,
             "avg_cost": 980.90, "current_price": 970.71,
             "unrealized_pl": -6.21, "pl_pct": -1.04,
             "market_value": 592.13, "stale_mark": False,
             "opened_at": opened},
        ])
        body = strategy._build_payload(snap, [], [], {}, {}, None, False,
                                       quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if ln.strip().startswith("LITE"))
        assert "held=3d" in line

    def test_option_line_shows_hold_age(self):
        opened = (self.datetime.now(self.timezone.utc)
                  - self.timedelta(hours=4)).isoformat()
        snap = self._snap([
            {"ticker": "NVDA", "type": "call", "qty": 1, "strike": 600.0,
             "expiry": "2026-12-19", "avg_cost": 5.0, "current_price": 6.0,
             "unrealized_pl": 100.0, "pl_pct": 20.0, "market_value": 600.0,
             "stale_mark": False, "opened_at": opened},
        ])
        body = strategy._build_payload(snap, [], [], {}, {}, None, False,
                                       quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if ln.strip().startswith("NVDA"))
        assert "held=4h" in line

    def test_no_opened_at_renders_no_token(self):
        snap = self._snap([
            {"ticker": "MU", "type": "stock", "qty": 0.5, "avg_cost": 724.12,
             "current_price": 724.12, "unrealized_pl": 0.0, "pl_pct": 0.0,
             "market_value": 362.06, "stale_mark": False},
        ])
        body = strategy._build_payload(snap, [], [], {}, {}, None, False,
                                       quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if ln.strip().startswith("MU"))
        assert "held=" not in line

    def test_hold_age_does_not_displace_stale_marker(self):
        # held= sits before the STALE MARK suffix; both must coexist so the
        # disposition signal never masks the unreliable-P/L warning.
        opened = (self.datetime.now(self.timezone.utc)
                  - self.timedelta(days=1)).isoformat()
        snap = self._snap([
            {"ticker": "MU", "type": "stock", "qty": 0.5, "avg_cost": 724.12,
             "current_price": 724.12, "unrealized_pl": 0.0, "pl_pct": 0.0,
             "market_value": 362.06, "stale_mark": True,
             "opened_at": opened},
        ])
        body = strategy._build_payload(snap, [], [], {}, {}, None, False,
                                       quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if ln.strip().startswith("MU"))
        assert "held=1d" in line
        assert "STALE MARK" in line
        assert line.index("held=1d") < line.index("STALE MARK")


class TestSignalAgeStr:
    """`_signal_age_str` — the pure news-freshness primitive surfaced into
    the Opus prompt's TOP SCORED SIGNALS lines."""

    from datetime import datetime, timedelta, timezone
    _NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)

    def test_minutes_render(self):
        seen = (self._NOW - self.timedelta(minutes=5)).isoformat()
        assert strategy._signal_age_str(seen, now=self._NOW) == "5m"

    def test_sub_minute_is_zero_m(self):
        seen = (self._NOW - self.timedelta(seconds=40)).isoformat()
        assert strategy._signal_age_str(seen, now=self._NOW) == "0m"

    def test_minutes_floor_not_round(self):
        # 92m 59s must read "92m" (floor), never round up to 93m.
        seen = (self._NOW - self.timedelta(minutes=92, seconds=59)).isoformat()
        assert strategy._signal_age_str(seen, now=self._NOW) == "92m"

    def test_never_rolls_up_to_hours(self):
        # Unlike _hold_age_str, a 2h-old signal stays minute-grained ("120m")
        # so minute freshness is never lost to an "Nh" bucket.
        seen = (self._NOW - self.timedelta(hours=2)).isoformat()
        assert strategy._signal_age_str(seen, now=self._NOW) == "120m"

    def test_missing_returns_empty(self):
        assert strategy._signal_age_str(None, now=self._NOW) == ""
        assert strategy._signal_age_str("", now=self._NOW) == ""

    def test_unparseable_returns_empty(self):
        assert strategy._signal_age_str("not-a-date", now=self._NOW) == ""

    def test_future_clamps_to_zero(self):
        # Wall clock stepped back — never render a negative age.
        seen = (self._NOW + self.timedelta(minutes=30)).isoformat()
        assert strategy._signal_age_str(seen, now=self._NOW) == "0m"

    def test_naive_timestamp_treated_as_utc(self):
        naive = ((self._NOW - self.timedelta(minutes=15))
                 .replace(tzinfo=None).isoformat())
        assert strategy._signal_age_str(naive, now=self._NOW) == "15m"

    def test_z_suffix_timestamp_parsed(self):
        # digital-intern first_seen values can carry a trailing Z.
        seen = ((self._NOW - self.timedelta(minutes=7))
                .isoformat().replace("+00:00", "Z"))
        assert strategy._signal_age_str(seen, now=self._NOW) == "7m"


class TestSignalAgeInPrompt:
    """`_build_payload` must surface `age=<Nm>` per signal when first_seen is
    present, and stay byte-identical when it is absent."""

    from datetime import datetime, timedelta, timezone

    def _snap(self):
        return {"cash": 100.0, "open_value": 0.0, "total_value": 100.0,
                "positions": []}

    def _sig(self, **over):
        s = {"ai_score": 7.5, "urgency": 1, "title": "NVDA beats earnings",
             "tickers": ["NVDA"]}
        s.update(over)
        return s

    def test_signal_line_shows_age(self):
        seen = (self.datetime.now(self.timezone.utc)
                - self.timedelta(minutes=8)).isoformat()
        sig = self._sig(first_seen=seen)
        body = strategy._build_payload(self._snap(), [sig], [], {}, {}, None,
                                       False, quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if "NVDA beats earnings" in ln)
        assert "age=8m" in line
        # The age token sits before the title, after urg.
        assert line.index("urg=") < line.index("age=8m") < line.index("NVDA beats")

    def test_no_first_seen_renders_no_token(self):
        sig = self._sig()  # no first_seen
        body = strategy._build_payload(self._snap(), [sig], [], {}, {}, None,
                                       False, quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if "NVDA beats earnings" in ln)
        assert "age=" not in line

    def test_malformed_first_seen_renders_no_token(self):
        sig = self._sig(first_seen="garbage")
        body = strategy._build_payload(self._snap(), [sig], [], {}, {}, None,
                                       False, quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if "NVDA beats earnings" in ln)
        assert "age=" not in line

    def test_ai_score_and_urgency_still_rendered(self):
        # The age token is additive — the pre-existing score/urgency fields
        # must be untouched.
        seen = (self.datetime.now(self.timezone.utc)
                - self.timedelta(minutes=3)).isoformat()
        sig = self._sig(ai_score=9.2, urgency=2, first_seen=seen)
        body = strategy._build_payload(self._snap(), [sig], [], {}, {}, None,
                                       False, quant_signals={})
        line = next(ln for ln in body.splitlines()
                    if "NVDA beats earnings" in ln)
        assert "[9.2]" in line
        assert "urg=2" in line
        assert "age=3m" in line
        assert "tickers=NVDA" in line


class TestWatchlistHygiene:
    """The live universe must not advertise permanently-delisted tickers.

    GOOGU / METAU (single-stock 2x ETFs) were liquidated and return a
    yfinance 404 with no quote. Listing them in WATCHLIST told Opus two
    untradeable names were available (it can never fill — market.get_price
    returns None → _execute BLOCKS) and made market.get_prices(WATCHLIST)
    re-404 every _DEAD_TTL window, cluttering runner.log. WATCHLIST and the
    SYSTEM_PROMPT 'LEVERAGE INSTRUMENTS AVAILABLE' text must stay in lockstep
    so the prompt never re-introduces what the universe drops (the recurring
    pass-#18 inconsistency concern).
    """

    DELISTED = ("GOOGU", "METAU")

    def test_watchlist_excludes_delisted(self):
        for t in self.DELISTED:
            assert t not in strategy.WATCHLIST, (
                f"{t} is permanently delisted; remove from WATCHLIST")

    def test_system_prompt_excludes_delisted(self):
        for t in self.DELISTED:
            assert t not in strategy.SYSTEM_PROMPT, (
                f"{t} is delisted; remove from the LEVERAGE INSTRUMENTS text")

    def test_watchlist_and_prompt_leverage_list_agree(self):
        # Every WATCHLIST leveraged ETF that the prompt is supposed to name
        # must actually appear in the prompt, and vice-versa for the ones we
        # kept — a divergence is how a delisted name silently survives in one
        # place. Spot-check the still-live 2x single-stock names.
        for t in ("NVDU", "MSFU", "AMZU", "TSLL"):
            assert t in strategy.WATCHLIST
            assert t in strategy.SYSTEM_PROMPT
