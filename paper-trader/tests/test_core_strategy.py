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
