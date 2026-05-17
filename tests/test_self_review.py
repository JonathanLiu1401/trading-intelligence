"""Tests for analytics/self_review.py and its wiring into strategy._build_payload.

The self-review block is the behavioural mirror fed back into the LIVE Opus
decision prompt. These tests assert *correct behaviour*, not just "no crash":

* it composes the three pure builders **verbatim** (single source of truth,
  AGENTS.md #10) — the sub-reports must be byte-identical to calling the
  builders directly with the same transformed inputs, so the dashboard and
  the in-prompt block can never drift;
* the asymmetry consumer must receive **reversed** (oldest→newest) trades —
  feeding the store-native order silently diverges the verdict from
  /api/trade-asymmetry, so an order-sensitive ledger pins the reversal;
* a strong negative-edge / pinned fixture must surface the real verdict
  strings (PAYOFF_TRAP, PINNED) in the prompt block;
* it is observational, never prescriptive — the framing our layer adds
  reaffirms autonomy and issues no directives (AGENTS.md #2/#12);
* it can NEVER raise: a failing builder degrades to "no mirror", and
  strategy.decide() swallows a self-review failure rather than skipping the
  trade ("no mirror this cycle", never "no decision this cycle").
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import strategy
from paper_trader import store as store_mod
from paper_trader.store import Store
from paper_trader.analytics import self_review as sr_mod
from paper_trader.analytics.self_review import build_self_review
from paper_trader.analytics.trade_asymmetry import build_trade_asymmetry
from paper_trader.analytics.capital_paralysis import build_capital_paralysis
from paper_trader.analytics.open_attribution import build_open_attribution

NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


# ───────────────────────── fixtures / builders ──────────────────────────

def _trade(tid, ticker, action, ts, qty, px):
    return {"id": tid, "timestamp": ts, "ticker": ticker, "action": action,
            "qty": qty, "price": px, "value": qty * px,
            "strike": None, "expiry": None, "option_type": None}


def _loss_ledger(n):
    """`n` independent losing round-trips, store-native **newest-first**
    (most recent first) — exactly what store.recent_trades() returns. Each is
    buy 100 → sell 90 (a $10·qty loss) in a strictly increasing time window so
    build_round_trips closes each as its own round-trip."""
    rows = []
    tid = 1
    for k in range(n):
        buy_ts = (NOW - timedelta(days=2 * (n - k) + 1)).isoformat()
        sell_ts = (NOW - timedelta(days=2 * (n - k))).isoformat()
        rows.append(_trade(tid, "LITE", "BUY", buy_ts, 1.0, 100.0))
        rows.append(_trade(tid + 1, "LITE", "SELL", sell_ts, 1.0, 90.0))
        tid += 2
    rows.reverse()  # newest-first (store-native)
    return rows


def _pos(ticker, qty, avg, cur, opened_days_ago):
    return {"ticker": ticker, "type": "stock", "qty": qty, "avg_cost": avg,
            "current_price": cur,
            "opened_at": (NOW - timedelta(days=opened_days_ago)).isoformat()}


def _eq(total, sp500, mins_ago):
    return {"timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
            "total_value": total, "cash": 6.0, "sp500_price": sp500}


# ───────────────────────────── NO_DATA path ─────────────────────────────

class TestNoData:
    def test_empty_inputs_return_usable_block_not_none(self):
        r = build_self_review({}, [], [], [], [], now=NOW)
        assert isinstance(r["prompt_block"], str) and r["prompt_block"]
        # honest fallback, not an empty section
        assert "No closed round-trips" in r["prompt_block"]
        # summary stays truthful about the composed no-data states rather than
        # collapsing to a single label — asymmetry (NO_DATA) contributes
        # nothing, the always-present capital/open-attribution states do
        assert r["summary"] == "capital=NO_DATA · open-vs-spy=NO_BENCHMARK"
        assert "edge=" not in r["summary"]  # NO_DATA asymmetry is omitted
        # sub-reports are still present (composed, not omitted)
        assert r["asymmetry"]["state"] == "NO_DATA"
        assert r["as_of"] == NOW.isoformat(timespec="seconds")

    def test_preamble_present_in_no_data(self):
        r = build_self_review({}, [], [], [], [], now=NOW)
        assert r["prompt_block"].startswith("SELF-REVIEW")


# ───────────────────── real verdict strings surface ─────────────────────

class TestPinnedPayoffTrapSurfaces:
    """21 all-loss round-trips → STABLE PAYOFF_TRAP; $6 cash of $1000 across
    two underwater names → PINNED. Both verdicts must reach the prompt block."""

    def setup_method(self):
        trades = _loss_ledger(21)
        portfolio = {"cash": 6.0, "total_value": 1000.0}
        positions = [_pos("LITE", 1.0, 800.0, 790.0, 5),
                     _pos("NVDA", 1.0, 200.0, 180.0, 5)]
        equity = [_eq(1010.0, 5000.0, 60), _eq(972.0, 5050.0, 1)]
        self.r = build_self_review(portfolio, positions, trades, [], equity,
                                   now=NOW)

    def test_asymmetry_is_stable_payoff_trap(self):
        a = self.r["asymmetry"]
        assert a["state"] == "STABLE"
        assert a["verdict"] == "PAYOFF_TRAP"
        assert a["n_round_trips"] == 21

    def test_capital_state_is_pinned(self):
        assert self.r["capital"]["state"] == "PINNED"

    def test_prompt_block_surfaces_both_verdicts(self):
        blk = self.r["prompt_block"]
        assert "PAYOFF_TRAP" in blk
        assert "PINNED" in blk
        assert "Behavioural edge:" in blk
        assert "Capital state:" in blk
        assert "Open-book vs SPY:" in blk

    def test_summary_is_compact_and_truthful(self):
        # e.g. "edge=PAYOFF_TRAP · capital=PINNED · open-vs-spy=SELECTION_..."
        s = self.r["summary"]
        assert "edge=PAYOFF_TRAP" in s
        assert "capital=PINNED" in s


# ───────────── single source of truth — no drift, correct order ─────────

class TestSingleSourceNoDrift:
    def test_subreports_are_builders_verbatim(self):
        trades = _loss_ledger(6)
        portfolio = {"cash": 6.0, "total_value": 1000.0}
        positions = [_pos("LITE", 1.0, 800.0, 790.0, 3)]
        decisions = [{"timestamp": (NOW - timedelta(minutes=10)).isoformat(),
                      "action_taken": "HOLD LITE → HOLD"}]
        equity = [_eq(1000.0, 5000.0, 30), _eq(990.0, 5010.0, 1)]

        r = build_self_review(portfolio, positions, trades, decisions, equity,
                              now=NOW)

        # asymmetry must equal the builder called with REVERSED trades
        assert r["asymmetry"] == build_trade_asymmetry(
            list(reversed(trades)), now=NOW)
        assert r["capital"] == build_capital_paralysis(
            portfolio, positions, trades, decisions, equity, now=NOW)
        assert r["open_attribution"] == build_open_attribution(
            positions, equity, now=NOW)

    def test_asymmetry_receives_reversed_trades_not_store_order(self):
        """Order-sensitive ledger: a partial re-entry whose round-trip
        pairing differs between chronological and reversed iteration. The
        self-review's asymmetry must match the REVERSED (chronological)
        reading and NOT the store-native order — otherwise the in-prompt
        verdict silently diverges from /api/trade-asymmetry."""
        # Chronological: BUY 2@100, SELL 1@130 (partial), SELL 1@80 → one
        # round-trip closing at a net loss. Fed in the wrong order the pairing
        # and P&L differ, so the two builder calls disagree — which is exactly
        # what pins that we reversed.
        chrono = [
            _trade(1, "AMD", "BUY", (NOW - timedelta(days=5)).isoformat(), 2.0, 100.0),
            _trade(2, "AMD", "SELL", (NOW - timedelta(days=3)).isoformat(), 1.0, 130.0),
            _trade(3, "AMD", "SELL", (NOW - timedelta(days=1)).isoformat(), 1.0, 80.0),
        ]
        store_native = list(reversed(chrono))  # newest-first

        r = build_self_review({}, [], store_native, [], [], now=NOW)
        correct = build_trade_asymmetry(chrono, now=NOW)
        wrong = build_trade_asymmetry(store_native, now=NOW)

        assert r["asymmetry"] == correct
        # guard: the order actually matters for this ledger (else the test
        # proves nothing)
        assert correct != wrong


# ─────────────────── observational, never prescriptive ──────────────────

class TestObservationalNotPrescriptive:
    def test_preamble_reaffirms_autonomy(self):
        r = build_self_review({}, [], _loss_ledger(21), [], [], now=NOW)
        blk = r["prompt_block"]
        low = blk.lower()
        assert "autonomy" in low
        assert "not directives" in low or "not directives or limits" in low

    def test_layer_adds_no_imperative_directives(self):
        """The builder headlines are observational facts; our framing must not
        inject directive verbs that would turn the mirror into a cage."""
        r = build_self_review(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("LITE", 1.0, 800.0, 790.0, 5)],
            _loss_ledger(21), [], [_eq(1000.0, 5000.0, 1)], now=NOW)
        # Inspect only the structural lines our module adds (labels + preamble)
        structural = [ln for ln in r["prompt_block"].splitlines()
                      if ln.strip().startswith(("SELF-REVIEW", "Behavioural edge:",
                                                "Capital state:", "Open-book vs SPY:"))
                      or ln.strip().startswith(("Behavioural", "Capital", "Open-book"))]
        joined = " ".join(structural).lower()
        for directive in ("you must", "you should", "do not ", "you need to",
                          "sell now", "stop trading"):
            assert directive not in joined


# ──────────────────── strategy._build_payload wiring ────────────────────

class TestBuildPayloadWiring:
    _SNAP = {"cash": 0.0, "open_value": 0.0, "total_value": 0.0,
             "positions": []}

    def test_block_is_injected_into_payload(self):
        out = strategy._build_payload(
            self._SNAP, [], [], {}, {}, None, False,
            self_review_block="SENTINEL_MIRROR_XYZ")
        assert "SENTINEL_MIRROR_XYZ" in out
        # placed between PORTFOLIO and WATCHLIST PRICES
        assert out.index("SENTINEL_MIRROR_XYZ") < out.index("WATCHLIST PRICES")
        assert out.index("PORTFOLIO:") < out.index("SENTINEL_MIRROR_XYZ")

    def test_none_block_is_backward_compatible(self):
        out = strategy._build_payload(self._SNAP, [], [], {}, {}, None, False)
        assert "SELF-REVIEW" not in out
        assert "WATCHLIST PRICES" in out  # still a valid payload, no crash


# ────────────────────────── never-raises guard ──────────────────────────

class TestNeverRaises:
    def test_failing_builder_degrades_to_no_mirror_not_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("simulated builder fault")

        # Break the asymmetry builder *as seen by self_review*.
        monkeypatch.setattr(sr_mod, "build_trade_asymmetry", _boom)
        r = build_self_review(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("LITE", 1.0, 800.0, 790.0, 5)],
            _loss_ledger(5), [], [_eq(1000.0, 5000.0, 1)], now=NOW)
        # No exception; still a usable block from the surviving builders.
        assert isinstance(r["prompt_block"], str) and r["prompt_block"]
        assert r["asymmetry"]["state"] == "ERROR"
        # the capital line still renders (PINNED) — one bad builder doesn't
        # sink the mirror
        assert "Capital state:" in r["prompt_block"]

    def test_decide_swallows_self_review_failure_and_still_returns(
            self, tmp_path, monkeypatch):
        """The live-cycle contract: a self-review fault must NOT skip the
        decision. decide() must return a summary even when build_self_review
        raises (failure mode = 'no mirror', never 'no decision')."""
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)

        # Make the self-review explode at the point decide() composes it.
        def _boom(*a, **k):
            raise RuntimeError("self-review blew up")
        monkeypatch.setattr(sr_mod, "build_self_review", _boom)

        from paper_trader import market
        monkeypatch.setattr(market, "is_market_open", lambda: False)
        monkeypatch.setattr(market, "get_prices", lambda tks: {})
        monkeypatch.setattr(market, "get_futures_price", lambda f: None)
        monkeypatch.setattr(market, "benchmark_sp500", lambda: None)
        monkeypatch.setattr(strategy.signals, "get_top_signals",
                            lambda *a, **k: [])
        monkeypatch.setattr(strategy.signals, "get_urgent_articles",
                            lambda *a, **k: [])
        monkeypatch.setattr(strategy.signals, "ticker_sentiments",
                            lambda *a, **k: [])
        monkeypatch.setattr(strategy, "get_quant_signals_live",
                            lambda tks: {})
        # No claude → NO_DECISION path, no retry (raw is None).
        monkeypatch.setattr(strategy, "_claude_call", lambda *a, **k: None)

        summary = strategy.decide()  # must not raise
        assert isinstance(summary, dict)
        assert summary["status"] == "NO_DECISION"
