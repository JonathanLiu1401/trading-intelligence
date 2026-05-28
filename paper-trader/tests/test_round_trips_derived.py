"""Tests for paper_trader.analytics.round_trips_derived + the
``/api/round-trips-derived``, ``/api/today-realized-pl-derived``, and
``/api/realized-pl-reconciliation`` endpoints.

The pure-builder tests pin algorithm correctness against
hand-constructed trade sequences; the endpoint tests drive
the real Flask views through ``app.test_client()`` against a fresh
temp Store so a CI run never touches the live paper_trader.db.

This data-integrity gap (round-trips hidden by position-row reactivation,
~$217 of realized P/L invisible from /api/closed-positions on 2026-05-27)
was the day's most material finding before this pass; these tests pin the
discovery so a regression in either the walker or the reconciliation
fingerprint is caught immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.round_trips_derived import (  # noqa: E402
    derive_round_trips,
    reconcile,
    summarize,
)


def _trade(ts, ticker, action, qty, value, *, option_type=None,
           expiry=None, strike=None, id_=None):
    """Build a trade row in the shape ``store.recent_trades(N)`` returns."""
    return {
        "id": id_,
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": (value / qty) if qty else 0.0,
        "value": value,
        "expiry": expiry,
        "strike": strike,
        "option_type": option_type,
    }


# ── Pure builder: derive_round_trips ─────────────────────────────────────────

class TestDeriveBasic:
    def test_empty_input_returns_empty_list(self):
        assert derive_round_trips([]) == []

    def test_none_input_returns_empty_list(self):
        assert derive_round_trips(None) == []

    def test_single_complete_round_trip(self):
        trades = [
            _trade("2026-05-26T11:41:12.724624+00:00", "MU", "BUY", 1.3, 976.03),
            _trade("2026-05-26T17:13:53.070980+00:00", "MU", "SELL", 1.3, 1156.35),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        rt = result[0]
        assert rt["ticker"] == "MU"
        assert rt["type"] == "stock"
        assert rt["realized_pl"] == 180.32
        assert rt["cost"] == 976.03
        assert rt["proceeds"] == 1156.35
        assert rt["n_trades"] == 2
        assert rt["opened_at"] == "2026-05-26T11:41:12.724624+00:00"
        assert rt["closed_at"] == "2026-05-26T17:13:53.070980+00:00"
        assert rt["realized_pl_pct"] == round(180.32 / 976.03 * 100.0, 2)

    def test_open_position_not_emitted(self):
        # BUY only — no SELL → round-trip never closes → not emitted.
        trades = [
            _trade("2026-05-27T20:15:49+00:00", "MU", "BUY", 1.0, 928.41),
        ]
        assert derive_round_trips(trades) == []

    def test_open_after_close_does_not_re_emit_prior_round_trip(self):
        # Two complete round-trips followed by an open BUY (the live MU
        # state on 2026-05-27): 2 round-trips emitted, the open lot ignored.
        trades = [
            _trade("2026-05-26T11:41:12+00:00", "MU", "BUY", 1.3, 976.03),
            _trade("2026-05-26T17:13:53+00:00", "MU", "SELL", 1.3, 1156.35),
            _trade("2026-05-27T12:52:28+00:00", "MU", "BUY", 1.0, 896.60),
            _trade("2026-05-27T20:09:13+00:00", "MU", "SELL", 1.0, 928.41),
            _trade("2026-05-27T20:15:49+00:00", "MU", "BUY", 1.0, 928.41),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        # Sorted newest-closed first.
        assert result[0]["closed_at"].startswith("2026-05-27T20:09:13")
        assert result[0]["realized_pl"] == 31.81
        assert result[1]["closed_at"].startswith("2026-05-26T17:13:53")
        assert result[1]["realized_pl"] == 180.32


class TestDeriveMultiLeg:
    def test_buy_add_buy_then_sell_all_is_one_round_trip(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "NVDA", "BUY", 1.0, 200.0),
            _trade("2026-05-19T11:00:00+00:00", "NVDA", "BUY", 1.0, 210.0),
            _trade("2026-05-19T15:00:00+00:00", "NVDA", "SELL", 2.0, 440.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        rt = result[0]
        # Cost = 200 + 210 = 410, proceeds = 440, realized = +30
        assert rt["cost"] == 410.0
        assert rt["proceeds"] == 440.0
        assert rt["realized_pl"] == 30.0
        assert rt["n_trades"] == 3
        # Opened_at = first BUY ts; closed_at = last SELL ts.
        assert rt["opened_at"] == "2026-05-19T10:00:00+00:00"
        assert rt["closed_at"] == "2026-05-19T15:00:00+00:00"

    def test_partial_sell_then_more_buys_then_full_close_is_one_round_trip(self):
        # BUY 1 → SELL 0.5 (held=0.5) → BUY 1.5 (held=2.0) → SELL 2.0 (held=0)
        # held only returns to ≈0 at the very end so this is ONE round-trip.
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "TSLA", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "TSLA", "SELL", 0.5, 55.0),
            _trade("2026-05-19T12:00:00+00:00", "TSLA", "BUY", 1.5, 165.0),
            _trade("2026-05-19T13:00:00+00:00", "TSLA", "SELL", 2.0, 230.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        rt = result[0]
        # Cost = 100 + 165 = 265, proceeds = 55 + 230 = 285, realized = +20
        assert rt["cost"] == 265.0
        assert rt["proceeds"] == 285.0
        assert rt["realized_pl"] == 20.0
        assert rt["n_trades"] == 4

    def test_two_independent_round_trips_on_same_key(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "SELL", 1.0, 105.0),
            _trade("2026-05-20T10:00:00+00:00", "MU", "BUY", 1.0, 110.0),
            _trade("2026-05-20T11:00:00+00:00", "MU", "SELL", 1.0, 112.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        # Newest-closed first.
        assert result[0]["closed_at"].startswith("2026-05-20T11:00:00")
        assert result[0]["realized_pl"] == 2.0
        assert result[1]["closed_at"].startswith("2026-05-19T11:00:00")
        assert result[1]["realized_pl"] == 5.0


class TestDeriveKeyIsolation:
    def test_different_tickers_dont_cross_contaminate(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "NVDA", "BUY", 1.0, 200.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
            _trade("2026-05-19T13:00:00+00:00", "NVDA", "SELL", 1.0, 195.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        by_tk = {rt["ticker"]: rt for rt in result}
        assert by_tk["MU"]["realized_pl"] == 5.0
        assert by_tk["NVDA"]["realized_pl"] == -5.0

    def test_stock_vs_option_same_ticker_dont_cross(self):
        # A stock BUY-SELL and an option BUY-SELL on the same ticker must
        # produce TWO independent round-trips, not collapse via the BUY
        # crossing into the SELL on the other key.
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY_CALL", 1.0, 500.0,
                   option_type="call", expiry="2026-06-19", strike=110.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
            _trade("2026-05-19T13:00:00+00:00", "MU", "SELL_CALL", 1.0, 600.0,
                   option_type="call", expiry="2026-06-19", strike=110.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        # The stock round-trip and option round-trip should both close
        # cleanly without contamination.
        stocks = [r for r in result if r["type"] == "stock"]
        calls = [r for r in result if r["type"] == "call"]
        assert len(stocks) == 1
        assert len(calls) == 1
        assert stocks[0]["realized_pl"] == 5.0
        assert calls[0]["realized_pl"] == 100.0
        assert calls[0]["expiry"] == "2026-06-19"
        assert calls[0]["strike"] == 110.0

    def test_same_ticker_two_strikes_dont_cross(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY_CALL", 1.0, 500.0,
                   option_type="call", expiry="2026-06-19", strike=110.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY_CALL", 1.0, 300.0,
                   option_type="call", expiry="2026-06-19", strike=120.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL_CALL", 1.0, 550.0,
                   option_type="call", expiry="2026-06-19", strike=110.0),
            _trade("2026-05-19T13:00:00+00:00", "MU", "SELL_CALL", 1.0, 320.0,
                   option_type="call", expiry="2026-06-19", strike=120.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        by_strike = {r["strike"]: r for r in result}
        assert by_strike[110.0]["realized_pl"] == 50.0
        assert by_strike[120.0]["realized_pl"] == 20.0


class TestDeriveBadInputs:
    def test_non_dict_rows_skipped(self):
        trades = [
            "not a dict",
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "SELL", 1.0, 105.0),
            42,
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        assert result[0]["realized_pl"] == 5.0

    def test_row_missing_ticker_skipped(self):
        trades = [
            {"timestamp": "2026-05-19T10:00:00+00:00",
             "action": "BUY", "qty": 1.0, "value": 100.0},
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        assert result[0]["ticker"] == "MU"

    def test_row_missing_action_skipped(self):
        trades = [
            {"timestamp": "2026-05-19T10:00:00+00:00",
             "ticker": "MU", "qty": 1.0, "value": 100.0},
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1

    def test_none_qty_treated_as_zero(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", None, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        # First BUY adds qty=0 → held stays 0 → start_idx anchors at it;
        # second BUY adds qty=1 → held=1; SELL closes the round-trip.
        # n_trades = 3 (the None-qty BUY counts as a trade — it ran, even
        # though it moved no shares; this mirrors store.closed_positions'
        # behaviour of walking every action in the slice).
        result = derive_round_trips(trades)
        assert len(result) == 1
        assert result[0]["realized_pl"] == 5.0

    def test_non_buy_sell_actions_ignored(self):
        # REBALANCE, OPEN, etc. — not BUY/SELL-prefixed — never advance
        # held and never start a round-trip. (Matches store.closed_positions
        # behaviour for the same reason.)
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "REBALANCE", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T12:00:00+00:00", "MU", "HOLD", 0.0, 0.0),
            _trade("2026-05-19T13:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        # The HOLD inside the slice IS counted in n_trades because it sits
        # within the round-trip's [open, close] index window — that's
        # fine; the realized_pl math correctly ignores its (zero) value.
        assert result[0]["realized_pl"] == 5.0


class TestDeriveBuyAliases:
    def test_buy_call_buy_put_are_buys(self):
        # The strategy emits BUY, BUY_CALL, BUY_PUT (and the SELL_*
        # counterparts). The walker matches via ``startswith("BUY")``.
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY_PUT", 1.0, 200.0,
                   option_type="put", expiry="2026-06-19", strike=100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "SELL_PUT", 1.0, 220.0,
                   option_type="put", expiry="2026-06-19", strike=100.0),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        assert result[0]["type"] == "put"
        assert result[0]["realized_pl"] == 20.0


class TestDeriveOrdering:
    def test_order_independent_input_yields_same_round_trips(self):
        # Pass trades newest-first (the order ``store.recent_trades`` gives)
        # — the walker must internally sort to chronological order before
        # walking.
        trades = [
            _trade("2026-05-19T13:00:00+00:00", "MU", "SELL", 1.0, 105.0,
                   id_=4),
            _trade("2026-05-19T12:00:00+00:00", "MU", "BUY", 1.0, 100.0,
                   id_=3),
            _trade("2026-05-19T11:00:00+00:00", "TSLA", "SELL", 1.0, 220.0,
                   id_=2),
            _trade("2026-05-19T10:00:00+00:00", "TSLA", "BUY", 1.0, 200.0,
                   id_=1),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 2
        by_tk = {r["ticker"]: r for r in result}
        assert by_tk["MU"]["realized_pl"] == 5.0
        assert by_tk["TSLA"]["realized_pl"] == 20.0

    def test_limit_caps_returned_length(self):
        # 5 independent round-trips on one key; limit=2 returns the 2
        # newest-closed only.
        trades: list[dict] = []
        for i, day in enumerate(range(20, 25)):
            trades.append(_trade(
                f"2026-05-{day:02d}T10:00:00+00:00", "MU", "BUY", 1.0, 100.0,
            ))
            trades.append(_trade(
                f"2026-05-{day:02d}T11:00:00+00:00", "MU", "SELL", 1.0, 105.0,
            ))
        result = derive_round_trips(trades, limit=2)
        assert len(result) == 2
        # Both should be from the latest days (24, 23).
        assert result[0]["closed_at"].startswith("2026-05-24")
        assert result[1]["closed_at"].startswith("2026-05-23")

    def test_limit_zero_returns_empty(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        assert derive_round_trips(trades, limit=0) == []

    def test_limit_none_returns_all(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            _trade("2026-05-19T11:00:00+00:00", "MU", "SELL", 1.0, 105.0),
            _trade("2026-05-20T10:00:00+00:00", "MU", "BUY", 1.0, 110.0),
            _trade("2026-05-20T11:00:00+00:00", "MU", "SELL", 1.0, 115.0),
        ]
        assert len(derive_round_trips(trades, limit=None)) == 2


class TestHoldDuration:
    def test_hold_seconds_and_days_computed(self):
        trades = [
            _trade("2026-05-19T10:00:00+00:00", "MU", "BUY", 1.0, 100.0),
            # Closes 2h = 7200s = 0.0833 days later.
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0),
        ]
        rt = derive_round_trips(trades)[0]
        assert rt["hold_seconds"] == 7200
        assert rt["hold_days"] == 0.0833

    def test_unparseable_timestamp_yields_none_holds(self):
        # An ISO-shaped but fromisoformat-incompatible timestamp (the
        # year-2099 ZZ suffix here) sorts lexically with the other ISO
        # strings — so the walker still pairs the trades into a
        # round-trip — but ``_hold_duration`` cannot parse it. Result:
        # the round-trip emits with hold_seconds / hold_days = None,
        # never raises.
        from paper_trader.analytics.round_trips_derived import _hold_duration
        # Sanity: the helper itself drops to (None, None) on a bad string.
        assert _hold_duration("not-an-iso", "2026-05-19T12:00:00+00:00") == (None, None)
        assert _hold_duration(None, "2026-05-19T12:00:00+00:00") == (None, None)
        assert _hold_duration("2026-05-19T10:00:00+00:00", None) == (None, None)
        # And: a round-trip with a None-timestamp BUY (the walker still
        # forms it via key grouping; only _hold_duration fails) drops the
        # hold-time fields to None without crashing.
        trades = [
            {"timestamp": None, "ticker": "MU", "action": "BUY",
             "qty": 1.0, "value": 100.0, "id": 1},
            _trade("2026-05-19T12:00:00+00:00", "MU", "SELL", 1.0, 105.0,
                   id_=2),
        ]
        result = derive_round_trips(trades)
        assert len(result) == 1
        assert result[0]["hold_seconds"] is None
        assert result[0]["hold_days"] is None


# ── reconcile ───────────────────────────────────────────────────────────────

class TestReconcile:
    def test_consistent_when_every_trades_rt_matches_a_position(self):
        # Identical realized P/L on identical key, opened_at-to-seconds
        # matches → fingerprint matches → no hidden round-trips.
        positions = [{
            "ticker": "MU", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-19T10:00:00.123456+00:00",
            "closed_at": "2026-05-19T11:00:00.789012+00:00",
            "realized_pl": 5.0, "cost": 100.0, "proceeds": 105.0,
        }]
        derived = [{
            "ticker": "MU", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-19T10:00:00.111+00:00",  # µs gap is fine
            "closed_at": "2026-05-19T11:00:00.222+00:00",
            "realized_pl": 5.0, "cost": 100.0, "proceeds": 105.0,
        }]
        r = reconcile(positions, derived)
        assert r["verdict"] == "CONSISTENT"
        assert r["n_hidden"] == 0
        assert r["hidden_realized_usd"] == 0.0
        assert r["hidden_tickers"] == []

    def test_hidden_when_trades_rt_has_no_position_match(self):
        positions = []
        derived = [{
            "ticker": "MU", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-26T11:41:12.724624+00:00",
            "closed_at": "2026-05-26T17:13:53.070980+00:00",
            "realized_pl": 180.32, "cost": 976.03, "proceeds": 1156.35,
        }]
        r = reconcile(positions, derived)
        assert r["verdict"] == "HIDDEN_REALIZED_PL"
        assert r["n_hidden"] == 1
        assert r["hidden_realized_usd"] == 180.32
        assert r["hidden_tickers"] == ["MU"]
        assert "MU" in r["headline"]
        assert "$180.32" in r["headline"]

    def test_no_data_verdict_on_both_empty(self):
        r = reconcile([], [])
        assert r["verdict"] == "NO_DATA"
        assert r["n_hidden"] == 0
        assert r["n_positions_visible"] == 0
        assert r["n_trades_derived"] == 0

    def test_microsecond_close_at_gap_does_not_mark_lot_as_hidden(self):
        # Live evidence: position.closed_at is set ~µs AFTER the SELL
        # trade.timestamp by upsert_position's UPDATE — they differ at
        # the microsecond grain but represent the SAME closed lot. The
        # cent-rounded realized_pl + opened_at-to-seconds fingerprint
        # collapses them; a closed_at-only fingerprint would mis-mark
        # every visible lot as also hidden.
        positions = [{
            "ticker": "NVDA", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-21T01:36:06.692530+00:00",
            "closed_at": "2026-05-24T02:08:08.434440+00:00",
            "realized_pl": -24.55, "cost": 670.30,
        }]
        derived = [{
            "ticker": "NVDA", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-21T01:36:06.684121+00:00",
            "closed_at": "2026-05-24T02:08:08.434223+00:00",
            "realized_pl": -24.55, "cost": 670.30,
        }]
        r = reconcile(positions, derived)
        assert r["verdict"] == "CONSISTENT"
        assert r["n_hidden"] == 0

    def test_multiple_hidden_round_trips_aggregated(self):
        # Live evidence on 2026-05-27: 2 hidden MU lots (+$180.32, +$31.81)
        # plus 1 hidden NVDA lot (+$4.97). Total = +$217.10.
        positions = []
        derived = [
            {"ticker": "MU", "type": "stock", "expiry": None, "strike": None,
             "opened_at": "2026-05-26T11:41:12+00:00",
             "closed_at": "2026-05-26T17:13:53+00:00",
             "realized_pl": 180.32, "cost": 976.03},
            {"ticker": "MU", "type": "stock", "expiry": None, "strike": None,
             "opened_at": "2026-05-27T12:52:28+00:00",
             "closed_at": "2026-05-27T20:09:13+00:00",
             "realized_pl": 31.81, "cost": 896.60},
            {"ticker": "NVDA", "type": "stock", "expiry": None, "strike": None,
             "opened_at": "2026-05-19T00:42:15+00:00",
             "closed_at": "2026-05-21T01:13:38+00:00",
             "realized_pl": 4.97, "cost": 1000.48},
        ]
        r = reconcile(positions, derived)
        assert r["verdict"] == "HIDDEN_REALIZED_PL"
        assert r["n_hidden"] == 3
        assert r["hidden_realized_usd"] == 217.10
        assert sorted(r["hidden_tickers"]) == ["MU", "NVDA"]

    def test_handles_none_inputs(self):
        # Both inputs None — never raise; same envelope.
        r = reconcile(None, None)
        assert r["verdict"] == "NO_DATA"
        # One side None — degrade gracefully.
        r2 = reconcile(None, [{
            "ticker": "MU", "type": "stock",
            "expiry": None, "strike": None,
            "opened_at": "2026-05-19T10:00:00+00:00",
            "closed_at": "2026-05-19T11:00:00+00:00",
            "realized_pl": 5.0,
        }])
        assert r2["verdict"] == "HIDDEN_REALIZED_PL"
        assert r2["n_hidden"] == 1


# ── summarize ───────────────────────────────────────────────────────────────

class TestSummarize:
    def test_empty_summary(self):
        s = summarize([])
        assert s["n"] == 0
        assert s["wins"] == 0
        assert s["losses"] == 0
        assert s["total_realized_pl"] == 0.0
        assert s["win_rate_pct"] is None
        assert s["avg_realized_pl_pct"] is None

    def test_summary_aggregates_wins_losses(self):
        derived = [
            {"realized_pl": 10.0, "cost": 100.0, "proceeds": 110.0},
            {"realized_pl": -5.0, "cost": 50.0, "proceeds": 45.0},
            {"realized_pl": 3.0, "cost": 30.0, "proceeds": 33.0},
        ]
        s = summarize(derived)
        assert s["n"] == 3
        assert s["wins"] == 2
        assert s["losses"] == 1
        assert s["flat"] == 0
        assert s["total_realized_pl"] == 8.0
        assert s["total_cost"] == 180.0
        assert s["total_proceeds"] == 188.0
        assert s["win_rate_pct"] == round(100 * 2 / 3, 2)
        # avg_realized_pl_pct = 8 / 180 * 100
        assert s["avg_realized_pl_pct"] == round(8.0 / 180.0 * 100.0, 2)

    def test_flat_round_trip_counted(self):
        s = summarize([{"realized_pl": 0.0, "cost": 100.0, "proceeds": 100.0}])
        assert s["wins"] == 0
        assert s["losses"] == 0
        assert s["flat"] == 1


# ── Endpoint envelopes (Flask test client) ──────────────────────────────────

class TestEndpoints:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from paper_trader import dashboard as dash_mod
        from paper_trader.store import Store
        db_path = tmp_path / "paper_trader.db"
        monkeypatch.setattr("paper_trader.store.DB_PATH", db_path)
        monkeypatch.setattr("paper_trader.store._singleton", None)
        fresh = Store()
        monkeypatch.setattr(dash_mod, "get_store", lambda: fresh)
        monkeypatch.setattr("paper_trader.store.get_store", lambda: fresh)
        dash_mod.app.config["TESTING"] = True
        return dash_mod.app.test_client(), fresh

    def test_round_trips_derived_empty_store_returns_empty_envelope(self, client):
        c, _ = client
        r = c.get("/api/round-trips-derived")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "OK"
        assert body["round_trips"] == []
        assert body["summary"]["n"] == 0
        assert "as_of" in body

    def test_round_trips_derived_with_real_trades(self, client):
        c, store = client
        # One complete round-trip in the trades table.
        store.record_trade("MU", "BUY", 1.0, 100.0)
        store.record_trade("MU", "SELL", 1.0, 105.0)
        r = c.get("/api/round-trips-derived")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "OK"
        assert len(body["round_trips"]) == 1
        rt = body["round_trips"][0]
        assert rt["ticker"] == "MU"
        assert rt["realized_pl"] == 5.0
        assert body["summary"]["n"] == 1
        assert body["summary"]["wins"] == 1

    def test_round_trips_derived_limit_clamped(self, client):
        c, _ = client
        # Out-of-range limit silently coerces. Envelope still well-formed.
        r = c.get("/api/round-trips-derived?limit=99999")
        assert r.status_code == 200
        assert r.get_json()["verdict"] == "OK"
        r2 = c.get("/api/round-trips-derived?limit=garbage")
        assert r2.status_code == 200
        assert r2.get_json()["verdict"] == "OK"

    def test_today_realized_pl_derived_envelope_shape(self, client):
        c, _ = client
        r = c.get("/api/today-realized-pl-derived")
        assert r.status_code == 200
        body = r.get_json()
        for k in ("verdict", "headline", "ny_date", "net_realized_usd",
                  "n_closes", "biggest_win", "biggest_loss", "closes",
                  "source", "as_of"):
            assert k in body, f"missing key: {k!r}"
        # Empty store → NO_CLOSES_TODAY.
        assert body["verdict"] == "NO_CLOSES_TODAY"
        # The new endpoint marks its source for callers comparing it to the
        # canonical /api/today-realized-pl.
        assert body["source"] == "trades-derived"

    def test_today_realized_pl_derived_surfaces_reactivation_hidden_close(
            self, client, monkeypatch):
        c, store = client
        # Simulate the live MU 2026-05-27 morning round-trip whose
        # position row was then reactivated (so /api/today-realized-pl
        # returns NO_CLOSES_TODAY despite the real SELL). The trades-derived
        # endpoint must surface it.
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        ny = ZoneInfo("America/New_York")
        today_ny = datetime.now(timezone.utc).astimezone(ny).date()
        # Put both trades at today's NY-midday so the filter catches them.
        morning = datetime(
            today_ny.year, today_ny.month, today_ny.day, 10, 0, tzinfo=ny,
        ).astimezone(timezone.utc).isoformat()
        afternoon = datetime(
            today_ny.year, today_ny.month, today_ny.day, 14, 0, tzinfo=ny,
        ).astimezone(timezone.utc).isoformat()
        # We need to inject the trades with specific timestamps; record_trade
        # uses _now() so we have to bypass it for the test.
        with store._lock:  # noqa: SLF001
            store.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, "
                "value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (morning, "MU", "BUY", 1.0, 100.0, 100.0, "", None, None, None),
            )
            store.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, "
                "value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (afternoon, "MU", "SELL", 1.0, 110.0, 110.0, "", None, None,
                 None),
            )
            # Now reactivate the position by inserting a third BUY trade
            # immediately after (mimics the live MU reactivation pattern).
            store.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, "
                "value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (afternoon, "MU", "BUY", 1.0, 110.0, 110.0, "", None, None,
                 None),
            )
            store.conn.commit()

        r = c.get("/api/today-realized-pl-derived")
        assert r.status_code == 200
        body = r.get_json()
        # The completed round-trip is visible despite the subsequent BUY
        # (which would have reactivated the position row in the live flow).
        assert body["verdict"] == "WINNING_DAY"
        assert body["n_closes"] == 1
        assert body["n_winners"] == 1
        assert body["net_realized_usd"] == 10.0
        assert body["closes"][0]["ticker"] == "MU"

    def test_reconciliation_endpoint_envelope_on_empty_store(self, client):
        c, _ = client
        r = c.get("/api/realized-pl-reconciliation")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "NO_DATA"
        assert body["n_hidden"] == 0
        assert body["hidden_realized_usd"] == 0.0
        for k in ("verdict", "headline", "n_positions_visible",
                  "n_trades_derived", "n_hidden", "hidden_realized_usd",
                  "hidden_tickers", "as_of"):
            assert k in body

    def test_reconciliation_endpoint_consistent_on_clean_state(self, client):
        c, store = client
        # A round-trip that goes to FULL closure (no reactivation) — the
        # position row closes_at IS NOT NULL, both sources see it.
        store.record_trade("DRAM", "BUY", 1.0, 100.0)
        store.upsert_position("DRAM", "stock", 1.0, 100.0)
        store.record_trade("DRAM", "SELL", 1.0, 105.0)
        store.upsert_position("DRAM", "stock", -1.0, 105.0)  # closes the lot
        r = c.get("/api/realized-pl-reconciliation")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "CONSISTENT"
        assert body["n_hidden"] == 0
        assert body["n_positions_visible"] == 1
        assert body["n_trades_derived"] == 1

    def test_reconciliation_endpoint_flags_hidden_after_reactivation(
            self, client):
        c, store = client
        # Two complete round-trips + a third open BUY that reactivates the
        # position row. /api/closed-positions sees 0 closed lots (row is
        # back to closed_at=NULL); the trades log sees 2 round-trips. The
        # diagnostic should flag both as hidden.
        store.record_trade("MU", "BUY", 1.0, 100.0)
        store.upsert_position("MU", "stock", 1.0, 100.0)
        store.record_trade("MU", "SELL", 1.0, 110.0)
        store.upsert_position("MU", "stock", -1.0, 110.0)
        store.record_trade("MU", "BUY", 1.0, 110.0)
        store.upsert_position("MU", "stock", 1.0, 110.0)  # reactivates
        store.record_trade("MU", "SELL", 1.0, 115.0)
        store.upsert_position("MU", "stock", -1.0, 115.0)
        store.record_trade("MU", "BUY", 1.0, 115.0)
        store.upsert_position("MU", "stock", 1.0, 115.0)  # reactivates again
        # Sanity: closed_positions should return 0 (row is currently open).
        assert store.closed_positions() == []
        # Reconciliation must now flag both round-trips as hidden.
        r = c.get("/api/realized-pl-reconciliation")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "HIDDEN_REALIZED_PL"
        assert body["n_hidden"] == 2
        # +10 + +5 = +15 hidden $.
        assert body["hidden_realized_usd"] == 15.0
        assert "MU" in body["hidden_tickers"]
