"""Tests for analytics.exit_trigger_pl_mix + /api/exit-trigger-pl-mix.

Locks the per-trigger round-trip rollup against hand-built trade slices.
Pins the verdict ladder and confirms the endpoint envelope shape via the
Flask test client (the recommended verification path per CLAUDE.md —
module __main__ smokes hit a different/empty DB).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.exit_trigger_pl_mix import (
    _classify_reason,
    _verdict,
    build,
    derive_trigger_round_trips,
)


def _t(
    *,
    action: str,
    ticker: str = "NVDA",
    qty: float = 1.0,
    price: float = 100.0,
    value: float | None = None,
    reason: str = "",
    timestamp: str = "2026-05-29T12:00:00+00:00",
    option_type: str | None = None,
    expiry: str | None = None,
    strike: float | None = None,
    id_: int = 0,
) -> dict:
    return {
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "value": value if value is not None else qty * price,
        "reason": reason,
        "timestamp": timestamp,
        "option_type": option_type,
        "expiry": expiry,
        "strike": strike,
        "id": id_,
    }


# ───────────────────────── _classify_reason ─────────────────────────


class TestClassifyReason:
    def test_hard_sl_prefix(self):
        assert _classify_reason(
            "HARD_SL: price 95.00 <= threshold 96.00"
        ) == "HARD_SL"

    def test_hard_tp_prefix(self):
        assert _classify_reason(
            "HARD_TP: price 110.00 >= threshold 109.00"
        ) == "HARD_TP"

    def test_discretionary_prose(self):
        # The most recent live MU rotation reason — must classify
        # as DISCRETIONARY, not HARD_*.
        assert _classify_reason(
            "Rotating fresh capital into the cleanest setup; thesis "
            "broken on NVDA after MACD bearish cross."
        ) == "DISCRETIONARY"

    def test_none(self):
        assert _classify_reason(None) == "DISCRETIONARY"

    def test_empty(self):
        assert _classify_reason("") == "DISCRETIONARY"

    def test_substring_not_matched(self):
        # "the HARD_SL fired earlier" must NOT be misclassified — only
        # the strict leading-prefix wins. The strategy.py exit reason
        # always starts with the marker; substring is operator prose.
        assert _classify_reason(
            "tightening risk because HARD_SL fired earlier"
        ) == "DISCRETIONARY"


# ───────────────────────── derive_trigger_round_trips ─────────────────────────


class TestDeriveTriggerRoundTrips:
    def test_empty_input(self):
        assert derive_trigger_round_trips([]) == []
        assert derive_trigger_round_trips(None) == []

    def test_open_position_not_emitted(self):
        # BUY with no closing SELL: held never returns to zero.
        rows = [_t(action="BUY", ticker="NVDA", qty=1, price=100,
                   timestamp="2026-05-29T12:00:00+00:00", id_=1)]
        assert derive_trigger_round_trips(rows) == []

    def test_single_clean_round_trip(self):
        rows = [
            _t(action="BUY", ticker="NVDA", qty=1, price=100,
               timestamp="2026-05-29T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=120,
               reason="HARD_TP: price 120.00 >= threshold 115.00",
               timestamp="2026-05-29T15:00:00+00:00", id_=2),
        ]
        out = derive_trigger_round_trips(rows)
        assert len(out) == 1
        rt = out[0]
        assert rt["ticker"] == "NVDA"
        assert rt["bucket"] == "HARD_TP"
        assert rt["realized_pl"] == 20.0
        assert rt["cost"] == 100.0
        assert rt["proceeds"] == 120.0

    def test_stock_and_option_share_ticker_do_not_bleed(self):
        # Stock leg + option leg on the same ticker each close as their
        # own round-trip, never collapse together.
        rows = [
            # Stock
            _t(action="BUY", ticker="NVDA", qty=1, price=100,
               timestamp="2026-05-28T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=110,
               reason="Thesis intact, taking +10%",
               timestamp="2026-05-28T15:00:00+00:00", id_=2),
            # Option leg (separate key)
            _t(action="BUY", ticker="NVDA", qty=1, price=5,
               option_type="call", expiry="2026-06-20", strike=200.0,
               timestamp="2026-05-29T12:00:00+00:00", id_=3),
            _t(action="SELL", ticker="NVDA", qty=1, price=8,
               option_type="call", expiry="2026-06-20", strike=200.0,
               reason="HARD_TP: price 8.00 >= threshold 7.50",
               timestamp="2026-05-29T14:00:00+00:00", id_=4),
        ]
        out = derive_trigger_round_trips(rows)
        assert len(out) == 2
        # Newest-closed first → option leg leads.
        assert out[0]["type"] == "call"
        assert out[0]["bucket"] == "HARD_TP"
        assert out[1]["type"] == "stock"
        assert out[1]["bucket"] == "DISCRETIONARY"

    def test_three_buckets_distinct(self):
        rows = [
            # HARD_SL trip
            _t(action="BUY", ticker="AMD", qty=1, price=520,
               timestamp="2026-05-28T18:44:45+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=505,
               reason="HARD_SL: price 505.21 <= threshold 508.67",
               timestamp="2026-05-29T17:25:12+00:00", id_=2),
            # HARD_TP trip
            _t(action="BUY", ticker="MU", qty=1, price=900,
               timestamp="2026-05-26T18:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=928,
               reason="HARD_TP: price 928.41 >= threshold 923.50",
               timestamp="2026-05-27T20:09:13+00:00", id_=4),
            # Discretionary trip
            _t(action="BUY", ticker="NVDA", qty=3, price=213,
               timestamp="2026-05-28T17:25:59+00:00", id_=5),
            _t(action="SELL", ticker="NVDA", qty=3, price=214,
               reason="Thesis WEAKENING; rotating fresh capital",
               timestamp="2026-05-28T22:45:59+00:00", id_=6),
        ]
        out = derive_trigger_round_trips(rows)
        assert len(out) == 3
        buckets = {r["ticker"]: r["bucket"] for r in out}
        assert buckets == {
            "AMD": "HARD_SL",
            "MU": "HARD_TP",
            "NVDA": "DISCRETIONARY",
        }


# ───────────────────────── _verdict ─────────────────────────


def _empty_buckets():
    return {b: {"n": 0, "total_pl_usd": 0.0} for b in
            ("HARD_SL", "HARD_TP", "DISCRETIONARY")}


class TestVerdict:
    def test_emerging_when_under_min(self):
        b = _empty_buckets()
        b["DISCRETIONARY"]["n"] = 3
        b["DISCRETIONARY"]["total_pl_usd"] = 50.0
        assert _verdict(b, min_for_verdict=4) == "EMERGING"

    def test_both_positive(self):
        b = _empty_buckets()
        b["DISCRETIONARY"]["n"] = 3
        b["DISCRETIONARY"]["total_pl_usd"] = 100.0
        b["HARD_TP"]["n"] = 2
        b["HARD_TP"]["total_pl_usd"] = 50.0
        # 5 total >= 4 min — verdict not withheld.
        assert _verdict(b, min_for_verdict=4) == "BOTH_POSITIVE"

    def test_both_negative(self):
        b = _empty_buckets()
        b["DISCRETIONARY"]["n"] = 3
        b["DISCRETIONARY"]["total_pl_usd"] = -100.0
        b["HARD_SL"]["n"] = 2
        b["HARD_SL"]["total_pl_usd"] = -50.0
        assert _verdict(b, min_for_verdict=4) == "BOTH_NEGATIVE"

    def test_mechanical_dominant(self):
        b = _empty_buckets()
        b["DISCRETIONARY"]["n"] = 2
        b["DISCRETIONARY"]["total_pl_usd"] = -20.0
        b["HARD_TP"]["n"] = 2
        b["HARD_TP"]["total_pl_usd"] = 80.0
        b["HARD_SL"]["n"] = 1
        b["HARD_SL"]["total_pl_usd"] = -10.0
        # mech = 80 - 10 = 70 > disc = -20 → mechanical dominant.
        assert _verdict(b, min_for_verdict=4) == "MECHANICAL_DOMINANT"

    def test_discretionary_dominant(self):
        b = _empty_buckets()
        b["DISCRETIONARY"]["n"] = 3
        b["DISCRETIONARY"]["total_pl_usd"] = 200.0
        b["HARD_SL"]["n"] = 2
        b["HARD_SL"]["total_pl_usd"] = -50.0
        # mech = -50, disc = 200 → disc dominant. Note: both > 0 path
        # requires disc > 0 AND mech > 0; here mech < 0 so it falls
        # through to the DISCRETIONARY_DOMINANT comparison.
        assert _verdict(b, min_for_verdict=4) == "DISCRETIONARY_DOMINANT"


# ───────────────────────── build (full envelope) ─────────────────────────


class TestBuild:
    def test_empty_returns_emerging(self):
        out = build([])
        assert out["verdict"] == "EMERGING"
        assert out["n_round_trips"] == 0
        assert out["buckets"]["HARD_SL"]["n"] == 0
        assert out["buckets"]["HARD_TP"]["n"] == 0
        assert out["buckets"]["DISCRETIONARY"]["n"] == 0
        assert "No closed round-trips" in out["headline"]

    def test_full_ledger_three_buckets(self):
        # 2 HARD_SL losers, 1 HARD_TP winner, 2 discretionary wins.
        # min_for_verdict=4 → verdict resolved.
        rows = [
            _t(action="BUY", ticker="AMD", qty=1, price=520,
               timestamp="2026-05-28T18:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=505,
               reason="HARD_SL: price 505 <= threshold 510",
               timestamp="2026-05-29T17:25:00+00:00", id_=2),
            _t(action="BUY", ticker="DRAM", qty=2, price=25,
               timestamp="2026-05-19T17:13:00+00:00", id_=3),
            _t(action="SELL", ticker="DRAM", qty=2, price=24,
               reason="HARD_SL: price 24 <= threshold 24.5",
               timestamp="2026-05-19T18:21:00+00:00", id_=4),
            _t(action="BUY", ticker="MU", qty=1, price=900,
               timestamp="2026-05-26T18:00:00+00:00", id_=5),
            _t(action="SELL", ticker="MU", qty=1, price=928,
               reason="HARD_TP: price 928 >= threshold 923",
               timestamp="2026-05-27T20:09:00+00:00", id_=6),
            _t(action="BUY", ticker="MU", qty=1, price=830,
               timestamp="2026-05-26T11:00:00+00:00", id_=7),
            _t(action="SELL", ticker="MU", qty=1, price=890,
               reason="Taking profit on the breakout; capital rotation",
               timestamp="2026-05-26T16:00:00+00:00", id_=8),
            _t(action="BUY", ticker="NVDA", qty=3, price=213,
               timestamp="2026-05-28T17:25:00+00:00", id_=9),
            _t(action="SELL", ticker="NVDA", qty=3, price=214,
               reason="Thesis weakening; sector rotation",
               timestamp="2026-05-28T22:45:00+00:00", id_=10),
        ]
        out = build(rows)
        assert out["n_round_trips"] == 5
        b = out["buckets"]
        assert b["HARD_SL"]["n"] == 2
        assert b["HARD_SL"]["losses"] == 2
        assert b["HARD_SL"]["wins"] == 0
        assert b["HARD_SL"]["win_rate_pct"] == 0.0
        assert b["HARD_TP"]["n"] == 1
        assert b["HARD_TP"]["wins"] == 1
        assert b["DISCRETIONARY"]["n"] == 2
        assert b["DISCRETIONARY"]["wins"] == 2
        assert b["DISCRETIONARY"]["win_rate_pct"] == 100.0
        # Totals — independently calculated.
        # AMD: -15, DRAM: -2 → SL = -17.
        # MU tp: +28 → TP = 28.
        # MU disc: +60, NVDA disc: +3 → DISC = 63.
        assert b["HARD_SL"]["total_pl_usd"] == -17.0
        assert b["HARD_TP"]["total_pl_usd"] == 28.0
        assert b["DISCRETIONARY"]["total_pl_usd"] == 63.0
        # Verdict: 5 trips >= 4 min; disc(+63) and tp(+28) both positive,
        # sl(-17) negative. mech = 28-17 = 11 > 0 AND disc = 63 > 0 →
        # BOTH_POSITIVE.
        assert out["verdict"] == "BOTH_POSITIVE"

    def test_partial_sell_does_not_close_until_flat(self):
        # BUY 2 + SELL 1 doesn't close; the trailing SELL 1 does.
        rows = [
            _t(action="BUY", ticker="NVDA", qty=2, price=100,
               timestamp="2026-05-28T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=110,
               reason="Half-out",
               timestamp="2026-05-28T13:00:00+00:00", id_=2),
            _t(action="SELL", ticker="NVDA", qty=1, price=120,
               reason="Closing the lot",
               timestamp="2026-05-28T14:00:00+00:00", id_=3),
        ]
        out = build(rows)
        assert out["n_round_trips"] == 1
        # The closing SELL is the last one; its reason → DISCRETIONARY.
        only = out["buckets"]["DISCRETIONARY"]
        assert only["n"] == 1
        # cost = 200; proceeds = 110+120 = 230; realized = 30.
        assert only["total_pl_usd"] == 30.0


# ───────────────────────── Endpoint envelope (Flask test client) ─────────────────────────


class TestEndpoint:
    def test_route_returns_envelope(self):
        # Verify the endpoint is wired, the envelope shape matches the
        # ERROR branch contract, and the JSON is well-formed. We don't
        # care about specific bucket counts — that's tested above.
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/exit-trigger-pl-mix")
        # Either 200 (live DB has rows) or 500 (ERROR envelope, e.g.
        # corrupt DB). Both must be JSON.
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert body is not None
        # Required envelope fields present in both branches.
        assert "verdict" in body
        assert "headline" in body
        assert "service" in body and body["service"] == "paper_trader"
        if resp.status_code == 200:
            assert "buckets" in body
            assert set(body["buckets"].keys()) >= {
                "HARD_SL", "HARD_TP", "DISCRETIONARY",
            }
            # Per-bucket shape.
            for b in ("HARD_SL", "HARD_TP", "DISCRETIONARY"):
                stat = body["buckets"][b]
                for key in ("n", "wins", "losses", "win_rate_pct",
                            "total_pl_usd", "avg_hold_days",
                            "median_hold_days"):
                    assert key in stat
            assert "n_round_trips" in body
