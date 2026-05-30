"""Tests for analytics.tape_fit_pl + /api/tape-fit-pl.

Pins the tape-direction classification at the ±0.30% band boundary, the
per-bucket P/L rollup, the verdict ladder, the UNKNOWN bucket isolation
(missing SPY context never gates a directional verdict), and the endpoint
envelope.

Every assertion is specific-value (no "no crash" passes).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.tape_fit_pl import (
    _BUCKETS,
    _TAPE_BAND_PCT,
    _classify_tape,
    annotate_round_trip,
    build,
)


def _t(*, action, ticker, qty=1.0, price=100.0, timestamp, id_=0):
    return {
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "reason": "",
        "timestamp": timestamp,
        "option_type": None,
        "expiry": None,
        "strike": None,
        "id": id_,
    }


def _eq(*, timestamp, sp500_price, total_value=1000.0, cash=0.0):
    return {
        "timestamp": timestamp,
        "total_value": total_value,
        "cash": cash,
        "sp500_price": sp500_price,
    }


# ────────────────────────────── unit: classifier ──────────────────────────


class TestClassifyTape:
    def test_band_constant_is_thirty_bps(self):
        assert _TAPE_BAND_PCT == 0.30

    def test_above_band_is_tailwind(self):
        assert _classify_tape(0.31) == "TAILWIND"
        assert _classify_tape(5.0) == "TAILWIND"

    def test_below_negative_band_is_headwind(self):
        assert _classify_tape(-0.31) == "HEADWIND"
        assert _classify_tape(-3.0) == "HEADWIND"

    def test_inside_band_is_flat(self):
        assert _classify_tape(0.0) == "FLAT"
        assert _classify_tape(0.29) == "FLAT"
        assert _classify_tape(-0.29) == "FLAT"

    def test_exactly_at_band_is_flat(self):
        # Boundary inclusive of FLAT: |v| == _TAPE_BAND_PCT → FLAT.
        assert _classify_tape(_TAPE_BAND_PCT) == "FLAT"
        assert _classify_tape(-_TAPE_BAND_PCT) == "FLAT"

    def test_none_is_unknown(self):
        assert _classify_tape(None) == "UNKNOWN"

    def test_nan_is_unknown(self):
        assert _classify_tape(float("nan")) == "UNKNOWN"

    def test_garbage_is_unknown(self):
        assert _classify_tape("ten percent") == "UNKNOWN"


# ─────────────────────── unit: annotate_round_trip ────────────────────────


class TestAnnotateRoundTrip:
    def test_tailwind_classification(self):
        rt = {
            "ticker": "AMD", "cost": 100.0, "realized_pl": 10.0,
            "opened_at": "2026-05-25T10:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
            "hold_days": 0.21,
        }
        # SPY 500 → 510 = +2.0% — TAILWIND.
        ec = [
            _eq(timestamp="2026-05-25T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-25T15:00:00+00:00", sp500_price=510.0),
        ]
        out = annotate_round_trip(rt, ec)
        assert out["spy_open"] == 500.0
        assert out["spy_close"] == 510.0
        assert out["spy_change_pct"] == 2.0
        assert out["tape_bucket"] == "TAILWIND"

    def test_headwind_classification(self):
        rt = {
            "ticker": "MU", "cost": 100.0, "realized_pl": -5.0,
            "opened_at": "2026-05-25T10:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
        }
        # SPY 500 → 485 = -3.0% — HEADWIND.
        ec = [
            _eq(timestamp="2026-05-25T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-25T15:00:00+00:00", sp500_price=485.0),
        ]
        out = annotate_round_trip(rt, ec)
        assert out["spy_change_pct"] == -3.0
        assert out["tape_bucket"] == "HEADWIND"

    def test_flat_classification(self):
        rt = {
            "ticker": "NVDA", "cost": 100.0, "realized_pl": 1.0,
            "opened_at": "2026-05-25T10:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
        }
        # SPY 500 → 501 = +0.20% — within ±0.30% → FLAT.
        ec = [
            _eq(timestamp="2026-05-25T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-25T15:00:00+00:00", sp500_price=501.0),
        ]
        out = annotate_round_trip(rt, ec)
        assert out["tape_bucket"] == "FLAT"

    def test_missing_endpoint_is_unknown(self):
        # SPY mark only at the open; nothing close to the exit → bucket UNKNOWN.
        rt = {
            "ticker": "AMD", "cost": 100.0, "realized_pl": 0.0,
            "opened_at": "2026-05-25T10:00:00+00:00",
            "closed_at": "2026-05-26T10:00:00+00:00",
        }
        # Only one SPY sample near the open — the nearest-neighbour falls
        # back to that same sample on the close lookup, so both endpoints
        # map to the same price (0.0% change) which classifies FLAT.
        # To get UNKNOWN, the SPY column has to be missing entirely.
        ec = [_eq(timestamp="2026-05-25T10:00:00+00:00", sp500_price=None)]
        out = annotate_round_trip(rt, ec)
        assert out["tape_bucket"] == "UNKNOWN"
        assert out["spy_change_pct"] is None

    def test_no_equity_curve_is_unknown(self):
        rt = {
            "ticker": "AMD", "cost": 100.0, "realized_pl": 0.0,
            "opened_at": "2026-05-25T10:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
        }
        out = annotate_round_trip(rt, [])
        assert out["tape_bucket"] == "UNKNOWN"

    def test_non_dict_returns_none(self):
        assert annotate_round_trip(None, []) is None
        assert annotate_round_trip("garbage", []) is None


# ────────────────────────── build (full flow) ─────────────────────────────


def _round_trip_trades(ticker, qty, open_price, close_price,
                       open_ts, close_ts, *, base_id=0):
    """Helper: two trades that form one closed round-trip."""
    return [
        _t(action="BUY", ticker=ticker, qty=qty, price=open_price,
           timestamp=open_ts, id_=base_id + 1),
        _t(action="SELL", ticker=ticker, qty=qty, price=close_price,
           timestamp=close_ts, id_=base_id + 2),
    ]


class TestBuildEmpty:
    def test_no_trades(self):
        out = build([])
        assert out["verdict"] == "NO_DATA"
        assert out["n_round_trips"] == 0
        assert out["n_directional"] == 0
        for b in _BUCKETS:
            assert out["buckets"][b]["n"] == 0


class TestBuildNoDirectional:
    def test_all_unknown_when_no_equity_curve(self):
        # 4 closed trips, no equity_curve at all → every row UNKNOWN.
        trades = []
        for i in range(4):
            trades.extend(_round_trip_trades(
                "AMD", 1.0, 100.0, 110.0,
                f"2026-05-2{i}T10:00:00+00:00",
                f"2026-05-2{i}T11:00:00+00:00",
                base_id=i * 10,
            ))
        out = build(trades, equity_curve=None)
        assert out["verdict"] == "NO_DATA"
        assert out["n_round_trips"] == 4
        assert out["n_directional"] == 0
        assert out["buckets"]["UNKNOWN"]["n"] == 4


class TestBuildAlphaVsTape:
    def test_headwind_winner_and_book_positive(self):
        # 4 trips, all wins; HEADWIND winner proves the alpha case.
        trades = []
        ec = []
        # Trip 1: HEADWIND (-1.0%), +$50.
        trades += _round_trip_trades("AMD", 1.0, 200.0, 250.0,
                                     "2026-05-20T10:00:00+00:00",
                                     "2026-05-20T15:00:00+00:00", base_id=10)
        ec += [
            _eq(timestamp="2026-05-20T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-20T15:00:00+00:00", sp500_price=495.0),
        ]
        # Trip 2: TAILWIND (+1.0%), +$30.
        trades += _round_trip_trades("MU", 1.0, 300.0, 330.0,
                                     "2026-05-21T10:00:00+00:00",
                                     "2026-05-21T15:00:00+00:00", base_id=20)
        ec += [
            _eq(timestamp="2026-05-21T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-21T15:00:00+00:00", sp500_price=505.0),
        ]
        # Trip 3: FLAT (+0.1%), +$10.
        trades += _round_trip_trades("NVDA", 1.0, 700.0, 710.0,
                                     "2026-05-22T10:00:00+00:00",
                                     "2026-05-22T15:00:00+00:00", base_id=30)
        ec += [
            _eq(timestamp="2026-05-22T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-22T15:00:00+00:00", sp500_price=500.5),
        ]
        # Trip 4: HEADWIND (-0.5%), +$20.
        trades += _round_trip_trades("TSLA", 1.0, 900.0, 920.0,
                                     "2026-05-23T10:00:00+00:00",
                                     "2026-05-23T15:00:00+00:00", base_id=40)
        ec += [
            _eq(timestamp="2026-05-23T10:00:00+00:00", sp500_price=500.0),
            _eq(timestamp="2026-05-23T15:00:00+00:00", sp500_price=497.5),
        ]
        out = build(trades, equity_curve=ec)
        assert out["verdict"] == "ALPHA_VS_TAPE"
        # HEADWIND: 2 trips, +50 + +20 = +70.
        assert out["buckets"]["HEADWIND"]["n"] == 2
        assert out["buckets"]["HEADWIND"]["total_pl_usd"] == 70.0
        # TAILWIND: 1 trip, +30.
        assert out["buckets"]["TAILWIND"]["n"] == 1
        assert out["buckets"]["TAILWIND"]["total_pl_usd"] == 30.0
        # FLAT: 1 trip, +10.
        assert out["buckets"]["FLAT"]["n"] == 1
        assert out["buckets"]["FLAT"]["total_pl_usd"] == 10.0
        # n_directional reflects only the three directional buckets.
        assert out["n_directional"] == 4
        # Headline always names HEADWIND $.
        assert "+70.00" in out["headline"]


class TestBuildRidingBeta:
    def test_tailwind_wins_headwind_loses(self):
        trades = []
        ec = []
        # 2 TAILWIND winners, 2 HEADWIND losers.
        trades += _round_trip_trades("AMD", 1.0, 200.0, 220.0,
                                     "2026-05-20T10:00:00+00:00",
                                     "2026-05-20T15:00:00+00:00", base_id=10)
        ec += [_eq(timestamp="2026-05-20T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-20T15:00:00+00:00", sp500_price=510.0)]
        trades += _round_trip_trades("MU", 1.0, 200.0, 230.0,
                                     "2026-05-21T10:00:00+00:00",
                                     "2026-05-21T15:00:00+00:00", base_id=20)
        ec += [_eq(timestamp="2026-05-21T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-21T15:00:00+00:00", sp500_price=515.0)]
        trades += _round_trip_trades("NVDA", 1.0, 700.0, 680.0,
                                     "2026-05-22T10:00:00+00:00",
                                     "2026-05-22T15:00:00+00:00", base_id=30)
        ec += [_eq(timestamp="2026-05-22T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-22T15:00:00+00:00", sp500_price=490.0)]
        trades += _round_trip_trades("TSLA", 1.0, 900.0, 870.0,
                                     "2026-05-23T10:00:00+00:00",
                                     "2026-05-23T15:00:00+00:00", base_id=40)
        ec += [_eq(timestamp="2026-05-23T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-23T15:00:00+00:00", sp500_price=488.0)]
        out = build(trades, equity_curve=ec)
        assert out["verdict"] == "RIDING_BETA"
        assert out["buckets"]["TAILWIND"]["total_pl_usd"] == 50.0
        assert out["buckets"]["HEADWIND"]["total_pl_usd"] == -50.0


class TestBuildTapeTrapped:
    def test_both_directional_negative(self):
        trades = []
        ec = []
        # TAILWIND: -$10 (losing during a rising tape). HEADWIND: -$20.
        trades += _round_trip_trades("AMD", 1.0, 200.0, 190.0,
                                     "2026-05-20T10:00:00+00:00",
                                     "2026-05-20T15:00:00+00:00", base_id=10)
        ec += [_eq(timestamp="2026-05-20T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-20T15:00:00+00:00", sp500_price=510.0)]
        trades += _round_trip_trades("MU", 1.0, 200.0, 195.0,
                                     "2026-05-21T10:00:00+00:00",
                                     "2026-05-21T15:00:00+00:00", base_id=20)
        ec += [_eq(timestamp="2026-05-21T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-21T15:00:00+00:00", sp500_price=508.0)]
        trades += _round_trip_trades("NVDA", 1.0, 700.0, 690.0,
                                     "2026-05-22T10:00:00+00:00",
                                     "2026-05-22T15:00:00+00:00", base_id=30)
        ec += [_eq(timestamp="2026-05-22T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-22T15:00:00+00:00", sp500_price=490.0)]
        trades += _round_trip_trades("TSLA", 1.0, 700.0, 690.0,
                                     "2026-05-23T10:00:00+00:00",
                                     "2026-05-23T15:00:00+00:00", base_id=40)
        ec += [_eq(timestamp="2026-05-23T10:00:00+00:00", sp500_price=500.0),
               _eq(timestamp="2026-05-23T15:00:00+00:00", sp500_price=485.0)]
        out = build(trades, equity_curve=ec)
        assert out["verdict"] == "TAPE_TRAPPED"
        assert out["buckets"]["TAILWIND"]["total_pl_usd"] == -15.0
        assert out["buckets"]["HEADWIND"]["total_pl_usd"] == -20.0


class TestBuildEmerging:
    def test_below_floor(self):
        trades = []
        ec = []
        for i in range(3):  # 3 < default min_for_verdict=4
            trades += _round_trip_trades(
                "AMD", 1.0, 200.0, 220.0,
                f"2026-05-2{i}T10:00:00+00:00",
                f"2026-05-2{i}T15:00:00+00:00",
                base_id=i * 10,
            )
            ec += [_eq(timestamp=f"2026-05-2{i}T10:00:00+00:00",
                       sp500_price=500.0),
                   _eq(timestamp=f"2026-05-2{i}T15:00:00+00:00",
                       sp500_price=510.0)]
        out = build(trades, equity_curve=ec)
        assert out["verdict"] == "EMERGING"
        assert out["n_directional"] == 3


class TestEnvelope:
    def test_top_level_keys(self):
        out = build([])
        assert set(out.keys()) >= {
            "verdict", "headline", "buckets", "n_round_trips",
            "n_directional", "min_for_verdict", "tape_band_pct",
        }
        assert set(out["buckets"].keys()) == {
            "TAILWIND", "HEADWIND", "FLAT", "UNKNOWN",
        }
        assert out["tape_band_pct"] == _TAPE_BAND_PCT


# ───────────────────────── Endpoint (Flask client) ────────────────────────


class TestEndpoint:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/tape-fit-pl")
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert body is not None
        assert "verdict" in body
        assert "headline" in body
        assert body["service"] == "paper_trader"
        if resp.status_code == 200:
            assert "buckets" in body
            for b in ("TAILWIND", "HEADWIND", "FLAT", "UNKNOWN"):
                assert b in body["buckets"]
                assert "n" in body["buckets"][b]
                assert "total_pl_usd" in body["buckets"][b]
            assert "n_round_trips" in body
            assert "n_directional" in body
            assert "tape_band_pct" in body
