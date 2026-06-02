"""Tests for analytics.sector_pl_history + /api/sector-pl-history.

Pins the time-windowed sector rollup against hand-built trade slices.
Verifies the verdict ladder, the per-window sort order, and the endpoint
envelope shape via the Flask test client.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.sector_pl_history import build


def _t(
    *,
    action: str,
    ticker: str,
    qty: float = 1.0,
    price: float = 100.0,
    timestamp: str,
    id_: int = 0,
) -> dict:
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


_NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


class TestBuildEmpty:
    def test_empty_trades(self):
        out = build([], now=_NOW)
        assert out["verdict"] == "NO_CLOSED_TRIPS"
        assert out["windows"]["last_7d"]["n_round_trips"] == 0
        assert out["windows"]["last_30d"]["n_round_trips"] == 0
        assert out["windows"]["all_time"]["n_round_trips"] == 0
        assert "No closed round-trips" in out["headline"]


class TestSingleSector:
    def test_one_sector_only(self):
        # MU + NVDA are both ``semis`` — single sector even with two
        # tickers.
        rows = [
            _t(action="BUY", ticker="NVDA", qty=1, price=200,
               timestamp="2026-05-28T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=210,
               timestamp="2026-05-28T15:00:00+00:00", id_=2),
            _t(action="BUY", ticker="MU", qty=1, price=900,
               timestamp="2026-05-29T12:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=920,
               timestamp="2026-05-29T15:00:00+00:00", id_=4),
        ]
        out = build(rows, now=_NOW)
        assert out["verdict"] == "SINGLE_SECTOR"
        all_time = out["windows"]["all_time"]
        assert len(all_time["sectors"]) == 1
        assert all_time["sectors"][0]["sector"] == "semis"
        assert all_time["n_round_trips"] == 2
        # +10 (NVDA) + +20 (MU) = +30.
        assert all_time["total_pl_usd"] == 30.0
        assert all_time["sectors"][0]["total_pl_usd"] == 30.0


class TestMixedSectors:
    def _rows(self):
        # Three sectors: semis (NVDA), broad (SPY), tech (AAPL).
        # NVDA wins +50, AAPL loses -30, SPY breaks even (+0).
        return [
            _t(action="BUY", ticker="NVDA", qty=1, price=200,
               timestamp="2026-05-28T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=250,
               timestamp="2026-05-28T15:00:00+00:00", id_=2),
            _t(action="BUY", ticker="AAPL", qty=1, price=180,
               timestamp="2026-05-29T12:00:00+00:00", id_=3),
            _t(action="SELL", ticker="AAPL", qty=1, price=150,
               timestamp="2026-05-29T15:00:00+00:00", id_=4),
            _t(action="BUY", ticker="SPY", qty=1, price=500,
               timestamp="2026-05-27T12:00:00+00:00", id_=5),
            _t(action="SELL", ticker="SPY", qty=1, price=500,
               timestamp="2026-05-27T15:00:00+00:00", id_=6),
        ]

    def test_leader_and_laggard_sort(self):
        out = build(self._rows(), now=_NOW)
        assert out["verdict"] == "MIXED"
        all_time = out["windows"]["all_time"]
        secs = all_time["sectors"]
        # Three sectors total. Sorted descending by total_pl_usd.
        sectors_by_name = {s["sector"]: s for s in secs}
        assert set(sectors_by_name) == {"semis", "broad", "tech"}
        # NVDA winner first, AAPL loser last.
        assert secs[0]["sector"] == "semis"
        assert secs[0]["total_pl_usd"] == 50.0
        assert secs[-1]["sector"] == "tech"
        assert secs[-1]["total_pl_usd"] == -30.0
        # Net = +50 - 30 + 0 = +20.
        assert all_time["total_pl_usd"] == 20.0
        # Headline names leader + laggard.
        assert "semis" in out["headline"]
        assert "tech" in out["headline"]

    def test_win_rate_per_sector(self):
        out = build(self._rows(), now=_NOW)
        sectors = {s["sector"]: s
                   for s in out["windows"]["all_time"]["sectors"]}
        # semis has 1 win, 0 losses → 100% (decided=1).
        assert sectors["semis"]["win_rate_pct"] == 100.0
        # tech has 0 wins, 1 loss → 0% (decided=1).
        assert sectors["tech"]["win_rate_pct"] == 0.0
        # broad has 1 flat trip → decided=0 → None.
        assert sectors["broad"]["win_rate_pct"] is None


class TestWindowFiltering:
    def test_7d_excludes_old_trips(self):
        # 1 trip 3d ago (in last_7d), 1 trip 20d ago (in last_30d only),
        # 1 trip 90d ago (all_time only).
        rows = [
            _t(action="BUY", ticker="NVDA", qty=1, price=200,
               timestamp="2026-05-27T12:00:00+00:00", id_=1),  # 3d ago
            _t(action="SELL", ticker="NVDA", qty=1, price=210,
               timestamp="2026-05-27T15:00:00+00:00", id_=2),
            _t(action="BUY", ticker="AAPL", qty=1, price=180,
               timestamp="2026-05-10T12:00:00+00:00", id_=3),  # 20d ago
            _t(action="SELL", ticker="AAPL", qty=1, price=200,
               timestamp="2026-05-10T15:00:00+00:00", id_=4),
            _t(action="BUY", ticker="SPY", qty=1, price=500,
               timestamp="2026-03-01T12:00:00+00:00", id_=5),  # 90d ago
            _t(action="SELL", ticker="SPY", qty=1, price=505,
               timestamp="2026-03-01T15:00:00+00:00", id_=6),
        ]
        out = build(rows, now=_NOW)
        # last_7d: only NVDA (+10).
        d7 = out["windows"]["last_7d"]
        assert d7["n_round_trips"] == 1
        assert d7["total_pl_usd"] == 10.0
        assert len(d7["sectors"]) == 1
        assert d7["sectors"][0]["sector"] == "semis"
        # last_30d: NVDA + AAPL (= +10 + +20 = +30).
        d30 = out["windows"]["last_30d"]
        assert d30["n_round_trips"] == 2
        assert d30["total_pl_usd"] == 30.0
        # all_time: all three (+10 + 20 + 5 = +35).
        dall = out["windows"]["all_time"]
        assert dall["n_round_trips"] == 3
        assert dall["total_pl_usd"] == 35.0


class TestUnknownTickerFallsBackToOther:
    def test_random_ticker_classified_as_other(self):
        # ``XYZZY`` is not in SECTOR_MAP → falls through to ``"other"``.
        rows = [
            _t(action="BUY", ticker="XYZZY", qty=1, price=100,
               timestamp="2026-05-28T12:00:00+00:00", id_=1),
            _t(action="SELL", ticker="XYZZY", qty=1, price=110,
               timestamp="2026-05-28T15:00:00+00:00", id_=2),
        ]
        out = build(rows, now=_NOW)
        assert out["verdict"] == "SINGLE_SECTOR"
        sec = out["windows"]["all_time"]["sectors"][0]
        assert sec["sector"] == "other"
        assert sec["total_pl_usd"] == 10.0


class TestEnvelopeShape:
    def test_shape_keys(self):
        out = build([], now=_NOW)
        assert set(out.keys()) >= {
            "verdict", "headline", "windows", "as_of",
        }
        for label in ("last_7d", "last_30d", "all_time"):
            w = out["windows"][label]
            assert set(w.keys()) == {
                "sectors", "n_round_trips", "total_pl_usd",
            }


# ───────────────────────── Endpoint envelope (Flask test client) ─────────────────────────


class TestEndpoint:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/sector-pl-history")
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert body is not None
        assert "verdict" in body
        assert "headline" in body
        assert body["service"] == "paper_trader"
        if resp.status_code == 200:
            assert "windows" in body
            for label in ("last_7d", "last_30d", "all_time"):
                assert label in body["windows"]
                assert "sectors" in body["windows"][label]
                assert "n_round_trips" in body["windows"][label]
                assert "total_pl_usd" in body["windows"][label]
