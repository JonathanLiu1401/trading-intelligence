"""Tests for paper_trader.analytics.closed_market_fill_skill.

Pins:
* the SESSION_ALIGNED × BALANCED × AFTER_HOURS_HEAVY × OVERNIGHT_DOMINATED
  × INSUFFICIENT_DATA ladder
* injectable ``is_open_fn`` so the test suite doesn't depend on the
  live NYSE calendar (mirrors the live ``market.is_market_open``)
* per-ticker breakdown sort + completeness
* closed_subbucket classification (weekend / overnight / holiday)
* envelope key stability across every verdict
* defensive: malformed trades / NaN / inf / wrong types degrade — never raise
* Flask route smoke
* live-data parity: the 2026-05-21 NVDA overnight cluster is detected
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.closed_market_fill_skill import (
    DEFAULT_AFTER_HOURS_PCT,
    DEFAULT_ALIGNED_PCT,
    DEFAULT_DOMINATED_PCT,
    DEFAULT_WINDOW_DAYS,
    MIN_FILLS_FOR_VERDICT,
    _closed_subbucket,
    _normalize_trade,
    build_closed_market_fill_skill,
)


def _now() -> datetime:
    return datetime(2026, 5, 23, 18, 0, 0, tzinfo=timezone.utc)


def _trade(action: str, ticker: str, hours_ago: float, price: float = 100.0,
           qty: float = 1.0, now=None) -> dict:
    now = now or _now()
    ts = now - timedelta(hours=hours_ago)
    return {
        "action": action,
        "ticker": ticker,
        "timestamp": ts.isoformat(),
        "price": price,
        "qty": qty,
    }


def _open_fn_always_true(_ts):
    return True


def _open_fn_always_false(_ts):
    return False


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "window_days",
    "thresholds", "stats", "per_ticker", "closed_subbucket",
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class TestNormalizeTrade:
    def test_basic(self):
        n = _normalize_trade(_trade("BUY", "NVDA", 1.0, 223.435))
        assert n is not None
        assert n["ticker"] == "NVDA" and n["action"] == "BUY"
        assert n["notional"] == 223.435

    def test_hold_dropped(self):
        assert _normalize_trade(_trade("HOLD", "NVDA", 1.0)) is None

    def test_negative_qty_uses_abs(self):
        n = _normalize_trade(_trade("BUY", "NVDA", 1.0, qty=-3.0))
        assert n is not None and n["qty"] == 3.0


class TestClosedSubbucket:
    def test_saturday_is_weekend(self):
        # 2026-05-23 is a Saturday.
        ts = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        assert _closed_subbucket(ts) == "weekend"

    def test_sunday_is_weekend(self):
        ts = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
        assert _closed_subbucket(ts) == "weekend"

    def test_weekday_overnight_classified(self):
        # Wednesday 03:00 UTC ≈ 23:00 NY Tuesday → overnight.
        # Use a clearly-weekday wall-time in NY.
        ts = datetime(2026, 5, 20, 3, 0, 0, tzinfo=timezone.utc)
        assert _closed_subbucket(ts) == "overnight"

    def test_known_holiday_classified(self):
        # 2026 NYSE holiday: Christmas Day (2026-12-25, a Friday).
        ts = datetime(2026, 12, 25, 15, 0, 0, tzinfo=timezone.utc)
        # When the date is in the holiday set, classify as holiday rather
        # than weekend/overnight. Test only fires if the live market
        # module's holiday set contains 2026-12-25.
        from paper_trader.market import NYSE_HOLIDAYS_2026
        if datetime(2026, 12, 25).date() in NYSE_HOLIDAYS_2026:
            assert _closed_subbucket(ts) == "holiday"


# ─────────────────────────────────────────────────────────────────────
# Envelope + verdict ladder
# ─────────────────────────────────────────────────────────────────────


class TestEnvelopeStability:
    def test_no_data_envelope(self):
        out = build_closed_market_fill_skill(None, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["per_ticker"] == []
        assert out["closed_subbucket"] == {"overnight": 0, "weekend": 0, "holiday": 0}

    def test_aligned_envelope(self):
        trades = [_trade("BUY", "NVDA", 10.0 - i) for i in range(6)]
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_fn_always_true,
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "SESSION_ALIGNED"
        assert out["stats"]["closed_pct"] == 0.0


class TestVerdictLadder:
    def test_insufficient_data_below_floor(self):
        trades = [_trade("BUY", "NVDA", 1.0 + i) for i in range(4)]  # 4 < 5
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_fn_always_true,
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_session_aligned_all_open(self):
        trades = [_trade("BUY", "NVDA", 10.0 - i) for i in range(10)]
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_fn_always_true,
        )
        assert out["verdict"] == "SESSION_ALIGNED"
        assert out["stats"]["n_closed"] == 0

    def test_balanced(self):
        # 3 closed + 7 open = 30% closed. Default aligned=25, after=50
        # ⇒ BALANCED.
        trades_closed = [_trade("BUY", "NVDA", 10.0 - i) for i in range(3)]
        trades_open = [_trade("BUY", "AAPL", 50.0 + i) for i in range(7)]

        # `_open_for_aapl(ts) is True` ⇔ ts older than 30h ago ⇒ "open".
        # closed trades sit at hours_ago=8..10 (well inside 30h, closed).
        # open trades sit at hours_ago=50..56 (well past 30h, open).
        def _open_for_aapl(ts):
            return ts < _now() - timedelta(hours=30)
        out = build_closed_market_fill_skill(
            trades_closed + trades_open, now=_now(),
            is_open_fn=_open_for_aapl,
        )
        assert out["verdict"] == "BALANCED"
        assert out["stats"]["n_closed"] == 3
        assert out["stats"]["n_open"] == 7
        assert out["stats"]["closed_pct"] == 30.0

    def test_after_hours_heavy(self):
        # 6 closed + 4 open = 60%. Default after_hours=50, dominated=75
        # ⇒ AFTER_HOURS_HEAVY.
        trades_closed = [_trade("BUY", "NVDA", 10.0 - i) for i in range(6)]
        trades_open = [_trade("BUY", "AAPL", 50.0 + i) for i in range(4)]

        def _open_for_aapl(ts):
            return ts < _now() - timedelta(hours=30)
        out = build_closed_market_fill_skill(
            trades_closed + trades_open, now=_now(),
            is_open_fn=_open_for_aapl,
        )
        assert out["verdict"] == "AFTER_HOURS_HEAVY"
        assert out["stats"]["closed_pct"] == 60.0

    def test_overnight_dominated(self):
        # 9 closed + 1 open = 90% closed ⇒ DOMINATED.
        trades_closed = [_trade("BUY", "NVDA", 10.0 - i * 0.1) for i in range(9)]
        trades_open = [_trade("BUY", "AAPL", 50.0)]

        def _open_for_aapl(ts):
            return ts < _now() - timedelta(hours=30)
        out = build_closed_market_fill_skill(
            trades_closed + trades_open, now=_now(),
            is_open_fn=_open_for_aapl,
        )
        assert out["verdict"] == "OVERNIGHT_DOMINATED"
        assert out["stats"]["closed_pct"] == 90.0


class TestPerTickerBreakdown:
    def test_per_ticker_worst_first(self):
        # NVDA 4/5 closed (80%), AAPL 1/5 closed (20%).
        trades = []
        for i in range(4):
            trades.append(_trade("BUY", "NVDA", 5.0 - i * 0.1))
        trades.append(_trade("BUY", "NVDA", 50.0))   # old ⇒ open
        for i in range(1):
            trades.append(_trade("BUY", "AAPL", 5.0 - i * 0.1))
        for i in range(4):
            trades.append(_trade("BUY", "AAPL", 50.0 + i))

        def _open_if_old(ts):
            return ts < _now() - timedelta(hours=30)
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_if_old,
        )
        assert len(out["per_ticker"]) == 2
        assert out["per_ticker"][0]["ticker"] == "NVDA"
        assert out["per_ticker"][0]["closed_pct"] >= out["per_ticker"][1]["closed_pct"]


class TestThresholdOverrides:
    def test_lower_dominated_promotes(self):
        # 60% closed. Default DOMINATED=75 ⇒ AFTER_HOURS_HEAVY.
        # Force dominated=55 ⇒ OVERNIGHT_DOMINATED.
        trades_closed = [_trade("BUY", "NVDA", 5.0 - i * 0.1) for i in range(6)]
        trades_open = [_trade("BUY", "AAPL", 50.0 + i) for i in range(4)]

        def _open_for_aapl(ts):
            return ts < _now() - timedelta(hours=30)
        out = build_closed_market_fill_skill(
            trades_closed + trades_open, now=_now(),
            is_open_fn=_open_for_aapl,
            dominated_pct=55.0,
        )
        assert out["verdict"] == "OVERNIGHT_DOMINATED"

    def test_scrambled_thresholds_dont_raise(self):
        # after_hours below aligned should auto-widen.
        trades = [_trade("BUY", "NVDA", 5.0 - i * 0.1) for i in range(10)]
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_fn_always_false,
            aligned_pct=80.0, after_hours_pct=10.0, dominated_pct=5.0,
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS


# ─────────────────────────────────────────────────────────────────────
# Defensive degradation
# ─────────────────────────────────────────────────────────────────────


class TestDefensiveDegradation:
    def test_malformed_trades_dropped(self):
        trades = [
            None, {}, {"action": "BUY"},
            {"action": "BUY", "ticker": "NVDA"},  # no price/ts
            {"action": "BUY", "ticker": "NVDA",
             "price": float("nan"), "qty": 1.0,
             "timestamp": _now().isoformat()},
            "garbage string",
            _trade("HOLD", "NVDA", 1.0),
        ]
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_open_fn_always_true,
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_is_open_fn_raises_treats_as_closed(self):
        def _boom(_ts):
            raise RuntimeError("nyse calendar exploded")
        trades = [_trade("BUY", "NVDA", 5.0 - i * 0.1) for i in range(10)]
        out = build_closed_market_fill_skill(
            trades, now=_now(), is_open_fn=_boom,
        )
        # Defensive: when is_open_fn raises, classify as CLOSED — the
        # conservative bucket — and continue. No exception escapes.
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["stats"]["n_closed"] == 10
        assert out["verdict"] == "OVERNIGHT_DOMINATED"


# ─────────────────────────────────────────────────────────────────────
# Live-data parity
# ─────────────────────────────────────────────────────────────────────


class TestLiveParityNvdaCluster:
    """Replays the 2026-05-21 NVDA cluster verbatim using a NY-time-aware
    is_open_fn that matches the documented session window. Asserts the
    builder flags it OVERNIGHT_DOMINATED (4/5 of these fired in the
    overnight bucket, the documented pathology)."""

    LIVE_NVDA_TRADES = [
        ("BUY",  "2026-05-20T21:10:10.495989+00:00", 1.0),
        ("BUY",  "2026-05-21T00:11:08.699815+00:00", 0.5),
        ("SELL", "2026-05-21T01:13:38.360935+00:00", 4.5),
        ("BUY",  "2026-05-21T01:36:06.684121+00:00", 2.0),
        ("BUY",  "2026-05-21T10:00:54.069646+00:00", 1.0),
    ]

    def test_overnight_dominated(self):
        # Use the LIVE is_market_open — these timestamps are NY-time
        # outside 09:30-16:00 on 2026-05-20/21 (Wed/Thu), so all 5
        # land in the CLOSED bucket. Verdict: OVERNIGHT_DOMINATED.
        from paper_trader.market import is_market_open
        trades = [
            {
                "action": a, "ticker": "NVDA", "timestamp": ts,
                "price": 223.435, "qty": q,
            }
            for a, ts, q in self.LIVE_NVDA_TRADES
        ]
        out = build_closed_market_fill_skill(
            trades,
            now=datetime(2026, 5, 23, 18, 0, 0, tzinfo=timezone.utc),
            is_open_fn=is_market_open,
        )
        assert out["verdict"] == "OVERNIGHT_DOMINATED"
        assert out["stats"]["n_closed"] == 5
        assert out["stats"]["n_open"] == 0
        # All 5 are weekday overnight (Wed/Thu late evening NY time).
        assert out["closed_subbucket"]["overnight"] == 5


# ─────────────────────────────────────────────────────────────────────
# Flask route
# ─────────────────────────────────────────────────────────────────────


class TestFlaskRoute:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/closed-market-fill-skill")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        for k in ("verdict", "headline", "stats", "thresholds",
                  "per_ticker", "closed_subbucket"):
            assert k in body, f"missing key: {k}"

    def test_route_clamps_invalid_params(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get(
            "/api/closed-market-fill-skill"
            "?window_days=garbage&aligned_pct=oops&dominated_pct=banana"
        )
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert isinstance(body, dict)
        assert "thresholds" in body
        # Defaults must have applied (not crashed).
        assert body["thresholds"]["aligned_pct"] == DEFAULT_ALIGNED_PCT
        assert body["thresholds"]["dominated_pct"] == DEFAULT_DOMINATED_PCT
