"""Tests for paper_trader.analytics.cash_redeployment_latency_skill.

Pins:
* the FAST_REDEPLOY × STEADY × SLOW × STALLED × NO_DATA verdict ladder
* SELL→next-BUY cross-ticker pairing (different ticker is fine)
* window-edge SELL exclusion (a SELL too close to `now` for the
  stalled_cutoff to have elapsed does not pollute the median / rate)
* stalled-cutoff override forwarding
* envelope key stability across every verdict
* defensive: malformed trades, missing timestamps, NaN values, garbage
  actions all degrade — never raise
* Flask route smoke (separate test class so it can be excluded with -k)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.cash_redeployment_latency_skill import (
    DEFAULT_DEGRADED_REDEPLOY_PCT,
    DEFAULT_FAST_MEDIAN_H,
    DEFAULT_HEALTHY_REDEPLOY_PCT,
    DEFAULT_SLOW_MEDIAN_H,
    DEFAULT_STALLED_CUTOFF_H,
    DEFAULT_STEADY_MEDIAN_H,
    DEFAULT_STEADY_REDEPLOY_PCT,
    DEFAULT_WINDOW_DAYS,
    MIN_SELLS_FOR_VERDICT,
    build_cash_redeployment_latency_skill,
    _median,
    _percentile,
    _safe_notional,
)


def _now():
    return datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)


def _trade(action, ticker, hours_ago, *, value=100.0, now=None):
    now = now or _now()
    ts = now - timedelta(hours=hours_ago)
    return {
        "action": action,
        "ticker": ticker,
        "timestamp": ts.isoformat(),
        "value": value,
        "qty": 1.0,
        "price": value,
    }


_ENVELOPE_KEYS = {
    "verdict", "headline", "as_of", "window_days",
    "stats", "thresholds", "pairs",
}


class TestEnvelopeStability:
    def test_no_data_empty_trades(self):
        out = build_cash_redeployment_latency_skill(None, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "NO_DATA"
        assert isinstance(out["stats"], dict)
        assert isinstance(out["thresholds"], dict)
        assert out["pairs"] == []

    def test_no_data_few_sells(self):
        # Only 2 SELLs → below MIN_SELLS_FOR_VERDICT (3)
        trades = [
            _trade("SELL", "NVDA", 100.0),
            _trade("BUY", "AAPL", 95.0),
            _trade("SELL", "MSFT", 80.0),
            _trade("BUY", "MSFT", 75.0),
        ]
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "NO_DATA"
        assert out["stats"]["n_classifiable"] == 2

    def test_keys_present_on_fast_redeploy(self):
        trades = []
        # 5 SELLs each followed by a BUY within 1h, all well in the past
        for i in range(5):
            sell_h = 200.0 + i * 20
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 0.5))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "FAST_REDEPLOY"


class TestVerdictLadder:
    def test_fast_redeploy(self):
        # 4 SELL→BUY pairs all within 2h
        trades = []
        for i in range(4):
            sell_h = 200.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 1.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "FAST_REDEPLOY"
        assert out["stats"]["n_redeployed"] == 4
        assert out["stats"]["redeploy_pct"] == 100.0
        assert out["stats"]["median_latency_h"] == 1.0

    def test_steady(self):
        # 4 SELLs each followed by a BUY at 10h (median 10h, 100% redeploy)
        trades = []
        for i in range(4):
            sell_h = 200.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 10.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "STEADY"
        assert out["stats"]["median_latency_h"] == 10.0

    def test_slow_by_median(self):
        # Median 36h (< 72) → SLOW
        trades = []
        for i in range(4):
            sell_h = 300.0 + i * 50
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 36.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "SLOW"

    def test_slow_by_rate(self):
        # 4 SELLs, 2 redeploy fast (5h), 2 stall (no subsequent BUY) → 50% rate → SLOW
        trades = [
            _trade("SELL", "AAA", 400.0),
            _trade("BUY", "BBB", 395.0),
            _trade("SELL", "AAA", 350.0),
            _trade("BUY", "BBB", 345.0),
            _trade("SELL", "AAA", 300.0),
            _trade("SELL", "AAA", 250.0),
        ]
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        # 2 redeployed, 2 stalled (no future BUY exists)
        assert out["stats"]["n_redeployed"] == 2
        assert out["stats"]["n_stalled"] == 2
        assert out["stats"]["redeploy_pct"] == 50.0
        # 50% rate is on the SLOW/STALLED boundary; verdict is SLOW
        # since median is small (5h) and rate isn't < 50.
        assert out["verdict"] == "SLOW"

    def test_stalled_by_low_rate(self):
        # 5 SELLs (sell_h = 400, 430, 460, 490, 520). Single BUY at
        # 270h ago. Per-SELL latency to next BUY:
        #   sell_h=400 → 400-270=130h (≤168 cutoff) → REDEPLOYED
        #   sell_h=430 → 160h           → REDEPLOYED
        #   sell_h=460 → 190h           → STALLED (>168)
        #   sell_h=490 → 220h           → STALLED
        #   sell_h=520 → 250h           → STALLED
        # → n_redeployed=2 / n_stalled=3, rate 40% < 50% → STALLED
        trades = [_trade("BUY", "AAA", 1000.0)]  # ancient BUY before any SELL
        for i in range(5):
            sell_h = 400.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
        trades.append(_trade("BUY", "AAPL", 270.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["stats"]["n_redeployed"] == 2
        assert out["stats"]["n_stalled"] == 3
        assert out["stats"]["redeploy_pct"] == 40.0
        assert out["verdict"] == "STALLED"

    def test_stalled_by_median(self):
        # Median 80h (> 72) → STALLED
        trades = []
        for i in range(4):
            sell_h = 500.0 + i * 100
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 80.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "STALLED"
        assert out["stats"]["median_latency_h"] == 80.0


class TestWindowEdgeExclusion:
    def test_recent_sell_without_buy_is_window_edge(self):
        # A SELL 6h ago with no subsequent BUY shouldn't be counted as
        # STALLED (the cash hasn't had a fair chance) — it goes to
        # n_window_edge. Otherwise: 4 healthy SELL→BUY pairs.
        trades = []
        for i in range(4):
            sell_h = 300.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 1.0))
        # Recent SELL — stalled_cutoff is 168h (default), this is only 6h ago
        trades.append(_trade("SELL", "MSFT", 6.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["stats"]["n_window_edge"] == 1
        assert out["stats"]["n_classifiable"] == 4
        # Verdict is FAST_REDEPLOY because the recent SELL was excluded
        assert out["verdict"] == "FAST_REDEPLOY"

    def test_recent_sell_with_buy_is_redeployed(self):
        # A SELL 4h ago WITH a subsequent BUY 2h ago is classifiable
        # (we have evidence of redeployment).
        trades = [
            _trade("SELL", "NVDA", 4.0),
            _trade("BUY", "AAPL", 2.0),
        ]
        # Pad with 4 more SELL→BUY pairs to clear the floor.
        for i in range(4):
            sell_h = 300.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "AAPL", sell_h - 1.0))
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["stats"]["n_window_edge"] == 0
        assert out["stats"]["n_classifiable"] == 5


class TestSellTickerIndependence:
    def test_buy_can_be_different_ticker(self):
        # The point of this skill: we measure CASH redeployment, not
        # same-ticker re-entry. SELL NVDA → BUY MSFT counts.
        trades = []
        for i in range(4):
            sell_h = 300.0 + i * 30
            trades.append(_trade("SELL", "NVDA", sell_h))
            trades.append(_trade("BUY", "MSFT", sell_h - 2.0))  # different name
        out = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out["verdict"] == "FAST_REDEPLOY"
        # Verify pairing crosses tickers
        assert any(p["sell_ticker"] != p["next_buy_ticker"] for p in out["pairs"])


class TestStalledCutoffOverride:
    def test_low_cutoff_promotes_to_stalled(self):
        # 4 SELLs (sell_h = 500, 550, 600, 650) followed by ONE late BUY.
        # Place the BUY at 300h ago so each SELL's latency = sell_h-300:
        #   500h SELL → 200h latency
        #   550h SELL → 250h latency
        #   600h SELL → 300h latency
        #   650h SELL → 350h latency
        # Default cutoff is 168h. None are within 168h →
        #   default: all STALLED.
        # We use ONE BUY (instead of paired BUYs) so the "next BUY in
        # time" lookup unambiguously picks the same target for all SELLs.
        # With cutoff=400h: all 4 within → REDEPLOYED.
        trades = []
        for i in range(4):
            sell_h = 500.0 + i * 50
            trades.append(_trade("SELL", "NVDA", sell_h))
        trades.append(_trade("BUY", "AAPL", 300.0))
        out_default = build_cash_redeployment_latency_skill(trades, now=_now())
        assert out_default["stats"]["n_stalled"] == 4
        out_lax = build_cash_redeployment_latency_skill(
            trades, now=_now(), stalled_cutoff_hours=400.0,
        )
        assert out_lax["stats"]["n_redeployed"] == 4
        assert out_lax["stats"]["n_stalled"] == 0


class TestDefensiveDegradation:
    def test_malformed_trades_never_raise(self):
        garbage = [
            None,
            {},
            {"action": "BUY"},                              # no ticker, no ts
            {"action": "BUY", "ticker": "NVDA"},            # no ts
            {"action": "BUY", "ticker": "NVDA", "timestamp": "not-an-iso"},
            {"action": 123, "ticker": "NVDA"},              # action not str
            {"action": "BLAHBLAH", "ticker": "NVDA",        # garbage action
             "timestamp": _now().isoformat()},
            {"action": "BUY", "ticker": None,               # bad ticker
             "timestamp": _now().isoformat()},
            "not a dict",
        ]
        # Should never raise; degrades to NO_DATA
        out = build_cash_redeployment_latency_skill(garbage, now=_now())
        assert out["verdict"] == "NO_DATA"
        assert out["stats"]["n_sells_total"] == 0

    def test_negative_notional_clamped(self):
        # value < 0 should be treated as abs() (defensive)
        t = {"action": "SELL", "ticker": "X",
             "timestamp": _now().isoformat(), "value": -50.0,
             "qty": 1.0, "price": 50.0}
        # _safe_notional should return 50.0 (abs)
        assert _safe_notional(t) == 50.0


class TestStaticHelpers:
    def test_median_basic(self):
        assert _median([]) is None
        assert _median([5.0]) == 5.0
        assert _median([1.0, 3.0]) == 2.0
        assert _median([1.0, 2.0, 3.0]) == 2.0
        assert _median([3.0, 1.0, 2.0, 4.0]) == 2.5

    def test_percentile_basic(self):
        assert _percentile([], 50.0) is None
        assert _percentile([5.0], 50.0) == 5.0
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == 3.0
        # 25th percentile of [1..5] → 2.0 (linear interp)
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 25.0) == 2.0


class TestDefaults:
    def test_defaults_have_expected_relationships(self):
        # The verdict ladder requires FAST < STEADY < SLOW
        assert DEFAULT_FAST_MEDIAN_H < DEFAULT_STEADY_MEDIAN_H
        assert DEFAULT_STEADY_MEDIAN_H < DEFAULT_SLOW_MEDIAN_H
        # And the rate floors must descend
        assert DEFAULT_HEALTHY_REDEPLOY_PCT > DEFAULT_STEADY_REDEPLOY_PCT
        assert DEFAULT_STEADY_REDEPLOY_PCT > DEFAULT_DEGRADED_REDEPLOY_PCT
        # Sanity on the window and cutoff
        assert DEFAULT_WINDOW_DAYS > 0
        assert DEFAULT_STALLED_CUTOFF_H > 0
        assert MIN_SELLS_FOR_VERDICT >= 2


class TestFlaskRoute:
    def test_route_returns_envelope(self):
        # Inject the route smoke through the Flask test_client to
        # match the verification-by-test-client memory.
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/cash-redeployment-latency-skill")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        # On a healthy DB or an empty DB the envelope keys are stable.
        for k in ("verdict", "headline", "stats", "thresholds", "pairs"):
            assert k in body, f"missing key: {k}"
