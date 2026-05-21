"""Tests for paper_trader.analytics.regime_leverage_fit_skill.

Pins:
* the BLIND_LEVERING × DANGEROUS_HEADWIND × ALIGNED × MISSED_TAILWIND
  × DEFENSIVE × NEUTRAL × NO_DATA verdict matrix
* priority ladder (BLIND_LEVERING fires before DANGEROUS_HEADWIND
  on bear+high-lev+high-flow)
* regime classifier boundaries (mom > bull_mom_pct, mom < bear_mom_pct,
  exact-boundary tie-break)
* trade-flow window enforcement (older trades dropped, future trades
  dropped)
* threshold override forwarding (custom bull_mom_pct rewrites verdict)
* envelope key stability across every verdict
* defensive: malformed positions, NaN values, garbage tickers, missing
  fields all degrade — never raise
* leveraged set parity with strategy._LEVERAGED_ETFS_LIVE (drift guard
  so a future addition to the strategy set fails this test)

Also adds a Flask-test-client route smoke for the dashboard layer.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.regime_leverage_fit_skill import (
    _LEVERAGED_ETFS,
    DEFAULT_ALIGNED_LEV_FLOOR,
    DEFAULT_BEAR_MOM_PCT,
    DEFAULT_BULL_MOM_PCT,
    DEFAULT_FLOW_WINDOW_HOURS,
    DEFAULT_HIGH_FLOW_PCT,
    DEFAULT_HIGH_LEV_FLOOR,
    DEFAULT_LOW_LEV_CEIL,
    build_regime_leverage_fit_skill,
    classify_regime,
    is_leveraged,
)


def _now():
    return datetime(2026, 5, 21, 6, 30, 0, tzinfo=timezone.utc)


def _pos(ticker, market_value):
    return {"ticker": ticker, "market_value": market_value}


def _trade(ticker, action, ts_dt, notional=None, *, qty=None, price=None,
           use_ts_key=False):
    out = {"ticker": ticker, "action": action}
    if use_ts_key:
        out["ts"] = ts_dt.isoformat()
    else:
        out["timestamp"] = ts_dt.isoformat()
    if notional is not None:
        out["notional"] = notional
    if qty is not None:
        out["qty"] = qty
    if price is not None:
        out["price"] = price
    return out


_ENVELOPE_KEYS = {
    "verdict", "headline", "as_of", "regime", "spy_mom_20d",
    "portfolio", "recent_flow", "thresholds",
}


class TestEnvelopeStability:
    def test_no_data_empty_everything(self):
        out = build_regime_leverage_fit_skill(
            None, None, None, None, None, now=_now(),
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "NO_DATA"
        # Even on NO_DATA the nested dicts are populated, never None.
        assert isinstance(out["portfolio"], dict)
        assert isinstance(out["recent_flow"], dict)
        assert isinstance(out["thresholds"], dict)

    def test_envelope_keys_under_active_regime(self):
        out = build_regime_leverage_fit_skill(
            [_pos("TQQQ", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["regime"] == "bull"
        assert out["portfolio"]["leveraged_pct"] == 40.0
        assert out["portfolio"]["n_leveraged_positions"] == 1


class TestRegimeClassifier:
    def test_bull(self):
        assert classify_regime(5.0) == "bull"

    def test_bear(self):
        assert classify_regime(-5.0) == "bear"

    def test_sideways(self):
        assert classify_regime(0.0) == "sideways"
        assert classify_regime(2.999) == "sideways"

    def test_boundary_exact_bull_threshold_is_sideways(self):
        # > bull_mom_pct, so equal is sideways.
        assert classify_regime(DEFAULT_BULL_MOM_PCT) == "sideways"

    def test_boundary_just_above_bull_is_bull(self):
        assert classify_regime(DEFAULT_BULL_MOM_PCT + 0.0001) == "bull"

    def test_unknown_on_none(self):
        assert classify_regime(None) == "unknown"

    def test_unknown_on_nan(self):
        assert classify_regime(float("nan")) == "unknown"

    def test_unknown_on_string(self):
        assert classify_regime("5.0") == "unknown"

    def test_custom_thresholds(self):
        assert classify_regime(1.5, bull_mom_pct=1.0, bear_mom_pct=-1.0) == "bull"


class TestIsLeveraged:
    def test_known_leveraged(self):
        assert is_leveraged("SOXL")
        assert is_leveraged("TQQQ")
        assert is_leveraged("SQQQ")  # inverse counts

    def test_known_unleveraged(self):
        assert not is_leveraged("NVDA")
        assert not is_leveraged("AAPL")

    def test_case_insensitive(self):
        assert is_leveraged("tqqq")
        assert is_leveraged("Soxl")

    def test_empty_or_none(self):
        assert not is_leveraged("")
        assert not is_leveraged(None)  # type: ignore[arg-type]


class TestVerdictMatrix:
    def test_aligned_bull_lev(self):
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 300.0), _pos("TQQQ", 200.0)],
            cash_usd=500.0, total_value_usd=1000.0,
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "ALIGNED"
        assert out["portfolio"]["leveraged_pct"] == 50.0

    def test_dangerous_headwind_bear_high_lev(self):
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "DANGEROUS_HEADWIND"

    def test_missed_tailwind_bull_low_lev(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "MISSED_TAILWIND"
        assert out["portfolio"]["leveraged_pct"] == 0.0

    def test_defensive_bear_low_lev(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "DEFENSIVE"

    def test_blind_levering_flow_overrides_static(self):
        # Bear regime + currently low-lev exposure but recent BUY flow into lev.
        # BLIND_LEVERING must fire because direction of change > static state.
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],  # 10% — would be DEFENSIVE on static
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() - timedelta(hours=2), notional=80.0),
            ],
            now=_now(),
        )
        assert out["verdict"] == "BLIND_LEVERING"
        assert out["recent_flow"]["n_leveraged_buys"] == 1
        assert out["recent_flow"]["buy_flow_pct"] == 8.0

    def test_blind_levering_priority_over_dangerous_headwind(self):
        # Bear + lev_pct already HIGH + recent flow — both could fire;
        # BLIND_LEVERING wins (priority documented in the docstring).
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 400.0)],  # 40% lev — DANGEROUS_HEADWIND candidate
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() - timedelta(hours=1), notional=100.0),
            ],
            now=_now(),
        )
        assert out["verdict"] == "BLIND_LEVERING"

    def test_neutral_mid_band(self):
        # bull regime, lev_pct between low_lev_ceil and aligned_lev_floor
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 150.0)],  # 15% — between ceil(10) and aligned(20)
            cash_usd=850.0, total_value_usd=1000.0,
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "NEUTRAL"

    def test_no_data_unknown_regime_empty_book(self):
        out = build_regime_leverage_fit_skill(
            [], cash_usd=0.0, total_value_usd=0.0,
            spy_mom_20d=None, recent_trades=[], now=_now(),
        )
        assert out["verdict"] == "NO_DATA"

    def test_unknown_regime_with_book_collapses_to_neutral(self):
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=None, recent_trades=[], now=_now(),
        )
        # Regime unknown but book is non-empty — withhold verdict, not NO_DATA.
        assert out["verdict"] == "NEUTRAL"


class TestFlowWindowEnforcement:
    def test_old_trade_dropped(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            # Trade beyond default 24h window — should not contribute.
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() - timedelta(hours=48), notional=80.0),
            ],
            now=_now(),
        )
        assert out["verdict"] == "DEFENSIVE"
        assert out["recent_flow"]["n_leveraged_buys"] == 0
        assert out["recent_flow"]["buy_flow_pct"] == 0.0

    def test_future_trade_dropped(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() + timedelta(hours=1), notional=80.0),
            ],
            now=_now(),
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 0

    def test_custom_window_extends(self):
        # Same 48h trade now in-window because flow_window_hours bumped.
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() - timedelta(hours=48), notional=80.0),
            ],
            now=_now(),
            flow_window_hours=72.0,
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 1
        assert out["verdict"] == "BLIND_LEVERING"

    def test_non_leveraged_trade_does_not_count(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("NVDA", "BUY",
                       _now() - timedelta(hours=1), notional=200.0),
            ],
            now=_now(),
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 0

    def test_sell_does_not_count_as_buy_flow(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("SOXL", "SELL",
                       _now() - timedelta(hours=1), notional=200.0),
            ],
            now=_now(),
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 0
        assert out["recent_flow"]["n_leveraged_sells"] == 1

    def test_ts_key_accepted(self):
        # Tests use `timestamp`, but builder accepts both keys.
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[
                _trade("TQQQ", "BUY",
                       _now() - timedelta(hours=1),
                       notional=80.0, use_ts_key=True),
            ],
            now=_now(),
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 1

    def test_value_key_used_as_notional_fallback(self):
        trade = {
            "ticker": "TQQQ", "action": "BUY",
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "value": 80.0,
        }
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[trade],
            now=_now(),
        )
        assert out["recent_flow"]["leveraged_buy_usd"] == 80.0
        assert out["recent_flow"]["buy_flow_pct"] == 8.0


class TestDefensiveDegradation:
    def test_malformed_position_degrades_value_to_zero(self):
        out = build_regime_leverage_fit_skill(
            [
                "not-a-dict",  # type: ignore[list-item]
                {"ticker": None, "market_value": 100.0},
                {"ticker": "SOXL", "market_value": "garbage"},
                {"ticker": "TQQQ", "market_value": 200.0},
            ],
            cash_usd=800.0, total_value_usd=1000.0,
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        # SOXL has a valid ticker but garbage mv → contributes 0; the
        # position is still reported (real position with a parsing
        # error is more honest than silent-drop). Only TQQQ's $200
        # contributes to leveraged_usd → 20% of $1000.
        assert out["portfolio"]["n_leveraged_positions"] == 2
        assert out["portfolio"]["leveraged_pct"] == 20.0
        assert out["portfolio"]["leveraged_usd"] == 200.0

    def test_nan_total_value_falls_back_to_cash_plus_open(self):
        out = build_regime_leverage_fit_skill(
            [_pos("SOXL", 200.0)],
            cash_usd=800.0, total_value_usd=float("nan"),
            spy_mom_20d=5.0, recent_trades=[], now=_now(),
        )
        # tv falls back to cash + open = 1000; 200/1000 = 20%
        assert out["portfolio"]["total_value_usd"] == 1000.0
        assert out["portfolio"]["leveraged_pct"] == 20.0

    def test_garbage_trade_skipped_no_raise(self):
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=5.0,
            recent_trades=[
                "not-a-dict",  # type: ignore[list-item]
                {"action": "BUY", "ticker": None},
                {"action": "INVALID_ACTION", "ticker": "TQQQ",
                 "timestamp": _now().isoformat()},
                {"action": "BUY", "ticker": "TQQQ", "timestamp": None},
            ],
            now=_now(),
        )
        assert out["recent_flow"]["n_leveraged_buys"] == 0

    def test_qty_price_fallback_when_no_notional(self):
        trade = {
            "ticker": "TQQQ", "action": "BUY",
            "timestamp": (_now() - timedelta(hours=1)).isoformat(),
            "qty": 4.0, "price": 25.0,
        }
        out = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0,
            recent_trades=[trade],
            now=_now(),
        )
        assert out["recent_flow"]["leveraged_buy_usd"] == 100.0
        assert out["verdict"] == "BLIND_LEVERING"


class TestThresholdOverrides:
    def test_custom_bull_threshold_flips_regime(self):
        # spy_mom_20d=4 — with default bull_mom_pct=3 -> bull;
        # with custom 5 -> sideways.
        out_default = build_regime_leverage_fit_skill(
            [_pos("SOXL", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=4.0, recent_trades=[], now=_now(),
        )
        assert out_default["regime"] == "bull"
        assert out_default["verdict"] == "ALIGNED"

        out_strict = build_regime_leverage_fit_skill(
            [_pos("SOXL", 400.0)],
            cash_usd=600.0, total_value_usd=1000.0,
            spy_mom_20d=4.0, recent_trades=[], now=_now(),
            bull_mom_pct=5.0,
        )
        assert out_strict["regime"] == "sideways"
        # 40% lev in sideways with no flow + no headwind: not BLIND_LEVERING,
        # not DANGEROUS_HEADWIND (bear-only), not DEFENSIVE (lev too high),
        # not ALIGNED (bull-only) — NEUTRAL.
        assert out_strict["verdict"] == "NEUTRAL"

    def test_custom_high_flow_pct_makes_blind_levering_harder(self):
        trades = [
            _trade("TQQQ", "BUY", _now() - timedelta(hours=1),
                   notional=80.0),
        ]
        out_default = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0, recent_trades=trades, now=_now(),
        )
        assert out_default["verdict"] == "BLIND_LEVERING"  # 8% > 5%

        out_strict = build_regime_leverage_fit_skill(
            [_pos("NVDA", 100.0)],
            cash_usd=900.0, total_value_usd=1000.0,
            spy_mom_20d=-5.0, recent_trades=trades, now=_now(),
            high_flow_pct=10.0,
        )
        # 8% < 10% — flow gate not crossed, falls back to DEFENSIVE.
        assert out_strict["verdict"] == "DEFENSIVE"

    def test_thresholds_echoed_back(self):
        out = build_regime_leverage_fit_skill(
            [], cash_usd=0.0, total_value_usd=0.0,
            spy_mom_20d=None, recent_trades=[], now=_now(),
            bull_mom_pct=2.0, bear_mom_pct=-1.0,
            high_lev_floor=25.0, aligned_lev_floor=15.0, low_lev_ceil=5.0,
            flow_window_hours=12.0, high_flow_pct=3.0,
        )
        thr = out["thresholds"]
        assert thr["bull_mom_pct"] == 2.0
        assert thr["bear_mom_pct"] == -1.0
        assert thr["high_lev_floor"] == 25.0
        assert thr["aligned_lev_floor"] == 15.0
        assert thr["low_lev_ceil"] == 5.0
        assert thr["flow_window_hours"] == 12.0
        assert thr["high_flow_pct"] == 3.0


class TestStrategySetParity:
    """Drift guard: this module's _LEVERAGED_ETFS must be a superset
    of strategy._LEVERAGED_ETFS_LIVE. A future addition to the
    strategy set must surface here or this test fails. Subset (not
    equal) because this module additionally includes the leveraged-
    short inverses (SQQQ/SPXS/SOXS/TECS/FNGD) so the regime endpoint
    can detect a flip into them — and intentionally extends with
    additional 2x / single-name leveraged names that the strategy
    advisor's vocabulary doesn't need."""
    def test_strategy_set_is_subset(self):
        from paper_trader.strategy import _LEVERAGED_ETFS_LIVE
        missing = set(_LEVERAGED_ETFS_LIVE) - _LEVERAGED_ETFS
        assert not missing, (
            f"strategy._LEVERAGED_ETFS_LIVE has names not in this "
            f"module's _LEVERAGED_ETFS — drift: {sorted(missing)}"
        )


class TestRouteSmoke:
    """Flask test client smoke — the route layer must compose the
    builder correctly and degrade to a structured ERROR envelope
    on internal failure (the established `never raises` contract)."""

    def _client(self):
        # Memory note: __main__ smoke hits a different DB; use Flask
        # test client + the live app object so the actual store is
        # consulted.
        from paper_trader.dashboard import app
        return app.test_client()

    def test_route_returns_envelope(self):
        cl = self._client()
        r = cl.get("/api/regime-leverage-fit-skill")
        # Either 200 (data) or 500 (ERROR envelope) — both are
        # acceptable. What's NOT acceptable is a missing-key crash.
        assert r.status_code in (200, 500)
        body = r.get_json()
        assert body is not None
        assert "verdict" in body
        assert "headline" in body

    def test_route_with_query_params(self):
        cl = self._client()
        r = cl.get(
            "/api/regime-leverage-fit-skill"
            "?flow_window_hours=12&high_flow_pct=3&bull_mom_pct=2"
        )
        assert r.status_code in (200, 500)
        body = r.get_json()
        assert "thresholds" in body
        if r.status_code == 200:
            assert body["thresholds"]["flow_window_hours"] == 12.0
            assert body["thresholds"]["high_flow_pct"] == 3.0
            assert body["thresholds"]["bull_mom_pct"] == 2.0
