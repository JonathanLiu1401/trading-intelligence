"""Tests for analytics/peer_earnings_shock.py — indirect 1σ exposure on
held leveraged ETFs from upcoming constituent (peer) earnings.

Hand-computed arithmetic + SSOT-composition invariants. The module
fuses ``etf_lookthrough`` ($-indirect-per-(ETF, underlying)) with
``event_calendar`` (which underlyings have imminent earnings).
Any drift from the SSOT chain (a recomputed indirect_usd, a σ band
that disagrees with the earnings_shock convention, a verdict emitted
before the sample gate, an aggregate that includes INSUFFICIENT rows)
fails an assertion here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.peer_earnings_shock import (
    DEFAULT_HORIZON_DAYS,
    MODERATE_BOOK_PCT,
    SEVERE_BOOK_PCT,
    build_peer_earnings_shock,
)


# A test-seam ETF table — TQQQ holds 9% NVDA at 3x leverage, plus a
# couple of other names; SOXL holds 10% NVDA + 7% AMD at 3x; SMH (1x)
# at 21% NVDA. Matches the real ``etf_lookthrough._ETF_LOOKTHROUGH``
# shape so we keep tests faithful to production but stable to a
# weight-update edit.
_TEST_MAP = {
    "TQQQ": {
        "leverage": 3.0,
        "holdings": [("NVDA", 9.0), ("MSFT", 9.0), ("AAPL", 8.5)],
    },
    "SOXL": {
        "leverage": 3.0,
        "holdings": [("NVDA", 10.0), ("AMD", 7.0)],
    },
    "SMH": {
        "leverage": 1.0,
        "holdings": [("NVDA", 21.0), ("AVGO", 9.0)],
    },
    "SQQQ": {
        "leverage": -3.0,
        "holdings": [("NVDA", 9.0), ("MSFT", 9.0)],
    },
}


def _snap(positions, cash=100.0, total_value=None):
    if total_value is None:
        total_value = cash + sum(
            (p.get("current_price") or p.get("avg_cost") or 0)
            * (p.get("qty") or 0) for p in positions
        )
    return {"cash": cash, "total_value": float(total_value),
            "positions": positions}


def _pos(ticker, qty, price, type_="stock"):
    return {"ticker": ticker, "qty": qty, "avg_cost": price,
            "current_price": price, "type": type_,
            "option_type": None, "strike": None, "expiry": None}


def _ec(events):
    """Minimal event_calendar report shape — only the keys the builder reads."""
    return {"events": events}


def _event(ticker, days_away, in_portfolio=False, tier="WATCH"):
    return {"ticker": ticker, "days_away": days_away,
            "in_portfolio": in_portfolio, "tier": tier,
            "earnings_date": "2026-05-21T16:00:00+00:00"}


# ─────────────────────────── empty-state ladder ─────────────────────────

class TestStateLadder:
    def test_no_data_on_empty_total_value(self):
        rep = build_peer_earnings_shock(
            {"total_value": 0, "positions": []}, _ec([]),
            lambda _t: 5.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] == "NO_DATA"
        assert rep["etf_rows"] == []
        assert rep["total_indirect_sigma_dollar"] is None

    def test_no_etf_held_when_only_direct_names(self):
        # Book holds NVDA directly only, no leveraged ETF.
        snap = _snap([_pos("NVDA", 1, 200.0)], total_value=300.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "NO_ETF_HELD"
        assert rep["verdict"] == "NO_ETF_HELD"
        assert rep["n_etfs_at_risk"] == 0

    def test_no_peer_events_when_calendar_empty(self):
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=200.0)
        rep = build_peer_earnings_shock(
            snap, _ec([]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "NO_PEER_EVENTS"
        assert rep["verdict"] == "NO_PEER_EVENTS"
        assert rep["n_peer_events"] == 0

    def test_no_peer_events_when_calendar_misses_constituents(self):
        # Calendar has earnings for a name TQQQ does NOT hold.
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=200.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("XYZ", 1.0)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        # peer_set has 1 entry but no overlap → no etf_rows → state
        # collapses to NO_PEER_EVENTS.
        assert rep["state"] == "NO_PEER_EVENTS"
        assert rep["etf_rows"] == []


# ───────────────────────── core arithmetic ──────────────────────────────

class TestArithmetic:
    def test_tqqq_nvda_indirect_sigma(self):
        # TQQQ position $148, NVDA 9% weight, leverage 3x →
        # indirect_usd = 148 * 3 * 0.09 = $39.96
        # σ 7% → indirect_sigma_dollar = 39.96 * 0.07 = $2.7972
        # On a $500 book: book_pct = 2.7972 / 500 * 100 = 0.5594%
        snap = _snap(
            [_pos("TQQQ", 1, 148.0)],
            total_value=500.0,
        )
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            sigma_provider=lambda tk: 7.0 if tk == "NVDA" else None,
            lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "OK"
        assert rep["n_etfs_at_risk"] == 1
        assert rep["n_peer_events"] == 1

        tqqq_row = rep["etf_rows"][0]
        assert tqqq_row["etf_ticker"] == "TQQQ"
        nvda = next(u for u in tqqq_row["underlyings"]
                    if u["underlying"] == "NVDA")
        # indirect_usd = 148 * 3 * 0.09 = 39.96
        assert nvda["indirect_usd"] == 39.96
        # sigma_dollar = 39.96 * 0.07 = 2.7972 → round 2 = 2.80
        assert nvda["indirect_sigma_dollar"] == 2.80
        # book_pct = 2.80 / 500 * 100 = 0.56 (rounded to 4dp)
        # exact: 2.7972 / 500 * 100 = 0.55944 → 0.5594
        assert nvda["indirect_sigma_book_pct"] == 0.5594
        # Aggregate matches (single etf, single underlying).
        assert rep["total_indirect_sigma_dollar"] == 2.80
        # Verdict LOW (0.56% << MODERATE 2.0%).
        assert rep["verdict"] == "LOW"

    def test_two_etfs_two_underlyings_aggregate(self):
        # TQQQ $148 → NVDA 9%, MSFT 9%; SOXL $80 → NVDA 10%, AMD 7%.
        # NVDA σ 7%, MSFT σ 4%, AMD σ 6%.
        # TQQQ_NVDA indirect_usd = 148 * 3 * 0.09 = 39.96; σ$ = 2.7972
        # TQQQ_MSFT indirect_usd = 148 * 3 * 0.09 = 39.96; σ$ = 1.5984
        # SOXL_NVDA indirect_usd = 80 * 3 * 0.10 = 24.00;   σ$ = 1.680
        # SOXL_AMD  indirect_usd = 80 * 3 * 0.07 = 16.80;   σ$ = 1.008
        # total = 2.7972 + 1.5984 + 1.680 + 1.008 = 7.0836
        # On a $1000 book, total_book_pct = 0.7084%
        snap = _snap(
            [_pos("TQQQ", 1, 148.0), _pos("SOXL", 1, 80.0)],
            total_value=1000.0,
        )
        rep = build_peer_earnings_shock(
            snap,
            _ec([_event("NVDA", 0.5), _event("MSFT", 2.0),
                 _event("AMD", 3.0)]),
            sigma_provider=lambda tk: {
                "NVDA": 7.0, "MSFT": 4.0, "AMD": 6.0,
            }.get(tk),
            lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "OK"
        assert rep["n_etfs_at_risk"] == 2
        assert rep["n_peer_events"] == 3
        # Float-tolerant equality (2-decimal rounding).
        assert abs(rep["total_indirect_sigma_dollar"] - 7.08) <= 0.01

    def test_negative_leverage_inverse_etf(self):
        # SQQQ -3x: indirect_usd is NEGATIVE for long-NVDA positions
        # (inverse ETF shorts the underlying). |σ$| still surfaces in
        # absolute terms because earnings σ is a magnitude — but the
        # signed indirect_usd flows through arithmetic.
        # SQQQ $100, NVDA 9%, leverage -3x → indirect_usd = -27.00
        # σ 7% → indirect_sigma_dollar = -27.00 * 0.07 = -1.89
        snap = _snap(
            [_pos("SQQQ", 1, 100.0)],
            total_value=500.0,
        )
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            sigma_provider=lambda tk: 7.0 if tk == "NVDA" else None,
            lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "OK"
        nvda = next(u for u in rep["etf_rows"][0]["underlyings"]
                    if u["underlying"] == "NVDA")
        assert nvda["indirect_usd"] == -27.00
        assert nvda["indirect_sigma_dollar"] == -1.89
        # Aggregate uses |sum| so an inverse ETF still surfaces risk
        # (a 7% NVDA move down means SQQQ goes UP 21%, still a real
        # P&L move — the magnitude IS the operator's $-at-risk).
        assert rep["total_indirect_sigma_dollar"] == 1.89

    def test_insufficient_sigma_excluded_from_aggregate(self):
        # Sigma provider returns None for some underlyings.
        snap = _snap(
            [_pos("TQQQ", 1, 100.0)],
            total_value=500.0,
        )
        rep = build_peer_earnings_shock(
            snap,
            _ec([_event("NVDA", 0.5), _event("MSFT", 2.0)]),
            sigma_provider=lambda tk: 7.0 if tk == "NVDA" else None,
            lookthrough_map=_TEST_MAP,
        )
        nvda = next(u for u in rep["etf_rows"][0]["underlyings"]
                    if u["underlying"] == "NVDA")
        msft = next(u for u in rep["etf_rows"][0]["underlyings"]
                    if u["underlying"] == "MSFT")
        assert nvda["row_state"] == "OK"
        assert msft["row_state"] == "INSUFFICIENT_SIGMA"
        assert msft["indirect_sigma_dollar"] is None
        # Aggregate only includes scored rows (NVDA only).
        # indirect_usd = 100 * 3 * 0.09 = 27.00; σ$ = 1.89
        assert rep["total_indirect_sigma_dollar"] == 1.89


# ───────────────────── horizon + filtering ──────────────────────────────

class TestHorizonAndFiltering:
    def test_events_beyond_horizon_dropped(self):
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=500.0)
        rep = build_peer_earnings_shock(
            snap,
            _ec([_event("NVDA", DEFAULT_HORIZON_DAYS + 1.0)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        # Beyond horizon → no peer events from the builder's view.
        assert rep["state"] == "NO_PEER_EVENTS"

    def test_custom_horizon_extends_window(self):
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=500.0)
        rep = build_peer_earnings_shock(
            snap,
            _ec([_event("NVDA", 10.0)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
            horizon_days=14.0,
        )
        assert rep["state"] == "OK"

    def test_past_event_dropped(self):
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=500.0)
        rep = build_peer_earnings_shock(
            snap,
            _ec([_event("NVDA", -1.0)]),  # already happened
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "NO_PEER_EVENTS"


# ───────────────────────── verdict band ─────────────────────────────────

class TestVerdictBand:
    def test_severe_when_total_book_pct_exceeds_threshold(self):
        # Engineer a position large enough to cross SEVERE.
        # TQQQ $5000 → NVDA 9% × 3x = $1350 indirect
        # σ 10% → $135 sigma_dollar. Book $1500.
        # book_pct = 135 / 1500 * 100 = 9.0% → SEVERE (>=5).
        snap = _snap([_pos("TQQQ", 50, 100.0)], total_value=1500.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            lambda _t: 10.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "OK"
        assert rep["verdict"] == "SEVERE"
        assert rep["total_indirect_sigma_book_pct"] >= SEVERE_BOOK_PCT

    def test_moderate_band(self):
        # Engineer book_pct in [2, 5).
        # TQQQ $300 → NVDA 9% × 3x = $81 indirect; σ 10% → $8.10 σ$.
        # Book $300 → book_pct = 2.7% → MODERATE.
        snap = _snap([_pos("TQQQ", 3, 100.0)], total_value=300.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            lambda _t: 10.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["verdict"] == "MODERATE"
        assert MODERATE_BOOK_PCT <= rep["total_indirect_sigma_book_pct"] < SEVERE_BOOK_PCT

    def test_low_band_default_live_shape(self):
        # The live live-book shape: TQQQ $148, total $1000.
        # 148 * 3 * 0.09 * 0.07 = 2.7972 → 0.28% book → LOW.
        snap = _snap([_pos("TQQQ", 1, 148.0)], total_value=1000.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["verdict"] == "LOW"


# ───────────────────────── never-raises ─────────────────────────────────

class TestNeverRaises:
    def test_sigma_provider_raises(self):
        def _boom(_tk):
            raise RuntimeError("yfinance died")
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=500.0)
        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            sigma_provider=_boom, lookthrough_map=_TEST_MAP,
        )
        # Provider exception → row reads INSUFFICIENT_SIGMA, builder
        # does NOT propagate.
        assert rep["state"] == "OK"
        nvda = rep["etf_rows"][0]["underlyings"][0]
        assert nvda["row_state"] == "INSUFFICIENT_SIGMA"

    def test_garbage_event_calendar(self):
        snap = _snap([_pos("TQQQ", 1, 100.0)], total_value=500.0)
        # Garbage events and missing keys must degrade gracefully.
        rep = build_peer_earnings_shock(
            snap, {"events": [None, {}, "string", {"ticker": "NVDA"}]},
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        # No usable events parsed → NO_PEER_EVENTS, no raise.
        assert rep["state"] == "NO_PEER_EVENTS"

    def test_none_snapshot_returns_no_data(self):
        rep = build_peer_earnings_shock(
            None, _ec([_event("NVDA", 0.5)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        assert rep["state"] == "NO_DATA"

    def test_response_shape_stable(self):
        rep = build_peer_earnings_shock(
            None, None, None, lookthrough_map=_TEST_MAP,
        )
        for k in ("as_of", "state", "headline", "horizon_days", "total_value",
                  "n_etfs_at_risk", "n_peer_events",
                  "total_indirect_sigma_dollar", "total_indirect_sigma_book_pct",
                  "verdict", "etf_rows"):
            assert k in rep, f"missing key {k}"


# ─────────────────────── SSOT composition ───────────────────────────────

class TestSSotComposition:
    def test_indirect_usd_matches_etf_lookthrough_arithmetic(self):
        # The indirect_usd surfaced HERE must byte-match what
        # etf_lookthrough would compute for the same snapshot.
        # SSOT invariant — a future divergence (a recompute drift)
        # fails this test.
        from paper_trader.analytics.etf_lookthrough import (
            build_etf_lookthrough,
        )
        snap = _snap([_pos("TQQQ", 2, 100.0)], total_value=500.0)
        lt = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # TQQQ_NVDA indirect from lookthrough.
        tqqq = next(p for p in lt["etf_positions"]
                    if p["ticker"] == "TQQQ")
        nvda_breakdown = next(h for h in tqqq["breakdown"]
                              if h["underlying"] == "NVDA")
        expected_indirect = nvda_breakdown["indirect_usd"]

        rep = build_peer_earnings_shock(
            snap, _ec([_event("NVDA", 0.5)]),
            lambda _t: 7.0, lookthrough_map=_TEST_MAP,
        )
        actual_indirect = rep["etf_rows"][0]["underlyings"][0]["indirect_usd"]
        assert actual_indirect == expected_indirect, (
            f"peer_earnings_shock indirect_usd {actual_indirect} drifted "
            f"from etf_lookthrough's {expected_indirect}")

    def test_horizon_default_matches_module_constant(self):
        # Ensures any retune of DEFAULT_HORIZON_DAYS is intentional.
        assert DEFAULT_HORIZON_DAYS == 7.0
