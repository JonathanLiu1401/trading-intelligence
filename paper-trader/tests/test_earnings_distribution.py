"""Unit tests for analytics/earnings_distribution.py — the empirical
observed-quantile complement to /api/earnings-shock.

These are pure-function tests (no Flask, no yfinance, no DB), so they run
fast and don't depend on the live trader state."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_trader.analytics.earnings_distribution import (
    DEFAULT_HORIZON_DAYS,
    ELEVATED_BOOK_PCT,
    MIN_HISTORY,
    MODERATE_BOOK_PCT,
    _observed_quartiles,
    build_earnings_distribution,
)


NOW = datetime(2026, 5, 19, 2, 30, tzinfo=timezone.utc)


def _event(ticker, days_away, tier="HELD_IMMINENT"):
    return {
        "ticker": ticker,
        "days_away": days_away,
        "earnings_date": "2026-05-20T00:00:00+00:00",
        "tier": tier,
    }


def _ec(events):
    return {"events": events}


def _stock_pos(ticker, qty, current_price, avg_cost=None):
    return {
        "ticker": ticker,
        "qty": qty,
        "current_price": current_price,
        "avg_cost": avg_cost if avg_cost is not None else current_price,
        "type": "stock",
    }


# ─────────────────────── observed quartiles helper ─────────────────────────


def test_observed_quartiles_returns_none_below_min_history():
    assert _observed_quartiles([]) is None
    assert _observed_quartiles([1.0]) is None
    assert _observed_quartiles([1.0, 2.0]) is None
    # n==MIN_HISTORY must succeed.
    q = _observed_quartiles([1.0, 2.0, 3.0])
    assert q is not None
    assert q["worst"] == 1.0
    assert q["best"] == 3.0


def test_observed_quartiles_linear_interpolation_matches_numpy_default():
    # Hand-checked NIST type 7 (numpy default) on [1,2,3,4,5]:
    #   q1   @ pos=1.0 → 2.0
    #   med  @ pos=2.0 → 3.0
    #   q3   @ pos=3.0 → 4.0
    q = _observed_quartiles([1, 2, 3, 4, 5])
    assert q["worst"] == 1
    assert q["q1"] == pytest.approx(2.0)
    assert q["median"] == pytest.approx(3.0)
    assert q["q3"] == pytest.approx(4.0)
    assert q["best"] == 5


def test_observed_quartiles_handles_unsorted_input():
    q = _observed_quartiles([5, 1, 3, 2, 4])
    assert q["worst"] == 1
    assert q["best"] == 5
    assert q["median"] == pytest.approx(3.0)


def test_observed_quartiles_handles_negative_values():
    # Earnings reactions can be negative — typical input for the builder.
    q = _observed_quartiles([-8.0, -2.0, 1.0, 3.0, 5.0])
    assert q["worst"] == -8.0
    assert q["best"] == 5.0


# ─────────────────────── state ladder: NO_DATA / NO_EVENTS / OK ───────────


def test_state_no_data_when_no_priced_positions():
    r = build_earnings_distribution(
        positions=[],
        total_value=1000.0,
        event_calendar_result=_ec([]),
        history_provider=lambda _t: [],
        now=NOW,
    )
    assert r["state"] == "NO_DATA"
    assert "no priced book" in r["headline"]


def test_state_no_data_when_total_value_zero():
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.0)],
        total_value=0.0,
        event_calendar_result=_ec([_event("NVDA", 1.0)]),
        history_provider=lambda _t: [-2.1, 3.7, 0.5, -1.2, 2.0, 1.0, -0.5, 4.0],
        now=NOW,
    )
    assert r["state"] == "NO_DATA"


def test_state_no_events_when_book_has_no_imminent_print():
    # Held position but no event-calendar event within horizon.
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.0)],
        total_value=1000.0,
        event_calendar_result=_ec([]),
        history_provider=lambda _t: [-2.1, 3.7, 0.5, -1.2],
        now=NOW,
    )
    assert r["state"] == "NO_EVENTS"
    assert r["verdict"] == "NO_EVENTS"


def test_state_ok_with_full_history_emits_observed_quartiles():
    # Live shape (2026-05-19): NVDA 2 shares @ $222.35, ~44% of $1000 book.
    history = [-2.1, 3.7, 0.5, -1.2, 2.0, 1.0, -0.5, 4.0]  # 8 prints (the live n)
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: history,
        now=NOW,
    )
    assert r["state"] == "OK"
    assert r["n_events"] == 1
    row = r["events"][0]
    assert row["ticker"] == "NVDA"
    assert row["state"] == "OK"
    assert row["n_history"] == 8
    q = row["observed_quartiles"]
    assert q["worst"] == pytest.approx(-2.1)
    assert q["best"] == pytest.approx(4.0)
    # Dollar quartiles must equal position_value × pct / 100.
    pos_value = 2 * 222.35
    assert row["dollar_quartiles"]["worst"] == pytest.approx(pos_value * -2.1 / 100, abs=0.01)
    assert row["dollar_quartiles"]["best"] == pytest.approx(pos_value * 4.0 / 100, abs=0.01)
    # Book pct quartiles must equal dollar_quartiles / total_value × 100.
    assert row["book_pct_quartiles"]["worst"] == pytest.approx(pos_value * -2.1 / 1000.0, abs=0.01)


# ─────────────────────── INSUFFICIENT_HISTORY semantics ───────────────────


def test_row_state_insufficient_history_when_below_min():
    # Two prints — below MIN_HISTORY=3 — must surface the event but withhold
    # the quantiles (the same discipline /api/earnings-shock applies).
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: [-1.0, 2.0],  # n=2 < MIN_HISTORY
        now=NOW,
    )
    assert r["state"] == "OK"  # the builder-level state is OK; rows can still INSUFFICIENT
    assert len(r["events"]) == 1
    row = r["events"][0]
    assert row["state"] == "INSUFFICIENT_HISTORY"
    assert row["observed_quartiles"] is None
    assert row["dollar_quartiles"] is None
    assert "distribution withheld" in row["headline"]
    assert f"need ≥{MIN_HISTORY}" in row["headline"]


def test_book_level_verdict_insufficient_when_all_rows_insufficient():
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: [1.0],  # single print
        now=NOW,
    )
    assert r["verdict"] == "INSUFFICIENT_HISTORY"


# ─────────────────────── downside semantics ───────────────────────────────


def test_downside_worst_is_zero_when_all_observations_positive():
    # If every observed print was a gain, "downside worst" must be 0 — not
    # a manufactured negative.
    history = [1.0, 2.0, 3.0, 4.0]
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: history,
        now=NOW,
    )
    row = r["events"][0]
    assert row["downside_worst_dollar"] == 0.0
    assert row["downside_worst_book_pct"] == 0.0
    # Row verdict should drop to LOW (|0|<MODERATE threshold).
    assert row["row_verdict"] == "LOW"


def test_downside_worst_dollar_is_negative_when_worst_is_negative():
    # qty=2 px=222.35 → pos_value=444.70. tv=1000 → weight 44.47%.
    # Use a -15% worst observation so book-pct impact crosses ELEVATED (>=5%):
    # |-15% × 44.47%| = 6.67% > 5%.
    history = [-15.0, -2.0, 1.0, 3.0]
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: history,
        now=NOW,
    )
    row = r["events"][0]
    expected_dollar = 2 * 222.35 * -15.0 / 100.0
    assert row["downside_worst_dollar"] == pytest.approx(expected_dollar, abs=0.01)
    assert row["row_verdict"] == "ELEVATED"


def test_row_verdict_thresholds():
    # Construct positions so the downside lands in each tier.
    # ELEVATED: |worst observed book pct| >= 5%
    # MODERATE: 2% <= ... < 5%
    # LOW: < 2%
    cases = [
        ([-10.0, 0.0, 0.0, 0.0], "ELEVATED"),  # 10%×44.47% book share = ~4.4% → moderate? recompute
    ]
    # Let me just spot-check directly:
    # qty=2 px=222.35 → pos_value=444.70. tv=1000. weight=44.47%.
    # worst=-10% → dollar=-44.47, book_pct=-4.447% → MODERATE (>=2 < 5).
    history = [-10.0, 0.0, 0.0, 0.0]
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=lambda _t: history,
        now=NOW,
    )
    assert r["events"][0]["row_verdict"] == "MODERATE"


def test_row_verdict_low_for_small_book_impact():
    # tiny weight + small move = LOW.
    history = [-1.0, 0.0, 0.5, 0.5]
    r = build_earnings_distribution(
        positions=[_stock_pos("MU", 1, 50.0)],
        total_value=10000.0,  # MU is 0.5% of book
        event_calendar_result=_ec([_event("MU", 5.0)]),
        history_provider=lambda _t: history,
        now=NOW,
    )
    assert r["events"][0]["row_verdict"] == "LOW"


# ─────────────────────── horizon + filtering ──────────────────────────────


def test_events_outside_horizon_are_excluded():
    # 15 days out, horizon is 7 → no row.
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 15.0)]),
        history_provider=lambda _t: [-2.0, 3.0, 1.0, -1.0],
        now=NOW,
    )
    assert r["state"] == "NO_EVENTS"


def test_negative_days_away_excluded():
    # past events shouldn't be scored as forward shock.
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", -1.0)]),
        history_provider=lambda _t: [-2.0, 3.0, 1.0],
        now=NOW,
    )
    assert r["state"] == "NO_EVENTS"


def test_only_held_positions_are_dollarized():
    # event_calendar may include WATCH (non-held) names; this builder must
    # skip them (only HELD events get dollarized).
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],  # NVDA held
        total_value=1000.0,
        event_calendar_result=_ec([
            _event("NVDA", 0.9),
            _event("MRVL", 7.0, tier="WATCH"),  # not held
        ]),
        history_provider=lambda _t: [-2.0, 3.0, 1.0, -1.0, 0.5],
        now=NOW,
    )
    assert r["n_events"] == 1
    assert r["events"][0]["ticker"] == "NVDA"


# ─────────────────────── aggregate downside ────────────────────────────────


def test_total_downside_book_pct_sums_absolute_downside_dollars():
    # Two held names with imminent prints, both with negative-tail histories.
    r = build_earnings_distribution(
        positions=[
            _stock_pos("NVDA", 2, 200.0),  # 400 / 1000 = 40%
            _stock_pos("MU",   1, 100.0),  # 100 / 1000 = 10%
        ],
        total_value=1000.0,
        event_calendar_result=_ec([
            _event("NVDA", 1.0),
            _event("MU",   5.0),
        ]),
        history_provider=lambda t: {
            "NVDA": [-5.0, 1.0, 2.0, -1.0],   # worst -5%
            "MU":   [-8.0, 0.0, 1.0, 2.0],    # worst -8%
        }[t],
        now=NOW,
    )
    # NVDA downside dollar = 400 * -5 / 100 = -20 (abs 20)
    # MU   downside dollar = 100 * -8 / 100 = -8  (abs 8)
    # total_downside_book_pct = -(20+8)/1000*100 = -2.8
    assert r["total_downside_book_pct"] == pytest.approx(-2.8, abs=0.01)


def test_events_sorted_by_days_away():
    r = build_earnings_distribution(
        positions=[
            _stock_pos("FAR", 1, 100.0),
            _stock_pos("NEAR", 1, 100.0),
        ],
        total_value=1000.0,
        event_calendar_result=_ec([
            _event("FAR",  5.0),
            _event("NEAR", 1.0),  # closer print first
        ]),
        history_provider=lambda _t: [-1.0, 2.0, 1.0],
        now=NOW,
    )
    assert [e["ticker"] for e in r["events"]] == ["NEAR", "FAR"]


# ─────────────────────── never-raises contract ─────────────────────────────


def test_builder_does_not_raise_when_history_provider_throws():
    def bad_provider(_t):
        raise RuntimeError("yfinance died")
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=bad_provider,
        now=NOW,
    )
    # Row degrades to INSUFFICIENT_HISTORY (history list is empty), no raise.
    assert r["state"] == "OK"
    assert r["events"][0]["state"] == "INSUFFICIENT_HISTORY"


def test_builder_handles_garbage_event_rows():
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result={"events": [None, "junk", {"no": "fields"}]},
        history_provider=lambda _t: [-1.0, 2.0, 1.0],
        now=NOW,
    )
    # No matching event → NO_EVENTS, no raise.
    assert r["state"] == "NO_EVENTS"


def test_builder_handles_none_history_provider():
    # None provider means "skip history entirely" — every row INSUFFICIENT.
    r = build_earnings_distribution(
        positions=[_stock_pos("NVDA", 2, 222.35)],
        total_value=1000.0,
        event_calendar_result=_ec([_event("NVDA", 0.9)]),
        history_provider=None,
        now=NOW,
    )
    assert r["state"] == "OK"
    assert r["events"][0]["state"] == "INSUFFICIENT_HISTORY"


def test_thresholds_consistent_with_earnings_shock():
    # Per AGENTS.md SSOT discipline, these must mirror earnings_shock to
    # keep the two endpoints' tier labels consistent on the same shape of risk.
    from paper_trader.analytics import earnings_shock as es
    assert ELEVATED_BOOK_PCT == es.ELEVATED_BOOK_PCT
    assert MODERATE_BOOK_PCT == es.MODERATE_BOOK_PCT
    assert MIN_HISTORY == es.MIN_HISTORY
    assert DEFAULT_HORIZON_DAYS == es.DEFAULT_HORIZON_DAYS
