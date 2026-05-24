"""Tests for paper_trader.analytics.initiation_drought.

The verdict drives operator action — EXPLORING ("keep watching"), STEADY
("normal mix"), RECYCLING ("exploration has stalled"), STUCK_ON_NAMES
("bot has abandoned the watchlist"). A silently-broken first-occurrence
walk or wrong recycle counter would misdirect, so every assertion below
is on a *specific* expected verdict / count / field.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import initiation_drought as idr  # noqa: E402

_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _trade(action: str, ticker: str, hours_ago: float) -> dict:
    """One trade row. Newest-first store convention: caller composes the list
    newest-first by passing the freshest ``hours_ago`` first."""
    ts = (_NOW - timedelta(hours=hours_ago)).isoformat()
    return {"action": action, "ticker": ticker, "timestamp": ts,
            "qty": 1.0, "price": 100.0, "value": 100.0}


# ─── NO_DATA cases ────────────────────────────────────────────────────

def test_empty_trades_returns_no_data():
    out = idr.build_initiation_drought([], now=_NOW)
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["total_buys"] == 0
    assert out["total_initiations"] == 0
    assert out["distinct_tickers"] == []
    assert out["headline"]


def test_none_trades_returns_no_data():
    out = idr.build_initiation_drought(None, now=_NOW)
    assert out["state"] == "NO_DATA"


def test_only_sells_returns_no_data_verdict():
    # No buys at all — there's no notion of "initiation" to grade.
    trades = [_trade("SELL", "NVDA", 1.0), _trade("SELL", "AMD", 5.0)]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "NO_DATA"
    assert out["total_buys"] == 0
    # Sells DO count toward total_trades but not total_buys.
    assert out["total_trades"] == 2


# ─── option BUY actions count as initiations ──────────────────────────

def test_buy_call_and_buy_put_count_as_buys():
    trades = [
        _trade("BUY_PUT", "QQQ", 1.0),
        _trade("BUY_CALL", "SPY", 2.0),
        _trade("BUY", "NVDA", 3.0),
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["total_buys"] == 3
    assert out["total_initiations"] == 3
    assert set(out["distinct_tickers"]) == {"SPY", "QQQ", "NVDA"}


# ─── insufficient history ─────────────────────────────────────────────

def test_insufficient_history_below_min_buys():
    # 4 buys, four distinct tickers — below the _INSUFFICIENT_HISTORY=5 cap.
    trades = [
        _trade("BUY", "NVDA", 1.0),
        _trade("BUY", "AMD", 2.0),
        _trade("BUY", "MU", 3.0),
        _trade("BUY", "TSM", 4.0),
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "INSUFFICIENT_HISTORY"
    assert out["total_buys"] == 4


# ─── EXPLORING: recent net-new initiation ─────────────────────────────

def test_exploring_verdict_when_last_init_under_24h():
    # ≥ 5 buys, last initiation was a brand-new ticker 6h ago.
    trades = [
        _trade("BUY", "MU",   6.0),   # newest — initiation of MU
        _trade("BUY", "NVDA", 30.0),
        _trade("BUY", "AMD",  31.0),
        _trade("BUY", "NVDA", 32.0),
        _trade("BUY", "TSM",  33.0),
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "EXPLORING"
    assert out["last_initiation_ticker"] == "MU"
    assert out["hours_since_last_initiation"] == 6.0
    # MU was the freshest BUY and the first MU buy in the ledger; recycles
    # since are zero by definition.
    assert out["recycles_since_last_initiation"] == 0


# ─── STUCK_ON_NAMES: many buys, ≤ 2 distinct tickers ─────────────────

def test_stuck_on_names_when_distinct_le_2_and_buys_ge_10():
    # The pathology this builder was written for: 13 buys, only NVDA+TQQQ.
    trades = []
    for i in range(13):
        ticker = "NVDA" if i % 2 == 0 else "TQQQ"
        # Oldest buy 100h ago; freshest 1h ago. Newest-first list shape.
        trades.append(_trade("BUY", ticker, 1.0 + i * 8.0))
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "STUCK_ON_NAMES"
    assert out["distinct_tickers_ever"] == 2
    assert set(out["distinct_tickers"]) == {"NVDA", "TQQQ"}
    assert out["total_buys"] == 13


def test_stuck_threshold_just_below_does_not_trigger():
    # 10 buys = _STUCK_MIN_BUYS exactly, but 3 distinct tickers > 2 cap.
    trades = []
    rotation = ["NVDA", "AMD", "MU"]
    for i in range(10):
        trades.append(_trade("BUY", rotation[i % 3], 1.0 + i * 8.0))
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] != "STUCK_ON_NAMES"


# ─── RECYCLING: many post-init re-cycles + cooled-off last init ──────

def test_recycling_verdict_when_recycles_pile_up_after_last_init():
    # 1 init 60h ago, then 6 same-ticker re-cycles (BUYs of the already-
    # known ticker). The recycle count since the last init is ≥ 5 and the
    # last init is > 48h ago, so RECYCLING.
    trades = [
        # Newest first → 6 NVDA re-cycles in the last 30 hours
        _trade("BUY", "NVDA", 1.0),
        _trade("BUY", "NVDA", 5.0),
        _trade("BUY", "NVDA", 9.0),
        _trade("BUY", "NVDA", 12.0),
        _trade("BUY", "NVDA", 20.0),
        _trade("BUY", "NVDA", 28.0),
        # AMD initiation 60h ago.
        _trade("BUY", "AMD", 60.0),
        # NVDA initiation oldest of all.
        _trade("BUY", "NVDA", 96.0),
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "RECYCLING"
    assert out["last_initiation_ticker"] == "AMD"
    # 6 NVDA buys after the AMD initiation, all re-cycles since the AMD
    # initiation was the most recent.
    assert out["recycles_since_last_initiation"] == 6
    assert out["hours_since_last_initiation"] == 60.0


def test_recycling_does_not_fire_with_recent_init():
    # 6 recycles but the LAST initiation was 12h ago — EXPLORING wins.
    trades = [
        _trade("BUY", "AMD", 12.0),   # most recent = initiation
        _trade("BUY", "NVDA", 13.0),
        _trade("BUY", "NVDA", 14.0),
        _trade("BUY", "NVDA", 15.0),
        _trade("BUY", "NVDA", 16.0),
        _trade("BUY", "NVDA", 17.0),
        _trade("BUY", "NVDA", 18.0),
        _trade("BUY", "NVDA", 96.0),  # initial NVDA buy
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "EXPLORING"


# ─── STEADY: typical exploration mix ──────────────────────────────────

def test_steady_when_diverse_and_no_extremes():
    # 6 buys across 5 distinct names; last init 36h ago — not exploring,
    # not stuck, not enough re-cycles to be recycling.
    trades = [
        _trade("BUY", "AMD",  36.0),  # most recent = AMD initiation
        _trade("BUY", "NVDA", 40.0),  # NVDA re-cycle
        _trade("BUY", "MU",   50.0),  # initiation
        _trade("BUY", "TSM",  60.0),  # initiation
        _trade("BUY", "SPY",  70.0),  # initiation
        _trade("BUY", "NVDA", 80.0),  # initiation
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["verdict"] == "STEADY"
    assert out["total_initiations"] == 5
    assert out["recycles_since_last_initiation"] == 0  # AMD was last, no recycles after
    # 1 recycle total, 6 buys.
    assert out["total_recycles"] == 1


# ─── chronology must reverse the newest-first input ───────────────────

def test_first_occurrence_is_based_on_chronological_order():
    # Newest first: NVDA, NVDA, AMD. Chronologically: AMD first, then NVDA,
    # then NVDA again. AMD is the *first* initiation in time; NVDA is the
    # second. Last initiation in time is NVDA (the older of the two NVDA
    # rows — the FIRST NVDA buy), with the most-recent NVDA row a re-cycle.
    trades = [
        _trade("BUY", "NVDA", 1.0),   # most-recent → re-cycle
        _trade("BUY", "NVDA", 5.0),   # the actual NVDA initiation chronologically
        _trade("BUY", "AMD",  10.0),  # oldest → first initiation
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    # Both are initiations once each.
    assert out["total_initiations"] == 2
    assert out["total_recycles"] == 1
    # The MOST RECENT initiation (in chronological order) was NVDA, 5h ago.
    assert out["last_initiation_ticker"] == "NVDA"
    assert out["hours_since_last_initiation"] == 5.0


# ─── watchlist coverage ───────────────────────────────────────────────

def test_watchlist_coverage_when_watchlist_provided():
    trades = [
        _trade("BUY", "NVDA", 1.0),
        _trade("BUY", "AMD",  10.0),
    ]
    wl = ["NVDA", "AMD", "MU", "TSM"]  # 4-name watchlist
    out = idr.build_initiation_drought(trades, watchlist=wl, now=_NOW)
    assert out["watchlist_size"] == 4
    assert out["watchlist_coverage_pct"] == 50.0
    # Unseen: MU + TSM.
    assert set(out["watchlist_unseen"]) == {"MU", "TSM"}


def test_watchlist_case_insensitive_and_dedupe():
    # Watchlist contains case variants and a duplicate — should normalise.
    trades = [_trade("BUY", "nvda", 1.0)]
    wl = ["NVDA", "nvda", "AMD"]
    out = idr.build_initiation_drought(trades, watchlist=wl, now=_NOW)
    # Distinct set is {NVDA, AMD} after upper-casing & dedupe.
    assert out["watchlist_size"] == 2
    assert out["watchlist_coverage_pct"] == 50.0


def test_no_watchlist_means_no_coverage_fields():
    trades = [_trade("BUY", "NVDA", 1.0)]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["watchlist_size"] is None
    assert out["watchlist_coverage_pct"] is None
    assert out["watchlist_unseen"] == []


# ─── garbage input never raises ──────────────────────────────────────

def test_garbage_timestamps_dont_raise():
    trades = [
        {"action": "BUY", "ticker": "NVDA", "timestamp": "not-a-date",
         "qty": 1, "price": 1, "value": 1},
        {"action": "BUY", "ticker": "AMD", "timestamp": None,
         "qty": 1, "price": 1, "value": 1},
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["state"] == "OK"
    # last_init_ts_obj is None for both → hours_since_last_initiation is None.
    assert out["hours_since_last_initiation"] is None
    assert out["total_initiations"] == 2


def test_missing_ticker_field_skipped_silently():
    trades = [
        _trade("BUY", "NVDA", 1.0),
        {"action": "BUY", "ticker": "", "timestamp": _NOW.isoformat(),
         "qty": 1, "price": 1, "value": 1},
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["total_buys"] == 1
    assert out["distinct_tickers"] == ["NVDA"]


# ─── headline / verdict_detail always present ────────────────────────

def test_headline_and_detail_always_populated():
    trades = [_trade("BUY", "NVDA", 5.0)]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["headline"]
    assert out["verdict_detail"]
    assert isinstance(out["headline"], str)
    assert isinstance(out["verdict_detail"], str)


# ─── recycle_rate arithmetic ─────────────────────────────────────────

def test_recycle_rate_arithmetic():
    # 5 buys, 2 unique tickers → 3 recycles / 5 buys = 0.6.
    trades = [
        _trade("BUY", "NVDA", 1.0),
        _trade("BUY", "NVDA", 2.0),
        _trade("BUY", "AMD",  3.0),
        _trade("BUY", "NVDA", 4.0),
        _trade("BUY", "NVDA", 5.0),
    ]
    out = idr.build_initiation_drought(trades, now=_NOW)
    assert out["total_buys"] == 5
    assert out["total_initiations"] == 2
    assert out["total_recycles"] == 3
    assert out["recycle_rate"] == 0.6
