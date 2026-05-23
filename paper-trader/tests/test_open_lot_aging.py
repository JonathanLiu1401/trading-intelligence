"""Tests for the per-open-lot aging builder + endpoint.

Layers pinned:

* **Builder** — ``paper_trader.analytics.open_lot_aging.build_open_lot_aging``.
  Asserts FIFO lot age computation, bucket boundaries (FRESH / NORMAL /
  MATURE / STALE), STALE_RED / STALE_GREEN / STALE_FLAT classification,
  per-position roll-up, aggregate verdict thresholds, attention-list
  filtering, options ×100 multiplier on lot dollars, and the never-raises
  / NO_DATA degradation contract.

* **Endpoint** — ``/api/open-lot-aging`` (plain GET). Byte-identical
  parity with the direct builder call (modulo ``as_of``).

* **SSOT** — the FIFO lots seen by ``open_lot_aging`` and
  ``cost_basis_ladder`` MUST match byte-for-byte (same primitive).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.analytics.open_lot_aging as ola
import paper_trader.analytics.cost_basis_ladder as cbl
from paper_trader.analytics.open_lot_aging import (
    build_open_lot_aging,
    _bucket_for_age,
    _classify_lot,
    FRESH_DAYS_MAX,
    NORMAL_DAYS_MAX,
    MATURE_DAYS_MAX,
    FLAT_PCT_TOL,
    STALE_BOOK_PCT_THRESHOLD,
    AGING_BOOK_PCT_THRESHOLD,
    BUCKET_FRESH, BUCKET_NORMAL, BUCKET_MATURE, BUCKET_STALE,
    POS_STALE_RED, POS_STALE_GREEN, POS_STALE_FLAT,
    POS_FRESH, POS_NORMAL, POS_MATURE_MIX, POS_NO_LOTS,
    AGG_NO_DATA, AGG_FRESH_BOOK, AGG_NORMAL_BOOK,
    AGG_AGING_BOOK, AGG_STALE_BOOK,
)


NOW = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)


def _ts(days_ago: float) -> str:
    """ISO timestamp `days_ago` days before NOW."""
    return (NOW - timedelta(days=days_ago)).isoformat()


def _trade(tid: int, ticker: str, action: str, qty: float, price: float,
           days_ago: float, ptype: str = "stock") -> dict:
    return {
        "id": tid,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "timestamp": _ts(days_ago),
        "type": ptype,
    }


# ───────────────────────────── bucket boundaries ───────────────────────────
def test_bucket_boundaries_are_left_closed():
    """FRESH = [0,1), NORMAL = [1,7), MATURE = [7,30), STALE = [30,∞)."""
    assert _bucket_for_age(0.0) == BUCKET_FRESH
    assert _bucket_for_age(0.99) == BUCKET_FRESH
    assert _bucket_for_age(1.0) == BUCKET_NORMAL
    assert _bucket_for_age(6.99) == BUCKET_NORMAL
    assert _bucket_for_age(7.0) == BUCKET_MATURE
    assert _bucket_for_age(29.99) == BUCKET_MATURE
    assert _bucket_for_age(30.0) == BUCKET_STALE
    assert _bucket_for_age(365.0) == BUCKET_STALE


def test_bucket_constants_locked():
    assert FRESH_DAYS_MAX == 1.0
    assert NORMAL_DAYS_MAX == 7.0
    assert MATURE_DAYS_MAX == 30.0
    assert FLAT_PCT_TOL == 0.5


# ───────────────────────────── lot classification ──────────────────────────
def test_stale_red_when_old_and_underwater():
    # 60 days old, -5% → STALE_RED
    assert _classify_lot(60.0, -5.0) == POS_STALE_RED


def test_stale_green_when_old_and_in_profit():
    assert _classify_lot(60.0, 5.0) == POS_STALE_GREEN


def test_stale_flat_when_old_and_near_zero():
    """A lot within ±FLAT_PCT_TOL is too flat to label red/green."""
    assert _classify_lot(60.0, 0.0) == POS_STALE_FLAT
    assert _classify_lot(60.0, 0.4) == POS_STALE_FLAT
    assert _classify_lot(60.0, -0.4) == POS_STALE_FLAT


def test_stale_flat_when_pl_pct_is_none():
    """A lot we can't price (no mark) must NOT be silently called green."""
    assert _classify_lot(60.0, None) == POS_STALE_FLAT


def test_young_lots_carry_only_bucket_label():
    """FRESH/NORMAL/MATURE lots don't earn red/green tags — bucket alone
    is informative enough at those ages."""
    assert _classify_lot(0.5, 50.0) == BUCKET_FRESH
    assert _classify_lot(3.0, -50.0) == BUCKET_NORMAL
    assert _classify_lot(10.0, 50.0) == BUCKET_MATURE


# ───────────────────────────── degradation ─────────────────────────────────
def test_no_positions_is_no_data():
    res = build_open_lot_aging([], [], now=NOW)
    assert res["state"] == AGG_NO_DATA
    assert res["positions"] == []
    assert res["attention"] == []


def test_none_inputs_are_no_data():
    res = build_open_lot_aging(None, None, now=NOW)
    assert res["state"] == AGG_NO_DATA


def test_positions_but_no_trades_yields_no_lots():
    """Positions present but no BUY trades → NO_DATA (no reconstructable
    lots), never a crash."""
    pos = [{"ticker": "NVDA", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None}]
    res = build_open_lot_aging(pos, [], now=NOW)
    assert res["state"] == AGG_NO_DATA
    assert res["n_positions"] == 1
    assert res["n_lots"] == 0


def test_malformed_trade_does_not_sink_result():
    pos = [{"ticker": "NVDA", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None}]
    trades = [
        {"bogus": "row"},  # missing every field
        _trade(1, "NVDA", "BUY", 1, 100.0, days_ago=2.0),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    # The good trade still reconstructs a lot.
    assert res["n_lots"] == 1


# ───────────────────────────── age computation ─────────────────────────────
def test_lot_age_in_days_is_precise():
    pos = [{"ticker": "NVDA", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 2, "expiry": None, "strike": None}]
    trades = [
        _trade(1, "NVDA", "BUY", 1, 100.0, days_ago=2.5),
        _trade(2, "NVDA", "BUY", 1, 100.0, days_ago=0.25),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    lots = res["positions"][0]["lots"]
    assert len(lots) == 2
    # Lots ordered chronologically (oldest first) — same as
    # cost_basis_ladder ordering.
    ages = [lot["age_days"] for lot in lots]
    assert ages[0] == 2.5
    assert ages[1] == 0.25


def test_oldest_lot_age_days_is_the_max_per_position():
    pos = [{"ticker": "X", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 3, "expiry": None, "strike": None}]
    trades = [
        _trade(1, "X", "BUY", 1, 100.0, days_ago=45.0),
        _trade(2, "X", "BUY", 1, 100.0, days_ago=3.0),
        _trade(3, "X", "BUY", 1, 100.0, days_ago=0.5),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["positions"][0]["oldest_lot_age_days"] == 45.0


# ───────────────────────────── per-position verdicts ───────────────────────
def test_position_stale_red_dominates_stale_green():
    """If a position holds BOTH a stale-red and a stale-green lot, the
    position verdict is STALE_RED (more decision-relevant pathology)."""
    pos = [{"ticker": "X", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 2, "expiry": None, "strike": None}]
    # Lot 1: bought at $120 60d ago → underwater at $100 → STALE_RED.
    # Lot 2: bought at $80 60d ago → in profit at $100 → STALE_GREEN.
    # Per-position verdict: STALE_RED.
    trades = [
        _trade(1, "X", "BUY", 1, 120.0, days_ago=60.0),
        _trade(2, "X", "BUY", 1, 80.0, days_ago=60.0),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["positions"][0]["verdict"] == POS_STALE_RED


def test_position_fresh_verdict_all_fresh_lots():
    pos = [{"ticker": "X", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None}]
    trades = [_trade(1, "X", "BUY", 1, 100.0, days_ago=0.5)]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["positions"][0]["verdict"] == POS_FRESH


def test_position_mature_mix_when_oldest_lot_is_mature():
    pos = [{"ticker": "X", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 2, "expiry": None, "strike": None}]
    trades = [
        _trade(1, "X", "BUY", 1, 100.0, days_ago=10.0),  # MATURE
        _trade(2, "X", "BUY", 1, 100.0, days_ago=0.5),   # FRESH
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["positions"][0]["verdict"] == POS_MATURE_MIX


# ───────────────────────────── aggregate verdict ───────────────────────────
def test_aggregate_stale_book_when_half_dollars_stale():
    """≥50 % of open-lot $ in STALE bucket → STALE_BOOK."""
    pos = [
        {"ticker": "A", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
        {"ticker": "B", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
    ]
    trades = [
        _trade(1, "A", "BUY", 1, 100.0, days_ago=60.0),  # $100 STALE
        _trade(2, "B", "BUY", 1, 100.0, days_ago=0.5),   # $100 FRESH
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    # 50% stale by value — exactly at the threshold → STALE_BOOK.
    assert res["state"] == AGG_STALE_BOOK
    assert res["stale_share_pct"] == 50.0


def test_aggregate_aging_book_when_half_dollars_mature_or_worse():
    pos = [
        {"ticker": "A", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
        {"ticker": "B", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
    ]
    trades = [
        _trade(1, "A", "BUY", 1, 100.0, days_ago=10.0),  # MATURE
        _trade(2, "B", "BUY", 1, 100.0, days_ago=0.5),   # FRESH
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["state"] == AGG_AGING_BOOK


def test_aggregate_fresh_book_when_all_lots_fresh():
    pos = [{"ticker": "A", "type": "stock", "current_price": 100.0,
            "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None}]
    trades = [_trade(1, "A", "BUY", 1, 100.0, days_ago=0.1)]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["state"] == AGG_FRESH_BOOK


def test_aggregate_normal_book_when_lots_are_normal_or_mature_minority():
    pos = [
        {"ticker": "A", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
        {"ticker": "B", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 3, "expiry": None, "strike": None},
    ]
    # A: $100 MATURE, B: $300 NORMAL. mature_share = 25 % < threshold.
    trades = [
        _trade(1, "A", "BUY", 1, 100.0, days_ago=10.0),
        _trade(2, "B", "BUY", 3, 100.0, days_ago=3.0),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["state"] == AGG_NORMAL_BOOK


# ───────────────────────────── sort + attention list ───────────────────────
def test_positions_sorted_oldest_first():
    pos = [
        {"ticker": "YOUNG", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
        {"ticker": "OLD", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
    ]
    trades = [
        _trade(1, "YOUNG", "BUY", 1, 100.0, days_ago=0.5),
        _trade(2, "OLD", "BUY", 1, 100.0, days_ago=40.0),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert [r["ticker"] for r in res["positions"]] == ["OLD", "YOUNG"]


def test_attention_list_excludes_fresh_and_normal_only_positions():
    pos = [
        {"ticker": "FRESH", "type": "stock", "current_price": 100.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
        {"ticker": "STALE", "type": "stock", "current_price": 90.0,
         "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None},
    ]
    trades = [
        _trade(1, "FRESH", "BUY", 1, 100.0, days_ago=0.5),
        _trade(2, "STALE", "BUY", 1, 100.0, days_ago=60.0),
    ]
    res = build_open_lot_aging(pos, trades, now=NOW)
    attention_tickers = [a["ticker"] for a in res["attention"]]
    assert "STALE" in attention_tickers
    assert "FRESH" not in attention_tickers


# ───────────────────────────── options ×100 multiplier ─────────────────────
def test_options_lot_value_includes_100x_multiplier():
    """One call contract at $5 should contribute $500 (not $5) to
    total_lot_value_usd so the STALE_BOOK share-of-dollars maths is
    honest about options exposure."""
    pos = [{"ticker": "NVDA", "type": "call", "current_price": 5.0,
            "avg_cost": 5.0, "qty": 1, "expiry": "2026-06-19",
            "strike": 220.0}]
    trades = [{
        "id": 1, "ticker": "NVDA", "action": "BUY_CALL",
        "qty": 1, "price": 5.0, "timestamp": _ts(2.0),
        "type": "call", "expiry": "2026-06-19", "strike": 220.0,
    }]
    res = build_open_lot_aging(pos, trades, now=NOW)
    assert res["total_lot_value_usd"] == 500.0


# ───────────────────────────── SSOT with cost_basis_ladder ─────────────────
def test_lots_match_cost_basis_ladder_byte_for_byte():
    """The FIFO lots seen by open_lot_aging MUST match cost_basis_ladder
    byte-for-byte — that's the whole point of reusing the primitive
    (otherwise the two builders could disagree on lot count / qty
    / price for the same position)."""
    pos = [{"ticker": "NVDA", "type": "stock", "current_price": 215.0,
            "avg_cost": 220.0, "qty": 3, "expiry": None, "strike": None}]
    trades = [
        _trade(11, "NVDA", "BUY", 2, 220.0, days_ago=3.0),
        _trade(12, "NVDA", "BUY", 1, 220.0, days_ago=2.0),
    ]
    cbl_res = cbl.build_cost_basis_ladder(pos, trades)
    cbl_lots = cbl_res["positions"][0]["lots"]
    ola_res = build_open_lot_aging(pos, trades, now=NOW)
    ola_lots = ola_res["positions"][0]["lots"]
    # Same lot count.
    assert len(cbl_lots) == len(ola_lots) == 2
    # Same trade_ids in the same order.
    assert [l["trade_id"] for l in cbl_lots] == [
        l["trade_id"] for l in ola_lots]
    # Same qty and price per lot.
    for a, b in zip(cbl_lots, ola_lots):
        assert a["qty"] == b["qty"]
        assert a["price"] == b["price"]


# ───────────────────────────── endpoint parity ─────────────────────────────
def test_endpoint_returns_builder_output_byte_for_byte(monkeypatch):
    fake_positions = [{
        "ticker": "X", "type": "stock", "current_price": 100.0,
        "avg_cost": 100.0, "qty": 1, "expiry": None, "strike": None,
    }]
    fake_trades = [_trade(1, "X", "BUY", 1, 100.0, days_ago=2.0)]

    class _FakeStore:
        def open_positions(self):
            return fake_positions

        def recent_trades(self, limit=2000):
            return fake_trades

    monkeypatch.setattr(dash, "get_store", lambda: _FakeStore())

    expected = build_open_lot_aging(fake_positions, fake_trades)
    with dash.app.test_client() as client:
        rv = client.get("/api/open-lot-aging")
    assert rv.status_code == 200
    got = rv.get_json()
    got.pop("as_of", None)
    expected.pop("as_of", None)
    assert got == expected
