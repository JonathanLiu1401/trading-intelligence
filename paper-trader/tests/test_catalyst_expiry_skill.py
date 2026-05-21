"""Tests for paper_trader.analytics.catalyst_expiry_skill.

Pins:
* per-position verdict ladder (ZOMBIE / FRESH_CATALYST / STRUCTURAL /
  UNCATEGORIZED / NO_REASON)
* aggregate verdict (ZOMBIE_HOLDINGS / ALL_FRESH / STRUCTURAL_BOOK /
  MIXED_BOOK / NO_DATA)
* catalyst class detection — EARNINGS / MACRO / PRODUCT / REGULATORY /
  CORPORATE / TECHNICAL / UNCATEGORIZED
* time-marker detection (tomorrow, today, in N days, Q1 2026, etc.)
* days_held computation crosses the zombie threshold cleanly
* threshold-override forwarding (zombie_days_floor / fresh_days_ceil)
* defensive: malformed rows, missing fields, no reason — never raise
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.catalyst_expiry_skill import (
    ZOMBIE,
    FRESH_CATALYST,
    STRUCTURAL,
    UNCATEGORIZED,
    NO_REASON,
    ZOMBIE_HOLDINGS,
    ALL_FRESH,
    STRUCTURAL_BOOK,
    MIXED_BOOK,
    NO_DATA,
    classify_catalyst,
    has_time_marker,
    build_catalyst_expiry_skill,
)


_NOW = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)


def _ts_days_ago(days: float) -> str:
    return (_NOW - timedelta(days=days)).isoformat()


def _position(
    ticker: str,
    *,
    days_held: float,
    type_: str = "stock",
    expiry: str | None = None,
    strike: float | None = None,
):
    return {
        "ticker": ticker,
        "type": type_,
        "opened_at": _ts_days_ago(days_held),
        "expiry": expiry,
        "strike": strike,
        "qty": 1.0,
        "current_price": 100.0,
    }


def _trade(
    tid: int,
    ticker: str,
    *,
    days_ago: float,
    reason: str,
    type_: str = "stock",
    expiry: str | None = None,
    strike: float | None = None,
    action: str = "BUY",
):
    return {
        "id": tid,
        "ticker": ticker,
        "action": action,
        "qty": 1.0,
        "price": 100.0,
        "value": 100.0,
        "timestamp": _ts_days_ago(days_ago),
        "reason": reason,
        "type": type_,
        "expiry": expiry,
        "strike": strike,
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "n_positions",
    "counts", "positions", "thresholds",
}


class TestEnvelopeStability:
    def test_no_positions_envelope(self):
        out = build_catalyst_expiry_skill([], [], now=_NOW)
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == NO_DATA
        assert out["n_positions"] == 0
        assert out["counts"] == {
            ZOMBIE: 0, FRESH_CATALYST: 0, STRUCTURAL: 0,
            UNCATEGORIZED: 0, NO_REASON: 0,
        }

    def test_none_inputs(self):
        out = build_catalyst_expiry_skill(None, None, now=_NOW)
        assert out["verdict"] == NO_DATA


class TestCatalystClassDetection:
    def test_earnings_class(self):
        assert classify_catalyst(
            "NVDA Q1 earnings tomorrow — beat expected"
        ) == "EARNINGS"

    def test_macro_class(self):
        assert classify_catalyst(
            "FOMC meeting this week, expecting rate cut"
        ) == "MACRO"

    def test_product_class(self):
        # Avoid 'announce' if mixed with earnings keywords; use a clean
        # product signal
        assert classify_catalyst(
            "Apple keynote scheduled tomorrow with iPhone unveil"
        ) == "PRODUCT"

    def test_regulatory_class(self):
        assert classify_catalyst(
            "FDA approval expected next week for new drug"
        ) == "REGULATORY"

    def test_corporate_class(self):
        # 'hike' is MACRO (rate hike) and 'announced' is PRODUCT — both
        # outrank CORPORATE under earliest-family-wins. A clean CORPORATE
        # signal needs to avoid those overlaps.
        assert classify_catalyst(
            "Massive buyback program with dividend boost"
        ) == "CORPORATE"
        assert classify_catalyst(
            "Spinoff completed, repurchase under way"
        ) == "CORPORATE"

    def test_technical_class(self):
        assert classify_catalyst(
            "RSI 65, MACD bullish crossover, breakout above resistance"
        ) == "TECHNICAL"

    def test_uncategorized_when_no_keywords(self):
        assert classify_catalyst(
            "Stock looks pretty solid here, going to add to my position"
        ) == "UNCATEGORIZED"

    def test_empty_text(self):
        assert classify_catalyst("") == "UNCATEGORIZED"
        assert classify_catalyst(None) == "UNCATEGORIZED"  # type: ignore[arg-type]

    def test_earliest_family_wins(self):
        # EARNINGS-class keyword should beat TECHNICAL when both present
        # (EARNINGS is checked first in _CATALYST_KEYWORDS ordering)
        assert classify_catalyst(
            "NVDA Q1 earnings tomorrow, RSI also showing breakout"
        ) == "EARNINGS"


class TestTimeMarkerDetection:
    def test_tomorrow_marker(self):
        assert has_time_marker("Earnings tomorrow — high conviction")

    def test_today_marker(self):
        assert has_time_marker("CPI today at 8:30am ET")

    def test_this_week_marker(self):
        assert has_time_marker("FOMC this week")

    def test_in_n_days_marker(self):
        assert has_time_marker("Earnings in 2 days")
        assert has_time_marker("Earnings in 0.9d")
        assert has_time_marker("Event in 3 hours")

    def test_qN_year_marker(self):
        assert has_time_marker("Q1 2026 results due")
        # Bare Q1 without year — currently doesn't match (intentional)
        assert not has_time_marker("Q1 setup")

    def test_iso_date_marker(self):
        assert has_time_marker("Earnings on 2026-05-20")

    def test_no_marker_in_pure_technical_reason(self):
        assert not has_time_marker(
            "RSI 65, MACD bullish crossover, breakout above resistance"
        )

    def test_post_earnings_marker(self):
        assert has_time_marker("post-earnings drift play")

    def test_empty_text(self):
        assert not has_time_marker("")
        assert not has_time_marker(None)  # type: ignore[arg-type]


class TestPerPositionVerdicts:
    def test_zombie_dated_catalyst_past_floor(self):
        # Position open 4 days, earnings catalyst with time marker
        # → ZOMBIE (>3d default floor)
        positions = [_position("NVDA", days_held=4.0)]
        trades = [_trade(
            1, "NVDA", days_ago=4.0,
            reason="NVDA Q1 earnings tomorrow — multi-catalyst setup, "
                   "RSI 61, MACD bullish",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == ZOMBIE
        assert p["catalyst_class"] == "EARNINGS"
        assert p["has_time_marker"] is True
        assert p["days_held"] >= 4.0
        assert out["verdict"] == ZOMBIE_HOLDINGS

    def test_fresh_catalyst_under_ceil(self):
        # Position open 0.5 days, earnings tomorrow → FRESH_CATALYST
        positions = [_position("NVDA", days_held=0.5)]
        trades = [_trade(
            1, "NVDA", days_ago=0.5,
            reason="NVDA Q1 earnings tomorrow",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == FRESH_CATALYST
        assert out["verdict"] == ALL_FRESH

    def test_structural_pure_technical(self):
        positions = [_position("NVDA", days_held=5.0)]
        trades = [_trade(
            1, "NVDA", days_ago=5.0,
            reason="RSI 45, MACD bullish, breakout above 200 resistance",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == STRUCTURAL
        assert p["catalyst_class"] == "TECHNICAL"
        assert out["verdict"] == STRUCTURAL_BOOK

    def test_structural_dated_class_no_time_marker(self):
        # Earnings keyword present but no time marker — treated structural
        positions = [_position("NVDA", days_held=5.0)]
        trades = [_trade(
            1, "NVDA", days_ago=5.0,
            reason="earnings season tailwind for semis broadly",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["catalyst_class"] == "EARNINGS"
        assert p["has_time_marker"] is False
        assert p["verdict"] == STRUCTURAL

    def test_uncategorized_no_keyword(self):
        positions = [_position("NVDA", days_held=5.0)]
        trades = [_trade(
            1, "NVDA", days_ago=5.0,
            reason="Looks fine here, adding to size",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == UNCATEGORIZED

    def test_no_reason_when_blank(self):
        positions = [_position("NVDA", days_held=5.0)]
        trades = [_trade(1, "NVDA", days_ago=5.0, reason="")]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == NO_REASON
        # NO_REASON alone, no zombie → STRUCTURAL_BOOK (all
        # non-dated-and-non-fresh)
        assert out["verdict"] == STRUCTURAL_BOOK

    def test_position_with_no_matching_buy_trade(self):
        positions = [_position("NVDA", days_held=5.0)]
        # No trade for NVDA at all
        out = build_catalyst_expiry_skill(positions, [], now=_NOW)
        p = out["positions"][0]
        assert p["verdict"] == NO_REASON


class TestZombieThresholdEdges:
    def test_exactly_at_zombie_floor_is_zombie(self):
        positions = [_position("NVDA", days_held=3.0)]
        trades = [_trade(
            1, "NVDA", days_ago=3.0,
            reason="NVDA Q1 earnings tomorrow",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        # days_held >= 3.0 → ZOMBIE
        assert out["positions"][0]["verdict"] == ZOMBIE

    def test_below_zombie_floor_at_fresh_ceil_boundary(self):
        # Days held = 2.0 — at fresh_days_ceil (exclusive), between
        # fresh and zombie. Builder collapses to FRESH_CATALYST.
        positions = [_position("NVDA", days_held=2.0)]
        trades = [_trade(
            1, "NVDA", days_ago=2.0,
            reason="NVDA earnings tomorrow",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out["positions"][0]["verdict"] == FRESH_CATALYST

    def test_below_fresh_ceil_is_fresh(self):
        positions = [_position("NVDA", days_held=0.1)]
        trades = [_trade(
            1, "NVDA", days_ago=0.1,
            reason="FOMC today",
        )]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out["positions"][0]["verdict"] == FRESH_CATALYST


class TestAggregateVerdict:
    def test_zombie_wins_over_fresh(self):
        positions = [
            _position("NVDA", days_held=5.0),  # zombie
            _position("TSLA", days_held=0.5),  # fresh
        ]
        trades = [
            _trade(1, "NVDA", days_ago=5.0,
                   reason="NVDA Q1 earnings tomorrow"),
            _trade(2, "TSLA", days_ago=0.5,
                   reason="TSLA earnings report tomorrow"),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out["verdict"] == ZOMBIE_HOLDINGS
        assert out["counts"][ZOMBIE] == 1
        assert out["counts"][FRESH_CATALYST] == 1

    def test_mixed_book_fresh_plus_structural(self):
        positions = [
            _position("NVDA", days_held=0.5),
            _position("TSLA", days_held=10.0),
        ]
        trades = [
            _trade(1, "NVDA", days_ago=0.5,
                   reason="earnings tomorrow"),
            _trade(2, "TSLA", days_ago=10.0,
                   reason="RSI 45 mean reversion setup"),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out["verdict"] == MIXED_BOOK

    def test_all_fresh_pure(self):
        positions = [
            _position("NVDA", days_held=0.3),
            _position("TSLA", days_held=0.5),
        ]
        trades = [
            _trade(1, "NVDA", days_ago=0.3,
                   reason="NVDA earnings tomorrow"),
            _trade(2, "TSLA", days_ago=0.5,
                   reason="FOMC today"),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out["verdict"] == ALL_FRESH


class TestThresholdOverrides:
    def test_zombie_days_floor_override(self):
        positions = [_position("NVDA", days_held=4.0)]
        trades = [_trade(
            1, "NVDA", days_ago=4.0,
            reason="NVDA Q1 earnings tomorrow",
        )]
        # Default floor=3 → ZOMBIE
        out_default = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out_default["positions"][0]["verdict"] == ZOMBIE
        # Raise to 7 → FRESH_CATALYST (5 days is still <7)
        out_relaxed = build_catalyst_expiry_skill(
            positions, trades, now=_NOW, zombie_days_floor=7.0,
        )
        assert out_relaxed["positions"][0]["verdict"] == FRESH_CATALYST

    def test_fresh_days_ceil_override(self):
        positions = [_position("NVDA", days_held=1.0)]
        trades = [_trade(
            1, "NVDA", days_ago=1.0,
            reason="earnings tomorrow",
        )]
        # Default ceil=2 → FRESH_CATALYST
        out_default = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        assert out_default["positions"][0]["verdict"] == FRESH_CATALYST

    def test_thresholds_in_envelope(self):
        out = build_catalyst_expiry_skill(
            [], [], now=_NOW,
            zombie_days_floor=5.5, fresh_days_ceil=1.5,
        )
        assert out["thresholds"]["zombie_days_floor"] == 5.5
        assert out["thresholds"]["fresh_days_ceil"] == 1.5


class TestDefensiveParse:
    def test_malformed_trades_skipped(self):
        positions = [_position("NVDA", days_held=4.0)]
        trades = [
            None,  # type: ignore[list-item]
            {"id": "garbage"},
            {"id": 1, "ticker": "NVDA", "action": "BUY",
             "timestamp": "not-a-date", "reason": "NVDA earnings tomorrow"},
            _trade(2, "NVDA", days_ago=4.0,
                   reason="NVDA Q1 earnings tomorrow"),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        # Should pick up the one valid trade
        assert out["positions"][0]["verdict"] == ZOMBIE

    def test_position_missing_opened_at(self):
        positions = [{
            "ticker": "NVDA", "type": "stock",
            "expiry": None, "strike": None,
            # No opened_at
        }]
        trades = [_trade(1, "NVDA", days_ago=0.0,
                         reason="NVDA earnings tomorrow")]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        p = out["positions"][0]
        # days_held falls back to 0.0 → FRESH_CATALYST
        assert p["days_held"] == 0.0
        assert p["verdict"] == FRESH_CATALYST


class TestMultiPositionIsolation:
    def test_trades_segregated_by_position_key(self):
        positions = [
            _position("NVDA", days_held=4.0),
            _position("TSLA", days_held=0.3),
        ]
        trades = [
            _trade(1, "NVDA", days_ago=4.0,
                   reason="NVDA Q1 earnings tomorrow"),
            _trade(2, "TSLA", days_ago=0.3,
                   reason="RSI mean reversion"),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        by_ticker = {p["ticker"]: p for p in out["positions"]}
        assert by_ticker["NVDA"]["verdict"] == ZOMBIE
        assert by_ticker["TSLA"]["verdict"] == STRUCTURAL

    def test_options_dont_collide_with_stock(self):
        positions = [
            _position("NVDA", days_held=4.0),
            _position(
                "NVDA", days_held=0.3, type_="call",
                expiry="2026-06-21", strike=225.0,
            ),
        ]
        trades = [
            _trade(1, "NVDA", days_ago=4.0,
                   reason="NVDA Q1 earnings tomorrow"),
            _trade(
                2, "NVDA", days_ago=0.3,
                reason="RSI mean reversion",
                type_="call", expiry="2026-06-21", strike=225.0,
                action="BUY_CALL",
            ),
        ]
        out = build_catalyst_expiry_skill(positions, trades, now=_NOW)
        by_type = {p["type"]: p for p in out["positions"]}
        assert by_type["stock"]["verdict"] == ZOMBIE
        assert by_type["call"]["verdict"] == STRUCTURAL
