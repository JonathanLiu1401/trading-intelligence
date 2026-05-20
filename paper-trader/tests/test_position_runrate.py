"""Tests for the position-runrate builder + endpoint.

These tests exercise the dollar-per-day-held arithmetic + verdict bands and
pin the edge cases that would actually mislead a trader: a stale-marked
position must NOT yield a falsely-FLAT verdict, a sub-1h fill must yield
FRESH (not noise-driven BLEEDING), and a deeply-losing position past the
band edge must yield BLEEDING with the correct dollar pace.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.position_runrate import build_position_runrate


NOW = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)


def _stock(ticker, qty, avg, cur, pl, hold_hours, *, stale=False, opt=False,
           strike=None, expiry=None):
    opened = (NOW - timedelta(hours=hold_hours)).isoformat()
    return {
        "ticker": ticker,
        "type": ("call" if opt == "call" else "put" if opt == "put" else "stock"),
        "qty": qty,
        "avg_cost": avg,
        "current_price": cur,
        "unrealized_pl": pl,
        "stale_mark": stale,
        "opened_at": opened,
        "strike": strike,
        "expiry": expiry,
    }


class TestEmptyAndDegenerate:
    def test_no_positions_returns_no_data(self):
        out = build_position_runrate([], total_value=1000.0, now=NOW)
        assert out["state"] == "NO_DATA"
        assert out["rows"] == []
        assert out["total_runrate_per_day_usd"] is None
        assert out["any_bleeding"] is False

    def test_none_positions_returns_no_data(self):
        out = build_position_runrate(None, total_value=1000.0, now=NOW)  # type: ignore
        assert out["state"] == "NO_DATA"

    def test_non_dict_rows_skipped(self):
        out = build_position_runrate(
            [None, "garbage", {}],  # type: ignore
            total_value=1000.0,
            now=NOW,
        )
        # Empty dict still produces a row (with mostly None fields) but the
        # None/string garbage entries are filtered. An empty-dict row has
        # no opened_at, so hold_seconds=0 ⇒ verdict reads FRESH (sub-1h)
        # rather than crashing; both FRESH and UNKNOWN are honest "no
        # actionable signal" reads for this degenerate row.
        assert out["state"] == "OK"
        assert len(out["rows"]) == 1
        assert out["rows"][0]["verdict"] in ("FRESH", "UNKNOWN")
        # And it must never accidentally claim "BLEEDING" / "WORKING" on
        # an empty row — that would be a contract violation.
        assert out["rows"][0]["runrate_per_day_usd"] is None


class TestRunrateMath:
    def test_5d_hold_5pct_loss_pace_is_per_day(self):
        # 5-day hold at -5% on a $200 cost basis: P/L is -$10 over 5 days
        # = -$2/day. The annualized return is -5% * 365/5 = -365%/yr, well
        # past the -100%/yr BLEEDING edge.
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=190.0, pl=-10.0,
                     hold_hours=5 * 24)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["hold_days"] == 5.0
        assert row["pl_pct"] == -5.0
        assert row["runrate_per_day_usd"] == -2.0
        assert row["annualized_pct"] == -365.0
        assert row["projected_pl_30d_usd"] == -60.0
        assert row["verdict"] == "BLEEDING"
        assert out["any_bleeding"] is True
        assert out["total_runrate_per_day_usd"] == -2.0

    def test_working_position_above_band(self):
        # 5-day hold at +1% on a $200 cost basis: annualized = +73%/yr,
        # past the +25%/yr WORKING edge.
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=202.0, pl=2.0,
                     hold_hours=5 * 24)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["pl_pct"] == 1.0
        assert row["runrate_per_day_usd"] == 0.4
        assert row["annualized_pct"] == 73.0
        assert row["verdict"] == "WORKING"
        assert out["any_bleeding"] is False

    def test_flat_position_inside_band(self):
        # 30-day hold at -0.5% on a $200 cost basis: annualized = -6%/yr,
        # well inside the band (>= -100%, <= 25%).
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=199.0, pl=-1.0,
                     hold_hours=30 * 24)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["verdict"] == "FLAT"
        assert out["any_bleeding"] is False


class TestEdgeCases:
    def test_stale_mark_yields_unknown_not_flat(self):
        # A stale mark (avg_cost == current_price after fallback) historically
        # read as a flat 0%/0$ position. This must be UNKNOWN, not FLAT.
        pos = _stock("MU", qty=1.0, avg=200.0, cur=200.0, pl=0.0,
                     hold_hours=48, stale=True)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["verdict"] == "UNKNOWN"
        assert row["runrate_per_day_usd"] is None
        assert row["pl_pct"] is None      # suppressed under stale_mark
        assert out["any_bleeding"] is False

    def test_fresh_fill_under_1h_yields_fresh(self):
        # 30-minute hold — noisy slope, verdict must read FRESH not BLEEDING.
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=190.0, pl=-10.0,
                     hold_hours=0.5)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["verdict"] == "FRESH"
        assert row["runrate_per_day_usd"] is None

    def test_future_opened_at_clamps_to_zero(self):
        # Wall-clock step-back gives an opened_at in the future. Don't crash;
        # treat as fresh (hold = 0).
        future = (NOW + timedelta(hours=1)).isoformat()
        pos = {
            "ticker": "NVDA", "type": "stock", "qty": 1.0,
            "avg_cost": 200.0, "current_price": 198.0, "unrealized_pl": -2.0,
            "opened_at": future, "stale_mark": False,
        }
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["hold_seconds"] == 0
        assert row["hold_days"] == 0.0
        assert row["verdict"] == "FRESH"

    def test_zero_avg_cost_suppresses_pl_pct(self):
        # An expired-worthless option settled at intrinsic 0 with avg_cost 0
        # must NOT divide by zero. (The intrinsic path avoids this but a
        # corrupt row coming in shouldn't crash either.)
        pos = _stock("X", qty=1.0, avg=0.0, cur=0.0, pl=0.0, hold_hours=48)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        row = out["rows"][0]
        assert row["pl_pct"] is None
        assert row["annualized_pct"] is None

    def test_options_multiplier_in_book_weight(self):
        # An option position's market value uses ×100 multiplier — verify
        # the book_weight_pct number reflects that, not the per-share price.
        pos = _stock("NVDA", qty=2.0, avg=5.0, cur=6.0, pl=200.0,
                     hold_hours=48, opt="call", strike=600.0, expiry="2026-12-19")
        out = build_position_runrate([pos], total_value=2000.0, now=NOW)
        row = out["rows"][0]
        # mv = 6.0 * 2 * 100 = 1200; weight = 1200/2000 = 60%
        assert row["book_weight_pct"] == 60.0
        assert row["strike"] == 600.0
        assert row["expiry"] == "2026-12-19"

    def test_aggregate_runrate_sums_only_valid_rows(self):
        # Mix: one valid bleeding row + one stale row → aggregate uses only
        # the valid row's runrate.
        pos1 = _stock("NVDA", qty=1.0, avg=200.0, cur=180.0, pl=-20.0,
                      hold_hours=5 * 24)  # -$4/day
        pos2 = _stock("MU", qty=1.0, avg=100.0, cur=100.0, pl=0.0,
                      hold_hours=48, stale=True)  # excluded from runrate
        out = build_position_runrate([pos1, pos2], total_value=2000.0, now=NOW)
        assert out["total_runrate_per_day_usd"] == -4.0
        assert out["any_bleeding"] is True
        assert out["worst_runrate"]["ticker"] == "NVDA"


class TestHeadline:
    def test_bleeding_headline_names_worst(self):
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=180.0, pl=-20.0,
                     hold_hours=5 * 24)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        assert "BLEEDING" in out["headline"]
        assert "NVDA" in out["headline"]
        assert "-4.00/day" in out["headline"] or "-4.0/day" in out["headline"]

    def test_working_book_says_earning(self):
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=210.0, pl=10.0,
                     hold_hours=30 * 24)  # +1.7%/yr; FLAT not WORKING
        # Force WORKING with a faster pace:
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=204.0, pl=4.0,
                     hold_hours=5 * 24)  # +2%/5d ~ 146%/yr
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        assert "earning" in out["headline"]
        assert "0.80/day" in out["headline"] or "$0.80" in out["headline"]

    def test_only_fresh_positions_says_not_yet_derivable(self):
        pos = _stock("NVDA", qty=1.0, avg=200.0, cur=199.0, pl=-1.0,
                     hold_hours=0.5)
        out = build_position_runrate([pos], total_value=1000.0, now=NOW)
        assert out["total_runrate_per_day_usd"] is None
        assert "not yet derivable" in out["headline"]


class TestEndpoint:
    """Lightweight smoke of the dashboard wiring."""

    def test_endpoint_returns_json(self, monkeypatch):
        from paper_trader import dashboard

        class FakeStore:
            def get_portfolio(self):
                opened = (NOW - timedelta(hours=48)).isoformat()
                return {
                    "cash": 100.0,
                    "total_value": 1000.0,
                    "positions": [{
                        "ticker": "NVDA", "type": "stock", "qty": 1.0,
                        "avg_cost": 200.0, "current_price": 195.0,
                        "unrealized_pl": -5.0, "stale_mark": False,
                        "opened_at": opened,
                    }],
                    "last_updated": NOW.isoformat(),
                }

            def open_positions(self):
                # Returns the same opened_at so the endpoint's join finds it.
                opened = (NOW - timedelta(hours=48)).isoformat()
                return [{
                    "ticker": "NVDA", "type": "stock", "qty": 1.0,
                    "expiry": None, "strike": None, "opened_at": opened,
                }]

        monkeypatch.setattr(dashboard, "get_store", lambda: FakeStore())
        client = dashboard.app.test_client()
        rv = client.get("/api/position-runrate")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["state"] == "OK"
        assert len(data["rows"]) == 1
        row = data["rows"][0]
        assert row["ticker"] == "NVDA"
        assert row["unrealized_pl"] == -5.0
        # 2-day hold, -2.5% return → annualized = -456%/yr ⇒ BLEEDING
        assert row["verdict"] == "BLEEDING"
