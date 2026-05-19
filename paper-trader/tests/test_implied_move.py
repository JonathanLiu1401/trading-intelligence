"""Tests for analytics/implied_move.py — options-market-implied move per
held position with an imminent earnings print.

The desk's pre-earnings frame is: backward-looking realized σ
(``earnings_shock``), backward-looking observed quartiles
(``earnings_distribution``), and **forward-looking market-priced**
implied move (this builder). The first two were already locked; this
fills the missing forward read.

Discriminating regressions locked here:
* the ATM-strike pick (min |strike − spot|) — a "first listed" or "exact
  match required" regression would mis-price every implied;
* the straddle mid math (``(bid+ask)/2`` when both > 0, else lastPrice
  > 0, else None — a one-sided bid silently averaged against ask=0 would
  halve every implied);
* the $-at-risk arithmetic (``position_value × implied_pct/100`` — a
  drift from this would mis-dollarize against the held book);
* held-but-distant exclusion (only events ≤ horizon_days scored);
* NaN-in-quote rejection (a half-NaN sum would propagate silently);
* never-raises contract on garbage rows (the ``_safe`` discipline);
* SSOT no-drift: held set + days_away consumed *verbatim* from the
  ``build_event_calendar`` events list — a regression that re-derives
  days_away here would shift the boundary;
* endpoint ↔ builder no-drift via the real Flask test_client.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import implied_move
from paper_trader.analytics.implied_move import (
    DEFAULT_HORIZON_DAYS,
    ELEVATED_BOOK_PCT,
    MODERATE_BOOK_PCT,
    _atm_row,
    _mid_or_last,
    _row_verdict,
    _safe_float,
    build_implied_move,
)


# ─────────────────────────── helpers ───────────────────────────

def _ec(events: list[dict]) -> dict:
    """A minimal build_event_calendar-shaped result for the builder."""
    return {"events": events, "source_ok": True, "state": "OK"}


def _ev(ticker: str, days_away: float) -> dict:
    """A minimal event-calendar event row."""
    return {
        "ticker": ticker,
        "days_away": days_away,
        "earnings_date": (datetime.now(timezone.utc)
                          + timedelta(days=days_away)).isoformat(),
        "tier": "HELD_IMMINENT" if days_away <= 3 else "HELD_SOON",
    }


def _chain(expiry: str, calls: list[dict], puts: list[dict]) -> dict:
    return {"expiry": expiry, "calls": calls, "puts": puts}


# ─────────────────────────── helper-fn locks ───────────────────────────

class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        # The discriminator: yfinance returns NaN for thin chains; naive
        # float(nan) is numeric but propagates through arithmetic.
        assert _safe_float(float("nan")) is None

    def test_string_garbage(self):
        assert _safe_float("not-a-number") is None

    def test_real_number(self):
        assert _safe_float("3.5") == 3.5
        assert _safe_float(0) == 0.0


class TestMidOrLast:
    def test_normal_mid(self):
        # bid=1.0, ask=1.4 → mid=1.2 (not lastPrice=99 — discriminator
        # for "mid wins over last when both quoted").
        assert _mid_or_last(
            {"bid": 1.0, "ask": 1.4, "lastPrice": 99.0}
        ) == pytest.approx(1.2)

    def test_one_sided_falls_back_to_last(self):
        # ask=0 ⇒ no real two-sided quote ⇒ use lastPrice
        assert _mid_or_last(
            {"bid": 1.0, "ask": 0.0, "lastPrice": 0.9}
        ) == 0.9

    def test_one_sided_no_last_returns_none(self):
        # ask=0 and lastPrice=0 ⇒ honest None, not 0.5
        assert _mid_or_last({"bid": 1.0, "ask": 0.0, "lastPrice": 0.0}) is None

    def test_nan_quote_returns_none(self):
        # bid=NaN ⇒ honest None even though ask is good (the discriminator
        # — a propagated NaN would silently corrupt the straddle sum).
        assert _mid_or_last(
            {"bid": float("nan"), "ask": 1.0, "lastPrice": 0.0}
        ) is None

    def test_garbage_input(self):
        assert _mid_or_last(None) is None
        assert _mid_or_last("not-a-dict") is None
        assert _mid_or_last({}) is None


class TestAtmRow:
    def test_picks_min_abs_distance(self):
        # Spot=$100. Strikes 95, 100, 105 → pick 100. The discriminator:
        # a "first listed" regression would pick 95.
        side = [
            {"strike": 95.0, "bid": 1.0, "ask": 1.5},
            {"strike": 100.0, "bid": 2.0, "ask": 2.5},
            {"strike": 105.0, "bid": 0.5, "ask": 1.0},
        ]
        assert _atm_row(side, 100.0)["strike"] == 100.0

    def test_picks_min_when_not_exact_match(self):
        # Spot=$98 with strikes 95/100/105 → pick 100 (|100-98|=2 < |95-98|=3).
        side = [
            {"strike": 95.0, "bid": 1.0, "ask": 1.5},
            {"strike": 100.0, "bid": 2.0, "ask": 2.5},
            {"strike": 105.0, "bid": 0.5, "ask": 1.0},
        ]
        assert _atm_row(side, 98.0)["strike"] == 100.0

    def test_empty_or_garbage(self):
        assert _atm_row([], 100.0) is None
        assert _atm_row(None, 100.0) is None
        assert _atm_row([{"strike": "garbage"}], 100.0) is None

    def test_invalid_spot(self):
        assert _atm_row([{"strike": 100.0}], 0) is None
        assert _atm_row([{"strike": 100.0}], -50) is None


class TestRowVerdict:
    def test_thresholds(self):
        assert _row_verdict(ELEVATED_BOOK_PCT) == "ELEVATED"
        assert _row_verdict(ELEVATED_BOOK_PCT - 0.01) == "MODERATE"
        assert _row_verdict(MODERATE_BOOK_PCT) == "MODERATE"
        assert _row_verdict(MODERATE_BOOK_PCT - 0.01) == "LOW"
        assert _row_verdict(0.0) == "LOW"
        # |negative| treated symmetrically
        assert _row_verdict(-ELEVATED_BOOK_PCT) == "ELEVATED"

    def test_none(self):
        assert _row_verdict(None) == "UNKNOWN"


# ─────────────────────────── builder state ladder ───────────────────────────

class TestStateLadder:
    def test_no_data_empty_book(self):
        r = build_implied_move([], 0.0, _ec([]), options_provider=None)
        assert r["state"] == "NO_DATA"
        assert r["headline"].startswith("Implied move: no priced book")
        assert r["n_events"] == 0

    def test_no_data_unpriceable_positions(self):
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 0, "current_price": 100, "type": "stock"}],
            0.0, _ec([{"ticker": "NVDA", "days_away": 1}]),
            options_provider=None,
        )
        assert r["state"] == "NO_DATA"

    def test_no_events_book_present_but_calendar_quiet(self):
        # Book is fine, but the calendar lists nothing — exercise NO_EVENTS
        # (distinct from NO_DATA, the operator can tell "calendar quiet"
        # from "book empty").
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100, "type": "stock"}],
            1000.0, _ec([]),
            options_provider=None,
        )
        assert r["state"] == "NO_EVENTS"
        assert r["verdict"] == "NO_EVENTS"
        assert r["n_events"] == 0

    def test_no_events_held_but_outside_horizon(self):
        # Earnings 30d away with default horizon=7d → dropped, not scored.
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100, "type": "stock"}],
            1000.0, _ec([_ev("NVDA", 30.0)]),
            options_provider=None,
        )
        assert r["state"] == "NO_EVENTS"

    def test_no_events_event_for_unheld_name(self):
        # The event-calendar can carry WATCH names; this builder is strictly
        # held-name-only (we dollarize against held value).
        r = build_implied_move(
            [{"ticker": "MSFT", "qty": 1, "current_price": 400, "type": "stock"}],
            400.0, _ec([_ev("NVDA", 1.0)]),
            options_provider=None,
        )
        assert r["state"] == "NO_EVENTS"


# ─────────────────────────── exact $-at-risk arithmetic ───────────────────────────

class TestExactArithmetic:
    def test_hand_computed_implied_book_pct(self):
        # Pinned scenario:
        #   NVDA 4 shares @ $100 spot, total book $1000.
        #   Chain: ATM call mid (bid 4, ask 5) = 4.5; ATM put mid (bid 3, ask 4) = 3.5.
        #   Straddle = 8.0. Spot = 100. Implied = 8.0% (and 1σ = 6.4%).
        #   Position value = 4 * 100 = 400. Implied $ = 400 * 0.08 = $32.
        #   Book $ % = 32 / 1000 * 100 = 3.20% → MODERATE.
        def provider(t, dte):
            assert t == "NVDA"
            return _chain("2026-05-23",
                          calls=[{"strike": 100.0, "bid": 4.0, "ask": 5.0,
                                  "lastPrice": 0.0, "impliedVolatility": 0.5}],
                          puts=[{"strike": 100.0, "bid": 3.0, "ask": 4.0,
                                 "lastPrice": 0.0, "impliedVolatility": 0.45}])

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 4, "current_price": 100.0,
              "avg_cost": 99.0, "type": "stock"}],
            1000.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=provider,
        )
        assert r["state"] == "OK"
        assert r["n_events"] == 1
        ev = r["events"][0]
        assert ev["state"] == "OK"
        assert ev["spot"] == 100.0
        assert ev["call_mid"] == 4.5
        assert ev["put_mid"] == 3.5
        assert ev["straddle"] == 8.0
        assert ev["implied_move_pct"] == 8.0
        assert ev["implied_one_sigma_pct"] == 6.4  # 0.8 * 8.0 by construction
        assert ev["implied_dollar_move"] == 32.0
        assert ev["implied_book_pct"] == 3.2
        assert ev["row_verdict"] == "MODERATE"
        # IV is decimal in the chain (0.5) → 50.0% in the output.
        assert ev["iv_atm"] == 50.0

    def test_elevated_verdict_when_implied_dwarfs_book(self):
        # NVDA 5 shares @ $200, book $1000. Straddle implies 20% move →
        # $200 dollar move = 20% of book → ELEVATED.
        def provider(t, dte):
            return _chain("2026-05-23",
                          calls=[{"strike": 200.0, "bid": 19.0, "ask": 21.0,
                                  "impliedVolatility": 0.6}],
                          puts=[{"strike": 200.0, "bid": 19.0, "ask": 21.0,
                                 "impliedVolatility": 0.6}])

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 5, "current_price": 200.0,
              "avg_cost": 200.0, "type": "stock"}],
            1000.0,
            _ec([_ev("NVDA", 0.5)]),
            options_provider=provider,
        )
        ev = r["events"][0]
        assert ev["implied_move_pct"] == 20.0
        assert ev["implied_dollar_move"] == 200.0
        assert ev["implied_book_pct"] == 20.0
        assert ev["row_verdict"] == "ELEVATED"
        assert r["verdict"] == "ELEVATED"

    def test_low_verdict_when_implied_small_pct_of_book(self):
        # 1 share @ $100 in a $10,000 book; implied $5 = 0.05% of book.
        def provider(t, dte):
            return _chain("2026-05-23",
                          calls=[{"strike": 100.0, "bid": 2.0, "ask": 3.0}],
                          puts=[{"strike": 100.0, "bid": 2.0, "ask": 3.0}])

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            10000.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=provider,
        )
        ev = r["events"][0]
        assert ev["row_verdict"] == "LOW"


# ─────────────────────────── degrade-honestly paths ───────────────────────────

class TestDegradePaths:
    def test_no_chain_when_provider_returns_none(self):
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=lambda t, dte: None,
        )
        ev = r["events"][0]
        assert ev["state"] == "NO_CHAIN"
        assert ev["implied_move_pct"] is None
        assert "chain unavailable" in ev["headline"]

    def test_no_chain_when_provider_raises(self):
        def boom(t, dte):
            raise RuntimeError("yfinance hiccup")

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=boom,
        )
        # The discriminator: builder is pure and never raises — a regression
        # that lets the provider exception bubble fails RED here.
        ev = r["events"][0]
        assert ev["state"] == "NO_CHAIN"

    def test_no_quotes_thin_chain(self):
        # Chain returned but every bid/ask/last is zero — withhold rather
        # than fabricate (the discriminator vs a regression that "rescues"
        # zeros into a synthetic implied of 0%).
        def provider(t, dte):
            return _chain("2026-05-23",
                          calls=[{"strike": 100.0, "bid": 0.0, "ask": 0.0,
                                  "lastPrice": 0.0}],
                          puts=[{"strike": 100.0, "bid": 0.0, "ask": 0.0,
                                 "lastPrice": 0.0}])

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=provider,
        )
        ev = r["events"][0]
        assert ev["state"] == "NO_QUOTES"
        assert ev["implied_move_pct"] is None

    def test_option_position_skipped_for_spot(self):
        # An option position on NVDA: its current_price is the option premium,
        # not the underlying. Without a clean spot we degrade to NO_CHAIN
        # (the discriminator: a regression that uses the premium as spot
        # would compute a nonsensical implied move).
        def provider(t, dte):
            # If we did try to fetch options on an option-position spot,
            # this provider would still pass; the row should degrade BEFORE
            # the provider runs because spot is None.
            return _chain("2026-05-23",
                          calls=[{"strike": 5.0, "bid": 0.5, "ask": 1.0}],
                          puts=[{"strike": 5.0, "bid": 0.5, "ask": 1.0}])

        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 5.0,
              "avg_cost": 4.0, "type": "call", "strike": 105.0,
              "expiry": "2026-05-23"}],
            500.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=provider,
        )
        ev = r["events"][0]
        assert ev["state"] == "NO_CHAIN"
        assert ev["spot"] is None

    def test_garbage_position_never_raises(self):
        # The ``_safe`` contract: a malformed row in the input list must not
        # crash the builder. Pin behaviour: bad rows are silently skipped.
        r = build_implied_move(
            [None,
             "not-a-dict",
             {"ticker": "NVDA", "qty": "garbage"},
             {"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            _ec([_ev("NVDA", 1.0)]),
            options_provider=lambda t, dte: None,
        )
        # Exactly one valid row makes it through, others silently dropped.
        assert r["n_events"] == 1

    def test_garbage_event_never_raises(self):
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            {"events": [None, "garbage", {"ticker": None}, _ev("NVDA", 1.0)]},
            options_provider=lambda t, dte: None,
        )
        assert r["n_events"] == 1

    def test_negative_days_away_dropped(self):
        # A "past" event leaking from a stale calendar — drop, never score.
        r = build_implied_move(
            [{"ticker": "NVDA", "qty": 1, "current_price": 100.0,
              "avg_cost": 100.0, "type": "stock"}],
            100.0,
            _ec([_ev("NVDA", -1.0)]),
            options_provider=lambda t, dte: None,
        )
        assert r["state"] == "NO_EVENTS"


# ─────────────────────────── sort + aggregate ───────────────────────────

class TestSortAndAggregate:
    def test_events_sorted_by_days_to_earnings(self):
        # NVDA in 3d, MU in 1d, AMD in 2d → expect MU, AMD, NVDA.
        def provider(t, dte):
            return _chain("2026-05-23",
                          calls=[{"strike": 100.0, "bid": 1.0, "ask": 2.0}],
                          puts=[{"strike": 100.0, "bid": 1.0, "ask": 2.0}])

        positions = [
            {"ticker": "NVDA", "qty": 1, "current_price": 100.0,
             "avg_cost": 100.0, "type": "stock"},
            {"ticker": "MU", "qty": 1, "current_price": 100.0,
             "avg_cost": 100.0, "type": "stock"},
            {"ticker": "AMD", "qty": 1, "current_price": 100.0,
             "avg_cost": 100.0, "type": "stock"},
        ]
        r = build_implied_move(
            positions, 300.0,
            _ec([_ev("NVDA", 3.0), _ev("MU", 1.0), _ev("AMD", 2.0)]),
            options_provider=provider,
        )
        order = [e["ticker"] for e in r["events"]]
        assert order == ["MU", "AMD", "NVDA"]

    def test_total_implied_book_pct_sums_abs(self):
        # Two events, each with $20 implied move on a $1000 book → total
        # 4.0% book impact (NOT averaged, NOT quadrature — the "honest
        # upper bound" semantic the earnings_shock precedent enforces).
        def provider(t, dte):
            # Straddle = 4.0; spot=100 → implied=4% → $4 dollar move on
            # a 1-share position. But we'll set qty=5 so dollar=$20.
            return _chain("2026-05-23",
                          calls=[{"strike": 100.0, "bid": 1.5, "ask": 2.5}],
                          puts=[{"strike": 100.0, "bid": 1.5, "ask": 2.5}])

        positions = [
            {"ticker": "NVDA", "qty": 5, "current_price": 100.0,
             "avg_cost": 100.0, "type": "stock"},
            {"ticker": "MU", "qty": 5, "current_price": 100.0,
             "avg_cost": 100.0, "type": "stock"},
        ]
        r = build_implied_move(
            positions, 1000.0,
            _ec([_ev("NVDA", 1.0), _ev("MU", 2.0)]),
            options_provider=provider,
        )
        # 4 (NVDA) + 4 (MU) = $40 implied total → 4.0% of $1000 book.
        # That doesn't actually equal the row-level $-at-risk sum because
        # the implied is 4% of $500 position value = $20, not $4.
        # Each event: implied_dollar = 500 * 0.04 = $20. Total = $40 → 4%.
        assert r["total_implied_book_pct"] == 4.0
        assert r["verdict"] == "MODERATE"


# ─────────────────────────── /api/implied-move parity ───────────────────────────

class TestImpliedMoveEndpoint:
    """Endpoint↔builder no-drift via the real Flask test_client (the
    ``test_event_calendar`` / ``test_stress_scenarios`` precedent). The
    discriminator: the route must serve the SAME builder against the live
    store's held names, with the chain provider routed through
    ``market.get_options_chain``. A regression that bypasses the builder
    (re-deriving implied on the route) or re-fetches the event calendar
    differently from the prompt fails RED here."""

    def test_endpoint_serves_builder_output(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.upsert_position("NVDA", "stock", 4.0, 100.0)
        s.update_portfolio(cash=600.0, total_value=1000.0,
                           positions=[{"ticker": "NVDA", "type": "stock",
                                       "qty": 4.0, "avg_cost": 100.0,
                                       "current_price": 100.0}])

        # Redirect event-calendar to a fixture file with NVDA in 1d.
        cal = tmp_path / "earnings_calendar.json"
        import json
        cal.write_text(json.dumps({
            "as_of": datetime.now(timezone.utc).isoformat(),
            "events": [{
                "ticker": "NVDA",
                "earnings_date": (datetime.now(timezone.utc)
                                  + timedelta(days=1)).isoformat(),
                "days_away": 0,
            }],
        }))
        from paper_trader.analytics import event_calendar as ec_mod
        monkeypatch.setattr(ec_mod, "_CANDIDATE_PATHS", (cal,))

        # Stub the options chain so the test is offline-deterministic.
        def fake_chain(ticker, target_dte=14):
            assert ticker == "NVDA"
            return {
                "ticker": ticker, "expiry": "2026-05-23",
                "calls": [{"strike": 100.0, "bid": 4.0, "ask": 5.0,
                           "lastPrice": 4.5, "impliedVolatility": 0.5}],
                "puts": [{"strike": 100.0, "bid": 3.0, "ask": 4.0,
                          "lastPrice": 3.5, "impliedVolatility": 0.5}],
            }
        from paper_trader import market as market_mod
        monkeypatch.setattr(market_mod, "get_options_chain", fake_chain)

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        # The route is SWR-cached; clear so this run is cold.
        try:
            dashboard._SWR_CACHE.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/implied-move")
        finally:
            s.close()

        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert "error" not in data, data
        # Recompute the builder with the SAME inputs and assert no drift.
        from paper_trader.analytics.event_calendar import build_event_calendar
        ec = build_event_calendar(
            [{"ticker": "NVDA", "type": "stock", "qty": 4, "avg_cost": 100.0,
              "current_price": 100.0}], {"NVDA"})
        expected = build_implied_move(
            [{"ticker": "NVDA", "type": "stock", "qty": 4, "avg_cost": 100.0,
              "current_price": 100.0}],
            1000.0, ec,
            options_provider=lambda t, dte: fake_chain(t, target_dte=dte),
        )
        # Build-time `as_of` differs by seconds; compare structural keys.
        for k in ("state", "n_events", "verdict"):
            assert data.get(k) == expected.get(k), (k, data.get(k), expected.get(k))
        assert data["events"][0]["implied_move_pct"] == \
            expected["events"][0]["implied_move_pct"]
        assert data["events"][0]["implied_dollar_move"] == \
            expected["events"][0]["implied_dollar_move"]
        assert data["events"][0]["implied_book_pct"] == \
            expected["events"][0]["implied_book_pct"]

    def test_endpoint_no_held_earnings_returns_no_events(
            self, tmp_path, monkeypatch):
        # An empty book → builder NO_DATA, but the held set is empty so the
        # endpoint never calls the options provider. The discriminator: a
        # regression that 500s on an empty book breaks the Discord chat
        # /api/analytics cross-fetch.
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        # No upsert_position — book is empty.

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        try:
            dashboard._SWR_CACHE.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/implied-move")
        finally:
            s.close()
        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert data["state"] in ("NO_DATA", "NO_EVENTS")
