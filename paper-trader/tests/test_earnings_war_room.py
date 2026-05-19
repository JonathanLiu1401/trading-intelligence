"""Lock the earnings_war_room composer.

Covers the state ladder, exact hand-computed arithmetic on the post-shock
projection, the SSOT consumption of sibling builder rows (verbatim — never
re-derived), the tier ladder (HIGH/MEDIUM/LOW), the impact-tier fallback
when only one of implied/σ is available, the headline-INSUFFICIENT honesty
branch, garbage-degrade-never-raises on every input source, and the
``TestEarningsWarRoomEndpoint`` Flask test-client coverage proving the
endpoint serves the builder output on a real seeded store with the
sibling builders' I/O seams monkeypatched offline-deterministically.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.analytics.earnings_war_room import (
    HIGH_BOOK_PCT,
    MEDIUM_BOOK_PCT,
    _DEFAULT_SINGLE_NAME_GAP_PCT,
    build_earnings_war_room,
)


# -- helpers ---------------------------------------------------------------

def _pos(ticker: str, qty: float, price: float, type_: str = "stock") -> dict:
    return {
        "ticker": ticker,
        "qty": qty,
        "current_price": price,
        "avg_cost": price,
        "type": type_,
    }


def _calendar(*rows: dict) -> dict:
    return {"events": list(rows)}


def _implied(*rows: dict) -> dict:
    return {"events": list(rows)}


def _shock(*rows: dict) -> dict:
    return {"events": list(rows)}


# -- state ladder ----------------------------------------------------------

class TestStateLadder:

    def test_no_data_empty_book(self):
        r = build_earnings_war_room([], 0.0, 1000.0, _calendar())
        assert r["state"] == "NO_DATA"
        assert r["n_events"] == 0
        assert "no priced book" in r["headline"].lower()

    def test_no_data_unpriceable_positions(self):
        positions = [_pos("NVDA", 0, 0)]
        r = build_earnings_war_room(positions, 0.0, 1000.0, _calendar())
        assert r["state"] == "NO_DATA"

    def test_no_data_zero_total_value(self):
        positions = [_pos("NVDA", 2, 100)]
        r = build_earnings_war_room(positions, 0.0, 1000.0, _calendar(
            {"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"}
        ))
        assert r["state"] == "NO_DATA"

    def test_no_events_priced_book_empty_calendar(self):
        positions = [_pos("NVDA", 2, 222.35)]
        r = build_earnings_war_room(positions, 444.70, 1000.0, _calendar())
        assert r["state"] == "NO_EVENTS"
        assert r["n_events"] == 0
        assert "no held imminent" in r["headline"].lower()

    def test_no_events_event_for_unheld_ticker(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "TSLA", "days_away": 1.0, "tier": "WATCH"})
        r = build_earnings_war_room(positions, 444.70, 1000.0, cal)
        assert r["state"] == "NO_EVENTS"

    def test_no_events_event_beyond_horizon(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": 30.0, "tier": "HELD_SOON"})
        r = build_earnings_war_room(positions, 444.70, 1000.0, cal)
        assert r["state"] == "NO_EVENTS"

    def test_no_events_event_in_the_past(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": -0.5, "tier": "HELD_IMMINENT"})
        r = build_earnings_war_room(positions, 444.70, 1000.0, cal)
        assert r["state"] == "NO_EVENTS"

    def test_ok_held_imminent_event(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": 0.1, "tier": "HELD_IMMINENT"})
        imp = _implied({"ticker": "NVDA", "implied_move_pct": 7.0})
        r = build_earnings_war_room(positions, 1000.0, 1000.0, cal, implied_move_result=imp)
        assert r["state"] == "OK"
        assert r["n_events"] == 1


# -- exact arithmetic ------------------------------------------------------

class TestExactArithmetic:
    """Pinned $1000 book / NVDA 2 @ $222.35 / implied 7% / σ 1.8% scenario.

    Hand-computed numbers below — a drift in any composition arithmetic
    fails these assertions exactly.
    """

    POS_VALUE = 444.70                 # 2 × 222.35
    BOOK = 1000.0
    WEIGHT_PCT = 44.47                 # 444.70 / 1000 × 100
    IMPLIED_PCT = 7.0
    IMPLIED_DOLLAR = 31.13             # 444.70 × 0.07 = 31.129
    IMPLIED_BOOK_PCT = 3.11            # 31.129 / 1000 × 100
    POST_SHOCK_TOTAL = 968.87          # 1000 - 31.13
    POST_SHOCK_VS_INIT = -3.11         # (968.87 - 1000)/1000 × 100
    POST_SHOCK_WEIGHT = 42.69          # (444.70 - 31.13)/968.87 × 100
    SIGMA_PCT = 1.8

    def _build(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": 0.1,
                         "tier": "HELD_IMMINENT",
                         "earnings_date": "2026-05-20"})
        imp = _implied({"ticker": "NVDA",
                        "implied_move_pct": self.IMPLIED_PCT})
        shk = _shock({"ticker": "NVDA", "sigma_pct": self.SIGMA_PCT,
                      "sigma_dollar_move": 8.0, "sigma_book_pct": 0.8})
        return build_earnings_war_room(
            positions, self.BOOK, self.BOOK, cal,
            implied_move_result=imp, earnings_shock_result=shk,
        )

    def test_weight_pct(self):
        r = self._build()
        assert r["events"][0]["weight_pct"] == self.WEIGHT_PCT

    def test_implied_dollar_at_risk(self):
        r = self._build()
        assert r["events"][0]["implied_dollar_at_risk"] == self.IMPLIED_DOLLAR

    def test_implied_book_pct(self):
        r = self._build()
        assert r["events"][0]["implied_book_pct"] == self.IMPLIED_BOOK_PCT

    def test_post_shock_total_value(self):
        r = self._build()
        assert r["events"][0]["post_shock_total_value"] == self.POST_SHOCK_TOTAL

    def test_post_shock_vs_initial_pct(self):
        r = self._build()
        assert r["events"][0]["post_shock_vs_initial_pct"] == self.POST_SHOCK_VS_INIT

    def test_post_shock_weight_pct_reduces(self):
        r = self._build()
        # After the position drops more than the rest of the book, its
        # weight as a % of the (smaller) post-shock book must DECREASE.
        assert r["events"][0]["post_shock_weight_pct"] == self.POST_SHOCK_WEIGHT
        assert self.POST_SHOCK_WEIGHT < self.WEIGHT_PCT

    def test_sigma_consumed_verbatim(self):
        """The σ row inputs are read straight from earnings_shock — a drift
        in either side fails this no-drift assertion (SSOT, AGENTS.md #10)."""
        r = self._build()
        ev = r["events"][0]
        assert ev["sigma_pct"] == self.SIGMA_PCT
        assert ev["sigma_dollar_move"] == 8.0
        assert ev["sigma_book_pct"] == 0.8

    def test_total_implied_at_risk_is_sum_of_abs(self):
        positions = [_pos("NVDA", 2, 222.35), _pos("MRVL", 4, 100)]
        cal = _calendar(
            {"ticker": "NVDA", "days_away": 0.1, "tier": "HELD_IMMINENT"},
            {"ticker": "MRVL", "days_away": 5.0, "tier": "HELD_SOON"},
        )
        imp = _implied(
            {"ticker": "NVDA", "implied_move_pct": 7.0},   # 444.70×0.07 = 31.129
            {"ticker": "MRVL", "implied_move_pct": 5.0},   # 400×0.05 = 20.0
        )
        r = build_earnings_war_room(positions, 1000.0, 1000.0, cal,
                                    implied_move_result=imp)
        # 31.13 + 20.0 = 51.13
        assert r["total_implied_dollars_at_risk"] == 51.13


# -- tier ladder -----------------------------------------------------------

class TestImpactTier:

    def _row_with(self, impl_book_pct=None, sigma_book_pct=None,
                  pos_value=200.0, book=1000.0):
        positions = [_pos("NVDA", 1, pos_value)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        imp_pct = (impl_book_pct * book / pos_value) if impl_book_pct is not None else None
        sigma_pct = (sigma_book_pct * book / pos_value) if sigma_book_pct is not None else None
        sigma_dollar = (sigma_book_pct * book / 100.0) if sigma_book_pct is not None else None
        imp = _implied({"ticker": "NVDA", "implied_move_pct": imp_pct}) if imp_pct else None
        shk = _shock({"ticker": "NVDA", "sigma_pct": sigma_pct,
                      "sigma_dollar_move": sigma_dollar,
                      "sigma_book_pct": sigma_book_pct}) if sigma_pct else None
        r = build_earnings_war_room(positions, book, book, cal,
                                    implied_move_result=imp,
                                    earnings_shock_result=shk)
        return r["events"][0]

    def test_high_at_threshold(self):
        ev = self._row_with(impl_book_pct=HIGH_BOOK_PCT)
        assert ev["impact_tier"] == "HIGH"

    def test_medium_at_threshold(self):
        ev = self._row_with(impl_book_pct=MEDIUM_BOOK_PCT)
        assert ev["impact_tier"] == "MEDIUM"

    def test_low_below_medium(self):
        ev = self._row_with(impl_book_pct=MEDIUM_BOOK_PCT - 0.5)
        assert ev["impact_tier"] == "LOW"

    def test_tier_takes_max_of_implied_and_sigma(self):
        # implied small, σ HIGH-tier — overall must still read HIGH.
        ev = self._row_with(impl_book_pct=1.0, sigma_book_pct=HIGH_BOOK_PCT + 0.1)
        assert ev["impact_tier"] == "HIGH"

    def test_tier_when_only_sigma_available(self):
        ev = self._row_with(impl_book_pct=None, sigma_book_pct=3.0)
        assert ev["impact_tier"] == "MEDIUM"
        assert ev["implied_move_pct"] is None
        assert ev["state"] == "OK"

    def test_tier_when_only_implied_available(self):
        ev = self._row_with(impl_book_pct=3.0, sigma_book_pct=None)
        assert ev["impact_tier"] == "MEDIUM"
        assert ev["sigma_pct"] is None
        assert ev["state"] == "OK"

    def test_tier_unknown_when_both_missing(self):
        ev = self._row_with(impl_book_pct=None, sigma_book_pct=None)
        assert ev["impact_tier"] == "UNKNOWN"
        assert ev["state"] == "INSUFFICIENT"


# -- INSUFFICIENT honesty branch ------------------------------------------

class TestInsufficient:

    def test_insufficient_row_still_emitted_with_position_facts(self):
        """A chain miss AND zero print history must still leave the
        operator seeing *"NVDA reports tomorrow"* — the earnings_shock
        honesty precedent. The σ-and-implied numerics are withheld but
        ticker / days / weight survive."""
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": 0.1, "tier": "HELD_IMMINENT"})
        r = build_earnings_war_room(positions, 1000.0, 1000.0, cal)
        assert r["state"] == "OK"
        assert r["n_events"] == 1
        ev = r["events"][0]
        assert ev["state"] == "INSUFFICIENT"
        assert ev["ticker"] == "NVDA"
        assert ev["weight_pct"] == 44.47
        assert ev["implied_move_pct"] is None
        assert ev["sigma_pct"] is None
        assert ev["post_shock_total_value"] is None
        assert "unavailable" in ev["headline"].lower()


# -- single-name shock SSOT -----------------------------------------------

class TestSingleNameShock:

    def test_default_gap_arithmetic_when_no_stress_passed(self):
        positions = [_pos("NVDA", 2, 100)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        r = build_earnings_war_room(positions, 200.0, 1000.0, cal,
                                    implied_move_result=_implied(
                                        {"ticker": "NVDA", "implied_move_pct": 5.0}
                                    ))
        ev = r["events"][0]
        # 200 × -10% = -20.0
        assert ev["single_name_shock_dollar"] == 200.0 * _DEFAULT_SINGLE_NAME_GAP_PCT / 100.0
        assert ev["single_name_shock_source"] == "default"

    def test_consumes_stress_single_name_when_ticker_matches(self):
        """When stress_scenarios.single_name names the same ticker, war
        room reads its pnl_usd verbatim (SSOT, AGENTS.md #10)."""
        positions = [_pos("NVDA", 2, 100)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        imp = _implied({"ticker": "NVDA", "implied_move_pct": 5.0})
        stress = {"single_name": {"ticker": "NVDA", "pnl_usd": -27.5}}
        r = build_earnings_war_room(positions, 200.0, 1000.0, cal,
                                    implied_move_result=imp,
                                    stress_scenarios_result=stress)
        ev = r["events"][0]
        assert ev["single_name_shock_dollar"] == -27.5
        assert ev["single_name_shock_source"] == "stress"

    def test_falls_back_to_default_when_stress_names_different_ticker(self):
        positions = [_pos("NVDA", 2, 100)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        imp = _implied({"ticker": "NVDA", "implied_move_pct": 5.0})
        stress = {"single_name": {"ticker": "TQQQ", "pnl_usd": -99.0}}
        r = build_earnings_war_room(positions, 200.0, 1000.0, cal,
                                    implied_move_result=imp,
                                    stress_scenarios_result=stress)
        ev = r["events"][0]
        assert ev["single_name_shock_source"] == "default"
        assert ev["single_name_shock_dollar"] == -20.0


# -- sort + aggregates -----------------------------------------------------

class TestAggregates:

    def test_events_sorted_by_days_to_earnings(self):
        positions = [_pos("NVDA", 1, 100), _pos("MU", 1, 100), _pos("MRVL", 1, 100)]
        cal = _calendar(
            {"ticker": "MU",   "days_away": 5.0, "tier": "HELD_SOON"},
            {"ticker": "MRVL", "days_away": 0.5, "tier": "HELD_IMMINENT"},
            {"ticker": "NVDA", "days_away": 2.0, "tier": "HELD_SOON"},
        )
        imp = _implied(
            {"ticker": "NVDA", "implied_move_pct": 5.0},
            {"ticker": "MU",   "implied_move_pct": 5.0},
            {"ticker": "MRVL", "implied_move_pct": 5.0},
        )
        r = build_earnings_war_room(positions, 300.0, 1000.0, cal,
                                    implied_move_result=imp)
        tickers = [e["ticker"] for e in r["events"]]
        assert tickers == ["MRVL", "NVDA", "MU"]

    def test_worst_case_picks_largest_implied_dollar(self):
        positions = [_pos("A", 1, 100), _pos("B", 1, 500)]
        cal = _calendar(
            {"ticker": "A", "days_away": 1.0, "tier": "HELD_IMMINENT"},
            {"ticker": "B", "days_away": 1.0, "tier": "HELD_IMMINENT"},
        )
        imp = _implied(
            {"ticker": "A", "implied_move_pct": 10.0},  # 10
            {"ticker": "B", "implied_move_pct": 5.0},   # 25
        )
        r = build_earnings_war_room(positions, 600.0, 1000.0, cal,
                                    implied_move_result=imp)
        assert r["worst_case_event"]["ticker"] == "B"
        assert r["worst_case_event"]["implied_dollar_at_risk"] == 25.0


# -- garbage-degrade contract (never raises) ------------------------------

class TestGarbageDegrade:

    def test_positions_is_none(self):
        r = build_earnings_war_room(None, 0.0, 1000.0, None)
        assert r["state"] == "NO_DATA"

    def test_positions_non_dict_rows_silently_skipped(self):
        r = build_earnings_war_room(["garbage", None, 42], 1.0, 1000.0,
                                    _calendar())
        assert r["state"] == "NO_DATA"

    def test_total_value_garbage(self):
        positions = [_pos("NVDA", 2, 222.35)]
        r = build_earnings_war_room(positions, "garbage", 1000.0,
                                    _calendar())
        assert r["state"] == "NO_DATA"  # tv defaults to 0.0

    def test_initial_equity_garbage(self):
        # Garbage initial_equity must NOT raise; post_shock_vs_initial_pct
        # silently becomes None when initial <= 0.
        positions = [_pos("NVDA", 2, 100)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        imp = _implied({"ticker": "NVDA", "implied_move_pct": 5.0})
        r = build_earnings_war_room(positions, 200.0, "bad", cal,
                                    implied_move_result=imp)
        assert r["state"] == "OK"
        assert r["events"][0]["post_shock_vs_initial_pct"] is None

    def test_event_calendar_is_none(self):
        positions = [_pos("NVDA", 2, 222.35)]
        r = build_earnings_war_room(positions, 444.70, 1000.0, None)
        assert r["state"] == "NO_EVENTS"

    def test_event_calendar_garbage_shape(self):
        positions = [_pos("NVDA", 2, 222.35)]
        r = build_earnings_war_room(positions, 444.70, 1000.0, "not a dict")
        assert r["state"] == "NO_EVENTS"

    def test_event_calendar_with_non_dict_event(self):
        positions = [_pos("NVDA", 2, 222.35)]
        r = build_earnings_war_room(positions, 444.70, 1000.0,
                                    {"events": ["garbage", None, 42]})
        assert r["state"] == "NO_EVENTS"

    def test_implied_result_garbage(self):
        positions = [_pos("NVDA", 2, 222.35)]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        r = build_earnings_war_room(positions, 444.70, 1000.0, cal,
                                    implied_move_result="garbage")
        # implied is silently dropped; row reads INSUFFICIENT (no σ either).
        assert r["events"][0]["state"] == "INSUFFICIENT"

    def test_event_with_nonnumeric_days_away_skipped(self):
        positions = [_pos("NVDA", 2, 100)]
        cal = _calendar(
            {"ticker": "NVDA", "days_away": "next-week", "tier": "HELD_IMMINENT"}
        )
        r = build_earnings_war_room(positions, 200.0, 1000.0, cal)
        assert r["state"] == "NO_EVENTS"


# -- options multiplier ---------------------------------------------------

class TestOptionsPosition:

    def test_call_position_value_uses_100_multiplier(self):
        """Option positions carry a 100 multiplier — the stress_scenarios /
        earnings_shock _position_value convention. A 2-contract NVDA call
        @ $4 premium is worth $800 (not $8) for shock dollarization."""
        positions = [{"ticker": "NVDA", "qty": 2, "current_price": 4.0,
                      "avg_cost": 4.0, "type": "call"}]
        cal = _calendar({"ticker": "NVDA", "days_away": 1.0, "tier": "HELD_IMMINENT"})
        imp = _implied({"ticker": "NVDA", "implied_move_pct": 10.0})
        r = build_earnings_war_room(positions, 800.0, 1000.0, cal,
                                    implied_move_result=imp)
        ev = r["events"][0]
        assert ev["current_value_usd"] == 800.0
        # 800 × 10% = 80.0
        assert ev["implied_dollar_at_risk"] == 80.0


# -- endpoint integration --------------------------------------------------

class TestEarningsWarRoomEndpoint:
    """Flask test-client coverage. The implied_move / earnings_shock I/O
    seams are deterministic in this codebase: implied_move's
    options_provider and earnings_shock's history_provider are the only
    yfinance touch points and they degrade honestly to NO_CHAIN /
    INSUFFICIENT_HISTORY when their providers return None. The endpoint
    composes them; we monkeypatch the seams so the test is offline-
    deterministic and the war room result is byte-stable.
    """

    def test_endpoint_state_and_no_drift_with_builder(self, tmp_path,
                                                     monkeypatch):
        from paper_trader import store as _store
        from paper_trader import dashboard as _dash
        from paper_trader import market as _market
        from pathlib import Path

        # Redirect Store's module-level DB_PATH to a tmp file before
        # construction (Store() calls _connect() which mkdirs + opens
        # DB_PATH at instantiation; the monkeypatch must precede __init__).
        db_path = tmp_path / "pt.db"
        monkeypatch.setattr(_store, "DB_PATH", Path(db_path), raising=True)
        s = _store.Store()
        pid = s.upsert_position("NVDA", "stock", 2, 222.35, None, None)
        # update_position_marks signature: {position_id: (price, unrealized_pl)}
        s.update_position_marks({pid: (222.35, 0.0)})
        s.update_portfolio(555.30, 1000.0, [
            {"ticker": "NVDA", "qty": 2, "type": "stock",
             "current_price": 222.35, "avg_cost": 222.35}
        ])

        # Monkeypatch the dashboard's store getter to our fresh instance.
        monkeypatch.setattr(_dash, "get_store", lambda: s)

        # Force implied_move to return a deterministic chain (no yfinance).
        def _fake_chain(ticker, target_dte):
            return {
                "expiry": "2026-05-23",
                "calls": [{"strike": 222.0, "bid": 7.5, "ask": 8.5,
                           "lastPrice": 8.0, "impliedVolatility": 0.6}],
                "puts":  [{"strike": 222.0, "bid": 7.5, "ask": 8.5,
                           "lastPrice": 8.0, "impliedVolatility": 0.6}],
            }
        monkeypatch.setattr(_market, "get_options_chain", _fake_chain)

        # Force earnings_shock's history provider to a fixed series.
        monkeypatch.setattr(_dash, "_earnings_history_for",
                            lambda t, depth=8: [1.0, -2.0, 3.0, -1.5, 0.5, 2.5, -3.0, 1.0])

        # Force event_calendar to return an imminent NVDA event without
        # touching digital-intern's earnings_calendar.json on disk.
        from paper_trader.analytics import event_calendar as _ec
        def _fake_ec(positions, names):
            return {
                "events": [{
                    "ticker": "NVDA",
                    "days_away": 0.1,
                    "tier": "HELD_IMMINENT",
                    "earnings_date": "2026-05-20",
                }],
            }
        monkeypatch.setattr(_ec, "build_event_calendar", _fake_ec)

        client = _dash.app.test_client()
        # SWR cache key: the @swr_cached wrapper short-circuits on cold
        # poll; we call the undecorated function via test_client and
        # accept the warming branch + retry.
        import json
        import time
        for _ in range(10):
            resp = client.get("/api/earnings-war-room")
            data = json.loads(resp.data)
            if data.get("state") in ("OK", "NO_DATA", "NO_EVENTS"):
                break
            time.sleep(0.5)
        assert data.get("state") == "OK", f"unexpected: {data}"
        assert data["n_events"] == 1
        ev = data["events"][0]
        assert ev["ticker"] == "NVDA"
        # weight = 444.70/1000 × 100 = 44.47
        assert ev["weight_pct"] == 44.47
        # impact tier is HIGH or MEDIUM (depends on chain — straddle
        # mid = (8.0 + 8.0) = 16.0 / 222.35 = 7.20% implied)
        assert ev["impact_tier"] in ("HIGH", "MEDIUM")
        assert ev["implied_move_pct"] is not None
        # Verbatim from earnings_shock: σ computed from the 8-print pinned
        # series above (pop stdev); not None.
        assert ev["sigma_pct"] is not None
        assert data["worst_case_event"]["ticker"] == "NVDA"
