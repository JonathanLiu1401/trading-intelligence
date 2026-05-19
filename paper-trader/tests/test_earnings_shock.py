"""Tests for analytics/earnings_shock.py — the pre-earnings dollarized
1σ shock for held positions with imminent prints.

The discriminating locks:

* **Population stdev (÷n) — SSOT with tail_risk.** The σ figure is
  asserted against a hand-computed ÷n stdev on a pinned reaction series
  (a regression to ÷(n−1) sample stdev would silently shift every σ
  figure here and fail the no-drift cross-check — the
  ``stress_scenarios`` / ``tail_risk`` `/n` precedent).
* **Per-event sample-size honesty.** A name with fewer than
  ``MIN_HISTORY=3`` historical prints reads ``INSUFFICIENT_HISTORY`` at
  the row level — the event still surfaces (so "NVDA reports tomorrow"
  is never hidden) but σ is **withheld**, never fabricated from one print
  (the ``build_correlation`` / ``build_news_velocity`` precedent).
* **State ladder NO_DATA / NO_EVENTS / OK** — NO_EVENTS is distinct from
  NO_DATA so the operator can tell *"calendar quiet"* from *"book empty"*.
* **SSOT no-drift with event_calendar.** The builder consumes the
  ``events`` list of ``build_event_calendar``'s result verbatim — held set
  & days_away come from the canonical earnings-tier source, so this
  builder and ``/api/event-calendar`` can never disagree.
* **Option ×100 dollarization** mirrors ``stress_scenarios``'s
  ``_position_betas`` (the ``recovery`` precedent on the multiplier-aware
  side; the breakeven path uses RAW ratio there — different concern).
* **Horizon filter** — held but distant (e.g. 30d-out) events excluded
  from shock scoring; past events excluded.
* **Never raises** on garbage rows / garbage events / history_provider
  raising (the ``_safe`` contract).
* **Reporter line + endpoint wiring** locks (no network in reporter, full
  history via /api/earnings-shock, swr prewarm coverage).
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.earnings_shock import (
    DEFAULT_HORIZON_DAYS,
    ELEVATED_BOOK_PCT,
    MIN_HISTORY,
    MODERATE_BOOK_PCT,
    _pop_stdev,
    build_earnings_shock,
)

_NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


# ───────────────────────── fixtures ─────────────────────────

# Hand-pinned reaction series → known μ, ÷n stdev.
# data = [-4, +6, -2, +8, +1, -3, +5, -1]; n=8; μ=1.25
# variance (÷n) = 17.9375; stdev = sqrt(17.9375) = 4.32290...
_HIST_NVDA = [-4.0, 6.0, -2.0, 8.0, 1.0, -3.0, 5.0, -1.0]
_HIST_MU_TWO = [2.0, -2.0]            # n=2 → still ≥ MIN_HISTORY? no, MIN_HISTORY=3
_HIST_AMD_THREE = [1.0, -1.0, 4.0]    # n=3 ≥ MIN_HISTORY; μ=4/3, var(÷n)=((1-4/3)^2+(-1-4/3)^2+(4-4/3)^2)/3


def _book(positions):
    return positions


def _ec(events):
    """Build a minimal event_calendar-shaped result."""
    return {"events": events, "as_of": _NOW.isoformat(), "source_ok": True}


def _imm_event(ticker, days, tier="HELD_IMMINENT"):
    """One imminent-earnings event in the build_event_calendar shape."""
    return {
        "ticker": ticker,
        "days_away": days,
        "tier": tier,
        "held": tier.startswith("HELD"),
        "earnings_date": "2026-05-20T20:00:00+00:00",
    }


# ───────────────────────── _pop_stdev ─────────────────────────


class TestPopStdev:
    """Population stdev (÷n) — the load-bearing SSOT with tail_risk's vol.
    A ÷(n−1) regression would silently shift every σ figure on the surface
    and fail the no-drift cross-check (the `_stdev_live` `/n` precedent)."""

    def test_population_not_sample(self):
        # Textbook: [2, 4, 4, 4, 5, 5, 7, 9]; μ=5; pop_var=4; pop_stdev=2.
        # A ÷(n−1) sample-stdev would give sqrt(32/7) ≈ 2.138 — caught here.
        assert _pop_stdev([2, 4, 4, 4, 5, 5, 7, 9]) == 2.0

    def test_handcomputed_nvda_series(self):
        # _HIST_NVDA: μ=1.25, var=17.9375, σ=sqrt(17.9375)
        got = _pop_stdev(_HIST_NVDA)
        assert math.isclose(got, math.sqrt(17.9375), rel_tol=1e-12)

    def test_too_few_samples_returns_none(self):
        assert _pop_stdev([]) is None
        assert _pop_stdev([1.0]) is None

    def test_garbage_skipped_never_raises(self):
        # str entries silently skipped; remaining 2 nums fit ÷n.
        assert math.isclose(_pop_stdev([1.0, "junk", 3.0]), 1.0, rel_tol=1e-12)


# ───────────────────────── state ladder ─────────────────────────


class TestStateLadder:
    def test_no_data_empty_book(self):
        r = build_earnings_shock([], 0.0, _ec([]), lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_DATA"
        assert r["events"] == []
        assert "no priced book" in r["headline"]

    def test_no_data_zero_total_value(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 1,
              "current_price": 100.0, "avg_cost": 100.0}]
        r = build_earnings_shock(b, 0.0, _ec([]), lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_DATA"

    def test_no_data_unpriceable_row(self):
        # qty=0 ⇒ value=0 ⇒ value_by_ticker empty ⇒ NO_DATA, not NO_EVENTS.
        b = [{"ticker": "NVDA", "type": "stock", "qty": 0,
              "current_price": 0.0, "avg_cost": 0.0}]
        r = build_earnings_shock(b, 1000.0, _ec([]),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_DATA"

    def test_no_events_when_book_priced_but_calendar_quiet(self):
        """Distinct from NO_DATA — operator must tell calendar-quiet from
        book-empty (the `build_correlation` honest-state-ladder precedent)."""
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        r = build_earnings_shock(b, 1000.0, _ec([]),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_EVENTS"
        assert r["events"] == []
        assert r["verdict"] == "NO_EVENTS"
        assert "no held name reports" in r["headline"]

    def test_no_events_when_event_is_for_non_held_ticker(self):
        """A WATCH-tier (non-held) event is dropped — the held-only shock
        scope (the reporter has zero use for a watchlist name's print)."""
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("AMD", 1.0, tier="WATCH")]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_EVENTS"

    def test_ok_when_at_least_one_held_imminent_event(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "OK"
        assert r["n_events"] == 1


# ───────────────────────── exact per-event math ─────────────────────────


class TestExactRowMath:
    def _build(self, history_provider=None):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 220.0}]
        events = [_imm_event("NVDA", 0.9)]
        return build_earnings_shock(
            b, 1000.0, _ec(events),
            history_provider or (lambda _t: _HIST_NVDA),
            now=_NOW,
        )

    def test_current_value_uses_current_price_not_avg_cost(self):
        r = self._build()
        row = r["events"][0]
        # 2 × 222.35 = 444.70 (NOT 2 × 220.0 = 440.0 — the avg_cost trap).
        assert row["current_value_usd"] == 444.70
        assert row["weight_pct"] == 44.47

    def test_sigma_pct_is_population_stdev_handcomputed(self):
        r = self._build()
        row = r["events"][0]
        # See _pop_stdev tests: σ on _HIST_NVDA = sqrt(17.9375) ≈ 4.235 → 4.24.
        assert row["sigma_pct"] == round(math.sqrt(17.9375), 2)
        assert row["sigma_pct"] == 4.24

    def test_sigma_dollar_is_position_value_times_sigma_frac(self):
        r = self._build()
        row = r["events"][0]
        # 444.70 × (4.3229.../100) = 19.225...
        expected = 444.70 * math.sqrt(17.9375) / 100.0
        assert row["sigma_dollar_move"] == round(expected, 2)

    def test_sigma_book_pct_is_dollar_over_total_value(self):
        r = self._build()
        row = r["events"][0]
        # sigma_dollar / 1000 * 100 = sigma_dollar / 10. Hand: ≈ 1.92.
        expected = 444.70 * math.sqrt(17.9375) / 100.0 / 1000.0 * 100.0
        assert row["sigma_book_pct"] == round(expected, 2)

    def test_three_sigma_down_stress_is_negative_three_times_sigma(self):
        r = self._build()
        row = r["events"][0]
        # 3σ down stress in dollars = -3 × σ_dollar (the asymmetric "what's
        # the bad tail look like" line — the stress_scenarios precedent).
        assert math.isclose(
            row["stress_3sigma_dollar_down"],
            -3.0 * row["sigma_dollar_move"],
            abs_tol=0.02,  # rounding tolerance on both legs
        )

    def test_history_mean_worst_best_exact(self):
        r = self._build()
        row = r["events"][0]
        # _HIST_NVDA: μ=1.25, min=-4, max=+8.
        assert row["history_mean_pct"] == 1.25
        assert row["history_worst_pct"] == -4.0
        assert row["history_best_pct"] == 8.0
        assert row["n_history"] == 8

    def test_low_verdict_when_sigma_book_pct_under_moderate(self):
        # Tiny position: 1 share × $5 = $5 → σ_book_pct ≈ 0.02 → LOW.
        b = [{"ticker": "NVDA", "type": "stock", "qty": 1,
              "current_price": 5.0, "avg_cost": 5.0}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        row = r["events"][0]
        assert abs(row["sigma_book_pct"]) < MODERATE_BOOK_PCT
        assert row["row_verdict"] == "LOW"

    def test_elevated_verdict_when_sigma_book_pct_at_or_above_threshold(self):
        # Position big enough to push σ_book_pct ≥ ELEVATED_BOOK_PCT=5.
        # 2 × 1000 = 2000; σ=4.32%; sigma_book_pct = 2000 × 4.32% / 2000 × 100% = 4.32?
        # Need sigma_book_pct ≥ 5: total_value smaller. Use tv=300, value=300 → σ_book_pct = σ_pct.
        b = [{"ticker": "NVDA", "type": "stock", "qty": 3,
              "current_price": 100.0, "avg_cost": 100.0}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 300.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        row = r["events"][0]
        # weight 100% × σ 4.32% = 4.32% of book → still MODERATE.
        # Use bigger σ via a steeper history.
        big_hist = [-15.0, 18.0, -20.0, 22.0, -10.0, 25.0]   # σ ≫ 5
        r2 = build_earnings_shock(b, 300.0, _ec(events),
                                  lambda _t: big_hist, now=_NOW)
        row2 = r2["events"][0]
        assert row2["sigma_book_pct"] >= ELEVATED_BOOK_PCT
        assert row2["row_verdict"] == "ELEVATED"


# ───────────────────────── option ×100 ─────────────────────────


class TestOptionValueMultiplier:
    def test_call_position_value_multiplied_by_100(self):
        """Options dollarize as price × qty × 100 (same as
        stress_scenarios._position_betas — SSOT). A dropped ×100 would
        understate every option position's earnings shock by 100×."""
        b = [{"ticker": "NVDA", "type": "call", "qty": 2,
              "current_price": 5.0, "avg_cost": 5.0,
              "strike": 220.0, "expiry": "2026-06-19"}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        row = r["events"][0]
        # 2 × 5 × 100 = 1000 (NOT 10).
        assert row["current_value_usd"] == 1000.0


# ───────────────────────── per-event sample-size honesty ─────────────────────────


class TestInsufficientHistory:
    def test_below_min_history_row_state_is_insufficient_and_sigma_withheld(self):
        b = [{"ticker": "MU", "type": "stock", "qty": 1,
              "current_price": 100.0, "avg_cost": 100.0}]
        events = [_imm_event("MU", 2.0)]
        # Only 2 prints, MIN_HISTORY=3 → row INSUFFICIENT.
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_MU_TWO, now=_NOW)
        row = r["events"][0]
        assert row["state"] == "INSUFFICIENT_HISTORY"
        assert row["sigma_pct"] is None
        assert row["sigma_dollar_move"] is None
        assert row["sigma_book_pct"] is None
        assert row["n_history"] == 2
        # The event surfaces — "MU reports in 2.0d" is never hidden.
        assert row["ticker"] == "MU"
        assert row["days_to_earnings"] == 2.0
        assert "σ withheld" in row["headline"]

    def test_no_history_provider_makes_every_row_insufficient(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 history_provider=None, now=_NOW)
        # Book-level state is still OK (events exist).
        assert r["state"] == "OK"
        # But every row is INSUFFICIENT and book σ aggregate withheld.
        row = r["events"][0]
        assert row["state"] == "INSUFFICIENT_HISTORY"
        assert r["total_sigma_book_pct"] is None
        assert r["verdict"] == "INSUFFICIENT_HISTORY"

    def test_exactly_min_history_emits_sigma(self):
        # MIN_HISTORY=3 boundary inclusive — n=3 must score, not withhold.
        b = [{"ticker": "AMD", "type": "stock", "qty": 1,
              "current_price": 100.0, "avg_cost": 100.0}]
        events = [_imm_event("AMD", 2.0)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_AMD_THREE, now=_NOW)
        row = r["events"][0]
        assert row["n_history"] == MIN_HISTORY
        assert row["state"] == "OK"
        assert row["sigma_pct"] is not None


# ───────────────────────── horizon + past filters ─────────────────────────


class TestHorizonAndPastFilters:
    def test_distant_held_event_dropped_from_shock_scope(self):
        """A held name 30d out is honest awareness on /api/event-calendar
        but not a shock candidate (would dilute the headline)."""
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", 30.0, tier="HELD_SOON")]
        # default horizon = 7d.
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_EVENTS"
        assert r["events"] == []

    def test_past_event_dropped(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", -2.0)]  # already reported
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "NO_EVENTS"

    def test_at_horizon_boundary_inclusive(self):
        # days_away == horizon → kept (the inclusive boundary the
        # event_calendar HELD_IMMINENT ≤3 precedent follows).
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", DEFAULT_HORIZON_DAYS, tier="HELD_SOON")]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "OK"


# ───────────────────────── never raises / _safe ─────────────────────────


class TestSafetyContract:
    def test_garbage_position_row_skipped_never_raises(self):
        b = [
            {"ticker": "NVDA", "type": "stock", "qty": "abc",   # bad qty
             "current_price": 100.0, "avg_cost": 100.0},
            {"ticker": "MU", "type": "stock", "qty": 1,         # good
             "current_price": 100.0, "avg_cost": 100.0},
        ]
        events = [_imm_event("MU", 2.0)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_AMD_THREE, now=_NOW)
        assert r["state"] == "OK"
        # Only the MU row survived value indexing.
        assert {e["ticker"] for e in r["events"]} == {"MU"}

    def test_garbage_event_skipped_never_raises(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 1,
              "current_price": 100.0, "avg_cost": 100.0}]
        events = [
            {"not": "an event"},                       # missing fields
            {"ticker": "NVDA", "days_away": "soon",    # non-numeric days
             "tier": "HELD_IMMINENT"},
            _imm_event("NVDA", 0.9),                   # good
        ]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "OK"
        assert r["n_events"] == 1                       # only the good one

    def test_history_provider_raising_degrades_to_insufficient(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 100.0, "avg_cost": 100.0}]
        events = [_imm_event("NVDA", 0.9)]

        def boom(_t):
            raise RuntimeError("yfinance exploded")
        r = build_earnings_shock(b, 1000.0, _ec(events), boom, now=_NOW)
        # The build never raises; the row degrades to INSUFFICIENT_HISTORY.
        row = r["events"][0]
        assert row["state"] == "INSUFFICIENT_HISTORY"
        assert row["n_history"] == 0

    def test_none_event_calendar_result_treated_as_empty(self):
        b = [{"ticker": "NVDA", "type": "stock", "qty": 2,
              "current_price": 100.0, "avg_cost": 100.0}]
        r = build_earnings_shock(b, 1000.0, None,
                                 lambda _t: _HIST_NVDA, now=_NOW)
        # Book is priced but no events available → NO_EVENTS, not NO_DATA.
        assert r["state"] == "NO_EVENTS"

    def test_held_set_case_insensitive(self):
        """value_by_ticker keys upper-cased so a lower-case position ticker
        still matches an upper-case event ticker (the upstream convention)."""
        b = [{"ticker": "nvda", "type": "stock", "qty": 2,
              "current_price": 222.35, "avg_cost": 222.35}]
        events = [_imm_event("NVDA", 0.9)]
        r = build_earnings_shock(b, 1000.0, _ec(events),
                                 lambda _t: _HIST_NVDA, now=_NOW)
        assert r["state"] == "OK"


# ───────────────────────── reporter line ─────────────────────────


class TestReporterEarningsShockLine:
    def test_no_data_suppressed(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            # Empty book → builder reads NO_DATA → reporter line "" (silent).
            assert reporter._earnings_shock_line(s) == ""
        finally:
            s.close()

    def test_no_events_suppressed(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            s.upsert_position("NVDA", "stock", 2.0, 222.35)
            s.update_portfolio(cash=555.30, total_value=1000.0, positions=[])
            # Stub the event_calendar to return no events → NO_EVENTS branch.
            monkeypatch.setattr(
                reporter, "_earnings_shock_line",
                reporter._earnings_shock_line,  # no-op rebind, see next stub
            )
            import paper_trader.analytics.event_calendar as ec_mod
            monkeypatch.setattr(
                ec_mod, "build_event_calendar",
                lambda *a, **k: {"events": [], "as_of": "x", "source_ok": True},
            )
            assert reporter._earnings_shock_line(s) == ""
        finally:
            s.close()

    def test_emits_dollarized_exposure_line_on_imminent(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        import paper_trader.analytics.event_calendar as ec_mod
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            s.upsert_position("NVDA", "stock", 2.0, 222.35)
            # Pin current_price via positions arg shape isn't necessary — the
            # store's open_positions returns avg_cost as current_price fallback
            # which the builder's _position_value uses. tv passed independently.
            s.update_portfolio(cash=555.30, total_value=1000.0, positions=[])
            monkeypatch.setattr(
                ec_mod, "build_event_calendar",
                lambda *a, **k: {
                    "events": [{"ticker": "NVDA", "days_away": 0.9,
                                "tier": "HELD_IMMINENT", "held": True,
                                "earnings_date": "2026-05-20T20:00:00+00:00"}],
                    "as_of": "x", "source_ok": True,
                },
            )
            line = reporter._earnings_shock_line(s)
            assert line != ""
            assert "PRE-EARNINGS RISK" in line
            assert "NVDA" in line
            assert "0.9d" in line
            # 2 × 222.35 = 444.70 dollarized exposure surfaces.
            assert "$444.70" in line
            # Book weight surfaces (444.70 / 1000 * 100 = 44.47).
            assert "44.5%" in line or "44.47%" in line
        finally:
            s.close()

    def test_builder_fault_degrades_to_empty_never_raises(
            self, tmp_path, monkeypatch):
        from paper_trader import reporter
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            # Patch the event_calendar import inside _earnings_shock_line
            # so the helper raises — the line must degrade to "" not raise.
            import paper_trader.analytics.event_calendar as ec_mod
            monkeypatch.setattr(
                ec_mod, "build_event_calendar",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            s.upsert_position("NVDA", "stock", 2.0, 222.35)
            s.update_portfolio(cash=555.30, total_value=1000.0, positions=[])
            assert reporter._earnings_shock_line(s) == ""
        finally:
            s.close()


# ───────────────────────── /api/earnings-shock endpoint ─────────────────────────


class TestEarningsShockEndpoint:
    def test_endpoint_returns_builder_shape_on_seeded_store(
            self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        from paper_trader import dashboard
        import paper_trader.analytics.event_calendar as ec_mod
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            s.upsert_position("NVDA", "stock", 2.0, 222.35)
            s.update_portfolio(cash=555.30, total_value=1000.0, positions=[])
            # Stub the event_calendar so we don't depend on a real
            # earnings_calendar.json being present in the test env.
            monkeypatch.setattr(
                ec_mod, "build_event_calendar",
                lambda *a, **k: {
                    "events": [{"ticker": "NVDA", "days_away": 0.9,
                                "tier": "HELD_IMMINENT", "held": True,
                                "earnings_date": "2026-05-20T20:00:00+00:00"}],
                    "as_of": "x", "source_ok": True,
                },
            )
            # And stub the yfinance I/O seam so the endpoint is fully offline.
            monkeypatch.setattr(
                dashboard, "_earnings_history_for",
                lambda ticker, depth=8: _HIST_NVDA,
            )
            dashboard.app.config["TESTING"] = True
            with dashboard.app.test_client() as client:
                rep = client.get("/api/earnings-shock").get_json()
            assert rep["state"] == "OK"
            assert rep["n_events"] == 1
            row = rep["events"][0]
            assert row["ticker"] == "NVDA"
            assert row["sigma_pct"] == 4.24
            assert row["current_value_usd"] == 444.70
        finally:
            s.close()

    def test_endpoint_in_swr_prewarm_set(self):
        """`earnings-shock` is @swr_cached, so prewarm coverage is required
        (the test_swr_prewarm_coverage prewarm==@swr_cached invariant)."""
        import inspect
        from paper_trader import dashboard
        src = inspect.getsource(dashboard._swr_prewarm)
        assert '"earnings-shock"' in src
