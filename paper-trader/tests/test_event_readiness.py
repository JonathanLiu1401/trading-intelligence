"""Pure-builder tests for ``analytics/event_readiness.py``.

Discriminating locks:

- **BLIND** when the bot is decision-blind (high empty-rate × few hours)
  even with positive cycles/hr — the headline operator scenario this
  endpoint exists to surface.
- **READY** when there's enough head-room for many usable decisions.
- **DEGRADED** between the two — the mid-band, calibration target.
- **NO_DECISIONS** when the bot has logged zero cycles in the velocity
  window (a strictly worse signal than a high empty-rate — we can't
  *expect* decisions if none are happening).
- **NO_EVENTS** when nothing held reports within the horizon.
- **exposure filter** — earnings events for tickers not in the held book
  are excluded (matches /api/earnings-risk SSOT: this is held-readiness,
  not the full calendar).
- **horizon filter** — events past the configured horizon are excluded.
- **option exposure** — uses the 100x multiplier (the /api/earnings-risk
  shape).
- **graceful inputs** — None / missing keys never raise.
- **velocity window** — decisions outside the window don't count.
- **recommended_action** — surfaces the empty-rate hint when high, so a
  chat-side enrichment doesn't have to re-derive it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.event_readiness import (
    build_event_readiness,
    _decision_velocity,
    _expected_decisions,
    _verdict_for,
)


NOW = datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc)


def _pos(ticker: str, qty: float = 2.0, price: float = 222.35,
         ptype: str = "stock") -> dict:
    return {
        "ticker": ticker,
        "qty": qty,
        "current_price": price,
        "avg_cost": price,
        "type": ptype,
    }


def _dec(minutes_ago: float, action: str = "BUY NVDA -> FILLED") -> dict:
    ts = NOW - timedelta(minutes=minutes_ago)
    return {
        "timestamp": ts.isoformat(timespec="seconds"),
        "action_taken": action,
    }


def _event(ticker: str, days_away: float,
           earnings_date: str | None = "2026-05-20T00:00:00+00:00") -> dict:
    return {
        "ticker": ticker,
        "days_away": days_away,
        "earnings_date": earnings_date,
    }


# ── verdict ladder (calibrated against the live cadence) ─────────────────
def test_blind_when_empty_rate_high_and_event_imminent():
    """The headline scenario: NVDA reports in 16h, 60 cyc/hr but Claude
    empty 80% of the time → expected = 60 × 16 × 0.2 = 192 ... that's
    actually READY. So pick numbers that *do* fall under the BLIND bar:
    low cycles/hr × short horizon × high empty-rate."""
    # 6 cycles in last 6h = 1 cyc/hr; 5 of them empty → empty_rate ≈ 0.83.
    decs = ([_dec(i * 60.0, "NO_DECISION") for i in range(5)]
            + [_dec(310.0, "HOLD NVDA -> HOLD")])
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 0.5)],   # 12h out
        now=NOW,
    )
    assert rep["worst_verdict"] == "BLIND"
    assert rep["n_events"] == 1
    ev = rep["events"][0]
    assert ev["ticker"] == "NVDA"
    assert ev["verdict"] == "BLIND"
    # exposure carries through verbatim (2 × 222.35 = 444.70)
    assert ev["exposure_usd"] == pytest.approx(444.70, abs=0.01)
    assert rep["exposure_at_risk_usd"] == pytest.approx(444.70, abs=0.01)
    # the empty-rate hint MUST surface in the action — the chat enrichment
    # depends on this so it doesn't have to re-derive
    assert "empty" in ev["recommended_action"].lower()


def test_ready_when_velocity_high_and_event_far():
    """60 cyc/hr at 10% empty over a 48h event → ~2600 expected → READY.

    Newest cycles must NOT all be NO_DECISION — otherwise the streak
    downgrade fires and we lose READY honestly."""
    # 360 cycles in 6h = 60/hr; ~36 NO_DECISION scattered (~10% empty).
    decs = []
    for i in range(360):
        # Sprinkle NO_DECISION at every 10th slot — keeps the streak
        # at most 9 (well under _STREAK_DEGRADE_CYCLES=10).
        action = ("NO_DECISION" if i % 10 == 5
                  else "HOLD NVDA -> HOLD")
        decs.append(_dec(i * 1.0, action))
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 2.0)],   # 48h out
        now=NOW,
    )
    assert rep["worst_verdict"] == "READY"
    assert rep["events"][0]["verdict"] == "READY"
    # nothing at risk — exposure_at_risk_usd only counts BLIND/DEGRADED rows
    assert rep["exposure_at_risk_usd"] == 0.0


def test_degraded_mid_band():
    """Sit in the mid band: enough velocity to land *some* decisions but
    not enough to be comfortable. expected ∈ [5, 30) → DEGRADED.

    Setup: 6 cyc/hr × 4h × 0.5 success = 12.0 expected → DEGRADED.
    Streak must be short (else BLIND downgrade kicks in)."""
    decs = []
    # 36 cycles across last 6h = 6/hr; alternating NO_DECISION + HOLD →
    # 50% empty. Newest is HOLD so streak = 0.
    for i in range(36):
        # newest (i=0) HOLD, i=1 NO_DECISION, i=2 HOLD, ... → max streak 1
        action = "HOLD NVDA -> HOLD" if i % 2 == 0 else "NO_DECISION"
        decs.append(_dec(i * 10.0, action))
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 4.0 / 24.0)],   # 4h out
        now=NOW,
    )
    assert rep["events"][0]["verdict"] == "DEGRADED"
    assert (_BLIND_MIN := 5.0) <= rep["events"][0][
        "expected_decisions_before_event"] < (_DEGRADED_MAX := 30.0)


# ── degenerate inputs collapse honestly, never raise ──────────────────────
def test_no_decisions_window_collapses_to_blind_signal():
    """If the bot hasn't been making cycles at all in the velocity window,
    *every* imminent event is BLIND — we can't expect decisions if none
    are happening. (Bot down / wedged supervisor scenario.)"""
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=[_dec(60 * 24 * 7, "HOLD NVDA -> HOLD")],   # 7d ago
        earnings_events=[_event("NVDA", 0.5)],
        now=NOW,
    )
    assert rep["events"][0]["verdict"] == "BLIND"
    assert rep["events"][0]["expected_decisions_before_event"] == 0.0


def test_no_events_means_readiness_is_moot():
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=[_dec(1.0)],
        earnings_events=[],
        now=NOW,
    )
    assert rep["worst_verdict"] == "NO_EVENTS"
    assert rep["n_events"] == 0
    assert rep["exposure_at_risk_usd"] == 0.0


def test_no_decisions_at_all_means_no_decisions_verdict():
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=[],
        earnings_events=[],
        now=NOW,
    )
    assert rep["worst_verdict"] == "NO_DECISIONS"


def test_event_outside_horizon_is_filtered():
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=[_dec(1.0)],
        earnings_events=[_event("NVDA", 30.0)],   # 30d out
        now=NOW,
    )
    assert rep["n_events"] == 0
    assert rep["worst_verdict"] == "NO_EVENTS"


def test_event_for_unheld_ticker_is_filtered():
    """This is held-readiness, not the full calendar — matches
    /api/earnings-risk's exposure-only filter."""
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=[_dec(1.0)],
        earnings_events=[_event("MRVL", 0.5)],
        now=NOW,
    )
    assert rep["n_events"] == 0


def test_options_use_100x_multiplier():
    rep = build_event_readiness(
        positions=[_pos("NVDA", qty=1.0, price=5.0, ptype="call")],
        decisions=[_dec(1.0)],
        earnings_events=[_event("NVDA", 0.5)],
        now=NOW,
    )
    # 1 × 5 × 100 = 500
    assert rep["events"][0]["exposure_usd"] == pytest.approx(500.0, abs=0.01)


@pytest.mark.parametrize("decs,evs,pos", [
    ([{"timestamp": None, "action_taken": None}],
     [_event("NVDA", 0.5)], [_pos("NVDA")]),
    ([_dec(1.0)],
     [None, {}, {"ticker": "", "days_away": 0.5},
      {"ticker": "NVDA", "days_away": "soon"}],
     [_pos("NVDA")]),
    ([_dec(1.0)],
     [_event("NVDA", 0.5)],
     [{"ticker": None}, {"qty": None}, {}]),
])
def test_malformed_inputs_do_not_raise(decs, evs, pos):
    """Pure/total: garbage rows downgrade to silence, never raise into
    the Flask handler."""
    rep = build_event_readiness(pos, decs, evs, now=NOW)
    assert isinstance(rep, dict)
    assert isinstance(rep["events"], list)
    assert rep["worst_verdict"] in {
        "BLIND", "DEGRADED", "READY",
        "IMMINENT_OVERDUE", "NO_EVENTS", "NO_DECISIONS",
    }


# ── velocity helper ───────────────────────────────────────────────────────
def test_velocity_only_counts_cycles_in_window():
    decs = ([_dec(60.0)] * 5                 # 5 cycles in last hour
            + [_dec(60.0 * 24)] * 100)       # 100 way outside the 6h window
    v = _decision_velocity(decs, NOW)
    assert v["n_cycles"] == 5
    assert v["cycles_per_hour"] == pytest.approx(5 / 6.0, abs=0.01)


def test_velocity_classifies_no_decision_correctly():
    """``"NO_DECISION"`` and bare-empty are both bucketed as empty —
    matches analytics/decision_drought._classify (SSOT)."""
    decs = [_dec(1.0, "NO_DECISION"), _dec(2.0, ""),
            _dec(3.0, "HOLD NVDA -> HOLD"), _dec(4.0, "BUY NVDA -> FILLED")]
    v = _decision_velocity(decs, NOW)
    assert v["n_cycles"] == 4
    assert v["n_empty"] == 2
    assert v["empty_rate"] == 0.5


# ── _expected_decisions guards ───────────────────────────────────────────
def test_expected_decisions_floors_at_zero():
    assert _expected_decisions({"cycles_per_hour": 10.0, "empty_rate": 1.0},
                               hours_until_event=1.0) == 0.0
    assert _expected_decisions({"cycles_per_hour": 10.0, "empty_rate": 0.5},
                               hours_until_event=-1.0) == 0.0


def test_imminent_overdue_verdict():
    """An event whose time has already passed — flag it explicitly so the
    operator notices a stale calendar."""
    assert _verdict_for(expected=999.0, hours_until=-0.5) == "IMMINENT_OVERDUE"


# ── current-streak downgrade (the live PARALYSIS regime) ─────────────────
def test_streak_downgrades_ready_to_blind():
    """The exact live failure mode /api/decision-drought catches as
    PARALYSIS: 6h average looks OK, but the newest 30 cycles are all
    NO_DECISION. A pure expected-decisions verdict says READY — but the
    bot is *currently stuck*, and that's what readiness must surface."""
    # 60 cyc/hr × 24h × 50% success = 720 expected → would be READY.
    # But the newest 30 cycles are NO_DECISION → streak=30 → BLIND override.
    decs = []
    for i in range(30):                                       # newest 30
        decs.append(_dec(i * 1.0, "NO_DECISION"))
    for i in range(30, 360):                                  # older 330
        action = "HOLD NVDA -> HOLD" if i % 2 == 0 else "NO_DECISION"
        decs.append(_dec(i * 1.0, action))
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 1.0)],   # 24h out
        now=NOW,
    )
    # the expected-decisions math would have said READY; the streak made
    # it BLIND. Verify both readings are visible.
    ev = rep["events"][0]
    assert ev["verdict"] == "BLIND"
    assert ev["base_verdict"] in ("READY", "DEGRADED")        # before downgrade
    assert ev["expected_decisions_before_event"] > 30.0       # would have been READY
    assert rep["velocity"]["current_streak_no_decision"] >= 30
    assert rep["velocity"]["minutes_since_last_real_decision"] is not None
    # the action MUST surface "PARALYSIS" so a chat enrichment can lift it
    assert "PARALYSIS" in ev["recommended_action"]


def test_streak_downgrades_ready_to_degraded():
    """Streak between 10 and 19 cycles drops a clean READY to DEGRADED."""
    decs = []
    for i in range(12):
        decs.append(_dec(i * 1.0, "NO_DECISION"))
    for i in range(12, 360):
        action = "HOLD NVDA -> HOLD" if i % 2 == 0 else "NO_DECISION"
        decs.append(_dec(i * 1.0, action))
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 1.0)],
        now=NOW,
    )
    ev = rep["events"][0]
    assert ev["verdict"] == "DEGRADED"
    assert ev["base_verdict"] == "READY"
    assert 10 <= rep["velocity"]["current_streak_no_decision"] < 20


def test_streak_velocity_metrics_have_minutes_since_real_decision():
    """The chat enrichment formats minutes_since_last_real_decision into
    'Xm since last real decision' — must be a positive number when a
    real decision exists ahead of the streak."""
    decs = ([_dec(1.0, "NO_DECISION")] * 25
            + [_dec(30.0, "HOLD NVDA -> HOLD")])
    rep = build_event_readiness(
        positions=[_pos("NVDA")],
        decisions=decs,
        earnings_events=[_event("NVDA", 0.5)],
        now=NOW,
    )
    v = rep["velocity"]
    assert v["current_streak_no_decision"] == 25
    assert v["minutes_since_last_real_decision"] is not None
    assert v["minutes_since_last_real_decision"] > 0
