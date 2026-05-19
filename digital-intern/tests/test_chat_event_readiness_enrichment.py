"""Pure-helper tests for the /api/chat event-readiness enrichment.

`_event_readiness_chat_lines` renders paper-trader's `/api/event-readiness`
(will the live trader actually be able to react before the next earnings
print?) into compact chat-context lines, so the analyst can answer
"is the book at risk?" without ignoring the prior question
"can the bot act in time?" — the live PARALYSIS failure mode silently
breaks the assumption every other chat block makes that the bot will
decide before the print.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_tail_risk_chat_lines` / `_macro_calendar_chat_lines`
/ `_earnings_shock_chat_lines`) the logic is a total/pure function unit-
tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `summary` string passes through UNCHANGED as the headline — no
  chat-side re-derived verdict that could drift from the trader endpoint.
  Likewise each event's recommended_action passes through verbatim.
- **healthy pipeline = silence**: READY / NO_EVENTS / NO_DECISIONS verdicts
  all collapse to `[]`, exactly as `_macro_calendar_chat_lines` /
  `_behavioural_chat_lines` omit non-actionable states — a chat must not
  carry "event-readiness fine" filler.
- **pure/total**: non-dict / missing keys / a malformed event row never
  raise and degrade to silence or the safe subset (the
  `_paper_trader_position_lines` precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _event_readiness_chat_lines


def _ev(ticker="NVDA", verdict="DEGRADED", hours=15.84, exposure=444.7,
        action="thin decision budget — monitor closely"):
    return {
        "ticker": ticker,
        "verdict": verdict,
        "base_verdict": verdict,
        "hours_until_event": hours,
        "exposure_usd": exposure,
        "expected_decisions_before_event": 21.0,
        "earnings_date": "2026-05-20T00:00:00+00:00",
        "days_away": hours / 24.0,
        "recommended_action": action,
    }


def _rep(worst="DEGRADED", events=None, summary=None):
    if events is None:
        events = [_ev()]
    if summary is None:
        summary = ("DEGRADED for 1 event(s), $445 at risk — "
                   "thin decision budget before the print(s)")
    return {
        "as_of": "2026-05-19T08:00:00+00:00",
        "horizon_days": 3.0,
        "velocity": {
            "n_cycles": 39, "n_empty": 31, "cycles_per_hour": 6.5,
            "empty_rate": 0.795, "current_streak_no_decision": 6,
            "minutes_since_last_real_decision": 41.11, "window_hours": 6.0,
        },
        "events": events,
        "n_events": len(events),
        "worst_verdict": worst,
        "exposure_at_risk_usd": sum(
            (e.get("exposure_usd") or 0)
            for e in events
            if isinstance(e, dict)
            and e.get("verdict") in {"BLIND", "DEGRADED"}),
        "summary": summary,
        "source_ok": True,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _event_readiness_chat_lines(bad) == []


# ── healthy pipeline collapses to silence ───────────────────────────────
def test_ready_is_silence():
    """READY = the pipeline is fine; chat must NOT carry "readiness OK"
    filler — that's the _macro_calendar_chat_lines / _behavioural_chat_
    lines silence precedent."""
    assert _event_readiness_chat_lines(_rep(worst="READY")) == []


def test_no_events_is_silence():
    """No held positions report within horizon → readiness is moot."""
    assert _event_readiness_chat_lines(_rep(worst="NO_EVENTS",
                                            events=[])) == []


def test_no_decisions_is_silence():
    """Bot has logged no recent cycles — readiness can't be computed.
    Silence here defers to the surrounding /api/decision-drought block
    which already screams about it."""
    assert _event_readiness_chat_lines(_rep(worst="NO_DECISIONS",
                                            events=[])) == []


def test_no_events_list_is_silence():
    """Missing / empty events list with any actionable worst_verdict still
    collapses to silence — without events there's nothing to surface."""
    assert _event_readiness_chat_lines(_rep(worst="BLIND",
                                            events=[])) == []


# ── SSOT: the builder's own summary is the verbatim headline ─────────────
def test_degraded_headline_is_builder_summary_verbatim():
    """A chat-side re-derived verdict that drifts from /api/event-readiness
    fails here — invariant #10."""
    rep = _rep(worst="DEGRADED",
               summary="DEGRADED for 1 event(s), $445 at risk — thin budget")
    out = _event_readiness_chat_lines(rep)
    assert out, "a DEGRADED event must surface"
    assert out[0] == rep["summary"]            # byte-identical SSOT


def test_blind_headline_is_builder_summary_verbatim():
    rep = _rep(worst="BLIND",
               summary="BLIND for 1 event(s), $445 at risk — bot stuck",
               events=[_ev(verdict="BLIND",
                           action="ACTIVE PARALYSIS streak: 25 NO_DECISION cycles")])
    out = _event_readiness_chat_lines(rep)
    assert out[0] == rep["summary"]


# ── per-event line restates builder fields, never recomputed ─────────────
def test_event_line_restates_builder_fields():
    """Per-row line restates the builder's own ticker / exposure_usd /
    hours_until_event / verdict / recommended_action — a restatement
    (the earnings_block precedent), never a recomputation."""
    rep = _rep(events=[_ev(ticker="NVDA", verdict="DEGRADED", hours=15.84,
                           exposure=444.7,
                           action="Claude-empty 79% — monitor closely")])
    blob = "\n".join(_event_readiness_chat_lines(rep))
    assert "NVDA" in blob
    assert "DEGRADED" in blob
    assert "15.8" in blob                                # hours_until_event
    assert "$445" in blob                                # exposure rounded
    # recommended_action passes through verbatim (the builder is SSOT for it)
    assert "Claude-empty 79%" in blob


def test_recommended_action_is_verbatim():
    """A chat-side re-derived recommendation that drifts from the builder
    fails here — invariant #10 also covers the per-event action."""
    rep = _rep(events=[_ev(action="ACTIVE PARALYSIS streak: 25 NO_DECISION cycles "
                           "(41m since last real decision) — restart paper-trader")])
    blob = "\n".join(_event_readiness_chat_lines(rep))
    assert ("ACTIVE PARALYSIS streak: 25 NO_DECISION cycles "
            "(41m since last real decision) — restart paper-trader") in blob


# ── multi-event filtering ────────────────────────────────────────────────
def test_only_actionable_events_surface():
    """A READY event mixed in with a DEGRADED one must NOT show up; the
    chat carries only the rows that need operator attention."""
    rep = _rep(
        worst="DEGRADED",
        events=[_ev(ticker="NVDA", verdict="DEGRADED"),
                _ev(ticker="MRVL", verdict="READY", exposure=0.0)])
    blob = "\n".join(_event_readiness_chat_lines(rep))
    assert "NVDA" in blob
    assert "MRVL" not in blob                            # READY is silence


def test_imminent_overdue_surfaces():
    """An OVERDUE event (calendar stale) is actionable — surface it."""
    rep = _rep(
        worst="IMMINENT_OVERDUE",
        summary="event time has passed for at least one held name — "
                "calendar may be stale",
        events=[_ev(ticker="NVDA", verdict="IMMINENT_OVERDUE", hours=-0.5,
                    action="event time has passed — verify earnings "
                           "calendar freshness")])
    out = _event_readiness_chat_lines(rep)
    assert out
    assert any("IMMINENT_OVERDUE" in ln for ln in out)


# ── malformed rows degrade to safe subset, never raise ───────────────────
def test_malformed_event_row_is_skipped():
    rep = _rep(events=[
        None,
        "garbage",
        {},
        _ev(verdict="DEGRADED"),
        {"ticker": "AMD", "verdict": None},
    ])
    out = _event_readiness_chat_lines(rep)
    blob = "\n".join(out)
    assert "NVDA" in blob
    assert "AMD" not in blob


def test_missing_hours_falls_back_to_imminent_label():
    rep = _rep(events=[_ev()])
    rep["events"][0].pop("hours_until_event")
    out = _event_readiness_chat_lines(rep)
    assert any("imminent" in ln for ln in out)


def test_no_summary_still_emits_event_lines():
    """A builder that returns events but no summary (defensive) still
    surfaces the per-event rows — the chat sees less context but is not
    empty."""
    rep = _rep()
    rep["summary"] = None
    out = _event_readiness_chat_lines(rep)
    assert out
    # no summary → no headline → first line is the event line
    assert "NVDA" in out[0]
