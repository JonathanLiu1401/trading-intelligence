"""Pure-helper tests for the /api/chat FOMC-awareness enrichment.

`_macro_calendar_chat_lines` renders paper-trader's `/api/macro-calendar`
(the forward FOMC rate-decision awareness block already fed into the live
trader's own decision prompt) into compact chat-context lines, so the
analyst can answer "is a rate decision about to move the whole book?" — the
single biggest market-wide event for a leveraged-ETF-heavy watchlist. The
chat had rich BACKWARD analytics but zero FORWARD macro-event awareness;
this closes that gap, exactly as `_baseline_compare_chat_lines` closed the
ML-gate-honesty gap.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_tail_risk_chat_lines` / `_baseline_compare_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:
- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `summary` string passes through UNCHANGED as the headline — no
  chat-side re-derived verdict that could drift from the trader endpoint.
- **no-FOMC is silence, not noise**: the builder sets `events: []` for every
  non-actionable case (no FOMC in horizon, schedule-not-loaded, builder
  error). All collapse to `[]` — a chat must not carry "no FOMC within 14d"
  filler, mirroring how `_behavioural_chat_lines` omits NO_DATA.
- **pure/total**: non-dict / missing keys / a malformed event row never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _macro_calendar_chat_lines


def _imminent_event(**over) -> dict:
    e = {
        "event": "FOMC",
        "label": "FOMC rate-decision statement",
        "when_utc": "2026-06-17T18:00:00+00:00",
        "when_et": "2026-06-17 14:00 ET",
        "hours_away": 33.6,
        "days_away": 1.4,
        "tier": "IMMINENT",
    }
    e.update(over)
    return e


def _rep(events=None, summary="FOMC in 1.4d (IMMINENT)", **over) -> dict:
    d = {
        "as_of": "2026-06-16T04:24:00+00:00",
        "summary": summary,
        "prompt_block": "MACRO CALENDAR (...)\n  FOMC rate-decision statement in 1.4d",
        "events": [_imminent_event()] if events is None else events,
        "source_ok": True,
        "schedule_valid_through": "2026-12-09T19:00:00+00:00",
    }
    d.update(over)
    return d


# ── pure/total contract ────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _macro_calendar_chat_lines(bad) == []


# ── no-FOMC / error / not-loaded all collapse to silence ───────────────
def test_no_events_is_silence():
    """Every non-actionable builder branch sets events:[] — the chat must
    not carry "no FOMC within 14d" / "macro calendar error" filler."""
    assert _macro_calendar_chat_lines({}) == []
    assert _macro_calendar_chat_lines({"events": []}) == []
    assert _macro_calendar_chat_lines({"events": None}) == []
    # the real no-FOMC payload (summary present, but no events)
    assert _macro_calendar_chat_lines(
        _rep(events=[], summary="no FOMC within 14d")) == []
    # the real builder-error payload
    assert _macro_calendar_chat_lines(
        _rep(events=[], summary="macro calendar error",
             source_ok=False, error="boom")) == []
    # the real schedule-not-loaded payload
    assert _macro_calendar_chat_lines(
        _rep(events=[], summary="no macro schedule loaded",
             source_ok=False)) == []


# ── SSOT: the builder's own summary is the verbatim headline ───────────
def test_imminent_headline_is_builder_summary_verbatim():
    """A chat-side re-derived verdict that drifts from /api/macro-calendar
    fails here — invariant #10."""
    rep = _rep(summary="FOMC in 1.4d (IMMINENT)")
    out = _macro_calendar_chat_lines(rep)
    assert out, "an imminent FOMC must surface"
    assert out[0] == rep["summary"]            # byte-identical SSOT


def test_event_detail_restates_builder_fields_not_recomputed():
    """Detail line restates the builder's own when_et / tier / day-timing —
    a restatement (the earnings_block precedent), never a recomputation."""
    rep = _rep(events=[_imminent_event(when_et="2026-06-17 14:00 ET",
                                       days_away=1.4, tier="IMMINENT")])
    blob = "\n".join(_macro_calendar_chat_lines(rep))
    assert "2026-06-17 14:00 ET" in blob
    assert "IMMINENT" in blob
    assert "1.4" in blob                       # the builder's days_away


def test_imminent_hours_uses_hours_not_days():
    """Within 24h the builder tiers IMMINENT_HOURS and the operator needs
    the HOUR figure (a day figure rounds a 6h-away decision to 0.2d)."""
    rep = _rep(
        summary="FOMC in 6.0h (IMMINENT_HOURS)",
        events=[_imminent_event(hours_away=6.0, days_away=0.25,
                                tier="IMMINENT_HOURS")])
    out = _macro_calendar_chat_lines(rep)
    assert out[0] == "FOMC in 6.0h (IMMINENT_HOURS)"
    blob = "\n".join(out)
    assert "6.0h" in blob
    assert "IMMINENT_HOURS" in blob
    # the day figure (0.2d / 0.25d) must NOT be the timing shown for an
    # hours-tier event — that would understate the imminence
    assert "0.2d" not in blob and "0.25d" not in blob


def test_multiple_events_each_get_a_line():
    rep = _rep(events=[
        _imminent_event(when_et="2026-06-17 14:00 ET", tier="IMMINENT"),
        _imminent_event(when_utc="2026-07-29T18:00:00+00:00",
                        when_et="2026-07-29 14:00 ET",
                        hours_away=1020.0, days_away=42.5, tier="UPCOMING"),
    ])
    out = _macro_calendar_chat_lines(rep)
    blob = "\n".join(out)
    assert "2026-06-17 14:00 ET" in blob
    assert "2026-07-29 14:00 ET" in blob


# ── partial / malformed never raises ───────────────────────────────────
def test_events_present_but_summary_missing_still_emits_no_raise():
    """A degraded payload (events present, summary missing/non-str) must
    still surface the event(s) and must NOT fabricate a headline or raise."""
    rep = _rep(events=[_imminent_event()])
    rep.pop("summary")
    out = _macro_calendar_chat_lines(rep)
    assert out, "events must still surface when summary is absent"
    assert "2026-06-17 14:00 ET" in "\n".join(out)

    rep2 = _rep(summary=None)
    out2 = _macro_calendar_chat_lines(rep2)
    assert out2 and "IMMINENT" in "\n".join(out2)


def test_malformed_event_row_is_skipped_never_raises():
    rep = _rep(events=["not-a-dict", None, 42, _imminent_event()])
    out = _macro_calendar_chat_lines(rep)        # must not raise
    assert any("2026-06-17 14:00 ET" in ln for ln in out)


def test_all_event_rows_malformed_degrades_to_headline_or_silence():
    """If every row is junk there is nothing to restate; the helper must
    not raise. With a real summary the SSOT headline still stands; without
    one it is silence."""
    assert _macro_calendar_chat_lines(
        _rep(events=["x", None]))[0] == "FOMC in 1.4d (IMMINENT)"
    assert _macro_calendar_chat_lines(
        _rep(events=["x", None], summary=None)) == []
