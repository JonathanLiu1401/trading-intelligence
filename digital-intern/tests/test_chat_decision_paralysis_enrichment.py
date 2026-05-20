"""Pure-helper tests for the /api/chat decision-paralysis enrichment.

`_decision_paralysis_chat_lines` renders paper-trader's
`/api/decision-paralysis` (consecutive HOLD-only / NO_DECISION streak
detector — the HOLD_LOCK pathology) into compact chat-context lines so
the analyst can answer "should I be doing something?" when the loop is
deciding every cycle but never moving the book.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_event_readiness_chat_lines` /
`_macro_calendar_chat_lines` / `_earnings_shock_chat_lines`) the logic is a
total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **healthy loop = silence**: ACTIVE / NO_DATA verdicts collapse to `[]`,
  matching the `_event_readiness_chat_lines` silence precedent — a chat
  must not carry "decision loop fine" filler.
- **pure/total**: non-dict / missing keys / unparseable counts never raise
  and degrade to silence or the safe subset (the
  `_paper_trader_position_lines` precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _decision_paralysis_chat_lines


def _rep(verdict="HOLD_LOCK", *, headline=None, hold_streak=12,
         passive_streak=12, nd_streak=0, hours_since_active=2.5):
    if headline is None:
        headline = (
            f"HOLD_LOCK — the last {hold_streak} consecutive cycles were "
            "HOLD (no FILLED/BLOCKED for "
            f"{hours_since_active:.1f}h). Opus is deciding every cycle but "
            "never moving the book; the prompt context may be identical "
            "cycle-to-cycle.")
    return {
        "as_of": "2026-05-19T12:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "n_decisions_scanned": 50,
        "n_decisions_24h": 24,
        "current_hold_streak": hold_streak,
        "current_no_decision_streak": nd_streak,
        "current_passive_streak": passive_streak,
        "longest_hold_streak_24h": passive_streak,
        "longest_passive_streak_24h": passive_streak,
        "hold_lock_threshold": 10,
        "idle_storm_threshold": 5,
        "passive_loop_threshold": 15,
        "hours_since_last_active": hours_since_active,
        "last_active_action": "BUY NVDA → FILLED",
        "last_active_ts": "2026-05-19T09:30:00+00:00",
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _decision_paralysis_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _decision_paralysis_chat_lines({}) == []
    assert _decision_paralysis_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize("verdict", ["ACTIVE", "NO_DATA", "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _decision_paralysis_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "HOLD_LOCK — the last 17 consecutive cycles were HOLD (no "
        "FILLED/BLOCKED for 4.4h). Opus is deciding every cycle but never "
        "moving the book; the prompt context may be identical cycle-to-cycle.")
    out = _decision_paralysis_chat_lines(_rep(headline=custom,
                                              hold_streak=17,
                                              passive_streak=17,
                                              hours_since_active=4.4))
    assert out[0] == custom              # exact char-for-char passthrough


# ── per-verdict actionability ───────────────────────────────────────────
def test_hold_lock_emits_detail_line():
    out = _decision_paralysis_chat_lines(
        _rep(verdict="HOLD_LOCK", hold_streak=12, passive_streak=12,
             hours_since_active=2.5))
    assert len(out) == 2
    assert "HOLD streak 12" in out[1]
    assert "2.5h ago" in out[1]
    # No NO_DECISION counter when streak is zero
    assert "NO_DECISION streak" not in out[1]


def test_idle_storm_emits_detail_with_nd_streak():
    out = _decision_paralysis_chat_lines(
        _rep(verdict="IDLE_STORM", hold_streak=0, passive_streak=8,
             nd_streak=8, hours_since_active=1.2))
    body = "\n".join(out)
    assert "IDLE_STORM" in body or "HOLD streak 0" in body
    assert "NO_DECISION streak 8" in body


def test_passive_loop_distinguishes_passive_from_hold():
    # Mixed run: hold_streak < passive_streak (NO_DECISIONs are present in
    # the passive band). Detail line should surface the larger passive
    # streak in addition to the hold streak.
    out = _decision_paralysis_chat_lines(
        _rep(verdict="PASSIVE_LOOP", hold_streak=4, passive_streak=16,
             nd_streak=0, hours_since_active=6.0))
    body = "\n".join(out)
    assert "passive streak 16" in body
    assert "HOLD streak 4" in body


def test_missing_hours_since_active_degrades_silently():
    rep = _rep(verdict="HOLD_LOCK",
               headline="HOLD_LOCK — synthetic (no timing)")
    rep["hours_since_last_active"] = None
    rep.pop("hours_since_last_active", None)
    out = _decision_paralysis_chat_lines(rep)
    # Headline still emitted; detail line may omit the timing fragment but
    # must not raise. Count-based fragments still present.
    body = "\n".join(out)
    assert "HOLD streak" in body
    assert "ago" not in body                    # no timing fragment


def test_garbage_counts_do_not_raise():
    rep = _rep(verdict="HOLD_LOCK")
    rep["current_hold_streak"] = "not-a-number"
    rep["current_passive_streak"] = None
    rep["hours_since_last_active"] = "soonish"
    out = _decision_paralysis_chat_lines(rep)
    # Headline still emitted at minimum.
    assert out and isinstance(out[0], str)


def test_empty_headline_omits_first_line_but_still_renders_detail():
    rep = _rep(verdict="HOLD_LOCK", headline="")
    out = _decision_paralysis_chat_lines(rep)
    # Empty headline filtered (the `_macro_calendar_chat_lines` precedent),
    # but the detail line still renders.
    assert all("HOLD_LOCK —" not in line for line in out)
    assert any("HOLD streak" in line for line in out)


def test_hold_streak_zero_is_omitted_from_detail_when_no_other_signal():
    # Extreme degenerate: HOLD_LOCK verdict but every counter is zero.
    # Should still emit the headline; detail line is fine to be empty.
    rep = _rep(verdict="HOLD_LOCK", headline="HOLD_LOCK — synthetic",
               hold_streak=0, passive_streak=0, hours_since_active=None)
    rep.pop("hours_since_last_active", None)
    out = _decision_paralysis_chat_lines(rep)
    assert out[0] == "HOLD_LOCK — synthetic"
