"""Pure-helper tests for the /api/chat idle-opportunity enrichment.

`_idle_opportunity_chat_lines` renders paper-trader's `/api/idle-opportunity`
(high-score watchlist arrivals during the *current* NO_DECISION drought)
into compact chat-context lines so the analyst answering "is the bot
missing anything RIGHT NOW?" sees the loudest live miss without
re-deriving the verdict.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_event_readiness_chat_lines` /
`_decision_paralysis_chat_lines` / `_opportunity_cost_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **healthy loop / honest silence = silence**: NO_DATA / NO_DROUGHT / OK
  with zero opportunities collapse to `[]`, matching the
  `_decision_paralysis_chat_lines` silence precedent. A drought-clear
  chat must not carry "loop is filling" filler.
- **HELD-name tag**: when the top opportunity's row carries ``held=True``,
  the detail line marks it ``(HELD)`` so the analyst can answer "the bot
  was BLIND on a position we own" without re-deriving holdings.
- **pure/total**: non-dict / missing keys / non-numeric counts never raise
  and degrade to silence or a safe partial line (the
  `_decision_paralysis_chat_lines` "garbage counts" precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _idle_opportunity_chat_lines


def _rep(state="OK", *, headline=None, n_opps=1, top_ticker="NVDA",
         top_score=8.0, dur=7.1, n_nd=2, held=False, opportunities=None):
    if headline is None:
        if state == "OK" and n_opps:
            held_tag = " (HELD)" if held else ""
            headline = (
                f"Idle opportunity: drought {dur:.1f}h ({n_nd} NO_DECISION) — "
                f"{n_opps} watchlist signal(s) ≥6.0 arrived; "
                f"loudest: {top_ticker}{held_tag} @ ai_score {top_score:.1f}."
            )
        elif state == "OK":
            headline = (
                f"Idle opportunity: drought {dur:.1f}h ({n_nd} NO_DECISION) — "
                "no live watchlist signals ≥6.0 arrived; the silence is honest."
            )
        elif state == "NO_DROUGHT":
            headline = ("Idle opportunity: no ongoing drought — the trader is "
                        "filling normally; nothing missed by definition.")
        else:
            headline = "Idle opportunity: no decisions recorded yet."

    if opportunities is None:
        opportunities = ([{
            "ticker": top_ticker,
            "top_score": top_score,
            "held": held,
            "article_count": 2,
            "max_urgency": 1,
            "top_title": "t",
            "top_source": "s",
            "top_url": "u",
            "top_first_seen": "2026-05-24T06:10:12+00:00",
        }] if n_opps else [])

    return {
        "as_of": "2026-05-24T09:38:38+00:00",
        "state": state,
        "headline": headline,
        "drought": {
            "duration_hours": dur,
            "n_no_decision": n_nd,
            "n_cycles": 28,
            "n_hold": 26,
            "n_blocked": 0,
            "kind": "DELIBERATE_HOLD",
            "ongoing": True,
            "start": "2026-05-24T02:25:20+00:00",
            "end": "2026-05-24T09:29:27+00:00",
        },
        "min_ai_score": 6.0,
        "n_opportunities": n_opps,
        "opportunities": opportunities,
        "missed_top_score": top_score if n_opps else None,
        "missed_top_ticker": top_ticker if n_opps else None,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _idle_opportunity_chat_lines(bad) == []


def test_missing_state_is_silence():
    assert _idle_opportunity_chat_lines({}) == []
    # state present but bogus → silence
    assert _idle_opportunity_chat_lines({"state": "BOGUS"}) == []


# ── silence verdicts (the healthy/no-regret branches) ───────────────────
@pytest.mark.parametrize("state", ["NO_DATA", "NO_DROUGHT", "ERROR"])
def test_non_actionable_states_silence(state):
    assert _idle_opportunity_chat_lines(_rep(state=state)) == []


def test_ok_with_zero_opportunities_silence():
    # The "silence is honest" branch — drought exists but nothing was
    # missed. Chat must collapse so the analyst doesn't see filler.
    rep = _rep(state="OK", n_opps=0)
    assert _idle_opportunity_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "Idle opportunity: drought 9.3h (3 NO_DECISION) — "
        "2 watchlist signal(s) ≥6.0 arrived; loudest: AMD @ ai_score 9.0.")
    out = _idle_opportunity_chat_lines(
        _rep(state="OK", headline=custom, n_opps=2,
             top_ticker="AMD", top_score=9.0, dur=9.3, n_nd=3))
    assert out[0] == custom              # exact char-for-char passthrough


def test_empty_headline_omits_first_line_but_still_renders_detail():
    rep = _rep(state="OK", headline="")
    out = _idle_opportunity_chat_lines(rep)
    # Empty headline filtered (the `_macro_calendar_chat_lines` precedent),
    # but the detail line still renders since n_opps > 0.
    assert all(not line.startswith("Idle opportunity") for line in out)
    body = "\n".join(out)
    assert "drought" in body
    assert "NVDA" in body


# ── actionable branches (the chat-relevant output) ──────────────────────
def test_ok_with_opps_emits_detail_line_with_drought_and_top():
    out = _idle_opportunity_chat_lines(
        _rep(state="OK", n_opps=1, top_ticker="NVDA", top_score=8.0,
             dur=7.1, n_nd=2))
    assert len(out) == 2
    body = "\n".join(out)
    assert "drought 7.1h" in body
    assert "2 NO_DECISION" in body
    assert "NVDA" in body
    assert "8.0" in body
    # No HELD suffix when the row didn't carry held=True
    assert "(HELD)" not in body


def test_held_top_opportunity_tags_chat_detail():
    # The "bot was BLIND on a position we OWN" path — operator-critical.
    # The chat must call this out distinctly from a watchlist-only miss.
    out = _idle_opportunity_chat_lines(
        _rep(state="OK", n_opps=1, top_ticker="MU", top_score=8.5,
             held=True))
    body = "\n".join(out)
    assert "MU (HELD)" in body


def test_missing_drought_field_does_not_raise():
    rep = _rep(state="OK", n_opps=1, top_ticker="NVDA", top_score=7.0)
    rep["drought"] = None
    out = _idle_opportunity_chat_lines(rep)
    # Headline still emitted; detail line skips the drought fragment but
    # still includes the loudest miss.
    body = "\n".join(out)
    assert "Idle opportunity" in body
    assert "NVDA" in body
    assert "7.0" in body
    assert "drought" not in body.split("\n")[-1]   # detail line skipped it


def test_garbage_numeric_fields_do_not_raise():
    rep = _rep(state="OK", n_opps=1)
    rep["drought"]["duration_hours"] = "bad"
    rep["drought"]["n_no_decision"] = None
    rep["missed_top_score"] = "fine"
    out = _idle_opportunity_chat_lines(rep)
    # At minimum the headline survives.
    assert out and isinstance(out[0], str)


def test_n_opportunities_must_be_positive_int():
    # Float/str/None for n_opportunities all collapse to silence (not a
    # positive integer count of missed signals = not actionable).
    for bad in [0, None, "1", 1.0, True]:
        rep = _rep(state="OK", n_opps=1)
        rep["n_opportunities"] = bad
        out = _idle_opportunity_chat_lines(rep)
        if bad in (0, None, True) or isinstance(bad, (str, float)):
            assert out == [], f"bad={bad!r} should silence"


def test_missing_opportunities_list_does_not_raise():
    rep = _rep(state="OK", n_opps=1)
    rep["opportunities"] = None
    out = _idle_opportunity_chat_lines(rep)
    body = "\n".join(out)
    # Headline + detail (sans HELD suffix) still render
    assert "Idle opportunity" in body or "NVDA" in body


def test_opportunities_list_with_non_dict_top_does_not_raise():
    rep = _rep(state="OK", n_opps=1)
    rep["opportunities"] = ["not-a-dict"]
    out = _idle_opportunity_chat_lines(rep)
    assert out and isinstance(out[0], str)
