"""Pure-helper tests for the /api/chat intent-followthrough enrichment.

`_intent_followthrough_chat_lines` renders paper-trader's
`/api/intent-followthrough` (the observational say-do gap detector that
grades whether the bot actually executes its own STANDING conditional
intents) into compact chat-context lines so the analyst can answer
"is the bot following through, or just talking?".

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_cash_redeployment_chat_lines` / `_macro_calendar_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **disciplined desk = silence**: DISCIPLINED / NO_DATA / NO_RESOLVED /
  ERROR collapse to `[]`, matching the `_decision_paralysis_chat_lines`
  silence precedent — chat must not carry "intent followthrough fine" filler.
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

from dashboard.web_server import _intent_followthrough_chat_lines


def _rep(verdict="DRIFTING", *, headline=None, n_followed=3, n_abandoned=4,
         n_pending=1, rate=0.43, preserve_dead=0, restraint_broken=0):
    if headline is None:
        headline = (
            f"{n_followed}/{n_followed + n_abandoned} followed, "
            f"{n_abandoned} abandoned ({100 * rate:.0f}% followthrough — "
            "drifting)")
    return {
        "as_of": "2026-05-24T12:00:00+00:00",
        "state": "OK",
        "verdict": verdict,
        "headline": headline,
        "n_intents": n_followed + n_abandoned + n_pending,
        "n_actionable": n_followed + n_abandoned + n_pending,
        "n_followed": n_followed,
        "n_pending": n_pending,
        "n_abandoned": n_abandoned,
        "followthrough_rate": rate,
        "abstention": {
            "n_preserve_deployed": 0,
            "n_preserve_active": 0,
            "n_preserve_dead": preserve_dead,
            "n_restraint_held": 0,
            "n_restraint_broken": restraint_broken,
        },
        "by_kind": {},
        "intents": [],
        "window_hours": 24.0,
        "stale_hours": 12.0,
        "eval_window_hours": 12.0,
        "discipline_floor": 0.66,
        "drifting_floor": 0.33,
        "abandoned_min_n": 3,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _intent_followthrough_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _intent_followthrough_chat_lines({}) == []


def test_missing_verdict_is_silence():
    assert _intent_followthrough_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize("verdict",
                         ["DISCIPLINED", "NO_DATA", "NO_RESOLVED",
                          "ERROR", "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _intent_followthrough_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim_drifting():
    custom = ("2/7 followed, 5 abandoned (29% followthrough — drifting)")
    out = _intent_followthrough_chat_lines(_rep(headline=custom,
                                                 n_followed=2,
                                                 n_abandoned=5,
                                                 rate=0.286))
    assert out[0] == custom              # exact char-for-char passthrough


def test_headline_passes_through_verbatim_abandoned():
    custom = ("0/6 followed; 6 abandoned (0% followthrough — bot states "
              "plans it does not execute)")
    out = _intent_followthrough_chat_lines(_rep(verdict="ABANDONED",
                                                 headline=custom,
                                                 n_followed=0,
                                                 n_abandoned=6,
                                                 rate=0.0))
    assert out[0] == custom


# ── actionable verdicts emit headline + detail ──────────────────────────
def test_drifting_emits_headline_plus_detail():
    out = _intent_followthrough_chat_lines(_rep(verdict="DRIFTING"))
    assert len(out) >= 2  # headline + at least one detail line
    assert "followed" in out[1].lower() or "abandoned" in out[1].lower()


def test_abandoned_emits_headline_plus_detail():
    out = _intent_followthrough_chat_lines(_rep(verdict="ABANDONED",
                                                 n_followed=0,
                                                 n_abandoned=5,
                                                 rate=0.0))
    assert len(out) >= 2
    assert "abandoned 5" in out[1]


def test_detail_shows_followthrough_rate():
    out = _intent_followthrough_chat_lines(_rep(verdict="DRIFTING",
                                                 n_followed=2,
                                                 n_abandoned=6,
                                                 rate=0.25))
    detail = out[1] if len(out) > 1 else ""
    assert "25%" in detail


def test_detail_includes_pending_when_present():
    out = _intent_followthrough_chat_lines(_rep(verdict="DRIFTING",
                                                 n_pending=4))
    detail = out[1] if len(out) > 1 else ""
    assert "4 pending" in detail


def test_detail_omits_pending_when_zero():
    out = _intent_followthrough_chat_lines(_rep(verdict="DRIFTING",
                                                 n_pending=0))
    detail = out[1] if len(out) > 1 else ""
    assert "pending" not in detail


def test_detail_surfaces_preserve_dead_abstention():
    out = _intent_followthrough_chat_lines(_rep(verdict="ABANDONED",
                                                 preserve_dead=3))
    detail = out[1] if len(out) > 1 else ""
    assert "dry-powder dead-weight" in detail


def test_detail_surfaces_restraint_broken_abstention():
    out = _intent_followthrough_chat_lines(_rep(verdict="DRIFTING",
                                                 restraint_broken=2))
    detail = out[1] if len(out) > 1 else ""
    assert "restraint broken" in detail


# ── defensive degradation on garbage fields ────────────────────────────
def test_garbage_count_fields_do_not_raise():
    bad = _rep(verdict="DRIFTING")
    bad["n_followed"] = "not a number"
    bad["n_abandoned"] = None
    bad["followthrough_rate"] = "x"
    bad["abstention"] = "not a dict"
    out = _intent_followthrough_chat_lines(bad)
    # Headline still surfaces; detail may collapse to nothing.
    assert isinstance(out, list)
    assert len(out) >= 1
    assert out[0] == bad["headline"]


def test_missing_headline_still_emits_detail():
    rep = _rep(verdict="DRIFTING")
    rep["headline"] = None
    out = _intent_followthrough_chat_lines(rep)
    # No headline → only the detail line (when computable).
    assert isinstance(out, list)
    # If a detail line was computable, it should be the first element.
    if out:
        assert "followthrough" in out[0].lower() or "followed" in out[0].lower()


def test_blank_headline_collapses_to_detail_only():
    rep = _rep(verdict="DRIFTING")
    rep["headline"] = "   "
    out = _intent_followthrough_chat_lines(rep)
    assert all("   " not in line for line in out)  # blank stripped
