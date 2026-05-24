"""Pure-helper tests for the /api/chat standing-intents enrichment.

`_standing_intents_chat_lines` renders paper-trader's
`/api/decision-conditionals` (STANDING conditional intents extracted from
recent decisions' reasoning) into compact chat-context lines. Answers
the forward-looking operator question no other reasoning block answers:
"what did the bot SAY it would do next, that it has not yet done?"

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_thesis_drift_chat_lines` / `_cash_redeployment_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED, and each
  surfaced intent's `text` field passes through verbatim (the
  ``_thesis_drift_chat_lines`` drift_reasons-verbatim precedent).
- **healthy = silence**: NO_INTENTS / NO_DATA collapse to `[]`, matching
  the ``_decision_paralysis_chat_lines`` silence precedent — never chat
  filler when the bot is reasoning without forward commitments.
- **STALE_INTENTS tags `[stale]` per-row** so the operator can see plans
  that aged out without action at a glance.
- **cap at 3 intents** to keep the chat block bounded — the builder
  itself caps deeper, this helper just shows the freshest slice.
- **pure/total**: non-dict / missing keys / unparseable rows never raise
  and degrade silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _standing_intents_chat_lines


def _intent(kind="watch-for", ticker="NVDA", text="wait for cash session to reassess",
            age_hours=0.5, stale=False):
    return {
        "decision_id": 100,
        "decision_ts": "2026-05-24T01:00:00+00:00",
        "ticker": ticker,
        "kind": kind,
        "text": text,
        "age_hours": age_hours,
        "stale": stale,
        "action_taken": "HOLD NVDA → HOLD",
    }


def _rep(verdict="STANDING_INTENTS", *, headline=None, intents=None,
         n_intents=None, n_stale=0):
    if intents is None:
        intents = [_intent()]
    if n_intents is None:
        n_intents = len(intents)
    if headline is None:
        headline = (
            f"{n_intents} standing intent(s) across 1 ticker(s): "
            f"{n_intents} watch-for"
        )
    return {
        "as_of": "2026-05-24T01:30:00+00:00",
        "state": "OK",
        "verdict": verdict,
        "headline": headline,
        "n_decisions_scanned": 50,
        "n_intents_raw": n_intents,
        "n_intents": n_intents,
        "n_stale": n_stale,
        "intents": intents,
        "by_kind": {"watch-for": n_intents},
        "window_hours": 24.0,
        "stale_hours": 12.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Silence precedent — non-actionable verdicts collapse to []
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verdict", ["NO_DATA", "NO_INTENTS", "UNKNOWN", None])
def test_silence_on_non_actionable_verdicts(verdict):
    rep = _rep(verdict=verdict)
    assert _standing_intents_chat_lines(rep) == []


def test_silence_on_missing_verdict():
    rep = _rep()
    rep.pop("verdict")
    assert _standing_intents_chat_lines(rep) == []


# ─────────────────────────────────────────────────────────────────────
# Defensive — non-dict / garbage degrades silently
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rep", [None, "string", 123, [], 0.5, True])
def test_silence_on_non_dict(rep):
    assert _standing_intents_chat_lines(rep) == []


def test_silence_on_garbage_intents_field():
    rep = _rep()
    rep["intents"] = "not-a-list"
    # The headline still appears; degrades silently on the rest.
    lines = _standing_intents_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("Standing intents:")


def test_garbage_intent_rows_skipped_not_raised():
    rep = _rep(intents=[
        None,
        {},                                                # missing required fields
        {"kind": 123, "text": "x"},                        # wrong type
        {"kind": "watch-for", "text": ""},                 # empty text
        _intent(),                                         # one valid row
    ])
    lines = _standing_intents_chat_lines(rep)
    # Headline + the one valid intent line.
    assert len(lines) == 2
    assert lines[1].startswith("  •")


# ─────────────────────────────────────────────────────────────────────
# Verbatim SSOT — headline + intent text pass through unchanged
# ─────────────────────────────────────────────────────────────────────

def test_headline_passes_through_verbatim():
    custom = "3 standing intent(s) across 2 ticker(s): 2 watch-for, 1 if-then"
    rep = _rep(headline=custom)
    lines = _standing_intents_chat_lines(rep)
    assert lines[0] == f"Standing intents: {custom}"


def test_intent_text_passes_through_verbatim():
    custom_text = "Wait for cash session to reassess NVDA price action and MRVL setup"
    rep = _rep(intents=[_intent(text=custom_text)])
    lines = _standing_intents_chat_lines(rep)
    assert any(custom_text in ln for ln in lines)


def test_intent_line_carries_kind_ticker_age():
    rep = _rep(intents=[_intent(
        kind="if-then", ticker="AMD", text="if it holds 200 will add",
        age_hours=2.5, stale=False,
    )])
    lines = _standing_intents_chat_lines(rep)
    assert any("[if-then]" in ln and "AMD" in ln and "2.5h" in ln for ln in lines)


# ─────────────────────────────────────────────────────────────────────
# Stale tagging
# ─────────────────────────────────────────────────────────────────────

def test_stale_intent_tagged():
    rep = _rep(verdict="STALE_INTENTS",
               intents=[_intent(stale=True, age_hours=18.0)],
               n_stale=1)
    lines = _standing_intents_chat_lines(rep)
    assert any("[stale]" in ln for ln in lines)


def test_fresh_intent_not_tagged_stale():
    rep = _rep(intents=[_intent(stale=False, age_hours=0.5)])
    lines = _standing_intents_chat_lines(rep)
    intent_lines = [ln for ln in lines if ln.startswith("  •")]
    assert intent_lines
    assert "[stale]" not in intent_lines[0]


# ─────────────────────────────────────────────────────────────────────
# Cap — at most 3 intents shown
# ─────────────────────────────────────────────────────────────────────

def test_cap_at_three_intents():
    five = [
        _intent(text=f"wait for catalyst {i} to confirm direction", age_hours=i * 0.5)
        for i in range(5)
    ]
    rep = _rep(intents=five, n_intents=5)
    lines = _standing_intents_chat_lines(rep)
    intent_lines = [ln for ln in lines if ln.startswith("  •")]
    assert len(intent_lines) == 3


def test_cap_preserves_order():
    five = [
        _intent(text=f"catalyst {i}", age_hours=i * 0.5)
        for i in range(5)
    ]
    rep = _rep(intents=five, n_intents=5)
    lines = _standing_intents_chat_lines(rep)
    intent_lines = [ln for ln in lines if ln.startswith("  •")]
    # The first three input rows survive.
    for i in range(3):
        assert f"catalyst {i}" in intent_lines[i]


# ─────────────────────────────────────────────────────────────────────
# Both actionable verdicts produce lines
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verdict", ["STANDING_INTENTS", "STALE_INTENTS"])
def test_both_actionable_verdicts_produce_output(verdict):
    rep = _rep(verdict=verdict)
    assert _standing_intents_chat_lines(rep) != []


# ─────────────────────────────────────────────────────────────────────
# Missing-ticker degrades to "—"
# ─────────────────────────────────────────────────────────────────────

def test_missing_ticker_renders_dash():
    rep = _rep(intents=[_intent(ticker=None)])
    lines = _standing_intents_chat_lines(rep)
    intent_lines = [ln for ln in lines if ln.startswith("  •")]
    assert "—" in intent_lines[0]


def test_missing_age_renders_question_mark():
    rep = _rep(intents=[{
        **_intent(),
        "age_hours": None,
    }])
    lines = _standing_intents_chat_lines(rep)
    intent_lines = [ln for ln in lines if ln.startswith("  •")]
    assert "(?)" in intent_lines[0]
