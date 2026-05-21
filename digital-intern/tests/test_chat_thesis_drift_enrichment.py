"""Pure-helper tests for the /api/chat thesis-drift enrichment.

`_thesis_drift_chat_lines` renders paper-trader's `/api/thesis-drift`
(every open position re-tested against the verbatim reason it was opened
for, graded INTACT / WEAKENING / BROKEN) into compact chat-context
lines. It lets the analyst answer "should the bot have already sold X?"
without re-deriving from raw signals — the answer is already in the
trader endpoint's `drift_reasons` field, surfaced verbatim.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_event_readiness_chat_lines` / `_macro_calendar_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` passes through UNCHANGED and each card's
  ``drift_reasons`` are surfaced verbatim — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **all-INTACT book = silence**: every position INTACT (or no positions)
  collapses to ``[]``, matching the `_decision_paralysis_chat_lines`
  silence precedent — a chat must not carry "all theses fine" filler.
- **pure/total**: non-dict / missing keys / unparseable shapes never
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

from dashboard.web_server import _thesis_drift_chat_lines


def _card(ticker, health, *, pl_pct=None, days_held=None, drift=None,
          typ="stock"):
    return {
        "ticker": ticker,
        "type": typ,
        "health": health,
        "pl_pct": pl_pct,
        "days_held": days_held,
        "drift_reasons": drift or [],
        "entry_reason": "stub entry reason",
    }


def _rep(cards, *, headline="1 open position(s): 1 weakening — NVDA "
                            "thesis weakening (-3.2% since entry)."):
    def _h(card):
        return card.get("health") if isinstance(card, dict) else None
    return {
        "as_of": "2026-05-21T00:00:00+00:00",
        "state": "OK" if cards else "NO_DATA",
        "headline": headline,
        "n_positions": len(cards),
        "counts": {
            "INTACT": sum(1 for c in cards if _h(c) == "INTACT"),
            "WEAKENING": sum(1 for c in cards if _h(c) == "WEAKENING"),
            "BROKEN": sum(1 for c in cards if _h(c) == "BROKEN"),
        },
        "positions": cards,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _thesis_drift_chat_lines(bad) == []


def test_missing_positions_is_silence():
    assert _thesis_drift_chat_lines({}) == []
    assert _thesis_drift_chat_lines({"state": "OK"}) == []
    assert _thesis_drift_chat_lines({"positions": None}) == []


def test_empty_positions_is_silence():
    assert _thesis_drift_chat_lines(_rep([])) == []


def test_no_data_state_is_silence():
    # A NO_DATA report (no open positions) collapses regardless of headline.
    rep = _rep([], headline="No open positions — no entry theses to re-test.")
    assert _thesis_drift_chat_lines(rep) == []


# ── healthy book = silence ──────────────────────────────────────────────
def test_all_intact_book_collapses_to_silence():
    cards = [
        _card("AAA", "INTACT", pl_pct=2.5, days_held=3.0),
        _card("BBB", "INTACT", pl_pct=0.5, days_held=1.5),
    ]
    rep = _rep(cards, headline="2 open position(s): 2 intact — "
                               "all open theses still intact.")
    # An INTACT book is silence — the chat never carries "all theses fine"
    # filler (the decision_paralysis ACTIVE silence precedent).
    assert _thesis_drift_chat_lines(rep) == []


# ── actionable verdicts surface verbatim ───────────────────────────────
def test_single_weakening_surfaces_headline_and_one_detail():
    cards = [
        _card(
            "NVDA", "WEAKENING",
            pl_pct=-3.20, days_held=1.98,
            drift=["P/L since entry -3.20%",
                   "5d momentum -1.05% (negative)"],
        ),
    ]
    rep = _rep(cards)
    lines = _thesis_drift_chat_lines(rep)
    assert len(lines) >= 2
    # First line is the builder's own headline — verbatim SSOT (invariant #10).
    assert lines[0] == rep["headline"]
    # Detail line carries ticker, health verdict, P/L and days_held — restated
    # from the card's OWN fields, never re-derived.
    detail = " ".join(lines[1:])
    assert "NVDA" in detail
    assert "WEAKENING" in detail
    assert "-3.20%" in detail
    assert "1.98d" in detail
    # drift_reasons must surface verbatim so Opus sees the *why*.
    assert "5d momentum -1.05% (negative)" in detail


def test_broken_and_weakening_both_surface_intact_does_not():
    cards = [
        _card("BRK", "BROKEN", pl_pct=-12.0, days_held=4.5,
              drift=["P/L since entry -12.00%"]),
        _card("WK",  "WEAKENING", pl_pct=-4.1, days_held=2.0,
              drift=["RSI 80 — overextended"]),
        _card("INT", "INTACT", pl_pct=1.0, days_held=1.0),
    ]
    rep = _rep(cards, headline="3 open position(s): 1 broken, 1 weakening, "
                               "1 intact — BRK thesis broken.")
    lines = _thesis_drift_chat_lines(rep)
    body = "\n".join(lines)
    assert "BRK" in body
    assert "BROKEN" in body
    assert "WK" in body
    assert "WEAKENING" in body
    # The INTACT sibling MUST NOT leak into chat — silence on the healthy slice.
    assert "INT " not in body
    assert "INTACT" not in body


def test_drift_reasons_passthrough_verbatim_no_rederivation():
    # The chat helper is forbidden from re-deriving the verdict text; the
    # builder's `drift_reasons` are an SSOT pass-through (the
    # _decision_paralysis_chat_lines / _event_readiness_chat_lines precedent).
    verbatim = "made-up-reason-token-not-derivable-by-chat"
    cards = [_card("X", "WEAKENING", pl_pct=-3.5, days_held=1.0,
                    drift=[verbatim, "another reason"])]
    rep = _rep(cards, headline="opaque headline string — verbatim")
    body = "\n".join(_thesis_drift_chat_lines(rep))
    assert verbatim in body
    assert "another reason" in body
    assert "opaque headline string — verbatim" in body


# ── degrade-safe on partial cards (the _paper_trader_position_lines
#     precedent: a missing field degrades, never raises) ────────────────
def test_card_missing_pl_pct_degrades_gracefully():
    cards = [_card("NA", "WEAKENING", pl_pct=None, days_held=1.0,
                    drift=["drift A"])]
    rep = _rep(cards)
    lines = _thesis_drift_chat_lines(rep)
    assert len(lines) >= 2
    detail = " ".join(lines[1:])
    assert "NA" in detail
    assert "WEAKENING" in detail
    # No P/L number when it's missing — the safe-subset render.
    assert "P/L" not in detail or "P/L None" not in detail


def test_card_missing_days_held_degrades_gracefully():
    cards = [_card("ND", "BROKEN", pl_pct=-10.0, days_held=None,
                    drift=["drift B"])]
    rep = _rep(cards)
    body = "\n".join(_thesis_drift_chat_lines(rep))
    assert "ND" in body
    assert "BROKEN" in body
    # Days held silent when missing — never "held Noned".
    assert "held None" not in body


def test_non_dict_card_inside_positions_is_skipped_not_raised():
    cards = [
        _card("OK", "WEAKENING", pl_pct=-3.5, days_held=1.0,
              drift=["drift OK"]),
        "garbage",                              # rogue list element
    ]
    rep = _rep(cards)
    # Total/pure — must not raise. The rogue element is dropped silently.
    body = "\n".join(_thesis_drift_chat_lines(rep))
    assert "OK" in body
    assert "WEAKENING" in body


def test_pure_no_network_required():
    # The contract is pure: no urllib, no sqlite, no /api/* hit. A patched
    # urlopen that errors must NOT be reached.
    import urllib.request as _u
    saved = _u.urlopen

    def _fail(*_a, **_kw):
        raise RuntimeError("helper hit network — pure contract violated")

    _u.urlopen = _fail
    try:
        cards = [_card("X", "WEAKENING", pl_pct=-3.5, days_held=1.0,
                        drift=["drift"])]
        _ = _thesis_drift_chat_lines(_rep(cards))
    finally:
        _u.urlopen = saved
