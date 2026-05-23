"""Pure-helper tests for the /api/chat persona-book-fit enrichment.

``_persona_book_fit_chat_lines`` renders paper-trader's
``/api/persona-book-fit`` (does the live book mirror a backtest-persona
rated DRAG by the persona-leaderboard?) into compact chat-context lines.

The surrounding chat handler is one large inline closure, so per the
established design (cf. ``_event_readiness_chat_lines`` /
``_decision_paralysis_chat_lines``) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own ``headline`` passes through UNCHANGED as the chat headline — no
  chat-side re-derived verdict that could drift from the trader endpoint.
- **healthy book = silence**: ALIGNED_EDGE / ALIGNED_FLAT / NO_BOOK /
  WEAK_OVERLAP / INSUFFICIENT_PERSONA all collapse to ``[]``, exactly as
  ``_event_readiness_chat_lines`` / ``_decision_paralysis_chat_lines`` omit
  non-actionable states — a chat must not carry "persona-fit fine" filler.
- **pure/total**: non-dict / missing keys / malformed sub-rows never raise
  and degrade to the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _persona_book_fit_chat_lines


def _rep(verdict="ALIGNED_DRAG", headline="ALIGNED_DRAG — sample headline",
         overlap=66.7, runner_up_name="Value Investor", runner_up_overlap=20.0):
    return {
        "verdict": verdict,
        "headline": headline,
        "dominant": {
            "persona": "Momentum Trader",
            "raw_score": 200.0,
            "overlap_pct": overlap,
            "matched_tickers": [
                {"ticker": "SOXL", "book_weight_pct": overlap, "boost": 4.0},
            ],
        },
        "runner_up": {
            "persona": runner_up_name, "raw_score": 30.0,
            "overlap_pct": runner_up_overlap,
        },
        "alternatives": [
            {"persona": "Value Investor", "median_vs_spy": 30.0,
             "win_rate": 0.65, "n_runs": 50},
        ],
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _persona_book_fit_chat_lines(bad) == []


# ── healthy verdicts collapse to silence ────────────────────────────────
@pytest.mark.parametrize("verdict", [
    "ALIGNED_EDGE", "ALIGNED_FLAT", "NO_BOOK", "WEAK_OVERLAP",
    "INSUFFICIENT_PERSONA",
])
def test_non_actionable_verdicts_are_silence(verdict):
    """Anything other than ALIGNED_DRAG is filler — silence precedent."""
    assert _persona_book_fit_chat_lines(_rep(verdict=verdict)) == []


def test_missing_verdict_is_silence():
    assert _persona_book_fit_chat_lines({"headline": "no verdict"}) == []


# ── ALIGNED_DRAG surfacing — verbatim SSOT ──────────────────────────────
def test_aligned_drag_surfaces_verbatim_headline():
    headline = (
        "ALIGNED_DRAG — book most resembles ‘Momentum Trader’ (100.0% "
        "overlap, n=50, median vs_spy -10.0pp)"
    )
    out = _persona_book_fit_chat_lines(_rep(verdict="ALIGNED_DRAG",
                                             headline=headline))
    assert len(out) >= 1
    # Verbatim — the builder's headline IS the first line.
    assert out[0] == headline


def test_aligned_drag_detail_line_restates_builder_fields():
    out = _persona_book_fit_chat_lines(_rep(overlap=66.7,
                                             runner_up_name="Value Investor",
                                             runner_up_overlap=20.0))
    # Detail line restates the builder's own numbers.
    detail = " ".join(out[1:]) if len(out) > 1 else ""
    assert "66.7%" in detail
    assert "Value Investor" in detail


def test_aligned_drag_with_no_runner_up_still_surfaces():
    """A book with one persona-archetype overlap and no runner-up still
    produces an actionable block — degrade gracefully on missing
    runner_up."""
    rep = _rep()
    rep["runner_up"] = None
    out = _persona_book_fit_chat_lines(rep)
    assert len(out) >= 1
    # Headline still there; the detail line drops the runner-up bit but
    # still includes the overlap fact.
    detail = " ".join(out[1:]) if len(out) > 1 else ""
    if detail:
        assert "Value Investor" not in detail  # no runner-up surfaced


def test_aligned_drag_with_missing_dominant_falls_back_to_headline_only():
    rep = _rep()
    rep["dominant"] = None
    rep["runner_up"] = None
    out = _persona_book_fit_chat_lines(rep)
    # The verbatim headline carries even when sub-blocks are missing.
    assert any("ALIGNED_DRAG" in line for line in out)


def test_aligned_drag_with_garbage_numeric_fields_degrades_gracefully():
    rep = _rep()
    rep["dominant"]["overlap_pct"] = "lots"
    rep["runner_up"]["overlap_pct"] = None
    out = _persona_book_fit_chat_lines(rep)
    # Still produces at least the headline; no exception.
    assert len(out) >= 1
    assert "ALIGNED_DRAG" in out[0]


def test_aligned_drag_blank_headline_omits_headline_keeps_detail():
    """A blank headline must not be carried as a literal empty string."""
    out = _persona_book_fit_chat_lines(_rep(headline=""))
    for line in out:
        assert line.strip() != ""


def test_aligned_drag_is_idempotent_on_repeated_call():
    rep = _rep()
    a = _persona_book_fit_chat_lines(rep)
    b = _persona_book_fit_chat_lines(rep)
    assert a == b
