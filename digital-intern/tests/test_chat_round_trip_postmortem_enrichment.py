"""Pure-helper tests for the /api/chat round-trip-postmortem enrichment.

`_round_trip_postmortem_chat_lines` renders paper-trader's
`/api/round-trip-postmortem` (post-exit price-drift verdict per closed
round-trip — CORRECT / PREMATURE / MISSED_RUNNER / WHIPSAW / NEUTRAL)
into compact chat-context lines so the analyst can answer "should we
have held that exit?" — the falsifiable hindsight question every other
realized-P&L chat surface (winner_autopsy / loser_autopsy / streak /
scorecard) reduces to a P&L number and walks away from.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_redeployment_chat_lines` /
`_decision_vapor_chat_lines` / `_decision_paralysis_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's top-level ``headline`` AND the surfaced worst-trip's own
  per-row ``headline`` BOTH pass through UNCHANGED — no chat-side
  paraphrase of the bot's own per-trip narrative.
- **healthy ladder = silence**: all-CORRECT/NEUTRAL exit ladders, plus
  state in {NO_DATA, INSUFFICIENT}, collapse to ``[]`` — matching the
  ``_decision_paralysis_chat_lines`` silence precedent.
- **worst-trip selection**: when multiple unfavourable verdicts exist,
  the chat surfaces the one with the LARGEST absolute post-exit drift
  (the most painful sample, by definition) — not the first/last in
  whatever order the builder returned them.
- **pure/total**: non-dict / missing keys / unparseable drifts never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _round_trip_postmortem_chat_lines


def _trip(ticker="NVDA", verdict="PREMATURE", post_exit_drift_pct=-3.0,
          headline=None):
    if headline is None:
        sign = "+" if post_exit_drift_pct >= 0 else ""
        headline = (
            f"{ticker}: sold (verdict {verdict}, drift "
            f"{sign}{post_exit_drift_pct:.2f}% post-exit).")
    return {
        "ticker": ticker,
        "verdict": verdict,
        "post_exit_drift_pct": post_exit_drift_pct,
        "headline": headline,
        "type": "stock",
    }


def _rep(state="OK", *, headline="2/4 exits ran against the bot.",
         trips=None, exit_quality_score=-0.5):
    if trips is None:
        trips = [
            _trip("NVDA", "CORRECT", -3.66,
                  "NVDA: sold $223.44; -3.66% post-exit. Exit captured the move."),
            _trip("TQQQ", "PREMATURE", 3.27,
                  "TQQQ: sold $75.34; +3.27% post-exit. May have exited too early."),
        ]
    return {
        "state": state,
        "exit_quality_score": exit_quality_score,
        "headline": headline,
        "n_input": len(trips),
        "n_scored": len(trips),
        "trips": trips,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _round_trip_postmortem_chat_lines(bad) == []


def test_missing_state_is_silence():
    assert _round_trip_postmortem_chat_lines({}) == []
    assert _round_trip_postmortem_chat_lines(
        {"headline": "x", "trips": [_trip()]}) == []


# ── insufficient/no-data = silence ──────────────────────────────────────
@pytest.mark.parametrize("state", ["NO_DATA", "INSUFFICIENT", None, "OTHER"])
def test_non_ok_states_silence(state):
    rep = _rep(state=state)
    assert _round_trip_postmortem_chat_lines(rep) == []


# ── healthy ladder = silence ────────────────────────────────────────────
def test_all_correct_trips_silence():
    rep = _rep(trips=[
        _trip("NVDA", "CORRECT", -5.0),
        _trip("TQQQ", "CORRECT", -2.0),
    ])
    assert _round_trip_postmortem_chat_lines(rep) == []


def test_all_neutral_trips_silence():
    rep = _rep(trips=[
        _trip("NVDA", "NEUTRAL", 0.1),
        _trip("TQQQ", "NEUTRAL", -0.2),
    ])
    assert _round_trip_postmortem_chat_lines(rep) == []


def test_mixed_correct_and_neutral_silence():
    rep = _rep(trips=[
        _trip("NVDA", "CORRECT", -5.0),
        _trip("TQQQ", "NEUTRAL", 0.5),
        _trip("MU", "CORRECT", -1.5),
    ])
    assert _round_trip_postmortem_chat_lines(rep) == []


def test_empty_trips_silence():
    rep = _rep(trips=[])
    assert _round_trip_postmortem_chat_lines(rep) == []


def test_missing_trips_silence():
    rep = _rep()
    rep.pop("trips", None)
    assert _round_trip_postmortem_chat_lines(rep) == []


# ── actionable verdicts surface ─────────────────────────────────────────
@pytest.mark.parametrize("verdict", ["PREMATURE", "MISSED_RUNNER", "WHIPSAW"])
def test_each_unfavourable_verdict_surfaces(verdict):
    rep = _rep(trips=[_trip("NVDA", verdict, 4.0)])
    out = _round_trip_postmortem_chat_lines(rep)
    assert len(out) >= 1


# ── verbatim SSOT (invariant #10) ───────────────────────────────────────
def test_top_level_headline_passes_through_verbatim():
    custom_hdr = "3/5 exits ran against the bot (premature/whipsaw/missed) — bot may be selling too early."
    rep = _rep(headline=custom_hdr, trips=[
        _trip("TQQQ", "MISSED_RUNNER", 8.5,
              "TQQQ: sold $75; +8.50% post-exit, 89h. Exited a runner."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    assert out[0] == custom_hdr            # exact char-for-char passthrough


def test_worst_trip_headline_passes_through_verbatim():
    custom_trip_hdr = (
        "TQQQ: sold $75.34 (+2.52%); $77.80 now (+3.27% post-exit, 89.9h). "
        "May have exited too early.")
    rep = _rep(trips=[
        _trip("NVDA", "CORRECT", -3.66, "NVDA: ok."),
        _trip("TQQQ", "PREMATURE", 3.27, custom_trip_hdr),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert custom_trip_hdr in body          # verbatim trip headline


# ── worst-trip selection ────────────────────────────────────────────────
def test_worst_trip_is_largest_absolute_drift():
    rep = _rep(trips=[
        _trip("AAA", "PREMATURE", 2.0,
              "AAA: sold; +2.00% post-exit."),
        _trip("BBB", "MISSED_RUNNER", 9.5,
              "BBB: sold; +9.50% post-exit — BIGGEST."),
        _trip("CCC", "WHIPSAW", -4.0,
              "CCC: sold; -4.00% post-exit."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "BBB: sold; +9.50% post-exit — BIGGEST." in body
    # The smaller-drift unfavourable trips should NOT appear in the chat
    # output (only one worst sample is surfaced, by design).
    assert "AAA: sold;" not in body
    assert "CCC: sold;" not in body


def test_worst_trip_ignores_correct_neutral_drifts():
    # A CORRECT trip with a -20% drift should NOT be picked as "worst";
    # only PREMATURE/MISSED_RUNNER/WHIPSAW are candidates.
    rep = _rep(trips=[
        _trip("CORRECT_BIG", "CORRECT", -20.0,
              "CORRECT_BIG: do not surface."),
        _trip("BAD_SMALL", "PREMATURE", 1.5,
              "BAD_SMALL: small but unfavourable."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "BAD_SMALL: small but unfavourable." in body
    assert "CORRECT_BIG" not in body


# ── garbage-input robustness ────────────────────────────────────────────
def test_garbage_trip_rows_skipped():
    rep = _rep(trips=[
        "not-a-dict",                  # filtered
        None,                          # filtered
        42,                            # filtered
        _trip("NVDA", "PREMATURE", 4.0,
              "NVDA: sold; +4.00% post-exit."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "NVDA: sold; +4.00% post-exit." in body


def test_unparseable_drift_treated_as_smallest():
    # An unparseable drift should not surface above a parseable one.
    rep = _rep(trips=[
        _trip("AAA", "PREMATURE", "x",
              "AAA: unparseable drift."),
        _trip("BBB", "MISSED_RUNNER", 1.0,
              "BBB: parseable drift."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "BBB: parseable drift." in body


def test_missing_worst_headline_omits_detail_line():
    rep = _rep(trips=[
        _trip("NVDA", "PREMATURE", 4.0, headline=""),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    # Top-level headline still surfaces (verbatim), worst-trip line omitted.
    assert len(out) == 1
    assert out[0]


def test_empty_top_level_headline_still_renders_worst():
    rep = _rep(headline="", trips=[
        _trip("NVDA", "PREMATURE", 4.0,
              "NVDA: sold; +4.00% post-exit."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "NVDA: sold; +4.00% post-exit." in body
    # Top-level headline omitted (empty string filtered).
    assert not any(line == "" for line in out)


def test_garbage_top_level_headline_omitted_detail_still_renders():
    rep = _rep(headline=42, trips=[          # non-string headline
        _trip("NVDA", "MISSED_RUNNER", 7.0,
              "NVDA: sold; +7.00% post-exit."),
    ])
    out = _round_trip_postmortem_chat_lines(rep)
    body = "\n".join(out)
    assert "NVDA: sold; +7.00% post-exit." in body


def test_returns_list_always():
    assert isinstance(_round_trip_postmortem_chat_lines({}), list)
    assert isinstance(_round_trip_postmortem_chat_lines(None), list)
    assert isinstance(_round_trip_postmortem_chat_lines(_rep()), list)
