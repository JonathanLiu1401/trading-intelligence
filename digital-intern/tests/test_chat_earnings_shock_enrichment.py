"""Pure-helper tests for the /api/chat pre-earnings shock enrichment.

`_earnings_shock_chat_lines` renders paper-trader's `/api/earnings-shock`
(the forward $-at-risk view per held imminent print) into compact chat-
context lines, so the analyst can answer "if NVDA gaps the typical 1σ on
its earnings release, what does it cost my book?" — the actual pre-print
question, which neither EARNINGS RADAR (timing-only) nor any sibling chat
block dollarizes. This closes that gap, in the same shape as
`_macro_calendar_chat_lines` / `_baseline_compare_chat_lines`.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_macro_calendar_chat_lines`,
`_baseline_compare_chat_lines`) the logic is a total/pure function unit-
tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT headline** (paper-trader invariant #10): the builder's
  own ``headline`` passes through UNCHANGED as the chat headline — a
  drift fails RED.
- **NO_DATA / NO_EVENTS is silence, not noise**: the empty book + quiet
  calendar paths must collapse to ``[]``, mirroring how
  ``_macro_calendar_chat_lines`` omits the no-FOMC and not-loaded branches.
- **INSUFFICIENT_HISTORY honesty**: a row with no usable history still
  surfaces the *event* (timing + exposure) but explicitly reports σ as
  *withheld* — never fabricated (the builder's per-row honesty contract;
  same shape as the baseline_compare INSUFFICIENT_DATA silence).
- **No exception leak**: a non-dict / malformed event row / missing keys
  never raises into the chat handler — the
  ``_macro_calendar_chat_lines`` malformed-row precedent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _earnings_shock_chat_lines


def _ok_event(**over) -> dict:
    e = {
        "ticker": "NVDA",
        "days_to_earnings": 0.9,
        "earnings_date": "2026-05-20T20:00:00+00:00",
        "tier": "HELD_IMMINENT",
        "current_value_usd": 444.70,
        "weight_pct": 44.47,
        "n_history": 8,
        "history_mean_pct": 1.25,
        "history_worst_pct": -4.0,
        "history_best_pct": 8.0,
        "state": "OK",
        "sigma_pct": 4.24,
        "sigma_dollar_move": 18.85,
        "sigma_book_pct": 1.88,
        "stress_3sigma_dollar_down": -56.55,
        "stress_3sigma_book_pct_down": -5.66,
        "row_verdict": "MODERATE",
        "headline": "NVDA in 0.9d: σ ±4.2% (n=8 prints, worst -4.0%, best +8.0%) → ±$18.85 (book ±1.88%); 3σ down stress $-56.55 (-5.66% of book).",
    }
    e.update(over)
    return e


def _insuff_event(**over) -> dict:
    e = {
        "ticker": "MU",
        "days_to_earnings": 2.0,
        "earnings_date": "2026-05-21T20:00:00+00:00",
        "tier": "HELD_IMMINENT",
        "current_value_usd": 250.00,
        "weight_pct": 25.00,
        "n_history": 1,
        "history_mean_pct": -3.0,
        "history_worst_pct": -3.0,
        "history_best_pct": -3.0,
        "state": "INSUFFICIENT_HISTORY",
        "sigma_pct": None,
        "sigma_dollar_move": None,
        "sigma_book_pct": None,
        "stress_3sigma_dollar_down": None,
        "stress_3sigma_book_pct_down": None,
        "row_verdict": "UNKNOWN",
        "headline": "MU: earnings in 2.0d — σ withheld (1 historical print, need ≥3)",
    }
    e.update(over)
    return e


def _rep(events=None, headline="Pre-earnings shock (1 held name ≤7d): worst is NVDA in 0.9d: σ ±4.2% (n=8 prints, worst -4.0%, best +8.0%) → ±$18.85 (book ±1.88%); 3σ down stress $-56.55 (-5.66% of book).", state="OK", **over) -> dict:
    d = {
        "as_of": "2026-05-19T01:17:55+00:00",
        "horizon_days": 7.0,
        "history_depth": 8,
        "n_events": 1,
        "events": [_ok_event()] if events is None else events,
        "total_sigma_book_pct": 1.88,
        "headline": headline,
        "verdict": "LOW",
        "state": state,
    }
    d.update(over)
    return d


# ── pure / total contract ─────────────────────────────────────────────


@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _earnings_shock_chat_lines(bad) == []


def test_missing_keys_is_silence_not_crash():
    """An empty dict / missing state must not raise into the chat handler."""
    assert _earnings_shock_chat_lines({}) == []
    assert _earnings_shock_chat_lines({"state": "OK"}) == []
    assert _earnings_shock_chat_lines({"state": "OK", "events": None}) == []
    assert _earnings_shock_chat_lines({"state": "OK", "events": []}) == []


# ── non-actionable states collapse to silence ─────────────────────────


def test_no_data_state_is_silence():
    """Empty book → NO_DATA → must be silence, mirroring how
    ``_macro_calendar_chat_lines`` omits the not-loaded branch — the chat
    must never become its own lying green light with a NO_DATA filler."""
    assert _earnings_shock_chat_lines(
        _rep(state="NO_DATA", events=[],
             headline="Earnings shock: no priced book to shock yet.")) == []


def test_no_events_state_is_silence():
    """Book priced but calendar quiet → NO_EVENTS → silence (the
    NO_EVENTS-distinct-from-NO_DATA contract; chat carries no "no held
    name reports within 7d" filler — same precedent)."""
    assert _earnings_shock_chat_lines(
        _rep(state="NO_EVENTS", events=[],
             headline="Earnings shock: no held name reports within 7d.")) == []


# ── SSOT: the builder's own headline is verbatim ──────────────────────


def test_ok_headline_is_builder_headline_verbatim():
    """A chat-side re-derived verdict that drifts from /api/earnings-shock
    fails here — invariant #10 (the ``_macro_calendar_chat_lines``
    SSOT-headline lock)."""
    rep = _rep()
    out = _earnings_shock_chat_lines(rep)
    assert out, "an OK event must surface"
    assert out[0] == rep["headline"]            # byte-identical SSOT


def test_event_detail_restates_builder_fields_not_recomputed():
    """Per-row line restates the builder's OWN fields — a restatement
    (the earnings_block / macro_calendar precedent), never a recomputation."""
    rep = _rep()
    blob = "\n".join(_earnings_shock_chat_lines(rep))
    # The detail line must restate ticker, timing, exposure, σ.
    assert "NVDA" in blob
    assert "0.9d" in blob
    assert "444.70" in blob or "$444.70" in blob
    assert "44.5%" in blob or "44.47%" in blob
    assert "±4.2%" in blob or "±$18.85" in blob


# ── INSUFFICIENT_HISTORY honesty ──────────────────────────────────────


def test_insufficient_history_event_surfaces_but_sigma_withheld():
    """A row with no usable history must STILL surface the event timing
    and exposure (so "MU reports in 2.0d" is never hidden), but σ must
    be reported as *withheld* — never fabricated. Mirrors the builder's
    own per-row honesty contract (and the baseline_compare INSUFFICIENT_DATA
    chat-silence precedent for verdicts)."""
    # The default _rep() headline talks about an OK NVDA event (σ ±4.2%),
    # which would be a builder-side SSOT mismatch for a payload carrying only
    # an insufficient-history MU event — override it to the matching insuff
    # headline so this test gates the per-row "no fabricated σ" behaviour
    # only, not a fixture mismatch on the SSOT headline line.
    rep = _rep(
        events=[_insuff_event()],
        headline="Pre-earnings shock (1 held name ≤7d): MU σ withheld "
                 "(1 historical print, need ≥3); event timing $250.00 "
                 "(25.0% of book).",
    )
    out = _earnings_shock_chat_lines(rep)
    blob = "\n".join(out)
    # Event timing + exposure surfaces (the always-visible parts).
    assert "MU" in blob
    assert "2.0d" in blob
    assert "$250.00" in blob
    assert "25.0%" in blob or "25.00%" in blob
    # σ MUST NOT appear as a fabricated numeric (1.0 or 0.0 or any
    # honest-looking but invented figure).
    assert "σ ±" not in blob
    # An explicit "withheld" disclosure is the honest carry-through —
    # mirrors the builder's INSUFFICIENT_HISTORY headline + the chat-
    # side baseline_compare INSUFFICIENT_DATA precedent.
    assert "withheld" in blob


def test_mixed_ok_and_insufficient_events_each_get_own_line():
    """A two-name book where NVDA has 8 prints and MU has 1 must surface
    BOTH events — the OK one with σ, the INSUFFICIENT one without."""
    rep = _rep(events=[_ok_event(), _insuff_event()], n_events=2)
    out = _earnings_shock_chat_lines(rep)
    blob = "\n".join(out)
    assert "NVDA" in blob
    assert "MU" in blob
    # NVDA's σ surfaces; MU's is explicitly withheld
    assert ("±4.2%" in blob) or ("±$18.85" in blob)
    assert "withheld" in blob


# ── partial / malformed never raises ──────────────────────────────────


def test_events_present_but_headline_missing_still_emits_no_raise():
    """A degraded payload (events present, headline missing/non-str)
    must still surface the events and must NOT fabricate a headline or
    raise (the ``_macro_calendar_chat_lines`` precedent)."""
    rep = _rep()
    rep.pop("headline")
    out = _earnings_shock_chat_lines(rep)
    assert out, "events must still surface when headline is absent"
    assert "NVDA" in "\n".join(out)

    rep2 = _rep(headline=None)
    out2 = _earnings_shock_chat_lines(rep2)
    assert out2 and "NVDA" in "\n".join(out2)


def test_malformed_event_row_is_skipped_never_raises():
    """Garbage rows in the events list must be skipped without raising
    (the ``_macro_calendar_chat_lines`` malformed-row precedent)."""
    rep = _rep(events=["not-a-dict", None, 42, _ok_event()])
    out = _earnings_shock_chat_lines(rep)        # must not raise
    blob = "\n".join(out)
    # The one good event still surfaces.
    assert "NVDA" in blob


def test_event_missing_ticker_is_skipped():
    """A row without a usable ticker is non-actionable — skip silently
    (the ``_macro_calendar_chat_lines`` malformed-row precedent)."""
    bad = _ok_event(ticker=None)
    good = _ok_event(ticker="AMD")
    rep = _rep(events=[bad, good])
    blob = "\n".join(_earnings_shock_chat_lines(rep))
    assert "AMD" in blob


def test_all_event_rows_malformed_degrades_to_headline_or_silence():
    """If every event row is junk, the helper must not raise. With a
    real headline the SSOT headline still stands; without one it's
    silence (the ``_macro_calendar_chat_lines`` precedent)."""
    assert _earnings_shock_chat_lines(
        _rep(events=["x", None]))[0] == _rep()["headline"]
    assert _earnings_shock_chat_lines(
        _rep(events=["x", None], headline=None)) == []
