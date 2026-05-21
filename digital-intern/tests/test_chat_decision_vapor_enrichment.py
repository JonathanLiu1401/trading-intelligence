"""Pure-helper tests for the /api/chat decision-vapor enrichment.

`_decision_vapor_chat_lines` renders paper-trader's
`/api/decision-vapor-skill` (per-FILLED-decision grounded-reasoning
detector — SPECIFIC / SEMI / VAPOR) into compact chat-context lines so
the analyst can answer "is the bot thinking, or rationalising?" — a
vapor trade that fails has nothing for the next decision to learn from.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_event_readiness_chat_lines` / `_macro_calendar_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string AND any VAPOR sample excerpt pass through
  UNCHANGED — no chat-side re-derived verdict or paraphrased excerpt that
  could drift from the trader endpoint.
- **grounded pool = silence**: SPECIFIC / NO_DATA collapse to `[]`, matching
  the `_decision_paralysis_chat_lines` silence precedent — a chat must not
  carry "reasoning is fine" filler.
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

from dashboard.web_server import _decision_vapor_chat_lines


def _sample(klass="VAPOR", excerpt="Strong setup, building position",
            action_taken="BUY NVDA → FILLED",
            ts="2026-05-21T09:00:00+00:00", id_=42):
    return {
        "id": id_, "ts": ts, "klass": klass, "action_taken": action_taken,
        "excerpt": excerpt, "has_numeric": False, "has_catalyst": False,
        "has_ticker": False,
    }


def _rep(verdict="VAPOR_DECISIONS", *, headline=None, n_filled=10,
         n_specific=2, n_semi=3, n_vapor=5, samples=None):
    if headline is None:
        if verdict == "VAPOR_DECISIONS":
            headline = (
                f"{n_vapor}/{n_filled} ({(n_vapor/n_filled)*100:.0f}%) "
                f"FILLED decisions read as vapor — missing numbers or catalysts"
            )
        elif verdict == "MIXED":
            headline = (
                f"mixed: {(n_specific/n_filled)*100:.0f}% specific / "
                f"{(n_vapor/n_filled)*100:.0f}% vapor "
                f"across {n_filled} FILLED decisions"
            )
        else:
            headline = "synthetic"
    if samples is None:
        samples = [_sample(klass="VAPOR")] if n_vapor > 0 else []
    return {
        "as_of": "2026-05-21T12:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "window_hours": 168.0,
        "stats": {
            "n_filled": n_filled,
            "n_specific": n_specific,
            "n_semi": n_semi,
            "n_vapor": n_vapor,
            "specific_pct": round((n_specific/n_filled)*100, 2),
            "semi_pct": round((n_semi/n_filled)*100, 2),
            "vapor_pct": round((n_vapor/n_filled)*100, 2),
        },
        "thresholds": {
            "vapor_pct_floor": 35.0,
            "vapor_pct_ceil": 15.0,
            "specific_pct_floor": 50.0,
            "min_filled_for_verdict": 5,
        },
        "samples": samples,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _decision_vapor_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _decision_vapor_chat_lines({}) == []
    assert _decision_vapor_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict", ["SPECIFIC", "NO_DATA", "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _decision_vapor_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "7/12 (58%) FILLED decisions read as vapor — missing numbers or "
        "catalysts")
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


def test_vapor_excerpt_passes_through_verbatim():
    raw_excerpt = (
        "Strong technical setup with momentum building. Position sizing "
        "reflects high conviction. Maintaining bias.")
    s = _sample(klass="VAPOR", excerpt=raw_excerpt,
                action_taken="BUY TQQQ → FILLED")
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", samples=[s]))
    body = "\n".join(out)
    # The full excerpt must appear unchanged — the chat is forbidden from
    # paraphrasing the bot's own words.
    assert raw_excerpt in body
    assert "BUY TQQQ → FILLED" in body


# ── per-verdict actionability ───────────────────────────────────────────
def test_vapor_decisions_emits_three_lines_with_exemplar():
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", n_filled=10, n_specific=2,
             n_semi=3, n_vapor=5))
    assert len(out) == 3
    assert "10 FILLED:" in out[1]
    assert "2 SPECIFIC" in out[1]
    assert "3 SEMI" in out[1]
    assert "5 VAPOR" in out[1]
    # Exemplar must be a VAPOR row, not SPECIFIC.
    assert out[2].lstrip().startswith("e.g. ")


def test_mixed_emits_headline_and_count_detail_no_exemplar():
    out = _decision_vapor_chat_lines(
        _rep(verdict="MIXED", n_filled=8, n_specific=4, n_semi=2,
             n_vapor=2))
    assert len(out) == 2                # no exemplar for MIXED
    assert "8 FILLED:" in out[1]
    assert "4 SPECIFIC" in out[1]
    assert "2 VAPOR" in out[1]


def test_vapor_decisions_skips_non_vapor_samples_for_exemplar():
    # First two samples are SPECIFIC / SEMI; only the third VAPOR row
    # should be surfaced as the exemplar.
    samples = [
        _sample(klass="SPECIFIC", excerpt="should be ignored 1", id_=1),
        _sample(klass="SEMI", excerpt="should be ignored 2", id_=2),
        _sample(klass="VAPOR", excerpt="this is the vapor one", id_=3,
                action_taken="SELL AMD → FILLED"),
    ]
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", n_vapor=1, samples=samples))
    body = "\n".join(out)
    assert "this is the vapor one" in body
    assert "should be ignored" not in body
    assert "SELL AMD → FILLED" in body


def test_vapor_decisions_no_samples_still_renders_count_line():
    # Builder may return an empty samples list (defensive degradation);
    # the helper must NOT raise and the count line must still render.
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", n_vapor=4, samples=[]))
    body = "\n".join(out)
    assert "VAPOR" in body
    # No exemplar line when no VAPOR sample is available.
    assert "e.g." not in body


def test_garbage_stats_do_not_raise():
    rep = _rep(verdict="MIXED")
    rep["stats"]["n_filled"] = "not-a-number"
    rep["stats"]["n_specific"] = None
    rep["stats"]["n_vapor"] = object()
    out = _decision_vapor_chat_lines(rep)
    # Headline still emitted at minimum, never raises.
    assert out and isinstance(out[0], str)


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(verdict="VAPOR_DECISIONS", headline="")
    out = _decision_vapor_chat_lines(rep)
    # No headline line, but count line still present.
    assert all(not line.lower().startswith("vapor") for line in out)
    body = "\n".join(out)
    assert "FILLED:" in body


def test_missing_stats_dict_degrades_silently():
    rep = _rep(verdict="MIXED")
    rep["stats"] = None
    out = _decision_vapor_chat_lines(rep)
    # Headline still emitted; count line absent — no exception raised.
    assert out and isinstance(out[0], str)
    assert all("FILLED:" not in line for line in out)


def test_vapor_sample_with_non_string_excerpt_is_skipped():
    samples = [
        _sample(klass="VAPOR", excerpt=None, id_=1),
        _sample(klass="VAPOR", excerpt=12345, id_=2),
        _sample(klass="VAPOR", excerpt="real text here", id_=3),
    ]
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", n_vapor=3, samples=samples))
    body = "\n".join(out)
    assert "real text here" in body


def test_vapor_sample_non_dict_does_not_raise():
    samples = ["not a dict", 42, None]
    out = _decision_vapor_chat_lines(
        _rep(verdict="VAPOR_DECISIONS", n_vapor=3, samples=samples))
    # No exemplar (none usable) but no exception either.
    assert out and isinstance(out[0], str)
