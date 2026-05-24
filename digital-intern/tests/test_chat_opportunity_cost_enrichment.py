"""Pure-helper tests for the /api/chat opportunity-cost enrichment.

`_opportunity_cost_chat_lines` renders paper-trader's `/api/opportunity-cost`
(the hindsight read on past HOLD-CASH / NO_DECISION sit-outs — graded
against forward returns of the top-news watchlist ticker) into compact
chat-context lines so the analyst can answer "is cash discipline
COSTING or SAVING alpha?".

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_cash_redeployment_chat_lines`) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **neutral cash discipline = silence**: NEUTRAL / NO_DATA / ERROR collapse
  to `[]`, matching the `_decision_paralysis_chat_lines` silence precedent
  — chat must not carry "sit-outs were neither costly nor saving" filler.
- **pure/total**: non-dict / missing keys / unparseable counts never raise
  and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _opportunity_cost_chat_lines


def _rep(verdict="MISSED_ALPHA", *, headline=None, n_sitout=20,
         n_classified=12, n_missed_runner=6, n_missed_ok=4, n_neutral=1,
         n_defensive=1, mean_3d=3.2):
    missed_pct = (
        round(100.0 * (n_missed_runner + n_missed_ok) / n_classified, 2)
        if n_classified else None
    )
    defensive_pct = (
        round(100.0 * n_defensive / n_classified, 2) if n_classified else None
    )
    if headline is None:
        if verdict == "MISSED_ALPHA":
            headline = (
                f"sit-out cost: {missed_pct:.0f}% of sit-outs preceded "
                f"a runner / ok move (mean 3d {mean_3d:+.2f}%, n={n_classified})")
        elif verdict == "DEFENSIVE_WIN":
            headline = (
                f"defensive sit-out paid: {defensive_pct:.0f}% of sit-outs "
                f"dodged a drawdown (mean 3d {mean_3d:+.2f}%, n={n_classified})")
        else:
            headline = "sit-outs were neutral"
    return {
        "as_of": "2026-05-24T12:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "window_hours": 168.0,
        "stats": {
            "n_sitout_total": n_sitout,
            "n_no_candidate": max(0, n_sitout - n_classified - 2),
            "n_no_fwd": 2,
            "n_classified": n_classified,
            "n_missed_runner": n_missed_runner,
            "n_missed_ok": n_missed_ok,
            "n_neutral": n_neutral,
            "n_defensive": n_defensive,
            "missed_pct": missed_pct,
            "defensive_pct": defensive_pct,
            "mean_fwd_3d_pct": mean_3d,
        },
        "thresholds": {
            "runner_pct_floor": 5.0,
            "ok_pct_floor": 1.0,
            "defensive_pct_ceil": -1.0,
            "missed_pct_floor": 50.0,
            "mean_fwd_pct_floor": 2.0,
            "min_decisions": 5,
        },
        "samples": [],
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _opportunity_cost_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _opportunity_cost_chat_lines({}) == []


def test_missing_verdict_is_silence():
    assert _opportunity_cost_chat_lines({"headline": "x"}) == []


# ── healthy / neutral = silence ─────────────────────────────────────────
@pytest.mark.parametrize("verdict",
                         ["NEUTRAL", "NO_DATA", "ERROR", "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _opportunity_cost_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim_missed_alpha():
    custom = ("sit-out cost: 83% of sit-outs preceded a runner / ok move "
              "(mean 3d +4.21%, n=12)")
    out = _opportunity_cost_chat_lines(_rep(headline=custom))
    assert out[0] == custom              # exact char-for-char passthrough


def test_headline_passes_through_verbatim_defensive_win():
    custom = ("defensive sit-out paid: 67% of sit-outs dodged a drawdown "
              "(mean 3d -3.18%, n=9)")
    out = _opportunity_cost_chat_lines(_rep(verdict="DEFENSIVE_WIN",
                                             headline=custom))
    assert out[0] == custom


# ── actionable verdicts emit headline + detail ──────────────────────────
def test_missed_alpha_emits_headline_plus_detail():
    out = _opportunity_cost_chat_lines(_rep(verdict="MISSED_ALPHA"))
    assert len(out) >= 2
    assert "sit-outs graded" in out[1]


def test_defensive_win_emits_headline_plus_detail():
    out = _opportunity_cost_chat_lines(_rep(verdict="DEFENSIVE_WIN",
                                             mean_3d=-3.5))
    assert len(out) >= 2
    # Detail line should show both percentages and mean 3d.
    detail = out[1]
    assert "missed" in detail.lower()
    assert "defensive" in detail.lower()


def test_detail_shows_mean_3d_with_sign():
    out = _opportunity_cost_chat_lines(_rep(mean_3d=4.2))
    detail = out[1] if len(out) > 1 else ""
    assert "+4.20%" in detail


def test_detail_shows_negative_mean_3d():
    out = _opportunity_cost_chat_lines(_rep(verdict="DEFENSIVE_WIN",
                                             mean_3d=-3.18))
    detail = out[1] if len(out) > 1 else ""
    assert "-3.18%" in detail


def test_detail_includes_unrated_count_when_relevant():
    # n_sitout=20, n_classified=12 → 8 sit-outs too recent to grade
    out = _opportunity_cost_chat_lines(_rep(n_sitout=20, n_classified=12))
    detail = out[1] if len(out) > 1 else ""
    assert "8 too recent to grade" in detail


def test_detail_omits_unrated_when_all_classified():
    out = _opportunity_cost_chat_lines(_rep(n_sitout=12, n_classified=12))
    detail = out[1] if len(out) > 1 else ""
    assert "too recent" not in detail


# ── defensive degradation on garbage fields ────────────────────────────
def test_garbage_stats_dict_does_not_raise():
    bad = _rep(verdict="MISSED_ALPHA")
    bad["stats"] = "not a dict"
    out = _opportunity_cost_chat_lines(bad)
    # Headline still surfaces; detail may collapse to nothing.
    assert isinstance(out, list)
    assert len(out) >= 1
    assert out[0] == bad["headline"]


def test_missing_stats_key_does_not_raise():
    bad = _rep(verdict="MISSED_ALPHA")
    del bad["stats"]
    out = _opportunity_cost_chat_lines(bad)
    assert isinstance(out, list)
    assert len(out) >= 1


def test_garbage_numeric_fields_skip_detail():
    bad = _rep(verdict="MISSED_ALPHA")
    bad["stats"]["missed_pct"] = "x"
    bad["stats"]["defensive_pct"] = None
    bad["stats"]["mean_fwd_3d_pct"] = "y"
    bad["stats"]["n_classified"] = "z"
    out = _opportunity_cost_chat_lines(bad)
    # Headline still present; detail line may not be emitted.
    assert out[0] == bad["headline"]


def test_blank_headline_collapses_to_detail_only():
    rep = _rep(verdict="MISSED_ALPHA")
    rep["headline"] = "   "
    out = _opportunity_cost_chat_lines(rep)
    # Blank headline is filtered; only detail (if any) remains.
    assert all("   " not in line for line in out)
