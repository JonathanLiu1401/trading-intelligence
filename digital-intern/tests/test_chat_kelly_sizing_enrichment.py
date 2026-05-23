"""Pure-helper tests for the /api/chat kelly-sizing enrichment.

`_kelly_sizing_chat_lines` renders paper-trader's `/api/kelly-sizing`
(Kelly-criterion sizing diagnostic — how the current top-position weight
compares to a Kelly-optimal allocation derived from realised win-rate ×
payoff ratio) into compact chat-context lines so the analyst can tell at
a glance whether the largest position is over- or under-sized vs the
statistical edge.

Per the established design (cf. `_decision_paralysis_chat_lines` /
`_regime_leverage_fit_chat_lines`) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED — no chat-side
  re-derived verdict that could drift from the trader endpoint.
- **healthy book = silence**: KELLY_ALIGNED / None / EMERGING-state
  payloads collapse to `[]`, matching the `_decision_paralysis_chat_lines`
  silence precedent — a chat must not carry "sizing fine" filler.
- **pure/total**: non-dict / missing keys / non-numeric values never raise
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

from dashboard.web_server import _kelly_sizing_chat_lines


def _rep(verdict="EXTREMELY_OVERSIZED", *, headline=None,
         full_kelly=62.5, half_kelly=31.25, top_pct=65.0,
         top_ticker="NVDA"):
    if headline is None:
        headline = (
            f"EXTREMELY_OVERSIZED — top position at {top_pct:.1f}% is "
            f"{top_pct / half_kelly:.2f}× half-Kelly ({half_kelly:.1f}%) and "
            f"above FULL Kelly ({full_kelly:.1f}%) — ruin tail of the long-run "
            f"growth curve.")
    return {
        "as_of": "2026-05-23T14:00:00+00:00",
        "state": "STABLE",
        "verdict": verdict,
        "verdict_reason": None,
        "headline": headline,
        "n_round_trips": 20,
        "n_wins": 14,
        "n_losses": 6,
        "actual_win_rate_pct": 70.0,
        "payoff_ratio": 4.0,
        "full_kelly_pct": full_kelly,
        "half_kelly_pct": half_kelly,
        "quarter_kelly_pct": full_kelly / 4.0 if full_kelly else None,
        "top_position_pct": top_pct,
        "top_position_ticker": top_ticker,
        "delta_vs_half_kelly_pct": (top_pct - half_kelly
                                    if top_pct is not None
                                    and half_kelly is not None else None),
        "stable_min_round_trips": 20,
        "thresholds": {},
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _kelly_sizing_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _kelly_sizing_chat_lines({}) == []
    assert _kelly_sizing_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize("verdict",
                         ["KELLY_ALIGNED", None, "OTHER",
                          "EMERGING_STATE", "NO_DATA"])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _kelly_sizing_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ─────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "EXTREMELY_OVERSIZED — top position at 65.4% is 2.08× half-Kelly "
        "(31.25%) and above FULL Kelly (62.5%) — ruin tail of the long-run "
        "growth curve.")
    out = _kelly_sizing_chat_lines(_rep(headline=custom, top_pct=65.4))
    assert out[0] == custom              # exact char-for-char passthrough


# ── per-verdict actionability ──────────────────────────────────────────
@pytest.mark.parametrize("verdict", ["UNDERSIZED", "OVERSIZED",
                                     "EXTREMELY_OVERSIZED", "NEGATIVE_EDGE"])
def test_actionable_verdicts_emit(verdict):
    out = _kelly_sizing_chat_lines(_rep(verdict=verdict))
    assert len(out) >= 1
    assert out[0]                          # non-empty headline


def test_detail_line_carries_kelly_targets_and_top_ticker():
    out = _kelly_sizing_chat_lines(_rep(top_pct=65.0, top_ticker="NVDA"))
    # Headline + detail.
    assert len(out) == 2
    detail = out[1]
    # Composition order: half-Kelly target, full-Kelly, current top with
    # ticker.
    assert "half-Kelly target 31.2%" in detail or "31.3%" in detail
    assert "full-Kelly 62.5%" in detail
    assert "current top 65.0%" in detail
    assert "(NVDA)" in detail


def test_detail_line_omits_ticker_when_missing():
    out = _kelly_sizing_chat_lines(_rep(top_ticker=None))
    detail = out[1]
    assert "(" not in detail or "NVDA" not in detail  # no orphan ticker tag


def test_missing_numeric_fields_degrade_silently():
    rep = _rep()
    # Strip every numeric — the helper must not raise; either silence or a
    # headline-only emit is acceptable.
    for key in ("full_kelly_pct", "half_kelly_pct", "top_position_pct",
                "top_position_ticker"):
        rep[key] = None
    out = _kelly_sizing_chat_lines(rep)
    # Headline still carries (verbatim from builder); detail line is empty
    # so the helper should emit only the headline.
    assert len(out) == 1
    assert "EXTREMELY_OVERSIZED" in out[0]


def test_non_numeric_top_pct_degrades(headline=None):
    rep = _rep()
    rep["top_position_pct"] = "not a number"
    rep["full_kelly_pct"] = True            # bool is rejected by _num
    out = _kelly_sizing_chat_lines(rep)
    # Headline emitted; detail composed only from the half-Kelly field.
    assert len(out) >= 1
