"""Pure-helper tests for the /api/chat all-cash-streak enrichment.

`_all_cash_streak_chat_lines` renders paper-trader's
`/api/all-cash-streak` (the chronic-flat-book surface) into compact
chat-context lines so the analyst can flag "the book has been 100%
cash for 22h+ — the loop may be too risk-off or stalled" — the
operator-visibility layer that cash_pct (point-in-time),
cash_redeployment latency (post-SELL sit), opportunity_cost (signal-
specific hindsight) and cash_drag (SPY-benchmarked dollar) all leave
unanswered.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_cash_redeployment_chat_lines` / `_decision_paralysis_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **healthy = silence**: BRIEF_HOLDOUT / NOT_ALL_CASH / NO_DATA /
  INSUFFICIENT_HISTORY all collapse to ``[]`` — short flats and
  not-flat books are not actionable.
- **state-OK gate**: state != "OK" collapses to silence — a builder
  that didn't run cleanly must not push a partial verdict to chat.
- **detail line fields**: when actionable, the detail line restates
  the current_streak's ``hours_elapsed_to_now`` / ``cash_usd`` /
  ``spy_return_pct`` / ``alpha_cost_usd`` verbatim — never a
  recomputation.
- **pure/total**: non-dict / missing keys / unparseable numbers never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _all_cash_streak_chat_lines


def _rep(state="OK", verdict="EXTENDED_HOLDOUT", *, headline=None,
         current_streak=None):
    if current_streak is None:
        current_streak = {
            "alpha_cost_usd": 0.0,
            "cash_usd": 987.39,
            "end_ts": "2026-05-25T00:14:39.431038+00:00",
            "hours": 22.11,
            "hours_elapsed_to_now": 22.23,
            "n_points": 84,
            "spy_return_pct": 0.0,
            "start_ts": "2026-05-24T02:08:08.434730+00:00",
        }
    if headline is None:
        hrs = current_streak.get("hours_elapsed_to_now") or current_streak.get("hours")
        cash = current_streak.get("cash_usd")
        spy = current_streak.get("spy_return_pct")
        ac = current_streak.get("alpha_cost_usd")
        ac_clause = (
            f"SPY {spy:+.2f}% → no alpha cost"
            if (spy is not None and ac is not None and ac == 0.0)
            else f"SPY {spy:+.2f}% → alpha_cost ${ac:.2f}"
            if (spy is not None and ac is not None)
            else "alpha cost unscored"
        )
        headline = f"all-cash {hrs:.1f}h on ${cash:.2f}; {ac_clause} — {verdict}"
    return {
        "as_of": "2026-05-25T00:22:13+00:00",
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "current_streak": current_streak,
        "newest_is_all_cash": True,
        "n_points": 518,
        "aggregate_flat_hours": 0.1,
        "aggregate_alpha_cost_usd": 0.0,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _all_cash_streak_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _all_cash_streak_chat_lines({}) == []


# ── state-gate ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad_state", ["error", "NO_DATA", "", None, 0])
def test_state_not_ok_is_silence(bad_state):
    """Builder that didn't run cleanly must not push a partial verdict."""
    assert _all_cash_streak_chat_lines(_rep(state=bad_state)) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize("v", [
    "BRIEF_HOLDOUT", "NOT_ALL_CASH", "NO_DATA", "INSUFFICIENT_HISTORY",
    "OK", "HEALTHY", "", None,
])
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _all_cash_streak_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize("v", ["EXTENDED_HOLDOUT", "PROLONGED_HOLDOUT"])
def test_actionable_verdicts_emit_lines(v):
    lines = _all_cash_streak_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_headline_passes_through_verbatim():
    rep = _rep(headline="custom — PROLONGED_HOLDOUT 168.0h $987.39 SPY +1.2% alpha $-12.34")
    rep["verdict"] = "PROLONGED_HOLDOUT"
    lines = _all_cash_streak_chat_lines(rep)
    assert lines[0] == "custom — PROLONGED_HOLDOUT 168.0h $987.39 SPY +1.2% alpha $-12.34"


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _all_cash_streak_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "flat 22.2h" in lines[0]
    assert "cash $987.39" in lines[0]


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_contains_all_four_fields():
    rep = _rep()
    rep["current_streak"] = {
        "alpha_cost_usd": -3.44,
        "cash_usd": 987.39,
        "hours": 18.0,
        "hours_elapsed_to_now": 22.23,
        "spy_return_pct": 0.96,
    }
    lines = _all_cash_streak_chat_lines(rep)
    assert len(lines) == 2
    detail = lines[1]
    assert detail.startswith("  ")
    assert "flat 22.2h" in detail
    assert "cash $987.39" in detail
    assert "SPY +0.96%" in detail
    assert "alpha_cost $-3.44" in detail


def test_detail_line_prefers_hours_elapsed_to_now_over_hours():
    """hours_elapsed_to_now is the live `now - start_ts` reading; the
    cached `hours` field can lag a polling interval. Always prefer
    the live one when available."""
    rep = _rep()
    rep["current_streak"] = {
        "alpha_cost_usd": 0.0, "cash_usd": 987.39,
        "hours": 18.0, "hours_elapsed_to_now": 22.23,
        "spy_return_pct": 0.0,
    }
    detail = _all_cash_streak_chat_lines(rep)[1]
    assert "flat 22.2h" in detail
    assert "flat 18.0h" not in detail


def test_detail_line_falls_back_to_hours_when_elapsed_missing():
    rep = _rep()
    rep["current_streak"] = {
        "alpha_cost_usd": 0.0, "cash_usd": 987.39,
        "hours": 18.0,
        "spy_return_pct": 0.0,
    }
    detail = _all_cash_streak_chat_lines(rep)[1]
    assert "flat 18.0h" in detail


def test_detail_line_omits_missing_fields():
    rep = _rep()
    rep["current_streak"] = {}
    lines = _all_cash_streak_chat_lines(rep)
    # Headline still present; detail line is suppressed because no
    # safe field survives — no empty "  " marker line either.
    assert len(lines) == 1


def test_missing_current_streak_degrades_to_headline_only():
    rep = _rep()
    rep["current_streak"] = None
    lines = _all_cash_streak_chat_lines(rep)
    assert lines == [rep["headline"]]


def test_non_dict_current_streak_degrades_to_headline_only():
    rep = _rep()
    rep["current_streak"] = "not-a-dict"
    lines = _all_cash_streak_chat_lines(rep)
    assert lines == [rep["headline"]]


def test_bool_field_treated_as_missing():
    """Defensive: bool is_a int in Python; never let True/False slip
    through as a numeric reading."""
    rep = _rep()
    rep["current_streak"] = {
        "cash_usd": True,
        "hours_elapsed_to_now": 22.23,
        "spy_return_pct": 0.0,
        "alpha_cost_usd": 0.0,
    }
    detail = _all_cash_streak_chat_lines(rep)[1]
    assert "cash $" not in detail
    assert "flat 22.2h" in detail


def test_dollar_thousands_formatting():
    rep = _rep()
    rep["current_streak"] = {
        "cash_usd": 12345.67,
        "hours_elapsed_to_now": 168.0,
        "spy_return_pct": 1.5,
        "alpha_cost_usd": 185.18,
    }
    detail = _all_cash_streak_chat_lines(rep)[1]
    assert "cash $12,345.67" in detail
    assert "alpha_cost $185.18" in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_fixture_2026_05_25():
    """The exact `/api/all-cash-streak` shape observed 2026-05-25 — the
    pathology this enrichment exists to surface in chat."""
    rep = {
        "as_of": "2026-05-25T00:22:13+00:00",
        "state": "OK",
        "verdict": "EXTENDED_HOLDOUT",
        "headline": (
            "all-cash 22.2h on $987.39; SPY +0.00% → no alpha cost "
            "— EXTENDED_HOLDOUT"
        ),
        "current_streak": {
            "alpha_cost_usd": 0.0,
            "cash_usd": 987.3909,
            "end_ts": "2026-05-25T00:14:39.431038+00:00",
            "hours": 22.11,
            "hours_elapsed_to_now": 22.23,
            "n_points": 84,
            "spy_return_pct": 0.0,
            "start_ts": "2026-05-24T02:08:08.434730+00:00",
        },
        "newest_is_all_cash": True,
        "n_points": 518,
    }
    lines = _all_cash_streak_chat_lines(rep)
    assert lines[0] == rep["headline"]
    assert "flat 22.2h" in lines[1]
    assert "cash $987.39" in lines[1]
    assert "SPY +0.00%" in lines[1]
    assert "alpha_cost $0.00" in lines[1]
