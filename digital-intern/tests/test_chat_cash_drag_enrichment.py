"""Pure-helper tests for the /api/chat cash-drag enrichment.

`_cash_drag_chat_lines` renders paper-trader's `/api/cash-drag`
(SPY-benchmarked $ cost of sitting in cash per rolling window) into
compact chat-context lines so the analyst can answer "is sitting in
cash actually costing me?" — the benchmarked-dollar follow-up that
cash_pct snapshots, cash_redeployment latency, and signal-specific
opportunity_cost all leave unanswered.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_redeployment_chat_lines` /
`_kelly_sizing_chat_lines` / `_decision_paralysis_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **healthy / unscored = silence**: NEUTRAL / HELPFUL_CASH /
  INSUFFICIENT / NO_DATA collapse to ``[]`` — cash that saved money
  or had no benchmark must not become chat filler.
- **worst-window selection**: when multiple windows are COSTLY_CASH,
  the chat detail line surfaces the one with the LARGEST dollar drag
  (ties broken by longer window_hours).
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

from dashboard.web_server import _cash_drag_chat_lines


def _win(window_hours=168.0, *, verdict="COSTLY_CASH",
         cash_drag_usd=3.44, sp500_return_pct=0.96, avg_cash_usd=358.73,
         state="OK", headline=None):
    if headline is None:
        if verdict == "COSTLY_CASH":
            headline = (
                f"{window_hours:.0f}h: cash cost you ${cash_drag_usd:.2f} "
                f"(SPY {sp500_return_pct:+.2f}%, "
                f"avg cash ${avg_cash_usd:.2f}).")
        elif verdict == "HELPFUL_CASH":
            headline = (
                f"{window_hours:.0f}h: cash SAVED you ${-cash_drag_usd:.2f} "
                f"(SPY {sp500_return_pct:+.2f}%, "
                f"avg cash ${avg_cash_usd:.2f}).")
        elif verdict == "NEUTRAL":
            headline = (
                f"{window_hours:.0f}h: cash essentially flat (SPY "
                f"{sp500_return_pct:+.2f}%, "
                f"avg cash ${avg_cash_usd:.2f}).")
        else:
            headline = f"{window_hours:.0f}h: insufficient history."
    return {
        "window_hours": window_hours,
        "verdict": verdict,
        "cash_drag_usd": cash_drag_usd,
        "sp500_return_pct": sp500_return_pct,
        "avg_cash_usd": avg_cash_usd,
        "state": state,
        "n_points": 100,
        "span_hours": window_hours,
        "headline": headline,
    }


def _rep(state="OK", verdict="COSTLY_CASH", *, headline=None, windows=None):
    if headline is None:
        headline = (
            "COSTLY_CASH — worst window: 168h: cash cost you $3.44 "
            "(SPY +0.96%, avg cash $358.73).")
    if windows is None:
        windows = [
            _win(24.0, verdict="NEUTRAL", cash_drag_usd=0.0,
                 sp500_return_pct=0.0, avg_cash_usd=526.64),
            _win(168.0, verdict="COSTLY_CASH", cash_drag_usd=3.44,
                 sp500_return_pct=0.96, avg_cash_usd=358.73),
        ]
    return {
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "windows": windows,
        "n_total_points": 453,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _cash_drag_chat_lines(bad) == []


def test_missing_state_is_silence():
    assert _cash_drag_chat_lines({}) == []
    assert _cash_drag_chat_lines({"verdict": "COSTLY_CASH"}) == []


# ── healthy / unscored = silence ────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict", ["NEUTRAL", "HELPFUL_CASH", "INSUFFICIENT", None, ""])
def test_non_actionable_top_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _cash_drag_chat_lines(rep) == []


@pytest.mark.parametrize(
    "state", ["NO_DATA", "INSUFFICIENT", None, "OTHER"])
def test_non_ok_states_silence(state):
    rep = _rep(state=state)
    assert _cash_drag_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_top_level_headline_passes_through_verbatim():
    custom = (
        "COSTLY_CASH — worst window: 720h: cash cost you $42.10 "
        "(SPY +3.50%, avg cash $750.00).")
    out = _cash_drag_chat_lines(_rep(headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── worst-window detail line ────────────────────────────────────────────
def test_worst_window_picks_highest_drag():
    rep = _rep(windows=[
        _win(24.0, cash_drag_usd=0.50, sp500_return_pct=0.10,
             avg_cash_usd=400.0),
        _win(168.0, cash_drag_usd=9.99, sp500_return_pct=2.50,
             avg_cash_usd=350.0),
        _win(720.0, cash_drag_usd=4.25, sp500_return_pct=1.20,
             avg_cash_usd=300.0),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    # 168h window has the highest drag ($9.99) — it must be surfaced.
    assert "window 168h" in body
    assert "drag $9.99" in body
    assert "SPY +2.50%" in body
    assert "avg cash $350" in body


def test_ties_broken_by_longer_window():
    rep = _rep(windows=[
        _win(168.0, cash_drag_usd=5.00),
        _win(720.0, cash_drag_usd=5.00),
        _win(24.0, cash_drag_usd=5.00),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    # Tie at $5.00; the longest window (720h) wins.
    assert "window 720h" in body


def test_helpful_and_neutral_windows_skipped_in_detail():
    rep = _rep(windows=[
        _win(24.0, verdict="NEUTRAL", cash_drag_usd=0.0),
        _win(168.0, verdict="HELPFUL_CASH", cash_drag_usd=-2.0,
             sp500_return_pct=-1.5, avg_cash_usd=400.0),
        _win(720.0, verdict="COSTLY_CASH", cash_drag_usd=1.10,
             sp500_return_pct=0.4, avg_cash_usd=300.0),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    # Only the COSTLY_CASH window contributes to the detail line.
    assert "window 720h" in body
    assert "drag $1.10" in body


def test_insufficient_window_skipped_in_detail():
    rep = _rep(windows=[
        _win(24.0, state="INSUFFICIENT", verdict=None,
             cash_drag_usd=None, sp500_return_pct=None,
             avg_cash_usd=None,
             headline="24h: insufficient history."),
        _win(168.0, verdict="COSTLY_CASH", cash_drag_usd=4.00,
             sp500_return_pct=1.20, avg_cash_usd=350.0),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    assert "window 168h" in body
    assert "drag $4.00" in body
    # Insufficient window must NOT appear in detail.
    assert "window 24h" not in body


def test_no_costly_windows_omits_detail():
    rep = _rep(verdict="COSTLY_CASH", windows=[
        _win(24.0, verdict="NEUTRAL", cash_drag_usd=0.0),
        _win(168.0, verdict="HELPFUL_CASH", cash_drag_usd=-2.0),
    ])
    out = _cash_drag_chat_lines(rep)
    # Headline still emits (top-level verdict is COSTLY_CASH so we trust
    # the trader endpoint's own framing), but the detail line is absent.
    # The detail line is uniquely identifiable by its leading indent ("  ")
    # and the "drag $" fragment — neither must appear when no COSTLY_CASH
    # window contributed.
    assert len(out) == 1
    assert not out[0].startswith("  ")
    assert "drag $" not in out[0]


# ── garbage-input robustness ────────────────────────────────────────────
def test_garbage_windows_skipped():
    rep = _rep(windows=[
        "not-a-dict",
        None,
        42,
        _win(168.0, cash_drag_usd=2.0, sp500_return_pct=0.5),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    assert "window 168h" in body
    assert "drag $2.00" in body


def test_unparseable_drag_falls_back_silently():
    rep = _rep(windows=[
        _win(168.0, cash_drag_usd="x",                # unparseable
             headline="168h: cash cost you (unparseable)."),
        _win(720.0, cash_drag_usd=1.50, sp500_return_pct=0.4,
             avg_cash_usd=300.0),
    ])
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    # Should pick the 720h window, not raise.
    assert "window 720h" in body


def test_non_list_windows_omits_detail():
    rep = _rep()
    rep["windows"] = "not-a-list"
    out = _cash_drag_chat_lines(rep)
    # Headline-only, never raises.
    assert len(out) == 1


def test_missing_windows_omits_detail():
    rep = _rep()
    rep.pop("windows", None)
    out = _cash_drag_chat_lines(rep)
    assert len(out) == 1


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(headline="")
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    assert "window 168h" in body
    assert "drag $3.44" in body


def test_garbage_headline_omits_first_line():
    rep = _rep(headline=42)
    out = _cash_drag_chat_lines(rep)
    body = "\n".join(out)
    assert "window 168h" in body


def test_returns_list_always():
    assert isinstance(_cash_drag_chat_lines({}), list)
    assert isinstance(_cash_drag_chat_lines(None), list)
    assert isinstance(_cash_drag_chat_lines(_rep()), list)
