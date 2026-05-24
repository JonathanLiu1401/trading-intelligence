"""Pure-helper tests for the /api/chat trade-asymmetry enrichment.

``_trade_asymmetry_chat_lines`` renders paper-trader's
``/api/trade-asymmetry`` (the payoff-ratio / disposition-effect
diagnostic — are we cutting winners short while letting losers run?)
into compact chat-context lines.

Discriminating locks (mirroring the
``test_chat_decision_paralysis_enrichment.py`` shape):

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict that could drift from the trader
  endpoint.
- **healthy / sample-thin record = silence**: EDGE_POSITIVE / FLAT /
  null (EMERGING / NO_DATA) all collapse to ``[]``, matching the
  ``_decision_paralysis_chat_lines`` silence precedent — a chat must
  not carry "behavioural edge fine" filler. The builder's own
  ``stable_min_round_trips=20`` gate keeps thin samples silent.
- **detail-line numeric SSOT**: each numeric fragment restates an
  endpoint-emitted field; none are recomputed.
- **pure/total**: non-dict / missing keys / unparseable numerics never
  raise and degrade to silence or the safe subset (the
  ``_decision_paralysis_chat_lines`` precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _trade_asymmetry_chat_lines


def _rep(
    verdict="PAYOFF_TRAP",
    *,
    headline=None,
    payoff_ratio=0.45,
    actual_win_rate_pct=72.0,
    breakeven_win_rate_pct=69.0,
    avg_winner_hold_days=0.8,
    avg_loser_hold_days=4.5,
    state="STABLE",
    n=25,
):
    if headline is None:
        headline = (
            f"{verdict} — $-1.20/trade over {n} round-trips. "
            f"Payoff {payoff_ratio:.2f} demands {breakeven_win_rate_pct:.1f}% "
            f"to break even (running {actual_win_rate_pct:.1f}%)."
        )
    return {
        "as_of": "2026-05-24T14:00:00+00:00",
        "verdict": verdict,
        "state": state,
        "headline": headline,
        "n_round_trips": n,
        "n_decided": n,
        "n_wins": int(n * actual_win_rate_pct / 100.0),
        "n_losses": int(n - n * actual_win_rate_pct / 100.0),
        "n_washes": 0,
        "payoff_ratio": payoff_ratio,
        "actual_win_rate_pct": actual_win_rate_pct,
        "breakeven_win_rate_pct": breakeven_win_rate_pct,
        "win_rate_gap_pct": actual_win_rate_pct - breakeven_win_rate_pct,
        "avg_winner_hold_days": avg_winner_hold_days,
        "avg_loser_hold_days": avg_loser_hold_days,
        "disposition_gap_days": avg_winner_hold_days - avg_loser_hold_days,
        "expectancy_usd": -1.2,
        "realized_pl_usd": -1.2 * n,
        "verdict_reason": "synthetic for test",
        "stable_min_round_trips": 20,
    }


# ── pure / total contract ───────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _trade_asymmetry_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _trade_asymmetry_chat_lines({}) == []


# ── healthy / sample-thin record = silence ──────────────────────────────
@pytest.mark.parametrize("v", ["EDGE_POSITIVE", "FLAT", None, "OTHER",
                               "UNKNOWN_FUTURE_VERDICT"])
def test_non_actionable_verdicts_silence(v):
    rep = _rep(verdict=v)
    rep["verdict"] = v        # _rep composes headline from verdict; pin verdict
    assert _trade_asymmetry_chat_lines(rep) == []


def test_emerging_no_data_states_silence_via_null_verdict():
    # When state is EMERGING or NO_DATA the builder sets verdict=None.
    rep = _rep(verdict=None, state="EMERGING")
    assert _trade_asymmetry_chat_lines(rep) == []
    rep = _rep(verdict=None, state="NO_DATA")
    assert _trade_asymmetry_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim_payoff_trap():
    custom = (
        "PAYOFF_TRAP — $-3.15/trade over 20 round-trips. Payoff 0.50 "
        "demands 66.7% to break even (running 50.0%)."
    )
    out = _trade_asymmetry_chat_lines(_rep(headline=custom))
    assert out[0] == custom


def test_headline_passes_through_verbatim_disposition_bleed():
    custom = (
        "DISPOSITION_BLEED — profitable ($+0.42/trade) but cutting "
        "winners 3.20d faster than losers over 25 round-trips. "
        "Disposition gap -3.20d."
    )
    out = _trade_asymmetry_chat_lines(_rep(verdict="DISPOSITION_BLEED",
                                           headline=custom))
    assert out[0] == custom


# ── actionable detail line composition ──────────────────────────────────
def test_payoff_trap_emits_detail_line_with_all_numerics():
    out = _trade_asymmetry_chat_lines(
        _rep(verdict="PAYOFF_TRAP",
             payoff_ratio=0.4957,
             actual_win_rate_pct=50.0,
             breakeven_win_rate_pct=66.86,
             avg_winner_hold_days=1.7649,
             avg_loser_hold_days=1.5346))
    assert len(out) == 2
    body = out[1]
    assert "payoff 0.50" in body
    assert "win-rate 50.0%" in body
    assert "66.9%" in body or "66.86%" in body or "66.9" in body
    assert "winner hold 1.76d" in body
    assert "loser hold 1.53d" in body


def test_disposition_bleed_emits_detail_line():
    out = _trade_asymmetry_chat_lines(
        _rep(verdict="DISPOSITION_BLEED",
             payoff_ratio=1.20,
             actual_win_rate_pct=80.0,
             breakeven_win_rate_pct=45.5,
             avg_winner_hold_days=0.4,
             avg_loser_hold_days=5.0))
    body = "\n".join(out)
    assert "payoff 1.20" in body
    assert "win-rate 80.0%" in body
    assert "winner hold 0.40d" in body
    assert "loser hold 5.00d" in body


# ── degraded inputs degrade to silence on that fragment, never raise ───
def test_missing_numerics_degrade_per_fragment():
    rep = _rep(verdict="PAYOFF_TRAP")
    rep["payoff_ratio"] = None
    rep["actual_win_rate_pct"] = None
    rep["breakeven_win_rate_pct"] = None
    rep["avg_winner_hold_days"] = None
    rep["avg_loser_hold_days"] = None
    out = _trade_asymmetry_chat_lines(rep)
    # Headline still emits; detail line is omitted because no fragments
    assert out[0] == rep["headline"]
    assert len(out) == 1


def test_missing_breakeven_keeps_actual_win_rate():
    rep = _rep(verdict="PAYOFF_TRAP",
               headline="PAYOFF_TRAP — synthetic for breakeven-missing test")
    rep["breakeven_win_rate_pct"] = None
    out = _trade_asymmetry_chat_lines(rep)
    # Headline passes through verbatim; the chat-composed DETAIL line is
    # the one that must omit the breakeven fragment when bwr is missing.
    detail_line = out[1] if len(out) > 1 else ""
    assert "win-rate 72.0%" in detail_line
    assert "break even" not in detail_line


def test_garbage_numerics_do_not_raise():
    rep = _rep(verdict="PAYOFF_TRAP")
    rep["payoff_ratio"] = "not-a-number"
    rep["actual_win_rate_pct"] = []
    rep["breakeven_win_rate_pct"] = None
    rep["avg_winner_hold_days"] = "soonish"
    rep["avg_loser_hold_days"] = object()
    out = _trade_asymmetry_chat_lines(rep)
    # Headline still emitted at minimum.
    assert out and isinstance(out[0], str)
    assert out[0] == rep["headline"]


def test_bool_is_not_treated_as_number():
    # Defends against bool being a subclass of int — payoff_ratio=True
    # must not render as "payoff 1.00".
    rep = _rep(verdict="PAYOFF_TRAP")
    rep["payoff_ratio"] = True
    rep["actual_win_rate_pct"] = False
    out = _trade_asymmetry_chat_lines(rep)
    body = "\n".join(out)
    assert "payoff 1.00" not in body
    assert "win-rate 0.0%" not in body


def test_empty_headline_omits_first_line():
    rep = _rep(verdict="PAYOFF_TRAP", headline="")
    out = _trade_asymmetry_chat_lines(rep)
    # No empty-string row; detail line still renders.
    assert "" not in out
    assert any("payoff" in line for line in out)


def test_missing_verdict_silence():
    rep = _rep(verdict="PAYOFF_TRAP")
    rep.pop("verdict")
    assert _trade_asymmetry_chat_lines(rep) == []
