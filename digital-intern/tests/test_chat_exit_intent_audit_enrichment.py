"""Pure-helper tests for the /api/chat exit-intent-audit enrichment.

`_exit_intent_audit_chat_lines` renders paper-trader's
`/api/exit-intent-audit` (per-closed-sell intent classification by exit
reason text — EARNINGS_CLEAR / STOP_LOSS / TARGET_HIT / THESIS_FLIP /
DEFENSIVE_CASH_RAISE / UNCLASSIFIED — rolled up to outcome per bucket)
into compact chat-context lines so the analyst can see whether the most
common stated reason to sell is also a money loser.

Per the established design (cf. `_decision_paralysis_chat_lines` /
`_regime_leverage_fit_chat_lines`) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED — no chat-side
  re-derived verdict that could drift from the trader endpoint.
- **healthy mix = silence**: DOMINANT_INTENT_HEALTHY / None collapse to
  `[]`, matching the `_decision_paralysis_chat_lines` silence precedent —
  chat must not carry "exit mix fine" filler.
- **pure/total**: non-dict / missing keys / non-list buckets / missing
  dominant bucket never raise and degrade to silence or the safe subset
  (the `_paper_trader_position_lines` precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _exit_intent_audit_chat_lines


def _rep(verdict="DOMINANT_INTENT_BLEED", *, headline=None,
         dominant="DEFENSIVE_CASH_RAISE", dom_n=6,
         dom_total=-12.5, dom_avg_pct=-1.75, dom_wr=33.3):
    if headline is None:
        headline = (
            f"DOMINANT_INTENT_BLEED — dominant exit intent is {dominant} "
            f"({dom_n}/10 round-trips) but it averages "
            f"${dom_total / dom_n:+.2f}/trip ({dom_avg_pct:+.2f}%) — the most "
            f"common reason to sell is also a money loser.")
    buckets = [
        {"intent": "EARNINGS_CLEAR", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
        {"intent": "STOP_LOSS", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
        {"intent": "TARGET_HIT", "n": 2,
         "total_pnl_usd": 4.0, "avg_pnl_usd": 2.0,
         "avg_pnl_pct": 2.5, "win_rate_pct": 100.0,
         "median_hold_days": 1.5, "n_wins": 2, "n_losses": 0,
         "examples": []},
        {"intent": "THESIS_FLIP", "n": 2,
         "total_pnl_usd": -3.0, "avg_pnl_usd": -1.5,
         "avg_pnl_pct": -1.0, "win_rate_pct": 0.0,
         "median_hold_days": 0.5, "n_wins": 0, "n_losses": 2,
         "examples": []},
        {"intent": dominant, "n": dom_n,
         "total_pnl_usd": dom_total,
         "avg_pnl_usd": dom_total / dom_n if dom_n else None,
         "avg_pnl_pct": dom_avg_pct, "win_rate_pct": dom_wr,
         "median_hold_days": 0.1, "n_wins": 2, "n_losses": 4,
         "examples": []},
        {"intent": "UNCLASSIFIED", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
    ] if dominant != "DEFENSIVE_CASH_RAISE" else [
        {"intent": "EARNINGS_CLEAR", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
        {"intent": "STOP_LOSS", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
        {"intent": "TARGET_HIT", "n": 2,
         "total_pnl_usd": 4.0, "avg_pnl_usd": 2.0,
         "avg_pnl_pct": 2.5, "win_rate_pct": 100.0,
         "median_hold_days": 1.5, "n_wins": 2, "n_losses": 0,
         "examples": []},
        {"intent": "THESIS_FLIP", "n": 2,
         "total_pnl_usd": -3.0, "avg_pnl_usd": -1.5,
         "avg_pnl_pct": -1.0, "win_rate_pct": 0.0,
         "median_hold_days": 0.5, "n_wins": 0, "n_losses": 2,
         "examples": []},
        {"intent": "DEFENSIVE_CASH_RAISE", "n": dom_n,
         "total_pnl_usd": dom_total,
         "avg_pnl_usd": dom_total / dom_n if dom_n else None,
         "avg_pnl_pct": dom_avg_pct, "win_rate_pct": dom_wr,
         "median_hold_days": 0.1, "n_wins": 2, "n_losses": 4,
         "examples": []},
        {"intent": "UNCLASSIFIED", "n": 0,
         "total_pnl_usd": 0.0, "avg_pnl_usd": None,
         "avg_pnl_pct": None, "win_rate_pct": None,
         "median_hold_days": None, "n_wins": 0, "n_losses": 0,
         "examples": []},
    ]
    return {
        "as_of": "2026-05-23T14:00:00+00:00",
        "state": "STABLE",
        "verdict": verdict,
        "verdict_reason": None,
        "headline": headline,
        "n_round_trips": 10,
        "dominant_intent": dominant,
        "buckets": buckets,
        "intent_order": ["EARNINGS_CLEAR", "STOP_LOSS", "TARGET_HIT",
                         "THESIS_FLIP", "DEFENSIVE_CASH_RAISE",
                         "UNCLASSIFIED"],
        "stable_min_round_trips": 10,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _exit_intent_audit_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _exit_intent_audit_chat_lines({}) == []
    assert _exit_intent_audit_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize("verdict",
                         ["DOMINANT_INTENT_HEALTHY", None, "OTHER",
                          "NO_DATA"])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _exit_intent_audit_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ─────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "DOMINANT_INTENT_BLEED — dominant exit intent is DEFENSIVE_CASH_RAISE "
        "(6/10 round-trips) but it averages $-2.08/trip (-1.75%) — the most "
        "common reason to sell is also a money loser.")
    out = _exit_intent_audit_chat_lines(_rep(headline=custom))
    assert out[0] == custom              # exact char-for-char passthrough


# ── per-verdict actionability ──────────────────────────────────────────
@pytest.mark.parametrize("verdict",
                         ["DOMINANT_INTENT_BLEED", "INTENT_UNCLEAR"])
def test_actionable_verdicts_emit(verdict):
    out = _exit_intent_audit_chat_lines(_rep(verdict=verdict))
    assert len(out) >= 1
    assert out[0]                          # non-empty headline


def test_detail_line_carries_dominant_bucket_stats():
    out = _exit_intent_audit_chat_lines(_rep())
    assert len(out) == 2
    detail = out[1]
    assert "dominant=DEFENSIVE_CASH_RAISE" in detail
    assert "n=6" in detail
    assert "$-12.50" in detail or "-12.5" in detail  # total_pnl
    assert "-1.75%" in detail                        # avg_pnl_pct
    assert "wr 33%" in detail                        # win-rate rounded


def test_intent_unclear_with_unclassified_dominant():
    rep = _rep(verdict="INTENT_UNCLEAR", dominant="UNCLASSIFIED",
               dom_total=8.0, dom_avg_pct=1.0, dom_wr=80.0)
    out = _exit_intent_audit_chat_lines(rep)
    assert len(out) >= 1
    assert "dominant=UNCLASSIFIED" in out[1] or "UNCLASSIFIED" in out[0]


def test_missing_buckets_list_degrades_silently():
    rep = _rep()
    rep["buckets"] = None
    out = _exit_intent_audit_chat_lines(rep)
    # Headline still emits + detail with dominant tag (from rep itself).
    assert len(out) >= 1
    assert out[0]


def test_missing_dominant_bucket_in_list_degrades():
    rep = _rep()
    # Drop the dominant bucket from the list. Headline still emits;
    # detail only contains the `dominant=` tag.
    rep["buckets"] = [b for b in rep["buckets"]
                      if b["intent"] != rep["dominant_intent"]]
    out = _exit_intent_audit_chat_lines(rep)
    assert len(out) >= 1
    if len(out) > 1:
        assert "dominant=DEFENSIVE_CASH_RAISE" in out[1]


def test_non_numeric_stats_degrade(headline=None):
    rep = _rep()
    # Corrupt the dominant bucket's stats — helper should still emit
    # the headline.
    for b in rep["buckets"]:
        if b["intent"] == rep["dominant_intent"]:
            b["n"] = "six"
            b["total_pnl_usd"] = None
            b["avg_pnl_pct"] = True       # bool is rejected by _num
            b["win_rate_pct"] = []
    out = _exit_intent_audit_chat_lines(rep)
    assert len(out) >= 1
    assert "DOMINANT_INTENT_BLEED" in out[0]
