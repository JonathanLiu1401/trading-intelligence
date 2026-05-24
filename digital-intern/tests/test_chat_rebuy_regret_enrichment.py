"""Pure-helper tests for the /api/chat rebuy-regret enrichment.

``_rebuy_regret_chat_lines`` renders paper-trader's ``/api/rebuy-regret``
(the DOLLAR sell-then-rebuy regret quantifier — did the desk save or
lose money on close→re-entry hops?) into compact chat-context lines.

Discriminating locks (mirroring the
``test_chat_decision_paralysis_enrichment.py`` shape):

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict that could drift from the trader
  endpoint.
- **saving / flat record = silence**: SAVINGS / NET_NEUTRAL / NO_DATA /
  NO_REBUYS / ERROR collapse to ``[]``, matching the
  ``_decision_paralysis_chat_lines`` silence precedent — a re-entry
  record that saves money or nets flat is not chat filler.
- **worst-ticker selection discriminator**: among ``per_ticker`` rows
  the surfaced detail picks the MAX ``net_regret_usd`` (only when
  positive, since negative regret = SAVINGS and would contradict the
  REGRETTING headline).
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

from dashboard.web_server import _rebuy_regret_chat_lines


def _rep(
    verdict="REGRETTING",
    *,
    headline=None,
    n_events=3,
    net_regret_usd=42.50,
    per_ticker=None,
):
    if headline is None:
        headline = (
            f"{verdict} — sold low and bought back higher by "
            f"${net_regret_usd:,.2f} net over {n_events} re-entry event(s)."
        )
    if per_ticker is None:
        per_ticker = [
            {"ticker": "NVDA", "n_events": 2,
             "net_regret_usd": 30.0, "worst_regret_usd": 25.0,
             "best_savings_usd": 0.0, "last_regret_usd": 15.0,
             "last_classification": "REGRET"},
            {"ticker": "AMD", "n_events": 1,
             "net_regret_usd": 12.5, "worst_regret_usd": 12.5,
             "best_savings_usd": 0.0, "last_regret_usd": 12.5,
             "last_classification": "REGRET"},
        ]
    return {
        "as_of": "2026-05-24T14:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "n_events": n_events,
        "n_round_trips": n_events + 1,
        "net_regret_usd": net_regret_usd,
        "median_regret_usd": net_regret_usd / max(n_events, 1),
        "best_savings_usd": 0.0,
        "neutral_event_count": 0,
        "per_ticker": per_ticker,
        "recent_events": [],
    }


# ── pure / total contract ───────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _rebuy_regret_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _rebuy_regret_chat_lines({}) == []


# ── healthy / flat / no-data = silence ──────────────────────────────────
@pytest.mark.parametrize("v", ["SAVINGS", "NET_NEUTRAL", "NO_DATA",
                               "NO_REBUYS", "ERROR", None, "OTHER"])
def test_non_actionable_verdicts_silence(v):
    rep = _rep()
    rep["verdict"] = v
    assert _rebuy_regret_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "REGRETTING — $87.20 net regret across 5 re-entry event(s) "
        "(4 regret, 0 saved, 1 neutral)."
    )
    out = _rebuy_regret_chat_lines(_rep(headline=custom))
    assert out[0] == custom


# ── actionable detail line composition ──────────────────────────────────
def test_regretting_emits_worst_ticker_detail():
    out = _rebuy_regret_chat_lines(_rep())
    assert len(out) == 2
    body = out[1]
    assert "NVDA" in body
    assert "$30" in body
    assert "2 event(s)" in body


def test_worst_ticker_picks_highest_net_regret():
    per_ticker = [
        {"ticker": "ZZZ", "n_events": 1, "net_regret_usd": 5.0,
         "worst_regret_usd": 5.0, "last_classification": "REGRET"},
        {"ticker": "AAA", "n_events": 7, "net_regret_usd": 99.0,
         "worst_regret_usd": 75.0, "last_classification": "REGRET"},
        {"ticker": "MID", "n_events": 2, "net_regret_usd": 40.0,
         "worst_regret_usd": 30.0, "last_classification": "REGRET"},
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    body = out[1]
    assert "AAA" in body
    assert "$99" in body
    assert "ZZZ" not in body
    assert "MID" not in body


def test_zero_net_regret_per_ticker_omits_detail_line():
    # If somehow every per_ticker is net 0/negative the detail line is dropped
    per_ticker = [
        {"ticker": "AAA", "n_events": 1, "net_regret_usd": 0.0,
         "last_classification": "NEUTRAL"},
        {"ticker": "BBB", "n_events": 1, "net_regret_usd": -5.0,
         "last_classification": "SAVED"},
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    # Headline still emits; no detail line because no positive-regret row
    assert len(out) == 1


def test_missing_n_events_omits_event_count_fragment():
    per_ticker = [
        {"ticker": "NVDA", "net_regret_usd": 25.0,
         "last_classification": "REGRET"},  # n_events missing
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    body = out[1]
    assert "NVDA" in body
    assert "$25" in body
    assert "event(s)" not in body


# ── degraded inputs degrade silently, never raise ──────────────────────
def test_garbage_per_ticker_does_not_raise():
    rep = _rep(per_ticker=[
        None, "string", 42, ["not", "dict"],
        {"ticker": "OK", "net_regret_usd": 17.0, "n_events": 2,
         "last_classification": "REGRET"},
    ])
    out = _rebuy_regret_chat_lines(rep)
    body = "\n".join(out)
    assert "OK" in body
    assert "$17" in body


def test_per_ticker_not_a_list_omits_detail():
    rep = _rep()
    rep["per_ticker"] = "not a list"
    out = _rebuy_regret_chat_lines(rep)
    # Headline still emits, detail omitted
    assert len(out) == 1


def test_per_ticker_missing_omits_detail():
    rep = _rep()
    rep.pop("per_ticker")
    out = _rebuy_regret_chat_lines(rep)
    assert len(out) == 1


def test_garbage_net_regret_per_ticker_ignored():
    per_ticker = [
        {"ticker": "BAD1", "net_regret_usd": "huge", "n_events": 1,
         "last_classification": "REGRET"},
        {"ticker": "BAD2", "net_regret_usd": None, "n_events": 1,
         "last_classification": "REGRET"},
        {"ticker": "GOOD", "net_regret_usd": 8.5, "n_events": 1,
         "last_classification": "REGRET"},
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    body = out[1]
    assert "GOOD" in body
    assert "$8.50" in body
    assert "BAD" not in body


def test_bool_per_ticker_regret_not_treated_as_number():
    per_ticker = [
        {"ticker": "FAKE", "net_regret_usd": True, "n_events": 1,
         "last_classification": "REGRET"},
        {"ticker": "REAL", "net_regret_usd": 3.0, "n_events": 1,
         "last_classification": "REGRET"},
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    body = out[1]
    assert "REAL" in body
    assert "FAKE" not in body


def test_empty_headline_omits_first_line():
    rep = _rep(headline="")
    out = _rebuy_regret_chat_lines(rep)
    # Empty-string row filtered; detail still renders.
    assert "" not in out
    assert any("NVDA" in line for line in out)


def test_missing_ticker_in_worst_row_renders_placeholder():
    per_ticker = [
        {"net_regret_usd": 50.0, "n_events": 1,
         "last_classification": "REGRET"},  # no ticker
    ]
    out = _rebuy_regret_chat_lines(_rep(per_ticker=per_ticker))
    body = out[1]
    assert "?" in body
    assert "$50" in body
