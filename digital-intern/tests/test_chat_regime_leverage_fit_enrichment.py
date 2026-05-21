"""Pure-helper tests for the /api/chat regime-leverage-fit enrichment.

`_regime_leverage_fit_chat_lines` renders paper-trader's
`/api/regime-leverage-fit-skill` (book-leverage alignment vs prevailing
SPY momentum regime) into compact chat-context lines so the analyst can
answer "are we positioned with or against the regime?" — the highest-
stakes structural question for the leveraged-ETF-heavy watchlist.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_event_readiness_chat_lines` / `_macro_calendar_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **regime-fit book = silence**: ALIGNED / DEFENSIVE / NEUTRAL / NO_DATA
  collapse to `[]`, matching the `_decision_paralysis_chat_lines` silence
  precedent — a chat must not carry "leverage tilt fine" filler.
- **pure/total**: non-dict / missing keys / unparseable values never raise
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

from dashboard.web_server import _regime_leverage_fit_chat_lines


def _rep(verdict="MISSED_TAILWIND", *, headline=None, regime="bull",
         spy_mom=4.22, lev_pct=0.0, lev_usd=0.0, n_lev_positions=0,
         buy_flow_pct=0.0, sell_flow_pct=0.0, flow_window_h=24.0):
    if headline is None:
        if verdict == "MISSED_TAILWIND":
            headline = (
                f"bull tape (spy_mom_20d={spy_mom:.2f}%) but only "
                f"{lev_pct}% leveraged")
        elif verdict == "DANGEROUS_HEADWIND":
            headline = (
                f"levered ({lev_pct}%) into bear "
                f"(spy_mom_20d={spy_mom:.2f}%)")
        elif verdict == "BLIND_LEVERING":
            headline = (
                f"levering into {regime} (spy_mom_20d={spy_mom:.2f}%; "
                f"recent leveraged buy flow {buy_flow_pct}% in "
                f"{flow_window_h:g}h)")
        else:
            headline = "synthetic"
    return {
        "as_of": "2026-05-21T12:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "regime": regime,
        "spy_mom_20d": spy_mom,
        "portfolio": {
            "cash_usd": 341.64,
            "total_value_usd": 1011.95,
            "open_value_usd": 670.30,
            "leveraged_usd": lev_usd,
            "leveraged_pct": lev_pct,
            "n_leveraged_positions": n_lev_positions,
            "leveraged_positions": [],
        },
        "recent_flow": {
            "window_hours": flow_window_h,
            "leveraged_buy_usd": 0.0,
            "leveraged_sell_usd": 0.0,
            "buy_flow_pct": buy_flow_pct,
            "sell_flow_pct": sell_flow_pct,
            "n_leveraged_buys": 0,
            "n_leveraged_sells": 0,
        },
        "thresholds": {
            "bull_mom_pct": 3.0,
            "bear_mom_pct": -3.0,
            "high_lev_floor": 30.0,
            "aligned_lev_floor": 20.0,
            "low_lev_ceil": 10.0,
            "flow_window_hours": 24.0,
            "high_flow_pct": 5.0,
        },
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _regime_leverage_fit_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _regime_leverage_fit_chat_lines({}) == []
    assert _regime_leverage_fit_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict", ["ALIGNED", "DEFENSIVE", "NEUTRAL", "NO_DATA",
                "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _regime_leverage_fit_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "levering into sideways (spy_mom_20d=0.42%; recent leveraged buy "
        "flow 8.5% in 24h)")
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="BLIND_LEVERING", headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── per-verdict actionability ───────────────────────────────────────────
def test_missed_tailwind_emits_regime_and_lev_detail():
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="MISSED_TAILWIND", regime="bull", spy_mom=4.22,
             lev_pct=0.0, lev_usd=0.0, n_lev_positions=0))
    assert len(out) == 2
    body = out[1]
    assert "regime=bull" in body
    assert "spy_mom_20d=4.22%" in body
    assert "leveraged=0.0%" in body


def test_dangerous_headwind_includes_lev_usd_and_position_count():
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="DANGEROUS_HEADWIND", regime="bear", spy_mom=-5.10,
             lev_pct=42.0, lev_usd=425.30, n_lev_positions=2))
    body = "\n".join(out)
    assert "regime=bear" in body
    assert "spy_mom_20d=-5.10%" in body
    assert "leveraged=42.0%" in body
    assert "$425" in body
    assert "2 pos" in body


def test_blind_levering_includes_flow_window():
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="BLIND_LEVERING", regime="sideways", spy_mom=0.42,
             lev_pct=15.0, lev_usd=150.0, n_lev_positions=1,
             buy_flow_pct=8.5, flow_window_h=24.0))
    body = "\n".join(out)
    assert "lev BUY flow 8.5%" in body
    assert "(24h)" in body


def test_zero_flow_omits_flow_fragment():
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="MISSED_TAILWIND", buy_flow_pct=0.0,
             sell_flow_pct=0.0))
    body = "\n".join(out)
    assert "lev BUY flow" not in body
    assert "lev SELL flow" not in body


def test_zero_lev_usd_omits_usd_fragment():
    out = _regime_leverage_fit_chat_lines(
        _rep(verdict="MISSED_TAILWIND", lev_pct=0.0, lev_usd=0.0,
             n_lev_positions=0))
    body = "\n".join(out)
    # leveraged=X% still present, but $ and "N pos" fragments omitted on zero.
    assert "leveraged=0.0%" in body
    assert "$" not in body
    assert "pos" not in body


def test_garbage_fields_do_not_raise():
    rep = _rep(verdict="MISSED_TAILWIND")
    rep["regime"] = 42
    rep["spy_mom_20d"] = "not-a-number"
    rep["portfolio"]["leveraged_pct"] = object()
    rep["recent_flow"] = "not-a-dict"
    out = _regime_leverage_fit_chat_lines(rep)
    # Headline still emitted at minimum, never raises.
    assert out and isinstance(out[0], str)


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(verdict="MISSED_TAILWIND", headline="")
    out = _regime_leverage_fit_chat_lines(rep)
    assert all(not line.startswith("bull tape") for line in out)
    body = "\n".join(out)
    assert "regime=bull" in body


def test_missing_portfolio_and_flow_dicts_degrade_silently():
    rep = _rep(verdict="MISSED_TAILWIND")
    rep["portfolio"] = None
    rep["recent_flow"] = None
    out = _regime_leverage_fit_chat_lines(rep)
    # Headline + regime/spy bit still emitted; lev + flow fragments absent.
    body = "\n".join(out)
    assert out and isinstance(out[0], str)
    assert "regime=bull" in body
    assert "leveraged=" not in body
    assert "lev BUY flow" not in body
