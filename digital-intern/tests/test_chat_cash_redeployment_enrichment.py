"""Pure-helper tests for the /api/chat cash-redeployment enrichment.

`_cash_redeployment_chat_lines` renders paper-trader's
`/api/cash-redeployment-latency-skill` (post-SELL cash-to-next-BUY latency
distribution — the sold-then-sat pathology) into compact chat-context
lines so the analyst can answer "did we sell into a bad thesis then sit
on the cash for days?" when the headline cash% looks fine.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_event_readiness_chat_lines` / `_macro_calendar_chat_lines`) the logic
is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own `headline` string passes through UNCHANGED — no chat-side re-derived
  verdict that could drift from the trader endpoint.
- **healthy cadence = silence**: FAST_REDEPLOY / STEADY / NO_DATA collapse
  to `[]`, matching the `_decision_paralysis_chat_lines` silence precedent
  — a chat must not carry "cash redeployment fine" filler.
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

from dashboard.web_server import _cash_redeployment_chat_lines


def _rep(verdict="STALLED", *, headline=None, median_h=120.0,
         p25_h=60.0, p75_h=200.0, n_stalled=2, n_classifiable=5,
         n_redeployed=3, total_freed=1500.0, total_redeployed=900.0):
    if headline is None:
        headline = (
            f"stalled: median {median_h:.1f}h, only "
            f"{n_redeployed}/{n_classifiable} redeployed "
            f"({(n_redeployed/n_classifiable)*100:.0f}%)")
    return {
        "as_of": "2026-05-21T12:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "window_days": 14.0,
        "stats": {
            "n_sells_total": n_classifiable,
            "n_classifiable": n_classifiable,
            "n_redeployed": n_redeployed,
            "n_stalled": n_stalled,
            "n_window_edge": 0,
            "redeploy_pct": round((n_redeployed/n_classifiable)*100, 2),
            "median_latency_h": median_h,
            "p25_latency_h": p25_h,
            "p75_latency_h": p75_h,
            "total_freed_usd": total_freed,
            "total_redeployed_usd": total_redeployed,
        },
        "thresholds": {
            "fast_median_h": 6.0,
            "steady_median_h": 24.0,
            "slow_median_h": 72.0,
            "healthy_redeploy_pct": 80.0,
            "steady_redeploy_pct": 70.0,
            "degraded_redeploy_pct": 50.0,
            "stalled_cutoff_h": 168.0,
            "min_sells_for_verdict": 3,
        },
        "pairs": [],
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _cash_redeployment_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _cash_redeployment_chat_lines({}) == []
    assert _cash_redeployment_chat_lines({"headline": "x"}) == []


# ── healthy = silence ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict", ["FAST_REDEPLOY", "STEADY", "NO_DATA", "OTHER", None])
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _cash_redeployment_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "stalled: 4/6 SELLs never redeployed within 168h "
        "(median latency undefined)")
    out = _cash_redeployment_chat_lines(
        _rep(headline=custom, verdict="STALLED"))
    assert out[0] == custom            # exact char-for-char passthrough


# ── per-verdict actionability ───────────────────────────────────────────
def test_stalled_emits_full_detail():
    out = _cash_redeployment_chat_lines(
        _rep(verdict="STALLED", median_h=120.0, p25_h=60.0,
             p75_h=200.0, n_stalled=2, n_classifiable=5,
             total_freed=1500.0, total_redeployed=900.0))
    assert len(out) == 2
    body = out[1]
    assert "p25/median/p75 = 60.0/120.0/200.0h" in body
    assert "2/5 SELLs never redeployed" in body
    assert "$600 freed but unworked" in body


def test_slow_emits_headline_and_detail():
    out = _cash_redeployment_chat_lines(
        _rep(verdict="SLOW", median_h=48.0, p25_h=24.0, p75_h=72.0,
             n_stalled=0, n_classifiable=4, total_freed=800.0,
             total_redeployed=800.0))
    body = "\n".join(out)
    assert "p25/median/p75 = 24.0/48.0/72.0h" in body
    # No stalled SELLs and no idle cash → those fragments are omitted.
    assert "never redeployed" not in body
    assert "freed but unworked" not in body


def test_median_without_percentiles_still_renders():
    rep = _rep(verdict="STALLED", median_h=80.0)
    rep["stats"]["p25_latency_h"] = None
    rep["stats"]["p75_latency_h"] = None
    out = _cash_redeployment_chat_lines(rep)
    body = "\n".join(out)
    assert "median latency 80.0h" in body
    assert "p25" not in body


def test_no_stalled_sells_omits_stalled_fragment():
    out = _cash_redeployment_chat_lines(
        _rep(verdict="SLOW", n_stalled=0, n_classifiable=3))
    body = "\n".join(out)
    assert "never redeployed" not in body


def test_idle_cash_only_when_nonzero():
    out = _cash_redeployment_chat_lines(
        _rep(verdict="SLOW", total_freed=500.0, total_redeployed=500.0))
    body = "\n".join(out)
    assert "freed but unworked" not in body


def test_garbage_stats_do_not_raise():
    rep = _rep(verdict="STALLED")
    rep["stats"]["median_latency_h"] = "not-a-number"
    rep["stats"]["p25_latency_h"] = None
    rep["stats"]["n_stalled"] = "x"
    rep["stats"]["total_freed_usd"] = object()
    out = _cash_redeployment_chat_lines(rep)
    # Headline still emitted at minimum, never raises.
    assert out and isinstance(out[0], str)


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(verdict="STALLED", headline="")
    out = _cash_redeployment_chat_lines(rep)
    assert all(not line.startswith("stalled:") for line in out)
    body = "\n".join(out)
    assert "p25/median/p75" in body


def test_missing_stats_dict_degrades_silently():
    rep = _rep(verdict="STALLED")
    rep["stats"] = None
    out = _cash_redeployment_chat_lines(rep)
    # Headline still emitted; detail line absent — no exception raised.
    assert out and isinstance(out[0], str)
    assert all("p25/median/p75" not in line for line in out)
