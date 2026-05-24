"""Pure-helper tests for the /api/chat no-decision-reasons enrichment.

`_no_decision_reasons_chat_lines` renders paper-trader's
`/api/no-decision-reasons` (per-bucket histogram of recent NO_DECISION
causes — host_saturated / cli_nonzero_rc / parse_failed / claude_timeout
/ claude_empty / blocked / unknown) into compact chat-context lines so
the analyst can answer "WHY is the bot silent right now?" without
parsing daemon logs — exactly the question the existing
decision-paralysis (the FACT) and runner-heartbeat (availability)
blocks leave open.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_redeployment_chat_lines` /
`_decision_vapor_chat_lines` / `_decision_paralysis_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` (which already inlines the
  ``recommendation`` verbatim) passes through UNCHANGED — no chat-side
  re-derived verdict, no chat-side re-derived recommendation, that
  could drift from the trader endpoint.
- **healthy/diffuse = silence**: NO_DATA / NORMAL / MIXED collapse to
  ``[]``, matching the ``_decision_paralysis_chat_lines`` silence
  precedent — the chat must not carry "everything fine" filler, nor a
  hand-wavy bucket histogram when no single cause owns the wedge.
- **pure/total**: non-dict / missing keys / unparseable bucket counts
  never raise and degrade to silence or the safe subset (the
  ``_paper_trader_position_lines`` precedent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _no_decision_reasons_chat_lines


def _rep(state="DOMINANT", *, headline=None, buckets=None, n_decisions=50,
         n_no_decision=20, dominant_bucket="host_saturated",
         dominant_pct=95.0):
    if headline is None:
        headline = (
            f"{n_no_decision}/{n_decisions} cycles NO_DECISION; "
            f"dominant cause: {dominant_bucket} ({dominant_pct:.0f}%) — "
            "Host saturation — too many concurrent Opus subprocesses "
            "(review agents / backtest committee). Reduce parallel "
            "Opus jobs, or wait for the storm to clear; a runner "
            "restart does NOT help.")
    if buckets is None:
        buckets = {"host_saturated": 19, "cli_nonzero_rc": 1}
    return {
        "state": state,
        "headline": headline,
        "buckets": buckets,
        "dominant_bucket": dominant_bucket,
        "dominant_pct": dominant_pct,
        "n_decisions": n_decisions,
        "n_no_decision": n_no_decision,
        "no_decision_pct": round(n_no_decision / n_decisions * 100, 1),
        "recommendation": (
            "Host saturation — too many concurrent Opus subprocesses "
            "(review agents / backtest committee). Reduce parallel Opus "
            "jobs, or wait for the storm to clear; a runner restart does "
            "NOT help."),
        "window": n_decisions,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _no_decision_reasons_chat_lines(bad) == []


def test_missing_state_is_silence():
    assert _no_decision_reasons_chat_lines({}) == []
    assert _no_decision_reasons_chat_lines({"headline": "x"}) == []


# ── healthy/diffuse = silence ───────────────────────────────────────────
@pytest.mark.parametrize(
    "state", ["NO_DATA", "NORMAL", "MIXED", "OTHER", None, ""])
def test_non_actionable_states_silence(state):
    rep = _rep(state=state)
    assert _no_decision_reasons_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "20/50 cycles NO_DECISION; dominant cause: parse_failed (85%) — "
        "Prompt shape regression — recent commits to strategy.py probably "
        "broke the JSON-only suffix; revert / verify _build_payload.")
    out = _no_decision_reasons_chat_lines(_rep(headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── bucket histogram detail line ────────────────────────────────────────
def test_bucket_histogram_emits_top_three():
    rep = _rep(buckets={
        "host_saturated": 19, "cli_nonzero_rc": 1,
        "claude_timeout": 5, "parse_failed": 2,
    })
    out = _no_decision_reasons_chat_lines(rep)
    assert len(out) >= 2
    # Ranked by count desc; only the top 3 surface.
    detail = out[1]
    assert "host_saturated: 19" in detail
    assert "claude_timeout: 5" in detail
    assert "parse_failed: 2" in detail
    # cli_nonzero_rc=1 is the 4th — should NOT appear.
    assert "cli_nonzero_rc" not in detail
    # And the order should be desc — host_saturated must come first.
    pos_host = detail.index("host_saturated")
    pos_timeout = detail.index("claude_timeout")
    pos_parse = detail.index("parse_failed")
    assert pos_host < pos_timeout < pos_parse


def test_zero_count_buckets_omitted():
    rep = _rep(buckets={
        "host_saturated": 19, "claude_empty": 0, "blocked": 0,
    })
    out = _no_decision_reasons_chat_lines(rep)
    detail = out[1] if len(out) >= 2 else ""
    assert "host_saturated: 19" in detail
    assert "claude_empty" not in detail
    assert "blocked" not in detail


def test_missing_buckets_omits_detail_line():
    rep = _rep()
    rep.pop("buckets", None)
    out = _no_decision_reasons_chat_lines(rep)
    # Headline emitted; no detail line raised.
    assert len(out) == 1


def test_empty_buckets_omits_detail_line():
    rep = _rep(buckets={})
    out = _no_decision_reasons_chat_lines(rep)
    assert len(out) == 1
    assert "bucket counts" not in out[0]


def test_non_dict_buckets_degrades_silently():
    rep = _rep()
    rep["buckets"] = ["host_saturated", 19]
    out = _no_decision_reasons_chat_lines(rep)
    assert len(out) == 1            # headline only — no detail line
    assert isinstance(out[0], str)


# ── garbage-input robustness ────────────────────────────────────────────
def test_unparseable_bucket_counts_skipped():
    rep = _rep(buckets={
        "host_saturated": 19,
        "claude_timeout": "x",         # non-numeric
        "blocked": None,
        "unknown": True,                # bool — explicitly rejected
        "parse_failed": 3,
    })
    out = _no_decision_reasons_chat_lines(rep)
    detail = out[1] if len(out) >= 2 else ""
    assert "host_saturated: 19" in detail
    assert "parse_failed: 3" in detail
    assert "claude_timeout" not in detail
    assert "blocked" not in detail
    assert "unknown" not in detail


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(headline="")
    out = _no_decision_reasons_chat_lines(rep)
    # Headline is empty so only the detail line should remain — and the
    # detail line must not include the empty headline string.
    assert all(not line.startswith("20/50 cycles") for line in out)
    body = "\n".join(out)
    assert "host_saturated: 19" in body


def test_garbage_headline_omits_first_line():
    rep = _rep()
    rep["headline"] = 42                # non-string
    out = _no_decision_reasons_chat_lines(rep)
    # Detail line should still render — never raises.
    body = "\n".join(out)
    assert "host_saturated: 19" in body


def test_returns_list_always():
    # Pure/total contract — the return type is a list[str] under all paths.
    assert isinstance(_no_decision_reasons_chat_lines({}), list)
    assert isinstance(_no_decision_reasons_chat_lines(None), list)
    assert isinstance(_no_decision_reasons_chat_lines(_rep()), list)
