"""Pure-helper tests for the /api/chat notify-health enrichment.

`_notify_health_chat_lines` renders paper-trader's `/api/notify-health`
(the Discord-channel delivery health of the live trader) into compact
chat-context lines so the analyst can flag "the trader can't reach
Discord right now" — the operator-fitness layer every existing
book/decision/skill block silently assumes is working.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_cash_redeployment_chat_lines` / `_decision_paralysis_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **healthy = silence**: HEALTHY / UNKNOWN / missing-verdict collapse
  to ``[]`` — a working Discord channel must not become chat filler.
- **detail line fields**: when DEGRADED, the detail line restates the
  builder's own ``consecutive_failures`` / ``last_error`` /
  ``restart_recommended`` verbatim — never a recomputation.
- **defensive long-error cap**: a multi-line traceback in
  ``last_error`` is truncated so the chat block can't be blown up by
  a pathological upstream value.
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

from dashboard.web_server import _notify_health_chat_lines


def _rep(verdict="DEGRADED", *, headline=None, consecutive_failures=1,
         last_error="openclaw timeout (60s)", restart_recommended=False,
         last_ok_ts=None, last_attempt_ts="2026-05-25T00:13:29.652642+00:00"):
    if headline is None:
        if verdict == "DEGRADED":
            headline = (
                f"Discord channel DARK — {consecutive_failures} consecutive "
                f"send failure, last OK never; last error: {last_error}"
            )
        elif verdict == "HEALTHY":
            headline = "Discord delivery healthy — last send OK."
        else:
            headline = "notify-health: unknown state."
    return {
        "as_of": "2026-05-25T00:22:13+00:00",
        "consecutive_failures": consecutive_failures,
        "headline": headline,
        "last_attempt_ts": last_attempt_ts,
        "last_error": last_error,
        "last_ok_ts": last_ok_ts,
        "restart_recommended": restart_recommended,
        "service": "paper_trader",
        "verdict": verdict,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _notify_health_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _notify_health_chat_lines({}) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize("v", ["HEALTHY", "UNKNOWN", "OK", "", None])
def test_non_degraded_verdicts_collapse_to_silence(v):
    assert _notify_health_chat_lines(_rep(verdict=v)) == []


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_headline_passes_through_verbatim():
    rep = _rep(headline="Discord channel DARK — 7 consecutive send failure(s), "
                        "last OK 2026-05-23T11:04:13+00:00; last error: HTTP 503")
    lines = _notify_health_chat_lines(rep)
    assert lines, "DEGRADED with a headline must produce ≥1 line"
    assert lines[0] == (
        "Discord channel DARK — 7 consecutive send failure(s), "
        "last OK 2026-05-23T11:04:13+00:00; last error: HTTP 503"
    )


def test_missing_headline_degrades_to_detail_only():
    rep = _rep(headline=None)
    rep["headline"] = None
    lines = _notify_health_chat_lines(rep)
    # detail line still surfaces from the consecutive_failures + last_error
    # restatement — no crash, no fabricated headline.
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "consecutive_failures=1" in lines[0]


def test_non_string_headline_falls_back_to_detail_only():
    rep = _rep()
    rep["headline"] = 42
    lines = _notify_health_chat_lines(rep)
    assert all("Discord channel DARK" not in ln for ln in lines)
    assert any("consecutive_failures=" in ln for ln in lines)


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_contains_all_three_fields():
    rep = _rep(consecutive_failures=3, last_error="HTTP 503",
               restart_recommended=True)
    lines = _notify_health_chat_lines(rep)
    assert len(lines) == 2
    detail = lines[1]
    assert detail.startswith("  ")
    assert "consecutive_failures=3" in detail
    assert "HTTP 503" in detail
    assert "restart_recommended=YES" in detail


def test_detail_line_restart_no_when_false():
    lines = _notify_health_chat_lines(_rep(restart_recommended=False))
    assert "restart_recommended=no" in lines[1]


def test_detail_line_omits_missing_fields():
    rep = _rep()
    rep["consecutive_failures"] = None
    rep["last_error"] = None
    rep["restart_recommended"] = None
    lines = _notify_health_chat_lines(rep)
    # Headline still present; detail line is suppressed because no
    # safe field survives — no empty "  " marker line either.
    assert lines == [rep["headline"]]


def test_long_last_error_is_truncated():
    """A pathological multi-line traceback must not blow up the chat block."""
    long_err = "Traceback (most recent call last):\n" + ("x" * 500)
    rep = _rep(last_error=long_err)
    lines = _notify_health_chat_lines(rep)
    assert len(lines) == 2
    detail = lines[1]
    assert "..." in detail
    # The detail line must not contain the entire long string.
    assert len(detail) < 400


def test_consecutive_failures_zero_still_emitted_as_int():
    """The DEGRADED path is the only one this helper sees; trust the
    builder's count. A 0 here would be an upstream bug worth surfacing."""
    rep = _rep(consecutive_failures=0)
    lines = _notify_health_chat_lines(rep)
    assert "consecutive_failures=0" in lines[1]


def test_bool_consecutive_failures_treated_as_missing():
    """Defensive: bool is_a int in Python; never let True/False slip
    through as an int count."""
    rep = _rep()
    rep["consecutive_failures"] = True
    lines = _notify_health_chat_lines(rep)
    assert all("consecutive_failures" not in ln for ln in lines)


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_fixture_2026_05_25():
    """The exact `/api/notify-health` shape observed 2026-05-25 — the
    pathology this enrichment exists to surface in chat."""
    rep = {
        "as_of": "2026-05-25T00:22:13+00:00",
        "consecutive_failures": 1,
        "headline": (
            "Discord channel DARK — 1 consecutive send failure, last OK "
            "never; last error: openclaw timeout (60s)"
        ),
        "last_attempt_ts": "2026-05-25T00:13:29.652642+00:00",
        "last_error": "openclaw timeout (60s)",
        "last_ok_ts": None,
        "restart_recommended": False,
        "service": "paper_trader",
        "verdict": "DEGRADED",
    }
    lines = _notify_health_chat_lines(rep)
    assert lines[0] == rep["headline"]
    assert "consecutive_failures=1" in lines[1]
    assert "openclaw timeout" in lines[1]
    assert "restart_recommended=no" in lines[1]
