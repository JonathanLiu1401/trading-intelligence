"""Pure-helper tests for the /api/chat passive-signal-density enrichment.

`_passive_signal_density_chat_lines` renders paper-trader's
`/api/passive-signal-density` (median news-signal count over the current
HOLD-only run — discriminates "informed passive: quiet news = correct
silence" from "deafening silence: loud news + idle engine") into compact
chat-context lines so the analyst can answer "is the bot quiet for the
right reason, or did it sit through real news?" — the structural
follow-up that decision-paralysis (length-only) and idle-opportunity
(forward signals) leave unanswered.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_no_decision_reasons_chat_lines` / `_round_trip_postmortem_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no
:8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **healthy / unscored = silence**: INFORMED_PASSIVE /
  SIGNAL_RICH_PASSIVE / NO_PASSIVE_RUN / INSUFFICIENT / NO_DATA all
  collapse to ``[]`` — only DEAFENING_SILENCE is an alert, mirroring
  the trader-side Discord block contract
  (``reporter._passive_signal_density_line`` — the two surfaces never
  disagree on what is the alert).
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

from dashboard.web_server import _passive_signal_density_chat_lines


def _rep(
    verdict="DEAFENING_SILENCE",
    *,
    headline=None,
    median_signal_count=12.0,
    n_passive=38,
    high_signal_threshold=10,
):
    if headline is None:
        headline = (
            "DEAFENING_SILENCE — 38 passive cycles with median 12.0 "
            "signals/cycle (>10). Engine sat through a loud news window."
        )
    return {
        "verdict": verdict,
        "state": "STABLE",
        "headline": headline,
        "median_signal_count": median_signal_count,
        "n_passive": n_passive,
        "high_signal_threshold": high_signal_threshold,
        "low_signal_median": 3,
        "min_passive_run": 5,
        "n_total_scanned": 377,
        "recent_signal_counts": [12, 14, 18, 11, 13],
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _passive_signal_density_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _passive_signal_density_chat_lines({}) == []
    assert _passive_signal_density_chat_lines({"headline": "anything"}) == []


# ── healthy / unscored = silence ────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict",
    [
        "INFORMED_PASSIVE",
        "SIGNAL_RICH_PASSIVE",
        "NO_PASSIVE_RUN",
        "INSUFFICIENT",
        "NO_DATA",
        None,
        "",
        "UNKNOWN_VERDICT",
    ],
)
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _passive_signal_density_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "DEAFENING_SILENCE — 99 passive cycles with median 42 "
        "signals/cycle (>10). The engine sat through a multi-day "
        "news firestorm."
    )
    out = _passive_signal_density_chat_lines(_rep(headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── detail line content ─────────────────────────────────────────────────
def test_detail_line_restates_median_and_passive_count():
    out = _passive_signal_density_chat_lines(
        _rep(median_signal_count=12.0, n_passive=38, high_signal_threshold=10)
    )
    body = "\n".join(out)
    assert "median 12 signals/cycle" in body
    assert "38 passive cycles" in body
    assert "high-signal floor >10" in body


def test_detail_line_with_non_integer_median():
    out = _passive_signal_density_chat_lines(
        _rep(median_signal_count=11.5, n_passive=20, high_signal_threshold=10)
    )
    body = "\n".join(out)
    # 11.5 should render with the fractional component (`:g`).
    assert "median 11.5 signals/cycle" in body


def test_detail_line_skips_missing_fields():
    rep = _rep(headline="DEAFENING_SILENCE — engine idle on loud wire.")
    rep.pop("median_signal_count", None)
    rep.pop("n_passive", None)
    out = _passive_signal_density_chat_lines(rep)
    body = "\n".join(out)
    # Only the threshold should appear; missing fields skipped silently.
    assert "high-signal floor >10" in body
    assert "median" not in body
    assert "passive cycles" not in body


def test_detail_line_skips_unparseable_fields():
    rep = _rep(
        median_signal_count="x",
        n_passive=True,                # bool must be rejected
        high_signal_threshold=None,
    )
    out = _passive_signal_density_chat_lines(rep)
    # Headline only; detail line is suppressed because no parts survived.
    assert len(out) == 1


def test_no_numeric_fields_emits_headline_only():
    rep = {
        "verdict": "DEAFENING_SILENCE",
        "headline": "DEAFENING_SILENCE — engine idle on loud wire.",
    }
    out = _passive_signal_density_chat_lines(rep)
    assert out == ["DEAFENING_SILENCE — engine idle on loud wire."]


# ── headline-empty fallback ─────────────────────────────────────────────
def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(headline="")
    out = _passive_signal_density_chat_lines(rep)
    body = "\n".join(out)
    assert body.startswith("  ")        # only the indented detail line
    assert "median 12 signals/cycle" in body


def test_garbage_headline_omits_first_line():
    rep = _rep(headline=42)
    out = _passive_signal_density_chat_lines(rep)
    body = "\n".join(out)
    assert "median 12 signals/cycle" in body


# ── shape stability ─────────────────────────────────────────────────────
def test_returns_list_always():
    assert isinstance(_passive_signal_density_chat_lines({}), list)
    assert isinstance(_passive_signal_density_chat_lines(None), list)
    assert isinstance(_passive_signal_density_chat_lines(_rep()), list)


def test_single_verdict_contract_matches_trader_discord_block():
    """The trader-side Discord block (reporter._passive_signal_density_line)
    ships ONLY on DEAFENING_SILENCE; this chat block must match that
    single-verdict contract so the two surfaces never disagree on what
    is "the alert"."""
    # Every other documented verdict in the ladder must be silent.
    for v in [
        "NO_PASSIVE_RUN",
        "INSUFFICIENT",
        "INFORMED_PASSIVE",
        "SIGNAL_RICH_PASSIVE",
        "NO_DATA",
    ]:
        assert _passive_signal_density_chat_lines(_rep(verdict=v)) == [], v
    # Only DEAFENING_SILENCE emits output.
    assert _passive_signal_density_chat_lines(
        _rep(verdict="DEAFENING_SILENCE")
    ) != []
