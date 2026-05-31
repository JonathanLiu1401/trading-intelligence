"""Pure-helper tests for the /api/chat repeat-loser enrichment.

``_repeat_loser_chat_lines`` renders paper-trader's ``/api/repeat-loser``
(the per-ticker chronic-pattern read — tickers where the bot has lost
the last N closed round-trips in a row on the same name) into compact
chat lines so the analyst can flag the structural blind spot every
aggregate realised-P&L surface erases: "you've lost the last 3 trips
on MU in a row".

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict or threshold restatement.
- **healthy = silence**: state OK / NO_DATA collapse to ``[]`` — a
  book with no offender must never become chat filler.
- **detail line fields**: when actionable, the detail line restates
  the worst offender's ``ticker`` / ``current_loss_streak`` /
  ``current_loss_usd`` / ``last_loss_exit_ts`` verbatim plus the
  builder's own ``threshold`` — never a recomputation.
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

from dashboard.web_server import _repeat_loser_chat_lines


def _rep(
    state="REPEAT_LOSER",
    *,
    headline=None,
    ticker="MU",
    streak=3,
    loss_usd=-45.50,
    last_exit="2026-05-28T14:02:43.368432+00:00",
    threshold=2,
    n_offenders=1,
    n_round_trips=9,
):
    if headline is None:
        headline = (
            f"REPEAT_LOSER — {ticker} on a {streak}-trip loss run "
            f"(net ${loss_usd:.2f}). Threshold {threshold}."
        )
    offenders = []
    if n_offenders > 0:
        offenders.append({
            "ticker": ticker,
            "current_loss_streak": streak,
            "current_loss_usd": loss_usd,
            "last_loss_exit_ts": last_exit,
            "n_round_trips": streak,
        })
    return {
        "state": state,
        "verdict": "REPEAT_LOSER" if n_offenders > 0 else None,
        "headline": headline,
        "n_offenders": n_offenders,
        "n_round_trips": n_round_trips,
        "threshold": threshold,
        "offenders": offenders,
        "per_ticker": {ticker: {
            "current_loss_streak": streak,
            "current_loss_usd": loss_usd,
            "last_loss_exit_ts": last_exit,
            "n_round_trips": streak,
        }},
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _repeat_loser_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _repeat_loser_chat_lines({}) == []


# ── silence on non-actionable states ────────────────────────────────────
@pytest.mark.parametrize(
    "s", ["OK", "NO_DATA", "", None, "WHATEVER", "ok"],  # case-sensitive
)
def test_non_actionable_states_collapse_to_silence(s):
    assert _repeat_loser_chat_lines(_rep(state=s)) == []


def test_actionable_state_emits_lines():
    lines = _repeat_loser_chat_lines(_rep(state="REPEAT_LOSER"))
    assert lines, "REPEAT_LOSER must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_actionable_headline_passes_through_verbatim():
    rep = _rep(
        headline=(
            "REPEAT_LOSER — MU on a 3-trip loss run (net $-45.50). "
            "Threshold 2."
        ),
    )
    assert _repeat_loser_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _repeat_loser_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "MU" in lines[0]


def test_empty_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = "   "
    lines = _repeat_loser_chat_lines(rep)
    # whitespace-only headline is rejected; only detail line remains
    assert len(lines) == 1
    assert lines[0].startswith("  ")


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full():
    rep = _rep(
        ticker="MU", streak=3, loss_usd=-45.50,
        last_exit="2026-05-28T14:02:43.368432+00:00",
        threshold=2,
    )
    detail = _repeat_loser_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "MU" in detail
    assert "3L in a row" in detail
    assert "-45.50" in detail or "$-45.50" in detail
    assert "2026-05-28" in detail
    assert "threshold 2" in detail


def test_detail_line_omits_missing_ticker():
    rep = _rep()
    rep["offenders"] = [{
        "current_loss_streak": 3,
        "current_loss_usd": -10.0,
    }]
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "3L in a row" in detail
    assert "MU" not in detail


def test_detail_line_omits_missing_streak():
    rep = _rep()
    rep["offenders"] = [{"ticker": "MU", "current_loss_usd": -10.0}]
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "MU" in detail
    assert "in a row" not in detail


def test_detail_line_omits_empty_offenders_list():
    rep = _rep()
    rep["offenders"] = []
    # No offender means no detail line, only the headline
    lines = _repeat_loser_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0] == rep["headline"]


def test_detail_line_skips_non_dict_offenders():
    rep = _rep()
    rep["offenders"] = ["not a dict", None, {"ticker": "MU", "current_loss_streak": 4}]
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "MU" in detail
    assert "4L in a row" in detail


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_streak_treated_as_missing():
    """bool is int in Python; never let True/False render as a streak count."""
    rep = _rep()
    rep["offenders"][0]["current_loss_streak"] = True
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "1L in a row" not in detail
    assert "True" not in detail


def test_bool_loss_usd_treated_as_missing():
    rep = _rep()
    rep["offenders"][0]["current_loss_usd"] = False
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "$0.00" not in detail
    assert "False" not in detail


def test_string_loss_usd_treated_as_missing():
    rep = _rep()
    rep["offenders"][0]["current_loss_usd"] = "-45.50"
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "-45.50" not in detail
    assert "MU" in detail  # still has ticker


def test_bool_threshold_treated_as_missing():
    rep = _rep()
    rep["threshold"] = True
    detail = _repeat_loser_chat_lines(rep)[1]
    assert "threshold True" not in detail
    assert "threshold 1" not in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_31_ok_is_silent():
    """The current live OK response — confirm we don't push chat filler."""
    rep = {
        "as_of": "2026-05-31T05:25:15+00:00",
        "headline": "OK — no ticker on a ≥2-loss run across 9 closed trips.",
        "n_offenders": 0,
        "n_round_trips": 9,
        "offenders": [],
        "per_ticker": {
            "AMD": {"current_loss_streak": 1, "current_loss_usd": -13.835,
                    "last_loss_exit_ts": "2026-05-29T17:25:12.691332+00:00",
                    "n_round_trips": 1},
            "MU": {"current_loss_streak": 1, "current_loss_usd": -22.765,
                   "last_loss_exit_ts": "2026-05-28T14:02:43.368432+00:00",
                   "n_round_trips": 3},
        },
        "prompt_block": None,
        "state": "OK",
        "threshold": 2,
        "verdict": None,
    }
    assert _repeat_loser_chat_lines(rep) == []


def test_repeat_loser_fixture_full_render():
    """The chronic-loser scenario this helper exists to surface."""
    rep = {
        "state": "REPEAT_LOSER",
        "verdict": "REPEAT_LOSER",
        "headline": (
            "REPEAT_LOSER — MU on a 3-trip loss run (net $-45.50). "
            "Threshold 2."
        ),
        "n_offenders": 1,
        "n_round_trips": 9,
        "threshold": 2,
        "offenders": [{
            "ticker": "MU",
            "current_loss_streak": 3,
            "current_loss_usd": -45.50,
            "last_loss_exit_ts": "2026-05-28T14:02:43.368432+00:00",
            "n_round_trips": 3,
        }],
        "per_ticker": {},
    }
    lines = _repeat_loser_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "MU" in detail
    assert "3L in a row" in detail
    assert "threshold 2" in detail
