"""Pure-helper tests for the /api/chat exit-only-streak enrichment.

``_exit_only_streak_chat_lines`` renders paper-trader's
``/api/exit-only-streak`` (the consecutive-SELLs-since-last-entry
detector at the book level) into compact chat lines so the analyst can
flag "the last 6 fills were all SELLs — the engine is liquidating, not
running the strategy" — a trade-direction sequence invisible to
``/api/streak`` (W/L on round-trips), ``/api/churn`` (cadence), and
``/api/cash-drag`` (idle-cash dollar cost).

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` passes through UNCHANGED — no chat-side
  re-derived verdict naming.
- **healthy = silence**: MOST_RECENT_IS_ENTRY / state != STABLE
  collapse to ``[]`` — a book whose newest fill is an entry must never
  become chat filler.
- **detail line fields**: when actionable, the detail line restates the
  builder's own ``exit_run_length`` / ``exit_run_tickers`` /
  ``hours_since_last_entry`` / ``most_recent_action`` verbatim — never
  a recomputation.
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

from dashboard.web_server import _exit_only_streak_chat_lines


def _rep(
    state="STABLE",
    verdict="DEFENSIVE_TRIM",
    *,
    headline=None,
    run_len=3,
    tickers=("NVDA", "MU", "AMD"),
    hours_since_entry=18.4,
    most_recent="SELL",
):
    if headline is None:
        headline = (
            f"{verdict} — {run_len} consec exits since last entry "
            f"({18.4:.1f}h ago)."
        )
    return {
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "exit_run_length": run_len,
        "exit_run_tickers": list(tickers),
        "hours_since_last_entry": hours_since_entry,
        "most_recent_action": most_recent,
        "n_entries": 15,
        "n_exits": 9,
        "n_total_fills": 24,
        "defensive_trim_min": 3,
        "defensive_liquidation_min": 6,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _exit_only_streak_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _exit_only_streak_chat_lines({}) == []


# ── silence on non-actionable verdicts/state ────────────────────────────
@pytest.mark.parametrize("s", ["NO_DATA", "", None, "EMERGING", "stable"])
def test_non_stable_state_collapses_to_silence(s):
    assert _exit_only_streak_chat_lines(_rep(state=s)) == []


@pytest.mark.parametrize(
    "v", ["MOST_RECENT_IS_ENTRY", "", None, "STABLE", "defensive_trim"],
)
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _exit_only_streak_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize(
    "v", ["DEFENSIVE_TRIM", "DEFENSIVE_LIQUIDATION"],
)
def test_actionable_verdicts_emit_lines(v):
    lines = _exit_only_streak_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_defensive_trim_headline_passes_through_verbatim():
    rep = _rep(
        verdict="DEFENSIVE_TRIM",
        headline=(
            "DEFENSIVE_TRIM — 3 consecutive exits since last entry "
            "(18.4h ago); newest exit MU."
        ),
    )
    assert _exit_only_streak_chat_lines(rep)[0] == rep["headline"]


def test_defensive_liquidation_headline_passes_through_verbatim():
    rep = _rep(
        verdict="DEFENSIVE_LIQUIDATION",
        run_len=7,
        headline=(
            "DEFENSIVE_LIQUIDATION — 7 consecutive exits since last entry; "
            "the engine is liquidating, not running the strategy."
        ),
    )
    assert _exit_only_streak_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _exit_only_streak_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "run=3 consec exits" in lines[0]


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_defensive_trim():
    rep = _rep(
        verdict="DEFENSIVE_TRIM", run_len=3,
        tickers=("NVDA", "MU", "AMD"),
        hours_since_entry=18.4,
        most_recent="SELL",
    )
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "run=3 consec exits" in detail
    assert "NVDA→MU→AMD" in detail
    assert "18.4h since last entry" in detail
    assert "most recent=SELL" in detail


def test_detail_line_caps_tickers_at_5():
    rep = _rep(tickers=("A", "B", "C", "D", "E", "F", "G"))
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "A→B→C→D→E" in detail
    assert "F" not in detail
    assert "G" not in detail


def test_detail_line_omits_missing_run_len():
    rep = _rep()
    rep["exit_run_length"] = None
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "run=" not in detail
    assert "NVDA→MU→AMD" in detail


def test_detail_line_omits_empty_tickers():
    rep = _rep(tickers=())
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "→" not in detail
    assert "run=3 consec exits" in detail


def test_detail_line_skips_non_string_tickers():
    rep = _rep()
    rep["exit_run_tickers"] = [None, 42, "MU", "", "  ", "AMD"]
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "MU→AMD" in detail


def test_detail_line_omits_missing_hours():
    rep = _rep()
    rep["hours_since_last_entry"] = None
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "since last entry" not in detail
    assert "run=3 consec exits" in detail


def test_detail_line_omits_missing_most_recent():
    rep = _rep()
    rep["most_recent_action"] = None
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "most recent=" not in detail
    assert "run=3 consec exits" in detail


def test_detail_line_all_fields_missing_suppresses_detail():
    rep = {
        "state": "STABLE",
        "verdict": "DEFENSIVE_TRIM",
        "headline": "DEFENSIVE_TRIM — sample text",
    }
    lines = _exit_only_streak_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0] == "DEFENSIVE_TRIM — sample text"


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_run_len_treated_as_missing():
    rep = _rep()
    rep["exit_run_length"] = True
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "run=1 consec exits" not in detail
    assert "run=True" not in detail


def test_bool_hours_treated_as_missing():
    rep = _rep()
    rep["hours_since_last_entry"] = False
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "since last entry" not in detail


def test_string_hours_treated_as_missing():
    rep = _rep()
    rep["hours_since_last_entry"] = "18.4"
    detail = _exit_only_streak_chat_lines(rep)[1]
    assert "since last entry" not in detail
    assert "run=3 consec exits" in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_31_most_recent_is_entry_is_silent():
    """The current live MOST_RECENT_IS_ENTRY response — confirm we don't push
    chat filler when the bot is in mixed-mode."""
    rep = {
        "as_of": "2026-05-31T05:25:15+00:00",
        "defensive_liquidation_min": 6,
        "defensive_trim_min": 3,
        "exit_run_length": 0,
        "exit_run_started_ts": None,
        "exit_run_tickers": [],
        "headline": "NEUTRAL — newest fill is an entry (BUY MU). 15 entries / 9 exits on record.",
        "hours_since_last_entry": 35.33,
        "last_entry_action": "BUY",
        "last_entry_ticker": "MU",
        "last_entry_ts": "2026-05-29T18:05:38.834995+00:00",
        "most_recent_action": "BUY",
        "most_recent_ts": "2026-05-29T18:05:38.834995+00:00",
        "n_entries": 15,
        "n_exits": 9,
        "n_total_fills": 24,
        "recent_sequence": ["X", "E", "X", "E", "X", "E", "X", "E", "E", "X", "X", "E"],
        "state": "STABLE",
        "verdict": "MOST_RECENT_IS_ENTRY",
    }
    assert _exit_only_streak_chat_lines(rep) == []


def test_defensive_trim_fixture_full_render():
    """The defensive-trim scenario this helper exists to surface."""
    rep = {
        "state": "STABLE",
        "verdict": "DEFENSIVE_TRIM",
        "headline": (
            "DEFENSIVE_TRIM — 4 consecutive exits since last entry "
            "(48.0h ago); newest exit AMD."
        ),
        "exit_run_length": 4,
        "exit_run_tickers": ["NVDA", "MU", "TQQQ", "AMD"],
        "hours_since_last_entry": 48.0,
        "most_recent_action": "SELL",
        "n_entries": 5,
        "n_exits": 8,
        "n_total_fills": 13,
        "defensive_trim_min": 3,
        "defensive_liquidation_min": 6,
    }
    lines = _exit_only_streak_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "run=4 consec exits" in detail
    assert "NVDA→MU→TQQQ→AMD" in detail
    assert "48.0h since last entry" in detail
    assert "most recent=SELL" in detail
