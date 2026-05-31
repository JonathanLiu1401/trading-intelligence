"""Pure-helper tests for the /api/chat catalyst-class-autopsy enrichment.

``_catalyst_class_autopsy_chat_lines`` renders paper-trader's
``/api/catalyst-class-autopsy`` (the per-catalyst-class win-rate / PnL
leaderboard over closed round-trips) into compact chat lines so the
analyst can answer the structural class-allocation question every other
realised-P&L block aggregates away: "of the 9 catalyst classes
(ML_ADVISOR / ANALYST_PT / TECHNICALS / EARNINGS_PLAY / MACRO /
BREAKING_NEWS / PUNDIT / SECTOR_SYMPATHY / CONCENTRATION), which biases
my realised P&L UP (lean INTO) or DOWN (lean OUT OF)?".

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` passes through UNCHANGED — no chat-side
  re-derived verdict or threshold restatement.
- **healthy = silence**: state != STABLE collapses to ``[]`` (NO_DATA /
  EMERGING — no class has crossed the sample-size gate). A STABLE-but-
  NEUTRAL panel where no class has reached BIASED also collapses to
  ``[]`` — the leaderboard is interesting but not actionable.
- **detail line fields**: when actionable, the detail line restates
  ``top_biased_winner`` / ``top_biased_loser`` /
  ``biased_wr_delta_pct`` / ``pool_win_rate_pct`` / ``n_round_trips``
  verbatim — never a recomputation.
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

from dashboard.web_server import _catalyst_class_autopsy_chat_lines


def _rep(
    state="STABLE",
    *,
    headline=None,
    top_winner="ML_ADVISOR",
    top_loser=None,
    delta=15.0,
    pool_wr=55.56,
    n_trips=9,
):
    if headline is None:
        if top_winner and top_loser:
            headline = (
                f"{top_winner} biases WIN, {top_loser} biases LOSS "
                f"(≥{delta:.0f}pp from pool {pool_wr:.1f}%)."
            )
        elif top_winner:
            headline = (
                f"{top_winner} is the earning catalyst class — "
                f"win-rate ≥{delta:.0f}% above pool."
            )
        elif top_loser:
            headline = (
                f"{top_loser} is the bleeding catalyst class — "
                f"win-rate ≥{delta:.0f}% below pool."
            )
        else:
            headline = (
                f"STABLE — no class biased (within ±{delta:.0f}pp of "
                f"pool {pool_wr:.1f}%)."
            )
    return {
        "state": state,
        "headline": headline,
        "top_biased_winner": top_winner,
        "top_biased_loser": top_loser,
        "best_class": top_winner or "ML_ADVISOR",
        "worst_class": top_loser or "EARNINGS_PLAY",
        "biased_wr_delta_pct": delta,
        "pool_win_rate_pct": pool_wr,
        "n_round_trips": n_trips,
        "n_scored": n_trips,
        "stable_min_trips_per_class": 4,
        "taxonomy": ["ML_ADVISOR", "EARNINGS_PLAY"],
        "classes": [],
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _catalyst_class_autopsy_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _catalyst_class_autopsy_chat_lines({}) == []


# ── silence on non-actionable states ────────────────────────────────────
@pytest.mark.parametrize(
    "s", ["NO_DATA", "EMERGING", "", None, "stable", "OK"],  # case-sensitive
)
def test_non_stable_states_collapse_to_silence(s):
    assert _catalyst_class_autopsy_chat_lines(
        _rep(state=s, top_winner="ML_ADVISOR")
    ) == []


def test_stable_but_no_bias_is_silence():
    """A STABLE-but-NEUTRAL leaderboard (no biased class) collapses."""
    assert _catalyst_class_autopsy_chat_lines(
        _rep(top_winner=None, top_loser=None)
    ) == []


def test_stable_with_winner_emits_lines():
    lines = _catalyst_class_autopsy_chat_lines(
        _rep(top_winner="ML_ADVISOR", top_loser=None)
    )
    assert lines, "STABLE with biased winner must produce ≥1 line"


def test_stable_with_loser_emits_lines():
    lines = _catalyst_class_autopsy_chat_lines(
        _rep(top_winner=None, top_loser="EARNINGS_PLAY")
    )
    assert lines, "STABLE with biased loser must produce ≥1 line"


def test_stable_with_both_emits_lines():
    lines = _catalyst_class_autopsy_chat_lines(
        _rep(top_winner="ML_ADVISOR", top_loser="EARNINGS_PLAY")
    )
    assert lines, "STABLE with biased winner + loser must produce ≥1 line"


@pytest.mark.parametrize("v", ["", "  ", None])
def test_whitespace_or_none_winner_treated_as_absent(v):
    """Whitespace-only or None winner string must not count as biased."""
    rep = _rep(top_winner=v, top_loser=None)
    assert _catalyst_class_autopsy_chat_lines(rep) == []


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_biased_winner_headline_passes_through_verbatim():
    rep = _rep(
        top_winner="ML_ADVISOR",
        headline=(
            "ML_ADVISOR is the earning catalyst class — win-rate "
            "≥15% above pool."
        ),
    )
    assert _catalyst_class_autopsy_chat_lines(rep)[0] == rep["headline"]


def test_biased_loser_headline_passes_through_verbatim():
    rep = _rep(
        top_winner=None,
        top_loser="EARNINGS_PLAY",
        headline=(
            "EARNINGS_PLAY is the bleeding catalyst class — win-rate "
            "≥15% below pool."
        ),
    )
    assert _catalyst_class_autopsy_chat_lines(rep)[0] == rep["headline"]


def test_both_biased_headline_passes_through_verbatim():
    rep = _rep(
        top_winner="ML_ADVISOR",
        top_loser="EARNINGS_PLAY",
        headline=(
            "ML_ADVISOR biases WIN, EARNINGS_PLAY biases LOSS "
            "(≥15pp from pool 55.6%)."
        ),
    )
    assert _catalyst_class_autopsy_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _catalyst_class_autopsy_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "winner=ML_ADVISOR" in lines[0]


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_biased_winner():
    rep = _rep(
        top_winner="ML_ADVISOR", top_loser=None,
        delta=15.0, pool_wr=55.56, n_trips=9,
    )
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "winner=ML_ADVISOR" in detail
    assert "loser=" not in detail
    assert "Δwr≥15pp" in detail
    assert "pool wr=55.6%" in detail
    assert "n=9 trips" in detail


def test_detail_line_full_biased_loser_only():
    rep = _rep(
        top_winner=None, top_loser="EARNINGS_PLAY",
        delta=15.0, pool_wr=50.0, n_trips=12,
    )
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "winner=" not in detail
    assert "loser=EARNINGS_PLAY" in detail
    assert "Δwr≥15pp" in detail


def test_detail_line_both_winner_and_loser():
    rep = _rep(top_winner="ML_ADVISOR", top_loser="EARNINGS_PLAY")
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "winner=ML_ADVISOR" in detail
    assert "loser=EARNINGS_PLAY" in detail


def test_detail_line_omits_missing_delta():
    rep = _rep()
    rep["biased_wr_delta_pct"] = None
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "Δwr" not in detail
    assert "winner=ML_ADVISOR" in detail


def test_detail_line_omits_missing_pool_wr():
    rep = _rep()
    rep["pool_win_rate_pct"] = None
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "pool wr" not in detail


def test_detail_line_omits_missing_n_trips():
    rep = _rep()
    rep["n_round_trips"] = None
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "n=" not in detail


def test_detail_line_all_optional_missing_keeps_winner_only():
    rep = {
        "state": "STABLE",
        "top_biased_winner": "ML_ADVISOR",
        "headline": "ML_ADVISOR is the earning catalyst class.",
    }
    lines = _catalyst_class_autopsy_chat_lines(rep)
    assert len(lines) == 2
    assert lines[1] == "  winner=ML_ADVISOR"


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_delta_treated_as_missing():
    rep = _rep()
    rep["biased_wr_delta_pct"] = True
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "Δwr≥1pp" not in detail
    assert "True" not in detail


def test_bool_pool_wr_treated_as_missing():
    rep = _rep()
    rep["pool_win_rate_pct"] = False
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "pool wr=0" not in detail
    assert "False" not in detail


def test_bool_n_trips_treated_as_missing():
    rep = _rep()
    rep["n_round_trips"] = True
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "n=1 trips" not in detail
    assert "n=True" not in detail


def test_string_delta_treated_as_missing():
    rep = _rep()
    rep["biased_wr_delta_pct"] = "15"
    detail = _catalyst_class_autopsy_chat_lines(rep)[1]
    assert "Δwr" not in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_31_biased_winner_emits():
    """The current live STABLE+ML_ADVISOR-winner response — confirm it surfaces."""
    rep = {
        "as_of": "2026-05-31T05:24:21+00:00",
        "best_class": "ML_ADVISOR",
        "biased_wr_delta_pct": 15.0,
        "classes": [],
        "headline": (
            "ML_ADVISOR is the earning catalyst class — win-rate "
            "≥15% above pool."
        ),
        "n_round_trips": 9,
        "n_scored": 9,
        "pool_win_rate_pct": 55.56,
        "stable_min_trips_per_class": 4,
        "state": "STABLE",
        "top_biased_loser": None,
        "top_biased_winner": "ML_ADVISOR",
        "worst_class": "EARNINGS_PLAY",
        "taxonomy": [
            "ML_ADVISOR", "EARNINGS_PLAY", "ANALYST_PT", "TECHNICALS",
            "MACRO", "BREAKING_NEWS", "PUNDIT", "SECTOR_SYMPATHY",
            "CONCENTRATION", "UNCLASSIFIED",
        ],
    }
    lines = _catalyst_class_autopsy_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "winner=ML_ADVISOR" in detail
    assert "loser=" not in detail
    assert "Δwr≥15pp" in detail
    assert "pool wr=55.6%" in detail
    assert "n=9 trips" in detail


def test_emerging_state_collapses_even_with_winner():
    """A class can be tagged biased before state crosses to STABLE; chat
    must still wait for the state gate."""
    rep = _rep(state="EMERGING", top_winner="ML_ADVISOR")
    assert _catalyst_class_autopsy_chat_lines(rep) == []
