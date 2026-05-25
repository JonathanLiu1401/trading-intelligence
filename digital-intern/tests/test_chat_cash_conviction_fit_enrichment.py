"""Pure-helper tests for the /api/chat cash-conviction-fit enrichment.

``_cash_conviction_fit_chat_lines`` renders paper-trader's
``/api/cash-conviction-fit`` (the cash-level-vs-loudest-live-signal
calibration check) into compact chat lines so the analyst can flag
"the book is structurally wrong for the current signal regime" — a
question none of the other cash surfaces (cash_pct, all_cash_streak,
cash_redeployment, cash_drag) answer.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict or threshold restatement.
- **balanced = silence**: BALANCED / NO_DATA collapse to ``[]`` —
  a correctly-calibrated book must never become chat filler; NO_DATA
  is a probe-side defect.
- **detail line fields**: when actionable, the detail line restates
  the builder's own ``portfolio.cash_pct`` / ``portfolio.cash_usd`` /
  ``top_signal.ticker`` / ``top_signal.ai_score`` /
  ``last_decision.verb`` / ``last_decision.age_min`` verbatim — never
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

from dashboard.web_server import _cash_conviction_fit_chat_lines


def _rep(
    verdict="IDLE_DESPITE_SURGE",
    *,
    headline=None,
    cash_pct=95.0,
    cash_usd=950.0,
    n_positions=0,
    total_value_usd=1000.0,
    top_ticker="NVDA",
    top_score=9.2,
    top_urgency=1,
    top_held=False,
    last_verb="HOLD",
    last_age_min=30.0,
    last_fill=False,
):
    if headline is None:
        headline = (
            f"{verdict} — {cash_pct:.0f}% cash, top signal {top_score:.1f} {top_ticker}"
        )
    return {
        "verdict": verdict,
        "headline": headline,
        "portfolio": {
            "cash_pct": cash_pct,
            "cash_usd": cash_usd,
            "n_positions": n_positions,
            "total_value_usd": total_value_usd,
        },
        "top_signal": {
            "ticker": top_ticker,
            "ai_score": top_score,
            "urgency": top_urgency,
            "source": "AlphaVantage",
            "held": top_held,
        },
        "last_decision": {
            "verb": last_verb,
            "age_min": last_age_min,
            "recent_fill": last_fill,
        },
        "thresholds": {
            "high_conviction_score": 8.0,
            "idle_cash_pct": 40.0,
            "low_conviction_score": 6.0,
            "overdeployed_cash_pct": 10.0,
            "recent_fill_max_min": 30.0,
        },
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _cash_conviction_fit_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _cash_conviction_fit_chat_lines({}) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize(
    "v", ["BALANCED", "NO_DATA", "", None, "OK", "WHATEVER"],
)
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _cash_conviction_fit_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize(
    "v", ["IDLE_DESPITE_SURGE", "OVERDEPLOYED", "IDLE_LOW_CONVICTION"],
)
def test_actionable_verdicts_emit_lines(v):
    lines = _cash_conviction_fit_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_idle_despite_surge_headline_passes_through_verbatim():
    rep = _rep(
        verdict="IDLE_DESPITE_SURGE",
        headline=(
            "IDLE_DESPITE_SURGE — 95% cash idle while NVDA screams ai_score "
            "9.2; last decision HOLD 30m ago."
        ),
    )
    assert _cash_conviction_fit_chat_lines(rep)[0] == rep["headline"]


def test_overdeployed_headline_passes_through_verbatim():
    rep = _rep(
        verdict="OVERDEPLOYED",
        headline=(
            "OVERDEPLOYED — only 5% cash with NVDA at ai_score 9.2; the "
            "book cannot add without trimming."
        ),
    )
    assert _cash_conviction_fit_chat_lines(rep)[0] == rep["headline"]


def test_idle_low_conviction_headline_passes_through_verbatim():
    rep = _rep(
        verdict="IDLE_LOW_CONVICTION",
        headline=(
            "IDLE_LOW_CONVICTION — 85% cash idle; loudest live signal only "
            "5.4. Cash idleness is correct — nothing is screaming."
        ),
    )
    assert _cash_conviction_fit_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _cash_conviction_fit_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "cash 95%" in lines[0]


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_idle_despite_surge():
    rep = _rep(
        verdict="IDLE_DESPITE_SURGE",
        cash_pct=95.0, cash_usd=950.0,
        top_ticker="NVDA", top_score=9.2,
        last_verb="HOLD", last_age_min=30.0,
    )
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "cash 95%" in detail
    assert "$950" in detail
    assert "top=NVDA" in detail
    assert "ai=9.2" in detail
    assert "last=HOLD" in detail
    assert "30m ago" in detail


def test_detail_line_omits_missing_portfolio():
    rep = _rep()
    rep["portfolio"] = None
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "cash" not in detail
    assert "top=NVDA" in detail


def test_detail_line_omits_missing_signal():
    rep = _rep()
    rep["top_signal"] = None
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "top=" not in detail
    assert "cash 95%" in detail


def test_detail_line_omits_missing_last_decision():
    rep = _rep()
    rep["last_decision"] = None
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "last=" not in detail
    assert "cash 95%" in detail


def test_detail_line_top_signal_with_only_ticker():
    rep = _rep()
    rep["top_signal"] = {"ticker": "NVDA"}
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "top=NVDA" in detail
    # No bare 'ai=' chunk when score missing
    assert "ai=" not in detail


def test_detail_line_top_signal_with_only_score():
    rep = _rep()
    rep["top_signal"] = {"ai_score": 9.2}
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "top_ai=9.2" in detail
    # We did not invent a ticker
    assert "top=NVDA" not in detail


def test_detail_line_last_verb_without_age():
    rep = _rep()
    rep["last_decision"] = {"verb": "BUY"}
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "last=BUY" in detail
    # No fabricated 'm ago'
    assert "m ago" not in detail


def test_detail_line_all_fields_missing_suppresses_detail():
    rep = {
        "verdict": "IDLE_DESPITE_SURGE",
        "headline": "IDLE_DESPITE_SURGE — sample text",
    }
    lines = _cash_conviction_fit_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0] == "IDLE_DESPITE_SURGE — sample text"


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_cash_pct_treated_as_missing():
    """bool is_a int in Python; never let True/False slip through."""
    rep = _rep()
    rep["portfolio"]["cash_pct"] = True
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "cash 1%" not in detail
    assert "cash True" not in detail


def test_bool_ai_score_treated_as_missing():
    rep = _rep()
    rep["top_signal"]["ai_score"] = False
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "ai=0" not in detail
    assert "ai=False" not in detail


def test_bool_age_min_treated_as_missing():
    rep = _rep()
    rep["last_decision"]["age_min"] = True
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "m ago" not in detail
    assert "last=HOLD" in detail


def test_string_score_treated_as_missing():
    rep = _rep()
    rep["top_signal"]["ai_score"] = "9.2"
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    assert "ai=9.2" not in detail
    # Fallback to ticker-only branch
    assert "top=NVDA" in detail


def test_empty_ticker_string_dropped():
    rep = _rep()
    rep["top_signal"]["ticker"] = "  "
    rep["top_signal"]["ai_score"] = 9.2
    detail = _cash_conviction_fit_chat_lines(rep)[1]
    # Fall through to top_ai-only branch since ticker is whitespace
    assert "top=" not in detail
    assert "top_ai=9.2" in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_25_balanced_is_silent():
    """The current live BALANCED response — confirm we don't push
    chat filler when the cash level fits the conviction."""
    rep = {
        "verdict": "BALANCED",
        "headline": "BALANCED — cash 100% vs top signal 6.0; level fits conviction.",
        "portfolio": {
            "cash_pct": 100.0, "cash_usd": 987.39,
            "n_positions": 0, "total_value_usd": 987.39,
        },
        "top_signal": {
            "ticker": "ORCL", "ai_score": 6.0, "urgency": 0,
            "source": "AlphaVantage/Foreign Policy Journal", "held": False,
        },
        "last_decision": {
            "verb": "HOLD", "age_min": 55.3, "recent_fill": False,
        },
    }
    assert _cash_conviction_fit_chat_lines(rep) == []


def test_idle_despite_surge_fixture_full_render():
    """The structural-wrong-book scenario this helper exists to surface."""
    rep = {
        "verdict": "IDLE_DESPITE_SURGE",
        "headline": (
            "IDLE_DESPITE_SURGE — 95% cash idle while NVDA screams ai_score "
            "9.2; last decision HOLD 30m ago."
        ),
        "portfolio": {
            "cash_pct": 95.0, "cash_usd": 950.0,
            "n_positions": 0, "total_value_usd": 1000.0,
        },
        "top_signal": {
            "ticker": "NVDA", "ai_score": 9.2, "urgency": 1,
            "source": "ticker_news", "held": False,
        },
        "last_decision": {
            "verb": "HOLD", "age_min": 30.0, "recent_fill": False,
        },
    }
    lines = _cash_conviction_fit_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "cash 95%" in detail
    assert "$950" in detail
    assert "top=NVDA ai=9.2" in detail
    assert "last=HOLD 30m ago" in detail
