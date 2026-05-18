"""Pure-helper tests for the /api/chat behavioural-diagnosis enrichment.

`_behavioural_chat_lines` composes the paper-trader's own self-review
verdicts (`/api/scorecard`, `/api/capital-paralysis`, `/api/churn`) into
compact chat-context lines. The surrounding chat handler is one large
inline closure, so per the design the new logic is a total/pure function
unit-tested here: a dropped verbatim headline, a wrong priority-precedence
branch, or a None/error-handling regression fails here without standing up
Flask or cross-fetching :8090.

The discriminating lock is **verbatim composition** (paper-trader
invariant #10 — single source of truth): each builder's own headline /
focus / flag / unlock-reason string must appear UNCHANGED in the output.
An inline re-derivation that drifts from the trader endpoint fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _behavioural_chat_lines


def _scorecard(focus: bool = True) -> dict:
    d = {
        "state": "FLAGS_PRESENT",
        "headline": "1 of 5 behavioural checks flagging: DEGRADED.",
    }
    if focus:
        d["focus"] = {
            "name": "decision_reliability",
            "label": "DEGRADED",
            "theme": "DECISION_INTEGRITY",
            "headline": "DEGRADED — current-regime parse-fail 25.5% over 94 cycle(s)",
        }
    return d


def _paralysis(unlock: bool = False) -> dict:
    d = {
        "state": "FREE" if not unlock else "PINNED",
        "headline": (
            "FREE — $18.49 cash (1.9%) available; the book can act on a "
            "new signal without selling."
        ),
        "flags": [
            "98.1% of book deployed",
            "LITE is 61% of the book",
            "inaction has cost -2.21% alpha (6 paralysis drought(s))",
        ],
        "recommended_unlock": None,
    }
    if unlock:
        d["recommended_unlock"] = {
            "ticker": "LITE",
            "frees_usd": 592.13,
            "pl_pct": -1.0,
            "reason": (
                "largest underwater name (-1.0%) — selling it frees "
                "$592.13 and restores the ability to act on a fresh signal"
            ),
        }
    return d


def _churn(state: str = "EMERGING") -> dict:
    return {
        "state": state,
        "headline": (
            "Emerging — 6 of 20 round-trips for a stable read. So far: 1 "
            "fast same-name re-entries (16.7%), 1.89 round-trips/day, "
            "0.27d median hold (verdict withheld until n≥20)."
        ),
    }


class TestBehaviouralChatLinesLiveShape:
    def test_exact_render_of_the_live_book_shape(self):
        lines = _behavioural_chat_lines(_scorecard(), _paralysis(), _churn())
        assert lines == [
            "Scorecard: 1 of 5 behavioural checks flagging: DEGRADED.",
            "  focus: DEGRADED — current-regime parse-fail 25.5% over 94 cycle(s)",
            "Capital: FREE — $18.49 cash (1.9%) available; the book can act "
            "on a new signal without selling.",
            "  • 98.1% of book deployed",
            "  • LITE is 61% of the book",
            "  • inaction has cost -2.21% alpha (6 paralysis drought(s))",
            "Churn: Emerging — 6 of 20 round-trips for a stable read. So "
            "far: 1 fast same-name re-entries (16.7%), 1.89 round-trips/day, "
            "0.27d median hold (verdict withheld until n≥20).",
            "▶ PRIORITY: DECISION_INTEGRITY — DEGRADED — current-regime "
            "parse-fail 25.5% over 94 cycle(s)",
        ]

    def test_verbatim_composition_not_rederived(self):
        sc, cp, ch = _scorecard(), _paralysis(), _churn()
        joined = "\n".join(_behavioural_chat_lines(sc, cp, ch))
        # Every builder's own string must pass through UNCHANGED.
        assert sc["headline"] in joined
        assert sc["focus"]["headline"] in joined
        assert cp["headline"] in joined
        for fl in cp["flags"]:
            assert fl in joined
        assert ch["headline"] in joined

    def test_flags_capped_at_three(self):
        cp = _paralysis()
        cp["flags"] = [f"flag-{i}" for i in range(7)]
        lines = _behavioural_chat_lines(None, cp, None)
        bullets = [l for l in lines if l.startswith("  • ")]
        assert bullets == ["  • flag-0", "  • flag-1", "  • flag-2"]


class TestPriorityPrecedence:
    def test_unlock_beats_focus(self):
        lines = _behavioural_chat_lines(
            _scorecard(focus=True), _paralysis(unlock=True), _churn("CHURNING")
        )
        prio = [l for l in lines if l.startswith("▶ PRIORITY:")]
        assert prio == [
            "▶ PRIORITY: sell LITE — largest underwater name (-1.0%) — "
            "selling it frees $592.13 and restores the ability to act on a "
            "fresh signal"
        ]

    def test_focus_beats_churn_when_no_unlock(self):
        lines = _behavioural_chat_lines(
            _scorecard(focus=True), _paralysis(unlock=False), _churn("CHURNING")
        )
        prio = [l for l in lines if l.startswith("▶ PRIORITY:")]
        assert prio == [
            "▶ PRIORITY: DECISION_INTEGRITY — DEGRADED — current-regime "
            "parse-fail 25.5% over 94 cycle(s)"
        ]

    def test_churn_churning_is_last_resort(self):
        lines = _behavioural_chat_lines(
            _scorecard(focus=False), _paralysis(unlock=False), _churn("CHURNING")
        )
        prio = [l for l in lines if l.startswith("▶ PRIORITY:")]
        assert prio == [
            "▶ PRIORITY: overtrading — Emerging — 6 of 20 round-trips for a "
            "stable read. So far: 1 fast same-name re-entries (16.7%), 1.89 "
            "round-trips/day, 0.27d median hold (verdict withheld until n≥20)."
        ]

    def test_no_priority_line_when_nothing_actionable(self):
        lines = _behavioural_chat_lines(
            _scorecard(focus=False), _paralysis(unlock=False), _churn("EMERGING")
        )
        assert not any(l.startswith("▶ PRIORITY:") for l in lines)
        # but the descriptive lines still render
        assert any(l.startswith("Scorecard:") for l in lines)


class TestIndependentDegradation:
    def test_each_input_none_drops_only_its_lines(self):
        only_churn = _behavioural_chat_lines(None, None, _churn())
        assert only_churn == [
            "Churn: Emerging — 6 of 20 round-trips for a stable read. So "
            "far: 1 fast same-name re-entries (16.7%), 1.89 round-trips/day, "
            "0.27d median hold (verdict withheld until n≥20)."
        ]

    def test_error_payload_treated_as_absent(self):
        lines = _behavioural_chat_lines(
            {"error": "boom"}, _paralysis(), {"error": "x"}
        )
        assert not any(l.startswith("Scorecard:") for l in lines)
        assert not any(l.startswith("Churn:") for l in lines)
        assert any(l.startswith("Capital:") for l in lines)

    def test_missing_state_treated_as_absent(self):
        # a dict with no 'state' is an unusable upstream payload
        lines = _behavioural_chat_lines({"headline": "x"}, _paralysis(), None)
        assert not any(l.startswith("Scorecard:") for l in lines)
        assert any(l.startswith("Capital:") for l in lines)

    def test_no_data_state_suppressed(self):
        lines = _behavioural_chat_lines(
            {"state": "NO_DATA", "headline": "no data"},
            {"state": "NO_DATA", "headline": "no data"},
            {"state": "NO_DATA", "headline": "no data"},
        )
        assert lines == []

    def test_all_absent_returns_empty(self):
        assert _behavioural_chat_lines(None, None, None) == []
        assert _behavioural_chat_lines("x", 7, ["a"]) == []
        assert _behavioural_chat_lines({}, {}, {}) == []
