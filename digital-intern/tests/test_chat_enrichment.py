"""Pure-helper tests for the /api/chat context enrichment (web_server.py).

`_tail_risk_chat_lines` and `_partition_thesis_articles` are the
unit-testable cores of the chat enrichment — the surrounding handler is
one large inline closure, so per the design the new logic was extracted
into these total/pure functions. A wrong gate branch, a dropped tie in
the dedup, an off-by-one cap, or a None-handling regression fails here
without needing to stand up Flask or the article DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import (
    _partition_thesis_articles,
    _tail_risk_chat_lines,
)


class TestTailRiskChatLines:
    def _ok_payload(self) -> dict:
        return {
            "tail_risk": {
                "state": "OK",
                "n_returns": 24,
                "var_95_pct": 10.0,
                "var_99_pct": 20.0,
                "cvar_95_pct": 15.0,
                "annualized_vol_pct": 117.9,
                "downside_deviation_pct": 79.37,
                "return_skew": 0.758,
                "worst_day_pct": -20.0,
                "max_consecutive_down_days": 2,
                "ulcer_index_pct": 18.89,
            }
        }

    def test_ok_renders_two_dense_lines_with_numbers(self):
        lines = _tail_risk_chat_lines(self._ok_payload())
        assert len(lines) == 2
        joined = "\n".join(lines)
        assert "95% 1-day VaR 10.00%" in joined
        assert "CVaR 15.00%" in joined
        assert "99% VaR 20.00%" in joined
        assert "ann.vol 117.90%" in joined
        assert "skew +0.76" in joined          # signed, 2dp
        assert "worst day -20.00%" in joined
        assert "max down-streak 2d" in joined
        assert "Ulcer 18.89%" in joined

    def test_insufficient_collapses_to_one_honest_line(self):
        lines = _tail_risk_chat_lines(
            {"tail_risk": {"state": "INSUFFICIENT", "n_returns": 5,
                           "min_returns": 20}}
        )
        assert len(lines) == 1
        assert "still building history" in lines[0]
        assert "5/20" in lines[0]
        # Must NOT leak a verdict-shaped VaR number.
        assert "VaR" not in lines[0]

    def test_no_data_is_empty(self):
        assert _tail_risk_chat_lines({"tail_risk": {"state": "NO_DATA"}}) == []

    def test_missing_key_is_empty(self):
        assert _tail_risk_chat_lines({"sharpe_annualized": 1.2}) == []

    def test_upstream_error_payload_is_empty(self):
        # /api/analytics returned {"error": ...} → no tail_risk key.
        assert _tail_risk_chat_lines({"error": "boom"}) == []

    def test_non_dict_inputs_never_raise(self):
        for bad in (None, [], "x", 3, {"tail_risk": "not-a-dict"}):
            assert _tail_risk_chat_lines(bad) == []  # type: ignore[arg-type]

    def test_ok_with_none_metrics_renders_na_not_crash(self):
        # Flat-book OK state: skew None, vol 0.0 — must degrade, not raise.
        lines = _tail_risk_chat_lines(
            {"tail_risk": {"state": "OK", "n_returns": 24,
                           "var_95_pct": 0.0, "var_99_pct": 0.0,
                           "cvar_95_pct": 0.0, "annualized_vol_pct": 0.0,
                           "downside_deviation_pct": 0.0,
                           "return_skew": None, "worst_day_pct": 0.0,
                           "max_consecutive_down_days": 0,
                           "ulcer_index_pct": 0.0}}
        )
        assert len(lines) == 2
        assert "skew n/a" in "\n".join(lines)


def _art(title: str, score: float = 5.0) -> dict:
    return {"title": title, "source": "rss", "ai_score": score, "summary": ""}


class TestPartitionThesisArticles:
    def test_dedups_case_insensitively_against_breaking(self):
        breaking = [_art("NVDA Beats Earnings"), _art("MU Guides Up")]
        thesis = [
            _art("  nvda beats earnings  "),   # dup (case/space) → dropped
            _art("Samsung HBM4 Ramp"),         # new → kept
            _art("MU GUIDES UP"),              # dup → dropped
        ]
        out = _partition_thesis_articles(breaking, thesis, max_thesis=8)
        assert [a["title"] for a in out] == ["Samsung HBM4 Ramp"]

    def test_internal_dedup_keeps_first_occurrence(self):
        thesis = [_art("AI capex cycle"), _art("ai CAPEX cycle"),
                  _art("Foundry pricing")]
        out = _partition_thesis_articles([], thesis, max_thesis=8)
        assert [a["title"] for a in out] == ["AI capex cycle", "Foundry pricing"]

    def test_cap_is_honored_and_order_preserved(self):
        thesis = [_art(f"Story {i}") for i in range(10)]
        out = _partition_thesis_articles([], thesis, max_thesis=3)
        assert [a["title"] for a in out] == ["Story 0", "Story 1", "Story 2"]

    def test_zero_or_negative_cap_returns_empty(self):
        thesis = [_art("X")]
        assert _partition_thesis_articles([], thesis, 0) == []
        assert _partition_thesis_articles([], thesis, -1) == []

    def test_blank_titles_skipped(self):
        thesis = [_art(""), _art("   "), _art("Real headline")]
        out = _partition_thesis_articles([], thesis, max_thesis=8)
        assert [a["title"] for a in out] == ["Real headline"]

    def test_no_overlap_passes_all_through_up_to_cap(self):
        breaking = [_art("Unrelated")]
        thesis = [_art("A"), _art("B")]
        out = _partition_thesis_articles(breaking, thesis, max_thesis=8)
        assert [a["title"] for a in out] == ["A", "B"]

    def test_empty_inputs(self):
        assert _partition_thesis_articles([], [], 8) == []
