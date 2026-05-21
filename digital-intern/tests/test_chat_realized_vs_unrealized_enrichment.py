"""Pure-helper tests for the /api/chat realized-vs-unrealized enrichment.

`_realized_vs_unrealized_chat_lines` renders paper-trader's
`/api/realized-vs-unrealized` (banked-vs-paper P&L split) into compact
chat-context lines so the analyst can answer the composition question
no other equity surface answers: "of today's net P&L, how much is
locked-in vs paper that can evaporate?"

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_macro_calendar_chat_lines` / `_cash_redeployment_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED.
- **healthy = silence**: BANKED / BALANCED / NO_DATA verdicts collapse
  to `[]`, matching the `_decision_paralysis_chat_lines` silence
  precedent — a chat must not carry "you're fine" filler.
- **pure/total**: non-dict / missing keys / unparseable values never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _realized_vs_unrealized_chat_lines


def _rep(verdict="PAPER_HEAVY", *, headline=None,
         realized=10.0, unrealized=40.0, net_pct=5.0):
    if headline is None:
        # Mirror the builder's "VERDICT — …" prefix so chat tests that
        # assert headline starts with the verdict pass regardless of
        # which actionable verdict the fixture is parameterized to.
        headline = (
            f"{verdict} — today's ${realized + unrealized:.2f} "
            f"({net_pct:+.2f}%) gain split realized ${realized:+.2f} / "
            f"unrealized ${unrealized:+.2f}.")
    return {
        "as_of": "2026-05-21T11:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "starting_value": 1000.0,
        "current_value": 1000.0 + realized + unrealized,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "net_pnl_usd": realized + unrealized,
        "realized_pnl_pct": realized / 10.0,
        "unrealized_pnl_pct": unrealized / 10.0,
        "net_pnl_pct": net_pct,
    }


class TestPureTotalContract:
    @pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
    def test_non_dict_is_silence(self, bad):
        assert _realized_vs_unrealized_chat_lines(bad) == []

    def test_missing_verdict_is_silence(self):
        assert _realized_vs_unrealized_chat_lines({}) == []
        assert _realized_vs_unrealized_chat_lines({"headline": "x"}) == []


class TestSilenceOnNonActionable:
    @pytest.mark.parametrize("verdict", ["BANKED", "BALANCED", "NO_DATA",
                                         None, "OTHER"])
    def test_non_actionable_verdicts_silence(self, verdict):
        rep = _rep(verdict=verdict)
        assert _realized_vs_unrealized_chat_lines(rep) == []


class TestVerbatimHeadlineSSOT:
    """Invariant #10 — chat must not re-derive the verdict."""

    @pytest.mark.parametrize("verdict",
                             ["DRAWING_DOWN", "LEAKING_PAPER", "PAPER_HEAVY"])
    def test_headline_passes_through_verbatim(self, verdict):
        custom = f"{verdict} — totally custom test string $42.42 [exact]"
        rep = _rep(verdict=verdict, headline=custom)
        lines = _realized_vs_unrealized_chat_lines(rep)
        assert lines[0] == custom

    def test_blank_headline_is_skipped_but_detail_kept(self):
        rep = _rep(verdict="PAPER_HEAVY", headline="")
        lines = _realized_vs_unrealized_chat_lines(rep)
        # No verbatim line, but the detail line still appears.
        for ln in lines:
            assert ln  # not empty
        joined = "\n".join(lines)
        assert "realized" in joined and "unrealized" in joined


class TestDetailLineComposition:
    def test_detail_restates_builder_fields(self):
        rep = _rep(verdict="PAPER_HEAVY", realized=5.0, unrealized=45.0,
                   net_pct=5.0)
        lines = _realized_vs_unrealized_chat_lines(rep)
        joined = "\n".join(lines)
        # Detail line restates the builder's own $-fields verbatim
        # (rounded but not re-computed).
        assert "+5.00" in joined or "$+5.00" in joined
        assert "+45.00" in joined or "$+45.00" in joined
        assert "+5.00%" in joined or "+5.0" in joined

    def test_missing_fields_degrade_silently(self):
        rep = {"verdict": "DRAWING_DOWN", "headline": "DD test"}
        # No realized / unrealized / net_pct — only the headline survives.
        lines = _realized_vs_unrealized_chat_lines(rep)
        assert lines == ["DD test"]

    def test_garbage_numeric_fields_skip_not_raise(self):
        rep = {
            "verdict": "PAPER_HEAVY",
            "headline": "ph",
            "realized_pnl_usd": "x",
            "unrealized_pnl_usd": None,
            "net_pnl_pct": True,        # bool must NOT pass the _num filter
        }
        lines = _realized_vs_unrealized_chat_lines(rep)
        # No detail line because no usable numerics.
        assert lines == ["ph"]

    def test_drawing_down_fires_with_negative_net(self):
        rep = _rep(verdict="DRAWING_DOWN", realized=-10.0, unrealized=-5.0,
                   net_pct=-1.5)
        lines = _realized_vs_unrealized_chat_lines(rep)
        assert lines
        assert "DRAWING_DOWN" in lines[0]
        assert "-1.50%" in lines[1] or "-1.5" in lines[1]

    def test_leaking_paper_shows_split_directionality(self):
        # The chat block must show realized > 0 + unrealized < 0 verbatim.
        rep = _rep(verdict="LEAKING_PAPER", realized=20.0, unrealized=-30.0,
                   net_pct=-1.0)
        lines = _realized_vs_unrealized_chat_lines(rep)
        joined = "\n".join(lines)
        assert "+20.00" in joined or "$+20.00" in joined
        assert "-30.00" in joined or "$-30.00" in joined


class TestAllActionableVerdictsFire:
    @pytest.mark.parametrize("verdict",
                             ["DRAWING_DOWN", "LEAKING_PAPER", "PAPER_HEAVY"])
    def test_each_actionable_emits_at_least_headline(self, verdict):
        rep = _rep(verdict=verdict)
        lines = _realized_vs_unrealized_chat_lines(rep)
        assert lines
        assert lines[0].startswith(verdict)
