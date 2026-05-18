"""Pure-helper tests for the /api/chat COVERAGE GAP enrichment (web_server.py).

A news analyst's most dangerous failure is a *silent* one: a high-value intel
channel goes dark and the chat simply contains nothing from it, so the absence
reads as "no news / calm" rather than "blind here". The 5h Opus briefing
already surfaces this via ``analysis.claude_analyst._coverage_gap_lines``; the
chat — the operator's primary interactive surface — did not, so it would
confidently answer "nothing notable on filings" while SEC 8-K had been dark
all session.

``_coverage_gap_chat_lines`` composes that **single source of truth verbatim**
(no re-derived gap logic — exactly the ``_behavioural_chat_lines`` /
``_game_plan_chat_lines`` invariant-#10 discipline) and obeys the shared total
contract: a non-dict, an empty report, or no curated channel disabled
contributes nothing — the block is omitted, never an exception into the chat
handler.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.claude_analyst import _coverage_gap_lines
from dashboard.web_server import _coverage_gap_chat_lines


def _disabled(fails: int, delivered: int = 0) -> dict:
    return {
        "disabled": True,
        "consecutive_failures": fails,
        "total_articles": delivered,
    }


def _healthy() -> dict:
    return {
        "disabled": False,
        "consecutive_failures": 0,
        "total_articles": 1234,
    }


class TestCoverageGapChatLines:
    def test_verbatim_composition_of_ssot(self):
        """The chat lines must be the briefing's own gap lines verbatim,
        only wrapped as bullets — not an independent re-derivation that can
        drift from the briefing the operator also reads."""
        report = {
            "sec_edgar": _disabled(932, delivered=0),   # priority 0
            "finnhub": _disabled(40, delivered=5),       # priority 1
            "rss": _healthy(),
        }
        out = _coverage_gap_chat_lines(report)
        expected = [f"• {ln}" for ln in _coverage_gap_lines(report)]
        assert out == expected
        assert out, "a disabled curated channel must produce gap lines"

    def test_filings_channel_ranked_first(self):
        """SEC filings are priority 0 — the operator must hear the most
        market-critical blindness first even if it has fewer dark hours."""
        report = {
            "nitter": _disabled(500, delivered=0),       # priority 3
            "sec_edgar": _disabled(10, delivered=0),     # priority 0
        }
        out = _coverage_gap_chat_lines(report)
        assert len(out) == 2
        assert "SEC 8-K filings" in out[0]
        assert "Nitter" in out[1]

    def test_dark_duration_and_session_blind_surfaced(self):
        """The line must carry the honest data: an estimated dark duration
        (~fails×cadence) and an explicit '0 delivered all session' so the
        chat can't soften a total blackout into a footnote."""
        # sec_edgar cadence is 300s → 932 fails ≈ 77.7h dark.
        report = {"sec_edgar": _disabled(932, delivered=0)}
        line = _coverage_gap_chat_lines(report)[0]
        assert "SEC 8-K filings" in line
        assert "DARK" in line
        assert "0 delivered all session" in line
        assert "77.7h" in line  # 932 * 300 / 3600

    def test_healthy_report_is_silent(self):
        """No curated channel disabled → no block. Silence here is correct;
        a 'coverage is fine' line would be exactly the noise the analyst
        persona complains about."""
        assert _coverage_gap_chat_lines(
            {"rss": _healthy(), "web": _healthy(), "gdelt": _healthy()}
        ) == []

    def test_uncurated_disabled_channel_ignored(self):
        """Only curated, analyst-meaningful channels surface — a disabled
        per-query gdelt junk key must NOT become a gap line (SSOT behaviour;
        we must not widen it)."""
        assert _coverage_gap_chat_lines(
            {"gdelt_query_aapl_xyz": _disabled(99, delivered=0)}
        ) == []

    def test_total_contract(self):
        assert _coverage_gap_chat_lines(None) == []
        assert _coverage_gap_chat_lines("nope") == []
        assert _coverage_gap_chat_lines({}) == []
        assert _coverage_gap_chat_lines([]) == []
        assert _coverage_gap_chat_lines(
            {"sec_edgar": "not-a-dict"}) == []
