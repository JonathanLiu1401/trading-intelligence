"""Pure-helper tests for ``_actionable_opportunities_chat_lines``.

The helper renders paper-trader's ``/api/actionable-opportunities``
(composite ranker for unheld watchlist names — quant pred × news burst ×
persistent hot-run) into compact chat-context lines. Pinned: silence on
healthy/insufficient/error, verbatim headline + per-ticker reasons on
actionable verdicts, defensive against non-dict / missing-key payloads.

Pure-helper tests — no Flask. Follows the
``test_chat_cash_conviction_fit_enrichment`` precedent.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _actionable_opportunities_chat_lines  # noqa: E402


class TestSilenceOnNonActionableVerdicts:
    def test_non_dict_returns_empty(self):
        assert _actionable_opportunities_chat_lines(None) == []
        assert _actionable_opportunities_chat_lines("") == []
        assert _actionable_opportunities_chat_lines([]) == []

    def test_insufficient_data_silence(self):
        rep = {"verdict": "INSUFFICIENT_DATA", "headline": "scorer not qualified"}
        assert _actionable_opportunities_chat_lines(rep) == []

    def test_all_quiet_silence(self):
        rep = {"verdict": "ALL_QUIET", "headline": "everything weak"}
        assert _actionable_opportunities_chat_lines(rep) == []

    def test_error_silence(self):
        rep = {"verdict": "ERROR"}
        assert _actionable_opportunities_chat_lines(rep) == []

    def test_missing_verdict_silence(self):
        assert _actionable_opportunities_chat_lines({}) == []


class TestActionableVerdicts:
    def test_high_conviction_emits_headline_and_top_pick(self):
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "HIGH CONVICTION: AMD — scorer +26.1% AND news HOT (6.0×)",
            "by_ticker": [
                {
                    "ticker": "AMD",
                    "actionability": "HIGH_CONVICTION",
                    "reasons": [
                        "scorer +26.1% predicted 5d return (STRONG_HOLD)",
                        "news HOT (6.0× baseline mention rate)",
                    ],
                }
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines[0] == rep["headline"]   # SSOT — verbatim
        # Detail line restates AMD and BOTH reasons.
        joined = " ".join(lines[1:])
        assert "AMD" in joined
        assert "HIGH_CONVICTION" in joined
        assert "+26.1%" in joined
        assert "6.0×" in joined

    def test_scorer_but_no_news_documents_live_failure_mode(self):
        """Live 2026-05-27: 46 STRONG_HOLD scorer picks but news silent
        on every one. The block must surface, not collapse to silence."""
        rep = {
            "verdict": "SCORER_BUT_NO_NEWS",
            "headline": (
                "SCORER-ONLY: AMD — scorer +26.1% but news is COLD. "
                "Strong quant pick the wire hasn't caught yet."
            ),
            "by_ticker": [
                {"ticker": "AMD", "actionability": "SCORER_ONLY",
                 "reasons": ["scorer +26.1% predicted 5d return (STRONG_HOLD)"]},
                {"ticker": "MU", "actionability": "SCORER_ONLY",
                 "reasons": ["scorer +24.7% predicted 5d return (STRONG_HOLD)"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines[0] == rep["headline"]
        joined = " ".join(lines[1:])
        assert "AMD" in joined
        assert "MU" in joined
        assert "SCORER_ONLY" in joined

    def test_news_confirmed_emits(self):
        rep = {
            "verdict": "NEWS_CONFIRMED",
            "headline": "NEWS-CONFIRMED: SOXX — scorer +8.5% AND news HOT",
            "by_ticker": [
                {"ticker": "SOXX", "actionability": "NEWS_CONFIRMED",
                 "reasons": ["scorer +8.5%", "news HOT"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines[0] == rep["headline"]
        assert any("SOXX" in ln for ln in lines[1:])

    def test_news_but_no_scorer_emits(self):
        rep = {
            "verdict": "NEWS_BUT_NO_SCORER",
            "headline": "NEWS-ONLY: QBTS — news BLAZING but scorer +1.2%",
            "by_ticker": [
                {"ticker": "QBTS", "actionability": "NEWS_ONLY",
                 "reasons": ["scorer +1.2%", "news BLAZING (12.5× baseline)"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines[0] == rep["headline"]

    def test_persistent_followup_emits(self):
        rep = {
            "verdict": "PERSISTENT_FOLLOWUP",
            "headline": "PERSISTENT: FOO — scorer +6.0% AND 10h contiguous heat",
            "by_ticker": [
                {"ticker": "FOO", "actionability": "PERSISTENT_FOLLOWUP",
                 "reasons": ["scorer +6.0%", "10.0h contiguous news heat"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines[0] == rep["headline"]
        joined = " ".join(lines[1:])
        assert "PERSISTENT_FOLLOWUP" in joined


class TestDetailDefensiveness:
    def test_caps_at_three_rows(self):
        rep = {
            "verdict": "SCORER_BUT_NO_NEWS",
            "headline": "many strong picks, news cold on all",
            "by_ticker": [
                {"ticker": f"T{i:02d}", "actionability": "SCORER_ONLY",
                 "reasons": [f"scorer +{20+i}.0%"]}
                for i in range(10)
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        # 1 headline + 3 detail rows = 4 lines max
        assert len(lines) <= 4

    def test_skips_non_actionable_rows_inside_actionable_payload(self):
        """A SCORER_BUT_NO_NEWS payload may carry WEAK rows alongside
        SCORER_ONLY ones. WEAK rows must not appear in chat (the actionable
        rank is the point)."""
        rep = {
            "verdict": "SCORER_BUT_NO_NEWS",
            "headline": "AMD STRONG_HOLD news cold",
            "by_ticker": [
                {"ticker": "AMD", "actionability": "SCORER_ONLY",
                 "reasons": ["scorer +26%"]},
                {"ticker": "BORING", "actionability": "WEAK",
                 "reasons": ["scorer +0.5%"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert any("AMD" in ln for ln in lines)
        assert all("BORING" not in ln for ln in lines)

    def test_missing_reasons_emits_ticker_only(self):
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "AMD scorer + news",
            "by_ticker": [
                {"ticker": "AMD", "actionability": "HIGH_CONVICTION"},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert any("AMD" in ln for ln in lines)

    def test_garbage_row_skipped(self):
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "AMD scorer + news",
            "by_ticker": [
                {"ticker": "AMD", "actionability": "HIGH_CONVICTION",
                 "reasons": ["good"]},
                "not-a-dict",
                {"ticker": "", "actionability": "HIGH_CONVICTION"},  # empty tk
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        joined = " ".join(lines)
        assert "AMD" in joined
        assert "not-a-dict" not in joined

    def test_headline_only_when_no_rows(self):
        """Builder might emit an actionable verdict with empty by_ticker
        (defensive path). The headline must still appear."""
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "actionable thing",
            "by_ticker": [],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        assert lines == ["actionable thing"]

    def test_blank_headline_skipped(self):
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "   ",
            "by_ticker": [
                {"ticker": "AMD", "actionability": "HIGH_CONVICTION",
                 "reasons": ["good"]},
            ],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        # No blank headline line — only the detail row.
        assert all(ln.strip() for ln in lines)
        assert any("AMD" in ln for ln in lines)


class TestSSOTContract:
    def test_headline_is_verbatim_not_re_derived(self):
        """The helper must NOT alter the headline string. Builder owns
        verdict-naming SSOT (paper-trader invariant #10)."""
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "HIGH CONVICTION: AMD — scorer +26.1% AND news HOT (6.0×)",
            "by_ticker": [{"ticker": "AMD", "actionability": "HIGH_CONVICTION"}],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        # Bit-for-bit equality — no chat-side re-phrasing.
        assert lines[0] == rep["headline"]

    def test_reasons_strings_pass_through_verbatim(self):
        """The builder's `reasons` strings already encode the % / × /
        hours phrasing. The chat helper must not paraphrase them."""
        rep = {
            "verdict": "HIGH_CONVICTION_FOUND",
            "headline": "AMD strong",
            "by_ticker": [{
                "ticker": "AMD",
                "actionability": "HIGH_CONVICTION",
                "reasons": [
                    "scorer +26.1% predicted 5d return (STRONG_HOLD)",
                    "news HOT (6.0× baseline mention rate)",
                ],
            }],
        }
        lines = _actionable_opportunities_chat_lines(rep)
        joined = "\n".join(lines)
        for r in rep["by_ticker"][0]["reasons"]:
            assert r in joined
