"""watchers/alert_agent.py — held-book relevance on the 🚨 BREAKING alert.

The alert is the analyst's most time-critical product and the persona is
explicitly "I react to events affecting MY positions". The prompt's mandatory
PORTFOLIO line previously relied entirely on Sonnet *inferring* held-ticker
relevance from the raw headline, so a real held-name break read identically to
generic macro colour. This pins the new ``_book_tickers`` helper, that the
``book:`` line + BOOK rule actually reach the Sonnet prompt, that a
no-position article emits NO ``book:`` line (never a fabricated one), and that
the held-ticker set is sourced verbatim from ``ml.features`` (single source of
truth — can never drift from the model's own ticker features or the briefing's
[BOOK:] tag). Read-only on the alert path: no ai_score / ml_score /
score_source / urgency / backtest-isolation surface is touched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import alert_agent
from ml.features import LIVE_PORTFOLIO_TICKERS


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class _StoreSpy:
    def __init__(self):
        self.marked = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)


# ── _book_tickers (pure) ─────────────────────────────────────────────────────

class TestBookTickersPure:
    def test_no_held_ticker_returns_empty(self):
        assert alert_agent._book_tickers(
            {"title": "Generic macro inflation print lands", "summary": ""}
        ) == []

    def test_single_held_ticker_in_title(self):
        assert alert_agent._book_tickers(
            {"title": "MU guides Q4 sharply above the Street", "summary": ""}
        ) == ["MU"]

    def test_multiple_held_tickers_sorted_deterministic(self):
        # Insertion order NVDA-then-MU; output must be sorted, stable.
        assert alert_agent._book_tickers(
            {"title": "NVDA and MU both surge on HBM demand", "summary": ""}
        ) == ["MU", "NVDA"]

    def test_matches_on_summary_not_only_title(self):
        # Title carries no ticker; summary does — parity with the briefing's
        # _book_tickers / ml.features ticker-density surface (title+summary).
        assert alert_agent._book_tickers(
            {"title": "Memory supply update hits the wire",
             "summary": "The note specifically flags AXTI substrate capacity."}
        ) == ["AXTI"]

    def test_muu_not_swallowed_by_mu_word_boundary(self):
        # \bMU\b must not match inside MUU; \bMUU\b must win.
        assert alert_agent._book_tickers(
            {"title": "MUU launches new product line", "summary": ""}
        ) == ["MUU"]

    def test_mu_not_matched_inside_company_name(self):
        # "Micron" contains 'mu' but not as a \b-delimited token → no false MU.
        assert alert_agent._book_tickers(
            {"title": "Micron expands Idaho fab", "summary": ""}
        ) == []

    def test_dedupes_repeated_mentions(self):
        assert alert_agent._book_tickers(
            {"title": "MU MU MU — MU keeps appearing", "summary": "MU again"}
        ) == ["MU"]

    def test_empty_and_missing_fields_safe(self):
        assert alert_agent._book_tickers({}) == []
        assert alert_agent._book_tickers(
            {"title": None, "summary": None}) == []

    def test_non_portfolio_ticker_not_flagged(self):
        # AAPL is a real ticker but not in the analyst's book → not flagged.
        assert "AAPL" not in LIVE_PORTFOLIO_TICKERS
        assert alert_agent._book_tickers(
            {"title": "AAPL hits a record high", "summary": ""}
        ) == []

    def test_set_is_sourced_from_ml_features_single_source_of_truth(self):
        # Drift guard: every ticker the alert flags must be one ml.features
        # already knows (no local copy that can silently diverge).
        found = alert_agent._book_tickers(
            {"title": " ".join(sorted(LIVE_PORTFOLIO_TICKERS)), "summary": ""}
        )
        assert set(found) == set(LIVE_PORTFOLIO_TICKERS)
        assert found == sorted(found)  # deterministic order


# ── end-to-end: book line + BOOK rule reach the Sonnet prompt ────────────────

class TestBookReachesSonnetPrompt:
    def _send(self, art, monkeypatch):
        spy = _StoreSpy()
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU"
                          ) as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True):
            ok = alert_agent.send_urgent_alert([art], spy)
        return ok, spy, mock_claude

    def test_held_row_emits_book_line_and_rule(self, monkeypatch):
        art = {
            "_id": "held1", "link": "https://reuters.com/x",
            "title": "MU guides Q4 revenue sharply above the Street",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        ok, spy, mock_claude = self._send(art, monkeypatch)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "book: MU — analyst HOLDS/watches these" in prompt
        # The BOOK rule must reach the prompt so Sonnet acts on the line.
        assert "BOOK: If an article carries a `book:` line" in prompt
        assert "PORTFOLIO line MUST" in prompt
        # Read-only contract preserved: exactly the live row marked alerted.
        assert spy.marked == ["held1"]

    def test_multi_ticker_book_line_is_sorted(self, monkeypatch):
        art = {
            "_id": "held2", "link": "https://reuters.com/y",
            "title": "NVDA and MU jump as Samsung HBM4 ships",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.2), "first_seen": _iso(0.05),
        }
        ok, _spy, mock_claude = self._send(art, monkeypatch)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "book: MU,NVDA — analyst HOLDS/watches these" in prompt

    def test_non_book_row_omits_book_line_no_fabrication(self, monkeypatch):
        art = {
            "_id": "nobook", "link": "https://reuters.com/z",
            "title": "Fed minutes reveal split on the next rate decision",
            "source": "rss", "ai_score": 8.0, "summary": "",
            "published": _iso(0.3), "first_seen": _iso(0.05),
        }
        ok, _spy, mock_claude = self._send(art, monkeypatch)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        # No held ticker in this headline → the per-row book: line is absent
        # (the BOOK *rule* is always in the static prompt; only the per-article
        # data line is conditional). Mirrors the related:-absent discipline.
        assert "\nbook:" not in prompt
