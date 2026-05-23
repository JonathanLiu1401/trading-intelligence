"""ALERT_PROMPT carries the LIVE held-book in the PORTFOLIO + BOOK slots.

Regression guard mirroring tests/test_urgency_portfolio_prompt.py (same drift
class). Before this fix, the Bloomberg BREAKING alert prompt hardcoded the
held set in two places:

  * the ``PORTFOLIO: [specific implication for LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS]``
    template line, and
  * the ``BOOK:`` rule that defined which tickers count as "live
    portfolio/watchlist positions the analyst actually has money in".

A position added in the trading UI (e.g. GOOG/COHR/NVDL on the 2026-05-23
live audit) was therefore invisible to the alert formatter — its PORTFOLIO
implication never named the held ticker, and the ``BOOK:`` rule's enumeration
did not list it, so Sonnet had no signal those rows touched the open book.
Same SSOT (``ml.features.LIVE_PORTFOLIO_TICKERS``) the urgency_scorer prompt
now uses, so the two paths can never silently disagree on what counts as
held.
"""
from __future__ import annotations

from watchers import alert_agent
from ml.features import LIVE_PORTFOLIO_TICKERS


class TestHeldBookPhrase:
    def test_returns_sorted_nonempty_slash_separated(self):
        line = alert_agent._held_book_phrase()
        assert line, "held-positions slot must never be blank"
        toks = [t.strip() for t in line.split("/")]
        # Deterministic, sorted order so the prompt is test-pinnable.
        assert toks == sorted(toks)
        # Multiple held names — the live book is never just one ticker.
        assert len(toks) >= 2

    def test_includes_every_live_portfolio_ticker(self):
        line = alert_agent._held_book_phrase()
        for t in LIVE_PORTFOLIO_TICKERS:
            assert t in line, f"{t} missing from alert prompt held-book line"

    def test_phrase_uses_slash_not_comma(self):
        """The alert template literal uses slash-separation
        ("LITE/MU/MSFT/..."); preserve that visual convention."""
        line = alert_agent._held_book_phrase()
        assert "/" in line
        assert "," not in line


class TestAlertPromptFormatsWithHeldBook:
    def test_alert_prompt_format_never_raises(self):
        """The prompt is a .format() template with literal {now_utc},
        {articles_text}, and the new {held_book} slot — formatting with all
        three must succeed."""
        out = alert_agent.ALERT_PROMPT.format(
            articles_text="[no articles]",
            now_utc="2026-05-23 14:00",
            held_book=alert_agent._held_book_phrase(),
        )
        assert "BREAKING" in out
        # PORTFOLIO line carries the live set, not the old literal.
        assert "PORTFOLIO:" in out
        for t in LIVE_PORTFOLIO_TICKERS:
            assert t in out, f"held ticker {t} not in formatted alert prompt"

    def test_alert_prompt_no_residual_hardcoded_literal(self):
        """The exact frozen literals that used to live in the template body
        must be gone — if they reappear, parameterization regressed."""
        # The two phrases the drift fix removed (the PORTFOLIO line example
        # and the BOOK-rule enumeration).
        assert "LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS" not in alert_agent.ALERT_PROMPT
        assert "LITE/LNOK/MUU/DRAM/SNDU/MU/MSFT/AXTI/ORCL/TSEM/QBTS/NVDA" not in alert_agent.ALERT_PROMPT

    def test_held_book_slot_appears_in_both_portfolio_and_book_sections(self):
        """The same {held_book} placeholder is consumed by two distinct
        sections of the prompt (PORTFOLIO template + BOOK rule); if either
        regresses to a hardcoded literal the two will silently disagree."""
        assert alert_agent.ALERT_PROMPT.count("{held_book}") == 2
