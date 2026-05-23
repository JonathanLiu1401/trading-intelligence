"""SYSTEM_PROMPT carries the LIVE held-book in the [BOOK:] rule.

Regression guard mirroring tests/test_alert_held_book_prompt.py +
tests/test_urgency_portfolio_prompt.py (same drift class — three prompts that
each enumerate the held universe; if any one regresses to a hardcoded literal
the three silently disagree on what counts as held).

Before this fix, the Opus heartbeat briefing's [BOOK:] rule hardcoded
``(LITE, LNOK, MUU, DRAM, MU, NVDA, MSFT, AXTI, ORCL, TSEM, QBTS)`` — a 2026-05-23
live audit found GOOG/COHR/NVDL held in portfolio.json yet absent from this
literal, so Opus had no signal those rows touched the analyst's open book
when ranking TOP SIGNALS / picking the LEAD / writing the PORTFOLIO table.
Same SSOT (``_BOOK_UNIVERSE`` = static ``_BOOK_TICKERS`` ∪
``ml.features.LIVE_PORTFOLIO_TICKERS``) the runtime ``_book_tickers`` and
``_BOOK_RE`` already use, so the prompt's enumeration can never disagree with
the actual ``[BOOK:]`` tag-stamping path.
"""
from __future__ import annotations

from analysis import claude_analyst
from analysis.claude_analyst import (
    SYSTEM_PROMPT,
    _BOOK_TICKERS,
    _BOOK_UNIVERSE,
    _held_book_phrase,
)


class TestHeldBookPhrase:
    def test_returns_canonical_order_nonempty(self):
        line = _held_book_phrase()
        assert line, "held-positions slot must never be blank"
        toks = [t.strip() for t in line.split(",")]
        # First token is the canonical core's first member (LITE) — order is
        # _BOOK_TICKERS then live-only additions, NOT alphabetical.
        assert toks[0] == _BOOK_TICKERS[0]
        # No duplicates.
        assert len(toks) == len(set(toks))

    def test_includes_full_universe_static_and_live(self):
        line = _held_book_phrase()
        for t in _BOOK_UNIVERSE:
            assert t in line, f"{t} missing from briefing prompt held-book line"

    def test_universe_extends_static_core(self):
        """The universe MUST be a superset of the static core — the union
        contract."""
        for t in _BOOK_TICKERS:
            assert t in _BOOK_UNIVERSE


class TestSystemPromptFormatsWithHeldBook:
    def test_system_prompt_format_never_raises(self):
        out = SYSTEM_PROMPT.format(held_book=_held_book_phrase())
        # The rule is still present after substitution.
        assert "[BOOK:" in out
        # Every held ticker reached the prompt.
        for t in _BOOK_UNIVERSE:
            assert t in out, f"held ticker {t} not in formatted briefing prompt"

    def test_system_prompt_no_residual_hardcoded_book_literal(self):
        """The exact frozen enumeration that used to live in the rule body
        must be gone — if it reappears, parameterization regressed."""
        assert (
            "(LITE, LNOK, MUU, DRAM, MU, NVDA, MSFT, AXTI, ORCL, TSEM, QBTS)"
            not in SYSTEM_PROMPT
        )

    def test_held_book_slot_appears_exactly_once(self):
        """One enumeration site in the prompt; another copy would silently
        drift if the helper changed."""
        assert SYSTEM_PROMPT.count("{held_book}") == 1


def test_held_book_is_ssot_with_runtime_book_regex():
    """Bridging guarantee: every ticker the briefing prompt enumerates as
    "held" is ALSO a ticker the runtime ``_BOOK_RE`` would actually tag with
    ``[BOOK:]`` when seen in a row's title/summary. If the two paths diverge
    (e.g. the prompt lists a ticker the regex won't match) Opus would be told
    to expect a tag the formatter never produces."""
    line = _held_book_phrase()
    for tkr in [t.strip() for t in line.split(",")]:
        # The runtime regex is case-sensitive uppercase with word boundaries —
        # a synthetic "TICKER announces $1B buyback" must match.
        sample = f"{tkr} announces $1B buyback today"
        assert claude_analyst._BOOK_RE.search(sample), (
            f"prompt lists {tkr} as held but _BOOK_RE would not tag it"
        )
