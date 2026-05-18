"""Held-book relevance tag in the 5h Opus heartbeat briefing.

The Opus digest ranks rows by importance but never tells Opus which rows touch
positions the analyst actually has money in while it composes the LEAD /
TOP SIGNALS / PORTFOLIO table (the Discord-only ``_format_portfolio_coverage``
line is appended *after* the briefing). ``claude_analyst._book_tickers`` +
the ``[BOOK: ...]`` newswire tag surface that, in the exact pure read-side
shape of the established ``[syndicated xN]`` / ``[model]`` / ``[ALERTED]``
tags: no DB write, no ai_score/ml_score/score_source/urgency touch, no row
mutation, backtest excluded upstream — all four load-bearing invariants intact.

These pin: ticker matching correctness (word-boundary, MU≠MUU, no match inside
"Micron"/"MUSEUM"), canonical de-duplicated ordering, the real-url snapshot
guard, read-only behaviour, daemon parity (anti-drift), and the SYSTEM_PROMPT
rule with its LEAD/PORTFOLIO consequence.
"""
from __future__ import annotations

import daemon
from analysis import claude_analyst


# ── _book_tickers: matching correctness ──────────────────────────────────────
class TestBookTickers:
    def test_single_held_ticker_matched(self):
        assert claude_analyst._book_tickers(
            {"title": "MU beats Q3 estimates", "summary": ""}
        ) == ["MU"]

    def test_no_held_ticker_is_empty(self):
        assert claude_analyst._book_tickers(
            {"title": "Fed holds rates steady amid inflation", "summary": "x"}
        ) == []

    def test_word_boundary_mu_not_inside_micron(self):
        # \bMU\b must NOT fire inside "Micron" or "MUSEUM" — the documented
        # case-sensitive word-boundary convention (ml.features._LIVE_RE).
        assert claude_analyst._book_tickers(
            {"title": "Micron museum MUSEUM micromu", "summary": ""}
        ) == []

    def test_muu_distinct_from_mu(self):
        # "MUU" alone must not also yield "MU" (no \bMU\b boundary inside MUU).
        assert claude_analyst._book_tickers(
            {"title": "MUU leveraged ETF jumps", "summary": ""}
        ) == ["MUU"]
        # Both present as distinct tokens → both, in canonical order.
        out = claude_analyst._book_tickers(
            {"title": "MUU rips while MU dips", "summary": ""}
        )
        assert out == ["MUU", "MU"]

    def test_canonical_order_and_dedup(self):
        # Mention order is NVDA, MU, MU, NVDA — output must be canonical
        # (_BOOK_TICKERS order: MU before NVDA) and de-duplicated.
        out = claude_analyst._book_tickers(
            {"title": "NVDA and MU rally", "summary": "MU up again, NVDA too"}
        )
        assert out == ["MU", "NVDA"]

    def test_summary_contributes(self):
        assert claude_analyst._book_tickers(
            {"title": "Photonics roundup", "summary": "LITE wins a big order"}
        ) == ["LITE"]

    def test_empty_inputs_safe(self):
        assert claude_analyst._book_tickers({}) == []
        assert claude_analyst._book_tickers(
            {"title": None, "summary": None}
        ) == []


# ── _build_payload: the rendered [BOOK: ...] token ───────────────────────────
def _row(payload: str, needle: str) -> str:
    for ln in payload.splitlines():
        if needle in ln and ln.lstrip()[:1].isdigit():
            return ln
    raise AssertionError(f"no numbered newswire row containing {needle!r}")


class TestBookTagRendering:
    def test_held_row_with_url_is_tagged(self):
        arts = [{
            "title": "MU guides Q4 DRAM ASP sharply higher",
            "source": "rss", "ai_score": 9.0, "summary": "HBM demand",
            "link": "https://reuters.com/x",
        }]
        out = claude_analyst._build_payload(arts, {}, [])
        line = _row(out, "MU guides Q4 DRAM")
        # MU and DRAM are both held names; canonical order MU before DRAM?
        # _BOOK_TICKERS order: DRAM(idx3) before MU(idx5) → "DRAM,MU".
        assert "[BOOK: DRAM,MU]" in line, line

    def test_general_market_row_has_no_book_tag(self):
        arts = [{
            "title": "Fed signals surprise hold, yields whip lower",
            "source": "rss", "ai_score": 8.0, "summary": "macro",
            "link": "https://reuters.com/fed",
        }]
        out = claude_analyst._build_payload(arts, {}, [])
        assert "[BOOK:" not in _row(out, "Fed signals surprise hold")

    def test_snapshot_row_without_url_never_tagged(self):
        # daemon prepends PORTFOLIO/OPTIONS snapshots with NO link/url and a
        # P&L body that legitimately lists held tickers ("MU -6.6%"). Without
        # the real-url guard this would falsely render [BOOK: MU]. Same
        # snapshot-exclusion discipline as [ALERTED] / _extract_briefing_labels.
        arts = [{"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
                 "ai_score": 10, "summary": "MU -6.6%  NVDA +1.2%"}]
        out = claude_analyst._build_payload(arts, {}, [])
        assert "[BOOK:" not in _row(out, "PORTFOLIO P&L SNAPSHOT")

    def test_url_alias_honoured(self):
        # Some callers carry `url` not `link`; the guard tolerates both.
        arts = [{"title": "AXTI signs multi-year InP wafer supply deal",
                 "source": "gdelt", "ai_score": 7.0, "summary": "",
                 "url": "https://gdelt/x"}]
        out = claude_analyst._build_payload(arts, {}, [])
        assert "[BOOK: AXTI]" in _row(out, "AXTI signs multi-year")

    def test_build_payload_does_not_mutate_caller_dicts(self):
        # heartbeat_worker feeds this same list onward to the briefing-label /
        # training path — read-only on the dicts is load-bearing.
        a = {"title": "NVDA H200 export approval expands to China firms",
             "source": "rss", "ai_score": 8.0, "summary": "z",
             "link": "https://reuters.com/n"}
        before = dict(a)
        claude_analyst._build_payload([a], {}, [])
        assert a == before


# ── anti-drift: local literal must equal the daemon source of truth ──────────
def test_book_tickers_parity_with_daemon():
    """_BOOK_TICKERS is a local mirror of daemon.PORTFOLIO_TICKERS (kept local
    to avoid the daemon import graph). If the daemon list changes and this
    doesn't, the briefing silently mis-tags the analyst's book — this pins
    them together, the same anti-drift discipline as _COVERAGE_POLL_SECS."""
    assert frozenset(claude_analyst._BOOK_TICKERS) == frozenset(
        daemon.PORTFOLIO_TICKERS
    ), "claude_analyst._BOOK_TICKERS drifted from daemon.PORTFOLIO_TICKERS"


def test_system_prompt_rule_present_with_consequence():
    sp = claude_analyst.SYSTEM_PROMPT
    assert "[BOOK:" in sp, "SYSTEM_PROMPT must define the [BOOK:...] tag"
    low = sp.lower()
    # The rule must state the LEAD/PORTFOLIO consequence, not merely define it.
    assert "lead" in low and "portfolio" in low
