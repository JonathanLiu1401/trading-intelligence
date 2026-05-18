"""BOOK HEAT — held-name distinct-story concentration in the 5h Opus digest.

The per-row ``[BOOK: ...]`` tag tells Opus WHICH rows touch the analyst's open
positions, but never that a single held name is the window's centre of gravity
(one MU story scoring 7 may not lead; MU spread across 6 *distinct* stories is
a magnitude signal in its own right). ``claude_analyst._book_heat_lines`` + the
``=== BOOK HEAT ===`` input block surface that in the exact pure read-side
shape of the established ``[syndicated xN]`` / ``[BOOK:]`` tags: no DB write,
no ai_score/ml_score/score_source/urgency touch, no row mutation, backtest
excluded upstream — all four load-bearing invariants intact by construction.

These pin: distinct-story counting (NOT syndicated-copy counting), the
threshold, the real-url snapshot guard, count-desc/canonical-tie ordering, the
max-lines cap, read-only behaviour, the _build_payload emission gate, and the
SYSTEM_PROMPT rule with its LEAD/PORTFOLIO consequence + "do not echo" framing.
"""
from __future__ import annotations

from analysis import claude_analyst


def _url_row(title, summary="", score=7.0):
    return {"title": title, "summary": summary, "source": "rss",
            "ai_score": score, "link": "https://reuters.com/" + title[:8]}


# ── _book_heat_lines: counting + threshold ───────────────────────────────────
class TestBookHeatLines:
    def test_below_threshold_is_empty(self):
        # 2 distinct MU stories < BOOK_HEAT_MIN_STORIES (3) → no heat.
        arts = [_url_row("MU smashes Q3 DRAM guidance"),
                _url_row("MU CFO departs abruptly midweek")]
        assert claude_analyst._book_heat_lines(arts) == []

    def test_at_threshold_emits_exact_count(self):
        arts = [_url_row("MU smashes Q3 DRAM guidance"),
                _url_row("MU CFO departs abruptly midweek"),
                _url_row("MU added to Goldman conviction list")]
        assert claude_analyst._book_heat_lines(arts) == [
            "MU — 3 distinct stories"
        ]

    def test_distinct_count_is_per_article_not_per_mention(self):
        # MU mentioned twice in ONE row counts once for that row.
        arts = [_url_row("MU and MU again headline one"),
                _url_row("MU second distinct story here"),
                _url_row("MU third distinct story here")]
        assert claude_analyst._book_heat_lines(arts) == [
            "MU — 3 distinct stories"
        ]

    def test_snapshot_rows_without_url_excluded(self):
        # Same real-url guard as the [BOOK:] tag: a snapshot P&L body that
        # lists held tickers must NOT manufacture phantom heat.
        snaps = [{"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
                  "ai_score": 10, "summary": "MU -6.6% MU lower MU again"}
                 for _ in range(5)]
        assert claude_analyst._book_heat_lines(snaps) == []

    def test_url_alias_honoured(self):
        rows = [{"title": f"AXTI InP wafer deal number {i}", "summary": "",
                 "source": "gdelt", "ai_score": 6.0, "url": f"https://g/{i}"}
                for i in range(3)]
        assert claude_analyst._book_heat_lines(rows) == [
            "AXTI — 3 distinct stories"
        ]

    def test_ordering_count_desc_then_canonical(self):
        # NVDA in 4 stories, MU in 3. Count desc → NVDA first even though MU
        # precedes NVDA in canonical _BOOK_TICKERS order.
        arts = ([_url_row(f"NVDA blackwell story {i}") for i in range(4)]
                + [_url_row(f"MU dram story {i}") for i in range(3)])
        assert claude_analyst._book_heat_lines(arts) == [
            "NVDA — 4 distinct stories",
            "MU — 3 distinct stories",
        ]

    def test_tie_breaks_to_canonical_order(self):
        # MU and NVDA both in 3 → tie broken by canonical order (MU < NVDA).
        arts = ([_url_row(f"MU memory story {i}") for i in range(3)]
                + [_url_row(f"NVDA gpu story {i}") for i in range(3)])
        assert claude_analyst._book_heat_lines(arts) == [
            "MU — 3 distinct stories",
            "NVDA — 3 distinct stories",
        ]

    def test_max_lines_cap(self):
        # 8 distinct held names each at threshold → capped at
        # _BOOK_HEAT_MAX_LINES (6).
        names = ["MU", "NVDA", "MSFT", "ORCL", "TSEM", "QBTS", "AXTI", "LITE"]
        arts = []
        for nm in names:
            arts += [_url_row(f"{nm} distinct story number {i}")
                     for i in range(3)]
        out = claude_analyst._book_heat_lines(arts)
        assert len(out) == claude_analyst._BOOK_HEAT_MAX_LINES == 6

    def test_empty_and_garbage_safe(self):
        assert claude_analyst._book_heat_lines([]) == []
        assert claude_analyst._book_heat_lines(
            [{"title": None, "summary": None, "link": "https://x/y"}]
        ) == []
        assert claude_analyst._book_heat_lines(
            [_url_row("Fed holds rates steady amid inflation worry")] * 5
        ) == []


# ── _build_payload: emission gate + honest distinct counting ─────────────────
def _has_heat_block(payload: str) -> bool:
    return "=== BOOK HEAT" in payload


class TestBuildPayloadEmission:
    def test_no_block_below_threshold(self):
        arts = [_url_row("MU smashes Q3 DRAM guidance", score=9.0),
                _url_row("MU CFO departs abruptly midweek", score=8.0)]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert not _has_heat_block(payload)

    def test_block_present_with_three_distinct_stories(self):
        arts = [_url_row("MU smashes Q3 DRAM guidance", score=9.0),
                _url_row("MU CFO departs abruptly midweek", score=8.0),
                _url_row("MU added to Goldman conviction list", score=7.0)]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert _has_heat_block(payload)
        assert "MU — 3 distinct stories" in payload

    def test_syndicated_copies_collapse_to_one_story(self):
        """Honest-counting contract: 5 syndicated copies of ONE MU headline
        are collapsed by _collapse_syndicated BEFORE heat is counted, so they
        are NOT 5 stories → no heat. The whole point of counting over the
        post-collapse digest."""
        same = "Micron shares surge after Q3 earnings blowout"
        arts = [{"title": same, "summary": "", "source": s, "ai_score": 9.0,
                 "link": f"https://{s}.com/x"}
                for s in ("reuters", "yahoo", "gdelt", "rss", "finnhub")]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert not _has_heat_block(payload), (
            "syndicated copies of one event must NOT inflate distinct-story heat"
        )

    def test_build_payload_does_not_mutate_caller_dicts(self):
        # heartbeat_worker feeds this same list onward to the briefing-label /
        # training path — read-only on the dicts is load-bearing.
        arts = [_url_row(f"MU distinct story headline {i}") for i in range(3)]
        before = [dict(a) for a in arts]
        claude_analyst._build_payload(arts, {}, [])
        assert arts == before


# ── SYSTEM_PROMPT rule ───────────────────────────────────────────────────────
def test_system_prompt_rule_present_with_consequence():
    sp = claude_analyst.SYSTEM_PROMPT
    assert "BOOK HEAT" in sp, "SYSTEM_PROMPT must define the BOOK HEAT hint"
    low = sp.lower()
    # Must state the LEAD/TOP-SIGNALS/PORTFOLIO consequence, not merely name it.
    assert "lead" in low and "portfolio" in low
    # Must explicitly say it is a hint, NOT a reproduced section (unlike
    # COVERAGE GAP) — otherwise Opus burns 1800-char budget echoing it.
    assert "do not echo" in low or "do not reproduce" in low or "hint only" in low
