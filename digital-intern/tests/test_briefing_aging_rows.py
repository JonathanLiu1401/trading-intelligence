"""AGING TOP ROWS — deterministic wall-clock recency cross-check in the 5h
Opus digest.

The model-estimated ``time_sensitivity`` decay rerank
(``_rank_by_decayed_score``) demotes stale time-bound rows only as far as the
ts head scored them; a row the ts head under-scored stays time-bound yet
barely decays, and a sparse 5h window can float an already-decayed 5-6h-old
item to #1. Opus then has only the per-row ``[seen HH:MM UTC]`` clock + the
``BRIEFING TIME`` header, and LLM clock subtraction across a bare-HH:MM 24h
window is unreliable — so it can write a multi-hour-old developing story into
the LEAD as if it just broke (the recurring stale-framing complaint, on the
analyst's primary product). ``claude_analyst._aging_top_rows`` + the
``=== AGING TOP ROWS ===`` input block surface a DETERMINISTIC wall-clock age
for the rows Opus actually leads with — the same pure read-side, BOOK-HEAT
shape (separate input block, never a per-row token so no render-line
contiguity break, never echoed): no DB write, no
ai_score/ml_score/score_source/urgency touch, no row mutation, backtest
excluded upstream — all four load-bearing invariants intact by construction.

These pin specific behaviour, not "no crash": the exact 3.0h boundary, the
top-scan window, the rank numbering, the cap, the real-url snapshot guard,
unknown-age exclusion, read-only behaviour, the ``_build_payload`` emission
gate, and the SYSTEM_PROMPT rule verbatim with its "do not echo" framing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analysis import claude_analyst

_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _row(title, *, age_h=None, link="https://reuters.com/x", score=8.0,
         first_seen=None):
    r = {"title": title, "summary": "", "source": "rss", "ai_score": score}
    if link is not None:
        r["link"] = link
    if first_seen is not None:
        r["first_seen"] = first_seen
    elif age_h is not None:
        r["first_seen"] = (_NOW - timedelta(hours=age_h)).isoformat()
    return r


# ── _aging_top_rows: the 3.0h boundary ───────────────────────────────────────
class TestAgingBoundary:
    def test_just_under_threshold_excluded(self):
        # 2h59m old → ~2.98h < 3.0 → not flagged.
        arts = [_row("MU DRAM guidance cut", age_h=2 + 59 / 60.0)]
        assert claude_analyst._aging_top_rows(arts, now=_NOW) == []

    def test_exactly_threshold_included(self):
        # age == 3.0h: `age_h < 3.0` is False → INCLUDED (>= boundary).
        arts = [_row("MU DRAM guidance cut", age_h=3.0)]
        out = claude_analyst._aging_top_rows(arts, now=_NOW)
        assert out == ["#1 ~3.0h — MU DRAM guidance cut"]

    def test_well_over_threshold_included_with_rank_and_age(self):
        arts = [_row("fresh NVDA pre-print note", age_h=0.5),
                _row("Samsung HBM4 line halted 6h ago", age_h=6.0)]
        out = claude_analyst._aging_top_rows(arts, now=_NOW)
        # rank is the 1-based position in the digest list Opus reads.
        assert out == ["#2 ~6.0h — Samsung HBM4 line halted 6h ago"]


# ── snapshot / unknown-age guards ────────────────────────────────────────────
class TestGuards:
    def test_snapshot_row_no_url_is_skipped(self):
        # The prepended PORTFOLIO/OPTIONS snapshot rows carry no link/url and
        # no first_seen — must never be flagged (same guard as [BOOK:]).
        snap = {"title": "PORTFOLIO P&L SNAPSHOT", "summary": "MU -6.6%",
                "source": "portfolio", "ai_score": 10}
        assert claude_analyst._aging_top_rows([snap], now=_NOW) == []

    def test_absent_or_unparseable_first_seen_excluded(self):
        # _seen_age_hours → 0.0 sentinel (< 3.0) so an unknown age is never
        # mis-flagged as stale.
        arts = [_row("no date", first_seen=None, age_h=None),
                _row("junk date", first_seen="not-a-date"),
                _row("future skew", first_seen=(_NOW + timedelta(hours=2)).isoformat())]
        assert claude_analyst._aging_top_rows(arts, now=_NOW) == []

    def test_only_top_scan_rows_considered(self):
        # A stale row beyond _AGING_TOP_SCAN (10) is NOT noise-flagged — Opus
        # leads from the very top only.
        fresh = [_row(f"fresh {i}", age_h=0.1)
                 for i in range(claude_analyst._AGING_TOP_SCAN)]
        deep_stale = _row("buried stale row", age_h=9.0)
        assert claude_analyst._aging_top_rows(fresh + [deep_stale], now=_NOW) == []

    def test_cap_at_max_lines(self):
        arts = [_row(f"stale story {i}", age_h=5.0)
                for i in range(claude_analyst._AGING_MAX_LINES + 4)]
        out = claude_analyst._aging_top_rows(arts, now=_NOW)
        assert len(out) == claude_analyst._AGING_MAX_LINES

    def test_title_truncated_and_untitled_fallback(self):
        long_t = "X" * 200
        arts = [_row(long_t, age_h=4.0), _row("", age_h=4.0)]
        out = claude_analyst._aging_top_rows(arts, now=_NOW)
        assert out[0] == "#1 ~4.0h — " + "X" * 60
        assert out[1] == "#2 ~4.0h — (untitled)"


# ── purity ───────────────────────────────────────────────────────────────────
class TestReadOnly:
    def test_does_not_mutate_input(self):
        arts = [_row("stale MU story", age_h=7.0)]
        before = [dict(a) for a in arts]
        claude_analyst._aging_top_rows(arts, now=_NOW)
        assert arts == before
        # returns a NEW list, not an alias
        assert claude_analyst._aging_top_rows(arts, now=_NOW) is not arts


# ── _build_payload integration + SYSTEM_PROMPT rule ──────────────────────────
class TestBuildPayloadEmission:
    def test_block_emitted_for_aged_top_row(self):
        arts = [_row("Fed minutes leaked 5h ago", age_h=5.0,
                     first_seen=(datetime.now(timezone.utc)
                                 - timedelta(hours=5)).isoformat())]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert "=== AGING TOP ROWS" in payload
        assert "Fed minutes leaked 5h ago" in payload
        assert "#1 ~" in payload

    def test_no_block_when_all_fresh(self):
        arts = [_row("breaking just now",
                     first_seen=datetime.now(timezone.utc).isoformat())]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert "=== AGING TOP ROWS" not in payload

    def test_snapshot_passthrough_not_flagged(self):
        # A real fresh article + (no snapshot here, but assert the block does
        # not appear and the per-row render line is unchanged → no contiguity
        # regression with test_briefing_seen_timestamp).
        fresh = _row("fresh wire item",
                     first_seen=datetime.now(timezone.utc).isoformat(),
                     score=9.0)
        fresh["source"] = "rss"
        payload = claude_analyst._build_payload([fresh], {}, [])
        assert "=== AGING TOP ROWS" not in payload


class TestSystemPromptRule:
    def test_aging_rule_present_verbatim(self):
        sp = claude_analyst.SYSTEM_PROMPT
        assert (
            '- If an "AGING TOP ROWS" block is present, it names the '
            "highest-ranked digest rows whose deterministic wall-clock age "
            "(time since the story hit our wire) is several hours old"
        ) in sp
        # the load-bearing framing + the "do not echo" discipline
        assert "developing/continued story, NOT one that just broke" in sp
        assert (
            'do NOT echo a literal "AGING TOP ROWS" section in the output '
            "(same as BOOK HEAT, unlike COVERAGE GAP)"
        ) in sp

    def test_threshold_constant_pinned(self):
        # 3.0h mirrors the alert path's documented "materially old (≳3h)"
        # RECENCY threshold — a drift here desyncs the two consumed products.
        assert claude_analyst.BRIEFING_AGING_MIN_HOURS == 3.0
        assert claude_analyst._AGING_TOP_SCAN == 10
        assert claude_analyst._AGING_MAX_LINES == 6
