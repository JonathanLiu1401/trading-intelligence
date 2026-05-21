"""Tests for analytics/briefing_coverage_audit.py.

Pure-builder discipline: pre-built dict inputs, injected ``now``,
hand-computed expected counts. The audit is the *retrospective* sibling of
the briefing's own ``_coverage_gap_lines`` / ``_book_silence_lines`` —
silent-coverage regressions (the audit drifting from what the briefing
actually printed, a state threshold flipping, a missed ticker getting
silently dropped) would all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.briefing_coverage_audit import (  # noqa: E402
    _BOOK_TICKERS,
    _book_tickers_in_text,
    build_briefing_coverage_audit,
)

NOW = datetime(2026, 5, 21, 12, 30, 0, tzinfo=timezone.utc)
BRIEFING_TS = NOW - timedelta(minutes=5)
WINDOW_START = NOW - timedelta(hours=5)


def _briefing(text: str, ts: datetime | None = None,
              article_count: int = 50) -> dict:
    return {
        "ts": (ts or BRIEFING_TS).isoformat(timespec="seconds"),
        "text": text,
        "article_count": article_count,
    }


def _art(title: str, urgency: int = 1,
         summary: str = "", source: str = "rss",
         age_hours: float = 1.0) -> dict:
    return {
        "title": title,
        "summary": summary,
        "urgency": urgency,
        "source": source,
        "first_seen": (NOW - timedelta(hours=age_hours)).isoformat(),
    }


# ──────────────────── envelope & defensive cases ────────────────────────

class TestEmptyAndDefensive:
    def test_none_briefing_returns_no_briefing(self):
        rep = build_briefing_coverage_audit(None, [], now=NOW)
        assert rep["state"] == "NO_BRIEFING"
        assert rep["briefing_ts"] is None
        assert rep["n_covered"] == 0
        assert rep["n_missed"] == 0
        assert rep["coverage_ratio"] is None
        assert rep["covered"] == []
        assert rep["missed"] == []
        assert "nothing to audit" in rep["headline"].lower()

    def test_non_dict_briefing_returns_no_briefing(self):
        for bad in ("string", 42, [1, 2, 3]):
            rep = build_briefing_coverage_audit(bad, [], now=NOW)  # type: ignore[arg-type]
            assert rep["state"] == "NO_BRIEFING"

    def test_empty_text_returns_no_briefing(self):
        rep = build_briefing_coverage_audit(_briefing(""), [], now=NOW)
        assert rep["state"] == "NO_BRIEFING"

    def test_missing_ts_returns_no_briefing(self):
        rep = build_briefing_coverage_audit(
            {"text": "MU rallies", "ts": None}, [], now=NOW,
        )
        assert rep["state"] == "NO_BRIEFING"

    def test_non_list_articles_collapses_to_no_urgent(self):
        # Briefing exists but the articles iterable is junk — should not
        # raise, just report NO_URGENT (no ticker flow tallied).
        for bad in (None, "string", 42, {"not": "a list"}):
            rep = build_briefing_coverage_audit(
                _briefing("MU rallies"), bad, now=NOW,  # type: ignore[arg-type]
            )
            assert rep["state"] == "NO_URGENT"

    def test_non_dict_article_rows_skipped(self):
        rep = build_briefing_coverage_audit(
            _briefing("MU rallies"),
            [None, "string", 42, _art("MU prints beat")],
            now=NOW,
        )
        # The valid row contributes; the junk ones don't crash anything.
        assert rep["state"] == "COMPLETE"
        assert rep["n_urgent_articles"] >= 1
        assert rep["n_unique_tickers"] == 1

    def test_envelope_key_stability_across_states(self):
        expected_keys = {
            "as_of", "state", "headline", "briefing_ts",
            "briefing_age_hours", "window_start", "window_end",
            "window_hours", "n_urgent_articles", "n_unique_tickers",
            "n_covered", "n_missed", "coverage_ratio",
            "covered", "missed", "card_cap",
        }
        # NO_BRIEFING
        e1 = build_briefing_coverage_audit(None, [], now=NOW)
        # NO_URGENT
        e2 = build_briefing_coverage_audit(
            _briefing("MU rallies"), [], now=NOW,
        )
        # COMPLETE
        e3 = build_briefing_coverage_audit(
            _briefing("MU and NVDA rallies on print"),
            [_art("MU rallies"), _art("NVDA prints beat")],
            now=NOW,
        )
        # THIN
        e4 = build_briefing_coverage_audit(
            _briefing("Macro line only — no ticker callouts"),
            [_art("MU rallies"), _art("NVDA prints beat"),
             _art("ORCL up 3%")],
            now=NOW,
        )
        for env in (e1, e2, e3, e4):
            assert set(env.keys()) == expected_keys


# ──────────────────────── NO_URGENT state ────────────────────────────────

class TestNoUrgent:
    def test_briefing_with_no_book_ticker_flow_is_no_urgent(self):
        # An urgent article that doesn't touch the book universe shouldn't
        # generate a coverage row — that's outside scope.
        rep = build_briefing_coverage_audit(
            _briefing("MU rallies"),
            [_art("BABA tariff news", urgency=2)],
            now=NOW,
        )
        assert rep["state"] == "NO_URGENT"
        # n_urgent_articles counts ALL urgent rows even if none touch the book
        assert rep["n_urgent_articles"] == 1
        assert rep["n_unique_tickers"] == 0

    def test_no_urgent_carries_briefing_age(self):
        rep = build_briefing_coverage_audit(
            _briefing("MU rallies", ts=NOW - timedelta(minutes=30)),
            [],
            now=NOW,
        )
        assert rep["state"] == "NO_URGENT"
        # 30 min ≈ 0.5h, rounded to 2dp
        assert abs(rep["briefing_age_hours"] - 0.5) < 0.01


# ──────────────────────── COMPLETE state ─────────────────────────────────

class TestCompleteState:
    def test_all_tickers_covered_is_complete(self):
        # 2 unique tickers, both mentioned in briefing → 100% → COMPLETE.
        rep = build_briefing_coverage_audit(
            _briefing("MU prints beat; NVDA guides above."),
            [_art("MU prints beat", urgency=2),
             _art("NVDA earnings crush", urgency=2)],
            now=NOW,
        )
        assert rep["state"] == "COMPLETE"
        assert rep["n_covered"] == 2
        assert rep["n_missed"] == 0
        assert rep["coverage_ratio"] == 1.0
        assert {r["ticker"] for r in rep["covered"]} == {"MU", "NVDA"}
        assert rep["missed"] == []
        assert "complete" in rep["headline"].lower()

    def test_partial_briefing_above_80pct_floor_is_complete(self):
        # 5 unique tickers, 4 covered → 80% exactly → COMPLETE.
        arts = [
            _art("MU prints", urgency=2),
            _art("NVDA prints", urgency=2),
            _art("MSFT prints", urgency=2),
            _art("ORCL prints", urgency=2),
            _art("AXTI prints", urgency=2),  # the miss
        ]
        rep = build_briefing_coverage_audit(
            _briefing("MU, NVDA, MSFT, ORCL all moved on prints."),
            arts, now=NOW,
        )
        assert rep["state"] == "COMPLETE"
        assert rep["coverage_ratio"] == 0.8
        assert rep["n_covered"] == 4
        assert rep["n_missed"] == 1


# ──────────────────────── PARTIAL state ──────────────────────────────────

class TestPartialState:
    def test_50pct_to_80pct_is_partial(self):
        # 4 tickers, 2 covered → 50% → PARTIAL (floor inclusive).
        arts = [
            _art("MU prints", urgency=2),
            _art("NVDA prints", urgency=2),
            _art("MSFT prints", urgency=2),  # miss
            _art("ORCL prints", urgency=2),  # miss
        ]
        rep = build_briefing_coverage_audit(
            _briefing("MU prints beat. NVDA earnings crush."),
            arts, now=NOW,
        )
        assert rep["state"] == "PARTIAL"
        assert rep["coverage_ratio"] == 0.5
        assert rep["n_covered"] == 2
        assert rep["n_missed"] == 2
        # Headline names the top miss
        assert "miss:" in rep["headline"].lower()

    def test_partial_just_below_complete_floor(self):
        # 5 tickers, 3 covered → 60% → PARTIAL.
        arts = [
            _art("MU prints", urgency=2),
            _art("NVDA prints", urgency=2),
            _art("MSFT prints", urgency=2),
            _art("ORCL prints", urgency=2),  # miss
            _art("AXTI prints", urgency=2),  # miss
        ]
        rep = build_briefing_coverage_audit(
            _briefing("MU, NVDA, MSFT moved hard today."),
            arts, now=NOW,
        )
        assert rep["state"] == "PARTIAL"
        assert rep["coverage_ratio"] == 0.6


# ──────────────────────── THIN state ─────────────────────────────────────

class TestThinState:
    def test_below_50pct_is_thin(self):
        # 4 tickers, 1 covered → 25% → THIN.
        arts = [
            _art("MU prints", urgency=2),
            _art("NVDA prints", urgency=2),  # miss
            _art("MSFT prints", urgency=2),  # miss
            _art("ORCL prints", urgency=2),  # miss
        ]
        rep = build_briefing_coverage_audit(
            _briefing("MU prints beat — broader market mixed."),
            arts, now=NOW,
        )
        assert rep["state"] == "THIN"
        assert rep["coverage_ratio"] == 0.25
        assert rep["n_missed"] == 3
        # Headline calls out THIN explicitly
        assert "thin" in rep["headline"].lower()

    def test_zero_coverage_is_thin(self):
        # 2 tickers urgent, 0 in briefing → 0% → THIN.
        rep = build_briefing_coverage_audit(
            _briefing("Macro recap only — no ticker callouts in this draft."),
            [_art("MU prints", urgency=2),
             _art("NVDA prints", urgency=2)],
            now=NOW,
        )
        assert rep["state"] == "THIN"
        assert rep["coverage_ratio"] == 0.0
        assert rep["n_covered"] == 0
        assert rep["n_missed"] == 2


# ──────────────────── ticker extraction / regex edges ───────────────────

class TestTickerExtraction:
    def test_word_boundary_keeps_mu_out_of_museum(self):
        # MU should NOT fire inside "Museum" — same word-boundary discipline
        # as claude_analyst._BOOK_RE.
        hits = _book_tickers_in_text("Crypto Museum opens in Singapore")
        assert "MU" not in hits

    def test_longest_first_alternation_prefers_muu_over_mu(self):
        # The "MUU" alternation must beat the "MU" alternation when the
        # title literally writes MUU (mirror of the claude_analyst regex).
        hits = _book_tickers_in_text("MUU declares dividend")
        assert "MUU" in hits

    def test_non_string_text_returns_empty(self):
        assert _book_tickers_in_text(None) == set()
        assert _book_tickers_in_text(42) == set()
        assert _book_tickers_in_text(["MU"]) == set()
        assert _book_tickers_in_text("") == set()

    def test_summary_contributes_to_ticker_extraction(self):
        # Title is generic; summary names MU. Audit must still count it.
        rep = build_briefing_coverage_audit(
            _briefing("MU jumps on guide."),
            [{"title": "Memory sector rallies",
              "summary": "MU and SK Hynix lead the tape.",
              "urgency": 2,
              "source": "rss",
              "first_seen": NOW.isoformat()}],
            now=NOW,
        )
        assert rep["n_unique_tickers"] == 1
        assert rep["covered"][0]["ticker"] == "MU"

    def test_invalid_urgency_does_not_crash(self):
        # Garbage urgency should still let the row contribute its ticker
        # mention (we already SQL-filtered urgency >= 1 at the route layer;
        # the builder should be tolerant of upstream surprises).
        rep = build_briefing_coverage_audit(
            _briefing("MU rallies."),
            [{"title": "MU prints", "urgency": "high", "summary": "",
              "source": "rss", "first_seen": NOW.isoformat()}],
            now=NOW,
        )
        assert rep["n_unique_tickers"] == 1
        # max_urgency falls back to 0 on garbage input
        assert rep["covered"][0]["max_urgency"] == 0


# ──────────────────── ranking / aggregation / card cap ──────────────────

class TestRanking:
    def test_missed_ranked_by_max_urgency_then_article_count(self):
        # Two missed tickers — the higher-urgency one ranks first.
        arts = [
            _art("MU prints", urgency=2),                  # covered
            _art("NVDA misses", urgency=1),                # missed, urg=1
            _art("ORCL crushes guide", urgency=2),         # missed, urg=2
            _art("ORCL added to index", urgency=2),        # missed, urg=2
        ]
        rep = build_briefing_coverage_audit(
            _briefing("MU prints beat."),
            arts, now=NOW,
        )
        assert rep["state"] in ("PARTIAL", "THIN")
        assert rep["missed"][0]["ticker"] == "ORCL"
        assert rep["missed"][0]["max_urgency"] == 2
        assert rep["missed"][0]["n_articles"] == 2

    def test_per_ticker_article_count_tallies_correctly(self):
        # MU touched by three rows → n_articles == 3.
        arts = [
            _art("MU prints beat", urgency=2),
            _art("MU guides above", urgency=2),
            _art("MU added to index", urgency=1),
        ]
        rep = build_briefing_coverage_audit(
            _briefing("Macro line only."),
            arts, now=NOW,
        )
        assert rep["state"] == "THIN"
        assert rep["missed"][0]["ticker"] == "MU"
        assert rep["missed"][0]["n_articles"] == 3
        assert rep["missed"][0]["max_urgency"] == 2

    def test_card_cap_truncates_display_but_not_counts(self):
        # 4 missed book tickers, card_cap=2 → 2 rows but n_missed still 4.
        arts = [
            _art("MU prints", urgency=2),
            _art("NVDA prints", urgency=2),
            _art("MSFT prints", urgency=2),
            _art("ORCL prints", urgency=2),
        ]
        rep = build_briefing_coverage_audit(
            _briefing("Macro recap only."),
            arts, now=NOW, card_cap=2,
        )
        assert rep["state"] == "THIN"
        assert rep["n_missed"] == 4
        assert len(rep["missed"]) == 2

    def test_window_metadata_passthrough(self):
        rep = build_briefing_coverage_audit(
            _briefing("MU rallies."),
            [_art("MU prints", urgency=2)],
            window_start=WINDOW_START,
            window_end=NOW,
            now=NOW,
        )
        assert rep["window_start"] == WINDOW_START.isoformat(timespec="seconds")
        assert rep["window_end"] == NOW.isoformat(timespec="seconds")
        # 5h window, rounded
        assert abs(rep["window_hours"] - 5.0) < 0.01


# ────────────── parity with claude_analyst._BOOK_TICKERS ────────────────

class TestBookTickerParityWithClaudeAnalyst:
    """The audit duplicates the ``_BOOK_TICKERS`` literal rather than
    importing claude_analyst (anti-import-cycle discipline). Verify the
    two literals can't silently diverge — same drift-guard discipline
    as ``test_briefing_book_tag.py`` exercises for the other consumers."""

    def test_audit_book_set_equals_claude_analyst_book_set(self):
        from analysis.claude_analyst import _BOOK_TICKERS as CA_BOOK
        assert set(_BOOK_TICKERS) == set(CA_BOOK)

    def test_audit_book_order_equals_claude_analyst_order(self):
        # The canonical rank for tie-breaks must match — otherwise the
        # ranked outputs of the two modules sort differently and the
        # operator sees different tickers as "top miss".
        from analysis.claude_analyst import _BOOK_TICKERS as CA_BOOK
        assert tuple(_BOOK_TICKERS) == tuple(CA_BOOK)
