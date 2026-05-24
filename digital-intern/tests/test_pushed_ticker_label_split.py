"""Per-held-ticker push calibration: real Discord pushes only, by score_source.

The audit lives at the intersection of three existing primitives, each of
which leaves a real gap:

  * ``urgency_label_split_by_ticker`` is gate-noise-inflated (urgency>=1
    includes rows gates marked alerted to drain the queue).
  * ``pushed_ticker_breakdown`` is push-correct but has no score_source axis.
  * ``alert_delivery_audit.delivered_by_source`` has both push-correctness AND
    score_source — but only aggregated, not per-held-ticker.

These tests pin the invariants the audit must preserve.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import pushed_ticker_label_split as P
from watchers.alert_dedup import _signature


def _fresh_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _row(*, _id, title, source="rss", summary="",
         ai_score=0, ml_score=None, score_source=None):
    return {
        "_id": _id, "title": title, "source": source, "summary": summary,
        "ai_score": ai_score, "ml_score": ml_score,
        "score_source": score_source,
    }


# ── pure function: empty inputs ─────────────────────────────────────────────
class TestEmptyInputs:
    def test_no_tickers_returns_zero_shape(self):
        out = P.compute_pushed_ticker_label_split([], set(), [])
        assert out["total_pushes"] == 0
        assert out["by_ticker"] == []
        assert out["silent_tickers"] == []

    def test_no_urgent_rows_all_silent(self):
        out = P.compute_pushed_ticker_label_split(
            [], set(), ["NVDA", "MU"])
        assert out["total_pushes"] == 0
        assert out["by_ticker"] == []
        # Silent because no pushes happened — every ticker is in the gap.
        assert sorted(out["silent_tickers"]) == ["MU", "NVDA"]

    def test_no_alerted_sigs_all_silent(self):
        """An urgency=2 row whose signature is NOT in alerted_sigs was
        gate-marked, not pushed — must be dropped, never counted per-ticker."""
        art = _row(_id="a", title="NVDA earnings beat Q3 long enough now",
                   ai_score=9.0, score_source="llm")
        out = P.compute_pushed_ticker_label_split(
            [art], set(), ["NVDA", "MU"])
        assert out["total_pushes"] == 0
        assert out["by_ticker"] == []
        assert "NVDA" in out["silent_tickers"]


# ── push-vs-gate-marked discrimination (the load-bearing invariant) ─────────
class TestPushDiscrimination:
    def test_only_signature_matching_rows_count(self):
        """The crux: alert_recency.db is the analyst-truth ledger. A row
        whose signature is NOT in alerted_sigs is gate-suppressed — invisible
        to the analyst. The per-ticker count must reflect what the analyst
        actually received, not what the dashboard ``urgent`` tile counts."""
        pushed_title = "NVDA beats Q3 estimates on AI chip demand cycle"
        gated_title = "Why NVDA Stock Is Trading Up Today on AI hype"  # recap
        pushed = _row(_id="p", title=pushed_title, ai_score=9.0,
                      score_source="llm")
        gated = _row(_id="g", title=gated_title, ml_score=9.5,
                     score_source="ml")
        sigs = {_signature(pushed_title)}
        out = P.compute_pushed_ticker_label_split(
            [pushed, gated], sigs, ["NVDA"])
        # Only the pushed row contributes; the gated recap is invisible.
        assert out["total_pushes"] == 1
        assert len(out["by_ticker"]) == 1
        row = out["by_ticker"][0]
        assert row["ticker"] == "NVDA"
        assert row["total"] == 1
        assert row["llm"] == 1
        assert row["ml"] == 0


# ── score_source attribution ────────────────────────────────────────────────
class TestScoreSourceAttribution:
    def test_ml_only_push_counted_as_ml(self):
        """A push whose label is score_source='ml' (ai_score=0, ml_score>0) is
        the calibration concern this whole module exists to surface."""
        title = "Memory prices climb on Samsung union strike disruption"
        art = _row(_id="a", title=title, ai_score=0, ml_score=9.5,
                   score_source="ml", summary="MU and Samsung sector context")
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, ["MU"])
        row = out["by_ticker"][0]
        assert row["ticker"] == "MU"
        assert row["ml"] == 1 and row["llm"] == 0
        assert row["llm_fraction"] == 0.0

    def test_llm_push_counted_as_llm_with_fraction_1(self):
        title = "Citi raises MU price target to 200 on DRAM cycle"
        art = _row(_id="a", title=title, ai_score=9.0, score_source="llm",
                   summary="Citi cycle commentary names MU")
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, ["MU"])
        row = out["by_ticker"][0]
        assert row["llm"] == 1 and row["ml"] == 0
        assert row["llm_fraction"] == 1.0

    def test_mixed_score_sources_per_ticker(self):
        """A ticker with both LLM-vetted AND ml-only pushes must show both
        counts AND the right fraction (canonical = (llm+briefing_boost)/total)."""
        ml_title = "NVDA Q1 revenue surge headlines memory expansion"
        llm_title = "Bank of America raises NVDA price target to 220"
        ml_art = _row(_id="m", title=ml_title, ai_score=0, ml_score=9.9,
                      score_source="ml")
        llm_art = _row(_id="l", title=llm_title, ai_score=9.0,
                       score_source="llm")
        sigs = {_signature(ml_title), _signature(llm_title)}
        out = P.compute_pushed_ticker_label_split(
            [ml_art, llm_art], sigs, ["NVDA"])
        row = out["by_ticker"][0]
        assert row["total"] == 2
        assert row["llm"] == 1 and row["ml"] == 1
        assert row["llm_fraction"] == 0.5

    def test_null_score_source_attributed_to_null(self):
        """Legacy/pre-migration rows with no score_source tag fall to 'null'
        bucket — same convention as urgency_label_split. They neither count as
        LLM-vetted nor inflate the ml-only number."""
        title = "Legacy row from before score_source migration here"
        art = _row(_id="a", title=title, ai_score=7.0, score_source=None)
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, [])
        # No tickers requested → nothing per-ticker, but total_pushes counts it.
        assert out["total_pushes"] == 1


# ── syndication fold (one push per signature, not per row) ──────────────────
class TestSyndicationFold:
    def test_multiple_urgency_rows_same_sig_count_once_per_ticker(self):
        """A single wire syndicated across GDELT/Reuters + Yahoo + Finnhub
        landed as 3 urgency=2 rows but is ONE Discord push (the alert_recency
        signature collapses them). The per-ticker count must reflect that:
        otherwise NVDA looks 3x more pushed than it actually was."""
        title = "NVDA smashes Q1 guidance and raises buyback authorization"
        copies = [
            _row(_id="r1", title=title, source="GDELT/reuters.com",
                 ai_score=0, ml_score=9.5, score_source="ml"),
            _row(_id="r2", title=title, source="rss",
                 ai_score=0, ml_score=9.4, score_source="ml"),
            _row(_id="r3", title=title, source="Finnhub/Yahoo",
                 ai_score=9.0, score_source="llm"),  # the LLM-vetted copy wins
        ]
        sigs = {_signature(title)}
        out = P.compute_pushed_ticker_label_split(copies, sigs, ["NVDA"])
        assert out["total_pushes"] == 1, (
            "syndicated copies must fold into ONE push the analyst's POV"
        )
        row = out["by_ticker"][0]
        assert row["total"] == 1
        # LLM-vetted copy wins source attribution — ground truth beats model.
        assert row["llm"] == 1 and row["ml"] == 0


# ── per-ticker matching surface (title + summary, whole-word) ───────────────
class TestTickerMatching:
    def test_title_only_match_counted(self):
        title = "MU shares climb 4% on Citi note long enough now"
        art = _row(_id="a", title=title, ai_score=9.0, score_source="llm")
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, ["MU", "NVDA"])
        tickers_seen = {r["ticker"] for r in out["by_ticker"]}
        assert tickers_seen == {"MU"}
        assert "NVDA" in out["silent_tickers"]

    def test_summary_match_counted(self):
        title = "Memory pricing analyst note from Citi covers field"
        art = _row(_id="a", title=title, summary="The note discusses MU specifically",
                   ai_score=9.0, score_source="llm")
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, ["MU"])
        assert out["by_ticker"][0]["ticker"] == "MU"

    def test_substring_does_not_match(self):
        """Whole-word match: ``DAMD`` must NOT count as AMD; ``MUTUAL`` must
        NOT count as MU. Same hygiene as ``_LIVE_RE``."""
        title = "MUTUAL fund flows from DAMD subsector this week here"
        art = _row(_id="a", title=title, ai_score=9.0, score_source="llm")
        out = P.compute_pushed_ticker_label_split(
            [art], {_signature(title)}, ["MU", "AMD"])
        # Neither held name appears as a whole word.
        assert out["by_ticker"] == []
        assert sorted(out["silent_tickers"]) == ["AMD", "MU"]


# ── ordering (most-ml-only-first, alphabetical tiebreak) ────────────────────
class TestOrdering:
    def test_sorted_most_ml_first_then_alpha(self):
        # Build 3 distinct pushes: NVDA gets 2 ml, MU gets 1 ml, AAPL gets 1 llm.
        a1 = _row(_id="a1", title="NVDA chip news one long enough now",
                  ai_score=0, ml_score=9, score_source="ml")
        a2 = _row(_id="a2", title="NVDA chip news two long enough now",
                  ai_score=0, ml_score=9, score_source="ml")
        a3 = _row(_id="a3", title="MU memory news one long enough now",
                  ai_score=0, ml_score=9, score_source="ml")
        a4 = _row(_id="a4", title="AAPL services news one long enough now",
                  ai_score=8, score_source="llm")
        sigs = {_signature(a["title"]) for a in (a1, a2, a3, a4)}
        out = P.compute_pushed_ticker_label_split(
            [a1, a2, a3, a4], sigs, ["NVDA", "MU", "AAPL"])
        order = [r["ticker"] for r in out["by_ticker"]]
        # NVDA (ml=2) > MU (ml=1) > AAPL (ml=0). AAPL appears last (lowest ml).
        assert order == ["NVDA", "MU", "AAPL"]


# ── load-bearing invariant: no reads of backtest:// / backtest_ are possible
#    via the pure helper (the SQL guard is on the DB shell). This pins the
#    pure-function contract — anything passed in is trusted; the shell is the
#    one that filters. Verify the DB-shell SQL contains the canonical clause.
class TestBacktestIsolationContract:
    def test_live_only_clause_matches_canonical(self):
        from storage.article_store import _LIVE_ONLY_CLAUSE
        assert P.LIVE_ONLY_CLAUSE == _LIVE_ONLY_CLAUSE, (
            "Anti-drift: the analytics-side clause must mirror the storage SSOT "
            "verbatim (same discipline as alert_delivery_audit.LIVE_ONLY_CLAUSE)."
        )


# ── DB shell integration: smoke test that run() can build a report when
#    both DBs are absent (degrades gracefully — never raises).
class TestRunDegradesGracefully:
    def test_missing_recency_db_yields_silent_book(self, monkeypatch, tmp_path):
        """A fresh install with no alert_recency.db must return a clean empty
        report, not crash. Same shape as alert_delivery_audit.run_audit."""
        # Point both DB paths at non-existent files.
        fake_articles = tmp_path / "articles.db"
        fake_recency = tmp_path / "alert_recency.db"
        # Create an empty articles.db so the SELECT works.
        import sqlite3
        c = sqlite3.connect(str(fake_articles))
        c.execute(
            "CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
            "source TEXT, published TEXT, kw_score REAL, ai_score REAL, "
            "urgency INTEGER, full_text BLOB, first_seen TEXT, cycle INTEGER, "
            "time_sensitivity REAL, ml_score REAL, score_source TEXT)"
        )
        c.commit()
        c.close()
        monkeypatch.setattr(P, "resolve_db_paths",
                            lambda: (fake_articles, fake_recency))
        report = P.run(tickers=["NVDA", "MU"], hours=6.0)
        assert report["total_pushes"] == 0
        assert sorted(report["silent_tickers"]) == ["MU", "NVDA"]
        assert "window_h" in report and "generated_at" in report
