"""Tests for analytics.collector_direct_urgent_audit.

Pure-function tests over synthetic rows + an end-to-end test that runs the
real read path against an in-memory ArticleStore. Assertions target specific
counts and verdicts so a regression in the classifier or aggregator surfaces
immediately.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from analytics import collector_direct_urgent_audit as cdua
from watchers.urgency_scorer import URGENT_THRESHOLD


# ── classify_row: the per-row discriminator ──────────────────────────────────
class TestClassifyRow:
    def test_no_scores_is_kw_only_uncorroborated(self):
        """The bug pattern: collector set urgency=1, pipeline never scored it."""
        f = cdua.classify_row(ai_score=0.0, ml_score=None)
        assert f["kw_only"] is True
        assert f["llm_urgent"] is False
        assert f["ml_urgent"] is False
        assert f["uncorroborated"] is True

    def test_llm_at_threshold_is_llm_urgent(self):
        f = cdua.classify_row(ai_score=URGENT_THRESHOLD, ml_score=None)
        assert f["llm_urgent"] is True
        assert f["kw_only"] is False
        assert f["uncorroborated"] is False

    def test_llm_below_threshold_not_llm_urgent_but_still_uncorroborated(self):
        f = cdua.classify_row(ai_score=URGENT_THRESHOLD - 0.01, ml_score=None)
        assert f["llm_urgent"] is False
        assert f["kw_only"] is False  # ai_score > 0 means LLM ran
        assert f["uncorroborated"] is True  # but didn't endorse urgent

    def test_ml_at_threshold_is_ml_urgent(self):
        f = cdua.classify_row(ai_score=0.0, ml_score=URGENT_THRESHOLD)
        assert f["ml_urgent"] is True
        assert f["kw_only"] is False
        assert f["uncorroborated"] is False

    def test_ml_below_threshold_remains_uncorroborated(self):
        """EXACTLY the commodity_futures noise pattern: collector wrote
        urgency=1, alert fired, scorer later assigned ml_score=6.32 (< 8.0).
        ml ran but did NOT endorse — so the alert was uncorroborated."""
        f = cdua.classify_row(ai_score=0.0, ml_score=6.32)
        assert f["ml_urgent"] is False
        assert f["kw_only"] is False  # ml ran
        assert f["uncorroborated"] is True

    def test_both_llm_and_ml_urgent_flags_set(self):
        f = cdua.classify_row(ai_score=9.0, ml_score=8.5)
        assert f["llm_urgent"] is True
        assert f["ml_urgent"] is True
        assert f["kw_only"] is False
        assert f["uncorroborated"] is False


# ── compute_audit: per-source aggregation ───────────────────────────────────
class TestComputeAudit:
    def test_basic_aggregation(self):
        """One source with 3 urgent rows: 2 kw-only-uncorroborated, 1 LLM."""
        rows = [
            ("commodity_futures", 0.0, None),     # kw_only, uncorroborated
            ("commodity_futures", 0.0, None),     # kw_only, uncorroborated
            ("commodity_futures", 9.0, None),     # LLM-confirmed
        ]
        audit = cdua.compute_audit(rows, min_per_source=1)
        assert len(audit) == 1
        r = audit[0]
        assert r["source"] == "commodity_futures"
        assert r["urgent_total"] == 3
        assert r["kw_only"] == 2
        assert r["llm_urgent"] == 1
        assert r["uncorroborated"] == 2
        # uncorroborated / urgent_total: 2/3 ≈ 0.6667
        assert abs(r["uncorroborated_fraction"] - 0.6667) < 0.001

    def test_min_per_source_filter(self):
        rows = [
            ("dxy", 0.0, None),                  # only 1 row
            ("rss", 0.0, None),
            ("rss", 0.0, None),
        ]
        audit = cdua.compute_audit(rows, min_per_source=2)
        sources = {r["source"] for r in audit}
        assert sources == {"rss"}, f"dxy had 1 row, should have been dropped: {sources}"

    def test_sort_order_uncorroborated_desc(self):
        rows = [
            ("a_quiet", 9.0, None),              # 0 uncorroborated
            ("z_loud", 0.0, None),               # 1 uncorroborated
            ("z_loud", 0.0, None),               # 2 uncorroborated
            ("a_quiet", 9.0, None),              # still 0
        ]
        audit = cdua.compute_audit(rows, min_per_source=1)
        assert [r["source"] for r in audit] == ["z_loud", "a_quiet"], (
            "primary sort key must be uncorroborated count descending"
        )

    def test_uncorroborated_fraction_zero_when_all_endorsed(self):
        """A source with ONLY LLM/ML-endorsed urgent rows has uncorroborated=0;
        the fraction is 0.0 (low noise risk)."""
        rows = [
            ("rss", 9.0, None),
            ("rss", 0.0, 9.0),
        ]
        audit = cdua.compute_audit(rows, min_per_source=1)
        r = audit[0]
        assert r["kw_only"] == 0
        assert r["uncorroborated"] == 0
        assert r["uncorroborated_fraction"] == 0.0

    def test_none_source_becomes_unknown(self):
        rows = [(None, 0.0, None), (None, 0.0, None)]
        audit = cdua.compute_audit(rows, min_per_source=1)
        assert audit[0]["source"] == "unknown"


# ── find_suspects: the analyst-actionable verdict ────────────────────────────
class TestFindSuspects:
    def test_high_uncorroborated_source_is_suspect(self):
        audit = [
            {"source": "commodity_futures", "urgent_total": 5,
             "kw_only": 5, "llm_urgent": 0, "ml_urgent": 0,
             "uncorroborated": 5, "uncorroborated_fraction": 1.0},
        ]
        suspects = cdua.find_suspects(audit, min_uncorroborated=3)
        assert len(suspects) == 1
        assert suspects[0]["source"] == "commodity_futures"

    def test_low_uncorroborated_does_not_qualify(self):
        audit = [
            {"source": "dxy", "urgent_total": 2,
             "kw_only": 2, "llm_urgent": 0, "ml_urgent": 0,
             "uncorroborated": 2, "uncorroborated_fraction": 1.0},
        ]
        # 2 < 3 min_uncorroborated
        assert cdua.find_suspects(audit, min_uncorroborated=3) == []

    def test_mostly_endorsed_does_not_qualify(self):
        """Source with 5 urgent rows but 4 LLM-endorsed: only 1 uncorroborated.
        Even with the count above the floor, the fraction (0.2) is well
        below the 0.8 noise threshold — the pipeline IS endorsing this
        collector's calls, so it should NOT be flagged."""
        audit = [
            {"source": "rss", "urgent_total": 5,
             "kw_only": 1, "llm_urgent": 4, "ml_urgent": 0,
             "uncorroborated": 1, "uncorroborated_fraction": 0.2},
        ]
        assert cdua.find_suspects(audit, min_uncorroborated=1) == []


# ── end-to-end: real load_urgent_rows against in-memory SQLite ───────────────
@pytest.fixture
def populated_db(tmp_path):
    """Real articles.db with a mix of direct-write urgent + LLM/ML scored."""
    db_path = tmp_path / "articles.db"
    con = sqlite3.connect(str(db_path))
    con.execute("""CREATE TABLE articles (
        id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL,
        source TEXT, published TEXT, kw_score REAL DEFAULT 0,
        ai_score REAL DEFAULT 0, urgency INTEGER DEFAULT 0,
        full_text BLOB, first_seen TEXT NOT NULL, cycle INTEGER DEFAULT 0,
        time_sensitivity REAL DEFAULT NULL,
        ml_score REAL DEFAULT NULL,
        score_source TEXT DEFAULT NULL
    )""")
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    # Tuple order matches INSERT column list:
    # (id, url, title, source, published, kw_score, ai_score, urgency,
    #  full_text, first_seen, cycle, time_sensitivity, ml_score, score_source)
    rows = [
        # commodity_futures: 3 direct-write urgent, no scoring → suspect
        ("cf1", "internal://c1", "WTI +2.1%", "commodity_futures",
         fresh, 6.0, 0.0, 1, None, fresh, 0, None, None, None),
        ("cf2", "internal://c2", "Brent +2.3%", "commodity_futures",
         fresh, 6.15, 0.0, 1, None, fresh, 0, None, None, None),
        ("cf3", "internal://c3", "Copper +2.0%", "commodity_futures",
         fresh, 6.0, 0.0, 1, None, fresh, 0, None, None, None),
        # rss: 2 LLM-confirmed urgent → not direct, not suspect
        ("r1", "https://reuters.com/a", "Fed cuts 50bp", "rss",
         fresh, 5.0, 9.0, 1, None, fresh, 0, None, None, "llm"),
        ("r2", "https://reuters.com/b", "Earnings beat", "rss",
         fresh, 5.0, 9.5, 1, None, fresh, 0, None, None, "llm"),
        # backtest row: should be EXCLUDED by _LIVE_ONLY_CLAUSE
        ("bt1", "backtest://run_1/x", "Synthetic", "backtest_run_1",
         fresh, 9.0, 9.0, 1, None, fresh, 0, None, None, None),
    ]
    for r in rows:
        con.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "full_text, first_seen, cycle, time_sensitivity, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            r,
        )
    con.commit()
    con.close()
    return db_path


class TestLoadAndAudit:
    def test_load_urgent_rows_excludes_backtest(self, populated_db):
        rows = cdua.load_urgent_rows(populated_db, hours=24)
        sources = [r[0] for r in rows]
        assert "backtest_run_1" not in sources, (
            "backtest row leaked past _LIVE_ONLY_CLAUSE — critical invariant"
        )
        assert sorted(sources) == ["commodity_futures"] * 3 + ["rss"] * 2

    def test_full_run_identifies_commodity_futures_as_suspect(self, populated_db):
        report = cdua.run(populated_db, hours=24, write=False)
        assert report["total_kw_only"] == 3
        assert report["total_uncorroborated"] == 3
        suspect_sources = {s["source"] for s in report["suspects"]}
        assert suspect_sources == {"commodity_futures"}, (
            f"3 unconfirmed commodity_futures direct-writes must be flagged "
            f"suspect; got suspects={suspect_sources}"
        )

    def test_full_run_rss_is_not_suspect(self, populated_db):
        report = cdua.run(populated_db, hours=24, write=False)
        rss = next((s for s in report["by_source"] if s["source"] == "rss"), None)
        assert rss is not None
        assert rss["llm_urgent"] == 2
        assert rss["kw_only"] == 0
        assert rss["uncorroborated"] == 0
        assert rss["source"] not in {s["source"] for s in report["suspects"]}
