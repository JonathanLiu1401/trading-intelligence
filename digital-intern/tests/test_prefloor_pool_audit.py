"""Pre-floor pool audit — assertion-pinned behavior + invariant.

The audit's purpose is to surface accumulated noise in the trainer's
strong-label pool from the urgency_scorer's 0.01 pre-filter floor (quote-
widget, recap-template, and Sonnet-omitted indices). The model trains on
``ai_score > 0`` rows (``STRONG_LABEL_WHERE``), so an unmonitored spike in
0.01 prefloor share is a silent training-signal collapse.

These tests pin:

  1. The canonical prefloor predicate (``score_source='llm'`` AND
     ``ai_score = 0.01``) actually maps to the prefloored rows — and
     never to real LLM-graded labels (Sonnet's integer scores,
     briefing_boost's 4.5, etc.).
  2. The verdict thresholds fire at the documented breakpoints.
  3. Backtest / opus-annotation rows can NEVER inflate the figure — the
     load-bearing CLAUDE.md §5 invariant.
  4. Per-source noise contribution is correctly attributed to the row's
     ``source`` column (not its url-host or anything else).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import prefloor_pool_audit as ppa


def _insert_raw(store, *, id, url, title, source, ai_score, score_source,
                kw_score=1.0, first_seen=None, urgency=0, ml_score=None):
    if first_seen is None:
        first_seen = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


class TestPrefloorCanonicalSignature:
    """``score_source='llm' AND ai_score = 0.01`` is the prefloor predicate.
    Real LLM labels (Sonnet integer 0..10, briefing_boost 4.5) must never
    be counted. A model self-prediction (score_source='ml') must never be
    counted regardless of its ml_score / ai_score."""

    def test_prefloor_noise_row_counted(self, store):
        _insert_raw(store, id="pf", url="https://x.com/1",
                    title="Why MU Stock Is Trading Up Today", source="rss",
                    ai_score=0.01, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 1
        assert r["real_llm_total"] == 0
        assert r["window_prefloor"] == 1

    def test_real_llm_label_excluded_from_prefloor(self, store):
        """Sonnet integer 5.0 must NOT be counted as prefloor noise."""
        _insert_raw(store, id="real", url="https://x.com/2",
                    title="MU beats estimates", source="rss",
                    ai_score=5.0, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 0
        assert r["real_llm_total"] == 1
        assert r["window_prefloor"] == 0
        assert r["window_real_llm"] == 1

    def test_briefing_boost_excluded_from_prefloor(self, store):
        """4.5 from briefing curation is NOT prefloor — different
        score_source tag too, but assert the value-based predicate alone
        is enough."""
        _insert_raw(store, id="bb", url="https://x.com/3",
                    title="Opus-curated NVDA earnings story", source="rss",
                    ai_score=4.5, score_source="briefing_boost")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 0
        assert r["real_llm_total"] == 0

    def test_ml_self_prediction_at_001_not_counted(self, store):
        """score_source='ml' with ai_score=0.01 (theoretically impossible
        but defense-in-depth) must NEVER be counted — the audit pertains
        to the LLM label pool, not model predictions."""
        _insert_raw(store, id="m", url="https://x.com/4",
                    title="Generic title", source="rss",
                    ai_score=0.0, ml_score=0.01, score_source="ml")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 0


class TestBacktestIsolation:
    """Load-bearing invariant: synthetic backtest / opus-annotation rows
    must NEVER inflate the audit. Mirrors the ``_LIVE_ONLY_CLAUSE``
    discipline carried by every analytics audit in this family."""

    def test_backtest_url_row_excluded(self, store):
        _insert_raw(store, id="bt1", url="backtest://run/1/BUY/MU",
                    title="Synthetic", source="backtest_run_1",
                    ai_score=0.01, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 0, (
            "backtest row leaked into prefloor count — _LIVE_ONLY_CLAUSE missing"
        )

    def test_opus_annotation_source_excluded(self, store):
        _insert_raw(store, id="op", url="https://x.com/5",
                    title="Annotation", source="opus_annotation_cycle_3",
                    ai_score=0.01, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 0


class TestVerdictBreakpoints:
    """Verdict thresholds fire at the documented values:
    < 70% → HEALTHY, 70-85% → ELEVATED, ≥ 85% → CONTAMINATED."""

    def test_healthy_below_70_pct(self, store):
        # 6 prefloor + 4 real = 60% prefloor share
        for i in range(6):
            _insert_raw(store, id=f"pf{i}", url=f"https://x.com/p{i}",
                        title="t", source="rss",
                        ai_score=0.01, score_source="llm")
        for i in range(4):
            _insert_raw(store, id=f"r{i}", url=f"https://x.com/r{i}",
                        title="t", source="rss",
                        ai_score=5.0, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["window_prefloor_share"] == pytest.approx(0.6)
        assert r["verdict"] == "HEALTHY"

    def test_elevated_at_75_pct(self, store):
        # 9 prefloor + 3 real = 75% prefloor share
        for i in range(9):
            _insert_raw(store, id=f"pf{i}", url=f"https://x.com/p{i}",
                        title="t", source="rss",
                        ai_score=0.01, score_source="llm")
        for i in range(3):
            _insert_raw(store, id=f"r{i}", url=f"https://x.com/r{i}",
                        title="t", source="rss",
                        ai_score=5.0, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["window_prefloor_share"] == pytest.approx(0.75)
        assert r["verdict"] == "ELEVATED"

    def test_contaminated_at_or_above_85_pct(self, store):
        # 17 prefloor + 3 real = 85% prefloor share
        for i in range(17):
            _insert_raw(store, id=f"pf{i}", url=f"https://x.com/p{i}",
                        title="t", source="rss",
                        ai_score=0.01, score_source="llm")
        for i in range(3):
            _insert_raw(store, id=f"r{i}", url=f"https://x.com/r{i}",
                        title="t", source="rss",
                        ai_score=5.0, score_source="llm")
        r = ppa.audit(store, window_hours=24)
        assert r["window_prefloor_share"] == pytest.approx(0.85)
        assert r["verdict"] == "CONTAMINATED"

    def test_empty_pool_is_healthy_not_error(self, store):
        """An empty store must return HEALTHY (0.0 share), not crash."""
        r = ppa.audit(store, window_hours=24)
        assert r["window_prefloor"] == 0
        assert r["window_real_llm"] == 0
        assert r["window_prefloor_share"] == 0.0
        assert r["verdict"] == "HEALTHY"


class TestPerSourceAttribution:
    """The top-N per-source breakdown surfaces who is generating noise so
    the analyst can decide where to tighten gates."""

    def test_top_sources_sorted_by_prefloor_count_desc(self, store):
        # Insert: 5 from sourceA, 3 from sourceB, 1 from sourceC
        for src, n in (("sourceA", 5), ("sourceB", 3), ("sourceC", 1)):
            for i in range(n):
                _insert_raw(store, id=f"{src}{i}",
                            url=f"https://x.com/{src}{i}",
                            title="t", source=src,
                            ai_score=0.01, score_source="llm")
        r = ppa.audit(store, window_hours=24, top_sources=10)
        per_src = r["per_source_top"]
        assert [row["source"] for row in per_src] == ["sourceA", "sourceB", "sourceC"]
        assert [row["prefloor_count"] for row in per_src] == [5, 3, 1]

    def test_top_sources_respects_limit(self, store):
        for src_n in range(5):
            _insert_raw(store, id=f"s{src_n}", url=f"https://x.com/s{src_n}",
                        title="t", source=f"src{src_n}",
                        ai_score=0.01, score_source="llm")
        r = ppa.audit(store, window_hours=24, top_sources=3)
        assert len(r["per_source_top"]) == 3


class TestWindowRestriction:
    """``window_hours`` cleanly restricts to first_seen within that window —
    older rows count toward lifetime totals but not window-restricted ones."""

    def test_old_prefloor_excluded_from_window(self, store):
        # Old (40h ago) + new (5min ago)
        old = (datetime.now(timezone.utc) - timedelta(hours=40)).isoformat()
        _insert_raw(store, id="old", url="https://x.com/old", title="t",
                    source="rss", ai_score=0.01, score_source="llm",
                    first_seen=old)
        _insert_raw(store, id="new", url="https://x.com/new", title="t",
                    source="rss", ai_score=0.01, score_source="llm")

        r = ppa.audit(store, window_hours=24)
        assert r["prefloor_total"] == 2          # lifetime counts both
        assert r["window_prefloor"] == 1         # window excludes old
        # per_source_top is window-restricted too
        assert sum(row["prefloor_count"] for row in r["per_source_top"]) == 1


class TestReadOnlyInvariant:
    """The audit must be 100% read-only — must not mutate any row."""

    def test_audit_does_not_mutate_articles(self, store):
        _insert_raw(store, id="x", url="https://x.com/1",
                    title="Some title here", source="rss",
                    ai_score=0.01, score_source="llm")
        # Snapshot every column we care about
        before = store.conn.execute(
            "SELECT id, url, title, source, ai_score, score_source, urgency, "
            "ml_score, kw_score FROM articles WHERE id='x'"
        ).fetchone()
        ppa.audit(store, window_hours=24)
        after = store.conn.execute(
            "SELECT id, url, title, source, ai_score, score_source, urgency, "
            "ml_score, kw_score FROM articles WHERE id='x'"
        ).fetchone()
        assert before == after, "audit mutated a row — must be read-only"
