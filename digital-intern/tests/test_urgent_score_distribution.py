"""Score-magnitude histogram of urgent rows — the missing calibration-axis
sibling to ``urgency_label_split`` (which counts by source-tag).

Pins:
  * the four-bucket layout actually buckets the COALESCED score
    (ai_score preferred over ml_score) so the histogram aligns with what the
    alerter / briefing reader actually saw;
  * the verdict ladder (BORDERLINE_HEAVY when >70% of urgent rows are at the
    8.0 threshold; the "over-confident urgency head" failure mode);
  * the per-bucket score_source split (so the analyst can read
    "80 of 87 borderline-8 rows are ML-only" — the most diagnostic single
    view of the unverified-rate problem);
  * the load-bearing invariants: ``_LIVE_ONLY_CLAUSE`` excludes
    backtest/opus rows; NO DB write (no ai_score/ml_score/score_source/
    urgency mutation).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _insert(store, *, id, score, score_source="llm", urgency=1,
            ai_score=None, ml_score=None, first_seen=None, source="rss",
            url=None):
    """Insert an urgent row with the unified score expressed via the
    canonical ai_score / ml_score split:

      * score_source='llm' → ai_score=score, ml_score=NULL
      * score_source='ml'  → ai_score=0,    ml_score=score
      * score_source='briefing_boost' → ai_score=score, ml_score=NULL
      * score_source=None  → ai_score=score, ml_score=NULL (legacy)
    """
    if first_seen is None:
        first_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
    if score_source == "ml":
        ai = 0.0 if ai_score is None else ai_score
        ml = score if ml_score is None else ml_score
    else:
        ai = score if ai_score is None else ai_score
        ml = ml_score
    if url is None:
        url = f"https://x.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, f"title {id}", source, "", 1.0, ai, urgency,
             first_seen, 0, ml, score_source),
        )
        store.conn.commit()


class TestEmptyDB:
    def test_no_urgent_rows_returns_no_data(self, store):
        out = store.urgent_score_distribution(hours=24)
        assert out["total"] == 0
        assert out["verdict"] == "NO_DATA"
        assert out["borderline_fraction"] == 0.0
        assert out["strong_fraction"] == 0.0
        # All five buckets emitted even with no rows — fixed-key contract.
        assert len(out["buckets"]) == 5


class TestBucketing:
    def test_score_8_lands_in_borderline_bucket(self, store):
        _insert(store, id="a", score=8.0, score_source="llm")
        out = store.urgent_score_distribution(hours=24)
        # Bucket index 3 is [8, 9); index 4 is [9, 10].
        assert out["buckets"][3]["count"] == 1
        assert out["buckets"][3]["by_source"]["llm"] == 1
        assert out["buckets"][4]["count"] == 0

    def test_score_9_lands_in_strong_bucket(self, store):
        _insert(store, id="a", score=9.0, score_source="llm")
        out = store.urgent_score_distribution(hours=24)
        assert out["buckets"][3]["count"] == 0
        assert out["buckets"][4]["count"] == 1

    def test_score_10_is_inclusive_in_strong_bucket(self, store):
        """Final bucket includes 10.0 inclusive — a max-clamped score
        must not silently fall off the right edge of the histogram."""
        _insert(store, id="a", score=10.0, score_source="llm")
        out = store.urgent_score_distribution(hours=24)
        assert out["buckets"][4]["count"] == 1
        assert out["total"] == 1


class TestUnifiedScoreCoalesce:
    def test_ml_only_row_buckets_by_ml_score(self, store):
        """A row with ai_score=0 + ml_score=9.5 (ML-only urgent) must
        bucket by ml_score — that's the score the alerter actually saw
        (COALESCE(NULLIF(ai_score,0), ml_score, 0))."""
        _insert(store, id="a", score=9.5, score_source="ml")
        out = store.urgent_score_distribution(hours=24)
        assert out["buckets"][4]["count"] == 1, "ml_score must populate histogram"
        assert out["buckets"][4]["by_source"]["ml"] == 1

    def test_ai_score_preferred_over_ml_score(self, store):
        """When BOTH columns are populated (LLM ground truth + a prior ML
        prediction), ai_score wins — same precedence as get_unalerted_urgent."""
        _insert(store, id="a", score=0,  # ignored; we set explicit columns:
                score_source="llm", ai_score=8.5, ml_score=9.9)
        out = store.urgent_score_distribution(hours=24)
        # 8.5 → borderline bucket. If ml_score wrongly won, 9.9 would land
        # in the strong bucket — explicit asymmetric values catch the bug.
        assert out["buckets"][3]["count"] == 1
        assert out["buckets"][4]["count"] == 0


class TestVerdictLadder:
    def test_borderline_heavy_when_over_70_pct_at_threshold(self, store):
        # 8 rows at 8.0 (borderline), 2 rows at 9.5 (strong). 80% borderline.
        for i in range(8):
            _insert(store, id=f"b{i}", score=8.1, score_source="ml")
        for i in range(2):
            _insert(store, id=f"s{i}", score=9.5, score_source="ml")
        out = store.urgent_score_distribution(hours=24)
        assert out["verdict"] == "BORDERLINE_HEAVY"
        assert out["borderline_fraction"] == 0.8

    def test_mixed_when_40_to_70_pct_at_threshold(self, store):
        # 5 borderline, 5 strong = 50%.
        for i in range(5):
            _insert(store, id=f"b{i}", score=8.0, score_source="ml")
        for i in range(5):
            _insert(store, id=f"s{i}", score=9.5, score_source="ml")
        out = store.urgent_score_distribution(hours=24)
        assert out["verdict"] == "MIXED"
        assert out["borderline_fraction"] == 0.5

    def test_well_calibrated_when_most_are_strong(self, store):
        # 1 borderline, 9 strong = 10% borderline.
        _insert(store, id="b0", score=8.0, score_source="ml")
        for i in range(9):
            _insert(store, id=f"s{i}", score=9.5, score_source="ml")
        out = store.urgent_score_distribution(hours=24)
        assert out["verdict"] == "WELL_CALIBRATED"


class TestScoreSourceSplit:
    def test_per_bucket_split_separates_llm_vs_ml(self, store):
        """The analyst's most diagnostic view: how many of the borderline-8
        rows are ML-only vs LLM-vetted? The split must populate inside
        each bucket, not just at the aggregate level."""
        _insert(store, id="llm1", score=8.2, score_source="llm")
        _insert(store, id="llm2", score=8.5, score_source="llm")
        _insert(store, id="ml1", score=8.1, score_source="ml")
        _insert(store, id="ml2", score=8.7, score_source="ml")
        _insert(store, id="ml3", score=8.3, score_source="ml")
        out = store.urgent_score_distribution(hours=24)
        bucket = out["buckets"][3]  # [8, 9)
        assert bucket["count"] == 5
        assert bucket["by_source"]["llm"] == 2
        assert bucket["by_source"]["ml"] == 3
        assert bucket["by_source"]["briefing_boost"] == 0
        assert bucket["by_source"]["null"] == 0

    def test_briefing_boost_counted_separately(self, store):
        _insert(store, id="bb", score=8.5, score_source="briefing_boost")
        out = store.urgent_score_distribution(hours=24)
        assert out["buckets"][3]["by_source"]["briefing_boost"] == 1


class TestBacktestIsolation:
    def test_backtest_urls_excluded(self, store):
        """The load-bearing invariant: backtest:// URLs and backtest_* /
        opus_annotation* sources must NEVER inflate the histogram —
        otherwise an injection burst could mask a real calibration issue."""
        _insert(store, id="live", score=9.0, score_source="llm",
                source="rss", url="https://reuters.com/x")
        _insert(store, id="bt", score=8.0, score_source="llm",
                source="backtest_run_1",
                url="backtest://run_1/2026-01-01/BUY/MU")
        _insert(store, id="opus", score=8.0, score_source="llm",
                source="opus_annotation_cycle_3",
                url="https://x.com/opus")
        _insert(store, id="btsrc", score=8.0, score_source="llm",
                source="backtest_run_42_winner",
                url="https://x.com/btsrc")
        out = store.urgent_score_distribution(hours=24)
        # Only the live row should appear.
        assert out["total"] == 1
        assert out["buckets"][4]["count"] == 1  # the score=9.0 row


class TestWindowBoundary:
    def test_rows_outside_window_excluded(self, store):
        """A row older than ``hours`` ago must not appear in the histogram."""
        in_window = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        outside = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).isoformat()
        _insert(store, id="in", score=9.0, score_source="llm",
                first_seen=in_window)
        _insert(store, id="old", score=9.0, score_source="llm",
                first_seen=outside)
        out = store.urgent_score_distribution(hours=24)
        assert out["total"] == 1

    def test_window_h_clamped_to_one(self, store):
        """A caller passing 0 / negative ``hours`` must not produce
        SQL "since=future" semantics; the impl clamps to >=1."""
        out = store.urgent_score_distribution(hours=0)
        # Doesn't raise, returns a structured dict (with hours=1).
        assert out["window_h"] == 1


class TestUrgencyState:
    def test_urgency_zero_rows_excluded(self, store):
        """Histogram is over urgency>=1 — never the un-flagged backlog."""
        _insert(store, id="urg", score=9.0, score_source="llm", urgency=1)
        _insert(store, id="non", score=9.0, score_source="llm", urgency=0)
        out = store.urgent_score_distribution(hours=24)
        assert out["total"] == 1

    def test_urgency_two_also_counted(self, store):
        """Both queued (urgency=1) and alerted (urgency=2) rows count —
        same surface as urgency_label_split."""
        _insert(store, id="q", score=8.5, score_source="llm", urgency=1)
        _insert(store, id="a", score=9.5, score_source="ml", urgency=2)
        out = store.urgent_score_distribution(hours=24)
        assert out["total"] == 2
