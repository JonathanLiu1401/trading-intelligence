"""Urgent-backlog aging snapshot — bin queued ``urgency=1`` rows by age.

Pins the analytics module that surfaces *where* in the 24h alerter window
the unalerted urgent backlog is concentrated. The dashboard's existing
``urgent_queue_health`` returns three coarse buckets (queued / near_reap /
overdue); this module bins finer so the analyst can diagnose "alerter is
keeping up" vs "Sonnet went dark ~12h ago".

Each test asserts a specific numeric outcome — not "does not crash".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analytics import urgent_backlog_aging as uba


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _seed(store, *, aid, urgency, first_seen, url=None, source="rss",
          score_source="llm"):
    store.conn.execute(
        "INSERT INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        " full_text, first_seen, cycle, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, url or f"http://x/{aid}", f"title-{aid}", source, "",
         1.0, 9.0, urgency, None, first_seen, 0, score_source),
    )
    store.conn.commit()


class TestBucketizePure:
    """The bin logic is a pure function — exercise it directly without a store."""

    def test_empty_input_emits_zero_filled_buckets(self):
        buckets, overdue = uba._bucketize([], bucket_h=4.0)
        # Six 4h buckets across the 24h window.
        assert len(buckets) == 6
        assert all(b["count"] == 0 for b in buckets)
        assert overdue == 0

    def test_bucket_edges_are_correct(self):
        buckets, _ = uba._bucketize([], bucket_h=4.0)
        assert [b["start_h"] for b in buckets] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
        assert [b["end_h"] for b in buckets] == [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]

    def test_fresh_row_lands_in_first_bucket(self):
        buckets, overdue = uba._bucketize([0.5, 1.0, 3.9], bucket_h=4.0)
        assert buckets[0]["count"] == 3
        assert sum(b["count"] for b in buckets[1:]) == 0
        assert overdue == 0

    def test_boundary_row_lands_in_next_bucket(self):
        # Exactly 4h → start of bucket 1, not last of bucket 0 (idx = 4 // 4 = 1).
        buckets, _ = uba._bucketize([4.0], bucket_h=4.0)
        assert buckets[0]["count"] == 0
        assert buckets[1]["count"] == 1

    def test_overdue_separates_from_in_window(self):
        buckets, overdue = uba._bucketize([23.9, 24.0, 30.0, 50.0], bucket_h=4.0)
        # 23.9h is still in-window (last bucket); 24.0+ is overdue.
        assert buckets[-1]["count"] == 1
        assert overdue == 3

    def test_negative_age_is_dropped(self):
        # A future-dated first_seen produces a negative age — defensive drop.
        buckets, overdue = uba._bucketize([-0.5, 1.0], bucket_h=4.0)
        assert sum(b["count"] for b in buckets) == 1
        assert overdue == 0

    def test_non_divisible_bucket_size(self):
        # 5h bucket on a 24h window → 5 buckets (5,5,5,5,4) but the
        # implementation just uses ceil so 5 buckets of nominal 5h each.
        buckets, _ = uba._bucketize([], bucket_h=5.0)
        # ceil(24 / 5) == 5
        assert len(buckets) == 5
        assert buckets[-1]["end_h"] == 24.0  # final bucket clipped at reap age

    def test_bucket_h_must_be_positive(self):
        import pytest
        with pytest.raises(ValueError):
            uba._bucketize([], bucket_h=0)


class TestVerdict:
    def test_overdue_loss_wins_over_everything(self):
        # Even with no queued rows the OVERDUE flag would never fire because
        # overdue is counted in queued. But once even one row is overdue, the
        # verdict is OVERDUE_LOSS regardless of how fresh the rest are.
        assert uba._verdict(queued=10, overdue=1, stuck_old_fraction=0.0) == "OVERDUE_LOSS"

    def test_empty_when_no_queue(self):
        assert uba._verdict(queued=0, overdue=0, stuck_old_fraction=0.0) == "EMPTY"

    def test_stuck_old_at_threshold(self):
        # Exactly STUCK_OLD_FRACTION counts as STUCK_OLD (inclusive).
        v = uba._verdict(queued=10, overdue=0,
                         stuck_old_fraction=uba.STUCK_OLD_FRACTION)
        assert v == "STUCK_OLD"

    def test_healthy_when_concentrated_fresh(self):
        v = uba._verdict(queued=10, overdue=0, stuck_old_fraction=0.1)
        assert v == "HEALTHY"


class TestAuditOnStore:
    """End-to-end: seed live rows into the in-memory store and verify the
    counts + verdict the audit reports."""

    def test_empty_store_reports_empty(self, store):
        report = uba.audit(store)
        assert report["queued"] == 0
        assert report["overdue"] == 0
        assert report["in_window"] == 0
        assert report["oldest_age_h"] is None
        assert report["median_age_h"] is None
        assert report["verdict"] == "EMPTY"

    def test_fresh_queued_row_lands_in_first_bucket(self, store):
        _seed(store, aid="fresh", urgency=1, first_seen=_iso(0.5))
        report = uba.audit(store, bucket_h=4.0)
        assert report["queued"] == 1
        assert report["overdue"] == 0
        assert report["buckets"][0]["count"] == 1
        assert sum(b["count"] for b in report["buckets"][1:]) == 0
        assert report["verdict"] == "HEALTHY"

    def test_overdue_row_drives_overdue_loss_verdict(self, store):
        _seed(store, aid="old", urgency=1, first_seen=_iso(30))
        _seed(store, aid="fresh", urgency=1, first_seen=_iso(1))
        report = uba.audit(store, bucket_h=4.0)
        assert report["queued"] == 2
        assert report["overdue"] == 1
        assert report["in_window"] == 1
        # Even ONE overdue row → OVERDUE_LOSS (silent missed push).
        assert report["verdict"] == "OVERDUE_LOSS"

    def test_stuck_old_verdict_when_mass_is_late_in_window(self, store):
        # 7 of 10 rows in the 12-24h band → stuck_old_fraction=0.7 → STUCK_OLD.
        for i in range(7):
            _seed(store, aid=f"old{i}", urgency=1, first_seen=_iso(18))
        for i in range(3):
            _seed(store, aid=f"fresh{i}", urgency=1, first_seen=_iso(1))
        report = uba.audit(store, bucket_h=4.0)
        assert report["queued"] == 10
        assert report["overdue"] == 0
        assert report["stuck_old_count"] == 7
        assert report["stuck_old_fraction"] == 0.7
        assert report["verdict"] == "STUCK_OLD"

    def test_alerted_row_is_not_counted(self, store):
        # urgency=2 means already pushed — never in the queued backlog.
        _seed(store, aid="done", urgency=2, first_seen=_iso(5))
        _seed(store, aid="queued", urgency=1, first_seen=_iso(5))
        report = uba.audit(store)
        assert report["queued"] == 1
        assert report["oldest_age_h"] is not None and 4.9 <= report["oldest_age_h"] <= 5.1

    def test_synthetic_backtest_row_is_filtered(self, store):
        # _LIVE_ONLY_CLAUSE excludes backtest:// URLs even on the audit path.
        # (Synthetic rows are urgency=0 by construction so this should not
        # happen anyway — the filter is defense-in-depth.)
        _seed(store, aid="bt", urgency=1, first_seen=_iso(5),
              url="backtest://run_1/2026-05-13/BUY/MU",
              source="backtest_run_1", score_source=None)
        _seed(store, aid="live", urgency=1, first_seen=_iso(5))
        report = uba.audit(store)
        assert report["queued"] == 1  # only the live one

    def test_opus_annotation_row_is_filtered(self, store):
        _seed(store, aid="opus", urgency=1, first_seen=_iso(5),
              source="opus_annotation_cycle_3", score_source=None)
        _seed(store, aid="live", urgency=1, first_seen=_iso(5))
        report = uba.audit(store)
        assert report["queued"] == 1

    def test_oldest_and_median_age_for_known_set(self, store):
        for i, h in enumerate([1, 2, 5, 5, 9]):
            _seed(store, aid=f"r{i}", urgency=1, first_seen=_iso(h))
        report = uba.audit(store)
        # Median of [1, 2, 5, 5, 9] = 5; max = 9.
        assert report["median_age_h"] == 5.0
        assert 8.9 <= report["oldest_age_h"] <= 9.1

    def test_audit_does_not_write_anything(self, store):
        # The four load-bearing invariants: audit is a pure READ.
        _seed(store, aid="r", urgency=1, first_seen=_iso(5))
        before = store.conn.execute(
            "SELECT urgency, ai_score, ml_score, score_source FROM articles "
            "WHERE id='r'"
        ).fetchone()
        uba.audit(store)
        after = store.conn.execute(
            "SELECT urgency, ai_score, ml_score, score_source FROM articles "
            "WHERE id='r'"
        ).fetchone()
        assert before == after


class TestStrictCliExitCode:
    """The --strict CLI contract: non-zero exit on STUCK_OLD / OVERDUE_LOSS so
    CI gates and cron alarms can wire on it."""

    def test_strict_exits_zero_on_healthy(self, store, capsys):
        _seed(store, aid="fresh", urgency=1, first_seen=_iso(1))
        # Run the audit + verdict directly (not through main, which opens its
        # own store). Replicate the CLI's exit-code logic.
        report = uba.audit(store)
        exit_code = 1 if report["verdict"] in ("STUCK_OLD", "OVERDUE_LOSS") else 0
        assert exit_code == 0

    def test_strict_exits_one_on_overdue(self, store):
        _seed(store, aid="old", urgency=1, first_seen=_iso(30))
        report = uba.audit(store)
        exit_code = 1 if report["verdict"] in ("STUCK_OLD", "OVERDUE_LOSS") else 0
        assert exit_code == 1
