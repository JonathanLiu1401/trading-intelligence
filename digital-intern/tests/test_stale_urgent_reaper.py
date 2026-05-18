"""Stale-urgent reaper invariant — ``ArticleStore.reap_stale_urgent`` and its
``purge_old`` wiring.

Live evidence (2026-05-18): 26 rows stuck at ``urgency=1`` since 2026-05-13 (5
days). ``get_unalerted_urgent`` only returns ``first_seen >= now-24h`` rows, so
once a still-pending urgent row ages past that window it is permanently
invisible to the alert worker — never alerted, never cleared, forever inflating
``stats()``'s ``urgent>=1`` tile. ``reap_stale_urgent`` demotes those aged-out
rows to ``urgency=0`` (the only state that is both honest — no analyst was ever
pushed — and corrective for the phantom-count bug).

These tests assert specific row states, not "no crash". They also pin the
load-bearing invariant that the reaper writes ONLY ``urgency`` (never
ai_score / ml_score / score_source) and never touches a synthetic row.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _seed(store, *, aid, urgency, first_seen, url="http://x/a",
          source="rss", ai_score=9.0, ml_score=None, score_source="llm"):
    store.conn.execute(
        "INSERT INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        " full_text, first_seen, cycle, time_sensitivity, ml_score, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, url, f"title-{aid}", source, "", 1.0, ai_score, urgency,
         None, first_seen, 0, None, ml_score, score_source),
    )
    store.conn.commit()


def _urgency(store, aid) -> int:
    return store.conn.execute(
        "SELECT urgency FROM articles WHERE id=?", (aid,)
    ).fetchone()[0]


class TestReapStaleUrgent:
    def test_aged_out_urgent_row_is_demoted_to_zero(self, store):
        _seed(store, aid="old", urgency=1, first_seen=_iso(48))
        n = store.reap_stale_urgent()
        assert n == 1
        assert _urgency(store, "old") == 0

    def test_in_window_urgent_row_is_left_alone(self, store):
        # 2h old — still alertable; the reaper must not touch it.
        _seed(store, aid="fresh", urgency=1, first_seen=_iso(2))
        assert store.reap_stale_urgent() == 0
        assert _urgency(store, "fresh") == 1

    def test_alerted_row_is_never_un_alerted(self, store):
        # urgency=2 aged out is the CORRECT end-state — must never regress.
        _seed(store, aid="alerted", urgency=2, first_seen=_iso(72))
        assert store.reap_stale_urgent() == 0
        assert _urgency(store, "alerted") == 2

    def test_normal_row_untouched(self, store):
        _seed(store, aid="normal", urgency=0, first_seen=_iso(72))
        assert store.reap_stale_urgent() == 0
        assert _urgency(store, "normal") == 0

    def test_only_urgency_is_written_scores_unchanged(self, store):
        _seed(store, aid="pin", urgency=1, first_seen=_iso(100),
              ai_score=8.5, ml_score=3.14, score_source="llm")
        store.reap_stale_urgent()
        row = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency "
            "FROM articles WHERE id=?", ("pin",)
        ).fetchone()
        # ai_score / ml_score / score_source byte-for-byte unchanged.
        assert row[0] == 8.5
        assert row[1] == 3.14
        assert row[2] == "llm"
        assert row[3] == 0  # only urgency moved

    def test_idempotent(self, store):
        _seed(store, aid="x", urgency=1, first_seen=_iso(48))
        assert store.reap_stale_urgent() == 1
        # Second pass: nothing left to reap.
        assert store.reap_stale_urgent() == 0
        assert _urgency(store, "x") == 0

    def test_synthetic_backtest_urgent_row_is_not_reaped(self, store):
        # Synthetic rows are urgency=0 by construction (invariant), so this
        # can't happen in production — but the _LIVE_ONLY_CLAUSE defense-in-
        # depth must keep the reaper off any training row even if it did.
        _seed(store, aid="bt", urgency=1, first_seen=_iso(99),
              url="backtest://run_1/2026-05-13/BUY/MU", source="backtest_run_1",
              score_source=None)
        assert store.reap_stale_urgent() == 0
        assert _urgency(store, "bt") == 1  # untouched — not a live row

    def test_custom_max_age_window(self, store):
        # A 10h-old row: untouched at the default 24h window, reaped at 6h.
        _seed(store, aid="mid", urgency=1, first_seen=_iso(10))
        assert store.reap_stale_urgent(max_age_hours=24) == 0
        assert _urgency(store, "mid") == 1
        assert store.reap_stale_urgent(max_age_hours=6) == 1
        assert _urgency(store, "mid") == 0

    def test_reaped_row_was_already_unreachable_by_alert_worker(self, store):
        """Ties the fix to the actual alert-path behaviour: the row the reaper
        clears is one ``get_unalerted_urgent`` already refuses to return, and
        the in-window one it keeps is one that path still serves — so demotion
        provably loses zero alert delivery."""
        _seed(store, aid="aged", urgency=1, first_seen=_iso(48))
        _seed(store, aid="live", urgency=1, first_seen=_iso(1))
        served_before = {r["_id"] for r in store.get_unalerted_urgent()}
        assert served_before == {"live"}  # aged-out one already invisible
        store.reap_stale_urgent()
        assert _urgency(store, "aged") == 0
        assert _urgency(store, "live") == 1
        served_after = {r["_id"] for r in store.get_unalerted_urgent()}
        assert served_after == {"live"}  # no delivery lost


class TestPurgeOldWiring:
    def test_purge_old_reaps_stale_urgent_and_deletes_old_rows(self, store):
        # An aged-out urgent row (5 days) and a >90d row that purge deletes.
        _seed(store, aid="stale_urg", urgency=1, first_seen=_iso(120))
        _seed(store, aid="ancient", urgency=0, first_seen=_iso(91 * 24))
        _seed(store, aid="keep", urgency=0, first_seen=_iso(1))
        deleted = store.purge_old()
        # Old row deleted; stale urgent demoted (not deleted — only 5d old);
        # fresh row untouched.
        assert deleted == 1
        assert store.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE id=?", ("ancient",)
        ).fetchone()[0] == 0
        assert _urgency(store, "stale_urg") == 0
        assert _urgency(store, "keep") == 0
        # purge_old still returns the delete count (unchanged contract).
        assert isinstance(deleted, int)
