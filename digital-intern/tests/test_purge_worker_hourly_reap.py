"""Hourly urgency=1 reap cadence in ``purge_worker`` — between full 6h purges.

Pins the fix for the recurring "aged-but-stuck urgent" failure mode where the
full purge cadence (6h) is the ONLY trigger for ``reap_stale_urgent`` past
worker startup, so a urgency=1 row that crossed the alerter's 24h fetch window
can linger up to ~6 hours past the cutoff — invisible to the alert worker
(push lost) yet still inflating the dashboard ``urgent`` tile and the
``overdue`` count in ``urgent_queue_health``. Live evidence (2026-05-23
16:30Z): 22 of 81 queued urgency=1 rows were >24h old (some 29-30h), never
alerted, awaiting the next purge_old fire.

The fix splits the cadence: the cheap reap (one indexed UPDATE) now fires
hourly on top of the existing 6h purge_old call. These tests do NOT spawn the
real worker loop (which would require a live store + sleep) — they pin the
constants and the per-tick decision logic so a future edit cannot silently
drift the cadence back to the 6h-only failure mode.
"""
from __future__ import annotations

import daemon


class TestUrgentReapCadenceConstant:
    def test_urgent_reap_interval_is_hourly(self):
        """The reap cadence must be ≤1h. 6h (the old behaviour) means up to 6h
        of stuck-urgent lifetime past the 24h cutoff."""
        assert daemon.URGENT_REAP_INTERVAL <= 3600
        # And shorter than the full purge cadence — otherwise the new in-between
        # reap is dominated by purge_old and the split serves no purpose.
        assert daemon.URGENT_REAP_INTERVAL < daemon.PURGE_INTERVAL

    def test_urgent_reap_interval_at_least_5_minutes(self):
        """Reaping more often than the worker's 5-min sleep tick would just
        no-op every wakeup; pin a sane minimum so a future edit doesn't make
        the loop hot."""
        assert daemon.URGENT_REAP_INTERVAL >= 300


class TestPurgeOldCallsReapStaleUrgent:
    """Cross-pin: the 6h purge_old path still wires reap_stale_urgent (the
    pre-existing path), so the hourly cadence is ADDITIVE and the worst case
    is still backstopped by the 6h purge. A regression that drops the
    in-purge_old reap would still be caught by ``test_stale_urgent_reaper``,
    but pinning the wiring here ties the two paths together explicitly."""

    def test_purge_old_demotes_aged_urgency_row(self, store):
        from datetime import datetime, timedelta, timezone
        iso = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " full_text, first_seen, cycle) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("stuck", "http://x/a", "title", "rss", "", 1.0, 9.0, 1, None, iso, 0),
        )
        store.conn.commit()
        store.purge_old()
        urgency = store.conn.execute(
            "SELECT urgency FROM articles WHERE id='stuck'"
        ).fetchone()[0]
        assert urgency == 0
