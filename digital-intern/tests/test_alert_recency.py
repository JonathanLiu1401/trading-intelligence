"""Cross-cycle (cross-time) syndication suppression on the urgent-alert path.

``watchers/alert_dedup.dedupe_urgent`` only collapses syndicated copies present
in the *same* ``get_unalerted_urgent()`` batch. Once a story is alerted it goes
``urgency=2`` and is excluded from every future batch — so a slower feed that
re-collects the **same event** as a NEW row (``urgency=1``) had nothing to be
deduped against and fired a SECOND standalone "🚨 BREAKING" push for an event
the analyst was already told about (observed live: the "US clears/approves H200
chip sales to 10 China firms" story alerted twice ~1.5h apart from different
sources).

``watchers/alert_recency`` closes that gap: it records the canonical signature
of every story that actually fired and suppresses a later urgent row whose
signature was alerted within ``ALERT_RECENCY_TTL_HOURS``.

Contract pinned here (assert specific behaviour, not no-crash):

  * ``partition_already_alerted`` is a pure split — a row whose canonical
    signature is in the recent set is suppressed; one whose signature is not
    is kept; an untitled row (empty signature) is NEVER suppressed; an empty
    recent set is a no-op (everything kept).
  * The signature is ``alert_dedup._signature`` verbatim (single source of
    truth) — a wire-prefixed / source-attributed repost of an already-alerted
    bare headline collapses to the same signature and is suppressed.
  * DB round-trip + TTL: a recorded signature is returned inside the window
    and NOT after it; old rows are pruned.
  * Best-effort: a recency-store open failure degrades to the pre-feature
    behaviour (empty set / record skipped) and never raises into the alert
    path.
  * End-to-end through ``send_urgent_alert``: the first fire records; a second
    cycle carrying a NEW-id row with the same headline is cross-suppressed
    (no Claude/Discord call, marked ``urgency=2`` so it exits the queue,
    returns False), while a genuinely distinct headline still fires.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent, alert_recency
from watchers.alert_dedup import _signature


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, url=None,
                    title="MU earnings blow past Q3 estimates sharply",
                    source="rss", ai_score=9.0, published=None,
                    first_seen=None):
    """Insert one live urgency=1 row exactly as the scoring path leaves it."""
    if url is None:
        url = f"https://example.com/{id}"
    if published is None:
        published = _iso(1)
    if first_seen is None:
        first_seen = _iso(0.08)
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published, 1.0, ai_score, 1,
             first_seen, 0, None, "llm", None),
        )
        store.conn.commit()


def _urgency_of(store, aid):
    return store.conn.execute(
        "SELECT urgency FROM articles WHERE id=?", (aid,)
    ).fetchone()[0]


@pytest.fixture
def recency_db(tmp_path, monkeypatch):
    """Redirect alert_recency to a per-test SQLite file (never the real one)."""
    db = tmp_path / "alert_recency.db"
    monkeypatch.setattr(alert_recency, "DB_PATH", db)
    return db


# ── pure helper: partition_already_alerted ──────────────────────────────────
class TestPartition:
    def test_in_recent_suppressed_not_in_recent_kept(self):
        a = {"_id": "1", "title": "Micron shares surge after Q3 earnings blowout"}
        b = {"_id": "2", "title": "Fed holds rates steady amid inflation"}
        recent = {_signature(a["title"])}
        kept, suppressed = alert_recency.partition_already_alerted([a, b], recent)
        assert kept == [b]
        assert suppressed == [a]

    def test_untitled_row_never_suppressed(self):
        # An empty signature must never match even if "" is in the set.
        a = {"_id": "1", "title": None}
        kept, suppressed = alert_recency.partition_already_alerted([a], {""})
        assert kept == [a] and suppressed == []

    def test_empty_recent_is_noop(self):
        rows = [{"_id": "1", "title": "Anything at all here"}]
        kept, suppressed = alert_recency.partition_already_alerted(rows, set())
        assert kept == rows and suppressed == []

    def test_signature_is_canonical_wire_repost_collapses(self):
        # The bare headline was alerted; a wire-prefixed + source-attributed
        # repost of it (a NEW row from a slower feed) must collapse to the
        # SAME signature and be suppressed — proves _signature reuse.
        bare = "Nvidia clears H200 chip sales to ten China firms"
        repost = "UPDATE 2-Nvidia clears H200 chip sales to ten China firms - Reuters"
        assert _signature(bare) == _signature(repost)
        recent = {_signature(bare)}
        kept, suppressed = alert_recency.partition_already_alerted(
            [{"_id": "x", "title": repost}], recent
        )
        assert suppressed and kept == []


# ── DB round-trip + TTL + prune ─────────────────────────────────────────────
class TestStore:
    def test_record_then_recent_within_window(self, recency_db):
        n = alert_recency.record_alerted(
            [{"_id": "a", "title": "Samsung HBM4 shipments begin amid worker strike"}]
        )
        assert n == 1
        sigs = alert_recency.recent_signatures()
        assert _signature("Samsung HBM4 shipments begin amid worker strike") in sigs

    def test_signature_expires_after_ttl(self, recency_db):
        t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        alert_recency.record_alerted(
            [{"_id": "a", "title": "Old breaking story about MU earnings beat"}],
            now=t0,
        )
        # 5h later — still inside the 6h TTL.
        within = alert_recency.recent_signatures(now=t0 + timedelta(hours=5))
        assert _signature("Old breaking story about MU earnings beat") in within
        # 7h later — outside the TTL, must not be returned.
        after = alert_recency.recent_signatures(now=t0 + timedelta(hours=7))
        assert _signature("Old breaking story about MU earnings beat") not in after

    def test_record_prunes_beyond_2x_ttl(self, recency_db):
        t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        alert_recency.record_alerted(
            [{"_id": "old", "title": "Ancient headline to be pruned away soon"}],
            now=t0,
        )
        # A later record 13h on (> 2x the 6h TTL) must prune the ancient row.
        alert_recency.record_alerted(
            [{"_id": "new", "title": "Fresh headline that stays in the table"}],
            now=t0 + timedelta(hours=13),
        )
        conn = alert_recency._connect()
        try:
            rows = {r[0] for r in conn.execute(
                "SELECT title FROM alerted_sig").fetchall()}
        finally:
            conn.close()
        assert any("Fresh headline" in t for t in rows)
        assert not any("Ancient headline" in t for t in rows)

    def test_repeat_record_bumps_hits_not_duplicate(self, recency_db):
        title = "Repeated wire headline carried by many feeds today"
        alert_recency.record_alerted([{"_id": "a", "title": title}])
        alert_recency.record_alerted([{"_id": "b", "title": title}])
        conn = alert_recency._connect()
        try:
            rows = conn.execute(
                "SELECT sig, hits FROM alerted_sig").fetchall()
        finally:
            conn.close()
        assert len(rows) == 1, "same signature must be one upserted row"
        assert rows[0][1] == 2, "second record bumps hits"


# ── best-effort degradation ─────────────────────────────────────────────────
class TestDegradation:
    def test_open_failure_yields_empty_set_and_zero(self, monkeypatch):
        import sqlite3

        def _boom():
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(alert_recency, "_connect", _boom)
        # Neither call may raise; both degrade to the pre-feature behaviour.
        assert alert_recency.recent_signatures() == set()
        assert alert_recency.record_alerted(
            [{"_id": "a", "title": "Some breaking headline here for sure"}]
        ) == 0


# ── end-to-end through send_urgent_alert ────────────────────────────────────
class TestEndToEnd:
    def test_second_cycle_same_event_is_cross_suppressed(
        self, store, recency_db, monkeypatch
    ):
        title = "US clears H200 chip sales to ten China firms after summit"
        _insert_urgent(store, id="h1", source="rss", title=title)
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        # Cycle 1: fires normally, records the signature.
        urgent1 = store.get_unalerted_urgent()
        assert len(urgent1) == 1
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING alert body") as mc1, \
             patch("notifier.discord_notifier.send", return_value=True) as ms1:
            ok1 = alert_agent.send_urgent_alert(urgent1, store)
        assert ok1 is True
        assert mc1.called and ms1.called
        assert _urgency_of(store, "h1") == 2

        # A slower feed re-collects the SAME event as a NEW row (new id,
        # different source). It is urgency=1 and the h1 copy is urgency=2
        # (excluded from get_unalerted_urgent) — dedupe_urgent cannot see it.
        _insert_urgent(store, id="h2", source="GDELT/techtimes.com",
                       title="US clears H200 chip sales to ten China firms — TechTimes")
        urgent2 = store.get_unalerted_urgent()
        assert [a["_id"] for a in urgent2] == ["h2"]
        with patch.object(alert_agent, "claude_call") as mc2, \
             patch("notifier.discord_notifier.send") as ms2:
            ok2 = alert_agent.send_urgent_alert(urgent2, store)
        # Cross-cycle duplicate: no Claude, no Discord, returns False,
        # and the row is marked alerted so it exits the urgent queue.
        assert ok2 is False
        assert not mc2.called, "must not burn a Sonnet call on a known repeat"
        assert not ms2.called, "must not post a duplicate BREAKING push"
        assert _urgency_of(store, "h2") == 2
        assert store.get_unalerted_urgent() == []

    def test_distinct_headline_still_fires_after_a_prior_alert(
        self, store, recency_db, monkeypatch
    ):
        _insert_urgent(store, id="p1", source="rss",
                       title="Micron guides Q4 DRAM pricing sharply higher")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="alert one"), \
             patch("notifier.discord_notifier.send", return_value=True):
            assert alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store) is True

        # A genuinely DIFFERENT story must not be muted by the recency gate.
        _insert_urgent(store, id="p2", source="rss",
                       title="Fed delivers surprise emergency rate cut today")
        with patch.object(alert_agent, "claude_call",
                          return_value="alert two") as mc, \
             patch("notifier.discord_notifier.send", return_value=True) as ms:
            ok = alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store)
        assert ok is True and mc.called and ms.called
        assert _urgency_of(store, "p2") == 2
