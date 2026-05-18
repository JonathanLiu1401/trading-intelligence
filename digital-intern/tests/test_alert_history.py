"""Cross-cycle (time-windowed) repeat-story suppression for urgent alerts.

``watchers.alert_dedup.dedupe_urgent`` collapses syndicated copies *within a
single 20s alert cycle*. It does nothing across cycles: a syndicated copy of an
already-delivered story that lands as a NEW row in a LATER cycle is a fresh
``_id``/``url``, independently crosses the urgency threshold, and fires a
SECOND Bloomberg "🚨 BREAKING" for the same event. That cross-cycle echo is the
single biggest residual duplicate-noise complaint from the analyst consuming
this push channel after the within-batch dedup landed.

``watchers.alert_history`` remembers the normalized story signature (the SAME
well-tested ``alert_dedup._signature`` primitive — no signature drift) of every
story actually *delivered*, with a TTL, persisted to a tiny JSON sidecar so the
documented restart-churn does not blow the cache every ~10-30 min. A repeat of
that signature within ``ALERT_REPEAT_TTL_SECS`` is suppressed; after the TTL it
can fire again (a genuinely developing story is not muted forever).

Contract pinned here (specific-value asserts, not no-crash). Same
defense-in-depth shape / invariant posture as
``_filter_low_authority_lone``: read-only on the alert path, suppressed rows
marked ``urgency=2`` so they leave the urgent queue, row stays in
``articles.db`` (ai_score/ml_score/score_source/backtest isolation untouched),
only the duplicate *push* is dropped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent, alert_history
from watchers.alert_history import (
    ALERT_REPEAT_TTL_SECS,
    partition_repeat,
    prune_history,
    record_alerted,
    load_history,
    save_history,
    _MAX_HISTORY,
)


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, url=None,
                   title="MU earnings blow past Q3 estimates sharply",
                   source="rss", ai_score=9.0, published=None, first_seen=None):
    """Insert one live urgency=1 row exactly as the scoring path leaves it."""
    if url is None:
        url = f"https://example.com/{id}"
    if published is None:
        published = _iso(1)              # 1h old — clears the staleness gate
    if first_seen is None:
        first_seen = _iso(0.08)          # ~5 min ago — inside the 24h window
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


# ── pure: prune_history ──────────────────────────────────────────────────────
class TestPruneHistory:
    def test_drops_entries_older_than_ttl_keeps_fresh(self):
        now = 1_000_000.0
        hist = {
            "fresh story a": now - 60.0,                      # 1 min ago — keep
            "edge story b": now - (ALERT_REPEAT_TTL_SECS - 1),  # just inside — keep
            "stale story c": now - (ALERT_REPEAT_TTL_SECS + 1),  # just past — drop
            "ancient story": now - 10 * ALERT_REPEAT_TTL_SECS,   # very old — drop
        }
        out = prune_history(hist, now)
        assert set(out) == {"fresh story a", "edge story b"}
        # Pure: input dict is not mutated.
        assert len(hist) == 4

    def test_caps_at_max_history_keeping_newest(self):
        now = 1_000_000.0
        # 50 over the cap, all fresh — the oldest 50 must be dropped.
        hist = {f"sig {i}": now - float(i) for i in range(_MAX_HISTORY + 50)}
        out = prune_history(hist, now)
        assert len(out) == _MAX_HISTORY
        # Newest (smallest age = smallest i) survive; oldest i are dropped.
        assert "sig 0" in out
        assert f"sig {_MAX_HISTORY - 1}" in out
        assert f"sig {_MAX_HISTORY}" not in out
        assert f"sig {_MAX_HISTORY + 49}" not in out


# ── pure: partition_repeat ───────────────────────────────────────────────────
class TestPartitionRepeat:
    def test_recently_delivered_signature_is_suppressed(self):
        now = 1_000_000.0
        hist = {"micron shares surge after q3 earnings blowout": now - 1800.0}
        rows = [
            {"_id": "echo",
             "title": "Micron shares surge after Q3 earnings blowout - Yahoo"},
            {"_id": "new",
             "title": "Fed delivers a surprise 50bp emergency rate cut today"},
        ]
        kept, suppressed = partition_repeat(rows, hist, now)
        assert {a["_id"] for a in kept} == {"new"}
        assert {a["_id"] for a in suppressed} == {"echo"}

    def test_signature_matches_across_wire_revisions(self):
        """The whole point: a later-cycle wire revision / attributed repost of
        an already-delivered story collapses to the same signature and is
        suppressed (reuses alert_dedup._signature normalization)."""
        now = 1_000_000.0
        hist = {"micron shares surge after q3 earnings blowout": now - 600.0}
        rows = [{"_id": "rev",
                 "title": "RPT-UPDATE 3-Micron shares surge after Q3 "
                          "earnings blowout (Reuters)"}]
        kept, suppressed = partition_repeat(rows, hist, now)
        assert kept == []
        assert [a["_id"] for a in suppressed] == ["rev"]

    def test_expired_signature_fires_again(self):
        """A story whose last delivery is older than the TTL is NOT muted
        forever — a genuinely developing/persistent story re-fires."""
        now = 1_000_000.0
        hist = {"micron shares surge after q3 earnings blowout":
                now - (ALERT_REPEAT_TTL_SECS + 1)}
        rows = [{"_id": "later",
                 "title": "Micron shares surge after Q3 earnings blowout"}]
        kept, suppressed = partition_repeat(rows, hist, now)
        assert [a["_id"] for a in kept] == ["later"]
        assert suppressed == []

    def test_untitled_and_brand_new_rows_are_kept(self):
        now = 1_000_000.0
        hist = {"already alerted story headline here": now - 60.0}
        rows = [
            {"_id": "untitled", "title": ""},
            {"_id": "none_title", "title": None},
            {"_id": "fresh", "title": "Completely unrelated breaking headline"},
        ]
        kept, suppressed = partition_repeat(rows, hist, now)
        assert {a["_id"] for a in kept} == {"untitled", "none_title", "fresh"}
        assert suppressed == []


# ── pure: record_alerted ─────────────────────────────────────────────────────
class TestRecordAlerted:
    def test_records_signature_per_delivered_row_skips_untitled(self):
        now = 1_000_000.0
        hist = {"old story still tracked": now - 120.0}
        batch = [
            {"_id": "a", "title": "Nvidia smashes Q3 expectations, guides "
                                  "Q4 sharply higher"},
            {"_id": "b", "title": ""},          # untitled — not recordable
        ]
        out = record_alerted(hist, batch, now)
        assert out["nvidia smashes q3 expectations guides"] == now
        assert "old story still tracked" in out          # prior entries kept
        # No empty-signature key was written.
        assert "" not in out
        # Pure: input not mutated.
        assert "nvidia smashes q3 expectations guides" not in hist

    def test_record_then_partition_round_trip_suppresses(self):
        now = 1_000_000.0
        delivered = [{"_id": "x",
                      "title": "Samsung confirms 50,000-worker HBM4 fab strike"}]
        hist = record_alerted({}, delivered, now)
        # 30 min later a syndicated copy of the same story arrives.
        later = now + 1800.0
        echo = [{"_id": "y",
                 "title": "BREAKING: Samsung confirms 50,000-worker HBM4 "
                          "fab strike | Bloomberg"}]
        kept, suppressed = partition_repeat(echo, hist, later)
        assert kept == []
        assert [a["_id"] for a in suppressed] == ["y"]


# ── persistence (best-effort, never raises) ──────────────────────────────────
class TestPersistence:
    def test_round_trip(self, tmp_path, monkeypatch):
        p = tmp_path / "alert_signature_history.json"
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", p)
        save_history({"a story sig": 123.0, "another sig": 456.5})
        out = load_history()
        assert out == {"a story sig": 123.0, "another sig": 456.5}

    def test_missing_file_is_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(alert_history, "_HISTORY_PATH",
                            tmp_path / "does_not_exist.json")
        assert load_history() == {}

    def test_corrupt_file_degrades_to_empty_never_raises(
        self, tmp_path, monkeypatch
    ):
        p = tmp_path / "alert_signature_history.json"
        p.write_text("{ this is not valid json :::")
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", p)
        assert load_history() == {}            # no exception


# ── end-to-end through send_urgent_alert ─────────────────────────────────────
class TestCrossCycleSuppressionEndToEnd:
    def test_repeat_story_next_cycle_is_suppressed(
        self, store, tmp_path, monkeypatch
    ):
        """Cycle 1: a fresh story fires (Claude+Discord, history persisted).
        Cycle 2: a NEW row (different id/url) carrying the SAME story headline
        is suppressed — no Claude, no Discord — and marked urgency=2 so it
        exits the urgent queue."""
        hp = tmp_path / "alert_signature_history.json"
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", hp)
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        title = "Micron guides Q4 revenue sharply above the Street consensus"
        _insert_urgent(store, id="c1", title=title, source="rss")
        urgent1 = store.get_unalerted_urgent()
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mc1, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as ms1:
            ok1 = alert_agent.send_urgent_alert(urgent1, store)
        assert ok1 is True
        mc1.assert_called_once()
        ms1.assert_called_once()
        assert _urgency_of(store, "c1") == 2

        # A later-cycle syndicated copy: distinct id + url, same story.
        _insert_urgent(store, id="c2", url="https://reuters.com/micron-q4",
                       title="UPDATE 1-" + title + " - Reuters",
                       source="gdelt_gkg/reuters.com")
        urgent2 = store.get_unalerted_urgent()
        assert [a["_id"] for a in urgent2] == ["c2"], "precondition"
        with patch.object(alert_agent, "claude_call") as mc2, \
             patch("notifier.discord_notifier.send") as ms2:
            ok2 = alert_agent.send_urgent_alert(urgent2, store)
        assert ok2 is False, "cross-cycle repeat must not re-fire"
        mc2.assert_not_called()
        ms2.assert_not_called()
        assert _urgency_of(store, "c2") == 2, \
            "repeat row must exit the urgent queue (no 20s re-fetch loop)"
        assert store.get_unalerted_urgent() == []

    def test_distinct_story_next_cycle_still_fires(
        self, store, tmp_path, monkeypatch
    ):
        """The gate is signature-scoped, not a blanket mute: a genuinely
        different urgent story in a later cycle fires normally."""
        hp = tmp_path / "alert_signature_history.json"
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", hp)
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        _insert_urgent(store, id="s1", source="rss",
                       title="Micron guides Q4 revenue sharply above Street")
        with patch.object(alert_agent, "claude_call", return_value="a"), \
             patch("notifier.discord_notifier.send", return_value=True):
            assert alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store) is True

        _insert_urgent(store, id="s2", source="rss",
                       title="Fed announces emergency 50bp inter-meeting cut")
        with patch.object(alert_agent, "claude_call",
                          return_value="b") as mc, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as ms:
            ok = alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store)
        assert ok is True, "an unrelated story must still fire"
        mc.assert_called_once()
        ms.assert_called_once()
        assert _urgency_of(store, "s2") == 2

    def test_discord_failure_does_not_record_so_retry_still_fires(
        self, store, tmp_path, monkeypatch
    ):
        """History is recorded ONLY on a successful delivery. A webhook outage
        on cycle 1 must NOT poison the cache — the next-cycle retry of the
        same story still fires (no false cross-cycle suppression of an event
        the analyst never actually received)."""
        hp = tmp_path / "alert_signature_history.json"
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", hp)
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        title = "AXTI signs a multi-year InP wafer supply agreement today"
        _insert_urgent(store, id="f1", title=title, source="rss")
        with patch.object(alert_agent, "claude_call", return_value="body"), \
             patch("notifier.discord_notifier.send", return_value=False):
            ok1 = alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store)
        assert ok1 is False
        # Discord failed → row stays urgency=1 (existing re-queue contract)
        # AND nothing recorded to history.
        assert _urgency_of(store, "f1") == 1
        assert load_history() == {}

        with patch.object(alert_agent, "claude_call",
                          return_value="body") as mc, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as ms:
            ok2 = alert_agent.send_urgent_alert(
                store.get_unalerted_urgent(), store)
        assert ok2 is True, "retry of an undelivered story must still fire"
        mc.assert_called_once()
        ms.assert_called_once()
        assert _urgency_of(store, "f1") == 2


class TestBacktestIsolationPreserved:
    def test_synthetic_repeat_never_recorded_or_suppressed_via_store(
        self, store, tmp_path, monkeypatch
    ):
        """A backtest:// row with a title identical to a delivered live story
        must not interfere: get_unalerted_urgent already excludes it, and the
        history layer is read-only on ai_score/ml_score/score_source. This
        pins that the new gate does not weaken invariant #1."""
        hp = tmp_path / "alert_signature_history.json"
        monkeypatch.setattr(alert_history, "_HISTORY_PATH", hp)
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        title = "NVDA blows past Q3 estimates, guides Q4 sharply higher"
        _insert_urgent(store, id="live1", title=title, source="rss")
        # Synthetic row, same headline, urgency=1 — must never surface live.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles (id,url,title,source,published,kw_score,"
                "ai_score,urgency,first_seen,cycle,ml_score,score_source,"
                "full_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("bt1", "backtest://run_9/2026-05-18/BUY/NVDA", title,
                 "backtest_run_9_winner", _iso(1), 1.0, 9.0, 1,
                 _iso(0.08), 0, None, None, None),
            )
            store.conn.commit()

        urgent = store.get_unalerted_urgent()
        assert [a["_id"] for a in urgent] == ["live1"], \
            "store must never surface the backtest row to the alert path"
        with patch.object(alert_agent, "claude_call", return_value="x"), \
             patch("notifier.discord_notifier.send", return_value=True):
            assert alert_agent.send_urgent_alert(urgent, store) is True

        # Invariant #1/#2: the synthetic row is byte-for-byte untouched —
        # still urgency=1, ai_score intact, score_source still NULL.
        row = store.conn.execute(
            "SELECT urgency, ai_score, ml_score, score_source FROM articles "
            "WHERE id='bt1'"
        ).fetchone()
        assert row == (1, 9.0, None, None)
