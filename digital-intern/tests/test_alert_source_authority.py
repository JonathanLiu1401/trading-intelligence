"""Source-authority gate on the urgent Bloomberg alert path.

``send_urgent_alert`` already drops synthetic (``_is_synthetic``) and stale
(``_article_age_ok``) rows at the formatter boundary. This pins a third
defense-in-depth filter of the same shape — ``_filter_low_authority_lone`` —
which exists because the ML urgency head demonstrably over-scores
social/forum sources: a lone Reddit/Nitter/StockTwits post (no corroboration)
was firing standalone "🚨 BREAKING" Bloomberg alerts into the analyst's push
channel (observed live in a 24h window: reddit/r/Daytrading and
reddit/r/ValueInvesting each fired solo).

Contract pinned here (assert specific behavior, not no-crash):

  * A LONE low-credibility row (``cred < ALERT_MIN_LONE_SOURCE_CRED``,
    ``dup_count`` <= 1) is suppressed: never reaches Claude/Discord, and is
    marked ``urgency=2`` UNCONDITIONALLY so it exits the urgent queue instead
    of being re-fetched every 20s cycle forever.
  * Corroboration is the escape valve: a story syndicated across ≥2 sources
    (``dup_count`` > 1 after ``dedupe_urgent``) still fires even if its
    representative copy is low-cred. The gate MUST run after dedup — a future
    refactor that moves it before dedup loses this and is caught here.
  * A credible OR unknown source (≥ threshold; DEFAULT_SOURCE_CRED=0.55 so a
    brand-new source is never gated) still fires normally.
  * On a mixed batch with a Discord failure, the suppressed row stays marked
    alerted (it is noise, never re-fire) while the *kept* row stays
    ``urgency=1`` — the existing re-queue-on-failure contract is preserved.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import alert_agent
from watchers.alert_agent import _filter_low_authority_lone


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, url=None, title="MU earnings blow past Q3 estimates sharply",
                    source="rss", ai_score=9.0, published=None, first_seen=None):
    """Insert one live urgency=1 row exactly as the scoring path leaves it."""
    if url is None:
        url = f"https://example.com/{id}"
    if published is None:
        published = _iso(1)          # 1h old — fresh, clears the staleness gate
    if first_seen is None:
        first_seen = _iso(0.08)      # ~5 min ago — inside the 24h first_seen window
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


# ── pure helper: the after-dedup corroboration partition ────────────────────
class TestFilterHelper:
    def test_partitions_lone_lowcred_vs_corroborated_vs_credible(self):
        """Direct pin of ``_filter_low_authority_lone`` so a future refactor
        that moves the gate before dedup (losing the ``dup_count`` escape
        valve) or flips the comparison is caught at the unit level."""
        rows = [
            {"_id": "lone_reddit", "source": "reddit/r/Daytrading", "dup_count": 1},
            {"_id": "lone_nitter", "source": "nitter", "dup_count": 1},
            {"_id": "synd_reddit", "source": "reddit/r/stocks", "dup_count": 3},
            {"_id": "credible",    "source": "rss",              "dup_count": 1},
            {"_id": "unknown_src", "source": "some-new-feed-2026", "dup_count": 1},
            {"_id": "missing_dc",  "source": "stocktwits"},  # no dup_count key
        ]
        kept, suppressed = _filter_low_authority_lone(rows)
        assert {a["_id"] for a in suppressed} == {
            "lone_reddit", "lone_nitter", "missing_dc"
        }, "only LONE known-low-cred social/forum rows are suppressed"
        assert {a["_id"] for a in kept} == {
            "synd_reddit", "credible", "unknown_src"
        }, "syndicated (dup_count>1), credible, and UNKNOWN sources are kept"

    def test_threshold_boundary_is_strict_less_than(self):
        """rss (0.65), scraped (0.50) and gdelt (0.58) all clear the 0.45
        bar; the captured tier is exactly reddit/twitter/stocktwits/nitter."""
        rows = [
            {"_id": "rss", "source": "rss", "dup_count": 1},
            {"_id": "scraped", "source": "scraped/finance.yahoo.com", "dup_count": 1},
            {"_id": "gdelt", "source": "gdelt_gkg/reuters.com", "dup_count": 1},
            {"_id": "twitter", "source": "twitter", "dup_count": 1},
        ]
        kept, suppressed = _filter_low_authority_lone(rows)
        assert {a["_id"] for a in kept} == {"rss", "scraped", "gdelt"}
        assert {a["_id"] for a in suppressed} == {"twitter"}


# ── end-to-end through send_urgent_alert ────────────────────────────────────
class TestLoneLowAuthoritySuppressed:
    def test_lone_reddit_suppressed_no_claude_no_discord_marked_alerted(
        self, store, monkeypatch
    ):
        """A solo reddit urgent row: never reaches Claude/Discord, returns
        False, and is marked urgency=2 so it leaves the urgent queue (no
        infinite re-fetch every 20s)."""
        _insert_urgent(store, id="r1", source="reddit/r/Daytrading",
                       title="Why I think MU is going to absolutely rip tomorrow")
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1, "precondition: store returns the live row"

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "r1") == 2, "suppressed row must exit the queue"
        assert store.get_unalerted_urgent() == []

    def test_all_suppressed_batch_marks_every_id_alerted(self, store, monkeypatch):
        """Two distinct lone low-auth rows (reddit + nitter): the whole batch
        is suppressed, no Claude/Discord, and BOTH ids are marked alerted."""
        _insert_urgent(store, id="r2", source="reddit/r/ValueInvesting",
                       title="DD: AXTI is a hidden InP supply-chain gem here")
        _insert_urgent(store, id="n1", source="nitter",
                       title="Unconfirmed chatter about an ORCL cloud outage")
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 2

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "r2") == 2
        assert _urgency_of(store, "n1") == 2


class TestCredibleAndCorroboratedStillFire:
    def test_lone_credible_source_fires_normally(self, store, monkeypatch):
        """A lone rss (cred 0.65 ≥ 0.45) urgent row is unaffected: Sonnet
        composes the alert, Discord accepts, exactly that id → urgency=2."""
        _insert_urgent(store, id="ok1", source="rss",
                       title="MU guides Q4 revenue sharply above the Street")
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        assert "MU guides Q4 revenue" in mock_claude.call_args.args[0]
        assert _urgency_of(store, "ok1") == 2

    def test_syndicated_low_authority_story_still_fires(self, store, monkeypatch):
        """The corroboration escape valve: the SAME headline carried by both a
        reddit copy and an rss copy collapses (dedupe_urgent) to one
        representative with dup_count=2 — the gate keeps it even though a copy
        is low-cred, and BOTH underlying ids end urgency=2 (cannot re-fire)."""
        shared_title = "Micron shares surge after Q3 earnings blowout"
        _insert_urgent(store, id="syn_rss", source="rss",
                       title=shared_title, ai_score=8.0)
        _insert_urgent(store, id="syn_reddit", source="reddit/r/stocks",
                       title=shared_title, ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert {a["_id"] for a in urgent} == {"syn_rss", "syn_reddit"}

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True, "a syndicated story must still fire"
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        assert "Micron shares surge" in mock_claude.call_args.args[0]
        # Both the winning representative and the merged-away copy are marked.
        assert _urgency_of(store, "syn_rss") == 2
        assert _urgency_of(store, "syn_reddit") == 2


class TestMixedBatchRequeueContract:
    def test_discord_failure_suppresses_noise_but_requeues_real(
        self, store, monkeypatch
    ):
        """Mixed batch: one lone reddit (suppressed) + one lone rss (kept).
        Discord POST fails. The suppressed reddit row must STILL be marked
        alerted (it is noise, never re-fire), while the kept rss row must stay
        urgency=1 so the next cycle retries it — the existing
        re-queue-on-failure contract is preserved alongside the new gate."""
        _insert_urgent(store, id="noise", source="reddit/r/wallstreetbets",
                       title="YOLO update: my MU calls are printing huge today")
        _insert_urgent(store, id="real", source="rss",
                       title="AXTI signs multi-year InP wafer supply agreement")
        urgent = store.get_unalerted_urgent()
        assert {a["_id"] for a in urgent} == {"noise", "real"}

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="alert body"), \
             patch("notifier.discord_notifier.send", return_value=False):
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False, "Discord failed → overall send failed"
        assert _urgency_of(store, "noise") == 2, \
            "suppressed noise is marked alerted even when Discord later fails"
        assert _urgency_of(store, "real") == 1, \
            "kept row must stay queued on Discord failure (re-queue contract)"
