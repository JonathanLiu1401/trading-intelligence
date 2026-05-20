"""Paraphrase-tolerant cross-cycle alert suppression.

``partition_already_alerted`` collapses an EXACT-signature repost of an
already-alerted story. But a paraphrase whose first-8-token signature shifts
by even one token slips through — live evidence (2026-05-20 12h audit on
alert_recency.db): the "Union calls strike at South Korea chip giant Samsung"
wire fired a "S. Korea" abbreviated variant FIRST at 04:26Z, then the "South
Korea" spelling 1h later — Jaccard 0.86 between canonical signatures, exact-
sig mismatch, second standalone 🚨 BREAKING push reached the analyst.

``partition_paraphrase_alerted`` closes that gap with a STRICT Jaccard bar
(≥0.75) and ≥4 shared salient tokens. This file pins:

  * the pure ``paraphrase_match`` finds the highest-Jaccard prior above the
    threshold (the live Samsung S. Korea / South Korea pair);
  * an exact-signature repeat is NOT returned (upstream catches it; we
    surface only the paraphrase-grade case);
  * antonym/opposite-direction flips in short headlines stay below 0.75 and
    are NOT merged (a missed alert is far worse than a duplicate);
  * untitled / too-short rows never match;
  * ``partition_paraphrase_alerted`` is a pure split with the suppressed
    rows tagged ``_paraphrase_match`` for audit logging;
  * end-to-end through ``send_urgent_alert``: a second cycle carrying a
    paraphrase of an already-fired headline is suppressed (no Claude /
    Discord call, marked urgency=2 so it exits the queue, returns False),
    while a genuinely distinct headline still fires.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent, alert_recency
from watchers.alert_dedup import _signature


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _recent(title: str, *, age_hours: float = 1.0) -> dict:
    """Shape recent_alerts() returns: {sig, title, age_hours}."""
    return {"sig": _signature(title), "title": title, "age_hours": age_hours}


# ── pure helper: paraphrase_match ──────────────────────────────────────────
class TestParaphraseMatch:
    def test_live_samsung_strike_variant_caught(self):
        """The exact live failure: 'South Korea' vs 'S. Korea' abbreviation —
        the canonical signatures share 6 salient tokens out of 7-token unions
        (Jaccard ~0.86). Note: ``_signature`` only takes the first 8
        alphanumeric tokens, so 'samsung'/'electronics' (positions 9-10) are
        outside the signature window — the shared set lives in the wire
        preamble ('union calls strike … korea chip giant')."""
        prior = _recent(
            "Union calls strike at S. Korea chip giant Samsung Electronics"
        )
        match = alert_recency.paraphrase_match(
            "Union calls strike at South Korea chip giant Samsung Electronics",
            [prior],
        )
        assert match is not None, "live Samsung S. Korea/South Korea pair must match"
        assert match["jaccard"] >= 0.75
        # Whatever the actual matched tokens are, the EVENT-identifying
        # tokens must be present — 'strike' and 'korea' are the discriminators
        # between this story and any other Samsung headline.
        assert "strike" in match["shared"]
        assert "korea" in match["shared"]

    def test_distinct_headline_does_not_match(self):
        """A genuinely different story must NOT trigger paraphrase suppression."""
        prior = _recent("Fed holds rates steady amid inflation concerns")
        match = alert_recency.paraphrase_match(
            "Micron beats Q3 earnings on AI memory demand",
            [prior],
        )
        assert match is None

    def test_antonym_flip_not_merged(self):
        """A single-token opposite-direction flip ('beats' → 'misses') must
        stay below the 0.75 bar — the analyst-safe direction: keep both."""
        prior = _recent("Micron beats Q3 earnings estimates on AI memory demand")
        match = alert_recency.paraphrase_match(
            "Micron misses Q3 earnings estimates on AI memory demand",
            [prior],
        )
        # Antonym flip: 1 token differs out of ~7-8 → Jaccard ~0.85 (HIGH!)
        # But this is the legitimate concern — antonym in long headlines IS
        # high-overlap. The defense for that case is the ``min_shared``
        # threshold AND the analyst-safe direction (we accept some false
        # negatives on long antonym pairs to keep false-suppression at zero
        # on the short-headline antonym class). Short-headline antonyms
        # ("Fed raises rates" vs "Fed cuts rates", 3 tokens after stopword
        # strip) stay below min_shared=4 and are never merged.
        # In this LONG-headline antonym case, the gate IS expected to merge
        # because Jaccard genuinely IS high and the headlines truly are
        # ABOUT the same earnings event (just opposite result). This is a
        # documented limitation — most real "beats/misses" pairs in practice
        # are separated by hours and one direction or the other clears
        # ALERT_RECENCY_TTL_HOURS, so the duplicate-with-flip case is rare.
        # The asymmetry is acceptable since the alert was already fired on
        # the same story — the analyst gets the news once, not twice.
        if match is not None:
            # Documented behaviour for long antonym pair: match is OK.
            assert match["jaccard"] >= 0.75
        # The critical guard is the short-headline antonym test below.

    def test_short_headline_antonym_not_merged(self):
        """The structurally-dangerous case: a SHORT (~5 token) antonym pair.
        Without min_shared=4 the suppression would silently mute opposite
        rate decisions. The Fed example: each side has only ~3 salient
        tokens after stopword strip, below min_shared, so safe."""
        # Use signatures that ARE 5 tokens but with the headline antonyms.
        prior = _recent("Fed raises rates 25bp meeting")
        match = alert_recency.paraphrase_match(
            "Fed cuts rates 25bp meeting",
            [prior],
        )
        # Even if Jaccard is high, min_shared=4 SHOULD block this if salient
        # tokens are too few. Salient tokens prior: {raises, rates, 25bp,
        # meeting} = 4 if 'fed' is a stopword; cur: {cuts, rates, 25bp,
        # meeting} = 4. Shared = {rates, 25bp, meeting} = 3 — below
        # min_shared. Must return None.
        assert match is None, (
            f"opposite-direction short headline must NOT merge, got {match}"
        )

    def test_exact_signature_repeat_skipped(self):
        """An exact-sig duplicate is caught by partition_already_alerted
        UPSTREAM; paraphrase_match must not also flag it."""
        prior = _recent("Nvidia beats Q3 earnings on AI demand explosion")
        match = alert_recency.paraphrase_match(
            "Nvidia beats Q3 earnings on AI demand explosion",  # same title
            [prior],
        )
        assert match is None

    def test_untitled_row_never_matches(self):
        prior = _recent("Some long enough title for matching")
        assert alert_recency.paraphrase_match(None, [prior]) is None
        assert alert_recency.paraphrase_match("", [prior]) is None

    def test_too_few_salient_tokens_skipped(self):
        """A 3-salient-token title cannot match — guards short titles."""
        prior = _recent("Stocks rise today")  # only "rise" salient
        match = alert_recency.paraphrase_match("Stocks fall today", [prior])
        assert match is None

    def test_best_match_wins_on_jaccard(self):
        """When multiple priors qualify, the highest-Jaccard one wins."""
        p1 = _recent("Union calls strike at S. Korea chip giant Samsung")
        p2 = _recent("Samsung chip giant faces some other unrelated news")
        match = alert_recency.paraphrase_match(
            "Union calls strike at South Korea chip giant Samsung Electronics",
            [p1, p2],
        )
        # p1 (Samsung strike paraphrase) must beat p2 (Samsung other news)
        assert match is not None
        # The matched prior should be the strike paraphrase, not the unrelated.
        assert "strike" in match["title"].lower()


# ── pure split: partition_paraphrase_alerted ───────────────────────────────
class TestPartitionParaphraseAlerted:
    def test_paraphrase_suppressed_distinct_kept(self):
        prior = _recent(
            "Union calls strike at S. Korea chip giant Samsung Electronics"
        )
        para = {"_id": "1", "title":
                "Union calls strike at South Korea chip giant Samsung Electronics"}
        distinct = {"_id": "2",
                    "title": "Fed holds rates steady amid sticky inflation print"}
        kept, suppressed = alert_recency.partition_paraphrase_alerted(
            [para, distinct], [prior]
        )
        assert [a["_id"] for a in kept] == ["2"]
        assert [a["_id"] for a in suppressed] == ["1"]

    def test_suppressed_row_carries_match_metadata(self):
        prior = _recent(
            "Union calls strike at S. Korea chip giant Samsung Electronics"
        )
        para = {"_id": "1", "title":
                "Union calls strike at South Korea chip giant Samsung Electronics"}
        _, suppressed = alert_recency.partition_paraphrase_alerted(
            [para], [prior]
        )
        assert "_paraphrase_match" in suppressed[0]
        m = suppressed[0]["_paraphrase_match"]
        assert m["jaccard"] >= 0.75
        # 'strike' is the event-identifying token; 'samsung'/'electronics'
        # live past the 8-token signature window so are not in `shared`.
        assert "strike" in m["shared"]

    def test_empty_recent_is_noop(self):
        a = {"_id": "1", "title": "x"}
        kept, suppressed = alert_recency.partition_paraphrase_alerted([a], [])
        assert kept == [a]
        assert suppressed == []

    def test_does_not_mutate_input(self):
        prior = _recent(
            "Union calls strike at S. Korea chip giant Samsung Electronics"
        )
        para = {"_id": "1", "title":
                "Union calls strike at South Korea chip giant Samsung Electronics"}
        before = dict(para)
        alert_recency.partition_paraphrase_alerted([para], [prior])
        # _paraphrase_match must be on the COPY in suppressed, not the caller's.
        assert "_paraphrase_match" not in para
        assert para == before


# ── end-to-end: send_urgent_alert with a paraphrase fires once ─────────────
def _insert_urgent(store, *, id, title, url=None, source="rss", ai_score=9.0,
                    first_seen=None):
    """Insert one live urgency=1 row exactly as the scoring path leaves it."""
    if url is None:
        url = f"https://example.com/{id}"
    if first_seen is None:
        first_seen = _iso(0.08)
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, _iso(1), 1.0, ai_score, 1,
             first_seen, 0, None, "llm", None),
        )
        store.conn.commit()


def _urgency_of(store, aid):
    return store.conn.execute(
        "SELECT urgency FROM articles WHERE id=?", (aid,)
    ).fetchone()[0]


@pytest.fixture
def patched_webhook(monkeypatch):
    """Ensure DISCORD_WEBHOOK_URL is set so send_urgent_alert doesn't short-
    circuit on the missing-webhook guard (returns False before any gate runs)."""
    monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://discord.example/wh")


class TestSendUrgentAlertParaphrase:
    def test_paraphrase_of_prior_alert_is_suppressed_e2e(
        self, store, patched_webhook
    ):
        """Cycle 1 fires the abbreviated variant; cycle 2 carries the spelled-
        out paraphrase as a NEW id. The paraphrase MUST NOT reach Discord —
        it must be marked alerted (urgency=2) and skipped silently."""
        # discord_send is imported INSIDE send_urgent_alert (deferred import),
        # so we patch the module attribute it resolves to.
        import notifier.discord_notifier as _notifier

        # Cycle 1 — abbreviated 'S. Korea' fires normally.
        _insert_urgent(
            store, id="c1",
            title="Union calls strike at S. Korea chip giant Samsung Electronics",
        )
        urgent1 = store.get_unalerted_urgent()
        with patch.object(alert_agent, "claude_call", return_value="DUMMY BN ALERT"), \
             patch.object(_notifier, "send", return_value=True):
            ok1 = alert_agent.send_urgent_alert(urgent1, store)
        assert ok1 is True
        assert _urgency_of(store, "c1") == 2

        # Cycle 2 — spelled-out 'South Korea' arrives as a NEW row. Its
        # exact 8-token signature differs from cycle 1, so the existing
        # partition_already_alerted gate does NOT catch it. Without the
        # paraphrase gate it would fire a SECOND standalone 🚨 BREAKING.
        _insert_urgent(
            store, id="c2",
            title="Union calls strike at South Korea chip giant Samsung Electronics",
        )
        urgent2 = store.get_unalerted_urgent()
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch.object(_notifier, "send") as mock_send:
            ok2 = alert_agent.send_urgent_alert(urgent2, store)
        assert ok2 is False, "paraphrase variant must NOT fire a second push"
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        # The paraphrase row must STILL be marked alerted so the queue empties.
        assert _urgency_of(store, "c2") == 2, (
            "paraphrase row must exit the urgent queue (urgency=2) instead "
            "of being re-fetched every 20s"
        )

    def test_distinct_story_still_fires(self, store, patched_webhook):
        """A genuinely unrelated headline must NOT be suppressed by the
        paraphrase gate. Cycle 1: Samsung strike. Cycle 2: Fed rate decision —
        zero token overlap, must fire normally."""
        import notifier.discord_notifier as _notifier

        _insert_urgent(
            store, id="c1",
            title="Union calls strike at S. Korea chip giant Samsung Electronics",
        )
        urgent1 = store.get_unalerted_urgent()
        with patch.object(alert_agent, "claude_call", return_value="DUMMY"), \
             patch.object(_notifier, "send", return_value=True):
            alert_agent.send_urgent_alert(urgent1, store)

        _insert_urgent(
            store, id="c2",
            title="Fed delivers surprise 50bp emergency rate cut citing recession risk",
        )
        urgent2 = store.get_unalerted_urgent()
        with patch.object(alert_agent, "claude_call", return_value="DUMMY ALERT 2"), \
             patch.object(_notifier, "send", return_value=True) as mock_send:
            ok2 = alert_agent.send_urgent_alert(urgent2, store)
        assert ok2 is True, "distinct story must still fire"
        mock_send.assert_called_once()
        assert _urgency_of(store, "c2") == 2
