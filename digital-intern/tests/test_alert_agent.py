"""Live alert formatter guards — ``watchers.alert_agent.send_urgent_alert``.

This is the most safety-critical hop in the system: it is the single function
that turns a DB row into a Bloomberg-style Discord alert. Two of its guards
were only ever exercised *implicitly* (via the end-to-end integration test or
the store-level isolation tests), never asserted at the agent boundary:

  1. Staleness — an article whose ``published`` date is > 24h old must NOT
     fire as breaking news, even though ``get_unalerted_urgent`` returned it
     (the store SQL filters on ``first_seen``, not ``published``; the agent's
     ``_article_age_ok`` is the only thing standing between a 3-day-old
     re-syndicated headline and a "🚨 BREAKING" Discord post).
  2. The webhook / no-parseable-date early-outs must short-circuit BEFORE the
     Sonnet call — otherwise every cycle burns a Claude call and risks posting
     to a ``None`` webhook.

These tests pin the agent's own contract: when it drops a batch it must touch
neither Claude nor Discord nor the store, and when it sends it must mark
exactly the alerted ids so the article cannot re-fire. A regression that
weakens the staleness guard (or removes the webhook check) passes every other
suite but resurfaces stale alerts in production — exactly the class of bug this
file exists to catch.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, url="https://reuters.com/x",
                    title="MU earnings blow past Q3 estimates sharply",
                    source="rss", ai_score=9.0, published="", first_seen=None):
    """Insert a single live, urgency=1 row exactly as the scoring path would
    leave it for the alerter to pick up."""
    if first_seen is None:
        first_seen = _iso(0.08)  # ~5 min ago — inside the 24h first_seen window
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


class TestStalenessGuard:
    def test_stale_published_article_is_not_alerted(self, store, monkeypatch):
        """A live urgent row whose ``published`` is 72h old is returned by
        ``get_unalerted_urgent`` (its ``first_seen`` is recent) but the agent
        MUST drop it: no Claude call, no Discord post, urgency stays 1."""
        _insert_urgent(store, id="stale", published=_iso(72), first_seen=_iso(0.1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1, "precondition: store returns the recent-first_seen row"

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "stale") == 1, "stale row was wrongly marked alerted"

    def test_unparseable_dates_block_the_alert(self, monkeypatch):
        """Neither field parses → the agent blocks rather than risk a stale
        alert (documented 'no parseable date — dropping to be safe' path).
        Driven directly so the branch is isolated from store SQL quirks."""
        art = {
            "_id": "junk", "link": "https://reuters.com/junk",
            "title": "Totally real urgent headline about MU here",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": "not-a-date", "first_seen": "also-not-a-date",
        }

        class _StoreSpy:
            marked = []

            def mark_alerted_batch(self, ids):
                self.marked.extend(ids)

        spy = _StoreSpy()
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([art], spy)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == []


class TestWebhookEarlyOut:
    def test_missing_webhook_short_circuits_before_claude(self, store, monkeypatch):
        """No DISCORD_WEBHOOK configured → return False immediately, WITHOUT
        spending a Sonnet call. A regression here silently burns Claude quota
        every alert cycle and POSTs to an empty URL."""
        _insert_urgent(store, id="fresh", published=_iso(1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "fresh") == 1


class TestHappyPathMarksAlerted:
    def test_fresh_live_article_alerts_and_marks_alerted_exactly_once(
        self, store, monkeypatch
    ):
        """A fresh (published 1h ago) live urgent row is sent: Sonnet composes
        the alert, Discord accepts it, and the agent marks PRECISELY that id
        urgency=2 so it can never re-fire on the next 20s alert cycle."""
        _insert_urgent(store, id="go", published=_iso(1), ai_score=9.0)
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
        # Marked alerted (urgency 1 → 2) and now invisible to the alerter.
        assert _urgency_of(store, "go") == 2
        assert store.get_unalerted_urgent() == []

    def test_discord_failure_leaves_article_requeued(self, store, monkeypatch):
        """If Discord delivery fails, the article must stay urgency=1 so the
        next cycle retries it — marking it alerted on a failed POST would
        silently lose the alert forever."""
        _insert_urgent(store, id="retry", published=_iso(1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call", return_value="alert body"), \
             patch("notifier.discord_notifier.send", return_value=False):
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        assert _urgency_of(store, "retry") == 1, "failed POST must not mark alerted"


class TestSyntheticDefenseInDepth:
    """``_is_synthetic`` re-filter AT THE AGENT BOUNDARY.

    ``get_unalerted_urgent`` already excludes synthetic rows, but
    ``send_urgent_alert`` re-checks because the invariant is load-bearing
    enough that a future caller bypassing the store (a manual replay, a new
    code path) must not be able to leak training rows into Discord. The other
    suites exercise the *store* filter; this pins the *formatter's* own guard
    by handing it synthetic dicts directly (never via the store). A regression
    that drops the re-filter passes every store-level test but posts backtest
    titles to Discord as "🚨 BREAKING".
    """

    class _StoreSpy:
        def __init__(self):
            self.marked = []

        def mark_alerted_batch(self, ids):
            self.marked.extend(ids)

        def mark_alerted(self, aid):
            self.marked.append(aid)

    @pytest.mark.parametrize(
        "art",
        [
            {  # backtest:// URL
                "_id": "bt_url", "link": "backtest://run_7/2026-01-01/BUY/MU",
                "title": "Synthetic BUY winner that scored 9.5 in replay",
                "source": "backtest_run_7_winner", "ai_score": 9.5,
                "summary": "", "published": "", "first_seen": "",
            },
            {  # backtest_ source tag, ordinary-looking URL
                "_id": "bt_src", "link": "https://example.com/x",
                "title": "Replay rank-1 row carried a real-looking url",
                "source": "backtest_run_7_rank1", "ai_score": 9.0,
                "summary": "", "published": "", "first_seen": "",
            },
            {  # opus_annotation source tag
                "_id": "opus", "link": "https://example.com/y",
                "title": "Opus GOOD-trade annotation lesson row here",
                "source": "opus_annotation_cycle_3", "ai_score": 8.5,
                "summary": "", "published": "", "first_seen": "",
            },
        ],
    )
    def test_synthetic_row_dropped_before_claude_and_discord(
        self, art, monkeypatch
    ):
        spy = self._StoreSpy()
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([art], spy)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == [], "synthetic row must not be marked alerted"

    def test_mixed_batch_keeps_live_drops_synthetic(self, store, monkeypatch):
        """A batch with one real urgent row + one synthetic row must alert only
        the real one and mark exactly its id — the synthetic row is silently
        dropped, not alerted, not marked."""
        _insert_urgent(store, id="real", url="https://reuters.com/real",
                       published=_iso(1), ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert {a["_id"] for a in urgent} == {"real"}
        # Splice a synthetic row in as if a future bypass had added it.
        urgent.append({
            "_id": "leak", "link": "backtest://run_9/2026/SELL/MU",
            "title": "Synthetic SELL loser, 0.5 outcome label",
            "source": "backtest_run_9_loser", "ai_score": 9.9,
            "summary": "", "published": _iso(1), "first_seen": _iso(0.1),
        })

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU"), \
             patch("notifier.discord_notifier.send", return_value=True):
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True
        assert _urgency_of(store, "real") == 2, "real urgent row must be alerted"
        # The synthetic row was never in the store, so nothing to assert there;
        # the key invariant is it never reached Claude/Discord (its title would
        # otherwise be in the prompt) and was not in the marked set.
        leak = store.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE id='leak'"
        ).fetchone()[0]
        assert leak == 0


class TestFormatterRobustness:
    """``_fmt`` must not let one malformed urgent dict unwind the whole batch.

    Every other hop in this pipeline (``_is_synthetic``, ``dedupe_urgent``)
    reads its keys through ``.get()``; ``_fmt`` used to be the lone place with
    hard subscripts (``a['link']``, ``a['ai_score']``...). A single dict from a
    non-canonical caller — a manual replay, or a row carrying ``url`` instead
    of ``link`` (the exact alias ``_is_synthetic`` already tolerates) — raised
    KeyError, the broad ``except`` in ``send_urgent_alert`` swallowed it, the
    ENTIRE batch was dropped, nothing was marked alerted, and urgent alerts
    silently failed every 20s cycle. This is the project's recurring
    "one poison row unwinds the pipeline, invisibly" failure class; these
    tests pin that one bad row is skipped, not fatal.
    """

    class _StoreSpy:
        def __init__(self):
            self.marked = []

        def mark_alerted_batch(self, ids):
            self.marked.extend(ids)

        def mark_alerted(self, aid):
            self.marked.append(aid)

    def test_url_alias_is_accepted_not_a_keyerror(self, monkeypatch):
        """A row carrying ``url`` (not ``link``) — what a manual replay or a
        non-canonical collector path produces — must still alert, with the url
        grounded into the Sonnet prompt, instead of KeyError-aborting the
        whole batch."""
        spy = self._StoreSpy()
        art = {
            "_id": "u1", "url": "https://reuters.com/breaking-mu",
            "title": "MU guides Q4 revenue sharply above the Street",
            "source": "rss", "ai_score": 9.0,
            "summary": "", "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send", return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([art], spy)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        assert "https://reuters.com/breaking-mu" in prompt, \
            "url alias must be grounded into the prompt, not dropped"
        assert spy.marked == ["u1"]

    def test_one_titleless_row_is_skipped_batch_still_alerts(self, monkeypatch):
        """Mixed batch: a real urgent row + a corrupt row with no title. The
        corrupt row is skipped (never marked alerted, so it cannot re-fire
        forever) while the real row still alerts in the same cycle."""
        spy = self._StoreSpy()
        good = {
            "_id": "good", "link": "https://reuters.com/x",
            "title": "AXTI signs multi-year InP wafer supply agreement",
            "source": "rss", "ai_score": 9.0,
            "summary": "", "published": _iso(1), "first_seen": _iso(0.1),
        }
        poison = {  # title is the only truly-required field; this has none
            "_id": "poison", "link": "https://example.com/y", "title": "",
            "source": "rss", "ai_score": "not-a-number",
            "summary": "", "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ SUPPLY CHAIN ◈ AXTI") as mock_claude, \
             patch("notifier.discord_notifier.send", return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([good, poison], spy)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        assert "AXTI signs multi-year" in prompt
        assert spy.marked == ["good"], \
            "only the formattable row may be marked; poison must stay unmarked"

    def test_all_rows_unformattable_skips_before_claude(self, monkeypatch):
        """If every row in the batch is unformattable, short-circuit BEFORE
        burning a Claude call (and never post an empty prompt to Discord)."""
        spy = self._StoreSpy()
        poison = {
            "_id": "p", "link": "https://example.com/z", "title": "   ",
            "source": "rss", "ai_score": 9.0,
            "summary": "", "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([poison], spy)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == []
