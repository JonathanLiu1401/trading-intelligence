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

These tests pin the agent's own contract: a dropped batch must never reach
Claude or Discord, and when it sends it must mark exactly the alerted ids so
the article cannot re-fire. A regression that weakens the staleness guard (or
removes the webhook check) passes every other suite but resurfaces stale alerts
in production — exactly the class of bug this file exists to catch.

Note the store-touch contract is deliberately asymmetric and mirrors the
production code: a *noise-suppression* drop (stale / quote-widget /
low-authority / cross-cycle) MARKS the dropped rows alerted (urgency=2) so they
exit the urgent queue instead of being re-fetched and re-dropped every 20s
cycle forever (a stale-by-`published` row only ages further — it can never
become a valid fresh alert). Only the synthetic-defense-in-depth re-filter and
the all-unformattable / no-webhook early-outs touch nothing. These tests pin
that exact split.
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
    def test_stale_published_article_is_dropped_and_exits_queue(self, store, monkeypatch):
        """A live urgent row whose ``published`` is 72h old is returned by
        ``get_unalerted_urgent`` (its ``first_seen`` is recent) but the agent
        MUST drop it as breaking news: no Claude call, no Discord post.

        It MUST also be marked alerted (urgency 1 → 2) so it exits the urgent
        queue — identical to the quote-widget / low-authority / cross-cycle
        suppression paths. Leaving it urgency=1 would re-fetch and re-drop it
        every 20s cycle until its first_seen ages out, then strand it as a
        permanent urgency=1 residue (the live bug this contract now guards:
        a stale-by-`published` row only ages further, so marking it loses no
        delivery). Store side-effect is urgency-only — ai_score/ml_score/
        score_source untouched."""
        _insert_urgent(store, id="stale", published=_iso(72), first_seen=_iso(0.1),
                       ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1, "precondition: store returns the recent-first_seen row"

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        # Marked alerted so it exits the queue (was urgency=1, now 2) — the
        # corrected contract: stale suppression mirrors quote-widget.
        assert _urgency_of(store, "stale") == 2, "stale row must exit the urgent queue"
        assert store.get_unalerted_urgent() == [], "stale row must not re-fire next cycle"
        # ai_score / score_source must be untouched (urgency-only side-effect).
        row = store.conn.execute(
            "SELECT ai_score, score_source, ml_score FROM articles WHERE id='stale'"
        ).fetchone()
        assert row == (9.0, "llm", None), "only urgency may change for a stale drop"

    def test_unparseable_dates_block_the_alert_and_exit_queue(self, monkeypatch):
        """Neither field parses → the agent blocks rather than risk a stale
        alert (documented 'no parseable date — dropping to be safe' path): no
        Claude call, no Discord post. Driven directly so the branch is isolated
        from store SQL quirks.

        It is ALSO marked alerted: a row with no parseable date in either field
        can never pass ``_article_age_ok``, so leaving it urgency=1 would have
        it re-fetched and re-dropped every 20s cycle forever (its corrupt
        first_seen string can still satisfy the store's ``first_seen >= cutoff``
        lexical compare). Exiting the queue is the corrected contract — same as
        every other noise-suppression drop."""
        art = {
            "_id": "junk", "link": "https://reuters.com/junk",
            "title": "Totally real urgent headline about MU here",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": "not-a-date", "first_seen": "also-not-a-date",
        }

        class _StoreSpy:
            def __init__(self):
                self.marked = []

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
        assert spy.marked == ["junk"], "undatable row must exit the urgent queue"

    def test_mixed_fresh_and_stale_alerts_fresh_and_evicts_stale(
        self, store, monkeypatch
    ):
        """The discriminating regression: a batch with one fresh urgent row +
        one stale-published one. The fresh row must alert (Claude+Discord, its
        title in the prompt); the stale row must be excluded from the prompt
        AND marked alerted so it exits the queue. After the cycle the queue is
        fully drained (neither row re-fires). Guards against a fix that evicts
        stale rows but accidentally also drops the fresh one, or vice-versa."""
        _insert_urgent(store, id="fresh", url="https://reuters.com/fresh",
                       title="MU guides Q4 revenue sharply above the Street",
                       published=_iso(1), first_seen=_iso(0.1), ai_score=9.0)
        _insert_urgent(store, id="old", url="https://reuters.com/old",
                       title="AXTI three-day-old re-syndicated supply headline",
                       published=_iso(72), first_seen=_iso(0.1), ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert {a["_id"] for a in urgent} == {"fresh", "old"}

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        assert "MU guides Q4 revenue" in prompt, "fresh row must reach the prompt"
        assert "three-day-old re-syndicated" not in prompt, \
            "stale row must never reach the alert prompt"
        assert _urgency_of(store, "fresh") == 2
        assert _urgency_of(store, "old") == 2, "stale row must exit the queue too"
        assert store.get_unalerted_urgent() == [], "queue fully drained, no re-fire"


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


class TestQuoteWidgetGate:
    """Defense-in-depth: a spaceless live ticker-tape title
    ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") that the urgency head
    over-scored must NOT fire a 🚨 BREAKING push, even if it reached the
    alerter from a non-web_scraper path. Suppressed rows must be marked
    alerted (exit the urgent queue) and must never burn a Claude call.
    """

    class _StoreSpy:
        def __init__(self):
            self.marked = []

        def mark_alerted_batch(self, ids):
            self.marked.extend(ids)

        def mark_alerted(self, aid):
            self.marked.append(aid)

    def _row(self, **kw):
        base = {
            "_id": "x", "link": "https://finance.yahoo.com/quote/NVDA/",
            "title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            "source": "scraped/finance.yahoo.com", "ai_score": 9.0,
            "summary": "", "published": _iso(0.2), "first_seen": _iso(0.1),
        }
        base.update(kw)
        return base

    def test_helper_rejects_widgets_accepts_real(self):
        f = alert_agent._looks_like_quote_widget
        # price glued to a decimal, parenthesised % change, /quote/ landing
        assert f({"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)"}) is True
        assert f({"title": "ETH-USDEthereum USD2,169.83"}) is True
        assert f({"title": "Some name (-3.65%)"}) is True
        assert f({"title": "NVIDIA Corporation overview",
                  "link": "https://finance.yahoo.com/quote/NVDA/"}) is True
        # `url` alias (the exact alias _is_synthetic/_fmt already tolerate)
        assert f({"title": "Nokia Oyj",
                  "url": "https://finance.yahoo.com/quote/NOK"}) is True
        # real headlines with $/%/comma numbers must survive
        for good in (
            "Nvidia Q3 revenue rises 22% to $35.1 billion, beats estimates",
            "Fed holds rates steady at 4.25%-4.50% as expected",
            "S&P 500 closes at 5,123.41 record high",
            "MU earnings blow past Q3 estimates sharply",
        ):
            assert f({"title": good, "link": "https://x.com/a/b"}) is False, good
        # a genuine article *under* a quote path is not the widget itself
        assert f({"title": "Nvidia tops Q3 estimates on AI demand",
                  "link": "https://finance.yahoo.com/quote/NVDA/news/nv-123"}
                 ) is False

    def test_all_widget_batch_suppressed_before_claude(self, monkeypatch):
        spy = self._StoreSpy()
        batch = [self._row(_id="a"),
                 self._row(_id="b", title="NOKNokia Oyj13.98-0.48(-3.35%)")]
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(batch, spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        # both marked alerted so they exit the urgent queue, not re-fetched
        assert sorted(spy.marked) == ["a", "b"]

    def test_mixed_batch_only_real_reaches_prompt(self, monkeypatch):
        spy = self._StoreSpy()
        widget = self._row(_id="w")
        real = {
            "_id": "r", "link": "https://reuters.com/mu-q3",
            "title": "MU earnings blow past Q3 estimates sharply",
            "source": "rss", "ai_score": 9.5, "summary": "",
            "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([widget, real], spy)
        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        assert "MU earnings blow past Q3 estimates" in prompt
        assert "NVDANVIDIA Corporation" not in prompt
        # widget suppressed-marked AND real alerted on success
        assert sorted(spy.marked) == ["r", "w"]

    def test_syndicated_widget_caught_before_dedup(self, monkeypatch):
        """Two copies of the same price tick from different collectors. The
        gate runs BEFORE dedupe_urgent, so BOTH ids are marked alerted — if
        it ran after dedup only the surviving representative's id would be."""
        spy = self._StoreSpy()
        batch = [
            self._row(_id="y1", source="scraped/finance.yahoo.com"),
            self._row(_id="y2", source="finnhub"),
        ]
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(batch, spy)
        assert ok is False
        mock_claude.assert_not_called()
        assert sorted(spy.marked) == ["y1", "y2"]

    # ── Quote-listing share-card fingerprint (_QW_LISTING) ──────────────────
    # "$NVIDIA (NVDA.US)$ - Moomoo" — a Google-News-indexed Moomoo/Futu/Webull
    # quote SHARE-CARD landing page, ML-relevance over-scored (live: 9.77,
    # ai_score 0) and fired urgency=2 🚨 BREAKING; recurring ≥6 prior passes.
    # A DISTINCT surface the two price/% fingerprints don't catch.
    def test_helper_rejects_quote_listing_share_card(self):
        f = alert_agent._looks_like_quote_widget
        for junk in (
            "$NVIDIA (NVDA.US)$ - Moomoo",          # the exact live row
            "$Tesla (TSLA.US)$ - Moomoo",
            "$Tencent (00700.HK)$ - Futu",          # HK numeric symbol
            "$Samsung Electronics (005930.KS)$ - Webull",
            "  $NIO Inc. (NIO.US)$",                # leading ws, no provider
        ):
            assert f({"title": junk}) is True, junk
        # Real "$TICKER ..." prose and $+paren headlines must SURVIVE — the
        # close ".EXCH)$" is the share-card-only discriminator.
        for good in (
            "$NVDA breaks out ahead of earnings (NYSE)",
            "$MU upgraded to Buy (price target $150.00)",
            "$TSLA: why I am buying the dip (analysis)",
            "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
            "Nvidia (NVDA) Q1 preview: all eyes on data center",
        ):
            assert f({"title": good, "link": "https://x.com/a/b"}) is False, good

    def test_quote_listing_share_card_suppressed_before_claude(self, monkeypatch):
        """The exact live row ($NVIDIA (NVDA.US)$ - Moomoo, GN: Nvidia,
        ml=9.77 → urgency=2) must NOT fire 🚨 BREAKING: no Claude call, no
        Discord, marked alerted so it exits the urgent queue."""
        spy = self._StoreSpy()
        row = self._row(
            _id="ql", source="GN: Nvidia", link="https://news.google.com/x",
            title="$NVIDIA (NVDA.US)$ - Moomoo", ai_score=9.77,
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["ql"]

    def test_quote_listing_mixed_batch_only_real_reaches_prompt(self, monkeypatch):
        spy = self._StoreSpy()
        listing = self._row(_id="ql", source="GN: Nvidia",
                             link="https://news.google.com/x",
                             title="$NVIDIA (NVDA.US)$ - Moomoo", ai_score=9.77)
        real = {
            "_id": "r", "link": "https://reuters.com/mu-q3",
            "title": "MU earnings blow past Q3 estimates sharply",
            "source": "rss", "ai_score": 9.5, "summary": "",
            "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([listing, real], spy)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "MU earnings blow past Q3 estimates" in prompt
        assert "$NVIDIA (NVDA.US)$" not in prompt
        assert sorted(spy.marked) == ["ql", "r"]

    # ── StockTwits sentiment fingerprint (_QW_STOCKTWITS_SENTIMENT) ─────────
    # ``[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓
    # of 30 msgs)`` — structured-data summary from
    # collectors/stocktwits_sentiment.py. ML head over-scores them (live 5h:
    # 130 rows, 45 ml>=5, several at 10.0). The stocktwits cred tier 0.30 < 0.45
    # already suppresses LONE pushes, but this is the defense-in-depth title
    # fingerprint so a syndicated copy (dup_count>1) is also caught — exactly
    # the same shape as the four prior quote-widget fingerprints.
    def test_helper_rejects_stocktwits_sentiment(self):
        f = alert_agent._looks_like_quote_widget
        for junk in (
            "[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30 msgs)",
            "[StockTwits Sentiment] ORCL Bullish: 80% Bullish / 0% Bearish (24↑ 0↓ of 30 msgs)",
            "[StockTwits Sentiment] LITE Bearish: 10% Bullish / 60% Bearish (3↑ 18↓ of 30 msgs)",
            "  [StockTwits Sentiment] MU Bullish: 30% Bullish / 0% Bearish",  # leading ws
        ):
            assert f({"title": junk}) is True, junk
        # Real headlines about StockTwits / sentiment must SURVIVE — only the
        # bracketed-marker prefix is the discriminator.
        for good in (
            "StockTwits announces new sentiment dashboard for retail traders",
            "Retail bullish on NVDA per StockTwits sentiment data",
            "Sentiment turns bullish ahead of NVDA earnings",
            "Bullish: NVDA breaks key resistance level",
        ):
            assert f({"title": good, "link": "https://x.com/a/b"}) is False, good

    def test_stocktwits_sentiment_suppressed_before_claude(self, monkeypatch):
        """A lone StockTwits sentiment row must NOT fire 🚨 BREAKING: no Claude
        call, no Discord, but marked alerted so it exits the urgent queue
        (same pre-fire-suppression mark discipline as other quote-widget gates)."""
        spy = self._StoreSpy()
        row = self._row(
            _id="st", source="stocktwits/sentiment",
            link="https://stocktwits.com/symbol/NVDA",
            title="[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30 msgs)",
            ai_score=0.0,
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["st"]

    # ── Image-credit fingerprint (_QW_IMAGE_CREDIT) ─────────────────────────
    # "Angela Weiss/AFP/Getty Images" — the hero-image photo credit on a news
    # page is wrapped inside the article's own <a> link, so web_scraper's
    # anchor-text fallback picks up the credit string as the article title.
    # Live evidence (2026-05-21 16:30:49Z, alert_recency.db): this exact title
    # fired a real 🚨 BREAKING push from ``scraped/www.bloomberg.com`` —
    # cred=0.90, well above the 0.45 lone-source bar (the authority gate
    # cannot catch this; content type IS the failure). ML urgency head scored
    # it 10.0 because the bloomberg.com URL + proper-noun tokens triggered
    # the high-relevance pattern recognition.
    def test_helper_rejects_image_credit_titles(self):
        f = alert_agent._looks_like_quote_widget
        for junk in (
            "Angela Weiss/AFP/Getty Images",            # the exact live noise
            "Tomohiro Ohsumi/Getty Images",
            "Timorthy A. Clary/AFP/Getty Images",       # initial-bearing variant
            "Anna Moneymaker/Getty Images",
            "Drew Angerer/AFP/Getty Images",
            "John Smith/Reuters",                       # 2-word + 1 agency
            "Mary Jane Doe/Bloomberg/Getty Images",     # 3-word + 2 agencies
            "  Angela Weiss/AFP/Getty Images  ",        # leading/trailing ws
        ):
            assert f({"title": junk}) is True, junk
        # Real headlines that mention agencies / slashes must SURVIVE — the
        # anchored ^...$ + Title-Case-Name + closed-agency-list trio is the
        # discriminator. Validated against the must-survive corpus including
        # cross-publisher prose and the AFP/Getty-launches edge case.
        for good in (
            "Reuters reports Q1 earnings beat",
            "Bloomberg: NVDA breaks $200",
            "Getty Images launches new product",
            "AFP Photo: 5 things to know about Q1",
            "MU drops 5%/Yahoo",                        # %/ slash mid-headline
            "Stock Market Today: Reuters/AP",           # colon-led list
            "Sam Altman/OpenAI says GPT-5 coming",      # OpenAI not in list
            "Reuters/Yahoo Finance reports earnings",   # Yahoo not in list
            "Apple/Microsoft deal closes",              # not agencies
            "AFP/Getty Images launches new service",    # mid-sentence content
            "Nvidia/AMD price war intensifies",         # both tickers
            "Q1 Earnings Preview",
            "Why Did Micron Stock Drop Today",
            "Tom Cruise",                                # 2-word name no /agency
        ):
            assert f({"title": good, "link": "https://x.com/a/b"}) is False, good

    def test_image_credit_suppressed_before_claude(self, monkeypatch):
        """The exact live row that fired 16:30:49Z (Angela Weiss/AFP/Getty
        Images from scraped/www.bloomberg.com, ml=10.0 → urgency=2) must NOT
        fire 🚨 BREAKING: no Claude call, no Discord, marked alerted so it
        exits the urgent queue. The bloomberg.com source is cred=0.90 so the
        existing source-authority gate cannot catch this — the title gate is
        the only thing standing between the photo credit and a Discord push."""
        spy = self._StoreSpy()
        row = self._row(
            _id="ic", source="scraped/www.bloomberg.com",
            link="https://www.bloomberg.com/news/articles/2026-05-21/trump-quantum",
            title="Angela Weiss/AFP/Getty Images",
            ai_score=0.0,
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["ic"]

    def test_image_credit_mixed_batch_only_real_reaches_prompt(self, monkeypatch):
        """A high-cred bloomberg.com batch with one photo credit and one real
        story: the credit is suppressed, the real story alerts, Claude only
        sees the real one in the prompt."""
        spy = self._StoreSpy()
        credit = self._row(
            _id="ic", source="scraped/www.bloomberg.com",
            link="https://www.bloomberg.com/news/articles/2026-05-21/x",
            title="Angela Weiss/AFP/Getty Images",
        )
        real = {
            "_id": "r", "link": "https://reuters.com/mu-q3",
            "title": "MU earnings blow past Q3 estimates sharply",
            "source": "rss", "ai_score": 9.5, "summary": "",
            "published": _iso(1), "first_seen": _iso(0.1),
        }
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([credit, real], spy)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "MU earnings blow past Q3 estimates" in prompt
        assert "Angela Weiss" not in prompt
        assert "Getty Images" not in prompt
        assert sorted(spy.marked) == ["ic", "r"]
