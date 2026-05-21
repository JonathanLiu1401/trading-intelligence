"""Quote-widget pre-filter on the urgency_scorer (Sonnet) path.

Sibling to ``tests/test_urgency_recap_prefilter.py`` (the recap-template
pre-filter on the same surface) and ``tests/test_alert_agent.py`` (the
alert-formatter quote-widget gate). Same single source of truth
(``watchers.alert_agent._looks_like_quote_widget``) so a regex tightening on
the alert path cannot silently re-admit quote widgets here.

Live evidence (2026-05-21, 30d audit of articles.db): 111 rows with
``score_source='llm'`` AND ``ai_score>0`` had quote-widget-shaped titles
("NVDANVIDIA Corporation227.13-8.61(-3.65%)" / "NQ=FNasdaq 100 Jun
2629,215.25-472.50(-1.59%)"). One was Sonnet-scored 8.0 (urgent territory).
Each of those rows then entered the trainer's strong-label pool tagged as
ground-truth LLM labels (``STRONG_LABEL_WHERE`` includes
``score_source='llm'``), polluting the learning signal with patently-not-prose
junk and wasting Sonnet quota on every cycle. The alert path's
defense-in-depth quote-widget gate already drops these BEFORE Discord, but
runs too late to prevent the training-pool contamination — the row already
carries ``ai_score=8`` by then.

The discriminating asserts:

  1. A quote-widget article in the input is floored to ``ai_score=0.01,
     urgency=0, score_source='llm'`` WITHOUT calling ``claude_call`` —
     proves the pre-filter actually saves Sonnet quota AND blocks the
     training-pool pollution path.
  2. A real urgent article in the same batch IS still sent to Sonnet and
     receives its Sonnet score.
  3. An all-quote-widget batch returns 0 urgent and never calls Sonnet at
     all (the analyst-facing "wasted-call rate" goes to zero on a
     pathological tape-flood window).
  4. The must-survive corpus (real headlines that look superficially similar
     — $-prefixed cashtags, real percent moves in prose, real "Q1" notes)
     is NEVER pre-floored — they reach Sonnet exactly as they did before.
  5. Lockstep parity: ``urgency_scorer`` uses the SAME
     ``_looks_like_quote_widget`` object as the alert formatter (single
     source of truth — a future agent that copies/forks the patterns into
     urgency_scorer breaks this assertion).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import urgency_scorer, alert_agent


def _insert(store, *, id, url=None, title="t", source="rss",
            kw_score=1.0, urgency=0, first_seen=None):
    if first_seen is None:
        first_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
    if url is None:
        url = f"https://example.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, urgency,
             first_seen, 0),
        )
        store.conn.commit()


def _patched_claude(response):
    body = json.dumps(response)
    return patch.object(urgency_scorer, "claude_call", return_value=body)


# ── Live-noise pre-floor (the actual win) ───────────────────────────────────


class TestQuoteWidgetPreFloorSkipsSonnet:
    def test_quote_widget_floored_to_noise_without_calling_sonnet(self, store):
        """The exact live-evidence row: a price-glue quote-widget title must
        exit with ai_score=0.01 / urgency=0 / score_source='llm' AND zero
        Sonnet calls — proves the pre-filter actually engages BEFORE
        claude_call AND blocks the training-pool pollution path."""
        title = "NVDANVIDIA Corporation227.13-8.61(-3.65%)"
        _insert(store, id="qw1", title=title,
                source="scraped/finance.yahoo.com")
        articles = [{"_id": "qw1", "title": title, "summary": "",
                     "link": "https://finance.yahoo.com/quote/NVDA"}]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 0
        assert mock_claude.call_count == 0, (
            "Sonnet was called on a quote-widget row — pre-filter did not engage"
        )
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id='qw1'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01)
        assert row[1] == 0
        assert row[2] == "llm", (
            "quote-widget row missing score_source='llm' — it would re-enter "
            "get_unscored every cycle and waste Sonnet calls forever"
        )

    def test_percent_paren_widget_floored(self, store):
        """The parenthesised-percent fingerprint (Nasdaq future tape):
        "NQ=FNasdaq 100 Jun 2629,215.25-472.50(-1.59%)" — distinct surface
        from price-glue, same noise class."""
        title = "NQ=FNasdaq 100 Jun 2629,215.25-472.50(-1.59%)"
        _insert(store, id="qw2", title=title,
                source="scraped/finance.yahoo.com")
        articles = [{"_id": "qw2", "title": title, "summary": "",
                     "link": "https://finance.yahoo.com/quote/NQ%3DF"}]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            urgency_scorer.score_batch(articles, store)
        assert mock_claude.call_count == 0
        row = store.conn.execute(
            "SELECT ai_score, score_source FROM articles WHERE id='qw2'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01)
        assert row[1] == "llm"

    def test_share_card_listing_widget_floored(self, store):
        """Moomoo/Futu/Webull share-card listing fingerprint:
        "$NVIDIA (NVDA.US)$ - Moomoo" — distinct surface, same class."""
        title = "$NVIDIA (NVDA.US)$ - Moomoo"
        _insert(store, id="qw3", title=title, source="GN: Nvidia")
        articles = [{"_id": "qw3", "title": title, "summary": "",
                     "link": "https://www.moomoo.com/quote/NVDA"}]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            urgency_scorer.score_batch(articles, store)
        assert mock_claude.call_count == 0
        row = store.conn.execute(
            "SELECT ai_score FROM articles WHERE id='qw3'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01)

    def test_all_quote_widget_batch_skips_sonnet_entirely(self, store):
        """A whole batch of quote-widget noise (a tape-flood window) must
        short-circuit BEFORE the Sonnet call. Hard guard against an
        analyst-facing wasted-quota spike."""
        titles = [
            "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            "NQ=FNasdaq 100 Jun 2629,215.25-472.50(-1.59%)",
            "MUMicron Technology114.50-2.30(-1.97%)",
            "$NVIDIA (NVDA.US)$ - Moomoo",
            "AAPLApple Inc.198.25+1.43(+0.73%)",
        ]
        for i, t in enumerate(titles):
            _insert(store, id=f"qw{i}", title=t,
                    source="scraped/finance.yahoo.com")
        articles = [
            {"_id": f"qw{i}", "title": t, "summary": "",
             "link": f"https://finance.yahoo.com/quote/X{i}"}
            for i, t in enumerate(titles)
        ]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 0
        assert mock_claude.call_count == 0, (
            "Sonnet was called on an all-quote-widget batch — the pathological "
            "tape-flood window the fix exists to prevent"
        )
        rows = {
            r[0]: (r[1], r[2], r[3]) for r in store.conn.execute(
                "SELECT id, ai_score, urgency, score_source FROM articles"
            ).fetchall()
        }
        for i in range(len(titles)):
            ai, urg, src = rows[f"qw{i}"]
            assert ai == pytest.approx(0.01), (
                f"quote-widget row {i} not floored: ai_score={ai}"
            )
            assert urg == 0
            assert src == "llm"

    def test_mixed_batch_sonnet_gets_only_real_rows(self, store):
        """A batch with one quote-widget + one real headline: Sonnet sees
        only the real one in its payload, the real one gets its real score,
        and the quote-widget one is floored to 0.01."""
        widget_title = "NVDANVIDIA Corporation227.13-8.61(-3.65%)"
        real_title = "MU earnings beat Q3 estimates"
        _insert(store, id="widget", title=widget_title,
                source="scraped/finance.yahoo.com")
        _insert(store, id="real", title=real_title, source="rss")
        articles = [
            {"_id": "widget", "title": widget_title, "summary": "",
             "link": "https://finance.yahoo.com/quote/NVDA"},
            {"_id": "real", "title": real_title, "summary": "",
             "link": "https://reuters.com/x"},
        ]
        captured: dict = {}

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            # Sonnet's payload should index the SOLE real article as 0.
            return json.dumps([
                {"index": 0, "score": 9.5, "reason": "earnings"}
            ])

        with patch.object(urgency_scorer, "claude_call", side_effect=_capture):
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 1, "real urgent row not classified urgent"
        # The widget title must NOT appear in the Sonnet prompt — proves it
        # was excluded from the LLM payload, not silently passed through.
        assert widget_title not in captured["prompt"]
        assert real_title in captured["prompt"]
        rows = {
            r[0]: (r[1], r[2]) for r in store.conn.execute(
                "SELECT id, ai_score, urgency FROM articles"
            ).fetchall()
        }
        assert rows["widget"][0] == pytest.approx(0.01)
        assert rows["widget"][1] == 0
        assert rows["real"][0] == pytest.approx(9.5)
        assert rows["real"][1] == 1

    def test_widget_and_recap_in_same_batch_both_filtered(self, store):
        """Both pre-filters engage independently. Sonnet sees neither."""
        widget = "NVDANVIDIA Corporation227.13-8.61(-3.65%)"
        recap = "Why Did Micron Stock Drop Today ? | The Motley Fool"
        real = "MU shares halted on pending news"
        _insert(store, id="w", title=widget,
                source="scraped/finance.yahoo.com")
        _insert(store, id="r", title=recap, source="rss")
        _insert(store, id="ok", title=real, source="rss")
        articles = [
            {"_id": "w", "title": widget, "summary": "",
             "link": "https://finance.yahoo.com/quote/NVDA"},
            {"_id": "r", "title": recap, "summary": "", "link": ""},
            {"_id": "ok", "title": real, "summary": "", "link": ""},
        ]

        def _capture(prompt, **kwargs):
            return json.dumps([{"index": 0, "score": 9.0, "reason": "x"}])

        with patch.object(urgency_scorer, "claude_call",
                          side_effect=_capture) as mock:
            urgency_scorer.score_batch(articles, store)
        # Sonnet must have been called once, but the prompt must NOT contain
        # either the widget or the recap title.
        assert mock.call_count == 1
        prompt = mock.call_args[0][0]
        assert widget not in prompt
        assert recap not in prompt
        assert real in prompt
        rows = {
            r[0]: r[1] for r in store.conn.execute(
                "SELECT id, ai_score FROM articles"
            ).fetchall()
        }
        assert rows["w"] == pytest.approx(0.01)
        assert rows["r"] == pytest.approx(0.01)
        assert rows["ok"] == pytest.approx(9.0)


# ── Must-survive corpus (no false-positives on real news) ───────────────────


class TestMustSurviveReachesSonnet:
    """Real breaking headlines MUST reach Sonnet exactly as before the fix.
    Mirrors the alert-path test_alert_agent.py quote-widget must-survive
    cases so a regex tightening that catches one of these fails both."""

    def _run(self, store, title, link=""):
        _insert(store, id="s", title=title)
        articles = [{"_id": "s", "title": title, "summary": "", "link": link}]
        with _patched_claude([{"index": 0, "score": 9.0, "reason": "real"}]):
            urgency_scorer.score_batch(articles, store)
        row = store.conn.execute(
            "SELECT ai_score, urgency FROM articles WHERE id='s'"
        ).fetchone()
        return row

    def test_cashtag_prose_survives(self, store):
        """Real cashtag prose ($NVDA breaks out) must NOT be caught by the
        share-card listing fingerprint — the listing pattern requires
        "(SYMBOL.EXCH)$" close, which prose doesn't have."""
        ai, urg = self._run(store, "$NVDA breaks out to new highs on AI demand")
        assert ai == pytest.approx(9.0), (
            "cashtag prose was incorrectly pre-floored as a share-card listing"
        )
        assert urg == 1

    def test_percent_with_space_survives(self, store):
        """A real "rises 22% to $35.1 billion" prose has a space between letter
        and number — must NOT match the price-glue fingerprint."""
        ai, urg = self._run(
            store, "Nvidia revenue rises 22% to $35.1 billion in Q3"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_real_market_high_survives(self, store):
        """A real "5,123.41 record high" with a comma but space-separated from
        any letter must NOT match the price-glue fingerprint."""
        ai, urg = self._run(
            store, "S&P 500 hits 5,123.41 record high on Fed pivot hopes"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_price_target_prose_survives(self, store):
        """A real "Price Target Cut to $223.00" headline must NOT match the
        listing fingerprint (no $/SYMBOL.EXCH/$ structure)."""
        ai, urg = self._run(
            store, "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_real_quote_scoped_article_survives(self, store):
        """A real article URL UNDER a /quote/ path ("/quote/NVDA/news/...")
        must NOT match the quote-landing-page fingerprint — anchored
        end-of-path so deeper article paths slip through."""
        ai, urg = self._run(
            store,
            "Nvidia earnings shock as guidance disappoints",
            link="https://finance.yahoo.com/quote/NVDA/news/article-123.html",
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1


# ── Lockstep parity with the alert-path gate (anti-drift) ───────────────────


def test_urgency_scorer_uses_alert_agent_quote_widget_gate():
    """urgency_scorer MUST resolve quote-widget fingerprints through the SAME
    ``_looks_like_quote_widget`` function the alert formatter uses — single
    source of truth. A future agent that copies/forks the patterns into
    urgency_scorer breaks this assertion. Mirrors the
    ``test_urgency_recap_prefilter.test_urgency_scorer_uses_alert_agent_gate``
    anti-drift discipline."""
    assert (urgency_scorer._looks_like_quote_widget
            is alert_agent._looks_like_quote_widget), (
        "urgency_scorer imported a fork of _looks_like_quote_widget — "
        "fingerprint drift across alert/scorer surfaces is now possible"
    )


def test_lockstep_with_alert_path_on_live_noise():
    """The live-noise titles caught on the alert path (and the 30d audit of
    score_source='llm' rows) MUST also be caught on the urgency-scorer
    pre-filter (same gate, by parity). A regex tightening that re-admits any
    of these fails both suites in lockstep."""
    titles = [
        "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
        "NQ=FNasdaq 100 Jun 2629,215.25-472.50(-1.59%)",
        "$NVIDIA (NVDA.US)$ - Moomoo",
    ]
    for t in titles:
        assert alert_agent._looks_like_quote_widget(
            {"title": t, "link": ""}
        ), f"alert_agent gate missed live widget: {t!r}"
        # urgency_scorer reuses the same function (the parity test above
        # asserts identity); this loop redundantly proves they agree on the
        # actual live corpus, not just the import.
        assert urgency_scorer._looks_like_quote_widget(
            {"title": t, "link": ""}
        ), f"urgency_scorer pre-filter missed live widget: {t!r}"
