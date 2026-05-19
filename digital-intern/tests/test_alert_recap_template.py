"""Recap / SEO template gate on the 🚨 BREAKING alert path.

A second, distinct surface the urgency head over-scores is the *recap /
preview / transcript-summary* template — content that is inherently
retrospective ("trading up TODAY", "Q1 Earnings Call Highlights", a date-
stamped "Stock Market Today, May 18:" wrap-up) or algorithmic mill output
("(LITE) Shares Fall 8.8% -- GF Value Says ..."). These NEVER warrant a
standalone 🚨 BREAKING push: by the time the recap was written the move
was already in the market.

Live evidence (2026-05-18/19, recurring across multiple alert cycles
inspected from the live articles.db urgency=2 set):

  - "Why Nvidia (NVDA) Stock Is Trading Up Today" — fired 22:34 and 00:12
  - "Why Did Micron Stock Drop Today ? | The Motley Fool" — fired 00:50
  - "Stock Market Today, May 18: Micron Falls as Memory Concerns Test
     AI Rally" — fired THREE times at 22:52
  - "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: ..." —
     fired twice ~14 min apart
  - "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says ..."
  - "Here What the Street Thinks About NVIDIA Corporation (NVDA)"

All from publishers above the ``ALERT_MIN_LONE_SOURCE_CRED`` 0.45 bar
(Finnhub 0.78, Motley Fool/yahoo ~0.65, GoogleNews 0.62) so the existing
source-authority gate does NOT catch them — failure is content type,
not publisher credibility.

Tests pin the gate's exact contract:
  1. Six fingerprints all match the LIVE evidence titles verbatim.
  2. Real breaking headlines (the must-survive corpus) are NEVER caught.
  3. ``_filter_recap_template_noise`` partitions correctly; the
     suppressed rows carry the ``_recap_fingerprint`` debug tag.
  4. Integration on ``send_urgent_alert``: an all-recap batch never
     reaches Claude/Discord and every dropped row is marked alerted
     (exits the urgent queue instead of churning every 20s); a mixed
     batch sends only the real story; the recap row is still marked
     alerted alongside it.
  5. Runs BEFORE dedup — a syndicated recap is suppressed on EVERY copy.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import alert_agent


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class _StoreSpy:
    """Minimal stand-in for ArticleStore — records what was marked alerted
    without touching SQLite. Mirrors the spy in test_alert_agent.py so the
    two suites stay shape-consistent."""

    def __init__(self):
        self.marked: list[str] = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)


def _row(_id="x", title="generic", source="rss", **kw) -> dict:
    base = {
        "_id": _id, "link": f"https://news.example.com/{_id}",
        "title": title, "source": source, "ai_score": 9.0,
        "summary": "", "published": _iso(0.2), "first_seen": _iso(0.1),
    }
    base.update(kw)
    return base


# ── _looks_like_recap_template: the LIVE noise must be caught ───────────────


class TestHelperCatchesLiveNoise:
    """Each of these titles fired a real 🚨 BREAKING Discord push live
    (2026-05-18/19) — pin them by the exact strings so a regex tightening
    that re-admits them fails this suite."""

    def test_why_x_stock_is_trading_up_today(self):
        # The live noisiest template — fired twice on 2026-05-18/19.
        for t in (
            "Why Nvidia (NVDA) Stock Is Trading Up Today",
            "Why Micron Stock Is Trading Down Today",
            "Why AMD Stock Is Trading Higher Today",
            "Why Tesla Stock Is Trading Lower Today",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed live recap template: {t!r}"
            assert name == "why_trading_today"

    def test_why_did_x_stock_drop_today_motley_fool(self):
        for t in (
            "Why Did Micron Stock Drop Today ? | The Motley Fool",
            "Why Did Nvidia Stock Surge Today",
            "Why Did AMD Stock Fall Today",
            "Why Did Lumentum Stock Plunge Today",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed live recap template: {t!r}"
            assert name == "why_did_stock"

    def test_stock_market_today_dated_wrapup(self):
        # The 22:52 triple-fire on Motley Fool + Nasdaq + YahooFinance.
        for t in (
            "Stock Market Today, May 18: Micron Falls as Memory Concerns Test AI Rally",
            "Stock Market Today: May 18 — Nvidia tops record high",
            "Stock Market Today, January 5: Fed Pause Drives Tech Higher",
            "Stock Market Today, September 30: End-of-Q3 Rotation",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed dated wrap-up: {t!r}"
            assert name == "market_today_dated"

    def test_earnings_call_highlights_transcript(self):
        # The QBTS double-fire on 2026-05-19 (01:03 + 01:17).
        for t in (
            "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: Surging Bookings Amid Revenue Decline",
            "Micron Technology Inc (MU) Q3 2026 Earnings Call Highlights",
            "Nvidia Corp (NVDA) Q1 2027 Earnings Call Recap",
            "Lumentum (LITE) Q2 2026 Earnings Call Takeaways",
            "AXT Inc (AXTI) Q4 2025 Earnings Call Transcript",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed earnings-call recap: {t!r}"
            assert name == "earnings_call_recap"

    def test_here_what_the_street_thinks_opinion_mill(self):
        for t in (
            "Here What the Street Thinks About ​NVIDIA Corporation ( NVDA )",
            "Here's What the Street Thinks About Micron",
            "Here is What the Street Thinks About AMD",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed Street-thinks recap: {t!r}"
            assert name == "street_thinks"

    def test_gf_value_says_gurufocus_mill(self):
        # Live: LITE 22:07 and 02:08 (~4h apart); AXTI 01:36 — all GuruFocus.
        for t in (
            "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says S - GuruFocus",
            "AXT Inc (AXTI) Shares Fall 14.3% -- GF Value Says Still Overvalued - GuruFocus",
            "Micron (MU) Shares Rise 5.2% -- GF Value Says Fairly Valued",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed GF Value recap: {t!r}"
            assert name == "gf_value_says"


# ── _looks_like_recap_template: real breaking headlines MUST survive ───────


class TestHelperPreservesRealBreaking:
    """The must-survive corpus — these are the kinds of headlines a real
    analyst NEEDS as urgent. A regex tightening that catches any of them is
    a regression."""

    def test_real_earnings_movers_survive(self):
        for t in (
            "MU earnings blow past Q3 estimates sharply",
            "Nvidia Q3 revenue rises 22% to $35.1 billion, beats estimates",
            "Micron raises guidance after DRAM ASP shock",
            "Lumentum cuts FY outlook on telecom weakness",
            "AMD beats on data-center revenue, AI demand strong",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on real earnings: {t!r}"

    def test_macro_breaking_survives(self):
        for t in (
            "Fed cuts rates by 50bp, citing labor weakness",
            "Fed holds rates steady at 4.25%-4.50% as expected",
            "CPI prints 0.3% MoM, hotter than expected",
            "Trump signs executive order on China chip exports",
            "China retaliates with new rare-earth export curbs",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on macro break: {t!r}"

    def test_ticker_action_survives(self):
        for t in (
            "MU shares halted on pending news",
            "NVDA halted limit-up after AI guidance",
            "TSLA tumbles 12% on delivery miss",
            "$NVDA breaks out to record high ahead of earnings",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on ticker action: {t!r}"

    def test_question_form_breaking_survives(self):
        """Mid-sentence "why" or future-tense questions are NOT the recap
        template — the discriminator is leading "Why <X> Stock Is Trading
        ... Today" / "Why Did <X> Stock <verb>"."""
        for t in (
            "Why investors are bullish on Nvidia ahead of earnings",
            "Why MU beat Q3 estimates",
            "Investors ask: why is the rally fading?",
            "Why the Fed may need to pause",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on question form: {t!r}"

    def test_earnings_preview_not_recap_survives(self):
        """A PREVIEW of an upcoming earnings call is forward-looking and
        actionable — it must NOT be caught by the call-highlights pattern.
        The discriminator is the verbs ``highlights|recap|takeaways|
        transcript|summary`` — a preview uses ``preview|ahead of|prelim``."""
        for t in (
            "Nvidia Q1 earnings preview: all eyes on data center",
            "MU Q3 earnings preview: DRAM outlook in focus",
            "Lumentum (LITE) Q2 earnings tomorrow: 5 things to watch",
            "AXT (AXTI) Q3 earnings ahead of the open Friday",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on earnings preview: {t!r}"

    def test_value_analyst_headlines_survive(self):
        """Real headlines that mention value/analyst takes (without the
        GuruFocus 'GF Value Says' tagline) must survive — the GF Value
        pattern is precision-targeted to the algorithmic mill."""
        for t in (
            "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
            "Wedbush says Nvidia likely to top estimates, offer strong guide",
            "Micron value case strengthens on memory cycle inflection",
            "Bank of America raises NVDA price target to $250",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on value/analyst: {t!r}"


# ── _filter_recap_template_noise: partitioning + tagging ───────────────────


class TestFilterPartitioning:
    def test_partition_separates_recap_from_real(self):
        real = _row(_id="r", title="MU earnings blow past Q3 estimates sharply")
        recap = _row(_id="c", title="Why Nvidia (NVDA) Stock Is Trading Up Today")
        kept, supp = alert_agent._filter_recap_template_noise([real, recap])
        assert [a["_id"] for a in kept] == ["r"]
        assert [a["_id"] for a in supp] == ["c"]

    def test_suppressed_carries_fingerprint_tag(self):
        recap = _row(_id="c", title="Stock Market Today, May 18: Micron Falls")
        kept, supp = alert_agent._filter_recap_template_noise([recap])
        assert kept == []
        assert supp[0]["_recap_fingerprint"] == "market_today_dated"

    def test_partition_does_not_mutate_caller(self):
        """``_filter_recap_template_noise`` adds ``_recap_fingerprint`` but
        must NOT mutate the caller's row (defensive copy — mirrors the
        ``_collapse_syndicated`` shallow-copy discipline). Suppressing a
        row should never silently rewrite the original DB-derived dict."""
        recap = _row(_id="c", title="Why Did Micron Stock Drop Today")
        original_keys = set(recap.keys())
        alert_agent._filter_recap_template_noise([recap])
        assert "_recap_fingerprint" not in recap, (
            "caller's row was mutated — _recap_fingerprint leaked back"
        )
        assert set(recap.keys()) == original_keys

    def test_empty_or_missing_title_does_not_match(self):
        for t in (None, "", "   "):
            kept, supp = alert_agent._filter_recap_template_noise(
                [_row(_id="x", title=t)]
            )
            assert len(kept) == 1 and supp == [], (
                f"recap gate over-suppressed a no-title row: title={t!r}"
            )


# ── Integration on send_urgent_alert ───────────────────────────────────────


class TestSendUrgentAlertIntegration:
    """The gate must wire into ``send_urgent_alert`` BEFORE dedup, mirror
    the quote-widget gate's mark-alerted-unconditionally discipline, and
    short-circuit a fully-noisy batch BEFORE the Claude call (no wasted
    quota, no Discord POST)."""

    def test_all_recap_batch_never_reaches_claude(self, monkeypatch):
        """All-recap batch: no Claude call, no Discord send, every id
        marked alerted (exits the urgent queue). Same shape as the
        quote-widget gate's all-noise test."""
        spy = _StoreSpy()
        batch = [
            _row(_id="a", title="Why Nvidia (NVDA) Stock Is Trading Up Today",
                 source="Finnhub/Yahoo"),
            _row(_id="b",
                 title="Stock Market Today, May 18: Micron Falls as Memory Concerns",
                 source="YahooFinance/005930.KS"),
            _row(_id="c",
                 title="D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
                 source="yfinance/GuruFocus.com"),
        ]
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(batch, spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        # ALL three suppressed and marked alerted — exits queue.
        assert sorted(spy.marked) == ["a", "b", "c"]

    def test_mixed_batch_only_real_reaches_prompt(self, monkeypatch):
        """A real story alongside two recap rows must fire (the real one),
        and BOTH recap rows must be marked alerted alongside it. The Sonnet
        prompt must contain ONLY the real headline."""
        spy = _StoreSpy()
        real = _row(_id="r", title="MU earnings blow past Q3 estimates sharply",
                    source="reuters", ai_score=9.5)
        recap1 = _row(_id="c1", title="Why Did Micron Stock Drop Today",
                      source="Motley Fool")
        recap2 = _row(_id="c2",
                      title="Lumentum (LITE) Shares Fall 8.8% -- GF Value Says S",
                      source="GoogleNews/GuruFocus")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([real, recap1, recap2], spy)
        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        # The real headline IS in the prompt.
        assert "MU earnings blow past Q3 estimates" in prompt
        # Neither recap headline leaked into the prompt.
        assert "Why Did Micron Stock Drop Today" not in prompt
        assert "GF Value Says" not in prompt
        # All three ids marked alerted: real on send-success, both recaps
        # unconditionally by the gate.
        assert sorted(spy.marked) == ["c1", "c2", "r"]

    def test_syndicated_recap_caught_before_dedup(self, monkeypatch):
        """A "Stock Market Today, May 18: ..." wrap-up syndicated across
        Motley Fool, Nasdaq Markets, and YahooFinance — live evidence — must
        suppress ALL three ids (recap gate runs BEFORE dedupe_urgent, so
        every copy is caught, not just the dedup survivor)."""
        spy = _StoreSpy()
        title = ("Stock Market Today, May 18: Micron Falls as Memory "
                 "Concerns Test AI Rally")
        batch = [
            _row(_id="m1", title=title, source="Motley Fool"),
            _row(_id="m2", title=title, source="Nasdaq Markets"),
            _row(_id="m3", title=title, source="YahooFinance/005930.KS"),
        ]
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(batch, spy)
        assert ok is False
        mock_claude.assert_not_called()
        assert sorted(spy.marked) == ["m1", "m2", "m3"], (
            "syndicated recap not suppressed on every copy — gate must "
            "run BEFORE dedupe_urgent"
        )

    def test_gate_runs_after_quote_widget_gate(self, monkeypatch):
        """Both gates run BEFORE dedup and both mark-alerted-unconditionally
        on suppress; this test pins the chained behaviour — a quote widget
        and a recap in the same batch are BOTH dropped, both marked. Any
        ordering regression that lets one gate's drop skip the other's
        mark-alerted call fails here."""
        spy = _StoreSpy()
        widget = _row(_id="w",
                      title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
                      link="https://finance.yahoo.com/quote/NVDA/",
                      source="scraped/finance.yahoo.com")
        recap = _row(_id="c", title="Why Nvidia Stock Is Trading Up Today",
                     source="Finnhub/Yahoo")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([widget, recap], spy)
        assert ok is False
        mock_claude.assert_not_called()
        assert sorted(spy.marked) == ["c", "w"]
