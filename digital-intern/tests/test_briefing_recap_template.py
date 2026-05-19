"""Briefing-path recap / SEO template noise gate (analysis/claude_analyst.py).

The 5h Opus heartbeat digest is the analyst's primary consumed product. The
ML urgency head over-scores the *recap / preview / transcript-summary*
template (content that is inherently retrospective: "Why X Stock Is Trading
Up Today", "Stock Market Today, May 18: ...", "Q1 2026 Earnings Call
Highlights", "Here's What the Street Thinks About ...", "GF Value Says ...").
The alert path already gates these via
watchers.alert_agent._filter_recap_template_noise; the briefing path did not.

Live evidence (2026-05-19 04:18Z heartbeat — the briefing actually surfaced
in the live DB): the TOP SIGNALS line "[00:50] 9.85 MU Motley: why MU
dropped (cont., ~3.5h old)" came directly from the "Why Did Micron Stock
Drop Today ? | The Motley Fool" row scored ml_score=9.8. A 6-hour scan of
``articles.db`` surfaced six other ai_score>=7 recap rows in the same
window (LITE/AXTI/QBTS via the same templates).

Tests pin the gate's exact contract — symmetric to
tests/test_alert_recap_template.py so the two surfaces can never drift:
  1. The six fingerprints match the LIVE evidence titles verbatim.
  2. Real breaking headlines (the must-survive corpus) are NEVER caught.
  3. ``_filter_recap_template_noise`` partitions correctly; the suppressed
     rows carry the ``_recap_fingerprint`` tag; the caller's row is never
     mutated (defensive shallow-copy discipline).
  4. Integration on ``_build_payload``: a recap row in the input does NOT
     reach the NEWSWIRE section (so it can never feed Opus's TOP SIGNALS).
  5. Runs BEFORE ``_collapse_syndicated`` — a syndicated recap is suppressed
     on EVERY copy (mirrors the alert-path "before dedup" discipline).
"""
from __future__ import annotations

import copy

import pytest

from analysis.claude_analyst import (
    _looks_like_recap_template,
    _filter_recap_template_noise,
    _build_payload,
)


# ── _looks_like_recap_template: the LIVE noise must be caught ───────────────


class TestHelperCatchesLiveNoise:
    """Pin each fingerprint by the exact live strings that fired on the alert
    path (2026-05-18/19) and the row found in the live briefing DB. A regex
    tightening that re-admits these into the briefing fails the suite."""

    def test_why_x_stock_is_trading_up_today(self):
        for t in (
            "Why Nvidia (NVDA) Stock Is Trading Up Today",
            "Why Micron Stock Is Trading Down Today",
            "Why AMD Stock Is Trading Higher Today",
            "Why Tesla Stock Is Trading Lower Today",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed live recap template: {t!r}"
            assert name == "why_trading_today"

    def test_why_did_x_stock_drop_today_motley_fool(self):
        # The exact title that landed in the 2026-05-19 04:18Z briefing.
        for t in (
            "Why Did Micron Stock Drop Today ? | The Motley Fool",
            "Why Did Nvidia Stock Surge Today",
            "Why Did AMD Stock Fall Today",
            "Why Did Lumentum Stock Plunge Today",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed live recap template: {t!r}"
            assert name == "why_did_stock"

    def test_stock_market_today_dated_wrapup(self):
        for t in (
            "Stock Market Today, May 18: Micron Falls as Memory Concerns Test AI Rally",
            "Stock Market Today: May 18 — Nvidia tops record high",
            "Stock Market Today, January 5: Fed Pause Drives Tech Higher",
            "Stock Market Today, September 30: End-of-Q3 Rotation",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed dated wrap-up: {t!r}"
            assert name == "market_today_dated"

    def test_earnings_call_highlights_transcript(self):
        # Live evidence: QBTS Q1 2026 Earnings Call Highlights — ml_score 9.8
        # AND a separate ai_score 8.0 copy in the same window.
        for t in (
            "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: Surging Bookings Amid Revenue Decline",
            "Micron Technology Inc (MU) Q3 2026 Earnings Call Highlights",
            "Nvidia Corp (NVDA) Q1 2027 Earnings Call Recap",
            "Lumentum (LITE) Q2 2026 Earnings Call Takeaways",
            "AXT Inc (AXTI) Q4 2025 Earnings Call Transcript",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed earnings-call recap: {t!r}"
            assert name == "earnings_call_recap"

    def test_here_what_the_street_thinks_opinion_mill(self):
        for t in (
            "Here What the Street Thinks About ​NVIDIA Corporation ( NVDA )",
            "Here's What the Street Thinks About Micron",
            "Here is What the Street Thinks About AMD",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed Street-thinks recap: {t!r}"
            assert name == "street_thinks"

    def test_gf_value_says_gurufocus_mill(self):
        # Live evidence: LITE 02:08 ai_score 8.0, AXTI 01:36 ai_score 8.0.
        for t in (
            "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says S - GuruFocus",
            "AXT Inc (AXTI) Shares Fall 14.3% -- GF Value Says Still Overvalued - GuruFocus",
            "Micron (MU) Shares Rise 5.2% -- GF Value Says Fairly Valued",
        ):
            hit, name = _looks_like_recap_template({"title": t})
            assert hit, f"missed GF Value recap: {t!r}"
            assert name == "gf_value_says"


# ── _looks_like_recap_template: real breaking headlines MUST survive ───────


class TestHelperPreservesRealBreaking:
    """The must-survive corpus — same set the alert-path gate is tested
    against, mirrored here so the two surfaces stay in lockstep. Any regex
    tightening that catches one of these is a real regression: the briefing
    needs these as TOP SIGNALS."""

    def test_real_earnings_movers_survive(self):
        for t in (
            "MU earnings blow past Q3 estimates sharply",
            "Nvidia Q3 revenue rises 22% to $35.1 billion, beats estimates",
            "Micron raises guidance after DRAM ASP shock",
            "Lumentum cuts FY outlook on telecom weakness",
            "AMD beats on data-center revenue, AI demand strong",
        ):
            hit, _ = _looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on real earnings: {t!r}"

    def test_macro_breaking_survives(self):
        for t in (
            "Fed cuts rates by 50bp, citing labor weakness",
            "Fed holds rates steady at 4.25%-4.50% as expected",
            "CPI prints 0.3% MoM, hotter than expected",
            "Trump signs executive order on China chip exports",
            "China retaliates with new rare-earth export curbs",
        ):
            hit, _ = _looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on macro break: {t!r}"

    def test_ticker_action_survives(self):
        for t in (
            "MU shares halted on pending news",
            "NVDA halted limit-up after AI guidance",
            "TSLA tumbles 12% on delivery miss",
            "$NVDA breaks out to record high ahead of earnings",
        ):
            hit, _ = _looks_like_recap_template({"title": t})
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
            hit, _ = _looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on question form: {t!r}"

    def test_earnings_preview_not_recap_survives(self):
        """A PREVIEW of an upcoming earnings call is forward-looking and
        actionable — it must NOT be caught by the call-highlights pattern
        (discriminator verbs are highlights|recap|takeaways|transcript|
        summary; a preview uses preview|ahead of|prelim)."""
        for t in (
            "Nvidia Q1 earnings preview: all eyes on data center",
            "MU Q3 earnings preview: DRAM outlook in focus",
            "Lumentum (LITE) Q2 earnings tomorrow: 5 things to watch",
            "AXT (AXTI) Q3 earnings ahead of the open Friday",
        ):
            hit, _ = _looks_like_recap_template({"title": t})
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
            hit, _ = _looks_like_recap_template({"title": t})
            assert not hit, f"false-positive on value/analyst: {t!r}"

    def test_synthetic_snapshot_rows_survive(self):
        """The prepended PORTFOLIO/OPTIONS snapshot rows (no link/url, title
        like "PORTFOLIO P&L SNAPSHOT") must always pass through cleanly —
        same precedent as the quote-widget gate's snapshot-safe contract."""
        for t in ("PORTFOLIO P&L SNAPSHOT", "OPTIONS SNAPSHOT",
                  "PORTFOLIO P&L SNAPSHOT (as of 04:18 UTC)"):
            hit, _ = _looks_like_recap_template({"title": t})
            assert not hit, f"snapshot row falsely flagged: {t!r}"


# ── _filter_recap_template_noise — pure partition, order, no mutation ──────


class TestFilterPartitioning:
    def test_partition_separates_recap_from_real(self):
        real = {"_id": "r",
                "title": "MU earnings blow past Q3 estimates sharply"}
        recap = {"_id": "c",
                 "title": "Why Nvidia (NVDA) Stock Is Trading Up Today"}
        kept, supp = _filter_recap_template_noise([real, recap])
        assert [a["_id"] for a in kept] == ["r"]
        assert [a["_id"] for a in supp] == ["c"]

    def test_suppressed_carries_fingerprint_tag(self):
        recap = {"_id": "c",
                 "title": "Stock Market Today, May 18: Micron Falls"}
        kept, supp = _filter_recap_template_noise([recap])
        assert kept == []
        assert supp[0]["_recap_fingerprint"] == "market_today_dated"

    def test_partition_does_not_mutate_caller(self):
        """Defensive shallow-copy — heartbeat_worker feeds source_articles
        onward to the briefing-label / training path, so a silent rewrite
        of the original DB-derived dict would leak the ``_recap_fingerprint``
        field downstream. Mirrors the alert-path mutation discipline."""
        recap = {"_id": "c", "title": "Why Did Micron Stock Drop Today",
                 "source": "Motley Fool", "ai_score": 9.8}
        before = copy.deepcopy(recap)
        _filter_recap_template_noise([recap])
        assert "_recap_fingerprint" not in recap, (
            "caller's row was mutated — _recap_fingerprint leaked back"
        )
        assert recap == before

    def test_empty_or_missing_title_does_not_match(self):
        for t in (None, "", "   "):
            kept, supp = _filter_recap_template_noise(
                [{"_id": "x", "title": t}]
            )
            assert len(kept) == 1 and supp == [], (
                f"recap gate over-suppressed a no-title row: title={t!r}"
            )

    def test_empty_input_returns_two_empty_lists(self):
        kept, supp = _filter_recap_template_noise([])
        assert kept == [] and supp == []

    def test_order_preserved_for_kept(self):
        rows = [
            {"_id": "1", "title": "Fed holds rates steady amid inflation concerns"},
            {"_id": "2", "title": "Why Nvidia Stock Is Trading Up Today"},
            {"_id": "3", "title": "MU beats Q3 estimates"},
            {"_id": "4", "title": "Lumentum (LITE) Shares Fall 8.8% -- GF Value Says"},
            {"_id": "5", "title": "Trump signs chip-export order"},
        ]
        kept, supp = _filter_recap_template_noise(rows)
        assert [a["_id"] for a in kept] == ["1", "3", "5"]
        assert [a["_id"] for a in supp] == ["2", "4"]


# ── _build_payload integration — recap rows never reach the Opus prompt ─────


def _digest_section(payload: str) -> str:
    return payload.split("=== NEWSWIRE (scored, ranked) ===", 1)[1]


class TestBuildPayloadIntegration:
    """The chain ``_filter_quote_widget_noise → _filter_recap_template_noise →
    _collapse_syndicated → ... → cap[:60]`` must drop recap rows BEFORE Opus
    sees the newswire. Pinning by payload-substring is robust against the
    surrounding briefing tags ([syndicated xN] / [model] / [BOOK:] / etc.)
    that change as features get added."""

    def test_recap_row_excluded_keeps_real(self):
        articles = [
            {"title": "Micron shares surge after Q3 earnings blowout",
             "source": "rss", "ai_score": 9,
             "summary": "DRAM pricing up sharply",
             "link": "https://example.com/micron-q3"},
            # The exact live row that surfaced in the 2026-05-19 04:18Z briefing
            # (ml_score 9.8 — model-only — landed at TOP SIGNALS rank 5).
            {"title": "Why Did Micron Stock Drop Today ? | The Motley Fool",
             "source": "Motley Fool", "ai_score": 9.85,
             "summary": "MU dropped 5.95% as memory rotation fears hit",
             "link": "https://www.fool.com/x"},
        ]
        payload = _build_payload(articles, {}, [])
        section = _digest_section(payload)
        assert "Micron shares surge after Q3 earnings blowout" in section
        # The recap-template row must not appear anywhere in the payload —
        # not in newswire, not in BOOK HEAT, not in AGING TOP ROWS.
        assert "Why Did Micron Stock Drop Today" not in payload
        # The real row still rendered with its score.
        assert "score=9" in section

    def test_all_recap_batch_degrades_to_no_articles_line(self):
        """A cycle whose entire input is recap-template noise must collapse
        to the same "(no high-relevance ...)" placeholder as the all-widget
        case — never silently send Opus a recap-only newswire."""
        articles = [
            {"title": "Why Nvidia (NVDA) Stock Is Trading Up Today",
             "source": "Finnhub", "ai_score": 8.0, "link": "https://x/1"},
            {"title": "Stock Market Today, May 18: Micron Falls",
             "source": "yfinance", "ai_score": 7.0, "link": "https://x/2"},
            {"title": "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
             "source": "GuruFocus", "ai_score": 8.0, "link": "https://x/3"},
            {"title": "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says",
             "source": "GuruFocus", "ai_score": 8.0, "link": "https://x/4"},
        ]
        payload = _build_payload(articles, {}, [])
        assert "(no high-relevance articles this cycle)" in payload
        # None of the live noise titles leaked into the payload.
        for needle in ("Why Nvidia", "Stock Market Today", "Earnings Call Highlights",
                       "GF Value Says"):
            assert needle not in payload

    def test_syndicated_recap_dropped_on_every_copy(self):
        """The "Stock Market Today, May 18: ..." wrap-up — live-evidenced
        triple-fire on Motley Fool + Nasdaq + YahooFinance — must be
        suppressed on EVERY copy. Gate must run BEFORE ``_collapse_syndicated``
        so the dedup layer never gets to "pick a survivor" from a recap
        cluster."""
        title = ("Stock Market Today, May 18: Micron Falls as Memory "
                 "Concerns Test AI Rally")
        articles = [
            {"title": title, "source": "Motley Fool", "ai_score": 7.0,
             "link": "https://fool.com/x"},
            {"title": title, "source": "Nasdaq Markets", "ai_score": 7.0,
             "link": "https://nasdaq.com/x"},
            {"title": title, "source": "YahooFinance", "ai_score": 7.0,
             "link": "https://yahoo.com/x"},
            # one real story so the empty-articles branch isn't taken
            {"title": "Fed cuts rates by 50bp, citing labor weakness",
             "source": "reuters", "ai_score": 9.5,
             "link": "https://reuters.com/x"},
        ]
        payload = _build_payload(articles, {}, [])
        section = _digest_section(payload)
        assert "Fed cuts rates by 50bp" in section
        # The recap title (in any copy) must not appear in the rendered newswire.
        assert "Stock Market Today, May 18" not in payload

    def test_build_payload_does_not_mutate_caller_articles(self):
        """heartbeat_worker keeps ``source_articles`` for the training-label
        path; the recap gate (and the surrounding chain) must not drop, tag
        or reorder the caller's list."""
        articles = [
            {"title": "Real headline about a Fed decision today",
             "source": "rss", "ai_score": 8, "summary": "",
             "link": "https://example.com/fed"},
            {"title": "Why Did Micron Stock Drop Today",
             "source": "Motley Fool", "ai_score": 9.0, "summary": "",
             "link": "https://example.com/recap"},
        ]
        before = copy.deepcopy(articles)
        _build_payload(articles, {}, [])
        assert articles == before, (
            "caller's source_articles was mutated by _build_payload"
        )

    def test_mixed_batch_real_keeps_score_recap_dropped(self):
        """Mixed input: assertion is on what appears (and what doesn't) in
        the rendered newswire — the exact section that feeds Opus."""
        articles = [
            {"title": "MU shares halted on pending news",
             "source": "reuters", "ai_score": 9.0,
             "summary": "Trading halt at NASDAQ",
             "link": "https://reuters.com/mu-halt"},
            {"title": "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says S",
             "source": "GuruFocus", "ai_score": 8.0, "summary": "",
             "link": "https://gurufocus.com/lite"},
            {"title": "Why Nvidia Stock Is Trading Up Today",
             "source": "Finnhub", "ai_score": 8.0, "summary": "",
             "link": "https://finnhub.io/nvda"},
        ]
        payload = _build_payload(articles, {}, [])
        section = _digest_section(payload)
        assert "MU shares halted on pending news" in section
        assert "GF Value Says" not in payload
        assert "Why Nvidia Stock Is Trading Up Today" not in payload


# ── Lockstep parity with the alert-path gate (anti-drift) ───────────────────


def test_alert_and_briefing_gates_agree_on_six_live_titles():
    """The alert-path and briefing-path gates are duplicated rather than
    cross-imported (the documented anti-import-cycle discipline). Pin them
    against drift: any live recap title the alert path catches, the
    briefing path must also catch, and with the same fingerprint name.

    A new fingerprint added to one but not the other (or a regex change in
    one that the other misses) fails this test BEFORE it can reach prod.
    """
    from watchers import alert_agent
    titles = [
        ("Why Nvidia (NVDA) Stock Is Trading Up Today", "why_trading_today"),
        ("Why Did Micron Stock Drop Today ? | The Motley Fool", "why_did_stock"),
        ("Stock Market Today, May 18: Micron Falls as Memory Concerns",
         "market_today_dated"),
        ("D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
         "earnings_call_recap"),
        ("Here's What the Street Thinks About NVIDIA", "street_thinks"),
        ("Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says",
         "gf_value_says"),
    ]
    for title, expected_name in titles:
        alert_hit, alert_name = alert_agent._looks_like_recap_template(
            {"title": title}
        )
        brief_hit, brief_name = _looks_like_recap_template({"title": title})
        assert alert_hit and brief_hit, (
            f"alert/briefing gates disagree on hit for {title!r} "
            f"(alert={alert_hit}, briefing={brief_hit})"
        )
        assert alert_name == brief_name == expected_name, (
            f"fingerprint names drifted for {title!r}: "
            f"alert={alert_name!r}, briefing={brief_name!r}, "
            f"expected={expected_name!r}"
        )


def test_alert_and_briefing_gates_agree_on_must_survive_corpus():
    """Symmetric anti-drift check: any real breaking headline the alert path
    survives MUST also survive the briefing gate. A briefing-side false-
    positive that the alert side doesn't have would silently bury real
    breaking news that the analyst's primary product NEEDS."""
    from watchers import alert_agent
    survivors = [
        "MU earnings blow past Q3 estimates sharply",
        "Fed cuts rates by 50bp, citing labor weakness",
        "MU shares halted on pending news",
        "Why investors are bullish on Nvidia ahead of earnings",
        "Nvidia Q1 earnings preview: all eyes on data center",
        "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
        "PORTFOLIO P&L SNAPSHOT",
    ]
    for title in survivors:
        alert_hit, _ = alert_agent._looks_like_recap_template({"title": title})
        brief_hit, _ = _looks_like_recap_template({"title": title})
        assert not alert_hit and not brief_hit, (
            f"gates disagree on survivor {title!r}: alert={alert_hit}, "
            f"briefing={brief_hit}"
        )
