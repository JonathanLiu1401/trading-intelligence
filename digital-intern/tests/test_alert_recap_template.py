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

    def test_earnings_call_recap_widened_variants(self):
        """Live evidence (2026-05-20, NVDA earnings night): the prior regex
        REQUIRED both a year ``20\\d{2}`` AND the literal ``call`` between
        ``earnings`` and the recap-noun, so two retrospective variants leaked
        through the gate and the urgency head over-scored them into the
        🚨 BREAKING channel. Pin each live failure-case title by the exact
        string so a future tightening that re-admits them fails this suite.

        Variant A — no year, still has ``Earnings Call Highlights``:
          - "NVIDIA Q1 Earnings Call Highlights"
        Variant B — has year, but ``Earnings Transcript`` (no ``Call`` bridge):
          - "Nvidia (NVDA) Q1 2027 Earnings Transcript - The Globe and Mail"
        """
        for t in (
            # Variant A — no year (NVIDIA repeated this exact title on
            # several syndicated feeds 2026-05-21 night).
            "NVIDIA Q1 Earnings Call Highlights",
            "MU Q3 Earnings Call Highlights",
            "Lumentum Q2 Earnings Call Recap",
            # Variant B — year, but no "Call" between Earnings and recap-noun
            # (Globe-and-Mail / Seeking Alpha transcript syndications).
            "Nvidia (NVDA) Q1 2027 Earnings Transcript - The Globe and Mail",
            "Micron Technology (MU) Q3 2026 Earnings Transcript",
            "AMD Q4 2025 Earnings Summary",
            "AXT Inc (AXTI) Q4 2025 Earnings Recap",
            # Combined: no year AND no Call.
            "Nvidia Q1 Earnings Transcript",
            "MSFT Q2 Earnings Summary",
            # FY-prefixed quarter form ("FY27Q1" / "Q1 FY2027") still works.
            "Nvidia Q1 FY2027 Earnings Call Highlights",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed widened earnings recap: {t!r}"
            assert name == "earnings_call_recap"

    def test_earnings_recap_widened_does_not_catch_real_news(self):
        """The widened ``_RT_EARNINGS_CALL`` must NEVER catch forward-looking
        previews, breaking earnings beats/misses, or mid-sentence references
        — otherwise the gate would suppress real news during earnings week.

        These come from the must-survive corpus and represent the kinds of
        headlines an analyst actually wants pushed."""
        for t in (
            # Forward-looking previews — explicit "preview" terminator.
            "Q3 2026 earnings preview: what to expect",
            "Q1 Earnings Preview: NVDA outlook",
            # Breaking earnings results — no recap noun.
            "Nvidia Q1 beats estimates",
            "Q1 earnings come in below estimates",
            "Nvidia Q1 earnings: revenue beats, guidance lifted",
            "Earnings beat sends NVDA higher in pre-market",
            "MU misses Q3 estimates, shares plunge",
            # Earnings call upcoming — no recap noun terminator.
            "NVDA Q2 2026 earnings call begins at 5pm ET",
            "Nvidia Q1 earnings call tonight at 5pm",
            # Mid-sentence year/quarter — no Earnings adjacency.
            "Nvidia 2027 outlook brightens after Q1 call",
            "Q1 was the best quarter for AMD earnings since 2020",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, f"widened earnings recap wrongly caught: {t!r} (name={name})"

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

    def test_earnings_tomorrow_preview_seo_mill(self):
        """Live evidence (2026-05-19/20, 36h articles.db scan, all urgency=2):
        6 distinct hits — DECK + SCVL (neither held; pure SEO spam) fired
        BREAKING pushes on 2026-05-20 at 03:57Z and 04:12Z, plus NVDA
        syndicated 4× across FinancialContent / StockStory / MSN /
        TradingView on 2026-05-19 (03:21Z, 05:16Z, 05:42Z, 14:51Z). The
        existing earnings_call_recap pattern is POST-earnings only
        (highlights/recap/takeaways/transcript/summary verb list); this
        PRE-earnings preview variant the SEO mills use leaked through. Pin
        each live failure-case title by the exact string."""
        for t in (
            # 2026-05-20 live noise (DECK + SCVL — not portfolio names).
            "FinancialContent - Deckers ( DECK ) Reports Earnings Tomorrow : What To Expect",
            "FinancialContent - Shoe Carnival ( SCVL ) Reports Earnings Tomorrow : What To Expect",
            # 2026-05-19 NVDA syndication (4 sources, same template).
            "FinancialContent - Nvidia ( NVDA ) Reports Earnings Tomorrow : What To Expect",
            "Nvidia (NVDA) Reports Earnings Tomorrow: What To Expect - StockStory",
            "Nvidia (NVDA) reports earnings tomorrow: What to expect - MSN",
            "Nvidia (NVDA) Reports Earnings Tomorrow: What To Expect - TradingView",
            # Plausible same-template variants (singular/plural, spacing).
            "Micron Technology (MU) Reports Earnings Tomorrow: What to Expect",
            "AMD ( AMD ) Reports Earnings Tomorrow : What To Expect - Yahoo",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed live earnings-tomorrow SEO mill: {t!r}"
            assert name == "earnings_tomorrow_preview"

    def test_why_x_stock_just_moved_motley_fool_variant(self):
        """Live evidence (2026-05-19 19:49Z): "Why Micron Stock Just Popped
        Again" was Sonnet-scored urgent=8 and fired a 🚨 BREAKING alert
        because the existing why_did_stock regex required "Did". The
        adverbial past-tense form ("Just/Now/Finally/Today/... Popped/Surged/
        Tumbled/...") is the same retrospective Motley-Fool/Zacks shape and
        must be suppressed. Pinned by the actual live title."""
        for t in (
            "Why Micron Stock Just Popped Again",
            "Why Nvidia Stock Just Surged",
            "Why AMD Stock Finally Tumbled",
            "Why Tesla Stock Suddenly Soared",
            "Why Microsoft Stock Now Crashed",
            "Why Lumentum Stock Today Plunged",
            "Why AXT Stock Just Jumped on Asia Demand",
            "Why Intel Stock Already Rebounded",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed live why-just-moved recap: {t!r}"
            assert name == "why_just_moved"

    def test_todays_movers_list_barrons_column(self):
        """Live evidence (2026-05-20, articles.db urgency=1 phantom queue):
        the Barron's daily "These Stocks Are Today's Movers" column was
        ML-flagged urgent (ml_score~9.x, score_source='ml') and syndicated
        across 5+ sources without the recap gate catching it — every
        distinct ticker-composition daily produces a fresh BREAKING-eligible
        candidate. Same retrospective-recap class as the date-stamped daily
        wrap-up (already caught by _RT_MARKET_TODAY). Pinned by the actual
        live titles from the 2026-05-20 urgency=1 queue scan."""
        for t in (
            # 2026-05-20 live phantom-queue copies (multi-source syndication).
            "These Stocks Are Today’s Movers: Nvidia, Micron, Intel, Meta",
            "These Stocks Are Today’s Movers: Micron, Intel, Lowe’s, Nvidia",
            # ASCII-apostrophe / no-apostrophe variants for parser tolerance.
            "These Stocks Are Today's Movers: Nvidia, AMD, MU",
            "These Stocks Are Todays Movers: AAPL, MSFT, NVDA",
            # Plausible same-template variants ("Top Movers" / "Biggest Movers").
            "These Stocks Are Today's Top Movers: NVDA, MU, INTC",
            "These Stocks Are Today's Biggest Movers: TSLA, NVDA, AMD",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed live Today's Movers recap: {t!r}"
            assert name == "todays_movers_list"

    def test_is_x_a_buy_after_earnings_recap(self):
        """Live evidence (2026-05-21, alert_recency.db pushed-alert audit —
        the canonical record of REAL Discord pushes, distinct from urgency=2
        in articles.db which also captures gate-suppressed rows): "Is Nvidia
        a Buy After Their Latest Earnings Report?" fired a real 🚨 BREAKING
        push at 04:46:07Z from `yfinance/Motley Fool` and repeated from
        `YahooFinance/NVDA` ml_score 9.79. Both variants — bare leading
        "Is X a Buy" and subject-leading "Subject Is Still/Now/It a Buy" —
        are the same post-event valuation-question SEO template. Pin each
        live failure-case title."""
        for t in (
            # Live failure case — exact title that fired.
            "Is Nvidia a Buy After Their Latest Earnings Report?",
            # Subject-leading variant — analyst-attributed retrospective call.
            "Tesla Is Still a Buy After Q1 Beat, Says Wedbush",
            "NVDA Is Now a Buy After Earnings, Says JPM",
            # Same template, other tickers / verbs / recap-nouns.
            "Is AMD a Buy After Q3 Results?",
            "Is Micron a Buy After Their Latest Quarter?",
            "Is Lumentum a Sell After Q2 Report?",
            "Is Oracle a Hold After Q4 Earnings?",
            "AMD Is It Still a Buy After Q1 Results, Says BofA",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed Is-X-a-Buy-After-earnings recap: {t!r}"
            assert name == "is_buy_after"

    def test_why_is_x_pct_since_earnings_recap(self):
        """Live evidence (2026-05-21, alert_recency.db pushed-alert audit):
        "Why Is AGNC Investment (AGNC) Down 7.2% Since Last Earnings Report?"
        fired a real 🚨 BREAKING push at 05:19:12Z. The discriminator is the
        TRIO of leading "^Why Is" + percent-move + "since" — by definition
        retrospective ("since" anchors the move BEFORE the article was
        written). Same retrospective class as why_did_stock but present-
        tense `Is` instead of past-tense `Did`, and requires the
        `% since` discriminator so real ongoing-move coverage survives."""
        for t in (
            # Live failure case — exact title that fired.
            "Why Is AGNC Investment (AGNC) Down 7.2% Since Last Earnings Report?",
            # Same template, other tickers / directions / spacing.
            "Why Is NVDA Up 12.5% Since Q3 Results?",
            "Why is MU down 5% since earnings",
            "Why Is Tesla Higher 15.2% Since Their Last Report?",
            "Why is Lumentum lower 8% since Q2",
            "Why Is AMD Down 4.5 % Since Last Earnings Call?",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed why-is-pct-since recap: {t!r}"
            assert name == "why_is_pct_since"

    def test_why_x_is_pct_after_recap(self):
        """Live evidence (2026-05-21, alert_recency.db pushed-alert audit):
        "Why AXT (AXTI) Is Down 14.2% After Betting Big On AI-Focused Indium
        Phosphide Expansion" fired a real 🚨 BREAKING push at 11:14:35Z. The
        existing four "Why X ..." recap fingerprints all use different
        phrasings:
          - _RT_WHY_TRADING requires "trading up/down today"
          - _RT_WHY_DID requires "Did" between Why and subject
          - _RT_WHY_JUST_MOVED requires past-tense verb after adverb
          - _RT_WHY_IS_PCT_SINCE requires explicit "% since" trio
          - _RT_WHY_STOCK_IS_AFTER requires "stock is" + state-verb + after +
            earnings-noun
        None catches the present-tense ``Is <direction> N% After <event>``
        shape with arbitrary terminator. Pin the live failure-case title plus
        plausible same-template siblings."""
        for t in (
            # Live failure case — exact title that fired.
            "Why AXT (AXTI) Is Down 14.2% After Betting Big On AI-Focused Indium Phosphide Expansion",
            # Same template, other tickers / directions / spacing.
            "Why MU Is Down 7% After Q3 Guidance Cut",
            "Why NVDA Is Up 5.2% After New AI Chip Launch",
            "Why Tesla Is Down 12% After Delivery Miss",
            "Why Lumentum (LITE) Is Lower 3.8% After Slowing Telecom Outlook",
            "Why AMD Are Up 4.5 % After Analyst Day",
            "Why Intel Was Down 2.1% After Foundry Loss",
            "Why Oracle is higher 6% After Cloud Beat",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed why-pct-after recap: {t!r}"
            assert name == "why_pct_after", (
                f"wrong fingerprint for {t!r}: got {name!r}"
            )

    def test_why_pct_after_does_not_over_catch(self):
        """The why_pct_after pattern requires the QUAD of leading ``^Why`` +
        subject + ``is/are/was/were`` + direction (``up|down|higher|lower``) +
        explicit % move + ``after``. Any real headline lacking one element of
        that quad MUST NOT match — otherwise this gate would suppress legit
        ongoing-move analysis. AGNC's ``% since`` variant is deliberately
        caught by a SIBLING fingerprint (``_RT_WHY_IS_PCT_SINCE``) — this one
        must scope to ``after`` only so the two never overlap."""
        for t in (
            # Missing %.
            "Why is Tesla down?",
            "Why is the rally fading?",
            "Why is MU up today",
            "Why Lumentum is lower after Q2",
            # Missing direction word.
            "Why AXTI Is 14.2% After Earnings",
            "Why Tesla Is Trading After Hours",
            # Missing is/are/was/were auxiliary.
            "Why AXTI Down 14% After Earnings",
            "Why investors are bullish on Nvidia",
            # Missing after.
            "Why AXTI Is Down 14.2% on Earnings",
            "Why MU Is Down 7% Today",
            # Forward-looking — "could/may/might" not "is".
            "Why NVDA Stock Could Rise 10% After Q1",
            "Why MU stock may climb 5% after Q3 results",
            # ``% since`` is the sibling pattern's territory — NOT this one.
            "Why Is AGNC Investment (AGNC) Down 7.2% Since Last Earnings Report?",
            # Real news that mentions a % move.
            "Nvidia Q1 revenue rises 22% to $35.1 billion, beats estimates",
            "MU shares halted on pending news",
            "Fed cuts rates by 50bp, citing labor weakness",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            # AGNC variant SHOULD be caught — but by why_is_pct_since, not us.
            if "since" in t.lower():
                assert hit and name == "why_is_pct_since", (
                    f"AGNC-shaped variant fingerprint drifted: {t!r} → {name!r}"
                )
                continue
            assert not hit, (
                f"why_pct_after over-caught: {t!r} (name={name})"
            )

    def test_why_x_stock_is_after_earnings_recap(self):
        """Live evidence (2026-05-21 NVDA earnings night, articles.db
        urgency=2 set): "Why Nvidia Stock Is Barely Moving After Earnings
        Crushed Expectations" fired a real 🚨 BREAKING push TWICE
        (10:37:16Z from `GN: Nvidia`/Barron's + 10:50:41Z from `GN: AI
        stocks`/MSN syndication). The cross-cycle dedup caught the third
        copy at 10:59:00Z but the analyst had already received two
        pushes. score_source='ml' on both — the urgency head over-scored
        the SEO post-event explainer template; the existing four "Why X
        Stock ..." recap variants all use a different phrasing and none
        catches the present-tense `Stock Is <state-verb> After <event>`
        shape.

        Pin each live failure-case title PLUS plausible same-template
        siblings (other verbs / adverbs / recap-nouns from the closed
        verb list) so a future regex tightening that re-admits any of
        them fails this suite. The "after" + earnings-noun terminator
        is the retrospective anchor — without it ("Why X stock is
        moving") the regex fails by design (must-survive corpus)."""
        for t in (
            # Live failure cases — exact titles that fired.
            "Why Nvidia Stock Is Barely Moving After Earnings Crushed Expectations - Barron's",
            "Why Nvidia stock is barely moving after earnings crushed expectations - MSN",
            "Why Nvidia Stock Is Barely Moving After Earnings Crushed Expectations",
            # Same template, other adverbs / tickers / verbs / recap-nouns.
            "Why NVDA Stock Is Down 3% After Q1 Earnings",
            "Why Tesla Stock Is Up After Earnings Beat",
            "Why Lumentum Stock Is Falling After Q2 Results",
            "Why MU Stock Is Surging After Earnings Beat",
            "Why AMD Stock Is Still Higher After Their Latest Report",
            "Why Intel Stock Is Now Trading Lower After Q3 Guidance",
            "Why Oracle Stock Is Just Soaring After Earnings",
            "Why Cisco Stock Is Currently Down After Q4 Results",
            # No adverb, simple state.
            "Why NVDA Stock Is Climbing After Earnings",
            "Why MU Stock Is Sliding After Q3",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed why-stock-is-after recap: {t!r}"
            assert name == "why_stock_is_after"

    def test_earnings_release_price_target_simplywallst_mill(self):
        """Live evidence (2026-05-24, 30-day articles.db scan): 9 such rows
        from the SimplyWallSt / scraped/finance.yahoo.com / GN: earnings
        channel; one reached urgency=2 at ml_score=9.83 on the
        scraped/finance.yahoo.com row (cred 0.65 — above the 0.45 lone-source
        bar; content type IS the failure). The canonical noisiest copy:
        "Earnings Release: Here's Why Analysts Cut Their The Home Depot, Inc.
         (NYSE:HD) Price Target To US$378.32".

        Pin the live failure-case plus same-template siblings across other
        verbs (raised / lowered / boosted / trimmed) and tickers
        (TELA, KROS, CURI, INTU). The discriminator is the SimplyWallSt
        triple signature: ``^Earnings Release:`` + ``Here's Why Analysts
        <verb>`` + ``Price Target`` — must-survive corpus below is the
        wider gate against false-positives on real PT-change wire copy."""
        for t in (
            # Live failure case — exact title from the live DB.
            "Earnings Release: Here's Why Analysts Cut Their The Home Depot, Inc. "
            "(NYSE:HD) Price Target To US$378.32",
            # Sibling live rows (TELA, KROS, CURI, INTU).
            "Earnings Release: Here's Why Analysts Cut Their TELA Bio, Inc. "
            "(NASDAQ:TELA) Price Target To US$2.25",
            "Earnings Release: Here's Why Analysts Cut Their Keros Therapeutics, "
            "Inc. (NASDAQ:KROS) Price Target",
            "Earnings Release: Here's Why Analysts Cut Their CuriosityStream Inc. "
            "(NASDAQ:CURI) Price Target",
            "Earnings Release: Here's Why Analysts Cut Their Intuit Inc. "
            "(NASDAQ:INTU) Price Target To US$700.00",
            # Same template, other verbs the SEO mill rotates through.
            "Earnings Release: Here's Why Analysts Raised Their NVIDIA Corp "
            "(NASDAQ:NVDA) Price Target To US$200.00",
            "Earnings Release: Here's Why Analysts Lowered Their AMD "
            "(NASDAQ:AMD) Price Target",
            "Earnings Release: Here's why analysts boosted their Apple Inc. "
            "(NASDAQ:AAPL) price target",
            # Apostrophe variants (ASCII straight + curly + bare s).
            "Earnings Release: Heres Why Analysts Trimmed Their MU "
            "(NASDAQ:MU) Price Target",
            "Earnings Release: Here’s Why Analysts Hiked Their TSLA Price Target",
            "Earnings Release: Here is Why Analysts Slashed Their META Price Target",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed earnings-release PT recap: {t!r}"
            assert name == "earnings_release_pt"

    def test_earnings_release_pt_does_not_catch_real_pt_changes(self):
        """The new gate must NOT catch real wire copy about analyst PT
        changes — the discriminator is the SimplyWallSt SEO mill's
        leading ``^Earnings Release:`` + ``Here's Why Analysts <verb>`` +
        ``Price Target`` triple. Real wire copy about PT actions does
        not lead with this exact SEO prefix."""
        for t in (
            # Real PT-change wires — no "Earnings Release:" lead.
            "MU price target raised 15% by Citi",
            "Citi raises NVDA price target to $250",
            "Analysts cut their PT on Tesla after Q1 miss",
            "JPMorgan resets Nvidia stock price target after earnings",
            "MU PT lifted to $190 at Morgan Stanley on DRAM surge",
            # "Earnings Release" tokens but no "Here's Why Analysts" bridge.
            "Earnings Release shows MU beat Q3 estimates",
            "Q1 Earnings Release: NVDA crushes guidance",
            # "Here's Why Analysts" mid-sentence — no "Earnings Release:" lead.
            "MU surges: here's why analysts cut their bearish thesis",
            # "Earnings Release: Here's Why" but no analysts/PT trio.
            "Earnings Release: Here's why MU shares are surging today",
            "Earnings Release: Here's Why The Stock Is Down 10%",
            # Real forward-looking analyst question.
            "Will analysts raise NVDA price target after Q1?",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on earnings-release PT pattern: {t!r} "
                f"(name={name!r})"
            )

    def test_whats_next_after_baystreet_seo_mill(self):
        """Live evidence (2026-05-23 17:39:55Z, articles.db urgency=2 set):
        "Baystreet . ca - What is Next After NVIDIA Trounces Expectations"
        reached urgency=2 (score_source='ml', ml_score=10.0) — Baystreet sits
        at SOURCE_CRED 0.55 (DEFAULT), ABOVE the 0.45 lone-source bar so the
        source-authority gate did not catch it. The pattern was defined in
        alert_agent.py but never wired into ``_RECAP_TEMPLATE_PATTERNS`` so
        every live "What's Next After ..." mill row sailed through.

        Pin the live failure-case PLUS plausible same-template siblings (other
        verbs / tickers / post-event terminators). The "after" + post-event
        terminator is the retrospective anchor — must-survive corpus (no
        "after" / no terminator) is unaffected."""
        for t in (
            # Live failure case — exact title from the GDELT/baystreet.ca feed.
            "Baystreet . ca - What is Next After NVIDIA Trounces Expectations",
            "What is Next After NVIDIA Trounces Expectations",
            "What's Next After NVIDIA Crushes Earnings",
            # Same template, other tickers / verbs / recap terminators.
            "What is Next After AMD Beats Q3 Expectations",
            "What's next after Apple beat Q4 earnings",
            "What's Next After Tesla Misses Q3 Results",
            "What is Next After Lumentum's Q1 Report",
            "What is Next After Snowflake's IPO",
            "What is Next After Cisco Trounced Guidance",
            "What is Next After NVDA Q1 Earnings",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed whats-next-after recap: {t!r}"
            assert name == "whats_next_after"

    def test_subject_led_pct_after_recap(self):
        """SUBJECT-led sibling of why_pct_after / why_stock_is_after — the
        MarketBeat / simplywall.st / bloomingbit / Yahoo recap mill states the
        price-attribution as fact (no leading ``Why``), so the existing
        Why-anchored gates miss it.

        Live evidence (2026-05-22..24, 7d articles.db urgency=2 scan, all
        fired or queued for BREAKING push):
          - "D-Wave Quantum (QBTS) Is Up 44.5% After $100 Million Planned
             Federal Equity Investment - simplywall.st" — ai=9.0 (Sonnet
             over-scored), urgency=2 alerted
          - "NVIDIA (NASDAQ:NVDA) Shares Down 1.9% After Analyst Downgrade -
             MarketBeat" — ml=9.98 score_source='ml', urgency=2
          - "Lumentum (NASDAQ:LITE) Shares Down 8.8% After Insider Selling -
             MarketBeat" — ml=9.63 score_source='ml', urgency=2 (held position!)
          - "MakeMyTrip (MMYT) Is Down 7.8% After Mixed FY26 Earnings ..." —
             ml=9.87 score_source='ml', urgency=2

        Pin the live failure cases plus plausible same-template siblings —
        any future regex tightening that re-admits one fails this suite."""
        for t in (
            # Live failure cases — exact titles that fired or queued.
            "D-Wave Quantum (QBTS) Is Up 44.5% After $100 Million Planned "
            "Federal Equity Investment - What's Changed - simplywall.st",
            "NVIDIA (NASDAQ:NVDA) Shares Down 1.9% After Analyst Downgrade - MarketBeat",
            "Lumentum (NASDAQ:LITE) Shares Down 8.8% After Insider Selling - MarketBeat",
            "MakeMyTrip (MMYT) Is Down 7.8% After Mixed FY26 Earnings Under "
            "Travel Disruptions And Ongoing Investments",
            # Same template, other tickers / directions / verb-bridge variants.
            "Apple Inc (AAPL) Stock Is Up 5.2% After Strong Q1",
            "Tesla (NASDAQ:TSLA) Shares Down 12% After Delivery Miss",
            "Oracle (ORCL) Is Higher 6% After Cloud Beat",
            "AMD (NASDAQ:AMD) Stock Was Down 3.5% After Foundry Loss",
            "Intel Stock Is Now Up 2.1% After Guidance Beat",
            "Micron (MU) Are Down 4% After Memory Glut Warning",
            "QBTS Stock Was Higher 8% After SEC Approval",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed subject-pct-after recap: {t!r}"
            assert name == "subject_pct_after", (
                f"wrong fingerprint for {t!r}: got {name!r}"
            )

    def test_gurufocus_recap_post_earnings_mill(self):
        """Live evidence (2026-05-23, 7d articles.db urgency=2 scan — NVDA
        earnings night syndication on GoogleNews/GuruFocus / GN: Nvidia /
        GN: earnings, all above the 0.45 lone-source bar so the authority
        gate cannot catch them; content type IS the failure):
          - "NVIDIA (NVDA) Reports Robust Earnings While Valuation Appears
             At - GuruFocus"  ml=9.98 score_source='ml' (urgency=2 ×2)
          - "NVIDIA (NVDA) Reports Strong Earnings Amid AI Investment Surge
             - GuruFocus"     ml=9.26 score_source='ml' (urgency=2)
          - "NVIDIA (NVDA) Stock Faces Setback Despite Strong Earnings Report
             - GuruFocus"     ml=9.83 score_source='ml' (urgency=2)
          - "NVIDIA (NVDA) Exceeds Earnings Expectations with Strong Future O
             - GuruFocus"     ai=8.00 score_source='llm' (urgency=2,
                                              Sonnet over-scored same template)

        Pin all four live failures plus same-template siblings across other
        qualitative adjectives (Mixed / Solid / Weak / Modest / Disappointing)
        and across the three orthogonal sub-templates. The qualitative
        adjective + earnings-noun pair / "Stock Faces Setback Despite" /
        "Exceeds Earnings Expectations" are SEO-mill specific — real wires
        cite specifics, not editorial qualifiers."""
        for t in (
            # Live failure cases — exact titles that reached urgency=2.
            "NVIDIA (NVDA) Reports Robust Earnings While Valuation Appears At - GuruFocus",
            "NVIDIA (NVDA) Reports Strong Earnings Amid AI Investment Surge - GuruFocus",
            "NVIDIA (NVDA) Stock Faces Setback Despite Strong Earnings Report - GuruFocus",
            "NVIDIA (NVDA) Exceeds Earnings Expectations with Strong Future O - GuruFocus",
            # Same template, other qualitative adjectives the mill rotates.
            "AMD (AMD) Reports Mixed Earnings Despite Sector Tailwind - GuruFocus",
            "Lumentum (LITE) Reports Solid Q1 Results - GuruFocus",
            "MakeMyTrip (MMYT) Reports Weak Quarter Despite Travel Recovery - GuruFocus",
            "ORCL Reports Modest Earnings, Margins Compress - GuruFocus",
            "Tesla (TSLA) Reports Disappointing Earnings as Margins Compress",
            # Other sub-template — Stock Faces Setback Despite.
            "Micron (MU) Stock Faces Setback Despite Strong DRAM Outlook",
            # Other sub-template — Exceeds/Outperforms Earnings Expectations.
            "AMD Exceeds Earnings Expectations on Data-Center Strength",
            "MSFT Outperforms Earnings Expectations Amid Azure Growth",
            "Lumentum Outperform Earnings Expectations With Solid Margins",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert hit, f"missed GuruFocus post-earnings recap: {t!r}"
            assert name == "gurufocus_recap", (
                f"wrong fingerprint for {t!r}: got {name!r}"
            )

    def test_gurufocus_recap_does_not_catch_real_earnings_wires(self):
        """The GuruFocus gate must NOT catch real wire copy. The qualitative
        adjective + earnings-noun pair / "Stock Faces Setback Despite" /
        "Exceeds Earnings Expectations" trio is the SEO-mill signature; real
        wires cite specifics (record $X revenue, beats estimates by $Y) and
        avoid the editorial qualifiers."""
        for t in (
            # Real specifics — no qualitative adjective + earnings-noun pair.
            "NVIDIA Reports Record $81.62 Billion Revenue as Gaming Income Disappears",
            "Nvidia Q1 revenue rises 22% to $35.1 billion, beats estimates",
            "Nvidia (NVDA) tops Q1 earnings and revenue estimates - MSN",
            "Nvidia Beats Estimates With $81.62 Billion in Q1 Revenue",
            # Real "Reports Record" / "Reports Best" — Record/Best NOT in
            # the qualitative-adjective list, so legit wires survive.
            "AMD reports record Q1 revenue, raises guidance",
            "NVIDIA reports record quarter results across segments",
            "Lumentum reports best EPS quarter in five years",
            # Real "beats" / "tops" / "blows past" wires.
            "MU earnings blow past estimates",
            "Apple beats Q4 earnings, raises FY outlook",
            "Tesla tops Q1 EPS forecast on automotive margins",
            # Real wires using "Stock falls" / "Stock drops" — not the
            # SEO-mill "Stock Faces Setback Despite" editorial framing.
            "Apple shares fall after weak guidance",
            "Tesla stock drops 5% on margin miss",
            "Lumentum stock falls on telecom outlook cut",
            # Real wires using "beats" / "tops" — not "Exceeds Earnings
            # Expectations" formal framing.
            "Nvidia beats Q1 estimates by $5B",
            "MSFT tops EPS forecast for fifth consecutive quarter",
            # Macro / unrelated.
            "Fed cuts rates by 50bp, citing labor weakness",
            "Trump signs executive order on chip exports",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"GuruFocus gate over-caught real wire: {t!r} (name={name!r})"
            )

    def test_subject_pct_after_does_not_overcatch_real_breaking(self):
        """The subject_pct_after pattern must NOT catch real wire copy. The
        QUAD discriminator (1..6 subject tokens + Shares/Stock/Is/Are/Was/Were
        + Up/Down/Higher/Lower + ``\\d+%`` + ``after``) is what makes the gate
        precise — any real news that fails one element survives.

        Critical false-positive guard: a real BREAKING headline with a
        trailing price footer ("Nvidia Beats Estimates ...; Shares Up 0.2%
        After Hours") has the verb bridge 9+ tokens from start, beyond the
        ≤6 subject-lead cap, so does NOT match — the BREAKING verb at the
        front (``Beats``) is what the analyst needs to see."""
        for t in (
            # Real news with trailing price footer — verb bridge too far in.
            "Nvidia Beats Estimates With $81.62 Billion in Q1 Revenue; Shares Up 0.2% After Hours - bloomingbit",
            "Fed cuts rates 50bp; stocks up 2% after surprise announcement",
            # Missing digit %.
            "Tesla shares jump after earnings beat",
            "Stock futures edge higher ahead of Nvidia earnings",
            "MU shares halted on pending news",
            # Missing direction word.
            "NVDA Shares 2.1% After Q1",
            # Missing ``after``.
            "QBTS Is Up 44.5% Today",
            "NVDA Shares Down 1.9% On Downgrade",
            # No verb bridge — just %/move language in prose.
            "MU revenue rises 22% after Q1 beats consensus",
            "Nvidia Q1 revenue rises 22% to $35.1 billion, beats estimates",
            # Why-led — caught by the sibling pattern, NOT this one.
            "Why NVDA Is Up 5.2% After New AI Chip Launch",
            "Why Nvidia Stock Is Barely Moving After Earnings Crushed Expectations",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            if t.lstrip().lower().startswith("why"):
                # The Why-led variants SHOULD be caught — by a sibling.
                assert hit, f"why-led variant unexpectedly not caught: {t!r}"
                assert name != "subject_pct_after", (
                    f"why-led {t!r} stolen by subject_pct_after (should be "
                    f"why_*): got name={name!r}"
                )
                continue
            assert not hit, (
                f"subject_pct_after over-caught real news: {t!r} (name={name})"
            )


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

    def test_earnings_tomorrow_preview_does_not_over_catch(self):
        """The earnings_tomorrow_preview pattern requires the four-part
        signature "Reports Earnings Tomorrow:What To Expect" — a real
        earnings-day/preview headline that uses a SUBSET of those tokens
        (just "tomorrow", just "what to expect", just "earnings today") must
        NOT match. Pins the SEO-mill discriminator against the genuine
        earnings-coverage corpus the analyst NEEDS as urgent. Live evidence:
        all 8 of the genuine NVDA-earnings-day pushes that fired alongside
        the SEO-mill noise on 2026-05-20."""
        for t in (
            # Real earnings-day NVDA pushes from 2026-05-20 (must survive).
            "Nvidia's Earnings Are Hours Away. Here Are 3 Things to Watch.",
            "Stock futures edge higher ahead of Nvidia earnings",
            "Nvidia stock erases early losses ahead of earnings: what to expect - Cryptonews.",
            "NVIDIA Earnings Today: Wall Street Expects EPS to Jump to $1.76 on $78.75B Revenue",
            "Nvidia to announce Q1 earnings tonight. What to expect and why there could be a",
            "Asian stocks extend losing streak as higher yields bite, Nvidia results in focus",
            "Nvidia Stock Heads Into Earnings With Biggest Short Position In S&P 500",
            # "Earnings tomorrow" alone (no SEO-mill "What To Expect" trailer).
            "MU reports earnings tomorrow at 4pm ET",
            # "What to expect" alone (real wire piece, no "Reports Earnings
            # Tomorrow" exact phrase).
            "Fed meeting tomorrow: what to expect from the SEP",
            # "Reports earnings" without "tomorrow" — POST-earnings, caught
            # by a different gate (or legitimate same-day coverage).
            "MU reports earnings: beats DRAM expectations",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on real earnings-day push: {t!r}"
            )

    def test_todays_movers_pattern_does_not_over_catch(self):
        """The todays_movers_list pattern is anchored to ``^These Stocks Are
        Today's Movers:`` — mid-sentence "today's movers" mentions, forward-
        looking "tomorrow's movers" / "next week's movers" / "premarket
        movers" analyses, and any real headline that does NOT lead with the
        exact bracketed-list signature must survive. Pins the SEO-mill
        discriminator against the genuine market-movers coverage corpus."""
        for t in (
            # Mid-sentence "today's movers" - real analysis copy.
            "Why Some Of Today's Movers Could Run Higher Tomorrow",
            "Today's premarket movers: NVDA leads, MU lags",
            # Forward-looking, NOT recap.
            "Tomorrow's Movers To Watch: Earnings Calendar Heavy",
            "Next week's biggest movers: 8 stocks on watch",
            # Different lead pattern - analysis, not recap.
            "Today's biggest market mover is the bond rout",
            "Premarket movers: NVDA up 2%, MU down 3%",
            # Mid-headline "today's" reference.
            "After-hours movers reflect today's session weakness",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on real movers headline: {t!r}"
            )

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

    def test_is_buy_after_does_not_catch_forward_or_macro(self):
        """The is_buy_after pattern must NEVER catch:
          - Forward-looking PRE-earnings questions (`before earnings`)
          - "X is a buy" without `after` (a standalone forward-looking call)
          - Non-investment "Is ... a ..." (Fed/macro/meta questions)
          - "after" + non-earnings context ("after the crash", "after this
            rally") — the recap-noun (earnings/results/report/quarter/Q[1-4])
            terminator is REQUIRED to keep this scoped to post-earnings."""
        for t in (
            # Forward-looking — "before" not "after".
            "NVDA: Is It a Buy Before Earnings?",
            "Is Tesla a Buy Before Q1 Results?",
            # Standalone "is a buy" — no "after" → still forward-looking.
            "AMD is a buy",
            "Is Nvidia a Buy",
            "Is AMD a Buy Right Now",
            # Macro / non-investment "is ... a ..." questions.
            "Is the Fed about to act?",
            "Is investing a hobby for retail traders",
            # "after" present but no earnings-noun → out of scope.
            "Is Bitcoin headed to 100k after this rally?",
            "Is AMD a buy after the crash?",
            "Is gold a buy after the dollar weakness?",
            # Real analyst raises / target changes / wires — never "is X a buy"
            "Bank of America raises NVDA price target to $250",
            "Wedbush says Nvidia likely to top estimates",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on forward-looking / macro: {t!r}"
            )

    def test_why_is_pct_since_does_not_catch_partial_signatures(self):
        """The why_is_pct_since pattern requires the TRIO of leading "^Why Is"
        + direction-word + percent + "since". A real headline that uses a
        SUBSET (just "why is", just "since", just a percent move) must NOT
        match — that scopes the SEO-mill discriminator tightly enough to
        preserve all real ongoing-move coverage the analyst NEEDS as urgent."""
        for t in (
            # Missing "since" — real ongoing-move questions.
            "Why is Tesla down?",
            "Why is the rally fading?",
            "Why is MU up 12% today",
            "Why is Nvidia higher 5%",
            # Missing percent — abstract "why is" without explicit move.
            "Why is NVDA down since earnings",
            # Missing "^Why Is" lead — mid-sentence/different verb.
            "MU down 7% since earnings",
            "Investors ask why Tesla is down 5% since Q3",
            "Why MU beat Q3 estimates",
            "Why investors are bullish on Nvidia ahead of earnings",
            # Real macro / breaking — none of the trio present.
            "Fed cuts rates by 50bp, citing labor weakness",
            "Nvidia Q1 revenue rises 22% to $35.1 billion, beats estimates",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on partial why-is-pct-since signature: {t!r}"
            )

    def test_why_stock_is_after_does_not_catch_forward_or_real_news(self):
        """The why_stock_is_after pattern requires the QUAD of leading
        ``^Why\\s+...\\s+Stock\\s+Is`` + (adverb)? + present-state action verb
        from a CLOSED list + ``\\bafter\\b`` + recap-noun terminator. A real
        forward-looking headline that uses a SUBSET (just "Why X stock is",
        "Stock Could Rise after earnings", "Stock Is the Best Buy" with no
        state verb, "Why X is up" with no "Stock") must NOT match. Pins the
        discriminator against the must-survive corpus so the analyst's real
        breaking news survives this gate."""
        for t in (
            # Missing "Stock" — question form about company not stock.
            "Why investors are bullish on Nvidia ahead of earnings",
            "Why NVDA is the best AI play",
            "Why Nvidia is winning the AI race",
            # Question form "Why is X stock" (not "Why X stock is") — real ongoing question.
            "Why is Tesla stock falling?",
            "Why is MU stock down today?",
            # No "after" + earnings-noun — forward-looking / present-tense.
            "Why Nvidia stock is moving today",
            "Why MU stock is up",
            "Why AMD stock is breaking out",
            # Verb is "could/may/might" — future-tense, not a present state.
            "Why Microsoft Stock Could Rise After Earnings",
            "Why MU stock may climb after Q3 results",
            "Why Tesla stock might tumble after Q4 report",
            # "after" present but action verb is non-action / not in the list.
            "Why Nvidia Stock Is the Best Buy After Q1",
            "Why AMD Stock Is a Buy After Earnings",
            "Why Tesla Stock Is Worth Watching After Q3",
            # "after" present but non-earnings context — out of scope.
            "Why Nvidia Stock Is Surging After the Fed Cut",
            "Why MU Stock Is Falling After the China News",
            "Why Tesla Stock Is Climbing After the Crash",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on why-stock-is-after pattern: {t!r} "
                f"(name={name})"
            )

    def test_whats_next_after_does_not_catch_forward_looking(self):
        """Forward-looking "What's next" or "Next after <non-event>" headlines
        are real ongoing analyst questions — they must NOT be caught. The
        retrospective anchor is BOTH the leading "what(?:'s|\\s+is)? next
        after" bridge AND a post-event terminator (trounces/crushes/beats/
        earnings/results/quarter/q1-q4/miss/ipo/guidance). Lose either and the
        title falls through."""
        for t in (
            # "What's next" without "after" — generic forward question.
            "What's next for the chip cycle?",
            "What's next for Apple in 2026",
            "What's next for the AI rally",
            # "What's next AFTER <non-event-terminator>" — the "after" is
            # there but the terminator isn't an earnings/recap noun.
            "What's next after the Fed cut rates",
            "What is next after the China trade deal",
            "What's next after the new product launch",
            # Real macro / forward questions about earnings season.
            "What comes next after Q3 results were already digested",  # uses "comes next", not "is/'s next"
            # Real breaking story with similar tokens but wrong shape.
            "Fed cuts rates 25bp; Powell signals more",
            "MU earnings beat: what we learned",
            # "Next earnings" without the "what's/is next after" bridge.
            "Nvidia next earnings call set for August 28",
        ):
            hit, name = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on whats-next-after pattern: {t!r} (name={name})"
            )

    def test_why_just_moved_does_not_catch_forward_looking(self):
        """The why-just-moved pattern requires an adverb between "Stock" and
        a PAST-TENSE verb — a forward-looking headline that uses the same
        verbs as nouns ("Stock Pop Could Continue") or present-tense
        ("Stock Slumps Today") must NOT match. Pins the discriminator."""
        for t in (
            # Forward-looking nouns — no past-tense verb after the adverb.
            "Why MU Stock Pop Could Continue",
            "Why Nvidia Stock Surge May Be Sustainable",
            "Why Tesla Stock Drop Is Likely Overdone",
            "Why AMD Stock Jump May Be the Start",
            # Present-tense verbs — recap-flavoured but a different
            # template the urgency_scorer prompt handles via the staleness
            # rule rather than this past-tense gate.
            "Why Microsoft Stock Slumps Today",
            "Why Intel Stock Soars on Foundry Win",
            # No adverb between "Stock" and verb — defaults to recap_did
            # surface (which requires "Did"), so this should fall through.
            "Why Microsoft Stock Drop Continues This Week",
        ):
            hit, _ = alert_agent._looks_like_recap_template({"title": t})
            assert not hit, (
                f"false-positive on forward-looking / non-past-tense: {t!r}"
            )


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

    def test_earnings_tomorrow_seo_mill_end_to_end(self, monkeypatch):
        """End-to-end: a FinancialContent SEO-mill 'Reports Earnings Tomorrow:
        What To Expect' row alongside a real urgent must (a) suppress the SEO
        row without a Discord push, (b) fire on the real urgent, (c) mark
        BOTH alerted. Pins the live failure-mode the new gate targets — DECK
        / SCVL fired BREAKING pushes from this template on 2026-05-20 with
        zero portfolio relevance (`score_source='ml'`, urgency head over-
        scored), exactly the noise this gate suppresses."""
        spy = _StoreSpy()
        real = _row(_id="real",
                    title="MU earnings blow past Q3 estimates sharply",
                    source="reuters", ai_score=9.5)
        seo = _row(_id="seo",
                   title="FinancialContent - Shoe Carnival ( SCVL ) "
                         "Reports Earnings Tomorrow : What To Expect",
                   source="GDELT/markets.financialcontent.com")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK",
                            "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU"
                          ) as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([real, seo], spy)
        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        # The real headline IS in the prompt; the SEO mill row is NOT.
        assert "MU earnings blow past Q3 estimates" in prompt
        assert "Reports Earnings Tomorrow" not in prompt
        assert "FinancialContent" not in prompt
        # Both ids marked: real on send-success, SEO unconditionally by gate.
        assert sorted(spy.marked) == ["real", "seo"]

    def test_new_is_buy_after_pattern_end_to_end(self, monkeypatch):
        """End-to-end pin for the new is_buy_after fingerprint: the live
        failure-case title from 2026-05-21 alert_recency.db must (a) be
        suppressed without a Discord push, (b) be marked alerted so it
        exits the urgent queue. Falsifies the live evidence: if this gate
        regresses the analyst will receive the same noisy post-event
        valuation-question push again on the next NVDA earnings cycle."""
        spy = _StoreSpy()
        row = _row(
            _id="ibA",
            title="Is Nvidia a Buy After Their Latest Earnings Report?",
            source="yfinance/Motley Fool",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["ibA"], (
            "is_buy_after recap not marked alerted — would re-fetch every "
            "20s and fire BREAKING on next batch"
        )

    def test_new_why_is_pct_since_pattern_end_to_end(self, monkeypatch):
        """End-to-end pin for the new why_is_pct_since fingerprint: the live
        failure-case title from 2026-05-21 alert_recency.db must be
        suppressed and marked alerted. Same shape as the is_buy_after
        end-to-end test — both target real BREAKING pushes that fired."""
        spy = _StoreSpy()
        row = _row(
            _id="wiP",
            title="Why Is AGNC Investment (AGNC) Down 7.2% Since Last Earnings Report?",
            source="GoogleNews/Zacks",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["wiP"]

    def test_new_why_stock_is_after_pattern_end_to_end(self, monkeypatch):
        """End-to-end pin for the new why_stock_is_after fingerprint: the live
        failure-case title from 2026-05-21 NVDA-earnings-night articles.db
        urgency=2 set must (a) be suppressed without a Discord push, (b) be
        marked alerted so it exits the urgent queue. Falsifies the live
        evidence: a regression here means the analyst will see "Why X Stock
        Is Barely Moving After Earnings" fire BREAKING twice (Barron's +
        MSN syndication) on the next earnings night, as it did on
        2026-05-21 (10:37:16Z + 10:50:41Z)."""
        spy = _StoreSpy()
        row = _row(
            _id="wSI",
            title="Why Nvidia Stock Is Barely Moving After Earnings Crushed Expectations - Barron's",
            source="GN: Nvidia",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([row], spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == ["wSI"], (
            "why_stock_is_after recap not marked alerted — would re-fetch "
            "every 20s and fire BREAKING on next batch"
        )

    def test_new_patterns_pre_floor_via_urgency_scorer_ssot(self):
        """Both new patterns are SSOT-shared with `watchers.urgency_scorer`
        (which imports `_looks_like_recap_template`); the urgency_scorer
        pre-floor path treats matching titles as noise (score 0.01, urgency 0)
        without calling Sonnet. Pin both patterns reach that path by asserting
        the imported helper returns the new fingerprint names — a future
        local fork that breaks SSOT would fail this guard."""
        from watchers import urgency_scorer as us
        # The urgency_scorer module imports _looks_like_recap_template from
        # alert_agent (the SSOT). If this import ever forks to a local copy,
        # the assertion below catches it.
        assert us._looks_like_recap_template is alert_agent._looks_like_recap_template
        hit_ib, name_ib = us._looks_like_recap_template(
            {"title": "Is Nvidia a Buy After Their Latest Earnings Report?"}
        )
        hit_wp, name_wp = us._looks_like_recap_template(
            {"title": "Why Is AGNC Down 7.2% Since Last Earnings Report?"}
        )
        assert hit_ib and name_ib == "is_buy_after"
        assert hit_wp and name_wp == "why_is_pct_since"

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
