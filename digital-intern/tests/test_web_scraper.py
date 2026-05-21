"""collectors/web_scraper.py — URL filtering + article extraction.

The scraper feeds ~60 sites into the ingest path every 60s and had no direct
coverage. The two pure helpers carry all the real logic; a regression in
either silently changes what enters the pipeline (junk links, or real
articles dropped):

  * `_is_article_url` — the SKIP_PATTERNS denylist and the path
    length/depth heuristic decide what counts as an article.
  * `_extract_articles` — anchor → article dict shaping: 15-char title
    floor, relative-URL resolution against base, per-page dedup, and the
    `source = "scraped/<netloc>"` tag that ml/features.py credibility
    scoring keys on.
"""
from __future__ import annotations

from urllib.parse import urlparse

from collectors import web_scraper


class TestIsArticleUrl:
    def test_skip_patterns_rejected(self):
        for bad in (
            "https://site.com/about/team",
            "https://site.com/login/",
            "https://site.com/privacy/policy",
            "https://site.com/careers/openings",
            "https://site.com/newsletter/subscribe/",
        ):
            assert web_scraper._is_article_url(bad) is False, bad

    def test_valid_deep_article_url_accepted(self):
        url = "https://site.com/markets/2026/big-semiconductor-story"
        assert web_scraper._is_article_url(url) is True

    def test_short_path_rejected(self):
        """path '/x' is len 2 — below the >10 floor."""
        assert web_scraper._is_article_url("https://site.com/x") is False

    def test_shallow_path_rejected(self):
        """A single-segment path (one '/') fails the >=2 depth rule even when
        it is long enough."""
        url = "https://site.com/news-page-here-and-long"
        assert urlparse(url).path.count("/") == 1
        assert web_scraper._is_article_url(url) is False


class TestExtractArticles:
    def test_extracts_filters_and_dedupes(self):
        html = """
        <html><body>
          <a href="/markets/2026/big-story-headline">A genuinely long article headline here</a>
          <a href="/markets/2026/big-story-headline">Duplicate link, different text but same URL</a>
          <a href="/about/team">About Us Page Link That Is Long Enough</a>
          <a href="/x">too short</a>
          <a href="/markets/2026/second-real-story">Second real story with a sufficiently long title</a>
        </body></html>
        """
        arts = web_scraper._extract_articles(html, "https://example.com/markets")

        urls = [a["link"] for a in arts]
        # dedup: the repeated href appears exactly once
        assert urls.count("https://example.com/markets/2026/big-story-headline") == 1
        # skip-pattern (/about/) and short-path (/x) links are excluded
        assert all("/about/" not in u for u in urls)
        assert "https://example.com/x" not in urls
        # both genuine articles survived
        assert "https://example.com/markets/2026/big-story-headline" in urls
        assert "https://example.com/markets/2026/second-real-story" in urls

    def test_relative_href_resolved_against_base_and_source_tag(self):
        html = (
            '<a href="/finance/2026/oracle-earnings-beat-report">'
            "Oracle earnings beat the Street consensus handily</a>"
        )
        arts = web_scraper._extract_articles(
            html, "https://www.example.com/finance/"
        )
        assert len(arts) == 1
        a = arts[0]
        assert a["link"] == "https://www.example.com/finance/2026/oracle-earnings-beat-report"
        assert a["source"] == "scraped/www.example.com"
        assert a["published"] == ""

    def test_short_anchor_text_rejected(self):
        """Anchor text below 15 chars is dropped even with a valid URL —
        prevents nav-chrome ('Home', 'More') polluting the feed."""
        html = (
            '<a href="/markets/2026/a-perfectly-valid-article-url">Read</a>'
        )
        arts = web_scraper._extract_articles(html, "https://example.com/")
        assert arts == []

    def test_title_truncated_to_200_chars(self):
        long_title = "X" * 400
        html = f'<a href="/markets/2026/very-long-title-article">{long_title}</a>'
        arts = web_scraper._extract_articles(html, "https://example.com/")
        assert len(arts) == 1
        assert len(arts[0]["title"]) == 200

    def test_malformed_html_returns_empty_list(self):
        """A parser failure must degrade to [] (the worker keeps running),
        never raise into the daemon thread."""
        assert web_scraper._extract_articles("<<<not html", "https://x.com/") == []
        assert web_scraper._extract_articles("", "https://x.com/") == []


class TestQuoteWidgetRejection:
    """Yahoo/Bloomberg list pages embed a live ticker-tape sidebar whose every
    <a href="/quote/TICKER"> entry is a price string with no spaces. Each price
    poll mints a new unique title → unbounded fake breaking news (3,476/5,847
    live scraped rows; one fired a real 🚨 BREAKING push). _looks_like_quote_widget
    must reject these without ever dropping a real headline."""

    # Live-observed quote-tape entries — every price tick is a distinct title.
    JUNK_TITLES = [
        "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
        "NVDANVIDIA Corporation225.32-10.42(-4.42%)",
        "NVDANVIDIA Corporation226.97-8.77(-3.72%)",
        "NOKNokia Oyj13.98-0.48(-3.35%)",
        "ETH-USDEthereum USD2,169.83",
    ]
    # Real headlines, including ones with $/%/comma numbers that a naive
    # "contains a number" filter would wrongly drop.
    REAL_TITLES = [
        "The Top 5 Analyst Questions From Motorola Solutions's Q1 Earnings Call",
        "Oil jumps as Trump warns 'Clock is Ticking' for Iran",
        "Stock futures fall after record-setting week for Wall Street",
        "Fed holds rates steady at 4.25%-4.50% as expected",
        "Nvidia Q3 revenue rises 22% to $35.1 billion, beats estimates",
        "Tesla stock up 3.2% after delivery beat",
        "S&P 500 closes at 5,123.41 record high",
        "Apple unveils iPhone 16 with A18 chip",
    ]

    def test_price_glue_titles_rejected(self):
        for t in self.JUNK_TITLES:
            assert web_scraper._looks_like_quote_widget(t, "") is True, t

    def test_real_headlines_accepted(self):
        for t in self.REAL_TITLES:
            assert web_scraper._looks_like_quote_widget(t, "https://x.com/a/b") is False, t

    def test_quote_landing_url_rejected_even_without_price_in_text(self):
        # Anchor text alone is clean (no glued price) — the /quote/ landing
        # path is the second independent fingerprint.
        assert web_scraper._looks_like_quote_widget(
            "NVIDIA Corporation overview", "https://finance.yahoo.com/quote/NVDA/"
        ) is True
        assert web_scraper._looks_like_quote_widget(
            "Nokia Oyj summary page", "https://finance.yahoo.com/quote/NOK"
        ) is True

    def test_real_quote_scoped_article_url_accepted(self):
        # A genuine article *under* a quote path must still pass — the URL
        # rule is anchored to end-of-path so deeper paths are not caught.
        assert web_scraper._looks_like_quote_widget(
            "Nvidia beats Q3 estimates on data-center demand",
            "https://finance.yahoo.com/quote/NVDA/news/nvidia-beats-q3-123",
        ) is False

    def test_extract_filters_every_price_tick_keeps_real(self):
        # Sidebar: 3 distinct NVDA price ticks (distinct hrefs+titles, each
        # would have minted its own article id) + 1 real article.
        html = """
        <html><body>
          <a href="/quote/NVDA/?p=1">NVDANVIDIA Corporation227.13-8.61(-3.65%)</a>
          <a href="/quote/NVDA/?p=2">NVDANVIDIA Corporation226.06-9.68(-4.11%)</a>
          <a href="/quote/NOK/?p=3">NOKNokia Oyj13.98-0.48(-3.35%)</a>
          <a href="/news/2026/nvidia-earnings-beat-the-street-q3">Nvidia tops Q3 estimates on AI demand, raises guidance</a>
        </body></html>
        """
        arts = web_scraper._extract_articles(html, "https://finance.yahoo.com/news/")
        titles = [a["title"] for a in arts]
        assert titles == ["Nvidia tops Q3 estimates on AI demand, raises guidance"]
        assert not any(web_scraper._QW_PRICE_GLUE.search(t) for t in titles)

    # ── Quote-listing share-card fingerprint (_QW_LISTING, lockstep) ────────
    # "$NVIDIA (NVDA.US)$ - Moomoo" — a Moomoo/Futu/Webull quote share-card
    # landing page. web_scraper does not normally ingest these (they arrive via
    # the Google News collector), but the third fingerprint is carried here
    # byte-identically so the three gates stay in lockstep.
    QUOTE_LISTING_TITLES = [
        "$NVIDIA (NVDA.US)$ - Moomoo",
        "$Tesla (TSLA.US)$ - Moomoo",
        "$Tencent (00700.HK)$ - Futu",
        "$Samsung Electronics (005930.KS)$ - Webull",
        "  $NIO Inc. (NIO.US)$",
    ]

    def test_quote_listing_share_card_rejected(self):
        for t in self.QUOTE_LISTING_TITLES:
            assert web_scraper._looks_like_quote_widget(t, "") is True, t

    def test_real_dollar_ticker_headlines_accepted(self):
        # Real "$TICKER ..." prose / $+paren headlines must still pass — the
        # discriminator is the glued "(SYM.EXCH)$" share-card close.
        for t in (
            "$NVDA breaks out ahead of earnings (NYSE)",
            "$MU upgraded to Buy (price target $150.00)",
            "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
        ):
            assert web_scraper._looks_like_quote_widget(
                t, "https://x.com/a/b") is False, t

    # ── Image-credit pseudo-article (_QW_IMAGE_CREDIT) ──────────────────────
    # The hero-image photo credit on news pages is wrapped inside the
    # article's own <a> link, so the anchor-text fallback in
    # `_extract_articles` picks up the credit string as the article title.
    # Live evidence (2026-05-21 16:30:49Z, alert_recency.db): "Angela Weiss/
    # AFP/Getty Images" fired a real 🚨 BREAKING push from
    # ``scraped/www.bloomberg.com`` — the highest-cred source tier so the
    # authority gate cannot catch it; content type IS the failure. Ingestion
    # gate here drops the credit string at the source.

    IMAGE_CREDIT_TITLES = [
        "Angela Weiss/AFP/Getty Images",                # the exact live noise
        "Tomohiro Ohsumi/Getty Images",                 # 2-word + 1 agency
        "Timorthy A. Clary/AFP/Getty Images",           # initial-bearing
        "Anna Moneymaker/Getty Images",
        "Drew Angerer/AFP/Getty Images",
        "John Smith/Reuters",                            # minimum match
        "Mary Jane Doe/Bloomberg/Getty Images",         # 3-word + 2 agencies
        "  Angela Weiss/AFP/Getty Images  ",             # leading/trailing ws
    ]

    def test_image_credit_titles_rejected(self):
        for t in self.IMAGE_CREDIT_TITLES:
            assert web_scraper._looks_like_quote_widget(t, "") is True, t

    def test_real_headlines_with_agency_names_accepted(self):
        # Real headlines that mention agencies / slashes must SURVIVE — the
        # anchored ^...$ + Title-Case-Name + closed-agency-list trio is the
        # discriminator (real headlines never END with the no-space "/Agency"
        # structure of a photo credit).
        for t in (
            "Reuters reports Q1 earnings beat",
            "Bloomberg: NVDA breaks $200",
            "Getty Images launches new product",
            "AFP Photo: 5 things to know about Q1",
            "MU drops 5%/Yahoo",
            "Stock Market Today: Reuters/AP",
            "Sam Altman/OpenAI says GPT-5 coming",      # OpenAI not in list
            "Reuters/Yahoo Finance reports earnings",   # Yahoo not in list
            "Apple/Microsoft deal closes",
            "AFP/Getty Images launches new service",    # mid-sentence content
            "Nvidia/AMD price war intensifies",
            "Tom Cruise",                                # 2-word name, no /agency
        ):
            assert web_scraper._looks_like_quote_widget(
                t, "https://x.com/a/b") is False, t

    def test_extract_articles_filters_image_credit_anchors(self):
        """End-to-end: a news page where the hero image (and its photo credit)
        is wrapped in the article's own <a> link must NOT produce a fake
        article whose title is the credit string. The exact bloomberg.com
        shape that fired the live BREAKING push at 16:30:49Z 2026-05-21."""
        html = """
        <html><body>
        <a href="/news/articles/2026-05-21/trump-quantum-firms">
          <img src="/photos/x.jpg" alt="Trump signs quantum order">
          Angela Weiss/AFP/Getty Images
        </a>
        <a href="/news/articles/2026-05-21/real-story">
          Trump signs $2B quantum executive order, AI stocks rally
        </a>
        </body></html>
        """
        arts = web_scraper._extract_articles(html, "https://www.bloomberg.com/")
        titles = [a["title"] for a in arts]
        # Credit string must NOT be a "title"; the real story survives.
        assert "Angela Weiss/AFP/Getty Images" not in titles
        assert any("$2B quantum" in t for t in titles), titles
