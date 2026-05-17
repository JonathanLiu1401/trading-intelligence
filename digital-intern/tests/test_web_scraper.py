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
