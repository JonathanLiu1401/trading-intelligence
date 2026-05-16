"""collectors/rss_collector.py — the requests-based fetch + dedup contract.

Two concerns, both previously uncovered (no test_rss_collector.py existed):

1. The unbounded-hang fix. ``_fetch_feed`` used to call
   ``feedparser.parse(url)``, which does its own urllib fetch with **no
   timeout**. rss_worker runs 32 parallel feed fetches every 30s; a single
   hung feed pinned one worker forever. The fix routes through
   ``requests.get(url, timeout=FETCH_TIMEOUT, headers={User-Agent: ...})``
   then ``feedparser.parse(resp.content)``. These tests pin that the
   timeout + UA are actually passed and that an HTTP error degrades to
   ``[]`` (worker keeps running) rather than raising into the daemon thread.

2. The dedup contract. ``collect_rss`` must drop (a) duplicate
   ``(link, title)`` pairs *within a single pass* (the ``seen_in_run`` set,
   e.g. the same story carried by two configured feeds) and (b) articles
   already recorded in ``seen_articles`` *across passes* (the persistent
   SQLite table). The connection-hardening change must not regress either.
"""
from __future__ import annotations

import pytest

from collectors import rss_collector

# A valid minimal RSS 2.0 document with one item. feedparser parses this for
# real (no network, deterministic) — mirrors what requests' .content returns.
_RSS_ONE_ITEM = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test Channel</title>
  <item>
    <title>Hello World Headline</title>
    <link>http://example.com/article-a</link>
    <description>Body text for the article.</description>
    <pubDate>Wed, 14 May 2026 10:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


class _FakeResp:
    def __init__(self, content=b"", raise_exc=None):
        self.content = content
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


def test_fetch_feed_passes_timeout_and_user_agent(monkeypatch):
    """The unbounded-hang fix: requests.get must get the bounded timeout and
    a browser UA (feedparser's default UA is 403'd by many CDN-fronted
    feeds)."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResp(content=_RSS_ONE_ITEM)

    monkeypatch.setattr(rss_collector.requests, "get", fake_get)

    out = rss_collector._fetch_feed(
        {"name": "TestFeed", "url": "http://example.com/rss"}
    )

    assert captured["url"] == "http://example.com/rss"
    assert captured["kwargs"]["timeout"] == rss_collector.FETCH_TIMEOUT
    assert rss_collector.FETCH_TIMEOUT > 0  # genuinely bounded, not 0/None
    headers = captured["kwargs"]["headers"]
    assert "User-Agent" in headers and "Mozilla" in headers["User-Agent"]

    # And the parsed result is well-formed (resp.content → feedparser).
    assert len(out) == 1
    art = out[0]
    assert art["title"] == "Hello World Headline"
    assert art["link"] == "http://example.com/article-a"
    assert art["summary"] == "Body text for the article."
    assert art["source"] == "TestFeed"
    assert "2026" in art["published"]


def test_fetch_feed_returns_empty_on_http_error(monkeypatch):
    """A non-200 (raise_for_status raises) must degrade to [] — the worker
    must never take an exception into the daemon thread."""
    def fake_get(url, **kwargs):
        return _FakeResp(raise_exc=rss_collector.requests.HTTPError("404"))

    monkeypatch.setattr(rss_collector.requests, "get", fake_get)
    assert rss_collector._fetch_feed({"name": "X", "url": "http://x"}) == []


def test_fetch_feed_no_url_is_empty():
    assert rss_collector._fetch_feed({"name": "no-url"}) == []


def test_fetch_feed_network_exception_is_swallowed(monkeypatch):
    """A raised requests exception (timeout, DNS, conn reset) → [] not crash."""
    def boom(url, **kwargs):
        raise rss_collector.requests.ConnectionError("dns fail")

    monkeypatch.setattr(rss_collector.requests, "get", boom)
    assert rss_collector._fetch_feed({"name": "X", "url": "http://x"}) == []


def test_collect_rss_dedups_within_run_and_across_runs(tmp_path, monkeypatch):
    """Two configured feeds carrying the *same* (link, title): the
    ``seen_in_run`` set collapses them to one new article in a single pass,
    and the persistent ``seen_articles`` table makes the next pass return
    zero new articles for the same story."""
    monkeypatch.setattr(rss_collector, "DB_PATH", tmp_path / "seen.db")
    # Two distinct feeds, identical article — exercises the in-run set.
    monkeypatch.setattr(
        rss_collector,
        "_load_sources",
        lambda: {"rss_feeds": [
            {"name": "FeedA", "url": "http://a/rss"},
            {"name": "FeedB", "url": "http://b/rss"},
        ]},
    )
    monkeypatch.setattr(
        rss_collector.requests, "get",
        lambda url, **kw: _FakeResp(content=_RSS_ONE_ITEM),
    )

    first = rss_collector.collect_rss()
    assert len(first) == 1, (
        "same (link,title) from two feeds must dedup to one within the run"
    )
    assert first[0]["title"] == "Hello World Headline"

    second = rss_collector.collect_rss()
    assert second == [], (
        "an article already in seen_articles must not be re-emitted on the "
        "next pass (persistent cross-run dedup)"
    )
