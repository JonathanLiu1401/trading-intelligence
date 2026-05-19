"""Unit tests for collectors.boj_press_collector.

Fully mocked — no network, isolated dedup DB. Asserts the standard collector
dict shape and the load-bearing behaviours the daemon relies on:

  * standard {title, link, summary, published, source} shape
  * ``source`` column = "boj_press" (not the URL or feed name) — dashboards
    and the briefing's source-grouping rely on this stable short name
  * dedup: a second run returns [] even though the underlying feed is unchanged
  * graceful [] on network failure (one bad fetch never aborts the daemon
    worker loop, mirroring fed_press_collector / ecb_press_collector behaviour)
  * malformed entries (missing title / link) are dropped, not crashed-on
"""
from __future__ import annotations

import pytest

from collectors import boj_press_collector as bc


class _FakeResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# Hand-rolled minimal RSS 2.0 sample mirroring the real BoJ whatsnew.xml shape
# (title, link, description/summary, pubDate). Three entries; one has missing
# title and must be dropped without crashing the collector.
_BOJ_RSS_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>Bank of Japan</title>
<link>https://www.boj.or.jp/en/</link>
<description>Test feed</description>

<item>
  <title>Summary of Opinions at the Monetary Policy Meeting on April 27 and 28, 2026</title>
  <link>https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260428.htm</link>
  <description>BoJ MPM Summary of Opinions release.</description>
  <pubDate>Fri, 15 May 2026 08:50:00 +0900</pubDate>
</item>

<item>
  <title>Speech by Deputy Governor HIMINO at the Japan Society of Monetary Economics</title>
  <link>https://www.boj.or.jp/en/about/press/koen_2026/ko260514a.htm</link>
  <description>Speech on the singleness of money and the role of central banks.</description>
  <pubDate>Wed, 14 May 2026 14:30:00 +0900</pubDate>
</item>

<item>
  <title></title>
  <link>https://www.boj.or.jp/en/empty</link>
  <description>This entry has an empty title and must be dropped.</description>
  <pubDate>Tue, 13 May 2026 09:00:00 +0900</pubDate>
</item>

</channel></rss>"""


@pytest.fixture(autouse=True)
def _isolate_dedup_db(tmp_path, monkeypatch):
    """Redirect boj_press_collector.DB_PATH so dedup never touches production.

    The collector opens a sqlite connection at module-import time only inside
    ``_ensure_db()`` (called from ``collect_boj_press``), so patching the
    module-level ``DB_PATH`` here, *before* the first call, is sufficient.
    """
    monkeypatch.setattr(bc, "DB_PATH", tmp_path / "seen_articles.db")


def _mock_boj(monkeypatch, *, payload=_BOJ_RSS_XML, status=200, boom=False):
    def _get(url, **kwargs):
        if boom:
            raise ConnectionError("network down")
        return _FakeResp(status_code=status, content=payload)

    monkeypatch.setattr(bc.requests, "get", _get)


def test_standard_shape_and_source_label(monkeypatch):
    _mock_boj(monkeypatch)
    arts = bc.collect_boj_press()
    # Two well-formed items; the empty-title one is dropped.
    assert len(arts) == 2
    for a in arts:
        # Standard collector dict — daemon._ingest requires these keys.
        assert set(a) >= {"title", "link", "summary", "published", "source"}
        # The `source` column on articles.db must be the short name, not the URL
        # or the raw feed string — dashboards / briefing group on this value.
        assert a["source"] == "boj_press"
        assert a["title"].strip(), "title was kept empty — daemon would drop it"
        assert a["link"].startswith("https://www.boj.or.jp/")
    titles = {a["title"] for a in arts}
    assert "Summary of Opinions at the Monetary Policy Meeting on April 27 and 28, 2026" in titles
    assert "Speech by Deputy Governor HIMINO at the Japan Society of Monetary Economics" in titles


def test_dedup_across_runs(monkeypatch):
    """Two consecutive collect() calls on the same feed -> first emits, second
    is empty. Mirrors the daemon's expectation that re-runs don't re-fire the
    same headline through the score / alert pipeline."""
    _mock_boj(monkeypatch)
    first = bc.collect_boj_press()
    assert len(first) == 2
    second = bc.collect_boj_press()
    assert second == []


def test_intra_run_dedup_on_identical_entries(monkeypatch):
    """The same item appearing twice within a single fetch (e.g. a transient
    feed glitch) must collapse to one article — the seen_in_run set."""
    dup_xml = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>x</title><link>x</link><description>x</description>
<item><title>Same MPM speech</title><link>https://boj/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 +0900</pubDate></item>
<item><title>Same MPM speech</title><link>https://boj/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 +0900</pubDate></item>
</channel></rss>"""
    _mock_boj(monkeypatch, payload=dup_xml)
    arts = bc.collect_boj_press()
    assert len(arts) == 1


def test_network_error_returns_empty(monkeypatch):
    _mock_boj(monkeypatch, boom=True)
    assert bc.collect_boj_press() == []


def test_non_200_returns_empty(monkeypatch):
    _mock_boj(monkeypatch, status=503, payload=b"<html>down</html>")
    assert bc.collect_boj_press() == []


def test_collect_alias_matches_named_function():
    # Daemon registration / introspection sometimes uses the bare ``collect``
    # alias; keep the two pointing at the same function so callers stay in sync.
    assert bc.collect is bc.collect_boj_press
