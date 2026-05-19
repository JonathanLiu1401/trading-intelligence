"""Unit tests for collectors.cftc_press_collector.

Fully mocked — no network, isolated dedup DB. Asserts the standard collector
dict shape and the load-bearing behaviours the daemon relies on:

  * standard {title, link, summary, published, source} shape
  * ``source`` column = "cftc_press" (not the URL or feed name) — dashboards
    and the briefing's source-grouping rely on this stable short name
  * dedup: a second run returns [] even though the underlying feed is unchanged
  * graceful [] on network failure (one bad fetch never aborts the daemon
    worker loop, mirroring fed_press_collector / boj_press_collector behaviour)
  * malformed entries (missing title / link) are dropped, not crashed-on
"""
from __future__ import annotations

import pytest

from collectors import cftc_press_collector as cc


class _FakeResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# Hand-rolled minimal RSS 2.0 sample mirroring the real CFTC PressReleases.xml
# shape (title, link, description/summary, pubDate). Three entries; one has
# missing title and must be dropped without crashing the collector.
_CFTC_RSS_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>CFTC Press Releases</title>
<link>https://www.cftc.gov/</link>
<description>Test feed</description>

<item>
  <title>CFTC Charges Major Crypto Exchange With Failure to Register</title>
  <link>https://www.cftc.gov/PressRoom/PressReleases/9001-26</link>
  <description>Enforcement action against a US-listed crypto venue for offering unregistered futures.</description>
  <pubDate>Fri, 15 May 2026 14:00:00 -0400</pubDate>
</item>

<item>
  <title>Commitments of Traders Update -- Week of May 12, 2026</title>
  <link>https://www.cftc.gov/MarketReports/CommitmentsOfTraders/2026-05-12</link>
  <description>Weekly COT release with institutional positioning across futures complexes.</description>
  <pubDate>Wed, 14 May 2026 15:30:00 -0400</pubDate>
</item>

<item>
  <title></title>
  <link>https://www.cftc.gov/empty</link>
  <description>This entry has an empty title and must be dropped.</description>
  <pubDate>Tue, 13 May 2026 09:00:00 -0400</pubDate>
</item>

</channel></rss>"""


@pytest.fixture(autouse=True)
def _isolate_dedup_db(tmp_path, monkeypatch):
    """Redirect cftc_press_collector.DB_PATH so dedup never touches production.

    The collector opens a sqlite connection at module-import time only inside
    ``_ensure_db()`` (called from ``collect_cftc_press``), so patching the
    module-level ``DB_PATH`` here, *before* the first call, is sufficient.
    """
    monkeypatch.setattr(cc, "DB_PATH", tmp_path / "seen_articles.db")


def _mock_cftc(monkeypatch, *, payload=_CFTC_RSS_XML, status=200, boom=False):
    def _get(url, **kwargs):
        if boom:
            raise ConnectionError("network down")
        return _FakeResp(status_code=status, content=payload)

    monkeypatch.setattr(cc.requests, "get", _get)


def test_standard_shape_and_source_label(monkeypatch):
    _mock_cftc(monkeypatch)
    arts = cc.collect_cftc_press()
    # Two well-formed items; the empty-title one is dropped.
    assert len(arts) == 2
    for a in arts:
        # Standard collector dict — daemon._ingest requires these keys.
        assert set(a) >= {"title", "link", "summary", "published", "source"}
        # The `source` column on articles.db must be the short name, not the URL
        # or the raw feed string — dashboards / briefing group on this value.
        assert a["source"] == "cftc_press"
        assert a["title"].strip(), "title was kept empty — daemon would drop it"
        assert a["link"].startswith("https://www.cftc.gov/")
    titles = {a["title"] for a in arts}
    assert "CFTC Charges Major Crypto Exchange With Failure to Register" in titles
    assert "Commitments of Traders Update -- Week of May 12, 2026" in titles


def test_dedup_across_runs(monkeypatch):
    """Two consecutive collect() calls on the same feed -> first emits, second
    is empty. Mirrors the daemon's expectation that re-runs don't re-fire the
    same headline through the score / alert pipeline."""
    _mock_cftc(monkeypatch)
    first = cc.collect_cftc_press()
    assert len(first) == 2
    second = cc.collect_cftc_press()
    assert second == []


def test_intra_run_dedup_on_identical_entries(monkeypatch):
    """The same item appearing twice within a single fetch (e.g. a transient
    feed glitch) must collapse to one article — the seen_in_run set."""
    dup_xml = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>x</title><link>x</link><description>x</description>
<item><title>Same CFTC enforcement</title><link>https://cftc/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 -0400</pubDate></item>
<item><title>Same CFTC enforcement</title><link>https://cftc/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 -0400</pubDate></item>
</channel></rss>"""
    _mock_cftc(monkeypatch, payload=dup_xml)
    arts = cc.collect_cftc_press()
    assert len(arts) == 1


def test_network_error_returns_empty(monkeypatch):
    _mock_cftc(monkeypatch, boom=True)
    assert cc.collect_cftc_press() == []


def test_non_200_returns_empty(monkeypatch):
    _mock_cftc(monkeypatch, status=503, payload=b"<html>down</html>")
    assert cc.collect_cftc_press() == []


def test_collect_alias_matches_named_function():
    # Daemon registration / introspection sometimes uses the bare ``collect``
    # alias; keep the two pointing at the same function so callers stay in sync.
    assert cc.collect is cc.collect_cftc_press
