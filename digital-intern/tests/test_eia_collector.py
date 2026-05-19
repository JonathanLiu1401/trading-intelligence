"""Unit tests for collectors.eia_collector.

Fully mocked — no network, isolated dedup DB. Mirrors the contract enforced
by test_ecb_press_collector.py / test_boj_press_collector.py:

  * standard {title, link, summary, published, source} shape
  * ``source`` column = the short feed name ("eia_today" / "eia_press"),
    not the URL — dashboards / briefing group on this value
  * dedup: a second collect() returns [] even though the underlying feed
    is unchanged (daemon re-runs must not re-fire the same headline)
  * intra-run dedup collapses identical entries inside one fetch
  * graceful [] on network failure / non-200 (one bad fetch never aborts
    the daemon worker loop, mirroring fed_press_collector behaviour)
  * malformed entries (missing title / link) are dropped, not crashed-on
"""
from __future__ import annotations

import pytest

from collectors import eia_collector as ec


class _FakeResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_EIA_TODAY_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>EIA Today in Energy</title>
<link>https://www.eia.gov/todayinenergy/</link>
<description>Test feed</description>

<item>
  <title>U.S. crude oil inventories fell by 4.5 million barrels last week</title>
  <link>https://www.eia.gov/todayinenergy/detail.php?id=12345</link>
  <description>Weekly Petroleum Status Report.</description>
  <pubDate>Wed, 14 May 2026 10:30:00 -0400</pubDate>
</item>

<item>
  <title>Natural gas spot prices rose sharply in the Northeast</title>
  <link>https://www.eia.gov/todayinenergy/detail.php?id=12346</link>
  <description>Cold snap drove demand higher.</description>
  <pubDate>Tue, 13 May 2026 10:30:00 -0400</pubDate>
</item>

<item>
  <title></title>
  <link>https://www.eia.gov/todayinenergy/empty</link>
  <description>This entry has an empty title and must be dropped.</description>
  <pubDate>Mon, 12 May 2026 10:30:00 -0400</pubDate>
</item>

</channel></rss>"""


_EIA_PRESS_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>EIA Press</title>
<link>https://www.eia.gov/pressroom/</link>
<description>Test feed</description>

<item>
  <title>EIA releases Short-Term Energy Outlook for May 2026</title>
  <link>https://www.eia.gov/pressroom/releases/press516.php</link>
  <description>STEO release.</description>
  <pubDate>Tue, 13 May 2026 09:00:00 -0400</pubDate>
</item>

</channel></rss>"""


@pytest.fixture(autouse=True)
def _isolate_dedup_db(tmp_path, monkeypatch):
    """Redirect eia_collector.DB_PATH so dedup never touches the shared
    production seen_articles.db. The collector opens its sqlite connection
    inside _ensure_db() (called from collect_eia), so patching the
    module-level DB_PATH before the first call is sufficient."""
    monkeypatch.setattr(ec, "DB_PATH", tmp_path / "seen_articles.db")


def _mock_eia(monkeypatch, *, today=_EIA_TODAY_XML, press=_EIA_PRESS_XML,
              status=200, boom=False):
    def _get(url, **kwargs):
        if boom:
            raise ConnectionError("network down")
        if "todayinenergy" in url:
            return _FakeResp(status_code=status, content=today)
        if "press_rss" in url:
            return _FakeResp(status_code=status, content=press)
        return _FakeResp(status_code=404, content=b"")

    monkeypatch.setattr(ec.requests, "get", _get)


def test_standard_shape_and_per_feed_source(monkeypatch):
    _mock_eia(monkeypatch)
    arts = ec.collect_eia()
    # Two well-formed today items + one press item; empty-title one dropped.
    assert len(arts) == 3
    sources = {a["source"] for a in arts}
    # Per-feed source labels — daemon dashboard groups on this.
    assert sources == {"eia_today", "eia_press"}
    for a in arts:
        assert set(a) >= {"title", "link", "summary", "published", "source"}
        assert a["title"].strip(), "title was kept empty — daemon would drop it"
        assert a["link"].startswith("https://www.eia.gov/")
    titles = {a["title"] for a in arts}
    assert "U.S. crude oil inventories fell by 4.5 million barrels last week" in titles
    assert "EIA releases Short-Term Energy Outlook for May 2026" in titles


def test_dedup_across_runs(monkeypatch):
    """Two consecutive collect_eia() calls on the same feeds -> first emits,
    second is empty. Mirrors the daemon's expectation that re-runs don't
    re-fire the same headline through the score / alert pipeline."""
    _mock_eia(monkeypatch)
    first = ec.collect_eia()
    assert len(first) == 3
    second = ec.collect_eia()
    assert second == []


def test_intra_run_dedup_on_identical_entries(monkeypatch):
    """The same item appearing twice within a single fetch (e.g. a transient
    feed glitch, or duplicated across the today/press feeds) must collapse
    to one article — the seen_in_run set."""
    dup_xml = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>x</title><link>x</link><description>x</description>
<item><title>Crude up</title><link>https://eia.gov/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 -0400</pubDate></item>
<item><title>Crude up</title><link>https://eia.gov/x</link>
<description>d</description><pubDate>Fri, 15 May 2026 10:00:00 -0400</pubDate></item>
</channel></rss>"""
    _mock_eia(monkeypatch, today=dup_xml, press=b"<rss/>")
    arts = ec.collect_eia()
    assert len(arts) == 1


def test_network_error_returns_empty(monkeypatch):
    _mock_eia(monkeypatch, boom=True)
    assert ec.collect_eia() == []


def test_non_200_returns_empty(monkeypatch):
    _mock_eia(monkeypatch, status=503, today=b"<html>down</html>",
              press=b"<html>down</html>")
    assert ec.collect_eia() == []


def test_collect_alias_matches_named_function():
    # Daemon registration / introspection sometimes uses the bare ``collect``
    # alias; keep the two pointing at the same function so callers stay in sync.
    assert ec.collect is ec.collect_eia
