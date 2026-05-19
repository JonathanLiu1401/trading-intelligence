"""Unit tests for collectors.cisa_kev_collector.

Fully mocked — no network, isolated dedup DB. Asserts the standard collector
dict shape and the load-bearing behaviours the daemon relies on:

  * standard {title, link, summary, published, source} shape
  * ``source`` column = "CISA/KEV" — dashboards group on this stable label
  * vendor → ticker mapping surfaces the ticker in the title so the heuristic
    scorer + ArticleNet keyword features can latch onto it
  * ransomware-flagged adds get a [RANSOMWARE] prefix the alert path can spot
  * dedup: a second run returns [] (KEV is polled hourly; the file changes
    rarely, so re-emitting the same row every hour would spam the pipeline)
  * graceful [] on network failure (one bad fetch never aborts the daemon)
  * malformed entries (missing cveID / vendorProject) are dropped, not crashed-on
  * max_items cap (default 50) prevents a fresh seen_articles.db from
    blasting 1500 historical CVEs into the pipeline on first run
"""
from __future__ import annotations

import json

import pytest

from collectors import cisa_kev_collector as kev


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


_SAMPLE_PAYLOAD = {
    "catalogVersion": "2026.05.15",
    "count": 4,
    "vulnerabilities": [
        {
            "cveID": "CVE-2026-42897",
            "vendorProject": "Microsoft",
            "product": "Exchange Server",
            "vulnerabilityName": "Exchange XSS in OWA",
            "shortDescription": "Cross-site scripting in OWA.",
            "knownRansomwareCampaignUse": "Unknown",
            "dateAdded": "2026-05-15",
            "dueDate": "2026-05-29",
        },
        {
            "cveID": "CVE-2026-31337",
            "vendorProject": "Fortinet",
            "product": "FortiOS",
            "vulnerabilityName": "FortiOS auth bypass",
            "shortDescription": "Authentication bypass in admin UI.",
            "knownRansomwareCampaignUse": "Known",
            "dateAdded": "2026-05-14",
            "dueDate": "2026-05-28",
        },
        {
            # Older entry — sort order check.
            "cveID": "CVE-2025-99999",
            "vendorProject": "SomePrivateVendor",
            "product": "Widget",
            "vulnerabilityName": "Old issue",
            "shortDescription": "Old.",
            "knownRansomwareCampaignUse": "Unknown",
            "dateAdded": "2025-01-01",
            "dueDate": "2025-01-15",
        },
        {
            # Malformed — no cveID; must be dropped.
            "cveID": "",
            "vendorProject": "Microsoft",
            "product": "Windows",
            "vulnerabilityName": "Bad row",
        },
    ],
}


@pytest.fixture(autouse=True)
def _isolate_dedup_db(tmp_path, monkeypatch):
    monkeypatch.setattr(kev, "DB_PATH", tmp_path / "seen_articles.db")


def _mock(monkeypatch, *, payload=_SAMPLE_PAYLOAD, status=200, boom=False):
    def _get(url, **kwargs):
        if boom:
            raise ConnectionError("network down")
        return _FakeResp(status_code=status, payload=payload)

    monkeypatch.setattr(kev.requests, "get", _get)


def test_standard_shape_and_source_label(monkeypatch):
    _mock(monkeypatch)
    arts = kev.collect_cisa_kev()
    # Three well-formed entries; the empty-cveID row is dropped.
    assert len(arts) == 3
    for a in arts:
        # Standard collector dict — daemon._ingest requires these keys.
        assert set(a) >= {"title", "link", "summary", "published", "source"}
        assert a["source"] == "CISA/KEV"
        assert a["title"].strip()
        assert a["link"].startswith("https://nvd.nist.gov/vuln/detail/CVE-")


def test_vendor_ticker_tag_in_title(monkeypatch):
    """Microsoft → $MSFT, Fortinet → $FTNT — must appear in the title so the
    heuristic scorer's keyword features fire on it. Without this tag the
    article looks generic to the pipeline."""
    _mock(monkeypatch)
    arts = kev.collect_cisa_kev()
    msft = next(a for a in arts if "Microsoft" in a["title"])
    ftnt = next(a for a in arts if "Fortinet" in a["title"])
    assert "$MSFT" in msft["title"]
    assert "$FTNT" in ftnt["title"]
    # Unmapped private vendors must NOT get a stray $ symbol.
    private = next(a for a in arts if "SomePrivateVendor" in a["title"])
    assert "$" not in private["title"]


def test_ransomware_prefix(monkeypatch):
    """``knownRansomwareCampaignUse == "Known"`` must surface as [RANSOMWARE]
    so the alert path can spot it (these are the highest-urgency KEV adds)."""
    _mock(monkeypatch)
    arts = kev.collect_cisa_kev()
    ftnt = next(a for a in arts if "Fortinet" in a["title"])
    msft = next(a for a in arts if "Microsoft" in a["title"])
    assert ftnt["title"].startswith("[RANSOMWARE]")
    assert not msft["title"].startswith("[RANSOMWARE]")


def test_summary_carries_ticker_and_due_date(monkeypatch):
    _mock(monkeypatch)
    arts = kev.collect_cisa_kev()
    msft = next(a for a in arts if "Microsoft" in a["title"])
    assert "MSFT" in msft["summary"]
    assert "2026-05-29" in msft["summary"]


def test_dedup_across_runs(monkeypatch):
    """Two consecutive collect() calls on the same feed -> first emits, second
    is empty. KEV polls hourly but the file changes ~daily, so re-emitting
    would flood the pipeline."""
    _mock(monkeypatch)
    first = kev.collect_cisa_kev()
    assert len(first) == 3
    second = kev.collect_cisa_kev()
    assert second == []


def test_network_error_returns_empty(monkeypatch):
    _mock(monkeypatch, boom=True)
    assert kev.collect_cisa_kev() == []


def test_non_200_returns_empty(monkeypatch):
    _mock(monkeypatch, status=503)
    assert kev.collect_cisa_kev() == []


def test_max_items_caps_yield(monkeypatch):
    """A fresh seen_articles.db on first run must not blast all 1500 historical
    KEV entries into the pipeline at once. max_items is the per-cycle ceiling."""
    bulk = {
        "vulnerabilities": [
            {
                "cveID": f"CVE-2026-{i:05d}",
                "vendorProject": "Microsoft",
                "product": "Windows",
                "vulnerabilityName": f"Issue {i}",
                "shortDescription": "test",
                "knownRansomwareCampaignUse": "Unknown",
                "dateAdded": f"2026-05-{(i % 28) + 1:02d}",
                "dueDate": "2026-06-01",
            }
            for i in range(120)
        ]
    }
    _mock(monkeypatch, payload=bulk)
    arts = kev.collect_cisa_kev(max_items=10)
    assert len(arts) == 10


def test_sorted_newest_first(monkeypatch):
    """The cap surfaces newest adds — the 2025 row must come last (or not at
    all once the cap kicks in). Verifies the sort step before the cap."""
    _mock(monkeypatch)
    arts = kev.collect_cisa_kev()
    # First emitted should be the 2026-05-15 row (Microsoft), last should be
    # the 2025-01-01 row (SomePrivateVendor).
    assert "Microsoft" in arts[0]["title"]
    assert "SomePrivateVendor" in arts[-1]["title"]
