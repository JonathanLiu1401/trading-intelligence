"""Unit tests for collectors.nasdaq_halts_collector.

Fully mocked — no network. Parses a static fixture in the real Nasdaq Trader
feed shape (``ndaq:`` namespace) and asserts the standard collector dict
shape plus the behaviours the daemon relies on: reason-code mapping,
HaltDate/HaltTime ET -> UTC conversion, pubDate fallback, low-signal
filtering, dedup, unique per-event link id, and graceful [] on
network / non-200 / malformed-XML failure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from collectors import nasdaq_halts_collector as nh

# Realistic feed body: namespaced payload fields, one LULD halt (not resumed),
# one news halt that has resumed, one low-signal IPO1, and a dup of the LULD.
_FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/">
 <channel>
  <title>Nasdaq Trading Halts</title>
  <item>
   <title>Trade Halt: EXMP</title>
   <pubDate>Mon, 18 May 2026 18:32:05 GMT</pubDate>
   <ndaq:CompanyName>Example Corp</ndaq:CompanyName>
   <ndaq:IssueSymbol>EXMP</ndaq:IssueSymbol>
   <ndaq:Market>NASDAQ</ndaq:Market>
   <ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
   <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
   <ndaq:HaltTime>14:32:01</ndaq:HaltTime>
   <ndaq:ResumptionDate></ndaq:ResumptionDate>
   <ndaq:ResumptionQuoteTime></ndaq:ResumptionQuoteTime>
   <ndaq:ResumptionTradeTime></ndaq:ResumptionTradeTime>
  </item>
  <item>
   <title>Trade Halt: NEWSY</title>
   <pubDate>Mon, 18 May 2026 17:01:00 GMT</pubDate>
   <ndaq:CompanyName>Newsy Inc</ndaq:CompanyName>
   <ndaq:IssueSymbol>NEWSY</ndaq:IssueSymbol>
   <ndaq:Market>NYSE</ndaq:Market>
   <ndaq:ReasonCode>T1</ndaq:ReasonCode>
   <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
   <ndaq:HaltTime>09:45:00</ndaq:HaltTime>
   <ndaq:ResumptionDate>05/18/2026</ndaq:ResumptionDate>
   <ndaq:ResumptionQuoteTime>10:10:00</ndaq:ResumptionQuoteTime>
   <ndaq:ResumptionTradeTime>10:15:00</ndaq:ResumptionTradeTime>
  </item>
  <item>
   <title>Trade Halt: IPOX</title>
   <pubDate>Mon, 18 May 2026 13:00:00 GMT</pubDate>
   <ndaq:CompanyName>NewListing Co</ndaq:CompanyName>
   <ndaq:IssueSymbol>IPOX</ndaq:IssueSymbol>
   <ndaq:Market>NASDAQ</ndaq:Market>
   <ndaq:ReasonCode>IPO1</ndaq:ReasonCode>
   <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
   <ndaq:HaltTime>08:00:00</ndaq:HaltTime>
  </item>
  <item>
   <title>Trade Halt: EXMP (dup)</title>
   <pubDate>Mon, 18 May 2026 18:32:05 GMT</pubDate>
   <ndaq:CompanyName>Example Corp</ndaq:CompanyName>
   <ndaq:IssueSymbol>EXMP</ndaq:IssueSymbol>
   <ndaq:Market>NASDAQ</ndaq:Market>
   <ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
   <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
   <ndaq:HaltTime>14:32:01</ndaq:HaltTime>
  </item>
 </channel>
</rss>"""


class _FakeResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _ok(content):
    def _get(url, **kwargs):
        return _FakeResp(200, content)
    return _get


def test_standard_shape_and_reason_mapping(monkeypatch):
    monkeypatch.setattr(nh.requests, "get", _ok(_FEED_XML))
    arts = nh.collect_nasdaq_halts()
    # IPO1 dropped (low signal), EXMP dup collapsed -> 2 events.
    assert len(arts) == 2
    by_sym = {a["_halt_symbol"]: a for a in arts}
    assert set(by_sym) == {"EXMP", "NEWSY"}

    exmp = by_sym["EXMP"]
    assert set(exmp) >= {"title", "link", "summary", "published", "source"}
    assert exmp["source"] == "nasdaq_halts"
    assert exmp["_halt_reason_code"] == "LUDP"
    assert exmp["_halt_resumed"] is False
    assert "Volatility trading pause (LULD)" in exmp["summary"]
    assert exmp["title"].startswith("HALT — EXMP Example Corp")
    # Per-event fragment keeps article_store's sha256(url||title) id unique
    # (digits of HaltDate 05/18/2026 + HaltTime 14:32:01).
    assert exmp["link"] == (
        "https://www.nasdaq.com/market-activity/stocks/exmp#halt-05182026143201"
    )


def test_resumed_halt_flagged_and_summarised(monkeypatch):
    monkeypatch.setattr(nh.requests, "get", _ok(_FEED_XML))
    newsy = {a["_halt_symbol"]: a for a in nh.collect_nasdaq_halts()}["NEWSY"]
    assert newsy["_halt_resumed"] is True
    assert newsy["title"].startswith("HALT / RESUME — NEWSY")
    assert "News pending" in newsy["summary"]
    assert "trade 05/18/2026 10:15:00 ET" in newsy["summary"]


def test_published_is_et_converted_to_utc(monkeypatch):
    # 2026-05-18 14:32:01 America/New_York is EDT (UTC-4) -> 18:32:01Z.
    monkeypatch.setattr(nh.requests, "get", _ok(_FEED_XML))
    exmp = {a["_halt_symbol"]: a for a in nh.collect_nasdaq_halts()}["EXMP"]
    pub = exmp["published"]
    parsed = datetime.fromisoformat(pub)
    assert parsed.utcoffset().total_seconds() == 0
    assert parsed == datetime(2026, 5, 18, 18, 32, 1, tzinfo=timezone.utc)


def test_include_low_signal_keeps_ipo(monkeypatch):
    monkeypatch.setattr(nh.requests, "get", _ok(_FEED_XML))
    arts = nh.collect_nasdaq_halts(include_low_signal=True)
    syms = {a["_halt_symbol"] for a in arts}
    assert "IPOX" in syms
    ipo = next(a for a in arts if a["_halt_symbol"] == "IPOX")
    assert ipo["_halt_reason_code"] == "IPO1"
    assert "IPO issue not yet trading" in ipo["summary"]


def test_fractional_halt_time_parsed_not_fallback():
    # The live feed ships HaltTime with milliseconds — it must use the halt
    # clock (precise), not fall back to the coarser pubDate.
    xml = b"""<rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel><item>
      <pubDate>Mon, 18 May 2026 23:59:59 GMT</pubDate>
      <ndaq:IssueSymbol>FRAC</ndaq:IssueSymbol>
      <ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
      <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
      <ndaq:HaltTime>14:32:01.056</ndaq:HaltTime>
    </item></channel></rss>"""
    art = nh._item_to_article(nh._parse_halts_xml(xml)[0], include_low_signal=False)
    # 14:32:01.056 EDT (UTC-4) -> 18:32:01.056Z (not the 23:59:59 pubDate).
    assert art["published"] == datetime(2026, 5, 18, 18, 32, 1, 56000,
                                        tzinfo=timezone.utc).isoformat()


def test_quote_only_resumption_renders_cleanly():
    # Only a quote time (no trade time) must not leave a dangling "trade  ET".
    xml = b"""<rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel><item>
      <ndaq:IssueSymbol>QONLY</ndaq:IssueSymbol>
      <ndaq:ReasonCode>T7</ndaq:ReasonCode>
      <ndaq:HaltDate>05/18/2026</ndaq:HaltDate>
      <ndaq:HaltTime>10:00:00</ndaq:HaltTime>
      <ndaq:ResumptionQuoteTime>10:30:00</ndaq:ResumptionQuoteTime>
    </item></channel></rss>"""
    art = nh._item_to_article(nh._parse_halts_xml(xml)[0], include_low_signal=False)
    assert art["_halt_resumed"] is True
    assert "Resumption: quote 10:30:00 ET." in art["summary"]
    assert "trade  ET" not in art["summary"]


def test_pubdate_fallback_when_halt_clock_missing():
    # No HaltDate/HaltTime -> falls back to RFC-822 pubDate (here UTC).
    xml = b"""<rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel><item>
      <pubDate>Mon, 18 May 2026 12:00:00 GMT</pubDate>
      <ndaq:IssueSymbol>NOCLK</ndaq:IssueSymbol>
      <ndaq:ReasonCode>T1</ndaq:ReasonCode>
    </item></channel></rss>"""
    art = nh._item_to_article(nh._parse_halts_xml(xml)[0], include_low_signal=False)
    assert art["published"] == datetime(2026, 5, 18, 12, 0, 0,
                                        tzinfo=timezone.utc).isoformat()


def test_unknown_code_has_generic_reason_and_no_symbol_skipped():
    xml = b"""<rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel>
      <item><ndaq:IssueSymbol>ZZZ</ndaq:IssueSymbol>
        <ndaq:ReasonCode>QQ9</ndaq:ReasonCode></item>
      <item><ndaq:ReasonCode>T1</ndaq:ReasonCode></item>
    </channel></rss>"""
    parsed = nh._parse_halts_xml(xml)
    arts = [nh._item_to_article(f, False) for f in parsed]
    arts = [a for a in arts if a is not None]
    assert len(arts) == 1  # the symbol-less item is skipped
    assert arts[0]["_halt_symbol"] == "ZZZ"
    assert "Trading halt (QQ9)" in arts[0]["summary"]


def test_graceful_empty_on_failures(monkeypatch):
    # Non-200, network exception, and malformed XML all -> [] (never raise).
    monkeypatch.setattr(nh.requests, "get",
                        lambda url, **k: _FakeResp(503, b""))
    assert nh.collect_nasdaq_halts() == []

    def _boom(url, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(nh.requests, "get", _boom)
    assert nh.collect_nasdaq_halts() == []

    monkeypatch.setattr(nh.requests, "get", _ok(b"<rss><not-closed>"))
    assert nh.collect_nasdaq_halts() == []
