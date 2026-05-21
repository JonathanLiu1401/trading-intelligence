"""Briefing-path quote-widget noise gate (analysis/claude_analyst.py).

The 5h Opus heartbeat digest is the analyst's primary consumed product. A
live ticker-tape pseudo-article ("NVDANVIDIA Corporation227.13-8.61(-3.65%)")
that enters via a non-web_scraper path (yahoo_ticker_rss / finnhub / replay)
and is ML-scored high used to surface there as a fake TOP SIGNAL — the
documented #1 noise complaint. The alert path and web_scraper already gate
it; these tests pin the equivalent gate now added to the briefing payload.

Assertions are on concrete behaviour (exact membership / counts / payload
substrings / input non-mutation), never just "did not crash".
"""
import copy

import pytest

from analysis.claude_analyst import (
    _looks_like_quote_widget,
    _filter_quote_widget_noise,
    _build_payload,
)


# ── _looks_like_quote_widget — the two title fingerprints + /quote/ path ─────

@pytest.mark.parametrize("title", [
    "NVDANVIDIA Corporation227.13-8.61(-3.65%)",   # letter-glued price
    "ETH-USDEthereum USD2,169.83",                  # comma-decimal glue
    "AAPLApple Inc.189.45+1.20(+0.64%)",            # glue + signed-pct-paren
    "Some headline (-3.65%)",                       # signed-pct parenthetical
])
def test_widget_titles_detected(title):
    assert _looks_like_quote_widget({"title": title, "link": ""}) is True


@pytest.mark.parametrize("title", [
    "Nvidia rises 22% to $35.1 billion in quarterly revenue",
    "S&P 500 hits 5,123.41 record high as rally broadens",
    "Apple's $1.50EPS beat sends shares higher",      # space after 's defeats glue
    "Micron shares surge after Q3 earnings blowout",
    "PORTFOLIO P&L SNAPSHOT",                          # prepended snapshot row
    "OPTIONS SNAPSHOT",
])
def test_real_headlines_not_flagged(title):
    assert _looks_like_quote_widget({"title": title, "link": ""}) is False


def test_quote_landing_path_detected_even_with_clean_title():
    art = {"title": "NVIDIA Corporation", "link": "https://finance.yahoo.com/quote/NVDA"}
    assert _looks_like_quote_widget(art) is True
    art2 = {"title": "NVIDIA Corporation", "link": "https://finance.yahoo.com/quote/NVDA/"}
    assert _looks_like_quote_widget(art2) is True


def test_real_quote_scoped_article_url_not_flagged():
    # A genuine article *under* a quote path must NOT be caught — the regex is
    # anchored to end-of-path so only the bare landing page matches.
    art = {
        "title": "Nvidia earnings preview: what to watch",
        "link": "https://finance.yahoo.com/quote/NVDA/news/headline-123",
    }
    assert _looks_like_quote_widget(art) is False


def test_url_alias_and_missing_fields_safe():
    # `url` alias honoured (some callers carry url not link); blank/None safe.
    assert _looks_like_quote_widget(
        {"title": "NVIDIA", "url": "https://finance.yahoo.com/quote/NVDA"}
    ) is True
    assert _looks_like_quote_widget({"title": "", "link": None}) is False
    assert _looks_like_quote_widget({}) is False


# ── Quote-listing share-card fingerprint (_QW_LISTING) ──────────────────────
# "$NVIDIA (NVDA.US)$ - Moomoo" — a Google-News-indexed Moomoo/Futu/Webull
# quote share-card landing page (NOT an article). ML-relevance over-scored
# (live: 9.77) so it reaches the top-60 newswire and renders as a fake
# "[HH:MM] [score] TOP SIGNAL" — recurring across ≥6 prior passes. A distinct
# surface the price-glue / parenthesised-% fingerprints don't catch.

@pytest.mark.parametrize("title", [
    "$NVIDIA (NVDA.US)$ - Moomoo",                  # the exact live row
    "$Tesla (TSLA.US)$ - Moomoo",
    "$Tencent (00700.HK)$ - Futu",                  # HK numeric symbol
    "$Samsung Electronics (005930.KS)$ - Webull",
    "  $NIO Inc. (NIO.US)$",                         # leading ws, no provider
])
def test_quote_listing_share_card_detected(title):
    assert _looks_like_quote_widget({"title": title, "link": ""}) is True


@pytest.mark.parametrize("title", [
    "$NVDA breaks out ahead of earnings (NYSE)",     # real $TICKER prose
    "$MU upgraded to Buy (price target $150.00)",
    "$TSLA: why I am buying the dip (analysis)",
    "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00 by Analysts",
    "Nvidia (NVDA) Q1 preview: all eyes on data center",
])
def test_real_dollar_ticker_headlines_not_flagged(title):
    # The share-card-only discriminator is the glued "(SYM.EXCH)$" close;
    # real "$TICKER ..." headlines must survive the gate.
    assert _looks_like_quote_widget({"title": title, "link": ""}) is False


def test_build_payload_excludes_quote_listing_keeps_real():
    articles = [
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "rss", "ai_score": 9, "summary": "DRAM pricing up",
         "link": "https://example.com/micron-q3"},
        {"title": "$NVIDIA (NVDA.US)$ - Moomoo",
         "source": "GN: Nvidia", "ai_score": 9.77, "summary": "",
         "link": "https://news.google.com/x"},
    ]
    payload = _build_payload(articles, {}, [])
    section = payload.split("=== NEWSWIRE (scored, ranked) ===", 1)[1]
    assert "Micron shares surge after Q3 earnings blowout" in section
    assert "$NVIDIA (NVDA.US)$" not in payload


# ── _filter_quote_widget_noise — pure partition, order, no mutation ──────────

def test_partition_keeps_real_drops_widgets_in_order():
    arts = [
        {"title": "Fed holds rates steady amid inflation concerns", "_id": "a"},
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)", "_id": "b"},
        {"title": "Micron beats Q3 estimates on DRAM strength", "_id": "c"},
        {"title": "TSMTaiwan Semiconductor 178.40+2.10(+1.19%)", "_id": "d"},
    ]
    kept, suppressed = _filter_quote_widget_noise(arts)
    assert [a["_id"] for a in kept] == ["a", "c"]
    assert [a["_id"] for a in suppressed] == ["b", "d"]


def test_filter_does_not_mutate_input_list_or_dicts():
    arts = [
        {"title": "Real market headline about earnings", "_id": "x"},
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)", "_id": "y"},
    ]
    snapshot = copy.deepcopy(arts)
    kept, suppressed = _filter_quote_widget_noise(arts)
    assert arts == snapshot, "input list/dicts must be untouched"
    assert kept is not arts and suppressed is not arts


def test_empty_input_returns_two_empty_lists():
    kept, suppressed = _filter_quote_widget_noise([])
    assert kept == [] and suppressed == []


# ── _build_payload integration — widgets never reach the Opus prompt ─────────

def _digest_section(payload: str) -> str:
    return payload.split("=== NEWSWIRE (scored, ranked) ===", 1)[1]


def test_build_payload_excludes_widget_rows_keeps_real():
    articles = [
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "rss", "ai_score": 9, "summary": "DRAM pricing up sharply",
         "link": "https://example.com/micron-q3"},
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
         "source": "yahoo_ticker_rss", "ai_score": 9.9,
         "summary": "", "link": "https://finance.yahoo.com/quote/NVDA"},
    ]
    payload = _build_payload(articles, {}, [])
    section = _digest_section(payload)
    assert "Micron shares surge after Q3 earnings blowout" in section
    assert "NVDANVIDIA Corporation227.13" not in payload
    # The real row still rendered with its score.
    assert "score=9" in section


def test_build_payload_all_widgets_degrades_to_no_articles_line():
    articles = [
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
         "source": "yahoo_ticker_rss", "ai_score": 9.9, "link": ""},
        {"title": "TSMTaiwan Semiconductor 178.40+2.10(+1.19%)",
         "source": "finnhub", "ai_score": 9.5, "link": ""},
    ]
    payload = _build_payload(articles, {}, [])
    assert "(no high-relevance articles this cycle)" in payload


def test_build_payload_empty_input_unchanged_behaviour():
    payload = _build_payload([], {}, [])
    assert "(no high-relevance articles this cycle)" in payload


def test_build_payload_does_not_mutate_caller_articles():
    # heartbeat_worker keeps `source_articles` for the training-label path;
    # _build_payload must not drop/reorder it as a side effect.
    articles = [
        {"title": "Real headline about a Fed decision today", "source": "rss",
         "ai_score": 8, "summary": "", "link": "https://example.com/fed"},
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)", "source": "yf",
         "ai_score": 9.9, "summary": "", "link": ""},
    ]
    before = copy.deepcopy(articles)
    _build_payload(articles, {}, [])
    assert articles == before, "caller's article list must be untouched"


# ── StockTwits Sentiment fingerprint (_QW_STOCKTWITS_SENTIMENT) ──────────────
# "[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30
# msgs)" — structured-data summary from collectors/stocktwits_sentiment.py.
# Live evidence (2026-05-21, 5h): 130 rows, 45 ML-scored >=5, several at the
# 10.0 ceiling — the briefing's per-domain cap admits up to 6 into the top
# pool every cycle, displacing real news in TOP SIGNALS.


@pytest.mark.parametrize("title", [
    "[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30 msgs)",
    "[StockTwits Sentiment] ORCL Bullish: 80% Bullish / 0% Bearish (24↑ 0↓ of 30 msgs)",
    "[StockTwits Sentiment] LITE Bearish: 10% Bullish / 60% Bearish (3↑ 18↓ of 30 msgs)",
    "  [StockTwits Sentiment] MU Bullish: 30% Bullish / 0% Bearish",  # leading ws
])
def test_stocktwits_sentiment_pseudo_detected(title):
    assert _looks_like_quote_widget({"title": title, "link": ""}) is True


@pytest.mark.parametrize("title", [
    # Real news about StockTwits or sentiment must not match — the leading
    # bracketed marker is the discriminator.
    "StockTwits announces new sentiment dashboard for retail traders",
    "Retail bullish on NVDA per StockTwits sentiment data",
    "Sentiment turns bullish ahead of NVDA earnings",
    "Bullish: NVDA breaks key resistance level",
])
def test_real_sentiment_headlines_not_flagged(title):
    assert _looks_like_quote_widget({"title": title, "link": ""}) is False


def test_build_payload_excludes_stocktwits_sentiment_keeps_real():
    """A real news headline must survive while the StockTwits Sentiment
    pseudo-article is dropped from the digest — analyst's primary product
    must not surface structured sentiment as a TOP SIGNAL."""
    articles = [
        {"title": "Micron beats Q3 estimates, raises guidance",
         "source": "rss", "ai_score": 9, "summary": "Strong DRAM demand",
         "link": "https://example.com/micron-q3"},
        {"title": "[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30 msgs)",
         "source": "stocktwits/sentiment", "ai_score": 0, "ml_score": 10.0,
         "summary": "", "link": "https://stocktwits.com/symbol/NVDA"},
    ]
    payload = _build_payload(articles, {}, [])
    section = _digest_section(payload)
    assert "Micron beats Q3 estimates" in section
    assert "StockTwits Sentiment" not in payload


def test_snapshot_rows_pass_through_payload():
    # The daemon prepends synthetic P&L / OPTIONS rows (ai_score=10, no url).
    # They must survive the gate and appear in the digest.
    articles = [
        {"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
         "summary": "grand_value $123,456", "ai_score": 10},
        {"title": "OPTIONS SNAPSHOT", "source": "options_monitor",
         "summary": "DRAM C59 ...", "ai_score": 10},
        {"title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
         "source": "yahoo_ticker_rss", "ai_score": 9.9, "link": ""},
    ]
    payload = _build_payload(articles, {}, [])
    assert "PORTFOLIO P&L SNAPSHOT" in payload
    assert "OPTIONS SNAPSHOT" in payload
    assert "NVDANVIDIA Corporation227.13" not in payload
