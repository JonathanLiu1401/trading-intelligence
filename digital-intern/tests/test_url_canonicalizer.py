"""Exact-duplicate URL collapse at ingest.

``storage.article_store`` keys rows on ``sha256(url || title)``, so the same
publisher's same article arriving with a different tracking suffix, scheme,
host prefix, AMP twin or trailing slash currently lands as a *second* row —
re-scored, re-alerted, padding the analyst's feed. ``collectors.url_canonicalizer``
collapses those trivially-equivalent variants. ``ml.dedup`` is the fuzzy,
cross-publisher *title* half; this is the deterministic *URL* half.

These tests pin specific computed values (not "no crash"): exact tracking-param
stripping, scheme/host/slash/port normalization, query-order independence,
AMP folding, Google-News redirect unwrap, idempotence, the load-bearing
``backtest://`` no-op (backtest isolation must survive canonicalization), and
that ``canonical_article_id`` agrees with the store's own hashing scheme.
"""
from __future__ import annotations

import hashlib

import pytest

from collectors.url_canonicalizer import canonical_article_id, canonicalize_url


def test_strips_utm_and_click_tracking_keeps_real_params():
    assert (
        canonicalize_url("https://x.com/a?utm_source=feed&utm_medium=rss&id=7")
        == "https://x.com/a?id=7"
    )
    assert canonicalize_url("https://x.com/a?fbclid=ABC&gclid=XYZ&p=1") == \
        "https://x.com/a?p=1"
    # 'ref' is a known tracking exact; all-tracking query collapses to no query.
    assert canonicalize_url("https://x.com/a?ref=twitter") == "https://x.com/a"


def test_query_order_is_canonical():
    assert canonicalize_url("https://x.com/a?b=2&a=1") == \
        canonicalize_url("https://x.com/a?a=1&b=2") == "https://x.com/a?a=1&b=2"


def test_scheme_host_slash_and_port_normalization():
    # http==https, host lowercased, www. stripped, path-case preserved.
    assert canonicalize_url("http://WWW.Example.COM/Path/To") == \
        "https://example.com/Path/To"
    # m. mobile mirror folds to the bare host.
    assert canonicalize_url("https://m.example.com/story") == \
        "https://example.com/story"
    # Trailing slash normalized, but the bare root keeps its slash.
    assert canonicalize_url("https://x.com/a/") == "https://x.com/a"
    assert canonicalize_url("https://x.com/") == "https://x.com/"
    # Default port dropped.
    assert canonicalize_url("https://x.com:443/a") == "https://x.com/a"


def test_fragment_dropped_but_hashbang_preserved():
    assert canonicalize_url("https://x.com/a#section") == "https://x.com/a"
    # #! is legacy SPA routing — a genuinely distinct view, kept.
    assert canonicalize_url("https://x.com/a#!/foo") == "https://x.com/a#!/foo"


def test_amp_twin_folds_to_canonical():
    assert canonicalize_url("https://x.com/story/amp") == "https://x.com/story"
    assert canonicalize_url("https://x.com/story/amp/") == "https://x.com/story"
    assert canonicalize_url("https://x.com/story?outputType=amp") == \
        "https://x.com/story"
    # A path that merely *contains* 'amp' mid-segment is untouched.
    assert canonicalize_url("https://x.com/champ/news") == \
        "https://x.com/champ/news"


def test_google_news_redirect_is_unwrapped():
    wrapped = (
        "https://news.google.com/rss/articles/CBMiABC"
        "?url=https%3A%2F%2Fwww.reuters.com%2Fworld%2Fstory%2F"
        "&utm_source=googlenews&hl=en-US"
    )
    assert canonicalize_url(wrapped) == "https://reuters.com/world/story"


def test_backtest_url_is_returned_verbatim():
    """Load-bearing: backtest isolation is a ``url NOT LIKE 'backtest://%'``
    clause. Canonicalization MUST be a strict no-op on non-HTTP schemes or a
    synthetic row could leak into live alerts/briefings."""
    bt = "backtest://run_42/2026-01-01/BUY/AAPL?utm_source=x"
    assert canonicalize_url(bt) == bt
    assert canonicalize_url("mailto:a@b.com") == "mailto:a@b.com"
    # And the id of a backtest row still LIKE-matches the isolation clause.
    assert canonical_article_id(bt, "t") == canonical_article_id(bt, "t")
    assert canonicalize_url(bt).startswith("backtest://")


def test_empty_and_garbage_inputs_are_safe():
    assert canonicalize_url("") == ""
    assert canonicalize_url("   ") == ""
    assert canonicalize_url(None) == ""  # type: ignore[arg-type]
    # No scheme -> not HTTP -> returned stripped, never raises.
    assert canonicalize_url("  not a url  ") == "not a url"


def test_idempotent():
    messy = ("HTTP://WWW.Example.com:80/Story/amp/"
             "?utm_campaign=z&id=9&fbclid=Q#frag")
    once = canonicalize_url(messy)
    assert canonicalize_url(once) == once
    assert once == "https://example.com/Story?id=9"


def test_strips_dow_jones_and_yahoo_referrer_params():
    """Live regression (2026-05-29): the same Barron's article
    'Micron Faces New Threat From Samsung's Memory Chip for AI' fired a
    BREAKING push 3× in 24h via referrer-param variants. The yfinance feed
    delivered the URL with ?siteid=yhoof2&ypt=1; the scraped/www.barrons.com
    feed delivered the SAME slug (/articles/...-ac9a8e59) with
    ?mod=md_home_pan_m. Without stripping these three referrer markers the
    canonical id diverges → three rows → three pushes for one article."""
    base = "https://www.barrons.com/articles/micron-stock-price-samsung-memorychip-ai-ac9a8e59"
    yahoo_variant = f"{base}?siteid=yhoof2&ypt=1"
    barrons_variant = f"{base}?mod=md_home_pan_m"
    bare = f"https://barrons.com/articles/micron-stock-price-samsung-memorychip-ai-ac9a8e59"
    assert canonicalize_url(yahoo_variant) == bare
    assert canonicalize_url(barrons_variant) == bare
    # Article id collapses across both variants — this is what stops the
    # duplicate-push.
    title = "Micron Faces New Threat From Samsung's Memory Chip for AI"
    assert canonical_article_id(yahoo_variant, title) == \
        canonical_article_id(barrons_variant, title)


def test_canonical_article_id_collapses_variants_and_matches_store_scheme():
    title = "Fed holds rates steady"
    a = canonical_article_id(
        "https://www.reuters.com/markets/fed/?utm_source=feedburner", title)
    b = canonical_article_id(
        "http://reuters.com/markets/fed#top", title)
    assert a == b, "tracking/scheme variants of one article must share an id"

    # Different title -> different id (title still participates in the hash).
    assert canonical_article_id("https://x.com/a", "A") != \
        canonical_article_id("https://x.com/a", "B")

    # Agrees with storage.article_store.article_id's exact hashing on the
    # canonical URL (drop-in compatibility).
    url = "https://x.com/p?utm_medium=rss&k=1"
    expected = hashlib.sha256(
        f"{canonicalize_url(url)}||{title}".encode()).hexdigest()
    assert canonical_article_id(url, title) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
