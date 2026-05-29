"""Tests for reporter._dead_tickers_line — Discord visibility for the
live trader's negative-cache state.

The producer ``market.dead_tickers()`` is locked by tests/test_market_*.
These tests pin the RENDER contract: silence-on-empty, format on
non-empty, ordering by ``seconds_dead`` desc, +N-more truncation
above 5, degrade-to-empty on any producer fault.
"""
from __future__ import annotations

import types

import pytest

from paper_trader import reporter


# ── _dead_age_token (compact age) ────────────────────────────────────────

@pytest.mark.parametrize(
    "secs,expected",
    [
        (None, ""),
        (-5, ""),               # clock skew clamp → empty
        ("garbage", ""),        # non-numeric → empty
        (0, "0s"),
        (12, "12s"),
        (59, "59s"),
        (60, "1m"),
        (120, "2m"),
        (3599, "59m"),
        (3600, "1h"),
        (7200, "2h"),
    ],
)
def test_dead_age_token_format(secs, expected):
    assert reporter._dead_age_token(secs) == expected


# ── _dead_tickers_line — silence on empty ────────────────────────────────

def _stub_market(rows):
    """Build a stub object whose ``dead_tickers()`` returns ``rows``."""
    return types.SimpleNamespace(dead_tickers=lambda: rows)


def test_empty_cache_returns_empty_string():
    """Silence-when-nothing-actionable — the healthy watchlist must NOT
    print a "0 dark" green-light line on every hourly. Mirror of
    ``_feed_health_line`` HEALTHY suppression."""
    assert reporter._dead_tickers_line(market_mod=_stub_market([])) == ""


def test_non_list_return_treated_as_empty():
    """A producer that returns a non-list (regression / bug) must degrade
    to silence rather than crash the report."""
    bad = types.SimpleNamespace(dead_tickers=lambda: None)
    assert reporter._dead_tickers_line(market_mod=bad) == ""


def test_producer_raises_degrades_to_empty():
    """Failure contract — any producer fault must degrade to "" so the
    hourly summary itself still ships (the additive-line discipline)."""
    def _boom():
        raise RuntimeError("yfinance unreachable")
    bad = types.SimpleNamespace(dead_tickers=_boom)
    assert reporter._dead_tickers_line(market_mod=bad) == ""


# ── _dead_tickers_line — formatted output ────────────────────────────────

def test_single_dark_ticker_uses_singular_word():
    """One dark symbol → ``symbol`` (singular), not ``symbols``. A small
    correctness pin so the count word never reads wrong on n=1."""
    rows = [{"ticker": "LITE", "seconds_dead": 240, "ttl_remaining_s": 60}]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert "1 watchlist symbol dark" in line
    assert "LITE 4m" in line
    assert "symbols" not in line  # plural must not appear


def test_two_dark_uses_plural():
    """Two dark symbols → ``symbols`` (plural)."""
    rows = [
        {"ticker": "LITE", "seconds_dead": 240, "ttl_remaining_s": 60},
        {"ticker": "MUU", "seconds_dead": 12, "ttl_remaining_s": 288},
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert "2 watchlist symbols dark" in line


def test_orders_by_seconds_dead_descending():
    """Worst-first ordering — the operator's eye must land on the longest-
    dark name. Inputs deliberately not pre-sorted (the producer sorts by
    ticker, this re-sort proves the render is independent)."""
    rows = [
        {"ticker": "AAA", "seconds_dead": 30},
        {"ticker": "BBB", "seconds_dead": 180},
        {"ticker": "CCC", "seconds_dead": 90},
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    # The order in the rendered bullets must be BBB → CCC → AAA.
    pos_b = line.find("BBB")
    pos_c = line.find("CCC")
    pos_a = line.find("AAA")
    assert pos_b < pos_c < pos_a, (
        f"expected BBB<CCC<AAA in line, got {line!r}")


def test_top_5_cap_with_more_suffix():
    """Above 5 dark → first 5 render explicitly, the rest collapse into
    ``+N more``. Keeps the Discord line a single readable sentence even
    on a wide yfinance outage."""
    rows = [
        {"ticker": f"T{i:02d}", "seconds_dead": (10 - i) * 30}
        for i in range(8)  # 8 dark tickers
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert "8 watchlist symbols dark" in line
    # First 5 in order of descending seconds_dead (T00 oldest → T04).
    for i in range(5):
        assert f"T{i:02d}" in line
    # The +3 more suffix surfaces the truncated count.
    assert "+3 more" in line
    # The 6th-8th names must NOT appear inline.
    for i in range(5, 8):
        assert f"T{i:02d}" not in line


def test_exactly_5_no_more_suffix():
    """Boundary — exactly 5 dark tickers all render without the
    ``+N more`` suffix."""
    rows = [
        {"ticker": f"T{i}", "seconds_dead": (5 - i) * 30}
        for i in range(5)
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert "5 watchlist symbols dark" in line
    assert "+0 more" not in line
    assert "more" not in line  # no +N more suffix at all


def test_missing_age_renders_bare_ticker():
    """A row without ``seconds_dead`` (defensive — the producer always
    writes it but a future schema change might drop it) renders the
    bare ticker without the age parenthetical, not a crash."""
    rows = [{"ticker": "LITE"}]  # no seconds_dead key
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert "LITE" in line
    assert "1 watchlist symbol dark" in line


def test_empty_ticker_dropped():
    """A row whose ``ticker`` is empty or whitespace contributes nothing
    to the inline bullet list — but the COUNT still includes it (the
    producer asserts a ticker is present, this is only defensive)."""
    rows = [
        {"ticker": "LITE", "seconds_dead": 120},
        {"ticker": "", "seconds_dead": 60},
        {"ticker": "   ", "seconds_dead": 30},
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    # Count reflects all 3 cache entries; only the valid one renders inline.
    assert "3 watchlist symbols dark" in line
    assert "LITE" in line


def test_all_empty_tickers_returns_empty_string():
    """If every row's ticker is blank, there is no useful operator signal
    to render — degrade to silence (defensive; the producer should never
    write blank rows)."""
    rows = [
        {"ticker": "", "seconds_dead": 60},
        {"ticker": "   ", "seconds_dead": 30},
    ]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert line == ""


def test_warning_icon_and_actionable_suffix():
    """The line must carry the ⚠️ icon (urgency-tier match with FEED
    HEALTH) and the actionable suffix that names the trader-visible
    impact — Opus reads N/A this cycle for the affected symbols."""
    rows = [{"ticker": "NVDA", "seconds_dead": 45}]
    line = reporter._dead_tickers_line(market_mod=_stub_market(rows))
    assert line.startswith("⚠️")
    assert "engine reads N/A this cycle" in line


def test_default_market_module_uses_real_import():
    """When no ``market_mod`` is passed, the function uses the module
    imported at top — the production path. Smoke-tests the integration
    by clearing the real ``_DEAD_CACHE`` and confirming the empty-cache
    silence contract holds end-to-end."""
    from paper_trader import market as _mkt
    saved = dict(_mkt._DEAD_CACHE)
    try:
        _mkt._DEAD_CACHE.clear()
        assert reporter._dead_tickers_line() == ""
    finally:
        _mkt._DEAD_CACHE.clear()
        _mkt._DEAD_CACHE.update(saved)
