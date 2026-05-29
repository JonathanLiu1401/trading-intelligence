"""``market.dead_tickers()`` exposes which watchlist symbols the live trader
is currently unable to fetch.

Today ``_DEAD_CACHE`` carries the live trader's "I cannot fetch this
symbol right now" state and is internal — the only operator signal is
a one-shot stderr line on the first dead-mark per TTL window. A
trader monitoring the desk has NO way to ask "which of my 50 watchlist
names is the engine flying blind on?". Every cycle, half-silently,
``WATCHLIST PRICES`` shows ``N/A`` for those tickers and Opus has no
idea they are *known broken* (e.g., a delisted leveraged ETF the
watchlist still references) vs. *transiently slow* (a yfinance hiccup
that will recover next cycle).

The accessor is the inert read other surfaces (a Discord hourly line,
``/api/dead-tickers``, an operator REPL) compose to render that view.
Pure read; never mutates the cache.
"""
from __future__ import annotations

import pytest

from paper_trader import market


@pytest.fixture(autouse=True)
def _clean_caches():
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()
    yield
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()


class TestDeadTickersAccessor:
    """``market.dead_tickers()`` introspects the negative cache. Pure read,
    never raises."""

    def test_empty_when_no_dead_tickers(self):
        # Fresh cache: empty list, never None.
        assert market.dead_tickers() == []

    def test_lists_currently_dead_ticker(self, monkeypatch):
        # Pin time so seconds_dead is exact.
        t0 = 1_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("LITE")
        # Probe at t0 + 30s
        monkeypatch.setattr(market.time, "time", lambda: t0 + 30.0)
        rows = market.dead_tickers()
        assert isinstance(rows, list)
        assert len(rows) == 1
        r = rows[0]
        assert r["ticker"] == "LITE"
        assert r["seconds_dead"] == 30
        assert r["ttl_remaining_s"] == int(market._DEAD_TTL - 30)
        assert isinstance(r["marked_at_ts"], float)

    def test_excludes_expired_entries(self, monkeypatch):
        """A dead entry whose TTL has elapsed is NOT returned (it would be
        re-fetched on the next get_price call, so it isn't "dark" anymore)."""
        t0 = 2_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("STALE")
        # Probe past the TTL.
        monkeypatch.setattr(market.time, "time",
                            lambda: t0 + market._DEAD_TTL + 1.0)
        assert market.dead_tickers() == []

    def test_returns_sorted_for_stable_output(self, monkeypatch):
        # Determinism makes a Discord line / dashboard table read consistently.
        t0 = 3_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        for t in ("ZZZ", "AAA", "MUU"):
            market._mark_dead(t)
        # Same probe time so all entries are "fresh".
        rows = market.dead_tickers()
        tickers = [r["ticker"] for r in rows]
        assert tickers == sorted(tickers), (
            f"dead_tickers() must return tickers in stable (sorted) order; "
            f"got {tickers}"
        )

    def test_multiple_entries_with_different_ages(self, monkeypatch):
        # An older mark has higher seconds_dead and lower ttl_remaining_s.
        t0 = 4_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("OLD")            # marked at t0
        monkeypatch.setattr(market.time, "time", lambda: t0 + 100.0)
        market._mark_dead("NEW")            # marked at t0+100
        # Probe at t0+150
        monkeypatch.setattr(market.time, "time", lambda: t0 + 150.0)
        rows = {r["ticker"]: r for r in market.dead_tickers()}
        assert rows["OLD"]["seconds_dead"] == 150
        assert rows["NEW"]["seconds_dead"] == 50
        assert rows["OLD"]["ttl_remaining_s"] < rows["NEW"]["ttl_remaining_s"]

    def test_pure_read_does_not_mutate_cache(self, monkeypatch):
        """Calling ``dead_tickers()`` must NOT clear the cache or refresh
        the timestamps — operators expect a snapshot, not a sweep."""
        t0 = 5_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("PIN")
        before_ts = market._DEAD_CACHE["PIN"]
        _ = market.dead_tickers()
        after_ts = market._DEAD_CACHE["PIN"]
        assert before_ts == after_ts
        # And the entry is still there.
        assert "PIN" in market._DEAD_CACHE

    def test_clear_dead_drops_from_accessor(self, monkeypatch):
        """``_clear_dead`` (called on a successful re-fetch) must drop the
        entry from ``dead_tickers()`` immediately, so a recovered ticker
        does not linger in the operator-visible dark list."""
        t0 = 6_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("FLIP")
        assert "FLIP" in {r["ticker"] for r in market.dead_tickers()}
        market._clear_dead("FLIP")
        assert market.dead_tickers() == []

    def test_wall_clock_step_back_clamps_seconds_dead(self, monkeypatch):
        """A wall-clock step-back (NTP correction) after marking would
        otherwise render a negative ``seconds_dead``; the accessor must
        clamp to 0 so the rendered value never reads as nonsense."""
        t0 = 7_000_000.0
        monkeypatch.setattr(market.time, "time", lambda: t0)
        market._mark_dead("SKEW")
        # Probe 5 seconds BEFORE the mark.
        monkeypatch.setattr(market.time, "time", lambda: t0 - 5.0)
        rows = market.dead_tickers()
        assert len(rows) == 1
        assert rows[0]["seconds_dead"] == 0
        # And ttl_remaining stays bounded by _DEAD_TTL.
        assert 0 <= rows[0]["ttl_remaining_s"] <= int(market._DEAD_TTL)
