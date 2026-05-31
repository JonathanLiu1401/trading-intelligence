"""Negative cache for `get_quant_signals_live` — locks the behaviour added by
the strategy.py fix that stops a delisted / short-history / wedged ticker
from re-hitting yfinance on every cycle.

A zombie WATCHLIST entry (e.g. the GOOGU / METAU 2026-05 delistings, future
churn) returning ``empty`` history previously fell through the ``continue``
in ``get_quant_signals_live`` without writing to ``_QUANT_CACHE``, so it
re-hit ``yf.Ticker(t).history(...)`` every cycle for the next 5 minutes of
``_QUANT_TTL`` — for every dead symbol. The negative cache mirrors
``market._DEAD_CACHE``'s discipline so a stable-zombie produces at most one
yfinance fetch per ``_QUANT_NEG_TTL`` window.

Each test verifies a SPECIFIC behaviour (call count, expiry, success-after-
recovery, exception path) by asserting concrete numeric counts — never just
"no exception was raised".
"""
from __future__ import annotations

import pandas as pd
import pytest

import paper_trader.strategy as strategy


@pytest.fixture(autouse=True)
def _reset_quant_caches():
    """Drop both caches before AND after every test in this module so a prior
    test's state can never leak in or out. The module-globals would otherwise
    accumulate across the (sometimes 800+) sibling tests in the strategy slice."""
    strategy._QUANT_CACHE.clear()
    strategy._QUANT_NEG_CACHE.clear()
    yield
    strategy._QUANT_CACHE.clear()
    strategy._QUANT_NEG_CACHE.clear()


class _CountingTicker:
    """A `yfinance.Ticker`-shaped double that records every `history()` call so
    a test can assert exact fetch counts.

    ``frames`` is a list of DataFrames the next N calls return in order; once
    exhausted the last frame is replayed (mirrors a stable steady state).
    Setting ``frames=[]`` with ``raise_with`` triggers an exception on every
    call — exercises the broad-`except` branch in ``get_quant_signals_live``.
    """

    calls_by_symbol: dict[str, int] = {}

    def __init__(self, sym):
        self.sym = sym

    @classmethod
    def reset(cls):
        cls.calls_by_symbol = {}

    def history(self, period="1y", auto_adjust=False):
        _CountingTicker.calls_by_symbol[self.sym] = (
            _CountingTicker.calls_by_symbol.get(self.sym, 0) + 1
        )
        frames = _CountingTicker._frames_by_symbol.get(self.sym, [])
        exc = _CountingTicker._raise_by_symbol.get(self.sym)
        if exc is not None:
            raise exc
        if not frames:
            return pd.DataFrame({"Close": [], "Volume": []})
        idx = min(_CountingTicker.calls_by_symbol[self.sym] - 1, len(frames) - 1)
        return frames[idx]


_CountingTicker._frames_by_symbol = {}
_CountingTicker._raise_by_symbol = {}


@pytest.fixture
def counting_yf(monkeypatch):
    """Install the counting ticker double and reset its state."""
    _CountingTicker.reset()
    _CountingTicker._frames_by_symbol = {}
    _CountingTicker._raise_by_symbol = {}
    monkeypatch.setattr("yfinance.Ticker", _CountingTicker)
    return _CountingTicker


# ────────────────────────── empty-history path ───────────────────────────


class TestEmptyHistoryNegativeCache:
    """An ``empty`` (zero-row) DataFrame is the canonical delisted-ticker
    yfinance signature. After one such call the negative cache must capture
    the symbol; the next call must NOT re-hit yfinance."""

    def test_empty_history_marks_negative_cache(self, counting_yf):
        # First call: yfinance returns empty → entry lands in _QUANT_NEG_CACHE.
        out1 = strategy.get_quant_signals_live(["ZOMBIE"])
        assert out1 == {}
        assert counting_yf.calls_by_symbol.get("ZOMBIE") == 1
        assert "ZOMBIE" in strategy._QUANT_NEG_CACHE

    def test_second_call_within_ttl_skips_yfinance(self, counting_yf):
        strategy.get_quant_signals_live(["ZOMBIE"])
        first_count = counting_yf.calls_by_symbol.get("ZOMBIE")

        # Second call within TTL: must short-circuit on the negative cache.
        out2 = strategy.get_quant_signals_live(["ZOMBIE"])
        assert out2 == {}
        assert counting_yf.calls_by_symbol.get("ZOMBIE") == first_count

    def test_expired_neg_entry_self_evicts(self, counting_yf, monkeypatch):
        # Mark the symbol expired by stamping a ts that is older than the TTL.
        import time
        monkeypatch.setattr(strategy, "_QUANT_NEG_TTL", 1.0)
        strategy._QUANT_NEG_CACHE["ZOMBIE"] = time.time() - 100.0  # ancient

        # _quant_neg_hit should return False on the expired entry AND evict it.
        assert strategy._quant_neg_hit("ZOMBIE", time.time()) is False
        assert "ZOMBIE" not in strategy._QUANT_NEG_CACHE


# ────────────────────────── short-history path ───────────────────────────


class TestShortHistoryNegativeCache:
    """A newly-listed symbol can return a non-empty frame that still has fewer
    than 60 closes (the floor get_quant_signals_live needs for RSI/MACD math).
    The previous code dropped this through ``continue`` without caching;
    the fix must mark it negative so the same too-short fetch isn't repeated."""

    def test_lt60_closes_marks_negative_cache(self, counting_yf):
        # 30 closes: under the 60-close floor.
        df = pd.DataFrame({"Close": [10.0] * 30, "Volume": [1_000_000.0] * 30})
        counting_yf._frames_by_symbol["NEWIPO"] = [df]

        out = strategy.get_quant_signals_live(["NEWIPO"])
        assert out == {}
        assert "NEWIPO" in strategy._QUANT_NEG_CACHE
        assert counting_yf.calls_by_symbol.get("NEWIPO") == 1

    def test_second_call_within_ttl_skips_short_history_lookup(
        self, counting_yf
    ):
        df = pd.DataFrame({"Close": [10.0] * 30, "Volume": [1_000_000.0] * 30})
        counting_yf._frames_by_symbol["NEWIPO"] = [df]

        strategy.get_quant_signals_live(["NEWIPO"])
        strategy.get_quant_signals_live(["NEWIPO"])
        # Only one yfinance call: second was dodged via the negative cache.
        assert counting_yf.calls_by_symbol.get("NEWIPO") == 1


# ─────────────────────────── exception path ──────────────────────────────


class TestExceptionNegativeCache:
    """yfinance can also raise (rate-limit, network blip, parse error).
    The except: branch in get_quant_signals_live previously fell through
    without caching; the fix must mark the symbol negative."""

    def test_exception_marks_negative_cache(self, counting_yf):
        counting_yf._raise_by_symbol["BOOM"] = RuntimeError("yfinance blew up")
        out = strategy.get_quant_signals_live(["BOOM"])
        assert out == {}
        assert "BOOM" in strategy._QUANT_NEG_CACHE
        assert counting_yf.calls_by_symbol.get("BOOM") == 1

    def test_exception_second_call_within_ttl_skips(self, counting_yf):
        counting_yf._raise_by_symbol["BOOM"] = RuntimeError("yfinance blew up")
        strategy.get_quant_signals_live(["BOOM"])
        strategy.get_quant_signals_live(["BOOM"])
        # Only the first attempt actually called yfinance.
        assert counting_yf.calls_by_symbol.get("BOOM") == 1


# ──────────────────────── recovery from neg cache ───────────────────────


class TestRecoveryAfterExpiry:
    """A symbol that recovers after the negative TTL window must be re-fetched
    on the next call AND populate the positive cache normally — the negative
    cache must not be a permanent ban."""

    def test_success_after_neg_ttl_expiry_populates_positive_cache(
        self, counting_yf, monkeypatch
    ):
        # Step 1: yfinance returns empty → negative cache.
        out = strategy.get_quant_signals_live(["RECOVER"])
        assert out == {}
        assert "RECOVER" in strategy._QUANT_NEG_CACHE

        # Step 2: arrange a real frame for the next call.
        good_df = pd.DataFrame({
            "Close": [100.0 + (i % 10) * 0.5 for i in range(250)],
            "Volume": [1_000_000.0] * 250,
        })
        counting_yf._frames_by_symbol["RECOVER"] = [good_df]

        # Force the negative entry to look expired so the next call re-fetches.
        monkeypatch.setattr(strategy, "_QUANT_NEG_TTL", 0.001)
        import time
        time.sleep(0.002)

        out2 = strategy.get_quant_signals_live(["RECOVER"])
        assert "RECOVER" in out2
        # Negative entry must have been evicted on the successful read path.
        assert "RECOVER" not in strategy._QUANT_NEG_CACHE
        # Positive cache populated.
        assert "RECOVER" in strategy._QUANT_CACHE


# ─────────────────── mixed-batch isolation behaviour ────────────────────


class TestMixedBatchIsolation:
    """A negative-cached zombie inside a batch must NOT block the surrounding
    healthy tickers from being fetched and returned."""

    def test_dead_ticker_does_not_starve_live_ticker(self, counting_yf):
        # Healthy ticker frame.
        good = pd.DataFrame({
            "Close": [100.0 + (i % 7) * 0.3 for i in range(250)],
            "Volume": [1_000_000.0] * 250,
        })
        counting_yf._frames_by_symbol["GOOD"] = [good]
        # Dead ticker returns empty (default frame_by_symbol path).
        # First call populates pos cache for GOOD, neg cache for DEAD.
        out1 = strategy.get_quant_signals_live(["DEAD", "GOOD"])
        assert "GOOD" in out1
        assert "DEAD" not in out1
        assert "DEAD" in strategy._QUANT_NEG_CACHE

        # Second call hits positive cache for GOOD (no new yf call), negative
        # cache for DEAD (no new yf call) → zero additional yfinance fetches.
        good_count_before = counting_yf.calls_by_symbol.get("GOOD", 0)
        dead_count_before = counting_yf.calls_by_symbol.get("DEAD", 0)
        out2 = strategy.get_quant_signals_live(["DEAD", "GOOD"])
        assert "GOOD" in out2
        assert "DEAD" not in out2
        assert counting_yf.calls_by_symbol.get("GOOD", 0) == good_count_before
        assert counting_yf.calls_by_symbol.get("DEAD", 0) == dead_count_before


# ─────────────────── private helpers behave correctly ───────────────────


class TestNegativeCacheHelpers:
    """`_quant_neg_hit` / `_quant_neg_mark` are the pure primitives the loop
    composes; pin their behaviour independently so a future refactor that
    only touches the helpers (no loop change) still has direct coverage."""

    def test_unmarked_symbol_is_a_miss(self):
        import time
        assert strategy._quant_neg_hit("FRESHSYM", time.time()) is False

    def test_marked_symbol_is_a_hit_within_ttl(self):
        import time
        now = time.time()
        strategy._quant_neg_mark("MARKED", now)
        assert strategy._quant_neg_hit("MARKED", now + 1.0) is True

    def test_marked_symbol_misses_after_ttl(self, monkeypatch):
        import time
        monkeypatch.setattr(strategy, "_QUANT_NEG_TTL", 1.0)
        now = time.time()
        strategy._quant_neg_mark("OLD", now)
        # 2s later (well past the 1s TTL) — must miss.
        assert strategy._quant_neg_hit("OLD", now + 2.0) is False

    def test_mark_is_idempotent_refreshing_timestamp(self):
        import time
        first = time.time()
        strategy._quant_neg_mark("RECYCLED", first)
        second = first + 100.0
        strategy._quant_neg_mark("RECYCLED", second)
        # The most recent mark wins — same key, just a refreshed timestamp.
        assert strategy._QUANT_NEG_CACHE["RECYCLED"] == second
