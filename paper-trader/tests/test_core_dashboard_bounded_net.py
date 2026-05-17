"""Tests for the bounded-network-I/O feature in paper_trader.dashboard.

`_bounded_call` is the second half of AGENTS.md invariant #7's deferred
concern: yfinance/`requests` has no total timeout, so a stalled HTTPS
socket inside an SWR background rebuild pins one of the 6 `_SWR_EXEC`
workers forever and every slow panel goes permanently dark on
`{"warming": true}`. `_daily_history_cached` (the shared yfinance fetch
behind /api/correlation, /api/news-edge, /api/source-edge,
/api/signal-followthrough) must now degrade in a bounded time instead of
never.

These tests assert the *actual bound* (a hang returns the default in well
under the hang duration) and that the pre-existing cache semantics are
unchanged — not merely that the code runs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


@pytest.fixture(autouse=True)
def _clear_px_cache(monkeypatch):
    """Each test starts with an empty daily-history cache so a prior
    test's bars can't satisfy a later cache-hit assertion."""
    monkeypatch.setattr(dashboard, "_NEWS_EDGE_PX_CACHE", {})
    yield


def _fake_yf_module(history_impl):
    """A stand-in `yfinance` module whose Ticker(...).history(...) delegates
    to `history_impl(ticker, period, auto_adjust)`."""
    class _FakeTicker:
        def __init__(self, ticker):
            self._t = ticker

        def history(self, period="3mo", auto_adjust=False):
            return history_impl(self._t, period, auto_adjust)

    import types
    m = types.ModuleType("yfinance")
    m.Ticker = _FakeTicker
    return m


def _frame(dates_closes):
    idx = pd.to_datetime([d for d, _ in dates_closes])
    return pd.DataFrame({"Close": [c for _, c in dates_closes]}, index=idx)


# ── _bounded_call ────────────────────────────────────────────────────────

class TestBoundedCall:
    def test_returns_value_when_fast(self):
        assert dashboard._bounded_call(lambda: 42, timeout_s=2.0,
                                       default=-1) == 42

    def test_returns_default_and_does_not_raise_when_fn_raises(self):
        def boom():
            raise RuntimeError("network down")
        assert dashboard._bounded_call(boom, timeout_s=2.0,
                                       default="DEF") == "DEF"

    def test_hang_returns_default_within_the_bound_not_the_hang(self):
        # The load-bearing property: a 3s hang must NOT block the caller
        # for 3s — it returns `default` within ~timeout_s.
        start = time.time()
        out = dashboard._bounded_call(lambda: time.sleep(3) or "late",
                                      timeout_s=0.3, default="BOUNDED")
        elapsed = time.time() - start
        assert out == "BOUNDED"
        assert elapsed < 1.5, f"bound not enforced: waited {elapsed:.2f}s"

    def test_submit_failure_degrades_to_default(self, monkeypatch):
        class _DeadPool:
            def submit(self, *a, **k):
                raise RuntimeError("pool shut down")
        monkeypatch.setattr(dashboard, "_NET_EXEC", _DeadPool())
        assert dashboard._bounded_call(lambda: 1, default=None) is None


# ── _daily_history_cached ────────────────────────────────────────────────

class TestDailyHistoryCached:
    def test_parses_and_caches_known_frame(self, monkeypatch):
        calls = []

        def hist(t, period, auto_adjust):
            calls.append(t)
            return _frame([("2026-05-10", 100.0),
                           ("2026-05-11", 101.5),
                           ("2026-05-12", float("nan"))])  # NaN row dropped

        monkeypatch.setitem(sys.modules, "yfinance",
                            _fake_yf_module(hist))

        bars = dashboard._daily_history_cached("NVDA")
        # NaN close excluded; dates formatted YYYY-MM-DD; floats coerced.
        assert bars == [("2026-05-10", 100.0), ("2026-05-11", 101.5)]

        # Second call inside the TTL must hit the cache (no 2nd fetch).
        bars2 = dashboard._daily_history_cached("NVDA")
        assert bars2 == bars
        assert calls == ["NVDA"], "cache miss — yfinance called twice"

    def test_hang_returns_empty_within_bound_and_caches_it(self, monkeypatch):
        # Shrink the network bound so the test is fast; a real hang would
        # otherwise dark the SWR pool forever.
        monkeypatch.setattr(dashboard, "_NET_TIMEOUT_S", 0.3)

        def hist(t, period, auto_adjust):
            time.sleep(2.0)
            return _frame([("2026-05-10", 1.0)])

        monkeypatch.setitem(sys.modules, "yfinance",
                            _fake_yf_module(hist))

        start = time.time()
        bars = dashboard._daily_history_cached("HANG")
        elapsed = time.time() - start
        assert bars == []
        assert elapsed < 1.5, (
            f"daily-history did not bound the hang: {elapsed:.2f}s")
        # Pre-existing semantics preserved: a failed fetch is cached empty
        # for the TTL (identical to the old `except: bars=[]` path).
        assert dashboard._NEWS_EDGE_PX_CACHE.get("HANG", (None,))[0] == []

    def test_fetch_exception_degrades_to_empty(self, monkeypatch):
        def hist(t, period, auto_adjust):
            raise ValueError("yfinance blew up")

        monkeypatch.setitem(sys.modules, "yfinance",
                            _fake_yf_module(hist))
        assert dashboard._daily_history_cached("ERR") == []
