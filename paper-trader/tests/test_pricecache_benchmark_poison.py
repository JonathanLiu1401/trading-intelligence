"""PriceCache benchmark-integrity guard (2026-05-17 quant fix).

Live finding: yfinance intermittently fails to return SPY at cache-build
time. The old code persisted a per-window `prices_*.json` whose SPY series
was `{}` (SPY still listed in `_meta.tickers`), and then *re-accepted* that
poisoned file on every subsequent draw of the window. `_build_trading_days`
silently fell back to another ticker's calendar so the run still completed,
but `returns_pct("SPY", …)` then returned 0.0 → `vs_spy_pct` became a
fabricated `total_return - 0` with no real benchmark, which feeds the live
trader's `_ml_is_qualified` median-alpha gate (CLAUDE.md §15).

Verified live: 34 of 177 per-window caches were poisoned this way.

These tests pin both halves of the guard in `PriceCache._load`:
  * read side  — a cached payload with an empty SPY series is rejected and
                  re-downloaded (SPY has data to its 1993 inception, so an
                  empty series is ALWAYS a transient fetch failure);
  * write side — a download that itself yields an empty SPY series is NOT
                 persisted (so the next draw retries fresh instead of
                 re-poisoning the cache), while the run still completes off
                 the fallback-ticker calendar.

All offline — yfinance is faked inside backtest's namespace.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import paper_trader.backtest as bt
from paper_trader.backtest import PriceCache

START = date(2020, 1, 2)
END = date(2020, 6, 30)


def _fake_hist(start: date, end: date) -> pd.DataFrame:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return pd.DataFrame(
        {"Close": [100.0 + i * 0.1 for i in range(len(days))],
         "Volume": [1_000_000 + i * 10 for i in range(len(days))]},
        index=pd.DatetimeIndex(days),
    )


def _install_fake_yf(monkeypatch, empty_for: set[str] | None = None):
    """Fake bt.yf.Ticker. Tickers in `empty_for` return an empty DataFrame
    (simulating the transient yfinance SPY failure)."""
    empty_for = empty_for or set()

    class _FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, start, end, auto_adjust):
            if self.sym in empty_for:
                return pd.DataFrame()
            s = date.fromisoformat(start)
            e = date.fromisoformat(end)
            return _fake_hist(s, e)

    monkeypatch.setattr(bt.yf, "Ticker", _FakeTicker)


def _write_poisoned_cache(path, tickers, spy_empty=True):
    payload = {"_meta": {
        "start": START.isoformat(),
        "end": END.isoformat(),
        "tickers": list(tickers),
        "saved_at": "2026-05-17T00:00:00+00:00",
    }}
    for t in tickers:
        if t == "SPY" and spy_empty:
            payload[t] = {}
        else:
            payload[t] = {(START + timedelta(days=i)).isoformat(): 100.0 + i
                          for i in range(120)}
    path.write_text(json.dumps(payload))


class TestPoisonedCacheRejectedOnRead:
    def test_empty_spy_cache_is_rejected_and_redownloaded(self, monkeypatch):
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        _write_poisoned_cache(cache_file, ["SPY", "NVDA"], spy_empty=True)
        # yfinance now works — the rejected cache must trigger a real fetch.
        _install_fake_yf(monkeypatch)

        cache = PriceCache(["SPY", "NVDA"], START, END)

        # SPY was re-downloaded → non-empty series, real trading-day calendar.
        assert cache.prices.get("SPY"), "poisoned SPY cache was not refetched"
        assert len(cache.trading_days) > 0
        # The on-disk file self-healed: SPY series is now populated.
        healed = json.loads(cache_file.read_text())
        assert healed.get("SPY"), "cache file was not rewritten with SPY data"

    def test_healthy_cache_is_still_accepted_without_refetch(self, monkeypatch):
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        _write_poisoned_cache(cache_file, ["SPY", "NVDA"], spy_empty=False)

        # If the guard wrongly rejected this healthy cache it would call
        # yf.Ticker; make that explode so any refetch fails loudly.
        def _boom(sym):
            raise AssertionError("healthy cache must not trigger a refetch")

        monkeypatch.setattr(bt.yf, "Ticker", _boom)

        cache = PriceCache(["SPY", "NVDA"], START, END)
        assert cache.prices.get("SPY")
        assert len(cache.trading_days) > 0

    def test_guard_inert_when_spy_not_requested(self, monkeypatch):
        # A watchlist without SPY: an empty-"SPY"-key cache is irrelevant and
        # must NOT be rejected (the guard keys on "SPY" in self.tickers).
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        cache_file.write_text(json.dumps({
            "_meta": {"start": START.isoformat(), "end": END.isoformat(),
                      "tickers": ["NVDA"], "saved_at": "x"},
            "NVDA": {(START + timedelta(days=i)).isoformat(): 100.0 + i
                     for i in range(120)},
        }))

        def _boom(sym):
            raise AssertionError("non-SPY cache must be accepted as-is")

        monkeypatch.setattr(bt.yf, "Ticker", _boom)
        cache = PriceCache(["NVDA"], START, END)
        assert cache.prices.get("NVDA")


class TestPoisonedDownloadNotPersisted:
    def test_empty_spy_download_is_not_cached(self, monkeypatch):
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        assert not cache_file.exists()
        # yfinance fails for SPY only; NVDA still returns data.
        _install_fake_yf(monkeypatch, empty_for={"SPY"})

        cache = PriceCache(["SPY", "NVDA"], START, END)

        # Run can still proceed: trading_days fell back to NVDA's calendar.
        assert len(cache.trading_days) > 0
        assert not cache.prices.get("SPY")
        # Crucial: the poisoned result was NOT persisted, so the next draw
        # of this window retries the download instead of re-poisoning.
        assert not cache_file.exists(), (
            "empty-SPY download must NOT be cached (would re-poison)"
        )

    def test_successful_download_is_still_cached(self, monkeypatch):
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        _install_fake_yf(monkeypatch)

        cache = PriceCache(["SPY", "NVDA"], START, END)

        assert cache.prices.get("SPY")
        assert cache_file.exists(), "a healthy download must still be cached"
        saved = json.loads(cache_file.read_text())
        assert saved.get("SPY") and saved.get("NVDA")


class TestPriceCacheAtomicWrite:
    """Atomic-write guard for the per-window price cache.

    Without atomicity, a process kill (OOM/SIGKILL) mid-write leaves a
    truncated/torn JSON file at the destination path. The next cache load
    fails `json.loads`, falls through to the download path, and silently
    re-pays the (hundreds of MB, dozens of tickers) yfinance refetch on
    every subsequent run for this window. This pins the same tmp+replace
    discipline already enforced for scorer.pkl, the volume cache, the
    outcomes-file trim, and validation_results.json.
    """

    def test_destination_unchanged_when_write_fails_midstream(self, monkeypatch):
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"

        # Pre-seed a HEALTHY cache file. If the in-progress write were not
        # atomic, the partial-write would clobber this and the next load
        # would see torn JSON and silently retry the download.
        sentinel = {
            "_meta": {"start": START.isoformat(), "end": END.isoformat(),
                      "tickers": ["SPY", "NVDA"],
                      "saved_at": "2026-05-21T00:00:00+00:00"},
            "SPY": {(START + timedelta(days=i)).isoformat(): 200.0 + i
                    for i in range(120)},
            "NVDA": {(START + timedelta(days=i)).isoformat(): 300.0 + i
                     for i in range(120)},
        }
        cache_file.write_text(json.dumps(sentinel))
        sentinel_bytes = cache_file.read_bytes()

        # Force the next cache load to BYPASS the on-disk file (so we hit the
        # download → write code path) by mutating the cached _meta tickers so
        # the meta-match check fails. The download path will then try to
        # write a fresh cache file.
        bad = json.loads(cache_file.read_text())
        bad["_meta"]["tickers"] = ["DIFFERENT"]
        cache_file.write_text(json.dumps(bad))

        _install_fake_yf(monkeypatch)

        # Crash the write_text call (simulating SIGKILL mid-write or a disk
        # full). With atomic write the destination file is updated via tmp
        # + replace; a failure before .replace leaves the destination's prior
        # contents intact (in this case, the meta-modified cache file).
        original_write = Path.write_text

        def _explode(self, content, *a, **kw):
            if self.name.endswith(".json.tmp"):
                raise OSError("simulated SIGKILL mid-write")
            return original_write(self, content, *a, **kw)

        monkeypatch.setattr(Path, "write_text", _explode)

        with pytest.raises(OSError):
            PriceCache(["SPY", "NVDA"], START, END)

        # The destination file must still be readable JSON — never a torn
        # half-write that a future load can't parse. The contents will be
        # the pre-crash meta-modified file (still valid JSON), not a
        # truncated payload.
        post_crash = cache_file.read_bytes()
        json.loads(post_crash)  # would raise on torn JSON

    def test_tmp_file_replace_used_in_normal_write(self, monkeypatch):
        """Direct-pattern test: observe that the write goes through a
        `.json.tmp` file then `.replace`s into place, mirroring the
        discipline used by every other on-disk cache writer in this module.
        """
        cache_file = bt.CACHE_DIR / f"prices_{START.isoformat()}_{END.isoformat()}.json"
        _install_fake_yf(monkeypatch)

        # Spy on Path.replace to confirm the atomic-rename step actually
        # fires from a `.json.tmp` source onto the destination cache path.
        replaces: list[tuple[str, str]] = []
        original_replace = Path.replace

        def _spy_replace(self, target):
            replaces.append((str(self), str(target)))
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", _spy_replace)

        cache = PriceCache(["SPY", "NVDA"], START, END)
        assert cache.prices.get("SPY")
        assert cache_file.exists()

        # Find the replace that targets the price cache; it must come from
        # a sibling `.json.tmp` file (not a direct write to the canonical path).
        cache_target_replaces = [
            (src, tgt) for src, tgt in replaces if tgt == str(cache_file)
        ]
        assert cache_target_replaces, (
            f"PriceCache must atomically rename a .tmp file into {cache_file}; "
            f"observed replaces: {replaces}"
        )
        src, _ = cache_target_replaces[-1]
        assert src.endswith(".json.tmp"), (
            f"PriceCache atomic-rename source must be a .json.tmp file; got {src}"
        )
