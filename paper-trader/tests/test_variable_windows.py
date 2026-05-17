"""Tests for variable backtest window selection + engine date plumbing."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

from run_continuous_backtests import _pick_window
from paper_trader.historical_collector import (
    _label_key,
    _parse_labels,
    _apply_labels,
)


# ─────────────────────────── _pick_window ───────────────────────────

class TestPickWindow:
    def test_same_seed_returns_identical_window(self):
        a_start, a_end = _pick_window(42)
        b_start, b_end = _pick_window(42)
        assert a_start == b_start
        assert a_end == b_end

    def test_different_seeds_give_different_windows(self):
        # Across a small sample, at least two seeds must yield different windows.
        # Probability of collision across 10 random seeds is vanishingly low.
        windows = {_pick_window(s) for s in range(10)}
        assert len(windows) > 1, "Different seeds should diverge"

    def test_duration_is_1_to_10_years(self):
        for seed in range(50):
            start, end = _pick_window(seed)
            days = (end - start).days
            # 1 yr = 365, 10 yr = 3650. Allow exact range.
            assert 365 <= days <= 10 * 365, (
                f"seed {seed}: duration {days}d not in [365, 3650]"
            )

    def test_window_ends_at_least_6_months_before_today(self):
        cutoff = date.today() - timedelta(days=180)
        for seed in range(50):
            _start, end = _pick_window(seed)
            assert end <= cutoff, (
                f"seed {seed}: end {end} not at least 6mo before today {date.today()}"
            )

    def test_window_starts_no_earlier_than_earliest(self):
        # Earliest start is SPY inception, 1993-02-01.
        for seed in range(50):
            start, _end = _pick_window(seed)
            assert start >= date(1993, 2, 1), (
                f"seed {seed}: start {start} before 1993-02-01"
            )


# ─────────────────────── engine date plumbing ───────────────────────

# yfinance is real-network. Patch it inside backtest's namespace so engine
# init returns synthetic OHLCV instead of calling out.
def _make_fake_hist(start: date, end: date):
    """Build a pandas DataFrame with a Close column for trading-day-ish dates."""
    import pandas as pd
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # business day
            days.append(cur)
        cur += timedelta(days=1)
    df = pd.DataFrame(
        {"Close": [100.0 + i * 0.1 for i in range(len(days))],
         "Volume": [1_000_000 + i * 1000 for i in range(len(days))]},
        index=pd.DatetimeIndex(days),
    )
    return df


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    """Redirect all backtest disk caches to a temp dir so tests don't pollute real cache."""
    import paper_trader.backtest as bt

    cache_dir = tmp_path / "backtest_cache"
    cache_dir.mkdir()
    gdelt_dir = cache_dir / "gdelt"
    gdelt_dir.mkdir()
    av_dir = cache_dir / "alphavantage"
    av_dir.mkdir()

    monkeypatch.setattr(bt, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(bt, "GDELT_CACHE", gdelt_dir)
    monkeypatch.setattr(bt, "AV_CACHE_DIR", av_dir)
    monkeypatch.setattr(bt, "AV_QUOTA_PATH", cache_dir / "av_quota.json")
    # Reset volume cache state so a previous test's data doesn't bleed in.
    monkeypatch.setattr(bt, "_VOLUME_CACHE", {})
    monkeypatch.setattr(bt, "_VOLUME_CACHE_DISK_LOADED", set())
    # Redirect backtest.db to a temp path too.
    monkeypatch.setattr(bt, "BACKTEST_DB", tmp_path / "backtest.db")
    # Tiny watchlist so test fakes stay cheap.
    monkeypatch.setattr(bt, "WATCHLIST", ["SPY", "AAPL"])

    yield cache_dir


class TestEngineWithCustomDates:
    def test_engine_initializes_with_custom_window(self, isolated_caches, monkeypatch):
        import paper_trader.backtest as bt

        # Fake the yfinance Ticker so PriceCache._load gets synthetic data.
        class _FakeTicker:
            def __init__(self, sym):
                self.sym = sym
            def history(self, start, end, auto_adjust):
                from datetime import date as _d
                s = _d.fromisoformat(start)
                e = _d.fromisoformat(end)
                return _make_fake_hist(s, e)

        monkeypatch.setattr(bt.yf, "Ticker", _FakeTicker)

        custom_start = date(2020, 1, 1)
        custom_end = date(2022, 1, 1)
        engine = bt.BacktestEngine(start=custom_start, end=custom_end)

        assert engine.start == custom_start
        assert engine.end == custom_end
        assert len(engine.prices.trading_days) > 0

    def test_backtest_run_stores_engine_dates_not_module_defaults(
        self, isolated_caches, monkeypatch
    ):
        """Sanity-check that the persisted DB row carries the engine's dates."""
        import paper_trader.backtest as bt

        class _FakeTicker:
            def __init__(self, sym):
                self.sym = sym
            def history(self, start, end, auto_adjust):
                from datetime import date as _d
                s = _d.fromisoformat(start)
                e = _d.fromisoformat(end)
                return _make_fake_hist(s, e)

        monkeypatch.setattr(bt.yf, "Ticker", _FakeTicker)

        custom_start = date(2020, 3, 1)
        custom_end = date(2020, 12, 31)
        engine = bt.BacktestEngine(start=custom_start, end=custom_end)
        engine.store.upsert_run(
            run_id=999, seed=1, status="running",
            start=custom_start, end=custom_end,
        )
        row = engine.store.conn.execute(
            "SELECT start_date, end_date FROM backtest_runs WHERE run_id=999"
        ).fetchone()
        assert row["start_date"] == custom_start.isoformat()
        assert row["end_date"] == custom_end.isoformat()


# ───────────────────── historical_collector pure logic ─────────────────────

class TestParseLabels:
    def test_basic_pipe_separated(self):
        raw = "0|7.5|1\n1|3.0|0\n2|0|0"
        assert _parse_labels(raw, expected=3) == [(7.5, 1), (3.0, 0), (0.0, 0)]

    def test_missing_lines_fallback_to_zero(self):
        raw = "0|5|1"
        # expected=3 → indices 1 and 2 get the (0.0, 0) fallback.
        result = _parse_labels(raw, expected=3)
        assert result == [(5.0, 1), (0.0, 0), (0.0, 0)]

    def test_relevance_clamped_to_range(self):
        raw = "0|99|1\n1|-5|0"
        assert _parse_labels(raw, expected=2) == [(10.0, 1), (0.0, 0)]

    def test_urgency_clamped_to_0_or_1(self):
        # Regex only accepts 0 or 1 in the urgency field; anything else is
        # treated as "no valid label" → fallback (0, 0).
        raw = "0|5|2"
        assert _parse_labels(raw, expected=1) == [(0.0, 0)]

    def test_garbage_lines_ignored(self):
        raw = "not a label\n\nrandom text\n0|4|1"
        assert _parse_labels(raw, expected=1) == [(4.0, 1)]


class TestApplyLabels:
    def test_keeps_existing_ai_score(self):
        articles = [{"title": "x", "source": "s", "ai_score": 9.9}]
        # Even if a label exists, an article with ai_score is left alone.
        labels = {_label_key(articles[0]): (1.0, 0)}
        out = _apply_labels(articles, labels)
        assert out[0]["ai_score"] == 9.9

    def test_fills_in_missing_score(self):
        articles = [{"title": "x", "source": "s"}]
        labels = {_label_key(articles[0]): (4.2, 1)}
        out = _apply_labels(articles, labels)
        assert out[0]["ai_score"] == 4.2
        assert out[0]["urgency"] == 1

    def test_unlabeled_article_unchanged(self):
        articles = [{"title": "x", "source": "s"}]
        out = _apply_labels(articles, labels={})
        assert "ai_score" not in out[0]


class TestLabelKey:
    def test_same_title_and_source_collide(self):
        a = {"title": "NVDA beats", "source": "reuters"}
        b = {"title": "NVDA beats", "source": "reuters"}
        assert _label_key(a) == _label_key(b)

    def test_different_titles_differ(self):
        a = {"title": "NVDA beats", "source": "reuters"}
        b = {"title": "NVDA misses", "source": "reuters"}
        assert _label_key(a) != _label_key(b)


# ────────────────── SEC EDGAR data flows into engine ──────────────────

class TestSecCacheConsumed:
    """Critical: collector data must actually feed the engine. Without this
    test, the SEC fetch could silently write to disk while pre-2015 backtests
    run purely on quant signals — exactly what the advisor caught."""

    def test_local_news_includes_sec_filings_in_window(self, isolated_caches,
                                                       monkeypatch):
        import paper_trader.backtest as bt

        # Synthetic price fake so engine init doesn't hit yfinance.
        class _FakeTicker:
            def __init__(self, sym):
                self.sym = sym
            def history(self, start, end, auto_adjust):
                from datetime import date as _d
                s = _d.fromisoformat(start)
                e = _d.fromisoformat(end)
                return _make_fake_hist(s, e)

        monkeypatch.setattr(bt.yf, "Ticker", _FakeTicker)

        # Seed a SEC EDGAR cache file BEFORE engine init.
        sec_dir = isolated_caches / "sec_edgar"
        sec_dir.mkdir()
        win_start = date(2010, 1, 1)
        win_end = date(2010, 6, 30)
        sec_file = sec_dir / f"AAPL_{win_start.isoformat()}_{win_end.isoformat()}.json"
        sec_file.write_text(json.dumps([
            {
                "title": "AAPL 8-K filing - March 15 2010",
                "url": "https://www.sec.gov/Archives/edgar/data/AAPL/0001.htm",
                "published": "2010-03-15",
                "source": "SEC/8-K/AAPL",
                "full_text": "AAPL 8-K filing - March 15 2010",
            },
            {
                # Out of window — must be filtered out by _merge_sec_cache.
                "title": "AAPL 10-K filing - January 2020",
                "url": "https://www.sec.gov/Archives/edgar/data/AAPL/0099.htm",
                "published": "2020-01-15",
                "source": "SEC/10-K/AAPL",
                "full_text": "AAPL 10-K filing - January 2020",
            },
        ]))

        # Point LOCAL_ARTICLES_DB at a nonexistent file so the local-DB load
        # path is a no-op and we test the SEC merge in isolation.
        monkeypatch.setattr(bt, "LOCAL_ARTICLES_DB",
                            isolated_caches / "no_such_db.db")

        engine = bt.BacktestEngine(start=win_start, end=win_end)

        in_window = engine._local_news.get("2010-03-15", [])
        assert any("8-K filing" in a["title"] for a in in_window), (
            "SEC filings inside window must be consumed by engine"
        )
        # The 2020-01-15 filing must NOT appear — it's outside the engine's
        # window and the engine would otherwise leak forward-looking signals.
        assert "2020-01-15" not in engine._local_news, (
            "SEC filings outside engine window must be filtered out"
        )

    def test_no_sec_cache_dir_is_tolerated(self, isolated_caches, monkeypatch):
        """Engine must initialize cleanly when sec_edgar/ doesn't exist —
        most cycles will never have it pre-warmed."""
        import paper_trader.backtest as bt

        class _FakeTicker:
            def __init__(self, sym):
                self.sym = sym
            def history(self, start, end, auto_adjust):
                from datetime import date as _d
                s = _d.fromisoformat(start)
                e = _d.fromisoformat(end)
                return _make_fake_hist(s, e)

        monkeypatch.setattr(bt.yf, "Ticker", _FakeTicker)
        monkeypatch.setattr(bt, "LOCAL_ARTICLES_DB",
                            isolated_caches / "no_such_db.db")
        # Explicitly do NOT create the sec_edgar/ directory.

        # Should not raise; _local_news just stays empty.
        engine = bt.BacktestEngine(start=date(2010, 1, 1), end=date(2010, 3, 1))
        assert engine._local_news == {}
