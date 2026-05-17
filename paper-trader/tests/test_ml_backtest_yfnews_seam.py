"""Regression locks for `BacktestEngine._fetch_yf_news` — a previously
zero-coverage seam (grepped every backtest symbol against tests/: no test
file referenced `_fetch_yf_news`).

Why this seam matters: `_fetch_yf_news` is the Tier-2 news source in
`_fetch_signals`. Its `providerPublishTime > sim_end_ts → skip` line is a
**load-bearing anti-contamination invariant**: a yfinance headline published
*after* the simulated date must never enter the article set, because those
articles flow into `_compute_decision_outcomes` → `decision_outcomes.jsonl` →
`DecisionScorer` training. If future news leaked in, the scorer would be
trained on information that did not exist at decision time — silent
forward-leakage that inflates backtest skill and is nearly impossible to
detect from the outside. A refactor that flips the `>` to `<`, drops the
`isinstance` numeric guard, or removes the `sim_date < cutoff` short-circuit
would reintroduce exactly that contamination. These tests pin the behaviour
with exact-value assertions, not ranges.

`_fetch_yf_news` references no instance attributes, so it is exercised via
`BacktestEngine.__new__` (no `__init__` → no PriceCache → no network).
`yfinance.Ticker` is mocked; the whole module is offline and deterministic.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import paper_trader.backtest as bt
from paper_trader.backtest import BacktestEngine


def _engine() -> BacktestEngine:
    # __new__ skips __init__ (PriceCache download / DB connect). _fetch_yf_news
    # is self-independent, so a bare instance is sufficient.
    return BacktestEngine.__new__(BacktestEngine)


def _sim_end_ts(sim_date: date) -> int:
    """Recompute the cutoff the production code uses verbatim."""
    return int(datetime(sim_date.year, sim_date.month, sim_date.day,
                         23, 59, 59, tzinfo=timezone.utc).timestamp())


def _patch_yf(monkeypatch, news_list: list[dict]) -> MagicMock:
    """Make `bt.yf.Ticker(<anything>).news` return `news_list`. Returns the
    Ticker mock so the caller can assert call counts."""
    ticker_obj = MagicMock()
    ticker_obj.news = news_list
    ticker_factory = MagicMock(return_value=ticker_obj)
    monkeypatch.setattr(bt.yf, "Ticker", ticker_factory)
    return ticker_factory


class TestForwardLeakGuard:
    """The core invariant: news timestamped after sim_date's end-of-day UTC
    must be excluded; on/before is kept."""

    def test_future_news_excluded_boundary_is_strict_gt(self, monkeypatch):
        sim_date = date.today()  # always >= today-30 cutoff
        end_ts = _sim_end_ts(sim_date)

        news = [
            # A: an hour before EOD → KEPT
            {"title": "NVDA earnings beat strong demand",
             "link": "https://x/a", "providerPublishTime": end_ts - 3600},
            # B: exactly at the EOD boundary → KEPT (guard is strict `>`)
            {"title": "AMD guidance raised rally",
             "link": "https://x/b", "providerPublishTime": end_ts},
            # C: one second past EOD → EXCLUDED (forward leakage)
            {"title": "MU record revenue surge",
             "link": "https://x/c", "providerPublishTime": end_ts + 1},
        ]
        _patch_yf(monkeypatch, news)

        out = _engine()._fetch_yf_news(["NVDA"], sim_date)
        urls = {a["url"] for a in out}

        # C is the forward-leak case — it must NOT be present.
        assert "https://x/c" not in urls, "future-dated news leaked forward"
        # A and B straddle the boundary and must both survive.
        assert urls == {"https://x/a", "https://x/b"}
        assert len(out) == 2
        # Every returned record carries the heuristic-scored shape.
        for a in out:
            assert set(a) == {"title", "url", "score", "tickers"}
            assert 0.0 <= a["score"] <= 5.0

    def test_missing_or_nonnumeric_timestamp_is_kept(self, monkeypatch):
        # The guard only excludes when pub_ts is *numeric* AND > cutoff.
        # A None / string timestamp must fall through and be KEPT — otherwise
        # legitimate undated headlines would be silently dropped.
        sim_date = date.today()
        news = [
            {"title": "NVDA upgrade outperform", "link": "https://x/none",
             "providerPublishTime": None},
            {"title": "AMD breakout buy rating", "link": "https://x/str",
             "providerPublishTime": "not-a-timestamp"},
            {"title": "MU missing ts key entirely", "link": "https://x/absent"},
        ]
        _patch_yf(monkeypatch, news)

        out = _engine()._fetch_yf_news(["NVDA"], sim_date)
        assert {a["url"] for a in out} == {
            "https://x/none", "https://x/str", "https://x/absent"}


class TestDedupAndEmptyTitle:
    def test_empty_title_skipped_and_url_deduped(self, monkeypatch):
        sim_date = date.today()
        end_ts = _sim_end_ts(sim_date)
        news = [
            {"title": "NVDA earnings beat", "link": "https://x/dup",
             "providerPublishTime": end_ts - 10},
            {"title": "", "link": "https://x/empty",
             "providerPublishTime": end_ts - 10},          # empty title → skip
            {"title": "NVDA second copy", "link": "https://x/dup",
             "providerPublishTime": end_ts - 5},            # dup url → skip
        ]
        _patch_yf(monkeypatch, news)

        out = _engine()._fetch_yf_news(["NVDA"], sim_date)
        assert [a["url"] for a in out] == ["https://x/dup"]
        assert out[0]["title"] == "NVDA earnings beat"  # first wins, not the dup


class TestStaleDateShortCircuit:
    def test_sim_date_older_than_30d_returns_empty_without_network(self, monkeypatch):
        # The `sim_date < date.today() - 30d` short-circuit prevents any
        # yfinance call for historical windows (the continuous loop's windows
        # always end >=180d before today). A regression that removes it would
        # both hit the network in offline backtests and risk leakage.
        old = date.today() - timedelta(days=40)
        factory = _patch_yf(monkeypatch, [
            {"title": "should never be read", "link": "https://x/never",
             "providerPublishTime": _sim_end_ts(old)},
        ])
        out = _engine()._fetch_yf_news(["NVDA", "SPY"], old)
        assert out == []
        factory.assert_not_called()  # no yf.Ticker(...) construction at all

    def test_ticker_fetch_exception_is_swallowed(self, monkeypatch):
        # Per-ticker yfinance failure must degrade to "no articles from that
        # ticker", never propagate (it runs inside the run thread's day loop).
        sim_date = date.today()
        boom = MagicMock(side_effect=RuntimeError("yf down"))
        monkeypatch.setattr(bt.yf, "Ticker", boom)
        out = _engine()._fetch_yf_news(["NVDA"], sim_date)
        assert out == []
