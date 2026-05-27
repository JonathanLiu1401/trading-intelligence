"""Pure-builder tests for analytics/ticker_news_burst_runner.py.

The verdict ladder, baseline-per-h floor, sort order, and shape MUST mirror
``ArticleStore.ticker_news_burst`` byte-for-byte (the daemon's in-process
counterpart, pinned by tests/test_ticker_news_burst.py). The endpoint reuses
this builder to avoid racing the writer connection (see ``_ro_query``'s
docstring).

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.ticker_news_burst_runner import (  # noqa: E402
    BLAZING_SPIKE,
    HOT_SPIKE,
    WARMING_SPIKE,
    build_ticker_news_burst,
    _normalise_tickers,
)


NOW = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)


def _at(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _a(title: str, minutes_ago: float) -> dict:
    return {"title": title, "first_seen": _at(minutes_ago)}


class TestNormaliseTickers:
    def test_uppercases_and_strips(self):
        assert _normalise_tickers(["nvda", " mu ", "AMD"]) == ["NVDA", "MU", "AMD"]

    def test_drops_too_short(self):
        assert _normalise_tickers(["A", "MU"]) == ["MU"]

    def test_drops_too_long(self):
        assert _normalise_tickers(["VERYLONGTICKER", "MU"]) == ["MU"]

    def test_drops_empty(self):
        assert _normalise_tickers(["", None, "MU"]) == ["MU"]

    def test_dedupes_preserving_order(self):
        assert _normalise_tickers(["MU", "NVDA", "MU"]) == ["MU", "NVDA"]


class TestBuilder:
    def test_empty_universe_returns_no_data(self):
        res = build_ticker_news_burst([], tickers=[], now=NOW)
        assert res["verdict"] == "NO_DATA"
        assert res["by_ticker"] == []
        assert res["hottest"] is None

    def test_all_cold_when_no_articles(self):
        res = build_ticker_news_burst([], tickers=["NVDA", "MU"], now=NOW)
        verdicts = {r["ticker"]: r["verdict"] for r in res["by_ticker"]}
        assert verdicts == {"NVDA": "COLD", "MU": "COLD"}
        assert res["verdict"] == "NO_DATA"
        assert res["hottest"] is None
        assert res["n_hot"] == 0

    def test_blazing_when_window_overwhelms_baseline(self):
        """5 fresh NVDA in window, 1 in 24h baseline. base_per_h = 1/24 ≈
        0.04, floored to 0.5 → spike = 5/0.5 = 10.0 → BLAZING."""
        articles = [_a("NVDA hits high", 30 + i) for i in range(5)]
        articles.append(_a("NVDA quarterly outlook", 60 * 5))
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"],
            window_h=1.0, baseline_h=24.0, now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        assert nvda["count_window"] == 5
        assert nvda["count_baseline"] == 1
        assert nvda["spike"] == pytest.approx(BLAZING_SPIKE, abs=0.01)
        assert nvda["verdict"] == "BLAZING"
        assert res["verdict"] == "BLAZING"
        assert "NVDA" in res["headline"]
        assert res["hottest"] == "NVDA"
        assert res["n_hot"] == 1

    def test_hot_threshold(self):
        """3 in window, 1 in baseline → spike = 3/0.5 = 6 (≥5) and cw=3 →
        HOT (NOT BLAZING — cw<5)."""
        articles = [_a("MU beats", 15 + i) for i in range(3)]
        articles.append(_a("MU thesis", 60 * 8))
        res = build_ticker_news_burst(
            articles, tickers=["MU"], window_h=1.0, baseline_h=24.0, now=NOW,
        )
        mu = next(r for r in res["by_ticker"] if r["ticker"] == "MU")
        assert mu["verdict"] == "HOT"
        assert res["verdict"] == "HOT"
        assert mu["spike"] >= HOT_SPIKE

    def test_warming_threshold(self):
        """2 in window. 4 in 24h baseline → 4/24 ≈ 0.167 → floor 0.5 →
        spike = 2/0.5 = 4 (≥2, <5) and cw=2 → WARMING."""
        articles = [_a("QBTS earnings", 20 + i) for i in range(2)]
        articles.extend([_a("QBTS history", 60 * (i + 2)) for i in range(4)])
        res = build_ticker_news_burst(
            articles, tickers=["QBTS"], window_h=1.0, baseline_h=24.0, now=NOW,
        )
        qbts = next(r for r in res["by_ticker"] if r["ticker"] == "QBTS")
        assert qbts["verdict"] == "WARMING"
        assert qbts["spike"] >= WARMING_SPIKE
        assert qbts["spike"] < HOT_SPIKE
        assert res["verdict"] == "WARMING"

    def test_normal_when_rate_matches_baseline(self):
        """Window matches the baseline per-hour rate → NORMAL."""
        articles = [_a("NVDA update", 20)]
        # 23 mentions spread across the 23h baseline window (-2h..-25h)
        articles.extend([_a("NVDA update", 60 * (i + 2)) for i in range(23)])
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"], window_h=1.0, baseline_h=24.0, now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        assert nvda["count_window"] == 1
        assert nvda["count_baseline"] == 23
        # spike = 1 / max(23/24, 0.5) ≈ 1.04 → NORMAL (below WARMING≥2)
        assert nvda["spike"] == pytest.approx(1.04, abs=0.05)
        assert nvda["verdict"] == "NORMAL"
        # No held ticker breaks out → top-level NORMAL (not NO_DATA)
        assert res["verdict"] == "NORMAL"

    def test_sort_by_spike_then_count(self):
        articles = []
        # NVDA: 5 in window, 1 in baseline → BLAZING spike 10
        articles.extend([_a("NVDA news", 10 + i) for i in range(5)])
        articles.append(_a("NVDA history", 180))
        # MU: 3 in window, 1 in baseline → HOT spike 6
        articles.extend([_a("MU news", 10 + i) for i in range(3)])
        articles.append(_a("MU history", 200))
        res = build_ticker_news_burst(
            articles, tickers=["MU", "NVDA"], now=NOW,
        )
        order = [r["ticker"] for r in res["by_ticker"]]
        assert order[0] == "NVDA"
        assert order[1] == "MU"
        assert res["hottest"] == "NVDA"
        assert res["n_hot"] == 2

    def test_word_boundary_no_substring_match(self):
        """AUTOMATIC must not match MAT or AMAT."""
        articles = [
            _a("AMAT reports quarterly revenue beat", 5),
            _a("AUTOMATIC SHIPMENTS RESUME", 10),
        ]
        res = build_ticker_news_burst(
            articles, tickers=["AMAT"], now=NOW,
        )
        amat = next(r for r in res["by_ticker"] if r["ticker"] == "AMAT")
        assert amat["count_window"] == 1

    def test_dollar_prefix_ticker_matches(self):
        res = build_ticker_news_burst(
            [_a("$NVDA breaking out on volume", 5)],
            tickers=["NVDA"], now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        assert nvda["count_window"] == 1

    def test_zero_baseline_zero_window_spike_is_none(self):
        res = build_ticker_news_burst([], tickers=["XYZ"], now=NOW)
        xyz = next(r for r in res["by_ticker"] if r["ticker"] == "XYZ")
        assert xyz["spike"] is None
        assert xyz["verdict"] == "COLD"

    def test_baseline_excludes_window(self):
        """A baseline window's right edge is the window's left edge; the same
        rows must NOT count in both buckets."""
        # 5 in window only, 0 in baseline
        articles = [_a("NVDA breaking", 10 + i) for i in range(5)]
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"], window_h=1.0, baseline_h=24.0, now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        assert nvda["count_baseline"] == 0
        assert nvda["count_window"] == 5
        assert nvda["verdict"] == "BLAZING"

    def test_future_dated_rows_excluded(self):
        """An article with ``first_seen`` in the future (clock skew) must
        not count toward either window — a wire-source can produce these and
        they would otherwise inflate the window count above the actual rate."""
        future = (NOW + timedelta(minutes=5)).isoformat()
        articles = [
            {"title": "NVDA future-dated", "first_seen": future},
            _a("NVDA real", 5),
        ]
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"], now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        # Only the one real mention should count.
        assert nvda["count_window"] == 1

    def test_unparseable_timestamp_skipped(self):
        """Bad first_seen value must be skipped, not raise."""
        articles = [
            {"title": "NVDA garbage ts", "first_seen": "not-a-date"},
            _a("NVDA real", 5),
        ]
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"], now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        assert nvda["count_window"] == 1

    def test_window_floor_at_005h(self):
        """An absurdly small window_h must floor at 0.05h (3 minutes) so
        the time-window arithmetic doesn't degenerate."""
        res = build_ticker_news_burst(
            [], tickers=["NVDA"], window_h=0.0, baseline_h=1.0, now=NOW,
        )
        assert res["window_h"] >= 0.05

    def test_baseline_minimum_15x_window(self):
        """baseline_h floors at 1.5 × window_h so the baseline window is
        always wider than the window itself (otherwise per-hour rate
        comparison is meaningless)."""
        res = build_ticker_news_burst(
            [], tickers=["NVDA"], window_h=4.0, baseline_h=1.0, now=NOW,
        )
        assert res["baseline_h"] >= 4.0 * 1.5

    def test_headline_describes_blazing_top(self):
        articles = [_a("NVDA hit", 30 + i) for i in range(5)]
        articles.append(_a("NVDA hit", 60 * 5))
        res = build_ticker_news_burst(
            articles, tickers=["NVDA"], window_h=1.0, baseline_h=24.0, now=NOW,
        )
        # The headline must encode (a) the verdict tier, (b) the top ticker,
        # (c) the window count, (d) the spike multiple — so a chat helper
        # can pass it verbatim.
        h = res["headline"]
        assert "BLAZING" in h
        assert "NVDA" in h
        assert "5" in h           # count_window
        assert "10" in h          # spike (rounded to 10.0)

    def test_n_hot_only_counts_hot_blazing(self):
        """n_hot must NOT include WARMING — mirrors the storage method's
        contract (the chat helper relies on it to gate the silence-on-WARMING
        threshold case)."""
        articles = []
        # WARMING NVDA
        articles.extend([_a("NVDA news", 20 + i) for i in range(2)])
        articles.extend([_a("NVDA history", 60 * (i + 2)) for i in range(4)])
        # HOT MU
        articles.extend([_a("MU news", 15 + i) for i in range(3)])
        articles.append(_a("MU thesis", 60 * 8))
        res = build_ticker_news_burst(
            articles, tickers=["NVDA", "MU"], now=NOW,
        )
        nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
        mu = next(r for r in res["by_ticker"] if r["ticker"] == "MU")
        assert nvda["verdict"] == "WARMING"
        assert mu["verdict"] == "HOT"
        assert res["n_hot"] == 1  # MU only — WARMING does not count
