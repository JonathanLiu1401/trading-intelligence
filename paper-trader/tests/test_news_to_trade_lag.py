"""Unit tests for paper_trader.analytics.news_to_trade_lag.

The builder composes a build_trade_attribution result and aggregates per-
trade minimum lags into a distribution + verdict. Tests assert the verdict
ladder thresholds, the NO_ATTRIBUTION trump rule (>50% trades without news
beats the median verdict), p25/p75/median computation, freshest-article
selection (min minutes_before_trade across attributed list), per-trade
classification, and degrade-never-raise on garbage input.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.news_to_trade_lag import build_news_to_trade_lag


def _attrib_result(per_trade_rows):
    """Construct a fake build_trade_attribution result for testing."""
    return {"state": "OK", "trades": per_trade_rows}


def _trade_row(ticker, ts, action, attributed, n_attributed=None):
    return {
        "ticker": ticker,
        "timestamp": ts,
        "action": action,
        "attributed": attributed,
        "n_attributed": n_attributed if n_attributed is not None else len(attributed),
    }


def _article(title, ai_score, minutes_before):
    return {
        "title": title,
        "ai_score": ai_score,
        "minutes_before_trade": minutes_before,
    }


class TestStateLadder:
    def test_none_input_no_data(self):
        out = build_news_to_trade_lag(None)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "NO_DATA"
        assert out["n_trades"] == 0

    def test_empty_trades_no_data(self):
        out = build_news_to_trade_lag({"state": "OK", "trades": []})
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "NO_DATA"

    def test_trades_without_attribution_no_attribution_state(self):
        """All trades have empty attributed lists → NO_ATTRIBUTION."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY", []),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY", []),
        ])
        out = build_news_to_trade_lag(result)
        assert out["state"] == "NO_ATTRIBUTION"
        assert out["verdict"] == "NO_ATTRIBUTION"
        assert out["n_trades"] == 2
        assert out["n_no_attribution"] == 2
        assert out["n_attributed"] == 0
        assert out["median_lag_minutes"] is None


class TestVerdictLadder:
    def test_reactive_fast_under_30min(self):
        """Median lag <30min → REACTIVE_FAST."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("hot news", 9.0, 10.0)]),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("more news", 8.0, 20.0)]),
            _trade_row("MU", "2026-05-19T12:00:00+00:00", "BUY",
                       [_article("yet more", 7.0, 25.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["state"] == "OK"
        assert out["verdict"] == "REACTIVE_FAST"
        assert out["median_lag_minutes"] == 20.0
        assert out["bucket_fast"] == 3
        assert out["bucket_reactive"] == 0
        assert out["bucket_delayed"] == 0

    def test_reactive_30_to_120(self):
        """Median lag 30..120 → REACTIVE."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("news", 9.0, 60.0)]),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("news", 8.0, 90.0)]),
            _trade_row("MU", "2026-05-19T12:00:00+00:00", "BUY",
                       [_article("news", 7.0, 100.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["verdict"] == "REACTIVE"
        assert out["median_lag_minutes"] == 90.0
        assert out["bucket_reactive"] == 3
        assert out["bucket_fast"] == 0
        assert out["bucket_delayed"] == 0

    def test_delayed_over_120(self):
        """Median lag >120 → DELAYED."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("stale news", 9.0, 180.0)]),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("stale news", 8.0, 200.0)]),
            _trade_row("MU", "2026-05-19T12:00:00+00:00", "BUY",
                       [_article("stale news", 7.0, 220.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["verdict"] == "DELAYED"
        assert out["median_lag_minutes"] == 200.0
        assert out["bucket_delayed"] == 3


class TestNoAttributionTrump:
    def test_majority_no_attribution_overrides_median(self):
        """>50% trades without news → NO_ATTRIBUTION verdict even if some
        attributed trade has a FAST lag."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("news", 9.0, 10.0)]),  # 1 attributed
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY", []),
            _trade_row("MU", "2026-05-19T12:00:00+00:00", "BUY", []),
            _trade_row("TSLA", "2026-05-19T13:00:00+00:00", "BUY", []),
        ])
        out = build_news_to_trade_lag(result)
        # 1/4 attributed (25%) → 3/4 (75%) no-attribution > 50% floor
        assert out["verdict"] == "NO_ATTRIBUTION"
        assert out["state"] == "OK"  # we did process trades
        assert out["n_no_attribution"] == 3
        assert out["n_attributed"] == 1
        assert out["no_attribution_pct"] == 75.0

    def test_below_floor_keeps_numeric_verdict(self):
        """<50% no-attribution → numeric verdict applies."""
        # 1 of 3 (33%) without attribution — under the floor.
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("news", 9.0, 10.0)]),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("news", 8.0, 15.0)]),
            _trade_row("MU", "2026-05-19T12:00:00+00:00", "BUY", []),
        ])
        out = build_news_to_trade_lag(result)
        assert out["verdict"] == "REACTIVE_FAST"
        assert out["no_attribution_pct"] < 50.0


class TestFreshestSelection:
    def test_min_lag_taken_from_attributed_list(self):
        """When a trade has multiple attributed articles, min lag wins."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY", [
                _article("old news", 9.0, 200.0),
                _article("hot news", 5.0, 5.0),  # freshest
                _article("middle news", 7.0, 60.0),
            ]),
        ])
        out = build_news_to_trade_lag(result)
        # Freshest = 5 min, NOT the top-scored article's 200 min.
        assert out["min_lag_minutes"] == 5.0
        assert out["median_lag_minutes"] == 5.0
        # top_score still reports the highest ai_score (9.0)
        pt = out["per_trade"][0]
        assert pt["top_score"] == 9.0
        assert pt["min_lag_minutes"] == 5.0


class TestStats:
    def test_quantiles_computed(self):
        """min/p25/median/p75/max correctly identify the distribution."""
        # Lags: 10, 20, 30, 40, 50 — median 30, p25 20, p75 40
        result = _attrib_result([
            _trade_row(f"T{i}", f"2026-05-19T{i:02d}:00:00+00:00", "BUY",
                       [_article("n", 5.0, m)])
            for i, m in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])
        ])
        out = build_news_to_trade_lag(result)
        assert out["min_lag_minutes"] == 10.0
        assert out["max_lag_minutes"] == 50.0
        assert out["median_lag_minutes"] == 30.0
        # Nearest-rank p25 / p75: ceil(0.25 * 5) = 2 → idx 1 = 20;
        # ceil(0.75 * 5) = 4 → idx 3 = 40
        assert out["p25_lag_minutes"] == 20.0
        assert out["p75_lag_minutes"] == 40.0


class TestPerTradeRows:
    def test_each_trade_classified(self):
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T12:00:00+00:00", "BUY",
                       [_article("n", 9.0, 10.0)]),     # REACTIVE_FAST
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("n", 8.0, 60.0)]),     # REACTIVE
            _trade_row("MU", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("n", 7.0, 200.0)]),    # DELAYED
            _trade_row("TSLA", "2026-05-19T09:00:00+00:00", "BUY", []),
        ])
        out = build_news_to_trade_lag(result)
        by_ticker = {r["ticker"]: r for r in out["per_trade"]}
        assert by_ticker["NVDA"]["classification"] == "REACTIVE_FAST"
        assert by_ticker["AMD"]["classification"] == "REACTIVE"
        assert by_ticker["MU"]["classification"] == "DELAYED"
        assert by_ticker["TSLA"]["classification"] == "NO_ATTRIBUTION"

    def test_per_trade_sorted_newest_first(self):
        """per_trade newest trade_ts first."""
        result = _attrib_result([
            _trade_row("OLDEST", "2026-05-19T09:00:00+00:00", "BUY",
                       [_article("n", 9.0, 10.0)]),
            _trade_row("NEWEST", "2026-05-19T12:00:00+00:00", "BUY",
                       [_article("n", 9.0, 10.0)]),
            _trade_row("MIDDLE", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("n", 9.0, 10.0)]),
        ])
        out = build_news_to_trade_lag(result)
        tickers = [r["ticker"] for r in out["per_trade"]]
        assert tickers == ["NEWEST", "MIDDLE", "OLDEST"]


class TestVerdictBoundaries:
    def test_at_30min_boundary_is_reactive_not_fast(self):
        """A min_lag of exactly 30 min is REACTIVE (strict <)."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("n", 9.0, 30.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["per_trade"][0]["classification"] == "REACTIVE"

    def test_at_120min_boundary_is_delayed_not_reactive(self):
        """A min_lag of exactly 120 min is DELAYED (strict <)."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("n", 9.0, 120.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["per_trade"][0]["classification"] == "DELAYED"


class TestDegradeNeverRaise:
    def test_none_input(self):
        out = build_news_to_trade_lag(None)
        assert out["state"] == "NO_DATA"

    def test_non_dict_input(self):
        out = build_news_to_trade_lag("garbage")
        assert out["state"] == "NO_DATA"

    def test_trades_not_a_list(self):
        out = build_news_to_trade_lag({"state": "OK", "trades": "garbage"})
        assert out["state"] == "NO_DATA"

    def test_garbage_articles_skipped(self):
        """Non-dict articles inside attributed don't crash."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [None, "garbage", 123,
                        _article("real", 9.0, 5.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert out["state"] == "OK"
        assert out["min_lag_minutes"] == 5.0

    def test_negative_lag_rejected(self):
        """A negative minutes_before_trade is impossible (article AFTER fill);
        should be dropped, not used."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY", [
                _article("bogus", 9.0, -5.0),
            ]),
        ])
        out = build_news_to_trade_lag(result)
        # The only attributed article got dropped → no-attribution
        assert out["n_no_attribution"] == 1
        assert out["per_trade"][0]["classification"] == "NO_ATTRIBUTION"

    def test_none_minutes_before_handled(self):
        """A None minutes_before_trade is dropped, not exploded."""
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY", [
                {"title": "x", "ai_score": 9.0, "minutes_before_trade": None},
            ]),
        ])
        out = build_news_to_trade_lag(result)
        # That trade has no usable lag → no-attribution counted.
        assert out["n_no_attribution"] == 1


class TestHeadline:
    def test_headline_includes_median_and_count(self):
        result = _attrib_result([
            _trade_row("NVDA", "2026-05-19T10:00:00+00:00", "BUY",
                       [_article("n", 9.0, 15.0)]),
            _trade_row("AMD", "2026-05-19T11:00:00+00:00", "BUY",
                       [_article("n", 8.0, 20.0)]),
        ])
        out = build_news_to_trade_lag(result)
        assert "min" in out["headline"]
        # Median value present
        assert "17" in out["headline"] or "18" in out["headline"]  # 17.5 rounds
