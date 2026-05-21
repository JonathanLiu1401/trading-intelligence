"""Tests for paper_trader.analytics.conviction_deployment.

Pins:
* the BUY-only filter (SELLs, options, non-positive value, malformed
  rows all degrade — never crash)
* the (trade_ts - window, trade_ts] peak-score window — articles
  exactly at the trade_ts are included; articles at the open edge
  are excluded
* equity-curve interpolation (nearest before-or-equal; starting
  capital fallback for trades pre-dating the curve)
* word-boundary ticker matching against article titles (MU does not
  match Museum; the convention mirrors briefing_coverage_audit)
* bucket cuts (<6, 6-7, 7-8, 8-9, 9+) including the high-cut boundary
* MONOTONIC / FLAT / INVERTED / INSUFFICIENT roll-up gated on
  per-bucket density and minimum populated buckets
* NO_DATA / EMERGING / STABLE state envelope keys present in every
  branch
* score_unavailable count for trades with no article matches
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.conviction_deployment import (
    STABLE_MIN_PER_BUCKET,
    STABLE_MIN_POPULATED_BUCKETS,
    MONOTONIC_SLOPE_PP,
    build_conviction_deployment,
)


def _trade(trade_id, action, ticker, *, qty, price, ts):
    return {
        "id": trade_id,
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "option_type": None,
        "type": "stock",
    }


def _article(title, score, ts, *, source="rss", urgency=1):
    return {
        "title": title,
        "ai_score": score,
        "first_seen": ts,
        "source": source,
        "urgency": urgency,
        "url": "https://example.com",
    }


def _equity(ts, total_value):
    return {"timestamp": ts, "total_value": total_value, "cash": 0.0}


_ENVELOPE_KEYS = {
    "as_of", "state", "verdict", "headline",
    "n_buys_scanned", "n_buys_with_score", "n_buys_score_unavailable",
    "window_hours_pre_trade", "stable_min_per_bucket",
    "stable_min_populated_buckets", "monotonic_slope_pp",
    "buckets", "evidence",
}


class TestEnvelopeStability:
    def test_no_data_keys(self):
        out = build_conviction_deployment([], [], [])
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "INSUFFICIENT"
        assert out["n_buys_scanned"] == 0
        assert len(out["buckets"]) == 5  # five buckets, always present
        for b in out["buckets"]:
            assert b["n_buys"] == 0

    def test_emerging_keys(self):
        # One BUY with a matching article — n_with_score=1 < threshold
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T01:00:00+00:00")]
        articles = [_article("NVDA crushes earnings", 9.0,
                             "2026-05-21T00:30:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["state"] == "EMERGING"
        assert out["n_buys_scanned"] == 1
        assert out["n_buys_with_score"] == 1
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["bucket"] == "9+"

    def test_defensive_non_dict_trades(self):
        # Garbage rows must not raise; the result envelope must still
        # be present.
        out = build_conviction_deployment(
            [None, "garbage", {"bad": "row"}, 42], [], [],
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["state"] == "NO_DATA"


class TestBuyOnlyFilter:
    def test_sells_excluded(self):
        trades = [
            _trade(1, "BUY", "NVDA", qty=1, price=100.0,
                   ts="2026-05-21T01:00:00+00:00"),
            _trade(2, "SELL", "NVDA", qty=1, price=110.0,
                   ts="2026-05-21T02:00:00+00:00"),
        ]
        out = build_conviction_deployment(trades, [], [])
        assert out["n_buys_scanned"] == 1
        assert out["evidence"][0]["action"] == "BUY"

    def test_option_legs_excluded(self):
        # BUY_CALL / BUY_PUT must not enter the curve — option
        # notional doesn't map to share-dollar deployment.
        trades = [
            {
                "id": 1, "timestamp": "2026-05-21T01:00:00+00:00",
                "ticker": "NVDA", "action": "BUY", "qty": 1,
                "price": 100.0, "value": 100.0,
                "option_type": "CALL", "type": "option",
            },
            _trade(2, "BUY", "AAPL", qty=2, price=50.0,
                   ts="2026-05-21T01:30:00+00:00"),
        ]
        out = build_conviction_deployment(trades, [], [])
        assert out["n_buys_scanned"] == 1
        assert out["evidence"][0]["ticker"] == "AAPL"

    def test_non_positive_value_excluded(self):
        trades = [
            _trade(1, "BUY", "NVDA", qty=0, price=100.0,
                   ts="2026-05-21T01:00:00+00:00"),
            _trade(2, "BUY", "NVDA", qty=1, price=0.0,
                   ts="2026-05-21T01:30:00+00:00"),
            _trade(3, "BUY", "AAPL", qty=1, price=10.0,
                   ts="2026-05-21T02:00:00+00:00"),
        ]
        out = build_conviction_deployment(trades, [], [])
        assert out["n_buys_scanned"] == 1
        assert out["evidence"][0]["ticker"] == "AAPL"

    def test_add_action_treated_as_buy(self):
        trades = [
            _trade(1, "ADD", "NVDA", qty=1, price=100.0,
                   ts="2026-05-21T01:00:00+00:00"),
        ]
        out = build_conviction_deployment(trades, [], [])
        assert out["n_buys_scanned"] == 1


class TestPeakScoreWindow:
    def test_score_inside_window_picked(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        # ai_score 7.5 is 2h before the BUY — inside the 6h default.
        articles = [_article("NVDA solid Q1", 7.5,
                             "2026-05-21T04:00:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] == 7.5
        assert out["evidence"][0]["bucket"] == "7-8"
        assert out["evidence"][0]["score_unavailable"] is False

    def test_score_outside_window_dropped(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        # ai_score 9.0 is 12h before — outside the 6h default.
        articles = [_article("NVDA old news", 9.0,
                             "2026-05-20T18:00:00+00:00")]
        out = build_conviction_deployment(
            trades, articles, [], window_hours_pre_trade=6.0,
        )
        assert out["evidence"][0]["peak_ai_score_pre"] is None
        assert out["evidence"][0]["score_unavailable"] is True
        assert out["n_buys_score_unavailable"] == 1

    def test_score_after_trade_excluded(self):
        # Future articles (ai_score arrives AFTER the trade) must not
        # poison the attribution — that would be look-ahead bias.
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("NVDA after-the-fact", 9.5,
                             "2026-05-21T07:00:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] is None

    def test_score_exactly_at_trade_ts_included(self):
        # The closed upper edge: an article first_seen exactly at the
        # trade ts is included (the (open, closed] interval).
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("NVDA right at the trade", 8.5,
                             "2026-05-21T06:00:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] == 8.5

    def test_peak_picked_when_multiple_articles(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [
            _article("NVDA quieter", 6.0, "2026-05-21T05:30:00+00:00"),
            _article("NVDA loudest", 9.2, "2026-05-21T05:45:00+00:00"),
            _article("NVDA medium", 7.0, "2026-05-21T05:55:00+00:00"),
        ]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] == 9.2
        assert out["evidence"][0]["top_article_title"] == "NVDA loudest"


class TestEquityInterpolation:
    def test_starting_capital_fallback_pre_curve(self):
        # Trade timestamp BEFORE the first equity point ⇒ use
        # starting_capital fallback.
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=200.0,
                         ts="2026-05-20T00:00:00+00:00")]
        equity = [_equity("2026-05-21T00:00:00+00:00", 2000.0)]
        out = build_conviction_deployment(
            trades, [], equity, starting_capital=1000.0,
        )
        # value 200 / 1000 = 20%
        assert out["evidence"][0]["equity_at_trade_usd"] == 1000.0
        assert out["evidence"][0]["size_pct"] == 20.0

    def test_nearest_before_picked(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T12:00:00+00:00")]
        equity = [
            _equity("2026-05-21T10:00:00+00:00", 1000.0),
            _equity("2026-05-21T11:00:00+00:00", 1100.0),
            _equity("2026-05-21T13:00:00+00:00", 1200.0),  # AFTER trade
        ]
        out = build_conviction_deployment(trades, [], equity)
        # Should pick 1100.0 (largest ≤ trade ts), not 1200.0
        assert out["evidence"][0]["equity_at_trade_usd"] == 1100.0


class TestTickerMatching:
    def test_word_boundary_match(self):
        # NVDA matches in title; MU does NOT match inside Museum.
        trades = [
            _trade(1, "BUY", "MU", qty=1, price=100.0,
                   ts="2026-05-21T06:00:00+00:00"),
        ]
        articles = [
            _article("Tour of the Museum opens today", 9.5,
                     "2026-05-21T05:00:00+00:00"),
            _article("MU posts strong Q1 results", 8.5,
                     "2026-05-21T05:30:00+00:00"),
        ]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] == 8.5

    def test_dollar_sign_prefix_matches(self):
        # Stocktwits-style $NVDA must match NVDA (regex \b treats $ as
        # a non-word char so the boundary is exactly before N).
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("$NVDA rip on results", 9.0,
                             "2026-05-21T05:00:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["peak_ai_score_pre"] == 9.0

    def test_other_ticker_does_not_match(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("AAPL beats on iPhone", 9.0,
                             "2026-05-21T05:00:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["score_unavailable"] is True


class TestBucketCuts:
    def test_high_bucket_open_ended(self):
        # 9.0, 9.5, 10.0 must all land in "9+".
        trades = []
        articles = []
        ts0 = "2026-05-21T06:00:00+00:00"
        for i, score in enumerate([9.0, 9.5, 10.0]):
            t = f"2026-05-21T0{i+6}:00:00+00:00"
            trades.append(_trade(i + 1, "BUY", f"AAA{i}", qty=1,
                                 price=100.0, ts=t))
            articles.append(_article(f"AAA{i} loud", score,
                                     f"2026-05-21T0{i+5}:30:00+00:00"))
        out = build_conviction_deployment(
            trades, articles, [], starting_capital=1000.0,
        )
        nine_plus = next(b for b in out["buckets"] if b["label"] == "9+")
        assert nine_plus["n_buys"] == 3

    def test_bucket_boundary_inclusive_lower(self):
        # ai_score 8.0 exactly belongs to "8-9", not "7-8".
        trades = [_trade(1, "BUY", "AAA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("AAA event", 8.0,
                             "2026-05-21T05:30:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["bucket"] == "8-9"

    def test_bucket_boundary_below_six(self):
        trades = [_trade(1, "BUY", "AAA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [_article("AAA mild", 5.99,
                             "2026-05-21T05:30:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["evidence"][0]["bucket"] == "<6"


def _stable_trade(i, ticker, score, *, qty=1, price=100.0):
    """Helper: emit a BUY + matching article that lands ticker in a
    bucket selected by score. Each trade lands at a distinct hour so
    the chronological order is stable."""
    h = i % 23
    t_ts = f"2026-05-21T{h:02d}:00:00+00:00"
    a_ts = f"2026-05-21T{h:02d}:30:00+00:00"
    # Articles in the SAME hour as the BUY trade — but 30 minutes AFTER
    # the trade — would be excluded by the window (open upper edge).
    # Push articles to 30 minutes BEFORE the trade by using the prior hour.
    a_ts = f"2026-05-21T{(h - 1) % 23:02d}:30:00+00:00"
    return (
        _trade(i, "BUY", ticker, qty=qty, price=price, ts=t_ts),
        _article(f"{ticker} catalyst", score, a_ts),
    )


class TestVerdictRollup:
    def test_insufficient_with_one_populated_bucket(self):
        # Three BUYs all in the same "9+" bucket — populated buckets=1
        # < STABLE_MIN_POPULATED_BUCKETS=2 ⇒ INSUFFICIENT.
        trades, articles = [], []
        for i in range(STABLE_MIN_PER_BUCKET):
            t, a = _stable_trade(i + 1, f"AAA{i}", score=9.5)
            trades.append(t)
            articles.append(a)
        out = build_conviction_deployment(trades, articles, [])
        # State could be EMERGING or STABLE depending on n_with_score
        # vs threshold; the verdict gate is what's load-bearing here.
        assert out["verdict"] == "INSUFFICIENT"

    def test_monotonic_when_size_scales_with_score(self):
        # Build STABLE_MIN_PER_BUCKET trades in each of 3 buckets:
        # "<6" sized small, "7-8" sized medium, "9+" sized large.
        # equity = 1000; sizes scaled accordingly.
        trades, articles = [], []
        i = 0
        # Sub-6 bucket: 5% size (price 50, qty 1)
        for _ in range(STABLE_MIN_PER_BUCKET):
            i += 1
            t, a = _stable_trade(i, f"LO{i}", score=4.0, qty=1, price=50.0)
            trades.append(t); articles.append(a)
        # 7-8 bucket: 15% size (price 150)
        for _ in range(STABLE_MIN_PER_BUCKET):
            i += 1
            t, a = _stable_trade(i, f"MD{i}", score=7.5, qty=1, price=150.0)
            trades.append(t); articles.append(a)
        # 9+ bucket: 35% size (price 350)
        for _ in range(STABLE_MIN_PER_BUCKET):
            i += 1
            t, a = _stable_trade(i, f"HI{i}", score=9.5, qty=1, price=350.0)
            trades.append(t); articles.append(a)
        out = build_conviction_deployment(
            trades, articles, [], starting_capital=1000.0,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "MONOTONIC"

    def test_flat_when_sizes_uniform(self):
        # Same dollar size across buckets ⇒ FLAT.
        trades, articles = [], []
        i = 0
        for score in (4.0, 7.5, 9.5):
            for _ in range(STABLE_MIN_PER_BUCKET):
                i += 1
                t, a = _stable_trade(i, f"X{i}", score=score,
                                     qty=1, price=100.0)
                trades.append(t); articles.append(a)
        out = build_conviction_deployment(
            trades, articles, [], starting_capital=1000.0,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "FLAT"

    def test_inverted_when_size_drops_with_score(self):
        # Counter-intuitive: smaller bets at higher conviction.
        trades, articles = [], []
        i = 0
        for score, price in ((4.0, 350.0), (7.5, 150.0), (9.5, 50.0)):
            for _ in range(STABLE_MIN_PER_BUCKET):
                i += 1
                t, a = _stable_trade(i, f"INV{i}", score=score,
                                     qty=1, price=price)
                trades.append(t); articles.append(a)
        out = build_conviction_deployment(
            trades, articles, [], starting_capital=1000.0,
        )
        assert out["state"] == "STABLE"
        assert out["verdict"] == "INVERTED"


class TestSlopeThreshold:
    def test_small_delta_below_slope_is_flat(self):
        # A < MONOTONIC_SLOPE_PP per-step delta should NOT read as
        # MONOTONIC — the curve must be informative at the noise grain.
        trades, articles = [], []
        i = 0
        # Bucket "<6" sized 10%, "7-8" sized 12%, "9+" sized 14%.
        # Step ≈ 2pp < MONOTONIC_SLOPE_PP (5pp default) ⇒ FLAT.
        prices = (100.0, 120.0, 140.0)
        for score, price in zip((4.0, 7.5, 9.5), prices):
            for _ in range(STABLE_MIN_PER_BUCKET):
                i += 1
                t, a = _stable_trade(i, f"S{i}", score=score,
                                     qty=1, price=price)
                trades.append(t); articles.append(a)
        out = build_conviction_deployment(
            trades, articles, [], starting_capital=1000.0,
        )
        # Verdict at this noise grain is FLAT, not MONOTONIC.
        assert out["verdict"] == "FLAT"


class TestScoreUnavailable:
    def test_unavailable_count_when_no_articles(self):
        trades = [
            _trade(1, "BUY", "OLD", qty=1, price=100.0,
                   ts="2025-01-01T00:00:00+00:00"),
            _trade(2, "BUY", "FRESH", qty=1, price=100.0,
                   ts="2026-05-21T06:00:00+00:00"),
        ]
        # Only the recent trade has a matching article.
        articles = [_article("FRESH big news", 8.5,
                             "2026-05-21T05:30:00+00:00")]
        out = build_conviction_deployment(trades, articles, [])
        assert out["n_buys_scanned"] == 2
        assert out["n_buys_with_score"] == 1
        assert out["n_buys_score_unavailable"] == 1
        # Old trade still appears in evidence with score_unavailable.
        ev = {e["ticker"]: e for e in out["evidence"]}
        assert ev["OLD"]["score_unavailable"] is True
        assert ev["FRESH"]["score_unavailable"] is False


class TestMalformedRows:
    def test_garbage_articles_do_not_crash(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        articles = [
            None,
            "garbage",
            {"first_seen": "not-a-timestamp", "ai_score": 9.0,
             "title": "NVDA"},
            {"first_seen": "2026-05-21T05:30:00+00:00",
             "ai_score": "not-a-number", "title": "NVDA"},
            # Valid one:
            _article("NVDA real", 8.0, "2026-05-21T05:30:00+00:00"),
        ]
        out = build_conviction_deployment(trades, articles, [])
        # The one valid article must be picked.
        assert out["evidence"][0]["peak_ai_score_pre"] == 8.0

    def test_garbage_equity_rows_do_not_crash(self):
        trades = [_trade(1, "BUY", "NVDA", qty=1, price=100.0,
                         ts="2026-05-21T06:00:00+00:00")]
        equity = [
            None,
            "x",
            {"timestamp": None, "total_value": 1500.0},
            {"timestamp": "bad", "total_value": 1500.0},
            {"timestamp": "2026-05-21T00:00:00+00:00",
             "total_value": "not-a-number"},
            _equity("2026-05-21T03:00:00+00:00", 2000.0),
        ]
        out = build_conviction_deployment(
            trades, [], equity, starting_capital=1000.0,
        )
        # Only the valid equity row at 2000 counts.
        assert out["evidence"][0]["equity_at_trade_usd"] == 2000.0
